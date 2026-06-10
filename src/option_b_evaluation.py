"""
src/option_b_evaluation.py

Scientific validation module (Option B) to evaluate the impact of rare mutation weighting
and attention modulation against baseline models across 18 HIV drugs.

IMPLEMENTATION COMPLETE: All 5 experiments are now implemented.
- Experiment 1: Temporal Robustness Evaluation
- Experiment 2: Rare Mutation Sensitivity
- Experiment 3: Overall Performance Delta
- Experiment 4: Calibration Stability Check  
- Experiment 5: Biological DRM Alignment
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, brier_score_loss, recall_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add workspace to path
workspace_dir = Path(__file__).resolve().parent.parent
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

# Imports from existing modules
from src.data_processing import (
    load_fasta, save_fasta, load_unified_data,
    PI_DRUGS, NRTI_DRUGS, NNRTI_DRUGS
)
from src.rare_mutations import (
    compute_mutation_frequencies,
    compute_rare_mutation_weights,
    normalize_weights_for_attention,
    AMINO_ACIDS, AA_TO_IDX
)
from src.models import (
    train_attention_model,
    EmbeddingDataset,
    collate_embeddings,
    AttentionWeightedClassifier
)
from src.evaluation import compute_calibration_metrics
from src.interpretability import load_known_drms, get_drug_specific_drms, compute_drm_enrichment
from src.feature_engineering import HIV_PROTEASE_REFERENCE, HIV_RT_REFERENCE
from src.temporal_validation import create_temporal_split

# Set styling for premium visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Color palette definition for premium aesthetics
COLOR_BASELINE = '#e74c3c'  # Crimson Red
COLOR_ENHANCED = '#2ecc71'  # Emerald Green
COLOR_MUT_RARE = '#f1c40f'  # Sun Yellow
COLOR_MUT_COMMON = '#3498db'  # Peter River Blue


def evaluate_cross_validation(
    X: List[np.ndarray],
    y: np.ndarray,
    rw: Optional[List[np.ndarray]] = None,
    n_splits: int = 5,
    seed: int = 42,
    epochs: int = 20
) -> Tuple[np.ndarray, List[AttentionWeightedClassifier]]:
    """
    Run 5-fold CV to get out-of-fold predictions and trained fold models.
    """
    # Adjust folds if class count is small
    min_class = np.min(np.bincount(y.astype(int)))
    effective_splits = min(n_splits, int(min_class))
    
    if effective_splits < 2:
        return np.full(len(y), np.nan), []
        
    cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    y_pred = np.zeros(len(y))
    fold_models = []
    
    for train_idx, val_idx in cv.split(np.zeros(len(y)), y):
        X_train_fold = [X[i] for i in train_idx]
        X_val_fold = [X[i] for i in val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        if rw is not None:
            rw_train_fold = [rw[i] for i in train_idx]
            rw_val_fold = [rw[i] for i in val_idx]
        else:
            rw_train_fold = None
            rw_val_fold = None
            
        model = train_attention_model(
            X_train_fold, y_train_fold,
            val_embeddings_list=X_val_fold,
            val_labels=y_val_fold,
            rare_mutation_weights_list=rw_train_fold,
            val_rare_mutation_weights_list=rw_val_fold,
            epochs=epochs,
            verbose=False
        )
        fold_models.append(model)
        
        model.eval()
        val_dataset = EmbeddingDataset(X_val_fold, y_val_fold, rare_mutation_weights_list=rw_val_fold)
        val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_embeddings)
        
        fold_preds = []
        with torch.no_grad():
            device = next(model.parameters()).device
            for val_batch in val_loader:
                if rw_val_fold is not None and len(val_batch) == 4:
                    v_emb, _, v_mask, v_rw = val_batch
                    v_rw = v_rw.to(device)
                else:
                    v_emb, _, v_mask = val_batch[:3]
                    v_rw = None
                v_emb = v_emb.to(device)
                v_mask = v_mask.to(device)
                logits, _ = model(v_emb, v_mask, rare_mutation_weights=v_rw)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                fold_preds.extend(probs)
                
        y_pred[val_idx] = fold_preds
        
    return y_pred, fold_models


def evaluate_temporal(
    X: List[np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    rw: Optional[List[np.ndarray]] = None,
    seed: int = 42,
    epochs: int = 20
) -> np.ndarray:
    """
    Train model on training split and predict on testing split.
    """
    X_train = [X[i] for i in train_idx]
    y_train = y[train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = y[test_idx]
    
    if rw is not None:
        rw_train = [rw[i] for i in train_idx]
        rw_test = [rw[i] for i in test_idx]
    else:
        rw_train = None
        rw_test = None
        
    model = train_attention_model(
        X_train, y_train,
        rare_mutation_weights_list=rw_train,
        epochs=epochs,
        verbose=False
    )
    
    model.eval()
    test_dataset = EmbeddingDataset(X_test, y_test, rare_mutation_weights_list=rw_test)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_embeddings)
    
    preds = []
    with torch.no_grad():
        device = next(model.parameters()).device
        for batch in test_loader:
            if rw_test is not None and len(batch) == 4:
                b_emb, _, b_mask, b_rw = batch
                b_rw = b_rw.to(device)
            else:
                b_emb, _, b_mask = batch[:3]
                b_rw = None
            b_emb = b_emb.to(device)
            b_mask = b_mask.to(device)
            logits, _ = model(b_emb, b_mask, rare_mutation_weights=b_rw)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds.extend(probs)
            
    return np.array(preds)


def classify_sequences_by_rarity(
    sequences: List[str],
    reference: str,
    freqs: np.ndarray,
    threshold: float = 0.05,
    percentile: float = 20.0
) -> np.ndarray:
    """
    Classify each sequence as containing a rare mutation (True) or not (False).
    A mutation is an amino acid at position i that is different from the reference sequence.
    """
    all_mutation_freqs = []
    for seq in sequences:
        for i in range(min(len(seq), len(reference))):
            aa = seq[i]
            ref_aa = reference[i]
            if aa != ref_aa and aa in AA_TO_IDX:
                all_mutation_freqs.append(freqs[i, AA_TO_IDX[aa]])
                
    cutoff = np.percentile(all_mutation_freqs, percentile) if len(all_mutation_freqs) > 0 else 0.0
    # Rarity threshold is the max of the 5% threshold and the bottom 20th percentile
    rarity_threshold = max(threshold, cutoff)
    
    has_rare_mutation = np.zeros(len(sequences), dtype=bool)
    for idx, seq in enumerate(sequences):
        for i in range(min(len(seq), len(reference))):
            aa = seq[i]
            ref_aa = reference[i]
            if aa != ref_aa and aa in AA_TO_IDX:
                freq = freqs[i, AA_TO_IDX[aa]]
                if freq < rarity_threshold:
                    has_rare_mutation[idx] = True
                    break
                    
    return has_rare_mutation


def compute_subset_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Helper to compute AUROC, Recall, and F1 score for a sequence subset.
    """
    if len(np.unique(y_true)) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(y_true, y_pred_proba)
        
    y_pred_bin = (y_pred_proba >= threshold).astype(int)
    recall = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    
    return auc, recall, f1


def get_attention_differential_for_model(
    model: AttentionWeightedClassifier,
    X: List[np.ndarray],
    y: np.ndarray,
    rw: Optional[List[np.ndarray]] = None,
    device: torch.device = None
) -> np.ndarray:
    """
    Extract attention differential (resistant - susceptible) using a trained AttentionWeightedClassifier model.
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    
    resistant_idx = np.where(y == 1)[0]
    susceptible_idx = np.where(y == 0)[0]
    
    res_weights = []
    susc_weights = []
    
    for idx in resistant_idx:
        emb = X[idx]
        rare_w = rw[idx] if rw is not None else None
        
        emb_tensor = torch.FloatTensor(emb).unsqueeze(0).to(device)
        mask = torch.ones(1, len(emb)).to(device)
        rw_tensor = torch.FloatTensor(rare_w).unsqueeze(0).to(device) if rare_w is not None else None
        
        with torch.no_grad():
            _, weights = model(emb_tensor, mask, rare_mutation_weights=rw_tensor)
        res_weights.append(weights.cpu().numpy().flatten())
        
    for idx in susceptible_idx:
        emb = X[idx]
        rare_w = rw[idx] if rw is not None else None
        
        emb_tensor = torch.FloatTensor(emb).unsqueeze(0).to(device)
        mask = torch.ones(1, len(emb)).to(device)
        rw_tensor = torch.FloatTensor(rare_w).unsqueeze(0).to(device) if rare_w is not None else None
        
        with torch.no_grad():
            _, weights = model(emb_tensor, mask, rare_mutation_weights=rw_tensor)
        susc_weights.append(weights.cpu().numpy().flatten())
        
    max_len = max(
        max(len(w) for w in res_weights) if len(res_weights) > 0 else 0,
        max(len(w) for w in susc_weights) if len(susc_weights) > 0 else 0
    )
    
    def pad_and_average(attention_list):
        if len(attention_list) == 0:
            return np.zeros(max_len)
        padded = np.zeros((len(attention_list), max_len))
        for i, a in enumerate(attention_list):
            padded[i, :len(a)] = a
        return np.mean(padded, axis=0)
        
    res_avg = pad_and_average(res_weights)
    susc_avg = pad_and_average(susc_weights)
    return res_avg - susc_avg


def get_reference_for_class(drug_class: str) -> str:
    """Get reference sequence for drug class."""
    if drug_class == 'PI':
        return HIV_PROTEASE_REFERENCE
    else:
        return HIV_RT_REFERENCE


def load_embeddings_data(data_dir: str, subset_size: int = 100) -> Dict:
    """
    Load subsampled embeddings and phenotype data.
    """
    data_dir = Path(data_dir)
    processed_dir = data_dir / 'processed'
    embeddings_dir = data_dir / 'embeddings'
    
    # Load unified data
    unified_data = load_unified_data(processed_dir)
    
    # Structure for storing prepared data
    sub_data = {}
    
    for drug_class in ['PI', 'NRTI', 'NNRTI']:
        print(f"Loading {drug_class} data...")
        
        # Load embeddings (subsampled)
        emb_file = embeddings_dir / f"{drug_class}_per_residue_sub_{subset_size}.npy"
        if not emb_file.exists():
            print(f"Warning: Embedding file not found: {emb_file}")
            continue
            
        per_residue_emb = np.load(emb_file, allow_pickle=True)
        per_residue_emb = [np.array(x, dtype=np.float32) for x in per_residue_emb]
        
        # Get full data
        full_data = unified_data.get(drug_class)
        if full_data is None:
            continue
            
        # Subsample based on phenotype density
        sequences = full_data['sequences'][:subset_size]
        seq_ids = full_data['seq_ids'][:subset_size]
        phenotypes = full_data['phenotypes'].iloc[:subset_size].reset_index(drop=True)
        drugs = full_data['drugs']
        reference = get_reference_for_class(drug_class)
        
        sub_data[drug_class] = {
            'sequences': sequences,
            'seq_ids': seq_ids,
            'phenotypes': phenotypes,
            'drugs': drugs,
            'per_residue': per_residue_emb,
            'reference': reference
        }
    
    return sub_data


def run_evaluation_pipeline(data_dir="data", results_dir="results/option_b_evaluation", seed=42, subset_size=100):
    """
    Main evaluation pipeline running all 5 experiments.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = results_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("OPTION B: SCIENTIFIC VALIDATION MODULE - ALL 5 EXPERIMENTS")
    print("="*80)
    
    # Load dataset
    print("\nLoading embeddings and phenotype data...")
    sub_data = load_embeddings_data(data_dir, subset_size)
    
    # Data frames to record experiment metrics
    drug_level_rows = []
    rare_mutation_rows = []
    calibration_rows = []
    drm_enrichment_rows = []
    
    # Storage for drop values to run paired Wilcoxon test
    baseline_drops = []
    enhanced_drops = []
    drugs_tested = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Option B scientific validation on {device}...")
    
    for drug_class, data in sub_data.items():
        print(f"\n==================== PROCESSING CLASS: {drug_class} ====================")
        sequences = data['sequences']
        phenotypes = data['phenotypes']
        drugs = data['drugs']
        per_residue = data['per_residue']
        reference = data['reference']
        
        # Binarize and align columns
        renamed_pheno = phenotypes.rename(columns={d: f"{d}_FC" for d in drugs})
        
        # Compute Rare Mutation frequencies and weights
        freqs = compute_mutation_frequencies(sequences, reference)
        raw_weights = compute_rare_mutation_weights(sequences, reference, frequencies=freqs)
        
        # Classify sequence rarity status
        has_rare_mutation = classify_sequences_by_rarity(sequences, reference, freqs)
        
        # Softmax-normalize rare mutation weights for attention
        norm_rare_weights = []
        for w in raw_weights:
            norm_w = normalize_weights_for_attention(w, method='softmax')
            norm_rare_weights.append(norm_w)
            
        for drug in drugs:
            print(f"\n--- Drug: {drug} ---")
            
            # Fetch valid samples for the drug
            fc_col = f"{drug}_FC"
            if fc_col not in renamed_pheno.columns:
                print(f"Skipping {drug}: column not found.")
                continue
                
            y_vals = renamed_pheno[fc_col].values
            valid_mask = ~np.isnan(y_vals)
            
            y_valid = (y_vals[valid_mask] >= 2.5).astype(int)
            X_valid = [per_residue[i] for i in range(len(per_residue)) if valid_mask[i]]
            rw_valid = [norm_rare_weights[i] for i in range(len(norm_rare_weights)) if valid_mask[i]]
            rare_flags_valid = has_rare_mutation[valid_mask]
            
            n_resistant = int(y_valid.sum())
            n_susceptible = len(y_valid) - n_resistant
            
            if len(np.unique(y_valid)) < 2 or min(n_resistant, n_susceptible) < 3:
                print(f"Skipping drug {drug}: insufficient samples (R={n_resistant}, S={n_susceptible})")
                continue
                
            # --- EVALUATION 1: Stratified CV ---
            print("Training Baseline CV...")
            y_pred_cv_base, base_cv_models = evaluate_cross_validation(
                X_valid, y_valid, rw=None, seed=seed, epochs=20
            )
            print("Training Enhanced CV...")
            y_pred_cv_enh, enh_cv_models = evaluate_cross_validation(
                X_valid, y_valid, rw=rw_valid, seed=seed, epochs=20
            )
            
            if np.isnan(y_pred_cv_base).any() or np.isnan(y_pred_cv_enh).any():
                print(f"Skipping drug {drug}: CV failed.")
                continue
                
            cv_auc_base = roc_auc_score(y_valid, y_pred_cv_base)
            cv_auc_enh = roc_auc_score(y_valid, y_pred_cv_enh)
            
            # --- EVALUATION 2: Temporal Robustness ---
            sub_pheno_valid = renamed_pheno.iloc[valid_mask].copy()
            # Try splitting
            try:
                # Support both 'seq_id' and 'SeqID'
                seq_col = 'seq_id' if 'seq_id' in sub_pheno_valid.columns else 'SeqID'
                train_idx, test_idx = create_temporal_split(sub_pheno_valid, seq_id_col=seq_col, cutoff_quantile=0.8)
                
                if len(train_idx) >= 10 and len(test_idx) >= 5:
                    y_tr = y_valid[train_idx]
                    y_te = y_valid[test_idx]
                    
                    if len(np.unique(y_tr)) >= 2 and len(np.unique(y_te)) >= 2:
                        y_pred_temp_base = evaluate_temporal(X_valid, y_valid, train_idx, test_idx, rw=None, seed=seed, epochs=20)
                        y_pred_temp_enh = evaluate_temporal(X_valid, y_valid, train_idx, test_idx, rw=rw_valid, seed=seed, epochs=20)
                        
                        temp_auc_base = roc_auc_score(y_te, y_pred_temp_base)
                        temp_auc_enh = roc_auc_score(y_te, y_pred_temp_enh)
                    else:
                        temp_auc_base, temp_auc_enh = np.nan, np.nan
                else:
                    temp_auc_base, temp_auc_enh = np.nan, np.nan
            except Exception as e:
                print(f"Temporal validation failed for {drug}: {e}")
                temp_auc_base, temp_auc_enh = np.nan, np.nan
                
            drop_base = cv_auc_base - temp_auc_base if not np.isnan(temp_auc_base) else np.nan
            drop_enh = cv_auc_enh - temp_auc_enh if not np.isnan(temp_auc_enh) else np.nan
            
            delta_auc = cv_auc_enh - cv_auc_base
            
            drug_level_rows.append({
                'drug': drug,
                'drug_class': drug_class,
                'baseline_cv_auc': cv_auc_base,
                'enhanced_cv_auc': cv_auc_enh,
                'baseline_temporal_auc': temp_auc_base,
                'enhanced_temporal_auc': temp_auc_enh,
                'baseline_drop': drop_base,
                'enhanced_drop': drop_enh,
                'delta_auc': delta_auc
            })
            
            if not np.isnan(drop_base) and not np.isnan(drop_enh):
                baseline_drops.append(drop_base)
                enhanced_drops.append(drop_enh)
                drugs_tested.append(drug)
                
            # --- EVALUATION 3: Rare Mutation Sensitivity ---
            # Evaluate metrics on rare vs common subsets
            # Rare subset
            if rare_flags_valid.any():
                y_true_rare = y_valid[rare_flags_valid]
                pred_base_rare = y_pred_cv_base[rare_flags_valid]
                pred_enh_rare = y_pred_cv_enh[rare_flags_valid]
                
                auc_base_r, rec_base_r, f1_base_r = compute_subset_metrics(y_true_rare, pred_base_rare)
                auc_enh_r, rec_enh_r, f1_enh_r = compute_subset_metrics(y_true_rare, pred_enh_rare)
                
                rare_mutation_rows.append({
                    'drug': drug,
                    'drug_class': drug_class,
                    'subset': 'rare',
                    'baseline_auc': auc_base_r,
                    'enhanced_auc': auc_enh_r,
                    'baseline_recall': rec_base_r,
                    'enhanced_recall': rec_enh_r,
                    'baseline_f1': f1_base_r,
                    'enhanced_f1': f1_enh_r,
                    'n_samples': len(y_true_rare)
                })
                
            # Common subset
            if (~rare_flags_valid).any():
                y_true_common = y_valid[~rare_flags_valid]
                pred_base_comm = y_pred_cv_base[~rare_flags_valid]
                pred_enh_comm = y_pred_cv_enh[~rare_flags_valid]
                
                auc_base_c, rec_base_c, f1_base_c = compute_subset_metrics(y_true_common, pred_base_comm)
                auc_enh_c, rec_enh_c, f1_enh_c = compute_subset_metrics(y_true_common, pred_enh_comm)
                
                rare_mutation_rows.append({
                    'drug': drug,
                    'drug_class': drug_class,
                    'subset': 'common',
                    'baseline_auc': auc_base_c,
                    'enhanced_auc': auc_enh_c,
                    'baseline_recall': rec_base_c,
                    'enhanced_recall': rec_enh_c,
                    'baseline_f1': f1_base_c,
                    'enhanced_f1': f1_enh_c,
                    'n_samples': len(y_true_common)
                })
                
            # --- EVALUATION 4: Calibration ---
            cal_base = compute_calibration_metrics(y_valid, y_pred_cv_base)
            cal_enh = compute_calibration_metrics(y_valid, y_pred_cv_enh)
            
            calibration_rows.append({
                'drug': drug,
                'drug_class': drug_class,
                'baseline_ece': cal_base['ece'],
                'enhanced_ece': cal_enh['ece'],
                'baseline_brier': cal_base['brier_score'],
                'enhanced_brier': cal_enh['brier_score']
            })
            
            # --- EVALUATION 5: Biological DRM Alignment ---
            # Train full baseline and enhanced models for this drug
            print("Training full cohort models for DRM alignment...")
            model_base = train_attention_model(X_valid, y_valid, rare_mutation_weights_list=None, epochs=20, verbose=False)
            model_enh = train_attention_model(X_valid, y_valid, rare_mutation_weights_list=rw_valid, epochs=20, verbose=False)
            
            diff_base = get_attention_differential_for_model(model_base, X_valid, y_valid, rw=None, device=device)
            diff_enh = get_attention_differential_for_model(model_enh, X_valid, y_valid, rw=rw_valid, device=device)
            
            known_drms = get_drug_specific_drms(drug, drug_class)
            
            for top_k in [10, 20, 30]:
                enrich_base = compute_drm_enrichment(diff_base, known_drms, top_k=top_k)
                enrich_enh = compute_drm_enrichment(diff_enh, known_drms, top_k=top_k)
                
                drm_enrichment_rows.append({
                    'drug': drug,
                    'drug_class': drug_class,
                    'model': 'baseline',
                    'top_k': top_k,
                    'observed_overlap': enrich_base['observed'],
                    'expected_overlap': enrich_base['expected'],
                    'enrichment_ratio': enrich_base['enrichment_ratio'],
                    'p_value': enrich_base['p_value']
                })
                
                drm_enrichment_rows.append({
                    'drug': drug,
                    'drug_class': drug_class,
                    'model': 'enhanced',
                    'top_k': top_k,
                    'observed_overlap': enrich_enh['observed'],
                    'expected_overlap': enrich_enh['expected'],
                    'enrichment_ratio': enrich_enh['enrichment_ratio'],
                    'p_value': enrich_enh['p_value']
                })
                
    # Create DataFrames
    df_drug_level = pd.DataFrame(drug_level_rows)
    df_rare = pd.DataFrame(rare_mutation_rows)
    df_cal = pd.DataFrame(calibration_rows)
    df_drm = pd.DataFrame(drm_enrichment_rows)
    
    # Save CSV files
    df_drug_level.to_csv(results_dir / "drug_level_comparison.csv", index=False)
    df_rare.to_csv(results_dir / "rare_mutation_performance.csv", index=False)
    df_cal.to_csv(results_dir / "calibration_comparison.csv", index=False)
    df_drm.to_csv(results_dir / "drm_enrichment_comparison.csv", index=False)
    
    # Run statistical summaries and paired Wilcoxon test on drops
    stat_summary_rows = []
    if len(baseline_drops) >= 5:
        w_stat, w_p = stats.wilcoxon(baseline_drops, enhanced_drops, alternative='greater')
        stat_summary_rows.append({
            'test_name': 'Temporal Robustness (Robustness Drop Comparison)',
            'statistic': w_stat,
            'p_value': w_p,
            'mean_baseline': np.mean(baseline_drops),
            'mean_enhanced': np.mean(enhanced_drops),
            'difference': np.mean(baseline_drops) - np.mean(enhanced_drops),
            'improvement_flag': w_p < 0.05
        })
        print(f"\nWilcoxon test on robustness drops: p-value = {w_p:.6f}")
    else:
        print("\nToo few temporal split points to run Wilcoxon test on robustness drops.")
        stat_summary_rows.append({
            'test_name': 'Temporal Robustness (Robustness Drop Comparison)',
            'statistic': np.nan,
            'p_value': np.nan,
            'mean_baseline': np.mean(baseline_drops) if len(baseline_drops) > 0 else np.nan,
            'mean_enhanced': np.mean(enhanced_drops) if len(enhanced_drops) > 0 else np.nan,
            'difference': np.nan,
            'improvement_flag': False
        })
        
    # Wilcoxon test on overall AUC
    if len(df_drug_level) >= 5:
        auc_base_all = df_drug_level['baseline_cv_auc'].values
        auc_enh_all = df_drug_level['enhanced_cv_auc'].values
        w_stat_auc, w_p_auc = stats.wilcoxon(auc_enh_all, auc_base_all, alternative='greater')
        stat_summary_rows.append({
            'test_name': 'Overall CV AUC Comparison',
            'statistic': w_stat_auc,
            'p_value': w_p_auc,
            'mean_baseline': np.mean(auc_base_all),
            'mean_enhanced': np.mean(auc_enh_all),
            'difference': np.mean(auc_enh_all) - np.mean(auc_base_all),
            'improvement_flag': w_p_auc < 0.05
        })
        print(f"Wilcoxon test on CV AUC: p-value = {w_p_auc:.6f}")
        
    df_stat = pd.DataFrame(stat_summary_rows)
    df_stat.to_csv(results_dir / "statistical_summary.csv", index=False)
    
    # --- VISUALIZATIONS ---
    # Plot 1: Temporal Robustness drop comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(drugs_tested))
    width = 0.35
    plt.bar(x - width/2, baseline_drops, width, label='Baseline Model', color=COLOR_BASELINE, alpha=0.85)
    plt.bar(x + width/2, enhanced_drops, width, label='Enhanced Model', color=COLOR_ENHANCED, alpha=0.85)
    plt.xticks(x, drugs_tested, rotation=45, ha='right')
    plt.xlabel('Drug')
    plt.ylabel('Robustness Drop (CV_AUC - Temporal_AUC)')
    plt.title('Temporal Robustness Drop comparison (Lower is Better)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(visuals_dir / 'temporal_robustness_comparison.png')
    plt.close()
    
    # Plot 2: Rare Mutation Performance comparison (AUROC, F1, Recall)
    if not df_rare.empty:
        # Group metrics for rare subset
        rare_df = df_rare[df_rare['subset'] == 'rare'].dropna()
        if not rare_df.empty:
            plt.figure(figsize=(12, 6))
            x_mut = np.arange(len(rare_df))
            width_mut = 0.25
            
            plt.bar(x_mut - width_mut, rare_df['baseline_auc'], width_mut, label='Baseline AUC', color=COLOR_BASELINE, alpha=0.7)
            plt.bar(x_mut, rare_df['enhanced_auc'], width_mut, label='Enhanced AUC', color=COLOR_ENHANCED, alpha=0.9)
            plt.bar(x_mut + width_mut, rare_df['enhanced_recall'], width_mut, label='Enhanced Recall', color=COLOR_MUT_RARE, alpha=0.8)
            
            plt.xticks(x_mut, rare_df['drug'].values, rotation=45, ha='right')
            plt.xlabel('Drug')
            plt.ylabel('Score')
            plt.title('Performance on Rare Mutation Sequences')
            plt.legend()
            plt.tight_layout()
            plt.savefig(visuals_dir / 'rare_mutation_performance.png')
            plt.close()
            
    # Plot 3: Delta AUC per drug bar chart
    if not df_drug_level.empty:
        plt.figure(figsize=(10, 6))
        sorted_dl = df_drug_level.sort_values('delta_auc', ascending=False)
        colors = [COLOR_ENHANCED if d >= 0 else COLOR_BASELINE for d in sorted_dl['delta_auc']]
        
        plt.bar(sorted_dl['drug'], sorted_dl['delta_auc'], color=colors, alpha=0.85)
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel('Drug')
        plt.ylabel('ΔAUC (Enhanced - Baseline)')
        plt.title('Overall Cross-Validation Delta AUC per Drug')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(visuals_dir / 'delta_auc_per_drug.png')
        plt.close()
        
    # Plot 4: Calibration ECE comparison
    if not df_cal.empty:
        plt.figure(figsize=(10, 6))
        x_cal = np.arange(len(df_cal))
        plt.bar(x_cal - width/2, df_cal['baseline_ece'], width, label='Baseline ECE', color=COLOR_BASELINE, alpha=0.85)
        plt.bar(x_cal + width/2, df_cal['enhanced_ece'], width, label='Enhanced ECE', color=COLOR_ENHANCED, alpha=0.85)
        plt.xticks(x_cal, df_cal['drug'].values, rotation=45, ha='right')
        plt.xlabel('Drug')
        plt.ylabel('Expected Calibration Error (ECE) (Lower is Better)')
        plt.title('Expected Calibration Error comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(visuals_dir / 'calibration_comparison.png')
        plt.close()
        
    # Plot 5: DRM Enrichment overlap comparison (using top_k=20)
    df_drm_20 = df_drm[df_drm['top_k'] == 20]
    if not df_drm_20.empty:
        # Pivot to align baseline and enhanced side-by-side
        pivot_drm = df_drm_20.pivot(index='drug', columns='model', values='observed_overlap').dropna()
        
        plt.figure(figsize=(10, 6))
        x_drm = np.arange(len(pivot_drm))
        plt.bar(x_drm - width/2, pivot_drm['baseline'], width, label='Baseline DRM Overlap', color=COLOR_BASELINE, alpha=0.85)
        plt.bar(x_drm + width/2, pivot_drm['enhanced'], width, label='Enhanced DRM Overlap', color=COLOR_ENHANCED, alpha=0.85)
        plt.xticks(x_drm, pivot_drm.index, rotation=45, ha='right')
        plt.xlabel('Drug')
        plt.ylabel('Observed DRM Overlap Count (out of 20)')
        plt.title('Clinical DRM Overlap Comparison (Higher is Better)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(visuals_dir / 'drm_enrichment_comparison.png')
        plt.close()
        
    print("\n" + "="*60)
    print("OPTION B EVALUATION RUN COMPLETED SUCCESSFULLY")
    print(f"All outputs saved under: {results_dir}")
    print("="*60)


if __name__ == "__main__":
    run_evaluation_pipeline(subset_size=100)

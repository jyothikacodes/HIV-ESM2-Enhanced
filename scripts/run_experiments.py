import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add workspace to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import modules from src
from src.data_processing import (
    load_fasta, save_fasta, load_unified_data,
    PI_DRUGS, NRTI_DRUGS, NNRTI_DRUGS
)
from src.feature_engineering import (
    load_esm2_model,
    batch_extract_per_residue_embeddings,
    HIV_PROTEASE_REFERENCE, HIV_RT_REFERENCE
)
from src.rare_mutations import (
    compute_mutation_frequencies,
    compute_rare_mutation_weights,
    normalize_weights_for_attention
)
from src.models import (
    train_attention_model,
    EmbeddingDataset,
    collate_embeddings,
    per_drug_training
)
from src.temporal_validation import (
    temporal_evaluation_extended,
    create_temporal_split
)
from src.ternary_classification import (
    per_drug_ternary_training,
    aggregate_ternary_results
)
from src.calibration import (
    calibrate_predictions,
    evaluate_calibration
)
from src.shap_explainability import (
    compute_shap_with_residue_mapping
)
from src.interpretability import (
    load_known_drms,
    compute_drm_enrichment
)

# Standard evaluation metrics helper
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, brier_score_loss, confusion_matrix
)

def compute_detailed_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Check classes
    if len(np.unique(y_true)) < 2:
        return {
            'auroc': np.nan, 'accuracy': np.nan, 'macro_f1': np.nan,
            'sensitivity': np.nan, 'specificity': np.nan, 'brier_score': np.nan
        }
        
    auroc = roc_auc_score(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    brier = brier_score_loss(y_true, y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'auroc': auroc,
        'accuracy': acc,
        'macro_f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'brier_score': brier
    }

# Helper to run stratified CV for attention model
def cv_attention_with_weights(
    X_valid, y_valid, rare_weights_valid=None, n_splits=5, random_state=42, epochs=20
):
    from sklearn.model_selection import StratifiedKFold
    
    # Adjust folds if class count is small
    min_class = np.min(np.bincount(y_valid.astype(int)))
    effective_splits = min(n_splits, int(min_class))
    
    if effective_splits < 2:
        return np.full(len(y_valid), np.nan)
        
    cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
    y_pred = np.zeros(len(y_valid))
    
    for train_idx, val_idx in cv.split(np.zeros(len(y_valid)), y_valid):
        X_train_fold = [X_valid[i] for i in train_idx]
        X_val_fold = [X_valid[i] for i in val_idx]
        y_train_fold = y_valid[train_idx]
        y_val_fold = y_valid[val_idx]
        
        if rare_weights_valid is not None:
            rw_train_fold = [rare_weights_valid[i] for i in train_idx]
            rw_val_fold = [rare_weights_valid[i] for i in val_idx]
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
        
    return y_pred


def select_and_extract_subsampled_data(data_dir, subset_size, seed=42):
    data_dir = Path(data_dir)
    processed_dir = data_dir / 'processed'
    embeddings_dir = data_dir / 'embeddings'
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    unified_data = load_unified_data(processed_dir)
    
    sub_data = {}
    model = None
    alphabet = None
    batch_converter = None
    device = None
    
    references = {
        'PI': HIV_PROTEASE_REFERENCE,
        'NRTI': HIV_RT_REFERENCE,
        'NNRTI': HIV_RT_REFERENCE
    }
    
    for drug_class, data in unified_data.items():
        print(f"\n--- Preparing Subsampled Dataset for {drug_class} (Size: {subset_size}) ---")
        seqs = data['sequences']
        seq_ids = data['seq_ids']
        phenotypes = data['phenotypes']
        drugs = data['drugs']
        
        # Check files (include subset_size in name to avoid cross-contamination)
        sub_fasta = processed_dir / f"{drug_class}_sequences_sub_{subset_size}.fasta"
        sub_csv = processed_dir / f"{drug_class}_phenotypes_sub_{subset_size}.csv"
        
        if sub_fasta.exists() and sub_csv.exists():
            print(f"Loading existing subsampled cohort from {sub_fasta.name}...")
            sub_seqs, sub_seq_ids = load_fasta(sub_fasta)
            sub_pheno = pd.read_csv(sub_csv).reset_index(drop=True)
        else:
            print("Selecting top sequences based on phenotype density...")
            # Count non-nulls in drug columns
            drug_cols = [c for c in drugs if c in phenotypes.columns]
            non_null_counts = phenotypes[drug_cols].notna().sum(axis=1)
            
            # Sort and take top subset_size
            sorted_idx = non_null_counts.sort_values(ascending=False).index
            top_idx = sorted_idx[:subset_size]
            
            sub_seqs = [seqs[i] for i in top_idx]
            sub_seq_ids = [seq_ids[i] for i in top_idx]
            sub_pheno = phenotypes.iloc[top_idx].copy().reset_index(drop=True)
            
            # Save
            save_fasta(sub_seqs, sub_seq_ids, sub_fasta)
            sub_pheno.to_csv(sub_csv, index=False)
            print(f"Saved subsampled fasta and CSV to processed/")
            
        # Check embeddings
        emb_per_residue_path = embeddings_dir / f"{drug_class}_per_residue_sub_{subset_size}.npy"
        
        if emb_per_residue_path.exists():
            print(f"Loading cached ESM-2 embeddings from {emb_per_residue_path.name}...")
            # Allow pickle is required since it is a list of arrays (variable seq lengths)
            per_residue = np.load(emb_per_residue_path, allow_pickle=True)
            # Convert back to list of float32 arrays
            per_residue = [np.array(x, dtype=np.float32) for x in per_residue]
        else:
            print("ESM-2 Embeddings cache not found. Extracting on CPU (this might take a few minutes)...")
            if model is None:
                print("Loading ESM-2 model (esm2_t33_650M_UR50D)...")
                model, alphabet, batch_converter, device = load_esm2_model()
                
            per_residue = batch_extract_per_residue_embeddings(
                sub_seqs, model, alphabet, batch_converter, device, batch_size=4
            )
            
            # Save embeddings
            np.save(emb_per_residue_path, np.array(per_residue, dtype=object), allow_pickle=True)
            print(f"Saved extracted embeddings to {emb_per_residue_path.name}")
            
        sub_data[drug_class] = {
            'sequences': sub_seqs,
            'seq_ids': sub_seq_ids,
            'phenotypes': sub_pheno,
            'drugs': drugs,
            'per_residue': per_residue,
            'reference': references[drug_class]
        }
        
    return sub_data


def run_all_experiments(sub_data, results_dir, seed=42):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("RUNNING ALL EXPERIMENTAL CHECKS & VALIDATION STEPS")
    print("="*60)
    
    baseline_vs_enhanced_rows = []
    ablation_rows = []
    temporal_rows = []
    ternary_rows = []
    shap_rows = []
    calibration_rows = []
    
    for drug_class, data in sub_data.items():
        print(f"\n==================== PROCESSING CLASS: {drug_class} ====================")
        
        sequences = data['sequences']
        phenotypes = data['phenotypes']
        drugs = data['drugs']
        per_residue = data['per_residue']
        reference = data['reference']
        
        # Rename columns to {drug}_FC to enable correct binarization internally
        renamed_pheno = phenotypes.rename(columns={d: f"{d}_FC" for d in drugs})
        
        # 1. Compute Rare Mutation weights
        print("Computing mutation frequencies and weights...")
        freqs = compute_mutation_frequencies(sequences, reference)
        raw_weights = compute_rare_mutation_weights(sequences, reference, frequencies=freqs)
        
        # Softmax-normalize rare mutation weights for attention
        norm_rare_weights = []
        for w in raw_weights:
            norm_w = normalize_weights_for_attention(w, method='softmax')
            norm_rare_weights.append(norm_w)
            
        # Process each drug
        for drug in drugs:
            print(f"\n--- Drug: {drug} ---")
            
            # Get valid mask for this drug
            fc_col = f"{drug}_FC"
            y_vals = renamed_pheno[fc_col].values
            valid_mask = ~np.isnan(y_vals)
            
            # Filter
            y_valid = (y_vals[valid_mask] >= 2.5).astype(int)
            X_valid = [per_residue[i] for i in range(len(per_residue)) if valid_mask[i]]
            rw_valid = [norm_rare_weights[i] for i in range(len(norm_rare_weights)) if valid_mask[i]]
            
            n_resistant = int(y_valid.sum())
            n_susceptible = len(y_valid) - n_resistant
            
            if len(np.unique(y_valid)) < 2 or min(n_resistant, n_susceptible) < 3:
                print(f"Skipping drug {drug}: insufficient sample counts (R={n_resistant}, S={n_susceptible})")
                continue
                
            print(f"Dataset summary: n_samples={len(y_valid)} (Resistant={n_resistant}, Susceptible={n_susceptible})")
            
            # --- TASK 1: Baseline vs Enhanced ---
            print("Training Baseline Attention Model (5-fold CV)...")
            y_pred_baseline = cv_attention_with_weights(
                X_valid, y_valid, rare_weights_valid=None, n_splits=5, random_state=seed, epochs=20
            )
            
            print("Training Enhanced Attention Model (with Rarity Weights, 5-fold CV)...")
            y_pred_enhanced = cv_attention_with_weights(
                X_valid, y_valid, rare_weights_valid=rw_valid, n_splits=5, random_state=seed, epochs=20
            )
            
            # Check for CV failure
            if np.isnan(y_pred_baseline).any() or np.isnan(y_pred_enhanced).any():
                print(f"Skipping drug {drug}: CV training failed to yield predictions.")
                continue
                
            baseline_metrics = compute_detailed_metrics(y_valid, y_pred_baseline)
            enhanced_metrics = compute_detailed_metrics(y_valid, y_pred_enhanced)
            
            baseline_vs_enhanced_rows.append({
                'drug': drug,
                'drug_class': drug_class,
                'baseline_auroc': baseline_metrics['auroc'],
                'baseline_accuracy': baseline_metrics['accuracy'],
                'baseline_macro_f1': baseline_metrics['macro_f1'],
                'baseline_sensitivity': baseline_metrics['sensitivity'],
                'baseline_specificity': baseline_metrics['specificity'],
                'baseline_brier_score': baseline_metrics['brier_score'],
                'enhanced_auroc': enhanced_metrics['auroc'],
                'enhanced_accuracy': enhanced_metrics['accuracy'],
                'enhanced_macro_f1': enhanced_metrics['macro_f1'],
                'enhanced_sensitivity': enhanced_metrics['sensitivity'],
                'enhanced_specificity': enhanced_metrics['specificity'],
                'enhanced_brier_score': enhanced_metrics['brier_score'],
                'n_samples': len(y_valid)
            })
            
            # --- TASK 2: Ablation Study ---
            print("Running Ablation: No-Attention (Mean pooling + Logistic Regression)...")
            # Extract mean pooled features
            X_mean_valid = np.array([np.mean(e, axis=0) for e in X_valid])
            
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import StratifiedKFold, cross_val_predict
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_mean_valid)
            cv_splits = min(5, min(n_resistant, n_susceptible))
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
            
            lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)
            y_pred_no_attn = cross_val_predict(lr_model, X_scaled, y_valid, cv=cv, method='predict_proba')[:, 1]
            
            no_attn_metrics = compute_detailed_metrics(y_valid, y_pred_no_attn)
            
            # Calibration model (Auto Calibration on Enhanced predictions)
            print("Running Calibration: Auto calibrating Enhanced model predictions...")
            y_pred_calibrated = np.zeros_like(y_pred_enhanced)
            for train_idx, val_idx in cv.split(y_pred_enhanced, y_valid):
                y_tr_true, y_val_true = y_valid[train_idx], y_valid[val_idx]
                y_tr_pred, y_val_pred = y_pred_enhanced[train_idx], y_pred_enhanced[val_idx]
                try:
                    y_pred_calibrated[val_idx] = calibrate_predictions(y_tr_true, y_tr_pred, y_val_pred, method='auto')
                except Exception:
                    y_pred_calibrated[val_idx] = y_val_pred # Fallback
                    
            cal_metrics = compute_detailed_metrics(y_valid, y_pred_calibrated)
            
            # Record Ablation Study
            configs = {
                'Full Model': enhanced_metrics,
                'No Rarity': baseline_metrics,
                'No Attention': no_attn_metrics,
                'With Calibration': cal_metrics,
                'Without Calibration': enhanced_metrics
            }
            
            for cfg, metrics in configs.items():
                ablation_rows.append({
                    'drug': drug,
                    'drug_class': drug_class,
                    'config': cfg,
                    'auroc': metrics['auroc'],
                    'accuracy': metrics['accuracy'],
                    'macro_f1': metrics['macro_f1'],
                    'brier_score': metrics['brier_score']
                })
                
            # --- TASK 3: Temporal Validation ---
            print("Running Temporal Validation split...")
            # We sort sequences chronologically using seq_id proxy
            # Load sub phenotypes to get seq_id column
            sub_pheno_valid = renamed_pheno.iloc[valid_mask].copy()
            # Perform temporal split on valid subset
            train_idx, test_idx = create_temporal_split(sub_pheno_valid, seq_id_col='seq_id', cutoff_quantile=0.8)
            
            if len(train_idx) >= 10 and len(test_idx) >= 5:
                # Prepare train/test sets
                X_tr = [X_valid[i] for i in train_idx]
                y_tr = y_valid[train_idx]
                rw_tr = [rw_valid[i] for i in train_idx]
                
                X_te = [X_valid[i] for i in test_idx]
                y_te = y_valid[test_idx]
                rw_te = [rw_valid[i] for i in test_idx]
                
                if len(np.unique(y_tr)) >= 2 and len(np.unique(y_te)) >= 2:
                    # Train attention model on training split
                    temp_model = train_attention_model(
                        X_tr, y_tr,
                        rare_mutation_weights_list=rw_tr,
                        epochs=20,
                        verbose=False
                    )
                    
                    # Evaluate on holdout split
                    temp_model.eval()
                    te_dataset = EmbeddingDataset(X_te, y_te, rare_mutation_weights_list=rw_te)
                    te_loader = DataLoader(te_dataset, batch_size=32, collate_fn=collate_embeddings)
                    
                    y_pred_temp = []
                    with torch.no_grad():
                        dev = next(temp_model.parameters()).device
                        for te_batch in te_loader:
                            if rw_te is not None and len(te_batch) == 4:
                                t_emb, _, t_mask, t_rw = te_batch
                                t_rw = t_rw.to(dev)
                            else:
                                t_emb, _, t_mask = te_batch[:3]
                                t_rw = None
                            t_emb = t_emb.to(dev)
                            t_mask = t_mask.to(dev)
                            logits, _ = temp_model(t_emb, t_mask, rare_mutation_weights=t_rw)
                            probs = torch.sigmoid(logits).cpu().numpy().flatten()
                            y_pred_temp.extend(probs)
                            
                    temp_metrics = compute_detailed_metrics(y_te, np.array(y_pred_temp))
                    
                    difference = enhanced_metrics['auroc'] - temp_metrics['auroc']
                    drop_pct = (difference / enhanced_metrics['auroc']) * 100 if enhanced_metrics['auroc'] > 0 else 0
                    
                    temporal_rows.append({
                        'drug': drug,
                        'drug_class': drug_class,
                        'cv_auroc': enhanced_metrics['auroc'],
                        'temporal_auroc': temp_metrics['auroc'],
                        'auroc_difference': difference,
                        'auroc_drop_pct': drop_pct,
                        'cv_macro_f1': enhanced_metrics['macro_f1'],
                        'temporal_macro_f1': temp_metrics['macro_f1'],
                        'macro_f1_change': temp_metrics['macro_f1'] - enhanced_metrics['macro_f1'],
                        'n_train': len(y_tr),
                        'n_test': len(y_te)
                    })
                    print(f"Temporal Holdout Results: AUROC={temp_metrics['auroc']:.4f} (CV={enhanced_metrics['auroc']:.4f}, drop={drop_pct:.1f}%)")
                else:
                    print("Skipping Temporal: training or testing split has single-class labels only")
            else:
                print("Skipping Temporal: training/testing split size too small")
                
            # --- TASK 4: Ternary Classification ---
            print("Running Ternary (3-class) Classification (5-fold CV)...")
            # Create a localized DataFrame for ternary call
            drug_pheno = phenotypes.iloc[valid_mask].copy()
            # Call training on mean pooled features
            ternary_dict = per_drug_ternary_training(
                X_mean_valid, drug_pheno, [drug],
                model_type='logistic',
                n_splits=5,
                random_state=seed
            )
            
            if drug in ternary_dict and not ternary_dict[drug].get('skipped', False):
                res = ternary_dict[drug]
                ternary_rows.append({
                    'drug': drug,
                    'drug_class': drug_class,
                    'auc_ovr': res['auc_ovr'],
                    'accuracy': res['accuracy'],
                    'macro_f1': res['macro_f1'],
                    'n_samples': res['n_samples'],
                    'class_0_count': res['class_distribution'].get(0, 0),
                    'class_1_count': res['class_distribution'].get(1, 0),
                    'class_2_count': res['class_distribution'].get(2, 0)
                })
                print(f"Ternary Results: AUC(OvR)={res['auc_ovr']:.4f}  Acc={res['accuracy']:.4f}  F1={res['macro_f1']:.4f}")
                
            # --- TASK 6: Calibration Quality Evaluation ---
            print("Evaluating Probability Calibration...")
            # Platt
            y_pred_platt = np.zeros_like(y_pred_enhanced)
            for train_idx, val_idx in cv.split(y_pred_enhanced, y_valid):
                y_pred_platt[val_idx] = calibrate_predictions(y_valid[train_idx], y_pred_enhanced[train_idx], y_pred_enhanced[val_idx], method='platt')
            
            # Isotonic
            y_pred_isotonic = np.zeros_like(y_pred_enhanced)
            for train_idx, val_idx in cv.split(y_pred_enhanced, y_valid):
                y_pred_isotonic[val_idx] = calibrate_predictions(y_valid[train_idx], y_pred_enhanced[train_idx], y_pred_enhanced[val_idx], method='isotonic')
                
            # Evaluate calibration quality
            raw_cal = evaluate_calibration(y_valid, y_pred_enhanced, y_pred_enhanced)
            platt_cal = evaluate_calibration(y_valid, y_pred_enhanced, y_pred_platt)
            isotonic_cal = evaluate_calibration(y_valid, y_pred_enhanced, y_pred_isotonic)
            auto_cal = evaluate_calibration(y_valid, y_pred_enhanced, y_pred_calibrated)
            
            # Choose best calibration method
            best_method = 'platt' if platt_cal['calibrated']['brier_score'] <= isotonic_cal['calibrated']['brier_score'] else 'isotonic'
            best_ece = platt_cal['calibrated']['ece'] if best_method == 'platt' else isotonic_cal['calibrated']['ece']
            best_mce = platt_cal['calibrated']['mce'] if best_method == 'platt' else isotonic_cal['calibrated']['mce']
            best_brier = platt_cal['calibrated']['brier_score'] if best_method == 'platt' else isotonic_cal['calibrated']['brier_score']
            
            calibration_rows.append({
                'drug': drug,
                'drug_class': drug_class,
                'raw_ece': raw_cal['raw']['ece'],
                'calibrated_ece': best_ece,
                'raw_mce': raw_cal['raw']['mce'],
                'calibrated_mce': best_mce,
                'raw_brier_score': raw_cal['raw']['brier_score'],
                'calibrated_brier_score': best_brier,
                'best_calibration_method': best_method
            })
            print(f"Calibration Results: Raw ECE={raw_cal['raw']['ece']:.4f} -> Calibrated ECE={best_ece:.4f} (via {best_method})")
            
        # --- TASK 5: SHAP explainability ---
        # Note: run SHAP per drug class by training a single Logistic Regression classifier 
        # on the aggregated dataset for a representative drug (or average across drugs in class)
        print(f"\nRunning SHAP Biological Validation for drug class: {drug_class}")
        # Select representative drug with highest sample count
        valid_counts = {}
        for drug in drugs:
            valid_counts[drug] = renamed_pheno[f"{drug}_FC"].notna().sum()
            
        rep_drug = max(valid_counts, key=valid_counts.get)
        rep_notna_mask = renamed_pheno[f"{rep_drug}_FC"].notna().values  # numpy boolean array
        rep_fc_values = renamed_pheno[f"{rep_drug}_FC"].values
        y_rep = (rep_fc_values[rep_notna_mask] >= 2.5).astype(int)
        X_rep = np.array([np.mean(e, axis=0) for i, e in enumerate(per_residue) if rep_notna_mask[i]])
        
        # In case representative selection is single-class
        if len(np.unique(y_rep)) >= 2:
            scaler = StandardScaler()
            X_rep_scaled = scaler.fit_transform(X_rep)
            
            shap_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)
            shap_model.fit(X_rep_scaled, y_rep)
            
            print(f"Computing SHAP values for {rep_drug} (n_samples={len(y_rep)}, background=50)...")
            shap_res = compute_shap_with_residue_mapping(
                shap_model, X_rep_scaled, reference, n_background=50, random_state=seed
            )
            
            # Compare with IAS-USA 2022 guidelines
            known_drms = load_known_drms(drug_class)
            enrichment = compute_drm_enrichment(shap_res['residue_importance'], known_drms, top_k=20)
            
            # Map top 20 residues for documentation
            top_residues_df = shap_res['top_residues']
            top_res_list = [f"{row['amino_acid']}{int(row['position'])}" for _, row in top_residues_df.iterrows()]
            
            # Map overlapping residues
            overlap_res_list = []
            for _, row in top_residues_df.iterrows():
                pos = int(row['position'])
                if pos in known_drms:
                    overlap_res_list.append(f"{row['amino_acid']}{pos}")
                    
            shap_rows.append({
                'drug_class': drug_class,
                'representative_drug': rep_drug,
                'overlap_score': enrichment['observed'] / 20.0,
                'hypergeometric_p_value': enrichment['p_value'],
                'observed_overlap': enrichment['observed'],
                'expected_overlap': enrichment['expected'],
                'top_20_residues': ', '.join(top_res_list),
                'overlapping_residues': ', '.join(overlap_res_list),
                'n_total_drms_in_ref': len(known_drms & set(range(1, len(reference) + 1)))
            })
            print(f"SHAP DRM Overlap for {drug_class}: Observed={enrichment['observed']} (Expected={enrichment['expected']:.2f}, p-value={enrichment['p_value']:.4f})")
        else:
            print(f"Skipping SHAP biological validation for {drug_class}: single-class labels")
            
    # Save all results
    pd.DataFrame(baseline_vs_enhanced_rows).to_csv(results_dir / 'baseline_vs_enhanced.csv', index=False)
    pd.DataFrame(ablation_rows).to_csv(results_dir / 'ablation_study.csv', index=False)
    pd.DataFrame(temporal_rows).to_csv(results_dir / 'temporal_validation.csv', index=False)
    pd.DataFrame(ternary_rows).to_csv(results_dir / 'ternary_results.csv', index=False)
    pd.DataFrame(shap_rows).to_csv(results_dir / 'shap_biological_validation.csv', index=False)
    pd.DataFrame(calibration_rows).to_csv(results_dir / 'calibration_evaluation.csv', index=False)
    
    print("\n" + "="*60)
    print("ALL SCIENTIFIC RESULT FILES WRITTEN SUCCESSFULLY")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run HIV Drug Resistance pipeline experiments")
    parser.add_argument('--subset_size', type=int, default=100, help="Subset size per drug class (e.g., 50, 100, 250)")
    parser.add_argument('--seed', type=int, default=42, help="Random state seed")
    parser.add_argument('--data_dir', type=str, default="data", help="Data directory path")
    parser.add_argument('--results_dir', type=str, default="results", help="Results output directory")
    args = parser.parse_args()
    
    # Run
    t_start = time.time()
    sub_data = select_and_extract_subsampled_data(args.data_dir, args.subset_size, args.seed)
    run_all_experiments(sub_data, args.results_dir, args.seed)
    print(f"\nExecution finished in {time.time() - t_start:.2f} seconds.")

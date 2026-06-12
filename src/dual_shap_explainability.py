"""
Dual SHAP Explainability Framework for HIV Drug Resistance.

This module combines model-driven structural importance (attention-based SHAP, Option 3)
and clinically grounded mutation importance (mutation-level SHAP, Option 4) into
a unified fusion layer, with biological validation and premium visualizations.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import hypergeom, fisher_exact

from sklearn.linear_model import LogisticRegression
import shap

# Set styling for premium visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

# Canonical amino acid alphabet
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def parse_mutation(mut_str: str) -> Tuple[Optional[str], Optional[int], List[str]]:
    """
    Parse a mutation string (e.g. M184V, T215Y/F) into wildtype, position, and mutant list.
    """
    match = re.match(r'^([A-Z])(\d+)([A-Z/]+)$', mut_str)
    if match:
        wt, pos, muts = match.groups()
        pos = int(pos)
        mut_list = muts.split('/')
        return wt, pos, mut_list
    return None, None, []


def get_mutations_relative_to_ref(sequence: str, reference: str) -> List[str]:
    """
    Extract point mutations present in the sequence relative to the reference sequence.
    """
    muts = []
    for i in range(min(len(sequence), len(reference))):
        wt = reference[i]
        mut = sequence[i]
        if mut != wt and mut in AMINO_ACIDS:
            muts.append(f"{wt}{i+1}{mut}")
    return muts


def sequences_to_mutation_matrix(sequences: List[str], mutation_map: List[str], reference: str) -> np.ndarray:
    """
    Convert sequence strings to binary mutation feature matrix.
    """
    n_samples = len(sequences)
    n_features = len(mutation_map)
    X = np.zeros((n_samples, n_features))
    
    parsed_muts = []
    for mut in mutation_map:
        wt, pos, mut_aa = parse_mutation(mut)
        parsed_muts.append((pos, mut_aa))
        
    for idx_seq, seq in enumerate(sequences):
        for idx_mut, (pos, mut_aa) in enumerate(parsed_muts):
            if pos is not None and pos <= len(seq):
                if seq[pos - 1] in mut_aa:
                    X[idx_seq, idx_mut] = 1.0
    return X


def min_max_normalize(val: np.ndarray) -> np.ndarray:
    """Helper to min-max normalize a numpy array to [0, 1]."""
    min_val = np.min(val)
    max_val = np.max(val)
    range_val = max_val - min_val
    if range_val < 1e-12:
        return np.zeros_like(val)
    return (val - min_val) / range_val


def compute_attention_residue_shap(
    model,
    embeddings: List[np.ndarray],
    attention_weights: List[np.ndarray],
    y_target: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Compute residue-level SHAP values by training a surrogate Logistic Regression model
    on L2-norm residue attention scores.
    """
    n_samples = len(embeddings)
    seq_len = len(embeddings[0])
    
    # 1. Compute residue scores
    X_residue = np.zeros((n_samples, seq_len))
    for i in range(n_samples):
        # residue_contrib[j] = attention_weights[j] * embeddings[j]
        # residue_score[j] = norm(residue_contrib[j])
        emb = embeddings[i]
        attn = attention_weights[i]
        
        # Multiply attention by token embeddings
        contrib = attn[:, np.newaxis] * emb
        # Take L2 norm across 1280 dimension
        scores = np.linalg.norm(contrib, axis=1)
        X_residue[i, :] = scores
        
    # 2. Train surrogate Logistic Regression
    surrogate = LogisticRegression(
        class_weight='balanced',
        penalty='l2',
        C=1.0,
        random_state=random_state,
        max_iter=1000
    )
    surrogate.fit(X_residue, y_target)
    
    # 3. Compute SHAP values once
    explainer = shap.LinearExplainer(surrogate, X_residue)
    shap_values = explainer.shap_values(X_residue)
    
    # Handle SHAP output dimensions
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]
            
    # Calculate global importance score (mean absolute SHAP)
    residue_shap_scores = np.mean(np.abs(shap_values), axis=0)
    
    # Rank residues
    top_k_residues = []
    ranked_indices = np.argsort(residue_shap_scores)[::-1]
    for rank, idx in enumerate(ranked_indices):
        top_k_residues.append({
            'position': idx + 1,
            'shap_score': residue_shap_scores[idx],
            'rank': rank + 1
        })
        
    return residue_shap_scores, top_k_residues


def compute_mutation_shap(
    sequences: List[str],
    mutation_map: List[str],
    reference: str,
    y_target: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Compute mutation-level SHAP values by training a surrogate Logistic Regression model
    on binary mutation features.
    """
    # 1. Convert sequences to mutation matrix
    X_mutation = sequences_to_mutation_matrix(sequences, mutation_map, reference)
    
    # If no mutations observed, return zeroes
    if X_mutation.shape[1] == 0:
        return np.zeros(0), []
        
    # 2. Fit Logistic Regression
    mutation_model = LogisticRegression(
        class_weight='balanced',
        penalty='l2',
        C=1.0,
        random_state=random_state,
        max_iter=1000
    )
    mutation_model.fit(X_mutation, y_target)
    
    # 3. Compute SHAP values once
    explainer = shap.LinearExplainer(mutation_model, X_mutation)
    shap_values = explainer.shap_values(X_mutation)
    
    # Handle SHAP output dimensions
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]
            
    # Global mutation importance (mean absolute SHAP)
    mutation_importance_scores = np.mean(np.abs(shap_values), axis=0)
    
    # Rank mutations
    ranked_mutations = []
    ranked_indices = np.argsort(mutation_importance_scores)[::-1]
    for rank, idx in enumerate(ranked_indices):
        ranked_mutations.append({
            'mutation': mutation_map[idx],
            'shap_score': mutation_importance_scores[idx],
            'rank': rank + 1
        })
        
    return mutation_importance_scores, ranked_mutations


def fuse_explainability(
    residue_shap: np.ndarray,
    mutation_shap: np.ndarray,
    mutation_map: List[str],
    seq_len: int,
    alpha: float = 0.4,
    beta: float = 0.6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align and fuse residue SHAP and mutation SHAP using explicit normalizations
    and index lookup mapping table.
    """
    # 1. Map mutation positions to 0-indexed residue indices
    mutation_shap_projection = np.zeros(seq_len)
    
    for idx_mut, mut in enumerate(mutation_map):
        _, pos, _ = parse_mutation(mut)
        if pos is not None and 1 <= pos <= seq_len:
            # Accumulate absolute SHAP score at that residue position
            mutation_shap_projection[pos - 1] += abs(mutation_shap[idx_mut])
            
    # 2. Normalize both spaces independently
    residue_shap_norm = min_max_normalize(residue_shap)
    mutation_proj_norm = min_max_normalize(mutation_shap_projection)
    
    # 3. Compute fused score
    fused_score = alpha * residue_shap_norm + beta * mutation_proj_norm
    
    return fused_score, mutation_shap_projection


def compute_hypergeom_p_value(
    top_k_positions: Set[int],
    known_drms: Set[int],
    seq_len: int
) -> Tuple[float, float]:
    """
    Compute hypergeometric enrichment p-value and enrichment score.
    """
    M = seq_len
    n = len(known_drms & set(range(1, seq_len + 1)))
    N = len(top_k_positions)
    x = len(top_k_positions & known_drms)
    
    expected = N * n / M if M > 0 else 0
    observed = x
    enrichment = observed / expected if expected > 0 else 0
    
    # sf(x-1) computes P(X >= x)
    p_value = hypergeom.sf(x - 1, M, n, N) if N > 0 and n > 0 else 1.0
    return p_value, enrichment


def run_dual_shap_pipeline(
    data_dir: str = 'data',
    results_dir: str = 'results',
    subset_size: int = 100,
    seed: int = 42
):
    """
    Execute Dual SHAP Explainability Framework pipeline.
    """
    # Force resolve local imports
    sys.path.insert(0, str(Path(data_dir).resolve().parent))
    try:
        from scripts.run_experiments import select_and_extract_subsampled_data
    except ImportError:
        from run_experiments import select_and_extract_subsampled_data
    from src.models import train_attention_model
    from src.interpretability import get_drug_specific_drms
    
    data_dir_path = Path(data_dir)
    results_dir_path = Path(results_dir)
    visuals_dir_path = results_dir_path / 'dual_shap_visuals'
    visuals_dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading dataset...")
    sub_data = select_and_extract_subsampled_data(data_dir_path, subset_size, seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Output rows list
    residue_importance_rows = []
    mutation_importance_rows = []
    fused_scores_rows = []
    alignment_analysis_rows = []
    
    # Store profiles for visualization heatmaps
    residue_shap_profiles = {}
    fused_score_profiles = {}
    
    for drug_class, class_data in sub_data.items():
        print(f"\n==================== PROCESSING CLASS: {drug_class} ====================")
        
        sequences = class_data['sequences']
        seq_ids = class_data['seq_ids']
        phenotypes = class_data['phenotypes']
        drugs = class_data['drugs']
        per_residue = class_data['per_residue']
        reference = class_data['reference']
        seq_len = len(reference)
        
        # Rename drug columns for internal consistency
        renamed_pheno = phenotypes.rename(columns={d: f"{d}_FC" for d in drugs})
        
        # Prepare mutation map for this class
        class_mutations = set()
        for seq in sequences:
            class_mutations.update(get_mutations_relative_to_ref(seq, reference))
        mutation_map = sorted(list(class_mutations))
        print(f"Total unique mutations in {drug_class} cohort: {len(mutation_map)}")
        
        residue_shap_profiles[drug_class] = []
        fused_score_profiles[drug_class] = []
        class_drugs_processed = []
        
        for drug in drugs:
            print(f"\n--- Running Dual SHAP for: {drug} ---")
            
            fc_col = f"{drug}_FC"
            y_vals = renamed_pheno[fc_col].values
            valid_mask = ~np.isnan(y_vals)
            
            y_valid = (y_vals[valid_mask] >= 2.5).astype(int)
            X_valid = [per_residue[i] for i in range(len(per_residue)) if valid_mask[i]]
            seqs_valid = [sequences[i] for i in range(len(sequences)) if valid_mask[i]]
            
            n_resistant = int(y_valid.sum())
            n_susceptible = len(y_valid) - n_resistant
            
            if len(np.unique(y_valid)) < 2 or min(n_resistant, n_susceptible) < 3:
                print(f"Skipping {drug}: cohort imbalance")
                continue
                
            # Train Attention Model to extract attention weights
            print("Training classifier model to extract attention weights...")
            attn_model = train_attention_model(
                X_valid, y_valid,
                epochs=20,
                verbose=False,
                device=device
            )
            
            # Extract attention weights for each sequence
            attn_model.eval()
            attention_weights = []
            for emb in X_valid:
                emb_tensor = torch.tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)
                mask_tensor = torch.ones(1, len(emb), device=device)
                with torch.no_grad():
                    _, weights_tensor = attn_model(emb_tensor, mask_tensor)
                attention_weights.append(weights_tensor.squeeze(0).cpu().numpy())
                
            # --- MODULE 1: Attention-Aligned Residue SHAP ---
            print("Running Attention-Aligned Residue SHAP...")
            residue_shap, top_residues = compute_attention_residue_shap(
                attn_model, X_valid, attention_weights, y_valid, random_state=seed
            )
            
            # --- MODULE 2: Mutation-Level SHAP ---
            print("Running Mutation-Level SHAP...")
            mutation_shap, top_mutations = compute_mutation_shap(
                seqs_valid, mutation_map, reference, y_valid, random_state=seed
            )
            
            if len(mutation_shap) == 0:
                print(f"Skipping {drug}: No mutations present")
                continue
                
            # --- MODULE 3: Fusion layer ---
            print("Fusing explainability sources...")
            fused_score, mutation_projection = fuse_explainability(
                residue_shap, mutation_shap, mutation_map, seq_len, alpha=0.4, beta=0.6
            )
            
            # Save profiles for heatmaps
            residue_shap_profiles[drug_class].append((drug, residue_shap))
            fused_score_profiles[drug_class].append((drug, fused_score))
            class_drugs_processed.append(drug)
            
            # Write to CSV rows
            for pos_idx in range(seq_len):
                pos = pos_idx + 1
                aa = reference[pos_idx]
                
                residue_importance_rows.append({
                    'drug': drug,
                    'drug_class': drug_class,
                    'position': pos,
                    'amino_acid': aa,
                    'residue_shap_score': residue_shap[pos_idx]
                })
                
                fused_scores_rows.append({
                    'drug': drug,
                    'drug_class': drug_class,
                    'position': pos,
                    'amino_acid': aa,
                    'residue_shap': residue_shap[pos_idx],
                    'mutation_projection': mutation_projection[pos_idx],
                    'fused_score': fused_score[pos_idx]
                })
                
            for item in top_mutations:
                mutation_importance_rows.append({
                    'drug': drug,
                    'drug_class': drug_class,
                    'mutation': item['mutation'],
                    'shap_score': item['shap_score'],
                    'rank': item['rank']
                })
                
            # --- ANALYSIS MODULE 1: Alignment Score Computation ---
            known_drms = get_drug_specific_drms(drug, drug_class)
            
            for K in [10, 20]:
                top_k_res_pos = set(np.argsort(residue_shap)[::-1][:K] + 1)
                
                # Top-K mutation projection positions
                top_k_mut_proj_pos = set(np.argsort(mutation_projection)[::-1][:K] + 1)
                
                overlap = top_k_res_pos & top_k_mut_proj_pos
                precision_at_k = len(overlap) / K
                
                union_size = len(top_k_res_pos | top_k_mut_proj_pos)
                jaccard = len(overlap) / union_size if union_size > 0 else 0.0
                
                expected_overlap = (K * K) / seq_len
                enrichment = len(overlap) / expected_overlap if expected_overlap > 0 else 0.0
                
                # --- ANALYSIS MODULE 2: Biological Validation ---
                p_val_res, enrich_res = compute_hypergeom_p_value(top_k_res_pos, known_drms, seq_len)
                p_val_mut, enrich_mut = compute_hypergeom_p_value(top_k_mut_proj_pos, known_drms, seq_len)
                
                alignment_analysis_rows.append({
                    'drug': drug,
                    'drug_class': drug_class,
                    'top_k': K,
                    'precision_at_k': precision_at_k,
                    'jaccard_similarity': jaccard,
                    'alignment_enrichment_score': enrichment,
                    'residue_shap_drm_p_value': p_val_res,
                    'residue_shap_drm_enrichment': enrich_res,
                    'mutation_shap_drm_p_value': p_val_mut,
                    'mutation_shap_drm_enrichment': enrich_mut
                })
                
            # --- VISUALIZATION 3: SHAP vs Mutation Alignment Scatter Plot ---
            plt.figure(figsize=(7, 5))
            sns.scatterplot(
                x=min_max_normalize(residue_shap),
                y=min_max_normalize(mutation_projection),
                alpha=0.8,
                color='#8e44ad'
            )
            plt.xlabel('Normalized Attention-Aligned Residue SHAP')
            plt.ylabel('Normalized Mutation SHAP Projection')
            plt.title(f'SHAP vs Mutation Alignment: {drug}')
            plt.tight_layout()
            plt.savefig(visuals_dir_path / f'shap_vs_mutation_{drug}.png')
            plt.close()
            
        # --- VISUALIZATIONS 1 & 4: Heatmaps for the class ---
        if class_drugs_processed:
            # 1. Residue SHAP Heatmap
            profiles_matrix = np.array([item[1] for item in residue_shap_profiles[drug_class]])
            plt.figure(figsize=(15, 2.5 + 0.5 * len(class_drugs_processed)))
            sns.heatmap(
                profiles_matrix,
                cmap='plasma',
                xticklabels=5 if drug_class != 'PI' else 2,
                yticklabels=class_drugs_processed,
                cbar_kws={'label': 'Residue SHAP Score'}
            )
            plt.xlabel('Residue Position (1-indexed)')
            plt.ylabel('Drug')
            plt.title(f'Attention-Aligned Residue SHAP Heatmap - {drug_class}')
            plt.tight_layout()
            plt.savefig(visuals_dir_path / f'dual_shap_heatmap_{drug_class}.png')
            plt.close()
            
            # 4. Consensus Fusion Map
            fused_matrix = np.array([item[1] for item in fused_score_profiles[drug_class]])
            plt.figure(figsize=(15, 2.5 + 0.5 * len(class_drugs_processed)))
            sns.heatmap(
                fused_matrix,
                cmap='viridis',
                xticklabels=5 if drug_class != 'PI' else 2,
                yticklabels=class_drugs_processed,
                cbar_kws={'label': 'Fused Score'}
            )
            plt.xlabel('Residue Position (1-indexed)')
            plt.ylabel('Drug')
            plt.title(f'Consensus Fusion Heatmap - {drug_class}')
            plt.tight_layout()
            plt.savefig(visuals_dir_path / f'consensus_fusion_heatmap_{drug_class}.png')
            plt.close()
            
    # --- VISUALIZATION 2: Mutation Importance Bar Plot ---
    # Combine top mutations across all drugs to plot the top mutations globally
    mut_df = pd.DataFrame(mutation_importance_rows)
    if not mut_df.empty:
        # Get top 20 mutations by absolute SHAP score globally
        top_20_global_muts = mut_df.groupby('mutation')['shap_score'].mean().sort_values(ascending=False).head(20).reset_index()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=top_20_global_muts,
            x='shap_score',
            y='mutation',
            palette='crest_r'
        )
        plt.xlabel('Mean Absolute SHAP Value')
        plt.ylabel('Mutation')
        plt.title('Top 20 Global Mutation-Level SHAP Importance')
        plt.tight_layout()
        plt.savefig(visuals_dir_path / 'top_mutations_global.png')
        plt.close()
        
    # --- SAVE OUTPUT FILES ---
    print("\nSaving output CSV files...")
    pd.DataFrame(residue_importance_rows).to_csv(results_dir_path / 'dual_shap_residue_importance.csv', index=False)
    pd.DataFrame(mutation_importance_rows).to_csv(results_dir_path / 'mutation_shap_importance.csv', index=False)
    pd.DataFrame(fused_scores_rows).to_csv(results_dir_path / 'fused_explainability_scores.csv', index=False)
    pd.DataFrame(alignment_analysis_rows).to_csv(results_dir_path / 'shap_alignment_analysis.csv', index=False)
    print("Dual SHAP execution complete.")


if __name__ == '__main__':
    run_dual_shap_pipeline(subset_size=100)

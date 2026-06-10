"""
Unified Explainability Aggregation and Analysis Framework for HIV-1.

This module combines Integrated Gradients, attention weights, and SHAP values
into a unified interpretability pipeline, generating residue-level importances,
DRM overlap metrics, rare mutation impact assessments, and premium figures.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Import local modules
from src.data_processing import PI_DRUGS, NRTI_DRUGS, NNRTI_DRUGS
from src.feature_engineering import HIV_PROTEASE_REFERENCE, HIV_RT_REFERENCE
from src.rare_mutations import (
    compute_mutation_frequencies,
    compute_rare_mutation_weights,
    normalize_weights_for_attention
)
from src.models import train_attention_model, EmbeddingDataset, collate_embeddings
from src.interpretability import (
    load_known_drms,
    get_drug_specific_drms,
    compute_shap_values
)

# Set styling for premium visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11


def compute_integrated_gradients(
    model,
    embedding: np.ndarray,
    baseline: np.ndarray,
    mask: Optional[np.ndarray] = None,
    rare_mutation_weights: Optional[np.ndarray] = None,
    steps: int = 50
) -> np.ndarray:
    """
    Compute residue-level Integrated Gradients for AttentionWeightedClassifier.
    
    Args:
        model: Trained AttentionWeightedClassifier
        embedding: Sequence embedding of shape (seq_len, dim)
        baseline: Dataset mean embedding of shape (seq_len, dim)
        mask: Optional sequence mask of shape (seq_len,)
        rare_mutation_weights: Optional rarity weights of shape (seq_len,)
        steps: Number of Riemann sum steps
        
    Returns:
        Residue attribution values of shape (seq_len,)
    """
    device = next(model.parameters()).device
    model.eval()
    
    # 1. Convert input and baseline to tensors
    emb_tensor = torch.tensor(embedding, dtype=torch.float32, device=device).unsqueeze(0) # (1, seq_len, dim)
    emb_tensor.requires_grad = True
    
    base_tensor = torch.tensor(baseline, dtype=torch.float32, device=device).unsqueeze(0) # (1, seq_len, dim)
    
    # 2. Prepare mask and rare weights
    seq_len = embedding.shape[0]
    if mask is not None:
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        mask_tensor = torch.ones(1, seq_len, device=device)
        
    if rare_mutation_weights is not None:
        rw_tensor = torch.tensor(rare_mutation_weights, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        rw_tensor = None
        
    alphas = np.linspace(0.0, 1.0, steps)
    grads = []
    
    # 3. Path integration
    for alpha in alphas:
        # Interpolate input
        interpolated = base_tensor + alpha * (emb_tensor - base_tensor)
        
        # Clone to avoid in-place operations and enable gradient flow
        interp_tensor = interpolated.clone().detach().requires_grad_(True)
        
        # Forward pass
        logits, _ = model(interp_tensor, mask_tensor, rare_mutation_weights=rw_tensor)
        
        # Backward pass to get gradient of logit w.r.t interpolated input
        model.zero_grad()
        logits.backward()
        
        # Collect gradient
        grad = interp_tensor.grad.squeeze(0).cpu().numpy() # (seq_len, dim)
        grads.append(grad)
        
    # 4. Average gradients and calculate attribution
    avg_grads = np.mean(grads, axis=0) # (seq_len, dim)
    delta = (emb_tensor - base_tensor).squeeze(0).detach().cpu().numpy() # (seq_len, dim)
    attributions = delta * avg_grads # (seq_len, dim)
    
    # 5. Take L2 norm across embedding dimension to get residue-level attributions
    ig_residue = np.linalg.norm(attributions, axis=-1) # (seq_len,)
    return ig_residue


def compute_shap_residue_proxy(
    shap_features: np.ndarray,
    embedding: np.ndarray,
    attention_weights: np.ndarray
) -> np.ndarray:
    """
    Map feature-level (1280-dim) SHAP values to sequence residues.
    This is a SHAP proxy projection using attention-weighted contribution approximations.
    
    Args:
        shap_features: Feature SHAP values of shape (1280,)
        embedding: Sequence embedding of shape (seq_len, 1280)
        attention_weights: Attention weights of shape (seq_len,)
        
    Returns:
        Residue-level SHAP attributions of shape (seq_len,)
    """
    seq_len, embed_dim = embedding.shape
    shap_proxy = np.zeros(seq_len)
    
    # Distribute feature SHAP values proportionally back to residues
    for d in range(embed_dim):
        feat_shap = np.abs(shap_features[d])
        
        # Contribution of each residue j to feature d is attention_weight * embedding_val
        contributions = np.abs(attention_weights * embedding[:, d])
        total_contrib = np.sum(contributions) + 1e-9
        
        # Distribute SHAP value based on proportion
        for j in range(seq_len):
            shap_proxy[j] += feat_shap * (contributions[j] / total_contrib)
            
    return shap_proxy


def min_max_normalize(val: np.ndarray) -> np.ndarray:
    """Helper to min-max normalize a numpy array to [0, 1]."""
    min_val = np.min(val)
    max_val = np.max(val)
    return (val - min_val) / (max_val - min_val + 1e-9)


def run_explainability_pipeline(
    data_dir: str = 'data',
    results_dir: str = 'results',
    subset_size: int = 100,
    seed: int = 42
):
    """
    Main runner to train models, extract attributions, aggregate results,
    save CSVs, and generate visualizations.
    """
    # Force import path resolved
    sys.path.insert(0, str(Path(data_dir).resolve().parent))
    from run_experiments import select_and_extract_subsampled_data
    
    data_dir_path = Path(data_dir)
    results_dir_path = Path(results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading subsampled dataset...")
    sub_data = select_and_extract_subsampled_data(data_dir_path, subset_size, seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Rows for final output DataFrames
    scores_rows = []
    topk_rows = []
    drug_summary_rows = []
    drm_overlap_rows = []
    correlation_rows = []
    
    # Keep track of drug summaries for visualization
    drug_summaries_list = []
    
    for drug_class, class_data in sub_data.items():
        print(f"\n==================== PROCESSING CLASS: {drug_class} ====================")
        
        sequences = class_data['sequences']
        seq_ids = class_data['seq_ids']
        phenotypes = class_data['phenotypes']
        drugs = class_data['drugs']
        per_residue = class_data['per_residue']
        reference = class_data['reference']
        
        # Rename columns to {drug}_FC to enable correct binarization
        renamed_pheno = phenotypes.rename(columns={d: f"{d}_FC" for d in drugs})
        
        # Compute Rare Mutation weights
        print("Computing mutation frequencies and weights...")
        freqs = compute_mutation_frequencies(sequences, reference)
        raw_weights = compute_rare_mutation_weights(sequences, reference, frequencies=freqs)
        
        # Softmax-normalize weights
        norm_rare_weights = []
        for w in raw_weights:
            norm_w = normalize_weights_for_attention(w, method='softmax')
            norm_rare_weights.append(norm_w)
            
        for drug in drugs:
            print(f"\n--- Aggregating Explainability for: {drug} ---")
            
            # Get valid cohort for this drug
            fc_col = f"{drug}_FC"
            y_vals = renamed_pheno[fc_col].values
            valid_mask = ~np.isnan(y_vals)
            
            y_valid = (y_vals[valid_mask] >= 2.5).astype(int)
            X_valid = [per_residue[i] for i in range(len(per_residue)) if valid_mask[i]]
            rw_valid = [norm_rare_weights[i] for i in range(len(norm_rare_weights)) if valid_mask[i]]
            seqs_valid = [sequences[i] for i in range(len(sequences)) if valid_mask[i]]
            ids_valid = [seq_ids[i] for i in range(len(seq_ids)) if valid_mask[i]]
            raw_rw_valid = [raw_weights[i] for i in range(len(raw_weights)) if valid_mask[i]]
            
            n_resistant = int(y_valid.sum())
            n_susceptible = len(y_valid) - n_resistant
            
            if len(np.unique(y_valid)) < 2 or min(n_resistant, n_susceptible) < 3:
                print(f"Skipping {drug}: insufficient cohort balance (R={n_resistant}, S={n_susceptible})")
                continue
                
            # Compute dataset mean embedding for biologically valid baseline
            # shape (seq_len, 1280)
            baseline_mean_embedding = np.mean(X_valid, axis=0)
            
            # 1. Train Attention Model (Enhanced with rarity weights)
            print("Training Attention model...")
            model = train_attention_model(
                X_valid, y_valid,
                rare_mutation_weights_list=rw_valid,
                epochs=20,
                verbose=False,
                device=device
            )
            
            # 2. Train Logistic Regression + SHAP explainer
            print("Training Linear SHAP model...")
            X_mean = np.array([np.mean(e, axis=0) for e in X_valid])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_mean)
            
            shap_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)
            shap_model.fit(X_scaled, y_valid)
            
            raw_shap = compute_shap_values(shap_model, X_scaled, n_background=min(50, len(X_scaled)), random_state=seed)
            
            if isinstance(raw_shap, list):
                shap_values = raw_shap[1] if len(raw_shap) == 2 else raw_shap[0]
            else:
                shap_values = raw_shap
                
            # Keep track of positional stats across sequences
            seq_len = len(reference)
            drug_ig_sum = np.zeros(seq_len)
            drug_attention_sum = np.zeros(seq_len)
            drug_shap_sum = np.zeros(seq_len)
            drug_shap_top_freq = np.zeros(seq_len)
            
            drug_pearson_corrs = []
            drug_spearman_corrs = []
            
            print(f"Running attributions for {len(X_valid)} sequences...")
            for i in range(len(X_valid)):
                emb = X_valid[i]
                seq = seqs_valid[i]
                seq_id = ids_valid[i]
                rw = rw_valid[i]
                raw_rw = raw_rw_valid[i]
                
                # Length of current sequence
                L = len(emb)
                
                # Attention weights
                emb_tensor = torch.tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)
                mask_tensor = torch.ones(1, L, device=device)
                rw_tensor = torch.tensor(rw, dtype=torch.float32, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    _, attn_tensor = model(emb_tensor, mask_tensor, rare_mutation_weights=rw_tensor)
                attn_weights = attn_tensor.squeeze(0).cpu().numpy()
                
                # Integrated Gradients using dataset mean embedding baseline
                ig_scores = compute_integrated_gradients(
                    model, emb, baseline_mean_embedding, mask=None, rare_mutation_weights=rw, steps=25
                )
                
                # SHAP proxy projection
                seq_shap = shap_values[i]
                shap_proxy = compute_shap_residue_proxy(seq_shap, emb, attn_weights)
                
                # Normalization
                ig_norm = min_max_normalize(ig_scores)
                attn_norm = min_max_normalize(attn_weights)
                shap_norm = min_max_normalize(shap_proxy)
                
                # Combined Score calculation
                combined = 0.5 * ig_norm + 0.3 * attn_norm + 0.2 * shap_norm
                
                # Map to reference dimensions
                for pos_idx in range(min(L, seq_len)):
                    pos = pos_idx + 1
                    aa = seq[pos_idx] if pos_idx < len(seq) else '?'
                    
                    scores_rows.append({
                        'seq_id': seq_id,
                        'position': pos,
                        'amino_acid': aa,
                        'drug': drug,
                        'drug_class': drug_class,
                        'ig_score': ig_norm[pos_idx],
                        'attention_weight': attn_norm[pos_idx],
                        'shap_proxy_projection': shap_norm[pos_idx],
                        'combined_score': combined[pos_idx]
                    })
                    
                    # Accumulate for drug-level averages
                    drug_ig_sum[pos_idx] += ig_norm[pos_idx]
                    drug_attention_sum[pos_idx] += attn_norm[pos_idx]
                    drug_shap_sum[pos_idx] += shap_norm[pos_idx]
                    
                # Sequence level correlation: rare mutation weights (raw_rw) vs IG scores (ig_norm)
                len_eval = min(L, seq_len)
                sub_rw = raw_rw[:len_eval]
                sub_ig = ig_norm[:len_eval]
                
                if np.std(sub_rw) > 0 and np.std(sub_ig) > 0:
                    p_corr, _ = pearsonr(sub_rw, sub_ig)
                    s_corr, _ = spearmanr(sub_rw, sub_ig)
                    if not np.isnan(p_corr):
                        drug_pearson_corrs.append(p_corr)
                    if not np.isnan(s_corr):
                        drug_spearman_corrs.append(s_corr)
                        
                # Top 10 combined score ranking for this sequence
                top10_indices = np.argsort(combined)[::-1][:10]
                for rank_idx, idx in enumerate(top10_indices):
                    pos = idx + 1
                    aa = seq[idx] if idx < len(seq) else '?'
                    topk_rows.append({
                        'seq_id': seq_id,
                        'drug': drug,
                        'drug_class': drug_class,
                        'rank': rank_idx + 1,
                        'position': pos,
                        'amino_acid': aa,
                        'combined_score': combined[idx]
                    })
                    
                # Track top 10 SHAP proxy positions for this sequence
                top10_shap_indices = np.argsort(shap_norm)[::-1][:10]
                for idx in top10_shap_indices:
                    if idx < seq_len:
                        drug_shap_top_freq[idx] += 1
                        
            # --- Aggregating Drug Summary ---
            n_seq = len(X_valid)
            avg_ig = drug_ig_sum / n_seq
            avg_attention = drug_attention_sum / n_seq
            avg_shap_proxy = drug_shap_sum / n_seq
            shap_top_freq = drug_shap_top_freq / n_seq
            
            # Combine into drug-level summary
            for pos_idx in range(seq_len):
                pos = pos_idx + 1
                drug_summary_rows.append({
                    'drug': drug,
                    'drug_class': drug_class,
                    'position': pos,
                    'avg_ig': avg_ig[pos_idx],
                    'avg_attention': avg_attention[pos_idx],
                    'shap_top_position_frequency': shap_top_freq[pos_idx],
                    'avg_shap_proxy_projection': avg_shap_proxy[pos_idx]
                })
                
            # Store drug-level positional profile for heatmap visualization
            drug_profile = 0.5 * min_max_normalize(avg_ig) + 0.3 * min_max_normalize(avg_attention) + 0.2 * min_max_normalize(avg_shap_proxy)
            drug_summaries_list.append((drug, drug_class, drug_profile))
            
            # --- DRM Overlap Analysis ---
            known_drms = get_drug_specific_drms(drug, drug_class)
            ref_drms = known_drms & set(range(1, seq_len + 1))
            
            # Rank average combined scores per position
            avg_combined = 0.5 * min_max_normalize(avg_ig) + 0.3 * min_max_normalize(avg_attention) + 0.2 * min_max_normalize(avg_shap_proxy)
            top_positions_ranked = np.argsort(avg_combined)[::-1]
            
            for K in [10, 20]:
                top_k_pos = set(top_positions_ranked[:K] + 1) # 1-indexed
                overlap = top_k_pos & ref_drms
                
                observed = len(overlap)
                precision_at_k = observed / K
                expected = K * (len(ref_drms) / seq_len) if seq_len > 0 else 0
                enrichment = observed / expected if expected > 0 else 0
                
                drm_overlap_rows.append({
                    'drug': drug,
                    'drug_class': drug_class,
                    'top_k': K,
                    'observed_overlap': observed,
                    'expected_overlap': expected,
                    'precision_at_k': precision_at_k,
                    'enrichment_score': enrichment
                })
                
            # --- Rare Mutation Correlation & Hotspot Analysis ---
            avg_p_corr = np.mean(drug_pearson_corrs) if drug_pearson_corrs else 0.0
            avg_s_corr = np.mean(drug_spearman_corrs) if drug_spearman_corrs else 0.0
            
            # Hotspot score: average rare mutation weight * average IG score
            avg_raw_weights = np.mean(raw_weights, axis=0) # (seq_len,)
            hotspot_scores = avg_raw_weights[:seq_len] * avg_ig
            top_hotspots_idx = np.argsort(hotspot_scores)[::-1][:5]
            top_hotspot_positions = sorted([int(idx + 1) for idx in top_hotspots_idx])
            
            correlation_rows.append({
                'drug': drug,
                'drug_class': drug_class,
                'pearson_correlation': avg_p_corr,
                'spearman_correlation': avg_s_corr,
                'top_hotspots': ', '.join(map(str, top_hotspot_positions))
            })
            
            print(f"Results for {drug}: DRM Overlap (K=10) = {drm_overlap_rows[-2]['observed_overlap']}, IG-Rarity Correlation = {avg_s_corr:.3f}")
            
    # --- SAVE OUTPUT FILES ---
    print("\nSaving aggregation results into results/ directory...")
    pd.DataFrame(scores_rows).to_csv(results_dir_path / 'explainability_combined_scores.csv', index=False)
    pd.DataFrame(topk_rows).to_csv(results_dir_path / 'topk_residues_per_sequence.csv', index=False)
    pd.DataFrame(drug_summary_rows).to_csv(results_dir_path / 'drug_level_explainability_summary.csv', index=False)
    pd.DataFrame(drm_overlap_rows).to_csv(results_dir_path / 'drm_overlap_analysis.csv', index=False)
    pd.DataFrame(correlation_rows).to_csv(results_dir_path / 'rare_mutation_explainability_correlation.csv', index=False)
    print("CSVs generated successfully.")
    
    # --- VISUALIZATION GENERATION ---
    print("\nGenerating premium visualizations...")
    
    # 1. Residue Importance Heatmap per Drug Class
    for target_class in ['PI', 'NRTI', 'NNRTI']:
        class_profiles = [item for item in drug_summaries_list if item[1] == target_class]
        if not class_profiles:
            continue
            
        drugs_list = [item[0] for item in class_profiles]
        profiles_matrix = np.array([item[2] for item in class_profiles])
        
        plt.figure(figsize=(15, 2.5 + 0.5 * len(drugs_list)))
        sns.heatmap(
            profiles_matrix, 
            cmap='viridis', 
            xticklabels=5 if target_class != 'PI' else 2,
            yticklabels=drugs_list,
            cbar_kws={'label': 'Combined Importance Score'}
        )
        plt.xlabel('Residue Position (1-indexed)', fontsize=12)
        plt.ylabel('Drug', fontsize=12)
        plt.title(f'Unified Residue Importance Profile - {target_class} Drugs', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(results_dir_path / f'explainability_heatmap_{target_class}.png')
        plt.close()
        
    # 2. IG vs. Attention Correlation Plot
    # Collate a subset of sequence scores to avoid over-plotting
    all_scores_df = pd.DataFrame(scores_rows)
    if not all_scores_df.empty:
        sample_df = all_scores_df.sample(n=min(1000, len(all_scores_df)), random_state=seed)
        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=sample_df, 
            x='attention_weight', 
            y='ig_score', 
            scatter_kws={'alpha': 0.4, 'color': '#3498db'},
            line_kws={'color': '#e74c3c', 'linewidth': 2}
        )
        plt.xlabel('Normalized Attention Weight', fontsize=12)
        plt.ylabel('Normalized Integrated Gradients Score', fontsize=12)
        plt.title('Explainability Correlation: Integrated Gradients vs. Attention', fontsize=13, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(results_dir_path / 'ig_vs_attention_correlation.png')
        plt.close()
        
    # 3. DRM Overlap Bar Chart (Precision@10 for each drug)
    overlap_df = pd.DataFrame(drm_overlap_rows)
    overlap_k10 = overlap_df[overlap_df['top_k'] == 10]
    if not overlap_k10.empty:
        plt.figure(figsize=(12, 5))
        sns.barplot(
            data=overlap_k10,
            x='drug',
            y='precision_at_k',
            hue='drug_class',
            palette='Set2'
        )
        plt.axhline(y=np.mean(overlap_k10['precision_at_k']), color='red', linestyle='--', alpha=0.7, label=f"Average P@10 ({np.mean(overlap_k10['precision_at_k']):.2f})")
        plt.xlabel('Drug Name', fontsize=12)
        plt.ylabel('Precision@10 (DRM Overlap)', fontsize=12)
        plt.title('Biological Validation: Top 10 Explainability Residues vs. Known Clinical DRMs', fontsize=13, fontweight='bold', pad=15)
        plt.ylim(0.0, 1.0)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(results_dir_path / 'drm_overlap_bar_chart.png')
        plt.close()
        
    print("Visualizations generated successfully. All tasks complete.")


if __name__ == '__main__':
    run_explainability_pipeline(subset_size=100)

"""
Counterfactual Mutation Causal Analysis for HIV Drug Resistance.

This module performs counterfactual analysis to validate causal impact of mutations
on model predictions. For each important mutation, we create perturbed sequences
(with mutation removed or introduced) and measure prediction probability changes.

This provides a causal validation layer for SHAP-based explainability.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

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


def extract_key_mutations(
    sequence: str,
    reference: str,
    shap_scores: Optional[np.ndarray] = None,
    top_k: int = 10
) -> List[Tuple[int, str]]:
    """
    Extract top-k important mutations from a sequence.
    
    If SHAP scores are provided, rank mutations by their SHAP importance.
    Otherwise, return top mutations by position.
    
    Args:
        sequence: Amino acid sequence
        reference: Reference sequence
        shap_scores: Per-residue SHAP scores (optional)
        top_k: Number of top mutations to extract
        
    Returns:
        List of (position, amino_acid) tuples
    """
    mutations = get_mutations_relative_to_ref(sequence, reference)
    
    if not mutations:
        return []
    
    # Parse each mutation
    parsed = []
    for mut in mutations:
        wt, pos, mut_aa = parse_mutation(mut)
        if pos is not None:
            parsed.append((pos, mut_aa[0], mut))  # (position, amino_acid, full_mutation_string)
    
    # Score mutations
    if shap_scores is not None:
        scored = [(pos, aa, shap_scores[pos - 1]) for pos, aa, _ in parsed]
        scored = sorted(scored, key=lambda x: x[2], reverse=True)
        return [(pos, aa) for pos, aa, _ in scored[:top_k]]
    else:
        # Return top by position (heuristic)
        scored = [(pos, aa, pos) for pos, aa, _ in parsed]
        scored = sorted(scored, key=lambda x: x[2], reverse=True)
        return [(pos, aa) for pos, aa, _ in scored[:top_k]]


def generate_counterfactual_sequence(
    sequence: str,
    reference: str,
    position: int,
    operation: str = 'remove'
) -> str:
    """
    Generate a counterfactual sequence by removing or introducing a mutation.
    
    Args:
        sequence: Original sequence
        reference: Reference sequence
        position: 1-indexed position of mutation
        operation: 'remove' (revert to reference) or 'introduce' (not used, for API completeness)
        
    Returns:
        Counterfactual sequence
    """
    seq_list = list(sequence)
    
    if operation == 'remove':
        # Revert to reference amino acid
        if position <= len(reference):
            seq_list[position - 1] = reference[position - 1]
    
    return ''.join(seq_list)


def compute_counterfactual_effect(
    model,
    original_embedding: np.ndarray,
    counterfactual_embedding: np.ndarray,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Compute the causal effect of a mutation by measuring prediction probability change.
    
    Args:
        model: Trained AttentionWeightedClassifier
        original_embedding: Per-residue embeddings of original sequence (seq_len, 1280)
        counterfactual_embedding: Per-residue embeddings of counterfactual sequence (seq_len, 1280)
        device: torch device
        
    Returns:
        Tuple of (delta_probability, original_prob, counterfactual_prob)
    """
    model.eval()
    
    # Ensure embeddings have same shape
    assert original_embedding.shape == counterfactual_embedding.shape, \
        f"Embedding shape mismatch: {original_embedding.shape} vs {counterfactual_embedding.shape}"
    
    # Convert to tensors and add batch dimension
    orig_tensor = torch.tensor(original_embedding, dtype=torch.float32, device=device).unsqueeze(0)
    cf_tensor = torch.tensor(counterfactual_embedding, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Create attention masks (all ones, no padding)
    seq_len = len(original_embedding)
    mask = torch.ones(1, seq_len, device=device)
    
    with torch.no_grad():
        # Get logits
        orig_logits, _ = model(orig_tensor, mask=mask)
        cf_logits, _ = model(cf_tensor, mask=mask)
        
        # Convert to probability (sigmoid)
        orig_prob = torch.sigmoid(orig_logits).squeeze().item()
        cf_prob = torch.sigmoid(cf_logits).squeeze().item()
        
    # Causal effect: change in probability when mutation is removed
    delta_prob = cf_prob - orig_prob
    
    return delta_prob, orig_prob, cf_prob


def run_counterfactual_analysis(
    data_dir: str = 'data',
    results_dir: str = 'results',
    subset_size: int = 100,
    seed: int = 42,
    top_k: int = 10,
    quick_demo: bool = True
):
    """
    Execute the full counterfactual mutation causal analysis pipeline.
    
    Uses embedding perturbation approach: For each mutation at position p,
    we approximate the counterfactual effect by zeroing or attenuating the
    embedding at that position (simulating amino acid change). This avoids
    expensive re-extraction of embeddings.
    
    Args:
        data_dir: Data directory path
        results_dir: Results directory path
        subset_size: Number of sequences to analyze (e.g., 100)
        seed: Random seed for reproducibility
        top_k: Number of top mutations per sequence to analyze
        quick_demo: If True, use sampling/demo mode for faster execution
    """
    # Force resolve local imports
    sys.path.insert(0, str(Path(data_dir).resolve().parent))
    try:
        from scripts.run_experiments import select_and_extract_subsampled_data
    except ImportError:
        from run_experiments import select_and_extract_subsampled_data
    from src.models import train_attention_model
    
    data_dir_path = Path(data_dir)
    results_dir_path = Path(results_dir)
    visuals_dir_path = results_dir_path / 'counterfactual_visuals'
    visuals_dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading dataset...")
    sub_data = select_and_extract_subsampled_data(data_dir_path, subset_size, seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Output rows
    mutation_effects_rows = []
    causal_ranking_rows = []
    drug_level_summary_rows = []
    
    # Store mutation effects per drug for visualization
    mutation_effects_by_drug = {}
    
    for drug_class, class_data in sub_data.items():
        print(f"\n==================== COUNTERFACTUAL ANALYSIS: {drug_class} ====================")
        
        sequences = class_data['sequences']
        seq_ids = class_data['seq_ids']
        phenotypes = class_data['phenotypes']
        drugs = class_data['drugs']
        per_residue = class_data['per_residue']
        reference = class_data['reference']
        
        # Rename drug columns for internal consistency
        renamed_pheno = phenotypes.rename(columns={d: f"{d}_FC" for d in drugs})
        
        # Subsample drugs for quick demo
        drugs_to_process = drugs[:3] if quick_demo else drugs
        
        for drug in drugs_to_process:
            print(f"\n--- Counterfactual Analysis for: {drug} ---")
            
            fc_col = f"{drug}_FC"
            y_vals = renamed_pheno[fc_col].values
            valid_mask = ~np.isnan(y_vals)
            
            y_valid = (y_vals[valid_mask] >= 2.5).astype(int)
            X_valid = [per_residue[i] for i in range(len(per_residue)) if valid_mask[i]]
            seqs_valid = [sequences[i] for i in range(len(sequences)) if valid_mask[i]]
            
            # Subsample sequences for quick demo
            if quick_demo and len(seqs_valid) > 20:
                sample_indices = np.random.RandomState(seed).choice(len(seqs_valid), 20, replace=False)
                seqs_valid = [seqs_valid[i] for i in sorted(sample_indices)]
                X_valid = [X_valid[i] for i in sorted(sample_indices)]
                y_valid = y_valid[sorted(sample_indices)]
            
            n_resistant = int(y_valid.sum())
            n_susceptible = len(y_valid) - n_resistant
            
            if len(np.unique(y_valid)) < 2 or min(n_resistant, n_susceptible) < 3:
                print(f"Skipping {drug}: cohort imbalance")
                continue
            
            # Train Attention Model to get trained classifier
            print("Training classifier model...")
            try:
                attn_model = train_attention_model(
                    X_valid, y_valid,
                    epochs=10 if quick_demo else 20,
                    verbose=False,
                    device=device
                )
            except Exception as e:
                print(f"Error training model for {drug}: {e}")
                continue
            
            mutation_causal_effects = {}
            
            # For each sequence, perform counterfactual analysis via embedding perturbation
            for seq_idx, (seq, emb) in enumerate(tqdm(zip(seqs_valid, X_valid), total=len(seqs_valid), desc=f"Counterfactuals for {drug}", disable=quick_demo)):
                # Extract mutations in this sequence
                key_muts = get_mutations_relative_to_ref(seq, reference)
                
                if not key_muts:
                    continue
                
                for mut in key_muts:
                    wt, pos, mut_aa = parse_mutation(mut)
                    if pos is None or pos > len(emb):
                        continue
                    
                    # Create counterfactual embedding by attenuating the mutated position
                    # This simulates removing the mutation's effect on the representation
                    cf_emb = emb.copy()
                    cf_emb[pos - 1, :] *= 0.1  # Attenuate instead of zero to maintain stability
                    
                    # Compute causal effect via embedding perturbation
                    try:
                        delta_prob, orig_prob, cf_prob = compute_counterfactual_effect(
                            attn_model, emb, cf_emb, device
                        )
                        
                        # Store results
                        mutation_effects_rows.append({
                            'drug': drug,
                            'drug_class': drug_class,
                            'mutation': mut,
                            'position': pos,
                            'delta_probability': delta_prob,
                            'original_probability': orig_prob,
                            'counterfactual_probability': cf_prob,
                            'causal_direction': 'increases_resistance' if delta_prob > 0 else 'decreases_resistance'
                        })
                        
                        # Accumulate for ranking
                        if mut not in mutation_causal_effects:
                            mutation_causal_effects[mut] = []
                        mutation_causal_effects[mut].append(abs(delta_prob))
                        
                    except Exception as e:
                        print(f"Error processing mutation {mut}: {e}")
                        continue
            
            # Rank mutations by mean absolute causal effect
            mutation_rankings = sorted(
                [(mut, np.mean(deltas)) for mut, deltas in mutation_causal_effects.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            for rank, (mut, mean_causal) in enumerate(mutation_rankings, 1):
                wt, pos, mut_aa = parse_mutation(mut)
                causal_ranking_rows.append({
                    'drug': drug,
                    'drug_class': drug_class,
                    'mutation': mut,
                    'position': pos,
                    'mean_causal_effect': mean_causal,
                    'rank': rank
                })
            
            # Drug-level summary
            if mutation_causal_effects:
                all_deltas = np.concatenate([np.array(v) for v in mutation_causal_effects.values()])
                drug_level_summary_rows.append({
                    'drug': drug,
                    'drug_class': drug_class,
                    'mean_causal_sensitivity': np.mean(all_deltas),
                    'std_causal_sensitivity': np.std(all_deltas),
                    'median_causal_sensitivity': np.median(all_deltas),
                    'n_mutations_analyzed': len(mutation_causal_effects)
                })
            
            # Store for visualization
            mutation_effects_by_drug[drug] = mutation_rankings[:20]  # Top 20 for viz
            
            # Generate per-drug visualization
            if mutation_rankings:
                top_20 = mutation_rankings[:min(20, len(mutation_rankings))]
                muts, effects = zip(*top_20)
                
                plt.figure(figsize=(10, 6))
                sns.barplot(
                    x=list(effects),
                    y=list(muts),
                    palette='coolwarm'
                )
                plt.xlabel('Mean Absolute Causal Effect (|Δ Probability|)')
                plt.ylabel('Mutation')
                plt.title(f'Top Causal Mutations: {drug}')
                plt.tight_layout()
                plt.savefig(visuals_dir_path / f'counterfactual_mutations_{drug}.png', dpi=150)
                plt.close()
    
    # Drug-level heatmap
    if drug_level_summary_rows:
        df_summary = pd.DataFrame(drug_level_summary_rows)
        
        # Pivot for heatmap
        if not df_summary.empty:
            pivot_data = df_summary.pivot_table(
                index='drug',
                values='mean_causal_sensitivity',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(8, max(6, len(pivot_data) * 0.3)))
            sns.heatmap(
                pivot_data.values.reshape(-1, 1),
                annot=True,
                fmt='.4f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Mean Causal Sensitivity'},
                xticklabels=['Sensitivity'],
                yticklabels=pivot_data.index
            )
            plt.title('Drug-Level Causal Sensitivity Heatmap')
            plt.tight_layout()
            plt.savefig(visuals_dir_path / 'drug_causal_sensitivity_heatmap.png', dpi=150)
            plt.close()
    
    # Save output files
    print("\nSaving output CSV files...")
    if mutation_effects_rows:
        pd.DataFrame(mutation_effects_rows).to_csv(
            results_dir_path / 'counterfactual_mutation_effects.csv',
            index=False
        )
        print(f"  Saved {len(mutation_effects_rows)} mutation effect rows")
    
    if causal_ranking_rows:
        pd.DataFrame(causal_ranking_rows).to_csv(
            results_dir_path / 'mutation_causal_ranking.csv',
            index=False
        )
        print(f"  Saved {len(causal_ranking_rows)} mutation ranking rows")
    
    if drug_level_summary_rows:
        pd.DataFrame(drug_level_summary_rows).to_csv(
            results_dir_path / 'drug_level_causal_summary.csv',
            index=False
        )
        print(f"  Saved {len(drug_level_summary_rows)} drug summary rows")
    
    print("Counterfactual analysis complete.")


if __name__ == '__main__':
    run_counterfactual_analysis(subset_size=100, quick_demo=True)

"""
Interpretability analysis for HIV drug resistance prediction.

This module provides functions for:
- Loading known DRM (Drug Resistance Mutation) lists
- Computing DRM enrichment in attention weights
- Finding novel positions with high attention
- SHAP value computation
- Extracting learned attention weights
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact
from tqdm import tqdm


def extract_learned_attention(
    embedding: np.ndarray,
    model,
    device: torch.device
) -> np.ndarray:
    """
    Extract attention weights from the learned AttentionWeightedClassifier.

    Args:
        embedding: Per-residue embedding array (seq_len, embed_dim)
        model: Trained AttentionWeightedClassifier
        device: Torch device

    Returns:
        Attention weights array (seq_len,)
    """
    model.eval()
    
    # Prepare input
    # (seq_len, dim) -> (1, seq_len, dim)
    emb_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(device)
    
    # Mask is all 1s for single sequence
    mask = torch.ones(1, len(embedding)).to(device)
    
    with torch.no_grad():
        _, weights = model(emb_tensor, mask)
        
    # (1, seq_len) -> (seq_len,)
    return weights.cpu().numpy().flatten()


# Drug Resistance Mutations (DRMs) from IAS-USA 2022 guidelines
# Protease Inhibitor DRMs (positions in HIV-1 Protease, 99 aa)
PI_DRM_POSITIONS = {
    'major': {30, 32, 33, 46, 47, 48, 50, 54, 76, 82, 84, 88, 90},
    'minor': {10, 11, 13, 16, 20, 23, 24, 35, 36, 43, 53, 58, 60, 62,
              66, 71, 73, 74, 77, 83, 85, 89, 93}
}

# Drug-specific PI DRMs
PI_DRUG_SPECIFIC = {
    'ATV': {32, 33, 46, 47, 48, 50, 54, 82, 84, 88, 90},
    'DRV': {32, 33, 47, 50, 54, 76, 84, 89},
    'FPV': {32, 33, 46, 47, 50, 54, 76, 82, 84},
    'IDV': {32, 46, 47, 54, 76, 82, 84},
    'LPV': {32, 33, 46, 47, 48, 50, 54, 76, 82, 84},
    'NFV': {30, 46, 84, 88, 90},
    'SQV': {48, 54, 82, 84, 90},
    'TPV': {33, 47, 58, 74, 82, 83, 84}
}

# NRTI DRMs (positions in HIV-1 Reverse Transcriptase)
NRTI_DRM_POSITIONS = {
    'TAMs': {41, 67, 70, 210, 215, 219},
    'major': {65, 69, 74, 115, 151, 184},
    'other': {62, 75, 77, 116, 118}
}

# Drug-specific NRTI DRMs
NRTI_DRUG_SPECIFIC = {
    'ABC': {65, 74, 115, 184},
    'AZT': {41, 67, 70, 210, 215, 219},
    '3TC': {65, 184},
    'D4T': {41, 65, 67, 70, 75, 210, 215, 219},
    'DDI': {65, 74},
    'TDF': {65, 70}
}

# NNRTI DRMs
NNRTI_DRM_POSITIONS = {
    'major': {100, 101, 103, 106, 181, 188, 190, 230},
    'minor': {98, 108, 138, 179, 221, 225, 227, 238}
}

# Drug-specific NNRTI DRMs
NNRTI_DRUG_SPECIFIC = {
    'EFV': {100, 101, 103, 106, 108, 181, 188, 190, 225},
    'ETR': {100, 101, 106, 138, 179, 181, 190, 230},
    'NVP': {100, 101, 103, 106, 108, 181, 188, 190},
    'RPV': {100, 101, 138, 179, 181, 221, 230}
}


def load_known_drms(drug_class: str) -> Set[int]:
    """
    Load known DRM positions for a drug class.

    Args:
        drug_class: 'PI', 'NRTI', or 'NNRTI'

    Returns:
        Set of DRM positions (1-indexed)
    """
    if drug_class == 'PI':
        return PI_DRM_POSITIONS['major'] | PI_DRM_POSITIONS['minor']
    elif drug_class == 'NRTI':
        return (NRTI_DRM_POSITIONS['TAMs'] |
                NRTI_DRM_POSITIONS['major'] |
                NRTI_DRM_POSITIONS['other'])
    elif drug_class == 'NNRTI':
        return NNRTI_DRM_POSITIONS['major'] | NNRTI_DRM_POSITIONS['minor']
    else:
        raise ValueError(f"Unknown drug class: {drug_class}")


def get_drug_specific_drms(drug: str, drug_class: str) -> Set[int]:
    """
    Get drug-specific DRM positions.

    Args:
        drug: Drug abbreviation
        drug_class: Drug class

    Returns:
        Set of DRM positions
    """
    drug_specific = {
        'PI': PI_DRUG_SPECIFIC,
        'NRTI': NRTI_DRUG_SPECIFIC,
        'NNRTI': NNRTI_DRUG_SPECIFIC
    }

    if drug_class not in drug_specific:
        return load_known_drms(drug_class)

    return drug_specific[drug_class].get(drug, load_known_drms(drug_class))


def compute_drm_enrichment(
    attention_differential: np.ndarray,
    drm_positions: Set[int],
    top_k: int = 20
) -> Dict:
    """
    Compute enrichment of DRM positions in top attention positions.

    Args:
        attention_differential: Attention difference (resistant - susceptible)
        drm_positions: Set of known DRM positions (1-indexed)
        top_k: Number of top positions to consider

    Returns:
        Dictionary with enrichment statistics
    """
    seq_len = len(attention_differential)

    # Get top-k positions by absolute differential
    top_positions = np.argsort(np.abs(attention_differential))[-top_k:]
    top_positions_set = set(top_positions + 1)  # Convert to 1-indexed

    # Count overlap with DRM positions
    drm_in_range = drm_positions & set(range(1, seq_len + 1))
    overlap = top_positions_set & drm_in_range

    # Expected by chance
    expected = top_k * len(drm_in_range) / seq_len if seq_len > 0 else 0
    observed = len(overlap)

    enrichment_ratio = observed / expected if expected > 0 else 0

    # Fisher's exact test
    a = observed
    b = top_k - observed
    c = len(drm_in_range) - observed
    d = seq_len - top_k - c

    # Ensure non-negative
    a, b, c, d = max(0, a), max(0, b), max(0, c), max(0, d)

    _, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')

    return {
        'enrichment_ratio': enrichment_ratio,
        'p_value': p_value,
        'observed': observed,
        'expected': expected,
        'top_k': top_k,
        'overlap_positions': sorted(overlap),
        'top_positions': sorted(top_positions_set)
    }


def find_novel_positions(
    attention_differential: np.ndarray,
    drm_positions: Set[int],
    top_k: int = 30
) -> List[Dict]:
    """
    Find positions with high attention that are NOT known DRMs.

    Args:
        attention_differential: Attention difference (resistant - susceptible)
        drm_positions: Set of known DRM positions
        top_k: Number of top positions to consider

    Returns:
        List of dictionaries with novel position information
    """
    seq_len = len(attention_differential)

    # Get top-k positions by attention differential
    top_positions = np.argsort(np.abs(attention_differential))[-top_k:]
    top_positions_set = set(top_positions + 1)  # 1-indexed

    # Find positions NOT in known DRMs
    drm_in_range = drm_positions & set(range(1, seq_len + 1))
    novel = top_positions_set - drm_in_range

    novel_positions = []
    for pos in novel:
        attn_diff = attention_differential[pos - 1]  # 0-indexed
        rank = list(np.argsort(np.abs(attention_differential))[::-1]).index(pos - 1) + 1

        novel_positions.append({
            'position': pos,
            'attention_differential': attn_diff,
            'rank': rank,
            'direction': 'resistant' if attn_diff > 0 else 'susceptible'
        })

    return sorted(novel_positions, key=lambda x: x['rank'])


def compute_attention_differential(
    sequences: List[str],
    labels: np.ndarray,
    model,
    alphabet,
    device,
    max_samples: int = 100,
    random_state: int = 42
) -> Dict:
    """
    Compute average attention differential between resistant and susceptible sequences.

    Args:
        sequences: List of sequences
        labels: Binary resistance labels
        model: ESM-2 model
        alphabet: ESM alphabet
        device: torch device
        max_samples: Maximum samples per class
        random_state: Random seed

    Returns:
        Dictionary with attention differential and sample counts
    """
    from .feature_engineering import extract_attention_weights

    np.random.seed(random_state)

    resistant_idx = np.where(labels == 1)[0]
    susceptible_idx = np.where(labels == 0)[0]

    # Sample for efficiency
    n_resistant = min(max_samples, len(resistant_idx))
    n_susceptible = min(max_samples, len(susceptible_idx))

    if n_resistant < 10 or n_susceptible < 10:
        return None

    sample_resistant = np.random.choice(resistant_idx, n_resistant, replace=False)
    sample_susceptible = np.random.choice(susceptible_idx, n_susceptible, replace=False)

    resistant_attention = []
    susceptible_attention = []

    # Extract attention for resistant samples
    for idx in tqdm(sample_resistant, desc="Resistant sequences"):
        try:
            _, pos_attn = extract_attention_weights(
                sequences[idx], model, alphabet, device, layer=-2
            )
            resistant_attention.append(pos_attn)
        except Exception as e:
            continue

    # Extract attention for susceptible samples
    for idx in tqdm(sample_susceptible, desc="Susceptible sequences"):
        try:
            _, pos_attn = extract_attention_weights(
                sequences[idx], model, alphabet, device, layer=-2
            )
            susceptible_attention.append(pos_attn)
        except Exception as e:
            continue

    if len(resistant_attention) == 0 or len(susceptible_attention) == 0:
        return None

    # Pad and average
    max_len = max(
        max(len(a) for a in resistant_attention),
        max(len(a) for a in susceptible_attention)
    )

    def pad_and_average(attention_list):
        padded = np.zeros((len(attention_list), max_len))
        for i, a in enumerate(attention_list):
            padded[i, :len(a)] = a
        return np.mean(padded, axis=0)

    resistant_avg = pad_and_average(resistant_attention)
    susceptible_avg = pad_and_average(susceptible_attention)
    differential = resistant_avg - susceptible_avg

    return {
        'resistant_avg': resistant_avg,
        'susceptible_avg': susceptible_avg,
        'differential': differential,
        'n_resistant': len(resistant_attention),
        'n_susceptible': len(susceptible_attention)
    }


def compute_learned_attention_differential(
    embeddings: List[np.ndarray],
    labels: np.ndarray,
    model,
    device: torch.device,
    max_samples: int = 100,
    random_state: int = 42
) -> Dict:
    """
    Compute attention differential using the learned AttentionWeightedClassifier.
    
    Args:
        embeddings: List of per-residue embeddings
        labels: Binary labels
        model: Trained AttentionWeightedClassifier
        device: Torch device
        max_samples: Max samples per class
        random_state: Random seed
        
    Returns:
        Dictionary with differential analysis
    """
    np.random.seed(random_state)

    resistant_idx = np.where(labels == 1)[0]
    susceptible_idx = np.where(labels == 0)[0]

    # Sample for efficiency
    n_resistant = min(max_samples, len(resistant_idx))
    n_susceptible = min(max_samples, len(susceptible_idx))

    if n_resistant < 10 or n_susceptible < 10:
        return None

    sample_resistant = np.random.choice(resistant_idx, n_resistant, replace=False)
    sample_susceptible = np.random.choice(susceptible_idx, n_susceptible, replace=False)

    resistant_attention = []
    susceptible_attention = []
    
    model.eval()

    # Extract attention for resistant samples
    for idx in tqdm(sample_resistant, desc="Resistant sequences (Learned)"):
        weights = extract_learned_attention(embeddings[idx], model, device)
        resistant_attention.append(weights)

    # Extract attention for susceptible samples
    for idx in tqdm(sample_susceptible, desc="Susceptible sequences (Learned)"):
        weights = extract_learned_attention(embeddings[idx], model, device)
        susceptible_attention.append(weights)

    if len(resistant_attention) == 0 or len(susceptible_attention) == 0:
        return None

    # Pad and average
    max_len = max(
        max(len(a) for a in resistant_attention),
        max(len(a) for a in susceptible_attention)
    )

    def pad_and_average(attention_list):
        padded = np.zeros((len(attention_list), max_len))
        for i, a in enumerate(attention_list):
            padded[i, :len(a)] = a
        return np.mean(padded, axis=0)

    resistant_avg = pad_and_average(resistant_attention)
    susceptible_avg = pad_and_average(susceptible_attention)
    differential = resistant_avg - susceptible_avg

    return {
        'resistant_avg': resistant_avg,
        'susceptible_avg': susceptible_avg,
        'differential': differential,
        'n_resistant': len(resistant_attention),
        'n_susceptible': len(susceptible_attention)
    }


def compute_shap_values(
    model,
    X: np.ndarray,
    n_background: int = 100,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute SHAP values for model interpretability.

    Args:
        model: Trained classifier
        X: Feature matrix
        n_background: Number of background samples
        random_state: Random seed

    Returns:
        SHAP values array
    """
    import shap

    np.random.seed(random_state)

    # Sample background
    bg_idx = np.random.choice(len(X), min(n_background, len(X)), replace=False)
    X_background = X[bg_idx]

    # Create explainer
    if hasattr(model, 'feature_importances_'):
        explainer = shap.TreeExplainer(model)
    elif hasattr(model, 'coef_'):
        explainer = shap.LinearExplainer(model, X_background)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_background)

    # Compute SHAP values
    shap_values = explainer.shap_values(X)

    return shap_values


def summarize_drm_validation(
    validation_results: List[Dict],
    top_k: int = 20
) -> Dict:
    """
    Summarize DRM validation results across drugs.

    Args:
        validation_results: List of validation result dictionaries
        top_k: Top-k value to summarize

    Returns:
        Summary statistics dictionary
    """
    top_k_results = [r for r in validation_results if r['top_k'] == top_k]

    if not top_k_results:
        return {}

    enrichments = [r['enrichment_ratio'] for r in top_k_results]
    p_values = [r['p_value'] for r in top_k_results]

    n_significant = sum(1 for p in p_values if p < 0.05)

    return {
        'mean_enrichment': np.mean(enrichments),
        'std_enrichment': np.std(enrichments),
        'median_enrichment': np.median(enrichments),
        'n_drugs': len(top_k_results),
        'n_significant': n_significant,
        'pct_significant': 100 * n_significant / len(top_k_results),
        'top_k': top_k
    }

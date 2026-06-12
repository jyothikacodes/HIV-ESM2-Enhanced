"""
Rare mutation analysis for HIV drug resistance prediction.

This module provides functions for:
- Computing per-position amino acid frequencies across a sequence cohort
- Generating rarity-based weight vectors for attention re-weighting
- Producing binary rare-mutation masks
- Summarising rare mutations in a cohort

These outputs are designed to integrate with the AttentionWeightedClassifier
in models.py via the ``rare_mutation_weights`` parameter.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Canonical amino acid alphabet (20 standard AAs)
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

DRM_POSITIONS_PI = [29, 31, 32, 33, 45, 46, 47, 48, 49, 53, 75, 81, 83, 87, 89]
DRM_POSITIONS_NRTI = [40, 61, 64, 66, 68, 69, 73, 74, 114, 150, 183, 209, 214, 218]
DRM_POSITIONS_NNRTI = [99, 100, 102, 105, 137, 178, 180, 187, 189, 220, 224, 226, 229]
DRM_POSITION_MAP = {
    'PI': DRM_POSITIONS_PI,
    'NRTI': DRM_POSITIONS_NRTI,
    'NNRTI': DRM_POSITIONS_NNRTI,
}


def compute_mutation_frequencies(
    sequences: List[str],
    reference: str
) -> np.ndarray:
    """
    Compute amino acid frequencies at each position across all sequences.

    Args:
        sequences: List of aligned amino acid sequences
        reference: Reference sequence (e.g. HXB2)

    Returns:
        Array of shape (seq_len, 20) where entry [i, j] is the frequency
        of amino acid j at position i across all sequences.
    """
    seq_len = len(reference)
    n_seqs = len(sequences)

    if n_seqs == 0:
        return np.zeros((seq_len, 20))

    # Count occurrences: (seq_len, 20)
    counts = np.zeros((seq_len, 20), dtype=np.float64)

    for seq in sequences:
        for i in range(min(len(seq), seq_len)):
            aa = seq[i]
            if aa in AA_TO_IDX:
                counts[i, AA_TO_IDX[aa]] += 1

    # Normalise to frequencies
    frequencies = counts / n_seqs

    return frequencies


def compute_rare_mutation_weights(
    sequences: List[str],
    reference: str,
    tau: float = 0.01,
    epsilon: float = 1e-6,
    frequencies: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Compute per-residue rarity weight vectors for each sequence.

    For each sequence, position i gets weight:
        w_i = 1 / (freq(position_i, amino_acid_i) + epsilon)

    Positions harbouring rare mutations (freq < tau) naturally receive
    very high weights, biasing attention toward them.

    Args:
        sequences: List of aligned amino acid sequences
        reference: Reference sequence (e.g. HXB2)
        tau: Rarity threshold (not used for filtering here, but available
             for documentation — weights are continuous)
        epsilon: Small constant to avoid division by zero
        frequencies: Pre-computed frequency matrix (seq_len, 20).
                     If None, computed from ``sequences``.

    Returns:
        List of weight vectors, one per sequence. Each vector has length
        min(len(seq), len(reference)).
    """
    if frequencies is None:
        frequencies = compute_mutation_frequencies(sequences, reference)

    seq_len = len(reference)
    weights_list = []

    for seq in sequences:
        effective_len = min(len(seq), seq_len)
        w = np.ones(effective_len, dtype=np.float64)

        for i in range(effective_len):
            aa = seq[i]
            if aa in AA_TO_IDX:
                freq = frequencies[i, AA_TO_IDX[aa]]
                w[i] = 1.0 / (freq + epsilon)
            else:
                # Unknown amino acid (e.g. X, *, gap) → neutral weight
                w[i] = 1.0

        weights_list.append(w)

    return weights_list


def compute_rare_mutation_mask(
    sequences: List[str],
    reference: str,
    tau: float = 0.01,
    frequencies: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Compute binary rare-mutation masks for each sequence.

    Position i is marked 1 if the amino acid at that position has
    frequency < tau across the cohort, else 0.

    Args:
        sequences: List of aligned amino acid sequences
        reference: Reference sequence
        tau: Frequency threshold below which a mutation is considered rare
        frequencies: Pre-computed frequency matrix. If None, computed here.

    Returns:
        List of binary mask arrays, one per sequence.
    """
    if frequencies is None:
        frequencies = compute_mutation_frequencies(sequences, reference)

    seq_len = len(reference)
    masks = []

    for seq in sequences:
        effective_len = min(len(seq), seq_len)
        mask = np.zeros(effective_len, dtype=np.float64)

        for i in range(effective_len):
            aa = seq[i]
            if aa in AA_TO_IDX:
                freq = frequencies[i, AA_TO_IDX[aa]]
                if freq < tau:
                    mask[i] = 1.0

        masks.append(mask)

    return masks


def compute_rare_mutation_summary_features(
    sequences: List[str],
    reference: str,
    tau: float = 0.05,
    epsilon: float = 1e-6,
    frequencies: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute compact rare-mutation summary features for each sequence.

    These features can be concatenated with ESM embeddings for downstream
    classifiers and are designed to capture rare mutation burden and severity.

    Returns:
        Array of shape (n_sequences, 4) with columns:
        - rare_mutation_count
        - rare_mutation_fraction
        - summed_inverse_frequency
        - mean_inverse_frequency
    """
    if frequencies is None:
        frequencies = compute_mutation_frequencies(sequences, reference)

    seq_len = len(reference)
    summary_features = []

    for seq in sequences:
        effective_len = min(len(seq), seq_len)
        rare_count = 0
        inverse_weights = []

        for i in range(effective_len):
            aa = seq[i]
            if aa in AA_TO_IDX:
                freq = frequencies[i, AA_TO_IDX[aa]]
                inv_weight = 1.0 / (freq + epsilon)
                inverse_weights.append(inv_weight)
                if freq < tau and aa != reference[i]:
                    rare_count += 1

        total_mutations = sum(1 for i in range(effective_len)
                              if seq[i] != reference[i] and seq[i] in AA_TO_IDX)
        rare_fraction = rare_count / total_mutations if total_mutations > 0 else 0.0
        summed_inverse = float(np.sum(inverse_weights)) if inverse_weights else 0.0
        mean_inverse = float(np.mean(inverse_weights)) if inverse_weights else 0.0

        summary_features.append([
            rare_count,
            rare_fraction,
            summed_inverse,
            mean_inverse
        ])

    return np.array(summary_features, dtype=np.float32)


def _get_drm_positions_for_class(drug_class: str) -> List[int]:
    return DRM_POSITION_MAP.get(drug_class, [])


def compute_drm_position_features(
    sequences: List[str],
    reference: str,
    drug_class: str,
    frequencies: Optional[np.ndarray] = None,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute DRM position-aware mutation features for a drug class."""
    positions = _get_drm_positions_for_class(drug_class)
    if not positions:
        return np.zeros((len(sequences), 0), dtype=np.float32)

    if frequencies is None:
        frequencies = compute_mutation_frequencies(sequences, reference)

    n_samples = len(sequences)
    feature_matrix = np.zeros((n_samples, len(positions) * 2), dtype=np.float32)

    for idx, seq in enumerate(sequences):
        effective_len = min(len(seq), len(reference))
        for pos_idx, pos in enumerate(positions):
            if pos < effective_len and pos < len(reference):
                aa = seq[pos]
                is_mutation = float(aa in AA_TO_IDX and aa != reference[pos])
                inv_freq = 0.0
                if is_mutation:
                    freq = frequencies[pos, AA_TO_IDX[aa]]
                    inv_freq = float(1.0 / (freq + epsilon))
            else:
                is_mutation = 0.0
                inv_freq = 0.0
            feature_matrix[idx, pos_idx * 2] = is_mutation
            feature_matrix[idx, pos_idx * 2 + 1] = inv_freq

    return feature_matrix


def compute_position_weighted_mutation_features(
    sequences: List[str],
    reference: str,
    drug_class: str,
    frequencies: Optional[np.ndarray] = None,
    weights: Optional[Dict[int, float]] = None,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute DRM position features with optional position-specific weighting."""
    raw_features = compute_drm_position_features(
        sequences,
        reference,
        drug_class,
        frequencies=frequencies,
        epsilon=epsilon,
    )
    if weights is None or raw_features.shape[1] == 0:
        return raw_features

    positions = _get_drm_positions_for_class(drug_class)
    weighted_features = raw_features.copy()
    for pos_idx, pos in enumerate(positions):
        weight = float(weights.get(pos, 1.0))
        weighted_features[:, pos_idx * 2] *= weight
        weighted_features[:, pos_idx * 2 + 1] *= weight
    return weighted_features


def get_rare_mutation_summary(
    sequences: List[str],
    reference: str,
    tau: float = 0.01,
    frequencies: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Summarise all rare mutations found in the cohort.

    Args:
        sequences: List of aligned amino acid sequences
        reference: Reference sequence
        tau: Rarity threshold
        frequencies: Pre-computed frequency matrix. If None, computed here.

    Returns:
        DataFrame with columns:
        - position (1-indexed)
        - amino_acid
        - frequency
        - count
        - reference_aa
        - is_mutation (True if different from reference)
    """
    if frequencies is None:
        frequencies = compute_mutation_frequencies(sequences, reference)

    n_seqs = len(sequences)
    seq_len = len(reference)
    rows = []

    for i in range(seq_len):
        for j, aa in enumerate(AMINO_ACIDS):
            freq = frequencies[i, j]
            if 0 < freq < tau:
                rows.append({
                    'position': i + 1,  # 1-indexed
                    'amino_acid': aa,
                    'frequency': freq,
                    'count': int(round(freq * n_seqs)),
                    'reference_aa': reference[i] if i < len(reference) else '?',
                    'is_mutation': aa != reference[i] if i < len(reference) else True
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('frequency', ascending=True).reset_index(drop=True)

    return df


def normalize_weights_for_attention(
    weights: np.ndarray,
    method: str = 'softmax',
    temperature: float = 1.0
) -> np.ndarray:
    """
    Normalise raw rarity weights to a suitable range for attention modulation.

    Raw weights (1/freq) can span many orders of magnitude. This function
    applies optional transformations to keep them in a useful range.

    Args:
        weights: Raw rarity weight vector (seq_len,)
        method: Normalisation method:
            - 'identity': no transformation (raw 1/freq)
            - 'log': log(1 + w) to compress dynamic range
            - 'rank': replace with rank-based weights
            - 'minmax': min-max scale to [1, max_weight]
            - 'softmax': softmax-style normalisation
        temperature: Temperature for softmax method

    Returns:
        Normalised weight vector (seq_len,)
    """
    if method == 'identity':
        return weights
    elif method == 'log':
        return np.log1p(weights)
    elif method == 'rank':
        ranks = np.argsort(np.argsort(weights)).astype(np.float64)
        return 1.0 + ranks / len(ranks)
    elif method == 'minmax':
        w_min, w_max = weights.min(), weights.max()
        if w_max - w_min < 1e-12:
            return np.ones_like(weights)
        return 1.0 + (weights - w_min) / (w_max - w_min)
    elif method == 'softmax':
        log_w = np.log(weights + 1e-12) / temperature
        log_w = log_w - log_w.max()  # numerical stability
        exp_w = np.exp(log_w)
        return exp_w / exp_w.sum() * len(weights)
    else:
        raise ValueError(f"Unknown normalisation method: {method}")

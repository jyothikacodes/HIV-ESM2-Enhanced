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

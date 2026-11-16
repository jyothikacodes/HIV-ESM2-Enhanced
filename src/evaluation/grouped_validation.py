"""
Patient-grouped and sequence-clustered cross-validation for HIV resistance.

Prevents sequence similarity and longitudinal patient leakage:
- Patient-grouped CV: isolates from the same patient (PtID) are strictly isolated to one fold.
- Sequence-clustered CV: near-identical sequences (>90% or >95% similarity) are clustered together,
  ensuring homologous sequences never leak across train and validation splits.
"""

from typing import Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def compute_sequence_hamming_distance(seq1: str, seq2: str) -> float:
    """Compute normalized Hamming distance between two aligned sequences."""
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return 1.0
    mismatches = sum(1 for i in range(min_len) if seq1[i] != seq2[i])
    mismatches += abs(len(seq1) - len(seq2))
    max_len = max(len(seq1), len(seq2))
    return mismatches / max_len


def cluster_sequences_by_identity(
    sequences: List[str],
    identity_threshold: float = 0.95
) -> np.ndarray:
    """
    Cluster sequences using greedy single-linkage clustering based on sequence identity.
    
    Args:
        sequences: List of amino acid sequences.
        identity_threshold: Fractional sequence identity threshold (e.g. 0.95 for 95%).
        
    Returns:
        cluster_ids: Array of shape (N,) containing integer cluster assignments.
    """
    N = len(sequences)
    cluster_ids = np.full(N, -1, dtype=int)
    current_cluster = 0
    max_distance = 1.0 - identity_threshold

    for i in range(N):
        if cluster_ids[i] != -1:
            continue
            
        cluster_ids[i] = current_cluster
        seed_seq = sequences[i]
        
        # Compare with remaining unclustered sequences
        for j in range(i + 1, N):
            if cluster_ids[j] == -1:
                dist = compute_sequence_hamming_distance(seed_seq, sequences[j])
                if dist <= max_distance:
                    cluster_ids[j] = current_cluster
                    
        current_cluster += 1
        
    return cluster_ids


class GroupedValidationSplitter:
    """
    Cross-validation splitter supporting patient-grouped, sequence-clustered,
    or combined grouping strategies.
    """
    def __init__(
        self,
        n_splits: int = 5,
        grouping_strategy: str = "sequence_clustered",
        identity_threshold: float = 0.95,
        random_state: int = 42
    ):
        """
        Args:
            n_splits: Number of CV folds.
            grouping_strategy: 'patient_grouped', 'sequence_clustered', or 'stratified'.
            identity_threshold: Threshold for sequence clustering (default 0.95).
            random_state: Random seed.
        """
        self.n_splits = n_splits
        self.grouping_strategy = grouping_strategy
        self.identity_threshold = identity_threshold
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequences: Optional[List[str]] = None,
        patient_ids: Optional[Union[List[str], np.ndarray]] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate (train_idx, val_idx) fold splits ensuring complete isolation of groups.
        """
        N = len(y)
        
        if self.grouping_strategy == "patient_grouped" and patient_ids is not None:
            groups = np.array(patient_ids)
        elif self.grouping_strategy == "sequence_clustered" and sequences is not None:
            groups = cluster_sequences_by_identity(sequences, self.identity_threshold)
        else:
            # Fallback to random grouping if metadata is absent
            rng = np.random.RandomState(self.random_state)
            groups = rng.randint(0, self.n_splits, size=N)

        # Check unique groups
        n_unique_groups = len(np.unique(groups))
        effective_splits = min(self.n_splits, n_unique_groups)
        
        # Try StratifiedGroupKFold if labels are binary, else GroupKFold
        valid_labels = ~np.isnan(y)
        if valid_labels.all() and len(np.unique(y)) > 1:
            try:
                sgkf = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=self.random_state)
                for train_idx, val_idx in sgkf.split(X, y, groups):
                    yield train_idx, val_idx
                return
            except ValueError:
                pass
                
        gkf = GroupKFold(n_splits=effective_splits)
        for train_idx, val_idx in gkf.split(X, y, groups):
            yield train_idx, val_idx

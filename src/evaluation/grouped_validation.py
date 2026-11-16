"""
Patient-grouped and sequence-clustered cross-validation for HIV resistance.

Prevents sequence similarity and longitudinal patient leakage:
- Patient-grouped CV: isolates from the same patient (PtID) are strictly isolated to one fold.
- Sequence-clustered CV: near-identical sequences (>90% or >95% similarity) are clustered together
  using transitive connected-component single-linkage clustering.
- Joint patient & sequence clustering: merges sequence clusters and patient IDs so that no sequence
  and no patient crosses train/test boundaries.
"""

from typing import Dict, Generator, List, Optional, Set, Tuple, Union
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
    Cluster sequences using transitive single-linkage (connected components)
    based on sequence identity.
    
    Args:
        sequences: List of amino acid sequences.
        identity_threshold: Fractional sequence identity threshold (e.g. 0.95 for 95%).
        
    Returns:
        cluster_ids: Array of shape (N,) containing integer cluster assignments.
    """
    N = len(sequences)
    max_distance = 1.0 - identity_threshold

    # Build adjacency graph
    adj: Dict[int, List[int]] = {i: [] for i in range(N)}
    for i in range(N):
        for j in range(i + 1, N):
            dist = compute_sequence_hamming_distance(sequences[i], sequences[j])
            if dist <= max_distance:
                adj[i].append(j)
                adj[j].append(i)

    # Find connected components via BFS
    visited = set()
    cluster_ids = np.full(N, -1, dtype=int)
    current_cluster = 0

    for i in range(N):
        if i not in visited:
            queue = [i]
            visited.add(i)
            cluster_ids[i] = current_cluster
            
            while queue:
                curr = queue.pop(0)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        cluster_ids[neighbor] = current_cluster
                        queue.append(neighbor)
                        
            current_cluster += 1

    return cluster_ids


def create_joint_patient_sequence_groups(
    sequences: Optional[List[str]] = None,
    patient_ids: Optional[Union[List[str], np.ndarray]] = None,
    identity_threshold: float = 0.95,
    n_samples: int = 0
) -> np.ndarray:
    """
    Construct joint groups merging patient identifiers and sequence clusters.
    Guarantees:
    - No two sequences with identity >= identity_threshold end up in different groups.
    - No two isolates from the same patient end up in different groups.
    """
    N = n_samples if n_samples > 0 else (len(sequences) if sequences is not None else len(patient_ids))
    if sequences is None and patient_ids is None:
        return np.arange(N)

    # Graph connecting sample indices
    adj: Dict[int, List[int]] = {i: [] for i in range(N)}

    # Connect samples from same patient
    if patient_ids is not None:
        pt_map: Dict[str, List[int]] = {}
        for idx, pt in enumerate(patient_ids):
            pt_map.setdefault(str(pt), []).append(idx)
        for pt, indices in pt_map.items():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    adj[indices[i]].append(indices[j])
                    adj[indices[j]].append(indices[i])

    # Connect samples within sequence identity threshold
    if sequences is not None:
        max_dist = 1.0 - identity_threshold
        for i in range(N):
            for j in range(i + 1, N):
                if compute_sequence_hamming_distance(sequences[i], sequences[j]) <= max_dist:
                    adj[i].append(j)
                    adj[j].append(i)

    # Transitive connected components
    visited = set()
    group_ids = np.full(N, -1, dtype=int)
    current_group = 0

    for i in range(N):
        if i not in visited:
            queue = [i]
            visited.add(i)
            group_ids[i] = current_group
            while queue:
                curr = queue.pop(0)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        group_ids[nbr] = current_group
                        queue.append(nbr)
            current_group += 1

    return group_ids


class GroupedValidationSplitter:
    """
    Cross-validation splitter supporting patient-grouped, sequence-clustered,
    and joint patient-sequence grouping strategies.
    """
    def __init__(
        self,
        n_splits: int = 5,
        grouping_strategy: str = "sequence_clustered",
        identity_threshold: float = 0.95,
        random_state: int = 42
    ):
        self.n_splits = n_splits
        self.grouping_strategy = grouping_strategy
        self.identity_threshold = identity_threshold
        self.random_state = random_state

    def get_groups(
        self,
        N: int,
        sequences: Optional[List[str]] = None,
        patient_ids: Optional[Union[List[str], np.ndarray]] = None
    ) -> np.ndarray:
        if self.grouping_strategy == "patient_grouped" and patient_ids is not None:
            return create_joint_patient_sequence_groups(sequences=None, patient_ids=patient_ids, n_samples=N)
        elif self.grouping_strategy == "sequence_clustered" and sequences is not None:
            return cluster_sequences_by_identity(sequences, self.identity_threshold)
        elif self.grouping_strategy == "joint_patient_sequence":
            return create_joint_patient_sequence_groups(
                sequences=sequences, patient_ids=patient_ids, identity_threshold=self.identity_threshold, n_samples=N
            )
        else:
            rng = np.random.RandomState(self.random_state)
            return rng.randint(0, self.n_splits, size=N)

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequences: Optional[List[str]] = None,
        patient_ids: Optional[Union[List[str], np.ndarray]] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        N = len(y)
        groups = self.get_groups(N, sequences=sequences, patient_ids=patient_ids)

        n_unique_groups = len(np.unique(groups))
        effective_splits = min(self.n_splits, n_unique_groups)
        
        valid_labels = ~np.isnan(y)
        if valid_labels.all() and len(np.unique(y)) > 1:
            try:
                sgkf = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=self.random_state)
                for train_idx, val_idx in sgkf.split(X, y, groups):
                    yield train_idx, val_idx
                return
            except ValueError:
                pass
                
        # GroupKFold with group-level shuffling
        rng = np.random.RandomState(self.random_state)
        unique_groups = np.unique(groups)
        rng.shuffle(unique_groups)
        
        # Partition groups into folds
        fold_groups = np.array_split(unique_groups, effective_splits)
        for fold in range(effective_splits):
            test_group_set = set(fold_groups[fold])
            val_idx = np.where(np.isin(groups, list(test_group_set)))[0]
            train_idx = np.where(~np.isin(groups, list(test_group_set)))[0]
            yield train_idx, val_idx

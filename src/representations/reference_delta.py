"""
Reference-contrastive (Delta-Embedding) representations for HIV-1 resistance prediction.

Experiment A:
Constructs contrastive embeddings relative to the consensus wildtype baseline:
    ΔE_i = E_i(mutant) - E_i(reference)

Hypothesis:
Raw ESM-2 representations are dominated by the invariant backbone fold (>90% identical).
Explicit reference subtraction isolates the directional biophysical perturbation
induced by resistance mutations, canceling out background invariant features.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np


RepresentationMode = Literal[
    "mutant_only",
    "delta_only",
    "mutant_plus_delta",
    "mutant_delta_mutation",
]


def compute_per_residue_delta(
    mutant_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    mutant_seq: Optional[str] = None,
    reference_seq: Optional[str] = None,
) -> np.ndarray:
    """
    Compute residue-level difference embedding between mutant and reference:
        ΔE_i = E_i(mutant) - E_i(reference)
        
    Args:
        mutant_embedding: (L_mut, D) array of per-residue embeddings.
        reference_embedding: (L_ref, D) array of per-residue reference embeddings.
        mutant_seq: Optional amino acid sequence of mutant isolate.
        reference_seq: Optional amino acid sequence of reference consensus.
        
    Returns:
        delta_embedding: (min(L_mut, L_ref), D) array of residue-wise perturbations.
    """
    if mutant_embedding.ndim != 2 or reference_embedding.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2D (seq_len, dim), got mutant {mutant_embedding.shape} "
            f"and reference {reference_embedding.shape}"
        )
    if mutant_embedding.shape[1] != reference_embedding.shape[1]:
        raise ValueError(
            f"Dimension mismatch between mutant ({mutant_embedding.shape[1]}) "
            f"and reference ({reference_embedding.shape[1]})."
        )
        
    min_len = min(len(mutant_embedding), len(reference_embedding))
    delta = mutant_embedding[:min_len] - reference_embedding[:min_len]
    
    # Optional sequence consistency verification:
    # If both sequences are provided and positions match wildtype, delta should theoretically be small
    if mutant_seq is not None and reference_seq is not None:
        seq_limit = min(min_len, len(mutant_seq), len(reference_seq))
        # Zero out tiny numerical noise at strictly identical positions if desired,
        # but in contextual PLMs, neighboring mutations can slightly perturb identical positions (allostery).
        pass
        
    return delta


def pool_embeddings(
    per_residue: np.ndarray,
    pooling_method: Literal["mean", "max", "norm"] = "mean",
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Pool a 2D per-residue embedding (L, D) into a 1D vector (D).
    
    Args:
        per_residue: (L, D) array.
        pooling_method: 'mean', 'max', or 'norm'.
        mask: Optional binary mask (L,) where 1=valid residue, 0=padding/gap.
    """
    if len(per_residue) == 0:
        raise ValueError("Cannot pool empty per-residue embedding.")
        
    if mask is not None:
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) == 0:
            return np.zeros(per_residue.shape[1], dtype=per_residue.dtype)
        per_residue = per_residue[valid_indices]
        
    if pooling_method == "mean":
        return np.mean(per_residue, axis=0)
    elif pooling_method == "max":
        return np.max(per_residue, axis=0)
    elif pooling_method == "norm":
        # L2-norm weighted pooling: positions with larger perturbation receive more weight
        norms = np.linalg.norm(per_residue, axis=1, keepdims=True) + 1e-8
        weights = norms / np.sum(norms)
        return np.sum(per_residue * weights, axis=0)
    else:
        raise ValueError(f"Unknown pooling method: {pooling_method}")


class ReferenceDeltaRepresentationBuilder:
    """
    Builder for multi-modal protein representations incorporating
    mutant embeddings, reference-contrastive delta embeddings, and sparse mutation vectors.
    """
    
    def __init__(
        self,
        reference_embedding: np.ndarray,
        reference_seq: str,
        mode: RepresentationMode = "mutant_plus_delta",
        pooling: Literal["mean", "max", "norm"] = "mean"
    ):
        """
        Args:
            reference_embedding: Per-residue embedding of wildtype reference sequence (L, D).
            reference_seq: Wildtype reference amino acid sequence.
            mode: One of ['mutant_only', 'delta_only', 'mutant_plus_delta', 'mutant_delta_mutation'].
            pooling: Pooling operation ('mean', 'max', 'norm').
        """
        if reference_embedding.ndim != 2:
            raise ValueError(f"reference_embedding must be 2D, got shape {reference_embedding.shape}")
        if len(reference_embedding) != len(reference_seq):
            raise ValueError(
                f"Length mismatch: reference_embedding length {len(reference_embedding)} "
                f"!= reference_seq length {len(reference_seq)}"
            )
            
        self.reference_embedding = reference_embedding
        self.reference_seq = reference_seq
        self.embed_dim = reference_embedding.shape[1]
        self.mode = mode
        self.pooling = pooling

    def build_sample_feature(
        self,
        mutant_embedding: np.ndarray,
        mutant_seq: Optional[str] = None,
        mutation_vector: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Assemble the final feature vector for a single sample according to self.mode.
        """
        mut_pooled = pool_embeddings(mutant_embedding, pooling_method=self.pooling)
        
        if self.mode == "mutant_only":
            return mut_pooled
            
        delta = compute_per_residue_delta(
            mutant_embedding,
            self.reference_embedding,
            mutant_seq=mutant_seq,
            reference_seq=self.reference_seq
        )
        delta_pooled = pool_embeddings(delta, pooling_method=self.pooling)
        
        if self.mode == "delta_only":
            return delta_pooled
        elif self.mode == "mutant_plus_delta":
            return np.concatenate([mut_pooled, delta_pooled], axis=0)
        elif self.mode == "mutant_delta_mutation":
            if mutation_vector is None:
                raise ValueError("mode='mutant_delta_mutation' requires mutation_vector to be provided.")
            return np.concatenate([mut_pooled, delta_pooled, mutation_vector], axis=0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def build_cohort_matrix(
        self,
        mutant_embeddings: List[np.ndarray],
        mutant_sequences: Optional[List[str]] = None,
        mutation_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Assemble feature matrix for a full cohort of sequences.
        
        Returns:
            X: Array of shape (N, feature_dim)
        """
        n_samples = len(mutant_embeddings)
        features_list = []
        
        for i in range(n_samples):
            seq = mutant_sequences[i] if mutant_sequences is not None else None
            mut_vec = mutation_matrix[i] if mutation_matrix is not None else None
            feat = self.build_sample_feature(
                mutant_embeddings[i],
                mutant_seq=seq,
                mutation_vector=mut_vec
            )
            features_list.append(feat)
            
        return np.vstack(features_list)

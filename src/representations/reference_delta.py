"""
Reference-contrastive (Delta-Embedding) representations for HIV-1 resistance prediction.

Experiment A & Ablations:
Constructs contrastive embeddings relative to the consensus wildtype baseline:
    ΔE_i = E_i(mutant) - E_i(reference)

Hypothesis:
Raw ESM-2 representations are dominated by the invariant backbone fold (>90% identical).
Explicit reference subtraction isolates the directional biophysical perturbation
induced by resistance mutations, canceling out background invariant features.

Supports controlled ablation modes:
1. mutant_only
2. reference_only
3. delta_only
4. mutant_plus_delta
5. mutant_plus_reference_plus_delta
6. mutant_delta_mutation

Includes alignment-aware coordinate verification and Protease (99 aa) vs RT (240 aa) checks.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np


RepresentationMode = Literal[
    "mutant_only",
    "reference_only",
    "delta_only",
    "mutant_plus_delta",
    "mutant_plus_reference_plus_delta",
    "mutant_delta_mutation",
]

ProteinType = Literal["PR", "RT", "auto"]

CANONICAL_LENGTHS = {
    "PR": 99,
    "RT": 240,
}


def compute_per_residue_delta(
    mutant_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    mutant_seq: Optional[str] = None,
    reference_seq: Optional[str] = None,
    protein_type: ProteinType = "auto",
    padding_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute residue-level difference embedding between mutant and reference:
        ΔE_i = E_i(mutant) - E_i(reference)
        
    Args:
        mutant_embedding: (L_mut, D) array of per-residue embeddings.
        reference_embedding: (L_ref, D) array of per-residue reference embeddings.
        mutant_seq: Optional amino acid sequence of mutant isolate.
        reference_seq: Optional amino acid sequence of reference consensus.
        protein_type: 'PR' (99 aa), 'RT' (240 aa), or 'auto'.
        padding_mask: Optional boolean/binary mask (L,) where 1=valid residue, 0=padding/gap.
        
    Returns:
        delta_embedding: (L, D) array of residue-wise perturbations.
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
        
    # Check protein coordinates
    L_ref = len(reference_embedding)
    if protein_type == "PR" and L_ref != CANONICAL_LENGTHS["PR"]:
        raise ValueError(f"Expected Protease reference length {CANONICAL_LENGTHS['PR']}, got {L_ref}")
    elif protein_type == "RT" and L_ref != CANONICAL_LENGTHS["RT"]:
        raise ValueError(f"Expected RT reference length {CANONICAL_LENGTHS['RT']}, got {L_ref}")

    # Check sequence alignment and indels
    L_mut = len(mutant_embedding)
    if mutant_seq is not None and reference_seq is not None:
        if len(mutant_seq) != L_mut:
            raise ValueError(f"mutant_seq length {len(mutant_seq)} != embedding length {L_mut}")
        if len(reference_seq) != L_ref:
            raise ValueError(f"reference_seq length {len(reference_seq)} != embedding length {L_ref}")
            
        # If lengths differ without gap characters, warn or raise
        if len(mutant_seq) != len(reference_seq):
            if "-" not in mutant_seq and "~" not in mutant_seq and "#" not in mutant_seq:
                raise ValueError(
                    f"Unaligned length mismatch: mutant ({len(mutant_seq)} aa) vs "
                    f"reference ({len(reference_seq)} aa). Naive subtraction across shifted indices "
                    f"causes catastrophic phase shift. Sequence alignment required."
                )

    min_len = min(L_mut, L_ref)
    delta = mutant_embedding[:min_len] - reference_embedding[:min_len]
    
    # Exclude padding positions if provided
    if padding_mask is not None:
        mask_slice = padding_mask[:min_len]
        delta[mask_slice == 0] = 0.0

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
    Builder for multi-modal protein representations supporting controlled ablations:
    - mutant_only: E_mut
    - reference_only: E_ref
    - delta_only: ΔE
    - mutant_plus_delta: E_mut ⊕ ΔE
    - mutant_plus_reference_plus_delta: E_mut ⊕ E_ref ⊕ ΔE
    - mutant_delta_mutation: E_mut ⊕ ΔE ⊕ M
    """
    
    def __init__(
        self,
        reference_embedding: np.ndarray,
        reference_seq: str,
        mode: RepresentationMode = "mutant_plus_delta",
        pooling: Literal["mean", "max", "norm"] = "mean",
        protein_type: ProteinType = "auto"
    ):
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
        self.protein_type = protein_type
        self.ref_pooled = pool_embeddings(reference_embedding, pooling_method=pooling)

    def build_sample_feature(
        self,
        mutant_embedding: np.ndarray,
        mutant_seq: Optional[str] = None,
        mutation_vector: Optional[np.ndarray] = None,
        padding_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Assemble the final feature vector for a single sample according to self.mode.
        """
        # Mode: reference_only (unconditioned baseline)
        if self.mode == "reference_only":
            return self.ref_pooled.copy()

        mut_pooled = pool_embeddings(mutant_embedding, pooling_method=self.pooling, mask=padding_mask)
        
        # Mode: mutant_only (Farquhar standard)
        if self.mode == "mutant_only":
            return mut_pooled
            
        delta = compute_per_residue_delta(
            mutant_embedding,
            self.reference_embedding,
            mutant_seq=mutant_seq,
            reference_seq=self.reference_seq,
            protein_type=self.protein_type,
            padding_mask=padding_mask
        )
        delta_pooled = pool_embeddings(delta, pooling_method=self.pooling, mask=padding_mask)
        
        # Mode: delta_only
        if self.mode == "delta_only":
            return delta_pooled
        # Mode: mutant_plus_delta
        elif self.mode == "mutant_plus_delta":
            return np.concatenate([mut_pooled, delta_pooled], axis=0)
        # Mode: mutant_plus_reference_plus_delta
        elif self.mode == "mutant_plus_reference_plus_delta":
            return np.concatenate([mut_pooled, self.ref_pooled, delta_pooled], axis=0)
        # Mode: mutant_delta_mutation
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
        mutation_matrix: Optional[np.ndarray] = None,
        padding_masks: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Assemble feature matrix for a full cohort of sequences.
        """
        n_samples = len(mutant_embeddings)
        features_list = []
        
        for i in range(n_samples):
            seq = mutant_sequences[i] if mutant_sequences is not None else None
            mut_vec = mutation_matrix[i] if mutation_matrix is not None else None
            pad_mask = padding_masks[i] if padding_masks is not None else None
            feat = self.build_sample_feature(
                mutant_embeddings[i],
                mutant_seq=seq,
                mutation_vector=mut_vec,
                padding_mask=pad_mask
            )
            features_list.append(feat)
            
        return np.vstack(features_list)

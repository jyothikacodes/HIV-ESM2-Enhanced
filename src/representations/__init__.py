"""Protein representation engineering for HIV-1 drug resistance."""

from .mutation_features import (
    CLINICAL_CUTOFFS,
    DRUG_CLASSES,
    HIV_PROTEASE_REFERENCE,
    HIV_RT_REFERENCE,
    STANDARD_AMINO_ACIDS,
    FoldSpecificMutationEncoder,
    binarize_resistance,
    categorize_resistance_3class,
    get_clinical_cutoff,
)
from .reference_delta import (
    ReferenceDeltaRepresentationBuilder,
    compute_per_residue_delta,
    pool_embeddings,
)
from .mutation_interactions import (
    EpistaticInteractionClassifier,
    PairwiseMutationInteractionEncoder,
)

__all__ = [
    "CLINICAL_CUTOFFS",
    "DRUG_CLASSES",
    "HIV_PROTEASE_REFERENCE",
    "HIV_RT_REFERENCE",
    "STANDARD_AMINO_ACIDS",
    "FoldSpecificMutationEncoder",
    "binarize_resistance",
    "categorize_resistance_3class",
    "get_clinical_cutoff",
    "ReferenceDeltaRepresentationBuilder",
    "compute_per_residue_delta",
    "pool_embeddings",
    "PairwiseMutationInteractionEncoder",
    "EpistaticInteractionClassifier",
]

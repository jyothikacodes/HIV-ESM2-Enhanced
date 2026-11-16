"""
Models for HIV-1 drug resistance prediction.

Maintains complete backward compatibility with legacy classifiers while
providing multi-task architectures, continuous regressors, and biologically guided attention.
"""

# Re-export legacy model functions and classes for backward compatibility
from .legacy_models import (
    AttentionWeightedClassifier,
    EmbeddingDataset,
    MultiHeadAttentionPoolingClassifier,
    collate_embeddings,
    compare_models,
    get_default_xgb_params,
    per_drug_training,
    train_attention_model,
    train_logistic_regression,
    train_multihead_attention_model,
    train_random_forest,
    train_svm,
    train_xgboost,
)

# New Research V2 modular components
from .multitask_drug_model import (
    MaskedMultiTaskBCELoss,
    MultiTaskDrugPredictor,
    MultiTaskDrugTrunkNet,
)
from .continuous_resistance import (
    ContinuousResistanceRegressor,
    compute_continuous_metrics,
    evaluate_continuous_as_ordinal,
)
from .biological_attention import (
    CANONICAL_DRM_POSITIONS,
    DRUG_SPECIFIC_PRIMARY_DRMS,
    BiologicallyGuidedAttention,
    BiologicallyGuidedClassifier,
    get_prior_drm_indicator,
)

__all__ = [
    # Legacy symbols
    "AttentionWeightedClassifier",
    "EmbeddingDataset",
    "MultiHeadAttentionPoolingClassifier",
    "collate_embeddings",
    "compare_models",
    "get_default_xgb_params",
    "per_drug_training",
    "train_attention_model",
    "train_logistic_regression",
    "train_multihead_attention_model",
    "train_random_forest",
    "train_svm",
    "train_xgboost",
    # V2 symbols
    "MaskedMultiTaskBCELoss",
    "MultiTaskDrugPredictor",
    "MultiTaskDrugTrunkNet",
    "ContinuousResistanceRegressor",
    "compute_continuous_metrics",
    "evaluate_continuous_as_ordinal",
    "CANONICAL_DRM_POSITIONS",
    "DRUG_SPECIFIC_PRIMARY_DRMS",
    "BiologicallyGuidedAttention",
    "BiologicallyGuidedClassifier",
    "get_prior_drm_indicator",
]

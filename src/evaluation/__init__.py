"""
Evaluation, validation protocols, and statistical testing.

Maintains backward compatibility with legacy evaluation functions while
providing leak-free nested CV, patient/sequence grouping, and statistical tests.
"""

# Re-export legacy evaluation functions for complete backward compatibility
from .legacy_evaluation import (
    bootstrap_auc,
    cluster_similarity_cv,
    compare_esm2_vs_baseline,
    compute_auc,
    compute_calibration_metrics,
    compute_classification_metrics,
    delong_test,
    isotonic_calibration,
    platt_scaling,
    stratified_cv,
    temperature_scaling,
)

# New Research V2 modular evaluation components
from .grouped_validation import (
    GroupedValidationSplitter,
    cluster_sequences_by_identity,
    compute_sequence_hamming_distance,
)
from .nested_cross_validation import (
    LeakFreeNestedCV,
    compute_binary_classification_metrics,
)
from .statistical_comparison import (
    benjamini_hochberg_correction,
    compute_bootstrap_confidence_interval,
    compute_delong_pvalue,
    paired_wilcoxon_multidrug_test,
)

__all__ = [
    # Legacy symbols
    "bootstrap_auc",
    "cluster_similarity_cv",
    "compare_esm2_vs_baseline",
    "compute_auc",
    "compute_calibration_metrics",
    "compute_classification_metrics",
    "delong_test",
    "isotonic_calibration",
    "platt_scaling",
    "stratified_cv",
    "temperature_scaling",
    # V2 symbols
    "GroupedValidationSplitter",
    "cluster_sequences_by_identity",
    "compute_sequence_hamming_distance",
    "LeakFreeNestedCV",
    "compute_binary_classification_metrics",
    "benjamini_hochberg_correction",
    "compute_bootstrap_confidence_interval",
    "compute_delong_pvalue",
    "paired_wilcoxon_multidrug_test",
]

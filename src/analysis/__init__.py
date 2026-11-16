"""Analysis modules for subtype robustness, calibration, and error breakdown."""

from .subtype_robustness import (
    compute_clade_b_vs_non_b_gap,
    evaluate_by_subtype,
)
from .calibration import (
    TemperatureScaler,
    compute_calibration_error,
)

__all__ = [
    "compute_clade_b_vs_non_b_gap",
    "evaluate_by_subtype",
    "TemperatureScaler",
    "compute_calibration_error",
]

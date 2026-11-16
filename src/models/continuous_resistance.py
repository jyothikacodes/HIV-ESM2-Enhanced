"""
Continuous and Ordinal Resistance Prediction on Phenotypic Fold-Change.

Experiment C:
Predicts log10(Fold-Change) directly as a continuous biophysical property,
avoiding loss of granular information caused by arbitrary binary binarization
and resolving class-imbalance failures on drugs with low resistant counts (e.g. TDF).

Includes:
- Continuous regression models (Ridge, ElasticNet, GradientBoosting).
- Regression evaluation metrics: R^2, RMSE, MAE, Pearson r, Spearman rho.
- Ordinal mapping to 3-class resistance (Susceptible, Intermediate, Resistant).
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    xgb = None
    _HAS_XGB = False

from ..representations.mutation_features import categorize_resistance_3class, get_clinical_cutoff


def compute_continuous_metrics(
    y_true_log_fc: np.ndarray,
    y_pred_log_fc: np.ndarray
) -> Dict[str, float]:
    """
    Compute rigorous continuous regression evaluation metrics on log10(FC).
    
    Args:
        y_true_log_fc: Ground-truth log10(Fold-Change) array.
        y_pred_log_fc: Predicted log10(Fold-Change) array.
        
    Returns:
        Dict containing R^2, RMSE, MAE, Pearson r, Spearman rho, and their p-values.
    """
    valid = (~np.isnan(y_true_log_fc)) & (~np.isnan(y_pred_log_fc))
    y_t = y_true_log_fc[valid]
    y_p = y_pred_log_fc[valid]
    
    if len(y_t) < 5:
        return {
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "n_samples": len(y_t),
        }
        
    r2 = r2_score(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    
    # Correlations
    pearson_r, pearson_p = stats.pearsonr(y_t, y_p)
    spearman_rho, spearman_p = stats.spearmanr(y_t, y_p)
    
    return {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "n_samples": int(len(y_t)),
    }


def evaluate_continuous_as_ordinal(
    y_true_log_fc: np.ndarray,
    y_pred_log_fc: np.ndarray,
    drug: str
) -> Dict[str, float]:
    """
    Transform continuous predictions back to linear fold-change and evaluate
    against clinical 3-class resistance boundaries (Susceptible / Intermediate / Resistant).
    """
    valid = (~np.isnan(y_true_log_fc)) & (~np.isnan(y_pred_log_fc))
    y_t = y_true_log_fc[valid]
    y_p = y_pred_log_fc[valid]
    
    if len(y_t) < 5:
        return {"ordinal_balanced_accuracy": np.nan, "ordinal_macro_f1": np.nan}
        
    # Reconvert from log10(FC) to linear FC
    true_linear_fc = np.power(10.0, y_t)
    pred_linear_fc = np.power(10.0, y_p)
    
    true_classes = categorize_resistance_3class(true_linear_fc, drug)
    pred_classes = categorize_resistance_3class(pred_linear_fc, drug)
    
    bal_acc = balanced_accuracy_score(true_classes, pred_classes)
    macro_f1 = f1_score(true_classes, pred_classes, average="macro", zero_division=0)
    
    return {
        "ordinal_balanced_accuracy": float(bal_acc),
        "ordinal_macro_f1": float(macro_f1),
    }


class ContinuousResistanceRegressor:
    """
    Fold-specific continuous resistance regressor on log10(Fold-Change).
    """
    def __init__(
        self,
        model_type: Literal["ridge", "elasticnet", "gradient_boosting", "xgboost"] = "gradient_boosting",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        random_state: int = 42
    ):
        self.model_type = model_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None

    def _init_model(self):
        if self.model_type == "ridge":
            return Ridge(alpha=self.alpha, random_state=self.random_state)
        elif self.model_type == "elasticnet":
            return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=self.random_state)
        elif self.model_type == "xgboost" and _HAS_XGB:
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=self.random_state,
                verbosity=0
            )
        else:
            # HistGradientBoostingRegressor standard fallback
            return HistGradientBoostingRegressor(
                max_iter=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=self.random_state
            )

    def fit(self, X: np.ndarray, y_linear_or_log_fc: np.ndarray, is_log10: bool = False) -> "ContinuousResistanceRegressor":
        """
        Fit regressor on training fold.
        Scales features internally strictly on training fold.
        """
        # Ensure log10 transformation
        if not is_log10:
            y_log = np.log10(np.clip(y_linear_or_log_fc, 1e-3, 1e5))
        else:
            y_log = y_linear_or_log_fc

        valid = ~np.isnan(y_log)
        X_clean = X[valid]
        y_clean = y_log[valid]

        X_scaled = self.scaler.fit_transform(X_clean)
        self.model = self._init_model()
        self.model.fit(X_scaled, y_clean)
        return self

    def predict_log_fc(self, X: np.ndarray) -> np.ndarray:
        """
        Predict log10(Fold-Change).
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_linear_fc(self, X: np.ndarray) -> np.ndarray:
        """
        Predict linear Fold-Change (10^y_pred).
        """
        log_preds = self.predict_log_fc(X)
        return np.power(10.0, log_preds)

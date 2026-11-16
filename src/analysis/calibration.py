"""
Probability calibration and reliability evaluation for HIV resistance models.

Computes:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score
- Reliability diagram bins
- Post-hoc calibration (Platt scaling, Isotonic regression, Temperature scaling)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def compute_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform"
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities in [0, 1].
        n_bins: Number of bins.
        strategy: 'uniform' (equal width) or 'quantile' (equal sample size).
    """
    valid = (~np.isnan(y_true)) & (~np.isnan(y_prob))
    yt = y_true[valid].astype(int)
    yp = np.clip(y_prob[valid], 1e-6, 1.0 - 1e-6)
    
    N = len(yt)
    if N == 0:
        return {"ece": np.nan, "mce": np.nan, "brier": np.nan}
        
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.percentile(yp, quantiles * 100)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (yp >= low) & (yp <= high)
        else:
            in_bin = (yp >= low) & (yp < high)
            
        count = in_bin.sum()
        if count > 0:
            acc = yt[in_bin].mean()
            conf = yp[in_bin].mean()
            diff = abs(acc - conf)
            
            ece += (count / N) * diff
            mce = max(mce, diff)
            
            bin_accs.append(acc)
            bin_confs.append(conf)
            bin_sizes.append(count)
        else:
            bin_accs.append(np.nan)
            bin_confs.append((low + high) / 2.0)
            bin_sizes.append(0)
            
    brier = brier_score_loss(yt, yp)
    
    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": float(brier),
        "bin_accuracies": np.array(bin_accs),
        "bin_confidences": np.array(bin_confs),
        "bin_sizes": np.array(bin_sizes),
        "bin_edges": bin_edges,
    }


class TemperatureScaler:
    """
    Post-hoc temperature scaling:
        P_calibrated = sigma(logit(P) / T)
    """
    def __init__(self):
        self.temperature = 1.0

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "TemperatureScaler":
        valid = (~np.isnan(y_true)) & (~np.isnan(y_prob))
        yt = y_true[valid]
        yp = np.clip(y_prob[valid], 1e-6, 1.0 - 1e-6)
        logits = np.log(yp / (1.0 - yp))

        def nll(T):
            if T <= 0:
                return 1e9
            scaled_logits = logits / T
            # Safe sigmoid and binary cross entropy
            p = 1.0 / (1.0 + np.exp(-scaled_logits))
            p = np.clip(p, 1e-8, 1.0 - 1e-8)
            return -np.mean(yt * np.log(p) + (1.0 - yt) * np.log(1.0 - p))

        res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x)
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        yp = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
        logits = np.log(yp / (1.0 - yp))
        scaled_logits = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled_logits))

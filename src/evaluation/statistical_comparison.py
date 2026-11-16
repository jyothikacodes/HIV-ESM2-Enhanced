"""
Statistical hypothesis testing and confidence interval estimation for model comparisons.

Provides:
- Paired DeLong test for comparing ROC AUCs between two models.
- Paired Wilcoxon signed-rank test across all 18 drugs.
- Benjamini-Hochberg False Discovery Rate (FDR) adjustment (q-values).
- Stratified bootstrap confidence intervals (95% CI).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def compute_delong_pvalue(
    y_true: np.ndarray,
    y_pred_model1: np.ndarray,
    y_pred_model2: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute paired DeLong test for comparing two ROC curves on the same cohort.
    
    Returns:
        (auc1, auc2, p_value)
    """
    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred_model1)) & (~np.isnan(y_pred_model2))
    yt = y_true[valid].astype(int)
    p1 = y_pred_model1[valid]
    p2 = y_pred_model2[valid]
    
    auc1 = roc_auc_score(yt, p1)
    auc2 = roc_auc_score(yt, p2)
    
    # Fast vectorized DeLong variance calculation
    pos_idx = np.where(yt == 1)[0]
    neg_idx = np.where(yt == 0)[0]
    m, n = len(pos_idx), len(neg_idx)
    
    if m == 0 or n == 0:
        return auc1, auc2, 1.0

    # Structural components
    v10_1 = np.array([np.mean(p1[pos_idx[i]] > p1[neg_idx]) + 0.5 * np.mean(p1[pos_idx[i]] == p1[neg_idx]) for i in range(m)])
    v01_1 = np.array([np.mean(p1[pos_idx] > p1[neg_idx[j]]) + 0.5 * np.mean(p1[pos_idx] == p1[neg_idx[j]]) for j in range(n)])
    
    v10_2 = np.array([np.mean(p2[pos_idx[i]] > p2[neg_idx]) + 0.5 * np.mean(p2[pos_idx[i]] == p2[neg_idx]) for i in range(m)])
    v01_2 = np.array([np.mean(p2[pos_idx] > p2[neg_idx[j]]) + 0.5 * np.mean(p2[pos_idx] == p2[neg_idx[j]]) for j in range(n)])
    
    s10 = np.cov(v10_1, v10_2)
    s01 = np.cov(v01_1, v01_2)
    
    s = s10 / m + s01 / n
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    
    if var_diff <= 0:
        return auc1, auc2, 1.0
        
    z = (auc1 - auc2) / np.sqrt(var_diff)
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return float(auc1), float(auc2), float(p_val)


def benjamini_hochberg_correction(p_values: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Perform Benjamini-Hochberg step-up procedure to control FDR at level alpha.
    
    Returns:
        q_values: Adjusted p-values maintaining monotonicity.
    """
    p_vals = np.array(p_values, dtype=float)
    n = len(p_vals)
    if n == 0:
        return np.array([])
        
    sorted_order = np.argsort(p_vals)
    sorted_p = p_vals[sorted_order]
    
    q_vals = np.zeros(n, dtype=float)
    accum_min = 1.0
    
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj_p = (sorted_p[i] * n) / rank
        accum_min = min(accum_min, adj_p)
        q_vals[sorted_order[i]] = min(1.0, accum_min)
        
    return q_vals


def paired_wilcoxon_multidrug_test(
    baseline_aucs: List[float],
    proposed_aucs: List[float]
) -> Dict[str, float]:
    """
    Paired Wilcoxon signed-rank test comparing baseline vs proposed across all 18 drugs.
    """
    b = np.array(baseline_aucs)
    p = np.array(proposed_aucs)
    diff = p - b
    
    res = stats.wilcoxon(diff, alternative="two-sided")
    
    return {
        "mean_baseline_auc": float(np.mean(b)),
        "mean_proposed_auc": float(np.mean(p)),
        "mean_delta_auc": float(np.mean(diff)),
        "median_delta_auc": float(np.median(diff)),
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "proposed_wins": int(np.sum(diff > 0)),
        "baseline_wins": int(np.sum(diff < 0)),
        "ties": int(np.sum(diff == 0)),
    }


def compute_bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Any = roc_auc_score,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute non-parametric bootstrap confidence interval for any metric.
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(random_state)
    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    yt = y_true[valid]
    yp = y_pred[valid]
    
    point_est = metric_fn(yt, yp)
    scores = []
    
    n_samples = len(yt)
    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n_samples, size=n_samples)
        if len(np.unique(yt[boot_idx])) < 2:
            continue
        score = metric_fn(yt[boot_idx], yp[boot_idx])
        scores.append(score)
        
    alpha = (1.0 - ci) / 2.0
    lower = np.percentile(scores, 100 * alpha)
    upper = np.percentile(scores, 100 * (1.0 - alpha))
    
    return float(point_est), float(lower), float(upper)

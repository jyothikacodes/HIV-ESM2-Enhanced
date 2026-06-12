"""
Model evaluation and statistical testing utilities.

This module provides functions for:
- Computing classification metrics (AUC-ROC, AUC-PR)
- Stratified cross-validation
- DeLong test for AUC comparison
- Calibration metrics and Platt scaling
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute ROC-AUC score.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities

    Returns:
        AUC-ROC score
    """
    return roc_auc_score(y_true, y_pred)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    auc_roc = roc_auc_score(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)

    # Confusion matrix components
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'brier_score': brier,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def stratified_cv(
    X: np.ndarray,
    y: np.ndarray,
    model,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[List[float], np.ndarray]:
    """
    Perform stratified k-fold cross-validation.

    Args:
        X: Feature matrix
        y: Labels
        model: Sklearn-compatible model with predict_proba
        n_splits: Number of folds
        random_state: Random seed

    Returns:
        Tuple of (cv_scores, all_predictions)
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_scores = []
    all_preds = np.zeros(len(y))

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]

        fold_auc = roc_auc_score(y_val, y_pred)
        cv_scores.append(fold_auc)

        all_preds[val_idx] = y_pred

    return cv_scores, all_preds


def delong_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray
) -> Tuple[float, float]:
    """
    DeLong test for comparing two AUC values.

    Uses the nonparametric placement value approach (DeLong et al., 1988)
    to test whether two ROC curves are significantly different.

    Args:
        y_true: True labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2

    Returns:
        Tuple of (z_statistic, p_value)
    """
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1).flatten()
    y_pred2 = np.asarray(y_pred2).flatten()

    pos = y_pred1[y_true == 1], y_pred2[y_true == 1]
    neg = y_pred1[y_true == 0], y_pred2[y_true == 0]
    m = len(pos[0])  # number of positives
    n = len(neg[0])  # number of negatives

    # Placement values for each model
    def _placement(pos_scores, neg_scores):
        """Compute placement values: fraction of negatives below each positive."""
        return np.array([
            np.mean(pos_s > neg_scores) + 0.5 * np.mean(pos_s == neg_scores)
            for pos_s in pos_scores
        ])

    def _placement_neg(pos_scores, neg_scores):
        """Compute placement values: fraction of positives above each negative."""
        return np.array([
            np.mean(pos_scores > neg_s) + 0.5 * np.mean(pos_scores == neg_s)
            for neg_s in neg_scores
        ])

    # Placement values for positives
    v10 = _placement(pos[0], neg[0])  # model 1
    v20 = _placement(pos[1], neg[1])  # model 2

    # Placement values for negatives
    v01 = _placement_neg(pos[0], neg[0])
    v02 = _placement_neg(pos[1], neg[1])

    # AUCs
    auc1 = np.mean(v10)
    auc2 = np.mean(v20)

    # Covariance matrix of (AUC1, AUC2)
    sx = np.cov(np.column_stack([v10, v20]).T)  # (2,2) covariance among positives
    sy = np.cov(np.column_stack([v01, v02]).T)  # (2,2) covariance among negatives

    S = sx / m + sy / n  # combined covariance

    # Contrast: AUC1 - AUC2
    diff = auc1 - auc2
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]

    if var_diff <= 0:
        return 0.0, 1.0

    z = diff / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p_value


def bootstrap_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for AUC-ROC.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level
        random_state: Random seed

    Returns:
        Tuple of (point_estimate, lower_ci, upper_ci)
    """
    np.random.seed(random_state)

    n = len(y_true)
    aucs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        aucs.append(roc_auc_score(y_true_boot, y_pred_boot))

    aucs = np.array(aucs)
    point_estimate = np.mean(aucs)

    alpha = (1 - confidence_level) / 2
    lower = np.percentile(aucs, 100 * alpha)
    upper = np.percentile(aucs, 100 * (1 - alpha))

    return point_estimate, lower, upper


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """
    Compute calibration metrics (ECE, MCE, Brier score).

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of calibration bins

    Returns:
        Dictionary with ECE, MCE, Brier score, and bin statistics
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins[1:-1])

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_pred_proba[mask].mean()
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_counts.append(mask.sum())

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)

    # Expected Calibration Error
    ece = np.sum(bin_counts * np.abs(bin_accs - bin_confs)) / len(y_true)

    # Maximum Calibration Error
    mce = np.max(np.abs(bin_accs - bin_confs)) if len(bin_accs) > 0 else 0

    # Brier score
    brier = brier_score_loss(y_true, y_pred_proba)

    return {
        'ece': ece,
        'mce': mce,
        'brier_score': brier,
        'bin_accuracies': bin_accs,
        'bin_confidences': bin_confs,
        'bin_counts': bin_counts
    }


def platt_scaling(
    y_true_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    y_pred_test: np.ndarray
) -> np.ndarray:
    """
    Apply Platt scaling for probability calibration.

    Args:
        y_true_cal: True labels for calibration set
        y_pred_cal: Predictions for calibration set
        y_pred_test: Predictions to calibrate

    Returns:
        Calibrated probabilities for test set
    """
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(y_pred_cal.reshape(-1, 1), y_true_cal)

    return lr.predict_proba(y_pred_test.reshape(-1, 1))[:, 1]


def isotonic_calibration(
    y_true_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    y_pred_test: np.ndarray
) -> np.ndarray:
    """
    Apply isotonic regression for probability calibration.

    Args:
        y_true_cal: True labels for calibration set
        y_pred_cal: Predictions for calibration set
        y_pred_test: Predictions to calibrate

    Returns:
        Calibrated probabilities for test set
    """
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_pred_cal, y_true_cal)

    return iso.predict(y_pred_test)


def temperature_scaling(
    y_true_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    y_pred_test: np.ndarray,
    eps: float = 1e-9
) -> np.ndarray:
    """
    Apply temperature scaling for binary probability calibration.

    Fits a single temperature T on the calibration set by minimizing
    negative log-likelihood, then rescales test logits.

    Args:
        y_true_cal: True labels for calibration set
        y_pred_cal: Predicted probabilities for calibration set
        y_pred_test: Predicted probabilities to calibrate
        eps: Numerical stability constant

    Returns:
        Temperature-scaled probabilities for test set
    """
    from scipy.optimize import minimize_scalar

    y_true_cal = np.asarray(y_true_cal, dtype=float)
    y_pred_cal = np.clip(np.asarray(y_pred_cal, dtype=float), eps, 1.0 - eps)
    y_pred_test = np.clip(np.asarray(y_pred_test, dtype=float), eps, 1.0 - eps)

    logits_cal = np.log(y_pred_cal / (1.0 - y_pred_cal))

    def nll(temperature: float) -> float:
        scaled = 1.0 / (1.0 + np.exp(-logits_cal / max(temperature, eps)))
        scaled = np.clip(scaled, eps, 1.0 - eps)
        return -np.mean(
            y_true_cal * np.log(scaled) + (1.0 - y_true_cal) * np.log(1.0 - scaled)
        )

    result = minimize_scalar(nll, bounds=(0.05, 20.0), method='bounded')
    temperature = max(float(result.x), eps)

    logits_test = np.log(y_pred_test / (1.0 - y_pred_test))
    scaled_test = 1.0 / (1.0 + np.exp(-logits_test / temperature))
    return np.clip(scaled_test, eps, 1.0 - eps)


def compare_esm2_vs_baseline(
    esm2_results: Dict,
    baseline_results: Dict,
    drugs: List[str]
) -> pd.DataFrame:
    """
    Compare ESM-2 model against baseline across all drugs.

    Args:
        esm2_results: Results dict from per_drug_training with ESM-2 features
        baseline_results: Results dict from per_drug_training with baseline features
        drugs: List of drugs to compare

    Returns:
        DataFrame with comparison statistics
    """
    comparison = []

    for drug in drugs:
        if drug not in esm2_results or drug not in baseline_results:
            continue

        esm2_auc = esm2_results[drug]['auc']
        baseline_auc = baseline_results[drug]['auc']

        # DeLong test
        y_true = esm2_results[drug]['y_true']
        y_esm2 = esm2_results[drug]['y_pred']
        y_baseline = baseline_results[drug]['y_pred']

        z_stat, p_value = delong_test(y_true, y_esm2, y_baseline)

        comparison.append({
            'drug': drug,
            'esm2_auc': esm2_auc,
            'baseline_auc': baseline_auc,
            'improvement': esm2_auc - baseline_auc,
            'delong_z': z_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'esm2_better': esm2_auc > baseline_auc
        })

    df = pd.DataFrame(comparison)

    # Add summary
    summary = pd.DataFrame([{
        'drug': 'MEAN',
        'esm2_auc': df['esm2_auc'].mean(),
        'baseline_auc': df['baseline_auc'].mean(),
        'improvement': df['improvement'].mean(),
        'delong_z': np.nan,
        'p_value': np.nan,
        'significant': np.nan,
        'esm2_better': df['esm2_better'].sum()
    }])

    return pd.concat([df, summary], ignore_index=True)

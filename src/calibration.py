"""
Unified calibration module for HIV drug resistance predictions.

Provides:
- Platt scaling and isotonic regression wrappers
- Auto-calibration (selects the best method via internal cross-validation)
- Multiclass (ternary) probability calibration using One-vs-Rest scaling
- Pre- vs Post-calibration evaluation (ECE, MCE, Brier score)
- Out-of-fold CV calibration for per-drug evaluation results
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, roc_auc_score

# Import existing calibration scaling functions from evaluation.py
from .evaluation import (
    platt_scaling,
    isotonic_calibration,
    temperature_scaling,
    compute_calibration_metrics
)


def calibrate_predictions(
    y_true_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    y_pred_test: np.ndarray,
    method: str = 'platt'
) -> np.ndarray:
    """
    Calibrate binary prediction probabilities using Platt scaling, isotonic regression,
    or automatically choosing the best method.

    Args:
        y_true_cal: Binary true labels for calibration set (n_samples_cal,)
        y_pred_cal: Raw predicted probabilities for calibration set (n_samples_cal,)
        y_pred_test: Raw predicted probabilities to be calibrated (n_samples_test,)
        method: Calibration method ('platt', 'isotonic', 'temperature', or 'auto')

    Returns:
        Calibrated probabilities for the test set
    """
    # Ensure inputs are numpy arrays
    y_true_cal = np.asarray(y_true_cal)
    y_pred_cal = np.asarray(y_pred_cal)
    y_pred_test = np.asarray(y_pred_test)

    if method == 'platt':
        return platt_scaling(y_true_cal, y_pred_cal, y_pred_test)
    elif method == 'isotonic':
        return isotonic_calibration(y_true_cal, y_pred_cal, y_pred_test)
    elif method == 'temperature':
        return temperature_scaling(y_true_cal, y_pred_cal, y_pred_test)
    elif method == 'auto':
        # Select best method using cross-validation on the calibration set
        n_splits = min(3, int(np.min(np.bincount(y_true_cal.astype(int)))))
        if n_splits < 2:
            return platt_scaling(y_true_cal, y_pred_cal, y_pred_test)

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        platt_brier = 0.0
        isotonic_brier = 0.0

        for train_idx, val_idx in cv.split(y_pred_cal, y_true_cal):
            y_train_c, y_val_c = y_true_cal[train_idx], y_true_cal[val_idx]
            y_pred_train_c, y_pred_val_c = y_pred_cal[train_idx], y_pred_cal[val_idx]

            try:
                p_cal = platt_scaling(y_train_c, y_pred_train_c, y_pred_val_c)
                platt_brier += brier_score_loss(y_val_c, p_cal)
            except Exception:
                platt_brier += 1.0

            try:
                i_cal = isotonic_calibration(y_train_c, y_pred_train_c, y_pred_val_c)
                isotonic_brier += brier_score_loss(y_val_c, i_cal)
            except Exception:
                isotonic_brier += 1.0

        best_method = 'platt' if platt_brier <= isotonic_brier else 'isotonic'
        return calibrate_predictions(y_true_cal, y_pred_cal, y_pred_test, method=best_method)
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def calibrate_ternary_predictions(
    y_true_cal: np.ndarray,
    y_pred_proba_cal: np.ndarray,
    y_pred_proba_test: np.ndarray,
    method: str = 'platt'
) -> np.ndarray:
    """
    Calibrate ternary (3-class) predictions using a One-vs-Rest strategy.

    Each class probability is calibrated independently, and then the probabilities
    are renormalized to sum to 1.

    Args:
        y_true_cal: True labels (0, 1, 2) for calibration set
        y_pred_proba_cal: Predicted class probabilities (n_samples_cal, 3)
        y_pred_proba_test: Predicted class probabilities to calibrate (n_samples_test, 3)
        method: Calibration method ('platt', 'isotonic', 'temperature', or 'auto')

    Returns:
        Calibrated class probabilities of shape (n_samples_test, 3)
    """
    y_true_cal = np.asarray(y_true_cal)
    y_pred_proba_cal = np.asarray(y_pred_proba_cal)
    y_pred_proba_test = np.asarray(y_pred_proba_test)

    n_classes = y_pred_proba_cal.shape[1]
    assert n_classes == 3, f"Expected 3 classes, got {n_classes}"

    calibrated_probs = []

    for c in range(3):
        # Define binary labels for class c vs all others
        y_true_cal_c = (y_true_cal == c).astype(int)
        y_pred_cal_c = y_pred_proba_cal[:, c]
        y_pred_test_c = y_pred_proba_test[:, c]

        # Calibrate class c probabilities
        try:
            # We clip the outputs of Platt/Isotonic scaling to ensure stability
            p_cal = calibrate_predictions(y_true_cal_c, y_pred_cal_c, y_pred_test_c, method=method)
            p_cal = np.clip(p_cal, 1e-9, 1.0 - 1e-9)
        except Exception:
            # If calibration fails (e.g. class not present in calibration fold), keep raw probability
            p_cal = y_pred_test_c

        calibrated_probs.append(p_cal)

    # Stack and renormalize
    calibrated_matrix = np.column_stack(calibrated_probs)
    row_sums = calibrated_matrix.sum(axis=1, keepdims=True)
    calibrated_matrix = calibrated_matrix / (row_sums + 1e-9)

    return calibrated_matrix


def evaluate_calibration(
    y_true: np.ndarray,
    y_pred_raw: np.ndarray,
    y_pred_calibrated: np.ndarray
) -> Dict:
    """
    Evaluate and compare pre- and post-calibration metrics.

    Args:
        y_true: True labels (binary or ternary)
        y_pred_raw: Raw predicted probabilities or classes
        y_pred_calibrated: Calibrated predicted probabilities or classes

    Returns:
        Dict comparing raw vs calibrated ECE, MCE, and Brier Score
    """
    # For binary inputs, we can directly compute ECE/MCE/Brier
    # For ternary inputs, we compute them in a One-vs-Rest fashion and average
    y_true = np.asarray(y_true)
    y_pred_raw = np.asarray(y_pred_raw)
    y_pred_calibrated = np.asarray(y_pred_calibrated)

    if len(y_pred_raw.shape) == 1 or y_pred_raw.shape[1] == 1:
        # Binary case
        raw_metrics = compute_calibration_metrics(y_true, y_pred_raw.flatten())
        cal_metrics = compute_calibration_metrics(y_true, y_pred_calibrated.flatten())
        return {
            'raw': raw_metrics,
            'calibrated': cal_metrics
        }
    else:
        # Multiclass (ternary) case: calculate OvR metrics and average
        n_classes = y_pred_raw.shape[1]
        raw_ece, raw_mce, raw_brier = [], [], []
        cal_ece, cal_mce, cal_brier = [], [], []

        for c in range(n_classes):
            y_true_c = (y_true == c).astype(int)
            
            raw_c = compute_calibration_metrics(y_true_c, y_pred_raw[:, c])
            cal_c = compute_calibration_metrics(y_true_c, y_pred_calibrated[:, c])

            raw_ece.append(raw_c['ece'])
            raw_mce.append(raw_c['mce'])
            raw_brier.append(raw_c['brier_score'])

            cal_ece.append(cal_c['ece'])
            cal_mce.append(cal_c['mce'])
            cal_brier.append(cal_c['brier_score'])

        return {
            'raw': {
                'ece': np.mean(raw_ece),
                'mce': np.mean(raw_mce),
                'brier_score': np.mean(raw_brier)
            },
            'calibrated': {
                'ece': np.mean(cal_ece),
                'mce': np.mean(cal_mce),
                'brier_score': np.mean(cal_brier)
            }
        }


def calibrate_per_drug(
    results: Dict,
    method: str = 'platt',
    n_splits: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Convenience function to calibrate per-drug cross-validation predictions.

    Performs out-of-fold calibration for each drug in the results dictionary.
    Supports both binary predictions (`y_pred` / `y_true`) and ternary predictions
    (`y_pred_proba` / `y_true`).

    Args:
        results: Dictionary containing per-drug evaluation results (e.g. from per_drug_training)
        method: Calibration method ('platt', 'isotonic', 'temperature', or 'auto')
        n_splits: Number of splits for out-of-fold calibration
        random_state: Random state for reproducibility

    Returns:
        A dictionary containing the calibrated results, with updated predictions and metrics.
    """
    calibrated_results = {}

    for drug, res in results.items():
        if res.get('skipped', False):
            calibrated_results[drug] = res.copy()
            continue

        y_true = np.asarray(res['y_true'])
        
        # 1. Determine if ternary or binary
        is_ternary = 'y_pred_proba' in res and res['y_pred_proba'].ndim == 2 and res['y_pred_proba'].shape[1] == 3
        
        if is_ternary:
            y_pred_raw = np.asarray(res['y_pred_proba'])
            y_pred_calibrated = np.zeros_like(y_pred_raw)
        else:
            y_pred_raw = np.asarray(res.get('y_pred', res.get('y_pred_proba', None)))
            if y_pred_raw is None:
                calibrated_results[drug] = res.copy()
                continue
            y_pred_calibrated = np.zeros_like(y_pred_raw)

        # 2. Out-of-fold calibration to avoid data leakage
        n_classes = len(np.unique(y_true))
        if n_classes < 2:
            calibrated_results[drug] = res.copy()
            continue

        # Adjust folds if sample counts are low
        min_class_count = np.min(np.bincount(y_true.astype(int)))
        effective_splits = min(n_splits, int(min_class_count))

        if effective_splits < 2:
            # Insufficient samples for CV calibration, calibrate in-sample (fallback)
            if is_ternary:
                y_pred_calibrated = calibrate_ternary_predictions(y_true, y_pred_raw, y_pred_raw, method=method)
            else:
                y_pred_calibrated = calibrate_predictions(y_true, y_pred_raw, y_pred_raw, method=method)
        else:
            cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
            
            for train_idx, val_idx in cv.split(y_pred_raw, y_true):
                y_train_true, y_val_true = y_true[train_idx], y_true[val_idx]
                
                if is_ternary:
                    y_train_pred, y_val_pred = y_pred_raw[train_idx], y_pred_raw[val_idx]
                    y_pred_calibrated[val_idx] = calibrate_ternary_predictions(
                        y_train_true, y_train_pred, y_val_pred, method=method
                    )
                else:
                    y_train_pred, y_val_pred = y_pred_raw[train_idx], y_pred_raw[val_idx]
                    y_pred_calibrated[val_idx] = calibrate_predictions(
                        y_train_true, y_train_pred, y_val_pred, method=method
                    )

        # 3. Compile calibrated result dictionary
        cal_res = res.copy()
        
        if is_ternary:
            cal_res['y_pred_proba'] = y_pred_calibrated
            # Update predicted class label by taking argmax
            cal_res['y_pred'] = np.argmax(y_pred_calibrated, axis=1)
            # Recompute macro F1 and accuracy
            from sklearn.metrics import accuracy_score, f1_score
            cal_res['accuracy'] = accuracy_score(y_true, cal_res['y_pred'])
            cal_res['macro_f1'] = f1_score(y_true, cal_res['y_pred'], average='macro')
            
            # Recalculate multi-class AUC
            try:
                cal_res['auc_ovr'] = roc_auc_score(y_true, y_pred_calibrated, multi_class='ovr', average='macro')
            except ValueError:
                pass
        else:
            cal_res['y_pred'] = y_pred_calibrated
            # Recompute binary AUC
            try:
                cal_res['auc'] = roc_auc_score(y_true, y_pred_calibrated)
            except ValueError:
                pass

        # Add pre/post calibration metrics
        cal_res['calibration_evaluation'] = evaluate_calibration(y_true, y_pred_raw, y_pred_calibrated)
        
        # Also store raw ECE/MCE/Brier for convenience
        metrics = evaluate_calibration(y_true, y_pred_raw, y_pred_calibrated)
        cal_res['ece_raw'] = metrics['raw']['ece']
        cal_res['ece_calibrated'] = metrics['calibrated']['ece']
        cal_res['brier_raw'] = metrics['raw']['brier_score']
        cal_res['brier_calibrated'] = metrics['calibrated']['brier_score']

        calibrated_results[drug] = cal_res

    return calibrated_results

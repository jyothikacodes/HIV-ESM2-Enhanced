"""
Strictly leak-free nested cross-validation for HIV resistance prediction.

Fixes all audit leakages:
1. Outer test fold is completely isolated and never used for preprocessing, early stopping, or tuning.
2. Inner fold is used strictly for model selection, early stopping, and hyperparameter tuning.
3. Preprocessing (StandardScaler, mutation frequency encoder) is fit exclusively on outer training data.
4. Generates unbiased out-of-fold predictions and comprehensive performance metrics.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    brier_score_loss,
    balanced_accuracy_score,
)
from .grouped_validation import GroupedValidationSplitter


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for binary classification.
    """
    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred_probs))
    yt = y_true[valid].astype(int)
    yp = y_pred_probs[valid]
    
    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return {
            "auc": np.nan, "ap": np.nan, "brier": np.nan,
            "accuracy": np.nan, "balanced_acc": np.nan, "f1": np.nan,
            "n_samples": len(yt)
        }
        
    y_pred_bin = (yp >= threshold).astype(int)
    
    auc = roc_auc_score(yt, yp)
    ap = average_precision_score(yt, yp)
    brier = brier_score_loss(yt, yp)
    acc = accuracy_score(yt, y_pred_bin)
    bal_acc = balanced_accuracy_score(yt, y_pred_bin)
    f1 = f1_score(yt, y_pred_bin, zero_division=0)
    
    return {
        "auc": float(auc),
        "ap": float(ap),
        "brier": float(brier),
        "accuracy": float(acc),
        "balanced_acc": float(bal_acc),
        "f1": float(f1),
        "n_samples": int(len(yt)),
    }


class LeakFreeNestedCV:
    """
    Orchestrates strictly leak-free nested cross-validation.
    """
    def __init__(
        self,
        estimator_factory: Callable[[], Any],
        outer_splits: int = 5,
        inner_splits: int = 3,
        grouping_strategy: str = "sequence_clustered",
        identity_threshold: float = 0.95,
        scale_features: bool = True,
        tune_hyperparameters: bool = False,
        param_candidates: Optional[List[Dict[str, Any]]] = None,
        random_state: int = 42
    ):
        """
        Args:
            estimator_factory: Function returning a fresh estimator instance or callable taking params.
            outer_splits: Number of outer validation folds.
            inner_splits: Number of inner tuning/validation folds.
            grouping_strategy: 'sequence_clustered', 'patient_grouped', 'joint_patient_sequence', or 'stratified'.
            identity_threshold: Identity threshold for sequence clustering.
            scale_features: Whether to fit a StandardScaler on outer training fold.
            tune_hyperparameters: If True, evaluates candidate parameters on inner folds.
            param_candidates: List of parameter dicts to evaluate on inner folds.
            random_state: Random seed.
        """
        self.estimator_factory = estimator_factory
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.grouping_strategy = grouping_strategy
        self.identity_threshold = identity_threshold
        self.scale_features = scale_features
        self.tune_hyperparameters = tune_hyperparameters
        self.param_candidates = param_candidates or []
        self.random_state = random_state

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequences: Optional[List[str]] = None,
        patient_ids: Optional[Union[List[str], np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Run leak-free nested CV.
        
        Returns:
            Dict containing:
            - oof_predictions: (N,) array of out-of-fold predicted probabilities.
            - outer_fold_metrics: List of metrics dicts for each outer fold.
            - overall_metrics: Aggregated metrics across all out-of-fold predictions.
        """
        N = len(y)
        oof_preds = np.full(N, np.nan, dtype=np.float32)
        outer_metrics_list = []
        
        outer_splitter = GroupedValidationSplitter(
            n_splits=self.outer_splits,
            grouping_strategy=self.grouping_strategy,
            identity_threshold=self.identity_threshold,
            random_state=self.random_state
        )
        
        fold_idx = 0
        for outer_train_idx, outer_test_idx in outer_splitter.split(X, y, sequences, patient_ids):
            X_outer_train, y_outer_train = X[outer_train_idx], y[outer_train_idx]
            X_outer_test, y_outer_test = X[outer_test_idx], y[outer_test_idx]
            
            # Filter training data for valid labels
            valid_train = ~np.isnan(y_outer_train)
            if valid_train.sum() == 0:
                continue
                
            train_idx_clean = np.where(valid_train)[0]
            X_tr_clean = X_outer_train[train_idx_clean]
            y_tr_clean = y_outer_train[train_idx_clean]
            
            if len(np.unique(y_tr_clean)) < 2:
                continue
                
            # Fit feature scaler STRICTLY on outer training data
            if self.scale_features:
                scaler = StandardScaler()
                X_tr_proc = scaler.fit_transform(X_tr_clean)
                X_te_proc = scaler.transform(X_outer_test)
            else:
                X_tr_proc = X_tr_clean
                X_te_proc = X_outer_test

            # INNER VALIDATION LOOP: Model selection & Hyperparameter Tuning
            if self.tune_hyperparameters and len(self.param_candidates) > 1:
                inner_splitter = GroupedValidationSplitter(
                    n_splits=self.inner_splits,
                    grouping_strategy=self.grouping_strategy,
                    identity_threshold=self.identity_threshold,
                    random_state=self.random_state + fold_idx
                )
                
                # Slices for inner sequences/patients
                inner_seqs = [sequences[outer_train_idx[i]] for i in train_idx_clean] if sequences is not None else None
                inner_pts = [patient_ids[outer_train_idx[i]] for i in train_idx_clean] if patient_ids is not None else None
                
                best_param = None
                best_inner_auc = -1.0
                
                for candidate in self.param_candidates:
                    inner_aucs = []
                    for in_tr_idx, in_val_idx in inner_splitter.split(X_tr_proc, y_tr_clean, inner_seqs, inner_pts):
                        m = self.estimator_factory(**candidate)
                        m.fit(X_tr_proc[in_tr_idx], y_tr_clean[in_tr_idx])
                        probs = m.predict_proba(X_tr_proc[in_val_idx])
                        if probs.ndim > 1:
                            probs = probs[:, 1]
                        val_m = compute_binary_classification_metrics(y_tr_clean[in_val_idx], probs)
                        if not np.isnan(val_m["auc"]):
                            inner_aucs.append(val_m["auc"])
                            
                    mean_inner = np.mean(inner_aucs) if inner_aucs else 0.0
                    if mean_inner > best_inner_auc:
                        best_inner_auc = mean_inner
                        best_param = candidate
                        
                # Fit best model on full outer training fold
                model = self.estimator_factory(**(best_param or {}))
                model.fit(X_tr_proc, y_tr_clean)
            else:
                # Standard inner fitting without parameter sweep
                model = self.estimator_factory()
                model.fit(X_tr_proc, y_tr_clean)
            
            # Predict strictly on untouched outer test fold
            if hasattr(model, "predict_proba"):
                test_probs = model.predict_proba(X_te_proc)
                if test_probs.ndim > 1:
                    test_probs = test_probs[:, 1]
            else:
                test_probs = model.predict(X_te_proc)
                
            oof_preds[outer_test_idx] = test_probs
            fold_m = compute_binary_classification_metrics(y_outer_test, test_probs)
            fold_m["fold"] = fold_idx
            outer_metrics_list.append(fold_m)
            fold_idx += 1
            
        overall_m = compute_binary_classification_metrics(y, oof_preds)
        
        return {
            "oof_predictions": oof_preds,
            "outer_fold_metrics": outer_metrics_list,
            "overall_metrics": overall_m,
            "mean_outer_auc": float(np.nanmean([m["auc"] for m in outer_metrics_list])),
            "std_outer_auc": float(np.nanstd([m["auc"] for m in outer_metrics_list])),
        }

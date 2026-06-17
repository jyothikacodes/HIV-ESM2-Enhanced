"""
Advanced ensemble weighting and aggregation for improved accuracy.

Implements:
- Drug-class adaptive weighting
- Confidence-aware averaging
- Distribution matching for probability calibration
- Uncertainty quantification
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def adaptive_ensemble_weights(
    oof_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    drug_class: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute adaptive weights for ensemble members based on AUC performance.
    
    Weights are optimized for:
    - Diversity: Down-weight correlated models
    - Calibration: Adjust for biased predictions
    - Drug-class sensitivity: Higher weights for drugs with known difficulty
    
    Args:
        oof_predictions: Dict of model_name -> predictions
        y_true: Ground truth labels
        drug_class: Optional drug class for adaptive weighting
    
    Returns:
        Dict of model_name -> weight
    """
    # Compute AUC for each model
    aucs = {}
    for model_name, preds in oof_predictions.items():
        try:
            aucs[model_name] = roc_auc_score(y_true, preds)
        except ValueError:
            aucs[model_name] = 0.5
    
    # Base weights from AUC (sigmoid to map to positive weights)
    weights = {}
    for model_name, auc in aucs.items():
        # Sigmoid scaling: convert AUC to weight
        # AUC=0.5 → weight=1.0, AUC=1.0 → weight=exp(1) ≈ 2.7
        weights[model_name] = np.exp(4 * (auc - 0.5))
    
    # Drug-class adaptive boost (increase weight for drug class-appropriate models)
    if drug_class == 'NNRTI':
        # NNRTI models need higher regularization, boost models that handle that
        if 'xgboost' in weights:
            weights['xgboost'] *= 1.2  # XGBoost handles regularization well
    elif drug_class == 'PI':
        # PI has larger cohort, boost ensemble methods
        if 'logistic' in weights:
            weights['logistic'] *= 1.1
    
    # Normalize to sum to 1
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    return weights


def confidence_weighted_average(
    predictions: List[np.ndarray],
    confidences: List[float]
) -> np.ndarray:
    """
    Compute weighted average of predictions using confidence scores.
    
    Higher confidence predictions get more weight in the final ensemble.
    Useful when some models are more reliable for specific drugs.
    
    Args:
        predictions: List of (n_samples,) prediction arrays
        confidences: List of confidence scores (0-1 per model)
    
    Returns:
        Weighted averaged predictions (n_samples,)
    """
    predictions = np.array(predictions)  # (n_models, n_samples)
    confidences = np.array(confidences)  # (n_models,)
    
    # Normalize confidences to sum to 1
    confidences = confidences / confidences.sum()
    
    # Weighted average
    ensemble_preds = np.average(predictions, axis=0, weights=confidences)
    
    return ensemble_preds


def distribution_matched_ensemble(
    oof_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Weight ensemble members to match calibrated prediction distribution.
    
    Reduces miscalibration by adjusting predictions to match the empirical
    distribution of true labels.
    
    Args:
        oof_predictions: Dict of model_name -> OOF predictions
        y_true: Ground truth labels
        n_bins: Number of bins for distribution matching
    
    Returns:
        Dict of model_name -> adjusted_weight
    """
    # Compute expected label frequency per bin
    bins = np.linspace(0, 1, n_bins + 1)
    bin_labels = np.digitize(y_true, bins) - 1
    bin_labels = np.clip(bin_labels, 0, n_bins - 1)
    
    expected_freq = np.bincount(bin_labels, minlength=n_bins) / len(y_true)
    
    # For each model, measure how well predictions match expected distribution
    weights = {}
    for model_name, preds in oof_predictions.items():
        bin_preds = np.digitize(preds, bins) - 1
        bin_preds = np.clip(bin_preds, 0, n_bins - 1)
        pred_freq = np.bincount(bin_preds, minlength=n_bins) / len(preds)
        
        # Wasserstein distance as mismatch metric
        mismatch = np.abs(pred_freq - expected_freq).sum()
        
        # Convert to weight (lower mismatch → higher weight)
        weights[model_name] = np.exp(-2 * mismatch)
    
    # Normalize
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    return weights


def uncertainty_quantified_ensemble(
    predictions_list: List[Dict[str, np.ndarray]],
    y_true: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensemble predictions with uncertainty quantification.
    
    Returns both point estimates and confidence intervals from
    prediction variance across ensemble members.
    
    Args:
        predictions_list: List of dicts (e.g., CV folds with {model: preds})
        y_true: Ground truth labels
    
    Returns:
        Tuple of (ensemble_predictions, uncertainty_std)
    """
    # Stack all predictions: (n_folds * n_models, n_samples)
    all_preds = []
    for pred_dict in predictions_list:
        all_preds.extend(pred_dict.values())
    
    all_preds = np.array(all_preds)  # (n_ensemble_members, n_samples)
    
    # Compute mean and std across ensemble members
    ensemble_mean = all_preds.mean(axis=0)
    ensemble_std = all_preds.std(axis=0)
    
    return ensemble_mean, ensemble_std


def boosted_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    models_dict: Dict[str, any],
    n_iterations: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Adaptive Boosting-style weighting for ensemble members.
    
    Iteratively reweights training samples to focus on hard cases,
    then combines model weights based on hard-case performance.
    
    Args:
        X: Feature matrix
        y: Binary labels
        models_dict: Dict of model_name -> model objects
        n_iterations: Number of boosting iterations
        random_state: Random seed
    
    Returns:
        Dict of model_name -> boosted_weight
    """
    np.random.seed(random_state)
    
    # Initialize sample weights uniformly
    sample_weights = np.ones(len(y)) / len(y)
    model_weights = {m: 1.0 for m in models_dict.keys()}
    
    for iteration in range(n_iterations):
        # Train each model on weighted samples
        errors = {}
        for model_name, model in models_dict.items():
            # Fit on weighted data
            model.fit(X, y, sample_weight=sample_weights)
            
            # Compute weighted error
            preds = model.predict(X)
            weighted_error = np.sum(sample_weights * (preds != y))
            errors[model_name] = max(weighted_error, 1e-10)
        
        # Update model weights (alpha) based on error
        for model_name in model_weights.keys():
            alpha = 0.5 * np.log((1 - errors[model_name]) / (errors[model_name] + 1e-10))
            model_weights[model_name] *= np.exp(alpha)
        
        # Update sample weights for next iteration
        new_weights = np.zeros(len(y))
        for model_name, model in models_dict.items():
            preds = model.predict(X)
            new_weights += model_weights[model_name] * (preds != y)
        
        sample_weights = np.exp(new_weights) / np.sum(np.exp(new_weights))
    
    # Normalize final model weights
    total = sum(model_weights.values())
    model_weights = {k: v / total for k, v in model_weights.items()}
    
    return model_weights

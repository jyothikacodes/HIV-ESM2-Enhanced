"""
Ternary (3-class) classification for HIV drug resistance.

This module extends the existing binary pipeline with an *optional*
three-class classification scheme:
    0 = Susceptible
    1 = Intermediate Resistance
    2 = Resistant

Controlled by a ``use_ternary`` flag — the existing binary pipeline is
completely untouched when this module is not invoked.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score
)
import xgboost as xgb


# ── Default ternary thresholds (FC values) ──────────────────────────────────
# Universal defaults: FC < 2.5 → Susceptible, 2.5 ≤ FC < 10 → Intermediate,
# FC ≥ 10 → Resistant.  These can be overridden per drug.

DEFAULT_TERNARY_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    # Format: drug -> (low_threshold, high_threshold)
    # When not listed here, the universal defaults are used.
    '_default': (2.5, 10.0),
}


def get_ternary_thresholds(drug: str) -> Tuple[float, float]:
    """
    Get ternary classification thresholds for a given drug.

    Args:
        drug: Drug abbreviation

    Returns:
        Tuple of (low_threshold, high_threshold) for FC values
    """
    return DEFAULT_TERNARY_THRESHOLDS.get(
        drug, DEFAULT_TERNARY_THRESHOLDS['_default']
    )


def extract_ternary_labels(
    phenotypes: pd.DataFrame,
    drug: str,
    low_threshold: Optional[float] = None,
    high_threshold: Optional[float] = None,
    fc_col_suffix: str = 'FC'
) -> np.ndarray:
    """
    Extract ternary resistance labels for a drug from fold-change values.

    Labels:
        0 = Susceptible (FC < low_threshold)
        1 = Intermediate (low_threshold <= FC < high_threshold)
        2 = Resistant (FC >= high_threshold)

    Args:
        phenotypes: DataFrame with phenotype data
        drug: Drug abbreviation
        low_threshold: FC threshold for susceptible/intermediate boundary.
                       If None, uses default for this drug.
        high_threshold: FC threshold for intermediate/resistant boundary.
                        If None, uses default for this drug.
        fc_col_suffix: Suffix for fold-change column (default 'FC')

    Returns:
        Array of ternary labels (0, 1, 2). NaN for missing values.
    """
    # Resolve thresholds
    if low_threshold is None or high_threshold is None:
        defaults = get_ternary_thresholds(drug)
        if low_threshold is None:
            low_threshold = defaults[0]
        if high_threshold is None:
            high_threshold = defaults[1]

    # Find the FC column
    fc_col = f"{drug}_{fc_col_suffix}" if fc_col_suffix else drug
    if fc_col not in phenotypes.columns:
        # Try bare drug name (raw FC values)
        if drug in phenotypes.columns:
            fc_col = drug
        else:
            raise ValueError(
                f"Neither '{fc_col}' nor '{drug}' found in phenotypes. "
                f"Available columns: {list(phenotypes.columns)}"
            )

    fc_values = phenotypes[fc_col].values.astype(float)
    labels = np.full(len(fc_values), np.nan, dtype=float)

    valid = ~np.isnan(fc_values)
    labels[valid & (fc_values < low_threshold)] = 0.0
    labels[valid & (fc_values >= low_threshold) & (fc_values < high_threshold)] = 1.0
    labels[valid & (fc_values >= high_threshold)] = 2.0

    return labels


def per_drug_ternary_training(
    X: np.ndarray,
    phenotypes: pd.DataFrame,
    drugs: List[str],
    model_type: str = 'logistic',
    n_splits: int = 5,
    random_state: int = 42,
    low_threshold: Optional[float] = None,
    high_threshold: Optional[float] = None
) -> Dict:
    """
    Train 3-class models for each drug and evaluate with cross-validation.

    This mirrors the interface of ``per_drug_training()`` from models.py
    but uses ternary labels and multi-class metrics.

    Args:
        X: Feature matrix (n_samples, n_features)
        phenotypes: DataFrame with drug resistance labels (FC values)
        drugs: List of drug names
        model_type: 'logistic' or 'xgboost'
        n_splits: Number of CV folds
        random_state: Random seed
        low_threshold: Override low threshold for all drugs
        high_threshold: Override high threshold for all drugs

    Returns:
        Dictionary with per-drug results including:
        - auc_ovr: One-vs-Rest AUC
        - accuracy: Classification accuracy
        - macro_f1: Macro-averaged F1
        - class_distribution: Count per class
    """
    results = {}

    for drug in drugs:
        try:
            y = extract_ternary_labels(
                phenotypes, drug,
                low_threshold=low_threshold,
                high_threshold=high_threshold
            )
        except ValueError as e:
            print(f"  Skipping {drug}: {e}")
            results[drug] = {
                'auc_ovr': np.nan, 'accuracy': np.nan,
                'macro_f1': np.nan, 'skipped': True,
                'reason': str(e)
            }
            continue

        # Filter valid samples
        valid_mask = ~np.isnan(y)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask].astype(int)

        n_classes = len(np.unique(y_valid))
        class_counts = {int(c): int((y_valid == c).sum()) for c in np.unique(y_valid)}

        if n_classes < 2:
            reason = f"only {n_classes} class(es) present"
            print(f"  Skipping {drug}: {reason}")
            results[drug] = {
                'auc_ovr': np.nan, 'accuracy': np.nan,
                'macro_f1': np.nan, 'skipped': True,
                'reason': reason, 'class_distribution': class_counts
            }
            continue

        # Ensure enough samples per class for stratified CV
        min_class_count = min(class_counts.values())
        effective_splits = min(n_splits, min_class_count)

        if effective_splits < 2:
            reason = f"min class count = {min_class_count}, need >= 2 for CV"
            print(f"  Skipping {drug}: {reason}")
            results[drug] = {
                'auc_ovr': np.nan, 'accuracy': np.nan,
                'macro_f1': np.nan, 'skipped': True,
                'reason': reason, 'class_distribution': class_counts
            }
            continue

        cv = StratifiedKFold(
            n_splits=effective_splits, shuffle=True,
            random_state=random_state
        )

        # Train and predict
        if model_type == 'logistic':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_valid)
            model = LogisticRegression(
                max_iter=1000, class_weight='balanced',
                random_state=random_state, solver='lbfgs'
            )
            y_pred_proba = cross_val_predict(
                model, X_scaled, y_valid, cv=cv, method='predict_proba'
            )
            y_pred = cross_val_predict(
                model, X_scaled, y_valid, cv=cv, method='predict'
            )

        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                random_state=random_state, eval_metric='mlogloss',
                n_jobs=-1, objective='multi:softprob',
                num_class=n_classes
            )
            y_pred_proba = cross_val_predict(
                model, X_valid, y_valid, cv=cv, method='predict_proba'
            )
            y_pred = cross_val_predict(
                model, X_valid, y_valid, cv=cv, method='predict'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Compute metrics
        accuracy = accuracy_score(y_valid, y_pred)
        macro_f1 = f1_score(y_valid, y_pred, average='macro')

        # One-vs-Rest AUC (handle case where some classes may be missing)
        try:
            if n_classes >= 3:
                auc_ovr = roc_auc_score(
                    y_valid, y_pred_proba, multi_class='ovr', average='macro'
                )
            else:
                # Fallback to binary AUC if only 2 classes
                auc_ovr = roc_auc_score(y_valid, y_pred_proba[:, 1])
        except ValueError:
            auc_ovr = np.nan

        results[drug] = {
            'auc_ovr': auc_ovr,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'n_samples': len(y_valid),
            'n_classes': n_classes,
            'class_distribution': class_counts,
            'y_true': y_valid,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
        }

        dist_str = ', '.join(f'{k}={v}' for k, v in sorted(class_counts.items()))
        print(f"  {drug}: AUC(OvR)={auc_ovr:.4f}  Acc={accuracy:.4f}  "
              f"F1={macro_f1:.4f}  (n={len(y_valid)}, {dist_str})")

    return results


def aggregate_ternary_results(results: Dict) -> pd.DataFrame:
    """
    Aggregate per-drug ternary results into a summary DataFrame.

    Args:
        results: Dictionary from per_drug_ternary_training()

    Returns:
        DataFrame with summary statistics per drug
    """
    rows = []

    for drug, res in results.items():
        if res.get('skipped', False):
            continue

        rows.append({
            'drug': drug,
            'auc_ovr': res['auc_ovr'],
            'accuracy': res['accuracy'],
            'macro_f1': res['macro_f1'],
            'n_samples': res.get('n_samples', 0),
            'n_classes': res.get('n_classes', 0),
            'class_distribution': str(res.get('class_distribution', {})),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Add mean row
    mean_row = pd.DataFrame([{
        'drug': 'MEAN',
        'auc_ovr': df['auc_ovr'].dropna().mean(),
        'accuracy': df['accuracy'].dropna().mean(),
        'macro_f1': df['macro_f1'].dropna().mean(),
        'n_samples': int(df['n_samples'].sum()),
        'n_classes': '',
        'class_distribution': '',
    }])
    df = pd.concat([df, mean_row], ignore_index=True)

    return df

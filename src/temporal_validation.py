"""
Extended temporal validation for HIV drug resistance prediction.

This module wraps and extends the existing temporal split and evaluation
functionality from subtype_analysis.py, adding:
- Temporal Accuracy
- Temporal Macro-F1
- Temporal Brier Score
- Side-by-side CV vs temporal comparison

The existing create_temporal_split() and temporal_holdout_evaluation()
in subtype_analysis.py are used as the foundation — not duplicated.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, brier_score_loss
)
import xgboost as xgb

from .subtype_analysis import create_temporal_split


def temporal_evaluation_extended(
    X: np.ndarray,
    phenotypes: pd.DataFrame,
    drugs: List[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_type: str = 'logistic',
    threshold: float = 0.5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluate model on temporal holdout with extended metrics.

    Extends the existing temporal_holdout_evaluation() with:
    - AUROC (already provided by the existing function)
    - Accuracy
    - Macro-F1
    - Brier Score

    Args:
        X: Feature matrix (n_samples, n_features)
        phenotypes: DataFrame with drug resistance labels
        drugs: List of drug names
        train_idx: Training set indices (older sequences)
        test_idx: Test set indices (newer sequences)
        model_type: 'logistic' or 'xgboost'
        threshold: Classification threshold for accuracy/F1
        random_state: Random seed

    Returns:
        DataFrame with per-drug temporal evaluation results including
        auroc, accuracy, macro_f1, and brier_score columns
    """
    rows = []

    for drug in drugs:
        # Resolve label column (same logic as existing pipeline)
        label_col = None
        for suffix in ['_class2', '_class3', '_FC', '']:
            candidate = f"{drug}{suffix}" if suffix else drug
            if candidate in phenotypes.columns:
                label_col = candidate
                break

        if label_col is None:
            print(f"  Skipping {drug}: no label column found")
            continue

        y = phenotypes[label_col].values.copy()

        # Binarize FC values if needed
        if label_col.endswith('_FC') or (
            not label_col.endswith('_class2') and
            not label_col.endswith('_class3')
        ):
            valid_fc = ~np.isnan(y)
            y_bin = np.full_like(y, np.nan, dtype=float)
            y_bin[valid_fc] = (y[valid_fc] >= 2.5).astype(float)
            y = y_bin

        # Get valid samples within each split
        train_valid = train_idx[~np.isnan(y[train_idx])]
        test_valid = test_idx[~np.isnan(y[test_idx])]

        if len(train_valid) < 10 or len(test_valid) < 5:
            print(f"  Skipping {drug}: insufficient samples in split")
            continue

        X_train = X[train_valid]
        y_train = y[train_valid].astype(int)
        X_test = X[test_valid]
        y_test = y[test_valid].astype(int)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"  Skipping {drug}: insufficient class diversity in split")
            continue

        # Train and predict
        if model_type == 'logistic':
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            model = LogisticRegression(
                max_iter=1000, class_weight='balanced',
                random_state=random_state
            )
            model.fit(X_train_s, y_train)
            y_pred_proba = model.predict_proba(X_test_s)[:, 1]
        elif model_type == 'xgboost':
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            model = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                scale_pos_weight=n_neg / n_pos if n_pos > 0 else 1.0,
                random_state=random_state, eval_metric='auc',
                n_jobs=-1
            )
            model.fit(X_train, y_train, verbose=0)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Compute all metrics
        y_pred_binary = (y_pred_proba >= threshold).astype(int)

        auroc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred_binary)
        macro_f1 = f1_score(y_test, y_pred_binary, average='macro')
        brier = brier_score_loss(y_test, y_pred_proba)

        rows.append({
            'drug': drug,
            'temporal_auroc': auroc,
            'temporal_accuracy': accuracy,
            'temporal_macro_f1': macro_f1,
            'temporal_brier_score': brier,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'n_resistant_train': int(y_train.sum()),
            'n_resistant_test': int(y_test.sum()),
            'test_prevalence': float(y_test.sum()) / len(y_test),
        })

        print(f"  {drug}: AUROC={auroc:.4f}  Acc={accuracy:.4f}  "
              f"F1={macro_f1:.4f}  Brier={brier:.4f}  "
              f"(train={len(y_train)}, test={len(y_test)})")

    return pd.DataFrame(rows)


def compare_temporal_vs_cv(
    temporal_results: pd.DataFrame,
    cv_results: Dict,
    metric: str = 'auroc'
) -> pd.DataFrame:
    """
    Compare temporal holdout results against cross-validation results.

    Args:
        temporal_results: DataFrame from temporal_evaluation_extended()
        cv_results: Dict from per_drug_training() (keyed by drug)
        metric: Which temporal metric to compare ('auroc', 'accuracy', 'macro_f1')

    Returns:
        DataFrame with side-by-side comparison for each drug
    """
    temporal_col = f'temporal_{metric}'
    rows = []

    for _, row in temporal_results.iterrows():
        drug = row['drug']

        if drug not in cv_results:
            continue

        cv_auc = cv_results[drug].get('auc', np.nan)
        temporal_val = row.get(temporal_col, np.nan)

        rows.append({
            'drug': drug,
            'cv_auroc': cv_auc,
            temporal_col: temporal_val,
            'difference': cv_auc - temporal_val,
            'n_cv': cv_results[drug].get('n_samples', 0),
            'n_temporal_test': row.get('n_test', 0),
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        # Add summary row
        mean_row = pd.DataFrame([{
            'drug': 'MEAN',
            'cv_auroc': df['cv_auroc'].mean(),
            temporal_col: df[temporal_col].mean(),
            'difference': df['difference'].mean(),
            'n_cv': int(df['n_cv'].sum()),
            'n_temporal_test': int(df['n_temporal_test'].sum()),
        }])
        df = pd.concat([df, mean_row], ignore_index=True)

    return df

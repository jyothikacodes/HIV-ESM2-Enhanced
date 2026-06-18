"""
Enhanced HIV Drug Resistance Models - PHASE 5 Improvements

Classifier helpers and per-drug dropout scheduling. Attention model classes
are consolidated in models.py; thin aliases are kept here for compatibility.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .models import AttentionWeightedClassifier, MultiHeadAttentionPoolingClassifier

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class RegularizedAttentionWeightedClassifier(AttentionWeightedClassifier):
    """Attention-weighted classifier with dropout regularization."""

    def __init__(
        self,
        input_dim: int = 1280,
        attention_hidden_dim: int = 256,
        dropout_rate: float = 0.1,
    ):
        super().__init__(input_dim, attention_hidden_dim, dropout=dropout_rate)


class ImprovedMultiHeadAttentionPoolingClassifier(MultiHeadAttentionPoolingClassifier):
    """Multi-head attention pooling with dropout regularization."""

    def __init__(
        self,
        input_dim: int = 1280,
        attention_hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__(input_dim, attention_hidden_dim, n_heads, dropout)


def get_dropout_schedule(drug_class: str, drug: str = None) -> float:
    """
    Get per-drug dropout rate based on drug class and size.

    - NNRTI: Aggressive dropout (0.3-0.5) due to small cohort
    - PI: Moderate dropout (0.2)
    - NRTI: Light dropout (0.1)
    """
    base_rates = {
        'NNRTI': 0.4,
        'PI': 0.2,
        'NRTI': 0.1,
    }
    fine_tuning = {
        'RPV': 0.5,
        'EFV': 0.45,
        'NVP': 0.4,
        'ETR': 0.35,
    }
    if drug and drug in fine_tuning:
        return fine_tuning[drug]
    return base_rates.get(drug_class, 0.1)


def _fit_classifier_improved(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    random_state: int = 42,
    drug_class: str = None,
    drug: str = None,
):
    """Enhanced classifier fitting with support for XGBoost on ESM features."""
    params = params or {}

    if model_type == 'xgboost_esm':
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        max_depth = params.get('max_depth', 6)
        if drug_class == 'NNRTI':
            max_depth = min(max_depth, 4)

        model = xgb.XGBClassifier(
            n_estimators=params.get('n_estimators', 300),
            max_depth=max_depth,
            learning_rate=params.get('learning_rate', 0.05),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            colsample_bylevel=params.get('colsample_bylevel', 0.8),
            reg_alpha=params.get('reg_alpha', 0.1),
            reg_lambda=params.get('reg_lambda', 1.0),
            scale_pos_weight=spw,
            random_state=random_state,
            eval_metric='auc',
            n_jobs=-1,
            tree_method='hist',
        )
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False,
            )
        else:
            model.fit(X_train, y_train, verbose=False)
        return model, None

    if model_type == 'logistic':
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(X_train)
        model = LogisticRegression(
            C=params.get('C', 1.0),
            max_iter=2000,
            class_weight='balanced',
            random_state=random_state,
        )
        model.fit(x_tr, y_train)
        return model, scaler

    if model_type == 'xgboost':
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        model = xgb.XGBClassifier(
            n_estimators=params.get('n_estimators', 300),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.05),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            reg_alpha=params.get('reg_alpha', 0.1),
            reg_lambda=params.get('reg_lambda', 1.0),
            scale_pos_weight=spw,
            random_state=random_state,
            eval_metric='auc',
            n_jobs=-1,
        )
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)
        return model, None

    if model_type == 'lightgbm':
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is required for LightGBM models")
        model = lgb.LGBMClassifier(
            n_estimators=params.get('n_estimators', 300),
            max_depth=params.get('max_depth', -1),
            learning_rate=params.get('learning_rate', 0.05),
            num_leaves=params.get('num_leaves', 31),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            reg_alpha=params.get('reg_alpha', 0.1),
            reg_lambda=params.get('reg_lambda', 1.0),
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        if X_val is not None and y_val is not None:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
        else:
            model.fit(X_train, y_train)
        return model, None

    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 300),
            max_depth=params.get('max_depth', 10),
            min_samples_leaf=params.get('min_samples_leaf', 2),
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model, None

    raise ValueError(f"Unknown model type: {model_type}")


def _predict_classifier_improved(
    model_type: str,
    model,
    scaler,
    X: np.ndarray,
) -> np.ndarray:
    """Enhanced prediction with support for XGBoost on ESM."""
    if model_type == 'logistic':
        return model.predict_proba(scaler.transform(X))[:, 1]
    if model_type in ('xgboost', 'xgboost_esm'):
        return model.predict_proba(X)[:, 1]
    return model.predict_proba(X)[:, 1]

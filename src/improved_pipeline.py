"""
Improved HIV drug resistance prediction pipeline.

Implements pooling comparisons, multi-pooling fusion, per-drug Optuna tuning,
weighted ensemble learning, SHAP feature selection, nested/repeated CV,
rare-mutation ablations, ternary vs binary evaluation, calibration comparison,
and publication-level reporting.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from .calibration import calibrate_predictions, evaluate_calibration
from .evaluation import bootstrap_auc, compute_calibration_metrics, delong_test
from .feature_engineering import (
    create_binary_mutation_encoding,
    max_pooling,
    mean_max_pooling,
    mean_pooling,
)
from .models import (
    EmbeddingDataset,
    collate_embeddings,
    train_attention_model,
    train_multihead_attention_model,
)
from .rare_mutations import (
    compute_drm_position_features,
    compute_mutation_frequencies,
    compute_rare_mutation_summary_features,
    compute_rare_mutation_weights,
    normalize_weights_for_attention,
)
from .subtype_analysis import create_temporal_split
from .ternary_classification import per_drug_ternary_training
from .visualization import plot_calibration_curve, plot_roc_curves

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from torch.utils.data import DataLoader


POOLING_METHODS = ('mean', 'max', 'mean_max', 'attention', 'fusion')
FEATURE_MODALITIES = (
    'esm_only',
    'mutation_only',
    'esm_mutation',
    'esm_rare_weighted',
)
CALIBRATION_METHODS = ('raw', 'platt', 'isotonic', 'temperature')
ENSEMBLE_MODELS = ('logistic', 'xgboost', 'lightgbm', 'rf')

# Reference benchmarks from HIV-ESM-2 paper and current pipeline (for summary table)
def load_cohort_for_improved_pipeline(
    data_dir: Union[str, Path],
    use_full_cohort: bool = True,
    subset_size: int = 250,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Load per-residue embeddings and phenotypes for the improved pipeline.

    Uses full cohort when per-residue `.npy` files exist; otherwise falls back
    to subsampled extraction via ``run_experiments.select_and_extract_subsampled_data``.
    """
    from pathlib import Path as _Path

    from .data_processing import load_unified_data
    from .feature_engineering import HIV_PROTEASE_REFERENCE, HIV_RT_REFERENCE

    data_dir = _Path(data_dir)
    processed_dir = data_dir / 'processed'
    embeddings_dir = data_dir / 'embeddings'

    references = {
        'PI': HIV_PROTEASE_REFERENCE,
        'NRTI': HIV_RT_REFERENCE,
        'NNRTI': HIV_RT_REFERENCE,
    }

    if use_full_cohort:
        unified = load_unified_data(processed_dir)
        sub_data = {}
        for drug_class, data in unified.items():
            per_res_path = embeddings_dir / f'{drug_class}_per_residue.npy'
            if not per_res_path.exists():
                continue
            per_residue = np.load(per_res_path, allow_pickle=True)
            per_residue = [np.array(x, dtype=np.float32) for x in per_residue]
            sub_data[drug_class] = {
                'sequences': data['sequences'],
                'seq_ids': data['seq_ids'],
                'phenotypes': data['phenotypes'],
                'drugs': data['drugs'],
                'per_residue': per_residue,
                'reference': references[drug_class],
            }
        if sub_data:
            return sub_data

    try:
        from scripts.run_experiments import select_and_extract_subsampled_data
    except ImportError:
        from run_experiments import select_and_extract_subsampled_data
    return select_and_extract_subsampled_data(data_dir, subset_size, seed)


def build_improved_pipeline_config(
    seed: int = 42,
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_repeats: int = 2,
    optuna_trials: int = 30,
    use_optuna: bool = True,
) -> Dict[str, Any]:
    """Build the config dict consumed by ``run_publication_evaluation``."""
    return {
        'random_state': seed,
        'outer_splits': outer_splits,
        'inner_splits': inner_splits,
        'n_repeats': n_repeats,
        'optuna_trials': optuna_trials,
        'use_optuna': use_optuna,
    }


BENCHMARK_REFERENCE = {
    'hiv_esm2_paper': {
        'mean_test_auc': 0.955,
        'mean_auc_drop': 0.042,
        'calibration': 'Partial',
        'interpretability': 'Attention + DRM',
        'clinical_utility': 'Binary resistance',
    },
    'current_pipeline': {
        'mean_test_auc': 0.9487,
        'mean_auc_drop': 0.0508,
        'calibration': 'Platt/isotonic (partial)',
        'interpretability': 'SHAP + DRM',
        'clinical_utility': 'Binary only (default path)',
    },
}


def _effective_splits(y: np.ndarray, n_splits: int) -> int:
    counts = np.bincount(y.astype(int))
    if len(counts) < 2 or counts.min() < 2:
        return 0
    return max(2, min(n_splits, int(counts.min())))


def pool_sequences(
    per_residue_list: List[np.ndarray],
    method: str = 'mean',
) -> np.ndarray:
    """Convert per-residue embeddings to a fixed-size feature matrix."""
    if method == 'mean':
        return np.array([mean_pooling(e) for e in per_residue_list], dtype=np.float32)
    if method == 'max':
        return np.array([max_pooling(e) for e in per_residue_list], dtype=np.float32)
    if method == 'mean_max':
        return np.array([mean_max_pooling(e) for e in per_residue_list], dtype=np.float32)
    raise ValueError(f"Unknown pooling method: {method}")


def extract_attention_pooled_vectors(
    model,
    embeddings_list: List[np.ndarray],
    rare_weights_list: Optional[List[np.ndarray]] = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract learned attention-pooled vectors from a trained model."""
    model.eval()
    device = next(model.parameters()).device
    dataset = EmbeddingDataset(
        embeddings_list,
        np.zeros(len(embeddings_list)),
        rare_mutation_weights_list=rare_weights_list,
    )
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_embeddings)
    pooled_vectors = []

    with torch.no_grad():
        for batch_data in loader:
            if rare_weights_list is not None and len(batch_data) == 4:
                batch_emb, _, batch_mask, batch_rw = batch_data
                batch_rw = batch_rw.to(device)
            else:
                batch_emb, _, batch_mask = batch_data[:3]
                batch_rw = None
            batch_emb = batch_emb.to(device)
            batch_mask = batch_mask.to(device)
            if hasattr(model, 'extract_pooled_vectors'):
                pooled = model.extract_pooled_vectors(
                    batch_emb,
                    mask=batch_mask,
                    rare_mutation_weights=batch_rw,
                )
            else:
                scores = model.attention(batch_emb).squeeze(-1)
                scores = scores.masked_fill(batch_mask == 0, -1e9)
                weights = torch.softmax(scores, dim=1)
                if batch_rw is not None:
                    weights = weights * batch_rw
                    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
                pooled = torch.bmm(weights.unsqueeze(1), batch_emb).squeeze(1)
            pooled_vectors.append(pooled.cpu().numpy())

    return np.vstack(pooled_vectors)


def cv_attention_predictions(
    per_residue_list: List[np.ndarray],
    y: np.ndarray,
    rare_weights_list: Optional[List[np.ndarray]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    epochs: int = 20,
    use_multihead: bool = False,
    n_heads: int = 4,
    attention_dropout: float = 0.1,
    return_pooled: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Out-of-fold attention model predictions (and optional pooled features)."""
    splits = _effective_splits(y, n_splits)
    if splits < 2:
        nan = np.full(len(y), np.nan)
        if return_pooled:
            dim = per_residue_list[0].shape[1]
            return nan, np.full((len(y), dim), np.nan)
        return nan

    input_dim = int(per_residue_list[0].shape[1])
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    y_pred = np.zeros(len(y))
    pooled_features = np.zeros((len(y), input_dim), dtype=np.float32)

    for train_idx, val_idx in cv.split(np.zeros(len(y)), y):
        x_train = [per_residue_list[i] for i in train_idx]
        x_val = [per_residue_list[i] for i in val_idx]
        y_train = y[train_idx]
        rw_train = [rare_weights_list[i] for i in train_idx] if rare_weights_list else None
        rw_val = [rare_weights_list[i] for i in val_idx] if rare_weights_list else None

        if use_multihead:
            model = train_multihead_attention_model(
                x_train,
                y_train,
                val_embeddings_list=x_val,
                val_labels=y[val_idx],
                rare_mutation_weights_list=rw_train,
                val_rare_mutation_weights_list=rw_val,
                input_dim=input_dim,
                attention_dim=input_dim if input_dim < 256 else 256,
                n_heads=n_heads,
                dropout=attention_dropout,
                epochs=epochs,
                verbose=False,
            )
        else:
            model = train_attention_model(
                x_train,
                y_train,
                val_embeddings_list=x_val,
                val_labels=y[val_idx],
                rare_mutation_weights_list=rw_train,
                val_rare_mutation_weights_list=rw_val,
                input_dim=input_dim,
                epochs=epochs,
                verbose=False,
            )
        model.eval()
        val_dataset = EmbeddingDataset(x_val, y[val_idx], rare_mutation_weights_list=rw_val)
        val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_embeddings)
        fold_preds = []
        with torch.no_grad():
            device = next(model.parameters()).device
            for val_batch in val_loader:
                if rw_val is not None and len(val_batch) == 4:
                    v_emb, _, v_mask, v_rw = val_batch
                    v_rw = v_rw.to(device)
                else:
                    v_emb, _, v_mask = val_batch[:3]
                    v_rw = None
                v_emb = v_emb.to(device)
                v_mask = v_mask.to(device)
                logits, _ = model(v_emb, v_mask, rare_mutation_weights=v_rw)
                fold_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())

        y_pred[val_idx] = fold_preds
        if return_pooled:
            pooled_features[val_idx] = extract_attention_pooled_vectors(model, x_val, rw_val)

    if return_pooled:
        return y_pred, pooled_features
    return y_pred


def build_fusion_features(
    per_residue_list: List[np.ndarray],
    sequences: List[str],
    reference: str,
    attention_pooled: np.ndarray,
    frequencies: Optional[np.ndarray] = None,
    rare_tau: float = 0.05,
) -> np.ndarray:
    """
    Concatenate mean, max, attention-pooled, and rare-mutation summary features.

    Rare-mutation features (4 dims per sequence): count, fraction, summed and mean
    inverse frequency — see ``compute_rare_mutation_summary_features``.
    """
    if attention_pooled is None:
        raise ValueError(
            "attention_pooled is required; compute with cv_attention_predictions(..., return_pooled=True)"
        )
    if len(sequences) != len(per_residue_list):
        raise ValueError("sequences and per_residue_list must have the same length")

    mean_feat = pool_sequences(per_residue_list, 'mean')
    max_feat = pool_sequences(per_residue_list, 'max')
    rare_feat = compute_rare_mutation_summary_features(
        sequences,
        reference,
        tau=rare_tau,
        frequencies=frequencies,
    )
    parts = [mean_feat, max_feat, attention_pooled, rare_feat]
    if drm_features is not None:
        parts.append(drm_features.astype(np.float32))
    return np.hstack(parts).astype(np.float32)


def scale_features(X: np.ndarray) -> np.ndarray:
    """Standard-scale feature matrix for sklearn / tree models."""
    return StandardScaler().fit_transform(X)


def _fit_classifier(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    random_state: int = 42,
):
    params = params or {}
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


def _predict_classifier(model_type: str, model, scaler, X: np.ndarray) -> np.ndarray:
    if model_type == 'logistic':
        return model.predict_proba(scaler.transform(X))[:, 1]
    return model.predict_proba(X)[:, 1]


def optuna_tune_classifier(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'xgboost',
    n_trials: int = 30,
    n_inner_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Tune hyperparameters with Optuna on inner stratified CV."""
    if not HAS_OPTUNA:
        warnings.warn("Optuna not installed; returning default parameters")
        return {'best_params': {}, 'best_auc': np.nan}

    splits = _effective_splits(y, n_inner_splits)
    if splits < 2:
        return {'best_params': {}, 'best_auc': np.nan}

    def objective(trial: optuna.Trial) -> float:
        if model_type == 'logistic':
            params = {'C': trial.suggest_float('C', 1e-3, 10.0, log=True)}
        elif model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            }
        elif model_type == 'lightgbm' and HAS_LIGHTGBM:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            }
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }

        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
        fold_aucs = []
        for train_idx, val_idx in cv.split(X, y):
            model, scaler = _fit_classifier(
                model_type, X[train_idx], y[train_idx], params=params, random_state=random_state
            )
            preds = _predict_classifier(model_type, model, scaler, X[val_idx])
            fold_aucs.append(roc_auc_score(y[val_idx], preds))
        return float(np.mean(fold_aucs))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {'best_params': study.best_params, 'best_auc': study.best_value}


def ensemble_soft_vote_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_types: Optional[List[str]] = None,
    params_per_model: Optional[Dict[str, Dict]] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Weighted soft-vote ensemble using OOF predictions to learn weights."""
    model_types = list(model_types or ENSEMBLE_MODELS)
    if not HAS_LIGHTGBM and 'lightgbm' in model_types:
        model_types.remove('lightgbm')

    splits = _effective_splits(y, n_splits)
    if splits < 2:
        return np.full(len(y), np.nan), {}

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    oof_preds = {m: np.zeros(len(y)) for m in model_types}

    for train_idx, val_idx in cv.split(X, y):
        for model_type in model_types:
            params = (params_per_model or {}).get(model_type, {})
            model, scaler = _fit_classifier(
                model_type,
                X[train_idx],
                y[train_idx],
                params=params,
                X_val=X[val_idx],
                y_val=y[val_idx],
                random_state=random_state,
            )
            oof_preds[model_type][val_idx] = _predict_classifier(
                model_type, model, scaler, X[val_idx]
            )

    # AUC-weighted soft voting (normalized)
    weights = {}
    for model_type, preds in oof_preds.items():
        weights[model_type] = max(roc_auc_score(y, preds), 0.01)
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    ensemble_pred = np.zeros(len(y))
    for model_type, preds in oof_preds.items():
        ensemble_pred += weights[model_type] * preds

    return ensemble_pred, weights


def stacked_ensemble_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_types: Optional[List[str]] = None,
    params_per_model: Optional[Dict[str, Dict]] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Stacked meta-learner ensemble using OOF base predictions."""
    model_types = list(model_types or ENSEMBLE_MODELS)
    if not HAS_LIGHTGBM and 'lightgbm' in model_types:
        model_types.remove('lightgbm')

    splits = _effective_splits(y, n_splits)
    if splits < 2:
        return np.full(len(y), np.nan), {'meta_weights': {}, 'oof_preds': {}}

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    oof_preds = {m: np.zeros(len(y)) for m in model_types}

    for train_idx, val_idx in cv.split(X, y):
        for model_type in model_types:
            params = (params_per_model or {}).get(model_type, {})
            model, scaler = _fit_classifier(
                model_type,
                X[train_idx],
                y[train_idx],
                params=params,
                X_val=X[val_idx],
                y_val=y[val_idx],
                random_state=random_state,
            )
            oof_preds[model_type][val_idx] = _predict_classifier(
                model_type, model, scaler, X[val_idx]
            )

    stacked_features = np.vstack([oof_preds[m] for m in model_types]).T
    meta_scaler = StandardScaler()
    stacked_scaled = meta_scaler.fit_transform(stacked_features)
    meta_model = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        random_state=random_state,
    )
    meta_model.fit(stacked_scaled, y)
    ensemble_pred = meta_model.predict_proba(stacked_scaled)[:, 1]

    return ensemble_pred, {
        'meta_model': meta_model,
        'meta_scaler': meta_scaler,
        'oof_preds': oof_preds,
        'model_types': model_types,
    }


def shap_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    top_fraction: float = 0.8,
    max_background: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select embedding dimensions by mean |SHAP| importance.

    Returns:
        feature_mask (bool), selected_indices
    """
    if not HAS_SHAP:
        mask = np.ones(X.shape[1], dtype=bool)
        return mask, np.arange(X.shape[1])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=random_state)
    model.fit(x_scaled, y)

    rng = np.random.RandomState(random_state)
    bg_idx = rng.choice(len(x_scaled), size=min(max_background, len(x_scaled)), replace=False)
    explainer = shap.LinearExplainer(model, x_scaled[bg_idx])
    shap_values = explainer.shap_values(x_scaled)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    importance = np.mean(np.abs(shap_values), axis=0)
    n_keep = max(1, int(len(importance) * top_fraction))
    selected = np.argsort(importance)[-n_keep:]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[selected] = True
    return mask, selected


def nested_cv_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'logistic',
    params: Optional[Dict] = None,
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_repeats: int = 1,
    tune_with_optuna: bool = False,
    optuna_trials: int = 20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Nested stratified CV with optional inner Optuna tuning.

    Reports train / validation (OOF) / test AUC, AUC drop, and bootstrap CIs.
    """
    outer_k = _effective_splits(y, outer_splits)
    if outer_k < 2:
        return {'skipped': True, 'reason': 'insufficient class counts'}

    train_aucs, val_aucs, test_aucs = [], [], []
    all_y_true, all_y_pred = [], []
    best_params = params or {}

    for repeat in range(n_repeats):
        seed = random_state + repeat
        outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=seed)

        for outer_train_idx, outer_test_idx in outer_cv.split(X, y):
            x_train, x_test = X[outer_train_idx], X[outer_test_idx]
            y_train, y_test = y[outer_train_idx], y[outer_test_idx]

            best_params = params or {}
            if tune_with_optuna and HAS_OPTUNA:
                tune_res = optuna_tune_classifier(
                    x_train,
                    y_train,
                    model_type=model_type,
                    n_trials=optuna_trials,
                    n_inner_splits=inner_splits,
                    random_state=seed,
                )
                best_params = tune_res.get('best_params', best_params)

            inner_k = _effective_splits(y_train, inner_splits)
            inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=seed)
            oof = np.zeros(len(y_train))

            for inner_train_idx, inner_val_idx in inner_cv.split(x_train, y_train):
                model, scaler = _fit_classifier(
                    model_type,
                    x_train[inner_train_idx],
                    y_train[inner_train_idx],
                    params=best_params,
                    X_val=x_train[inner_val_idx],
                    y_val=y_train[inner_val_idx],
                    random_state=seed,
                )
                oof[inner_val_idx] = _predict_classifier(
                    model_type, model, scaler, x_train[inner_val_idx]
                )

                train_pred = _predict_classifier(
                    model_type, model, scaler, x_train[inner_train_idx]
                )
                if len(np.unique(y_train[inner_train_idx])) > 1:
                    train_aucs.append(roc_auc_score(y_train[inner_train_idx], train_pred))

            if len(np.unique(y_train)) > 1:
                val_aucs.append(roc_auc_score(y_train, oof))

            final_model, final_scaler = _fit_classifier(
                model_type, x_train, y_train, params=best_params, random_state=seed
            )
            test_pred = _predict_classifier(model_type, final_model, final_scaler, x_test)
            if len(np.unique(y_test)) > 1:
                test_aucs.append(roc_auc_score(y_test, test_pred))

            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(test_pred.tolist())

    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)
    _, ci_low, ci_high = bootstrap_auc(y_true_arr, y_pred_arr, random_state=random_state)

    mean_train = float(np.mean(train_aucs)) if train_aucs else np.nan
    mean_val = float(np.mean(val_aucs)) if val_aucs else np.nan
    mean_test = float(np.mean(test_aucs)) if test_aucs else np.nan
    auc_drop = mean_val - mean_test if not np.isnan(mean_val) and not np.isnan(mean_test) else np.nan

    return {
        'train_auc': mean_train,
        'val_auc': mean_val,
        'test_auc': mean_test,
        'auc_drop': auc_drop,
        'test_auc_ci_low': ci_low,
        'test_auc_ci_high': ci_high,
        'y_true': y_true_arr,
        'y_pred': y_pred_arr,
        'best_params': best_params if tune_with_optuna else params,
    }


def compare_pooling_strategies(
    per_residue_list: List[np.ndarray],
    sequences: List[str],
    reference: str,
    y: np.ndarray,
    rare_weights_list: Optional[List[np.ndarray]] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Benchmark mean, max, mean_max, attention, and fusion pooling."""
    rows = []
    splits = _effective_splits(y, n_splits)
    if splits < 2:
        return pd.DataFrame()

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)

    for method in ('mean', 'max', 'mean_max'):
        X = pool_sequences(per_residue_list, method)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=random_state)
        preds = cross_val_predict(model, x_scaled, y, cv=cv, method='predict_proba')[:, 1]
        rows.append({
            'pooling': method,
            'auc': roc_auc_score(y, preds),
            'n_features': X.shape[1],
        })

    attn_preds = cv_attention_predictions(
        per_residue_list, y, rare_weights_list=rare_weights_list, n_splits=splits, random_state=random_state
    )
    rows.append({
        'pooling': 'attention',
        'auc': roc_auc_score(y, attn_preds),
        'n_features': per_residue_list[0].shape[1],
    })

    _, attn_pooled = cv_attention_predictions(
        per_residue_list, y, rare_weights_list=rare_weights_list,
        n_splits=splits, random_state=random_state, return_pooled=True,
    )
    fusion_x = build_fusion_features(
        per_residue_list,
        sequences,
        reference,
        attention_pooled=attn_pooled,
        frequencies=compute_mutation_frequencies(sequences, reference),
    )
    scaler = StandardScaler()
    fusion_scaled = scaler.fit_transform(fusion_x)
    model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=random_state)
    fusion_preds = cross_val_predict(model, fusion_scaled, y, cv=cv, method='predict_proba')[:, 1]
    rows.append({
        'pooling': 'fusion',
        'auc': roc_auc_score(y, fusion_preds),
        'n_features': fusion_x.shape[1],
    })

    return pd.DataFrame(rows)


def compare_feature_modalities(
    per_residue_list: List[np.ndarray],
    sequences: List[str],
    reference: str,
    y: np.ndarray,
    rare_weights_list: Optional[List[np.ndarray]] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare ESM-only, mutation-only, combined, and rare-weighted attention."""
    splits = _effective_splits(y, n_splits)
    if splits < 2:
        return pd.DataFrame()

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    mutation_x = create_binary_mutation_encoding(sequences, reference).astype(np.float32)
    esm_x = pool_sequences(per_residue_list, 'mean')
    combined_x = np.hstack([esm_x, mutation_x])

    rows = []
    configs = {
        'esm_only': esm_x,
        'mutation_only': mutation_x,
        'esm_mutation': combined_x,
    }

    for name, features in configs.items():
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(features)
        model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=random_state)
        preds = cross_val_predict(model, x_scaled, y, cv=cv, method='predict_proba')[:, 1]
        rows.append({'modality': name, 'auc': roc_auc_score(y, preds)})

    rare_preds = cv_attention_predictions(
        per_residue_list, y, rare_weights_list=rare_weights_list, n_splits=splits, random_state=random_state
    )
    rows.append({'modality': 'esm_rare_weighted', 'auc': roc_auc_score(y, rare_preds)})
    return pd.DataFrame(rows)


def compare_calibration_methods_oof(
    y: np.ndarray,
    y_pred_raw: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """OOF calibration comparison: raw, platt, isotonic, temperature."""
    splits = _effective_splits(y, n_splits)
    if splits < 2:
        return pd.DataFrame()

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    rows = []

    for method in CALIBRATION_METHODS:
        if method == 'raw':
            calibrated = y_pred_raw.copy()
        else:
            calibrated = np.zeros_like(y_pred_raw)
            for train_idx, val_idx in cv.split(y_pred_raw, y):
                calibrated[val_idx] = calibrate_predictions(
                    y[train_idx], y_pred_raw[train_idx], y_pred_raw[val_idx], method=method
                )

        metrics = compute_calibration_metrics(y, calibrated)
        rows.append({
            'method': method,
            'ece': metrics['ece'],
            'mce': metrics['mce'],
            'brier_score': metrics['brier_score'],
            'auc': roc_auc_score(y, calibrated),
        })

    return pd.DataFrame(rows)


def apply_oof_calibration(
    y: np.ndarray,
    y_pred_raw: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    OOF-calibrate predictions with Platt, isotonic, and temperature scaling.

    Selects the calibrated method with the lowest Brier score and returns
    calibrated probabilities plus per-method metrics.
    """
    calib_df = compare_calibration_methods_oof(
        y, y_pred_raw, n_splits=n_splits, random_state=random_state
    )
    if calib_df.empty:
        return {
            'calibration_comparison': calib_df,
            'best_method': 'raw',
            'y_pred_calibrated': y_pred_raw.copy(),
            'calibrated_ece': np.nan,
            'calibrated_brier': np.nan,
            'calibrated_auc': np.nan,
        }

    calibrated_methods = calib_df[calib_df['method'] != 'raw']
    if calibrated_methods.empty:
        best_method = 'raw'
        best_row = calib_df[calib_df['method'] == 'raw'].iloc[0]
        y_cal = y_pred_raw.copy()
    else:
        best_row = calibrated_methods.sort_values('brier_score').iloc[0]
        best_method = str(best_row['method'])
        splits = _effective_splits(y, n_splits)
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
        y_cal = np.zeros_like(y_pred_raw)
        for train_idx, val_idx in cv.split(y_pred_raw, y):
            y_cal[val_idx] = calibrate_predictions(
                y[train_idx],
                y_pred_raw[train_idx],
                y_pred_raw[val_idx],
                method=best_method,
            )

    return {
        'calibration_comparison': calib_df,
        'best_method': best_method,
        'y_pred_calibrated': y_cal,
        'calibrated_ece': float(best_row['ece']),
        'calibrated_brier': float(best_row['brier_score']),
        'calibrated_auc': float(best_row['auc']),
    }


def evaluate_temporal_drug(
    fusion_x: np.ndarray,
    y: np.ndarray,
    drug_pheno: pd.DataFrame,
    cv_auc: float,
    model_type: str = 'xgboost',
    params: Optional[Dict] = None,
    seq_id_col: Optional[str] = None,
    cutoff_quantile: float = 0.8,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Temporal holdout evaluation on fusion features for a single drug."""
    seq_col = seq_id_col
    if seq_col is None:
        for candidate in ('seq_id', 'SeqID', 'SeqId'):
            if candidate in drug_pheno.columns:
                seq_col = candidate
                break

    if seq_col is None:
        return {
            'temporal_auc': np.nan,
            'temporal_auc_drop': np.nan,
            'skipped': True,
            'reason': 'missing sequence id column',
        }

    pheno_reset = drug_pheno.reset_index(drop=True)
    try:
        train_idx, test_idx = create_temporal_split(
            pheno_reset, seq_id_col=seq_col, cutoff_quantile=cutoff_quantile
        )
    except Exception as exc:
        return {
            'temporal_auc': np.nan,
            'temporal_auc_drop': np.nan,
            'skipped': True,
            'reason': str(exc),
        }

    if len(train_idx) < 10 or len(test_idx) < 5:
        return {
            'temporal_auc': np.nan,
            'temporal_auc_drop': np.nan,
            'skipped': True,
            'reason': 'insufficient temporal split sizes',
        }

    y_train, y_test = y[train_idx], y[test_idx]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {
            'temporal_auc': np.nan,
            'temporal_auc_drop': np.nan,
            'skipped': True,
            'reason': 'single-class temporal split',
        }

    x_train, x_test = fusion_x[train_idx], fusion_x[test_idx]
    model, scaler = _fit_classifier(
        model_type,
        x_train,
        y_train,
        params=params or {},
        random_state=random_state,
    )
    y_pred = _predict_classifier(model_type, model, scaler, x_test)
    temporal_auc = float(roc_auc_score(y_test, y_pred))
    temporal_drop = float(cv_auc - temporal_auc) if not np.isnan(cv_auc) else np.nan

    return {
        'temporal_auc': temporal_auc,
        'temporal_auc_drop': temporal_drop,
        'n_train': int(len(train_idx)),
        'n_test': int(len(test_idx)),
        'skipped': False,
    }


def evaluate_drug_improved(
    drug: str,
    drug_class: str,
    per_residue_list: List[np.ndarray],
    sequences: List[str],
    phenotypes: pd.DataFrame,
    reference: str,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run full improved evaluation for a single drug."""
    config = config or {}
    seed = config.get('random_state', 42)
    fc_col = f"{drug}_FC" if f"{drug}_FC" in phenotypes.columns else drug

    y_vals = phenotypes[fc_col].values.astype(float)
    valid_mask = ~np.isnan(y_vals)
    y = (y_vals[valid_mask] >= 2.5).astype(int)

    x_list = [per_residue_list[i] for i in range(len(per_residue_list)) if valid_mask[i]]
    seq_valid = [sequences[i] for i in range(len(sequences)) if valid_mask[i]]

    if len(np.unique(y)) < 2 or min(y.sum(), len(y) - y.sum()) < 3:
        return {'drug': drug, 'skipped': True, 'reason': 'insufficient samples'}

    freqs = compute_mutation_frequencies(seq_valid, reference)
    raw_weights = compute_rare_mutation_weights(seq_valid, reference, frequencies=freqs)
    norm_weights = [normalize_weights_for_attention(w, method='softmax') for w in raw_weights]

    pooling_df = compare_pooling_strategies(
        x_list, seq_valid, reference, y, norm_weights, random_state=seed
    )
    modality_df = compare_feature_modalities(x_list, seq_valid, reference, y, norm_weights, random_state=seed)

    use_multihead = config.get('use_multihead_attention', False)
    n_heads = config.get('attention_n_heads', 4)
    attention_dropout = config.get('attention_dropout', 0.1)
    _, attn_pooled = cv_attention_predictions(
        x_list,
        y,
        rare_weights_list=norm_weights,
        random_state=seed,
        use_multihead=use_multihead,
        n_heads=n_heads,
        attention_dropout=attention_dropout,
        return_pooled=True,
    )
    drm_features = compute_drm_position_features(
        seq_valid,
        reference,
        drug_class,
        frequencies=freqs,
    )
    fusion_x = build_fusion_features(
        x_list,
        seq_valid,
        reference,
        attention_pooled=attn_pooled,
        frequencies=freqs,
        drm_features=drm_features,
    )
    fusion_scaled = scale_features(fusion_x)

    use_optuna = config.get('use_optuna', True) and HAS_OPTUNA
    nested = nested_cv_evaluation(
        fusion_x,
        y,
        model_type='xgboost',
        outer_splits=config.get('outer_splits', 5),
        inner_splits=config.get('inner_splits', 3),
        n_repeats=config.get('n_repeats', 1),
        tune_with_optuna=use_optuna,
        optuna_trials=config.get('optuna_trials', 20),
        random_state=seed,
    )

    baseline_nested = nested_cv_evaluation(
        pool_sequences(x_list, 'mean'),
        y,
        model_type='logistic',
        outer_splits=config.get('outer_splits', 5),
        inner_splits=config.get('inner_splits', 3),
        n_repeats=config.get('n_repeats', 1),
        tune_with_optuna=False,
        random_state=seed,
    )

    shap_mask, _ = shap_feature_selection(fusion_x, y, top_fraction=0.8, random_state=seed)
    nested_shap = nested_cv_evaluation(
        fusion_x[:, shap_mask],
        y,
        model_type='xgboost',
        params=nested.get('best_params'),
        outer_splits=config.get('outer_splits', 5),
        n_repeats=1,
        tune_with_optuna=False,
        random_state=seed,
    )

    xgb_params = nested.get('best_params') or {}
    if config.get('use_stacked_ensemble', True):
        ensemble_pred, ensemble_meta = stacked_ensemble_cv(
            fusion_scaled,
            y,
            params_per_model={'xgboost': xgb_params},
            random_state=seed,
        )
        ensemble_weights = {'stacked_meta': 1.0}
    else:
        ensemble_pred, ensemble_weights = ensemble_soft_vote_cv(
            fusion_scaled,
            y,
            params_per_model={'xgboost': xgb_params},
            random_state=seed,
        )

    drug_pheno = phenotypes.iloc[valid_mask].copy()
    calibration_result = apply_oof_calibration(y, ensemble_pred, random_state=seed)

    temporal = evaluate_temporal_drug(
        fusion_x,
        y,
        drug_pheno,
        cv_auc=nested.get('val_auc', np.nan),
        model_type='xgboost',
        params=xgb_params,
        random_state=seed,
    )

    ternary_res = per_drug_ternary_training(
        pool_sequences(x_list, 'mean'),
        drug_pheno,
        [drug],
        model_type='logistic',
        random_state=seed,
    )
    ternary = ternary_res.get(drug, {})

    return {
        'drug': drug,
        'drug_class': drug_class,
        'n_samples': len(y),
        'pooling_comparison': pooling_df,
        'modality_comparison': modality_df,
        'nested_cv': nested,
        'baseline_nested_cv': baseline_nested,
        'nested_cv_shap_selected': nested_shap,
        'ensemble_weights': ensemble_weights,
        'ensemble_auc': roc_auc_score(y, ensemble_pred),
        'ternary': ternary,
        'calibration': calibration_result['calibration_comparison'],
        'calibration_result': calibration_result,
        'temporal': temporal,
        'y_true': y,
        'y_pred_ensemble': ensemble_pred,
        'y_pred_calibrated': calibration_result['y_pred_calibrated'],
        'optuna_params': nested.get('best_params', {}),
        'skipped': False,
    }


def generate_summary_comparison_table(
    improved_mean_auc: float,
    improved_mean_auc_drop: float,
    improved_calibration_ece: float,
) -> pd.DataFrame:
    """Compare original paper, current pipeline, and improved pipeline."""
    rows = [
        {
            'pipeline': 'Original HIV-ESM-2 Paper',
            'mean_test_auc': BENCHMARK_REFERENCE['hiv_esm2_paper']['mean_test_auc'],
            'mean_auc_drop': BENCHMARK_REFERENCE['hiv_esm2_paper']['mean_auc_drop'],
            'generalization': 'Good',
            'calibration': BENCHMARK_REFERENCE['hiv_esm2_paper']['calibration'],
            'interpretability': BENCHMARK_REFERENCE['hiv_esm2_paper']['interpretability'],
            'clinical_utility': BENCHMARK_REFERENCE['hiv_esm2_paper']['clinical_utility'],
            'improves_mean_auc': np.nan,
            'improves_generalization': np.nan,
            'improves_calibration': np.nan,
        },
        {
            'pipeline': 'Current Pipeline',
            'mean_test_auc': BENCHMARK_REFERENCE['current_pipeline']['mean_test_auc'],
            'mean_auc_drop': BENCHMARK_REFERENCE['current_pipeline']['mean_auc_drop'],
            'generalization': 'Moderate',
            'calibration': BENCHMARK_REFERENCE['current_pipeline']['calibration'],
            'interpretability': BENCHMARK_REFERENCE['current_pipeline']['interpretability'],
            'clinical_utility': BENCHMARK_REFERENCE['current_pipeline']['clinical_utility'],
            'improves_mean_auc': np.nan,
            'improves_generalization': np.nan,
            'improves_calibration': np.nan,
        },
        {
            'pipeline': 'Improved Pipeline',
            'mean_test_auc': improved_mean_auc,
            'mean_auc_drop': improved_mean_auc_drop,
            'generalization': 'Strong' if improved_mean_auc_drop < 0.034 else 'Moderate',
            'calibration': f'ECE={improved_calibration_ece:.4f}',
            'interpretability': 'SHAP + attention + rare mutations',
            'clinical_utility': 'Binary + ternary + calibrated probabilities',
            'improves_mean_auc': improved_mean_auc > BENCHMARK_REFERENCE['current_pipeline']['mean_test_auc'],
            'improves_generalization': improved_mean_auc_drop < BENCHMARK_REFERENCE['current_pipeline']['mean_auc_drop'],
            'improves_calibration': improved_calibration_ece < 0.05,
        },
    ]
    return pd.DataFrame(rows)


def run_publication_evaluation(
    sub_data: Dict,
    results_dir: Union[str, Path],
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run full improved pipeline across all drugs and write publication outputs.
    """
    results_dir = Path(results_dir)
    improved_dir = results_dir / 'improved_pipeline'
    fig_dir = improved_dir / 'figures'
    improved_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    config = config or {}
    drug_rows = []
    pooling_rows = []
    modality_rows = []
    calibration_rows = []
    ece_rows = []
    brier_rows = []
    temporal_rows = []
    baseline_comparison_rows = []
    ternary_rows = []
    nested_rows = []
    hyperparameter_rows = []
    all_results_for_plots = {}
    summary = pd.DataFrame()
    saved_summary_calibration_curve = False

    for drug_class, data in sub_data.items():
        sequences = data['sequences']
        phenotypes = data['phenotypes']
        drugs = data['drugs']
        per_residue = data['per_residue']
        reference = data['reference']

        renamed = phenotypes.rename(columns={d: f"{d}_FC" for d in drugs if d in phenotypes.columns})

        for drug in drugs:
            print(f"\n[Improved Pipeline] {drug_class} / {drug}")
            res = evaluate_drug_improved(
                drug, drug_class, per_residue, sequences, renamed, reference, config
            )
            if res.get('skipped'):
                continue

            nested = res['nested_cv']
            baseline = res['baseline_nested_cv']
            calib = res['calibration_result']
            temporal = res.get('temporal', {})

            drug_rows.append({
                'drug': drug,
                'drug_class': drug_class,
                'n_samples': res['n_samples'],
                'train_auc': nested['train_auc'],
                'val_auc': nested['val_auc'],
                'test_auc': nested['test_auc'],
                'auc_drop': nested['auc_drop'],
                'test_auc_ci_low': nested['test_auc_ci_low'],
                'test_auc_ci_high': nested['test_auc_ci_high'],
                'baseline_test_auc': baseline.get('test_auc', np.nan),
                'baseline_auc_drop': baseline.get('auc_drop', np.nan),
                'ensemble_auc': res['ensemble_auc'],
                'calibrated_auc': calib.get('calibrated_auc', np.nan),
                'calibrated_ece': calib.get('calibrated_ece', np.nan),
                'calibrated_brier': calib.get('calibrated_brier', np.nan),
                'best_calibration_method': calib.get('best_method', 'raw'),
                'temporal_auc': temporal.get('temporal_auc', np.nan),
                'temporal_auc_drop': temporal.get('temporal_auc_drop', np.nan),
                'shap_selected_test_auc': res['nested_cv_shap_selected']['test_auc'],
            })

            baseline_comparison_rows.append({
                'drug': drug,
                'drug_class': drug_class,
                'improved_test_auc': nested['test_auc'],
                'baseline_test_auc': baseline.get('test_auc', np.nan),
                'delta_auc': nested['test_auc'] - baseline.get('test_auc', np.nan),
            })

            if res.get('optuna_params'):
                hyperparameter_rows.append({
                    'drug': drug,
                    'drug_class': drug_class,
                    **res['optuna_params'],
                })

            ece_rows.append({
                'drug': drug,
                'drug_class': drug_class,
                'method': calib.get('best_method', 'raw'),
                'ece': calib.get('calibrated_ece', np.nan),
            })
            brier_rows.append({
                'drug': drug,
                'drug_class': drug_class,
                'method': calib.get('best_method', 'raw'),
                'brier_score': calib.get('calibrated_brier', np.nan),
            })

            if not temporal.get('skipped', True):
                temporal_rows.append({
                    'drug': drug,
                    'drug_class': drug_class,
                    'cv_auc': nested.get('val_auc', np.nan),
                    'temporal_auc': temporal.get('temporal_auc', np.nan),
                    'temporal_auc_drop': temporal.get('temporal_auc_drop', np.nan),
                    'n_train': temporal.get('n_train', 0),
                    'n_test': temporal.get('n_test', 0),
                })

            for _, row in res['pooling_comparison'].iterrows():
                pooling_rows.append({'drug': drug, 'drug_class': drug_class, **row.to_dict()})
            for _, row in res['modality_comparison'].iterrows():
                modality_rows.append({'drug': drug, 'drug_class': drug_class, **row.to_dict()})
            for _, row in res['calibration'].iterrows():
                calibration_rows.append({'drug': drug, 'drug_class': drug_class, **row.to_dict()})

            ternary = res.get('ternary', {})
            if ternary and not ternary.get('skipped'):
                ternary_rows.append({
                    'drug': drug,
                    'drug_class': drug_class,
                    'auc_ovr': ternary.get('auc_ovr', np.nan),
                    'accuracy': ternary.get('accuracy', np.nan),
                    'macro_f1': ternary.get('macro_f1', np.nan),
                    'weighted_f1': f1_score(
                        ternary.get('y_true', []),
                        ternary.get('y_pred', []),
                        average='weighted',
                        zero_division=0,
                    ) if 'y_true' in ternary else np.nan,
                    'n_samples': ternary.get('n_samples', res['n_samples']),
                })

            nested_rows.append({
                'drug': drug,
                'drug_class': drug_class,
                **{k: nested[k] for k in ('train_auc', 'val_auc', 'test_auc', 'auc_drop')},
            })

            y_plot = res['y_pred_calibrated']
            all_results_for_plots[drug] = {
                'y_true': res['y_true'],
                'y_pred': y_plot,
                'auc': calib.get('calibrated_auc', res['ensemble_auc']),
            }

            # Per-drug ROC and calibration figures (calibrated OOF predictions)
            fpr, tpr, _ = roc_curve(res['y_true'], y_plot)
            y_cal = y_plot

            cal_path = fig_dir / f'calibration_{drug}.png'
            plot_calibration_curve(
                res['y_true'],
                res['y_pred_ensemble'],
                y_cal,
                title=f'{drug} Calibration ({calib.get("best_method", "raw")})',
                save_path=str(cal_path),
            )
            if not saved_summary_calibration_curve:
                import shutil
                shutil.copy(cal_path, fig_dir / 'calibration_curves.png')
                saved_summary_calibration_curve = True
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"AUC={calib.get('calibrated_auc', res['ensemble_auc']):.3f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title(f'ROC — {drug}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / f'roc_{drug}.png', dpi=150)
            plt.close()

            prec, rec, _ = precision_recall_curve(res['y_true'], y_plot)
            plt.figure(figsize=(6, 5))
            plt.plot(rec, prec)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'PR — {drug}')
            plt.tight_layout()
            plt.savefig(fig_dir / f'pr_{drug}.png', dpi=150)
            plt.close()

    drug_df = pd.DataFrame(drug_rows)
    if not drug_df.empty:
        drug_df.to_csv(improved_dir / 'drug_wise_performance.csv', index=False)
        drug_df.to_csv(improved_dir / 'drugwise_auc_summary.csv', index=False)

        mean_test = drug_df['test_auc'].mean()
        mean_drop = drug_df['auc_drop'].mean()
        mean_temporal = drug_df['temporal_auc'].dropna().mean()
        mean_temporal_drop = drug_df['temporal_auc_drop'].dropna().mean()
        mean_ece = drug_df['calibrated_ece'].dropna().mean()
        mean_brier = drug_df['calibrated_brier'].dropna().mean()

        baseline_df = pd.DataFrame(baseline_comparison_rows)
        if len(baseline_df) >= 5:
            valid_pairs = baseline_df.dropna(subset=['improved_test_auc', 'baseline_test_auc'])
            if len(valid_pairs) >= 5:
                stat, pval = stats.wilcoxon(
                    valid_pairs['improved_test_auc'].values,
                    valid_pairs['baseline_test_auc'].values,
                )
            else:
                stat, pval = np.nan, np.nan
        else:
            stat, pval = np.nan, np.nan

        summary = pd.DataFrame([{
            'mean_test_auc': mean_test,
            'mean_baseline_test_auc': drug_df['baseline_test_auc'].mean(),
            'mean_val_auc': drug_df['val_auc'].mean(),
            'mean_train_auc': drug_df['train_auc'].mean(),
            'mean_auc_drop': mean_drop,
            'mean_temporal_auc': mean_temporal,
            'mean_temporal_auc_drop': mean_temporal_drop,
            'mean_ensemble_auc': drug_df['ensemble_auc'].mean(),
            'mean_calibrated_ece': mean_ece,
            'mean_calibrated_brier': mean_brier,
            'wilcoxon_statistic_improved_vs_baseline': stat,
            'p_value_improved_vs_baseline': pval,
            'n_drugs': len(drug_df),
            'target_auc_met': mean_test > 0.968,
            'target_drop_met': mean_drop < 0.034,
            'target_temporal_drop_met': mean_temporal_drop < 0.034 if not np.isnan(mean_temporal_drop) else False,
        }])
        summary.to_csv(improved_dir / 'aggregate_metrics.csv', index=False)

        comparison = generate_summary_comparison_table(
            mean_test,
            mean_temporal_drop if not np.isnan(mean_temporal_drop) else mean_drop,
            mean_ece if not np.isnan(mean_ece) else 0.1,
        )
        comparison.to_csv(improved_dir / 'pipeline_comparison_summary.csv', index=False)

    pd.DataFrame(pooling_rows).to_csv(improved_dir / 'pooling_comparison.csv', index=False)
    pd.DataFrame(modality_rows).to_csv(improved_dir / 'rare_mutation_modality_comparison.csv', index=False)
    pd.DataFrame(calibration_rows).to_csv(improved_dir / 'calibration_comparison.csv', index=False)
    pd.DataFrame(ece_rows).to_csv(improved_dir / 'ece_scores.csv', index=False)
    pd.DataFrame(brier_rows).to_csv(improved_dir / 'brier_scores.csv', index=False)
    pd.DataFrame(ternary_rows).to_csv(improved_dir / 'ternary_classification_results.csv', index=False)
    pd.DataFrame(ternary_rows).to_csv(improved_dir / 'ternary_results.csv', index=False)
    pd.DataFrame(nested_rows).to_csv(improved_dir / 'nested_cv_results.csv', index=False)
    pd.DataFrame(temporal_rows).to_csv(improved_dir / 'temporal_validation.csv', index=False)
    pd.DataFrame(baseline_comparison_rows).to_csv(improved_dir / 'baseline_comparison.csv', index=False)
    if hyperparameter_rows:
        pd.DataFrame(hyperparameter_rows).to_csv(improved_dir / 'hyperparameters.csv', index=False)

    if all_results_for_plots:
        plot_roc_curves(all_results_for_plots, save_path=str(fig_dir / 'roc_all_drugs.png'))

    print(f"\nImproved pipeline results written to {improved_dir}")
    return {
        'drug_performance': drug_df,
        'summary': summary if not drug_df.empty else pd.DataFrame(),
        'output_dir': improved_dir,
    }

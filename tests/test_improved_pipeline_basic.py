"""Synthetic smoke test for improved pipeline (no HIVDB data required)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.feature_engineering import HIV_PROTEASE_REFERENCE
from src.improved_pipeline import (
    apply_oof_calibration,
    build_fusion_features,
    compare_calibration_methods_oof,
    compare_pooling_strategies,
    ensemble_soft_vote_cv,
    stacked_ensemble_cv,
    nested_cv_evaluation,
)
from src.models import MultiHeadAttentionPoolingClassifier
from src.rare_mutations import compute_drm_position_features
from src.evaluation import temperature_scaling


def _synthetic_cohort(n=80, embed_dim=32, seq_len=50, seed=0):
    rng = np.random.RandomState(seed)
    per_residue = [rng.randn(seq_len, embed_dim).astype(np.float32) for _ in range(n)]
    sequences = [HIV_PROTEASE_REFERENCE[:seq_len] for _ in range(n)]
    y = rng.binomial(1, 0.35, size=n).astype(int)
    return per_residue, sequences, y


def test_temperature_scaling():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.2, 0.7, 0.3, 0.8, 0.6, 0.4])
    out = temperature_scaling(y, p, p)
    assert out.shape == p.shape
    assert np.all((out >= 0) & (out <= 1))


def test_pooling_and_nested_cv():
    per_residue, sequences, y = _synthetic_cohort()
    pooling_df = compare_pooling_strategies(
        per_residue, sequences, HIV_PROTEASE_REFERENCE, y, n_splits=3
    )
    assert len(pooling_df) == 5
    assert 'auc' in pooling_df.columns

    x_mean = np.array([e.mean(axis=0) for e in per_residue])
    nested = nested_cv_evaluation(x_mean, y, outer_splits=3, inner_splits=2, n_repeats=1)
    assert not nested.get('skipped')
    assert 0 <= nested['test_auc'] <= 1


def test_fusion_includes_rare_mutation_features():
    per_residue, sequences, _ = _synthetic_cohort()
    embed_dim = per_residue[0].shape[1]
    attn = np.random.randn(len(per_residue), embed_dim).astype(np.float32)
    fusion = build_fusion_features(
        per_residue, sequences, HIV_PROTEASE_REFERENCE, attention_pooled=attn
    )
    assert fusion.shape[1] == embed_dim * 2 + embed_dim + 4


def test_multihead_attention_pooling_classifier():
    model = MultiHeadAttentionPoolingClassifier(input_dim=16, attention_hidden_dim=8, n_heads=3, dropout=0.2)
    x = torch.randn(4, 12, 16)
    mask = torch.ones(4, 12)
    logits, weights = model(x, mask)
    assert logits.shape == (4, 1)
    assert weights.shape == (4, 12, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4, 3), atol=1e-5)


def test_stacked_ensemble_cv():
    rng = np.random.RandomState(1)
    X = rng.randn(40, 16).astype(np.float32)
    y = rng.binomial(1, 0.4, size=40).astype(int)
    preds, info = stacked_ensemble_cv(X, y, n_splits=4)
    assert preds.shape == (40,)
    assert np.all((preds >= 0) & (preds <= 1))
    assert 'meta_model' in info


def test_drm_position_features():
    sequences = ['AAAAA', 'AACAA', 'AAGAA']
    reference = 'AAAAA'
    features = compute_drm_position_features(sequences, reference, 'PI')
    assert features.shape[0] == len(sequences)
    assert features.shape[1] % 2 == 0
    assert np.all(features[:, 0] >= 0)


def test_ensemble_and_calibration():
    per_residue, sequences, y = _synthetic_cohort()
    embed_dim = per_residue[0].shape[1]
    attn = np.random.randn(len(per_residue), embed_dim).astype(np.float32)
    fusion = build_fusion_features(
        per_residue, sequences, HIV_PROTEASE_REFERENCE, attention_pooled=attn
    )
    preds, weights = ensemble_soft_vote_cv(fusion, y, n_splits=3)
    assert len(weights) >= 2
    calib = compare_calibration_methods_oof(y, preds, n_splits=3)
    assert set(calib['method']) == {'raw', 'platt', 'isotonic', 'temperature'}
    applied = apply_oof_calibration(y, preds, n_splits=3)
    assert 'calibrated_ece' in applied
    assert len(applied['y_pred_calibrated']) == len(y)


if __name__ == '__main__':
    test_temperature_scaling()
    test_fusion_includes_rare_mutation_features()
    test_pooling_and_nested_cv()
    test_ensemble_and_calibration()
    print('Improved pipeline synthetic tests passed.')

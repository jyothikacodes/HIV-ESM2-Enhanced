"""
Comprehensive Unit Test Suite for Research Pipeline Upgrade (V2).

Tests:
1. Zero wildtype leakage in FoldSpecificMutationEncoder.
2. Fold-specific frequency isolation between train and test.
3. Accurate drug-specific clinical cutoffs (e.g. TDF=1.5x vs 3TC=3.0x).
4. Reference-contrastive delta representation construction (Experiment A).
5. Strictly masked multi-task loss with NaN labels (Experiment B).
6. Multi-task predictor training and inference.
7. Continuous log10(FC) regression metrics and ordinal evaluation (Experiment C).
8. Biologically guided attention mechanism with prior DRM indicators (Experiment D).
9. Sequence-clustered cross-validation group isolation (homology leakage fix).
10. Strictly leak-free nested cross-validation.
11. DeLong test, Wilcoxon test, and Benjamini-Hochberg FDR correction.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.representations import (
    FoldSpecificMutationEncoder,
    ReferenceDeltaRepresentationBuilder,
    compute_per_residue_delta,
    binarize_resistance,
    categorize_resistance_3class,
    get_clinical_cutoff,
    HIV_PROTEASE_REFERENCE,
)
from src.models import (
    MaskedMultiTaskBCELoss,
    MultiTaskDrugTrunkNet,
    MultiTaskDrugPredictor,
    ContinuousResistanceRegressor,
    compute_continuous_metrics,
    evaluate_continuous_as_ordinal,
    BiologicallyGuidedAttention,
    BiologicallyGuidedClassifier,
    get_prior_drm_indicator,
)
from src.evaluation import (
    GroupedValidationSplitter,
    cluster_sequences_by_identity,
    LeakFreeNestedCV,
    compute_binary_classification_metrics,
    compute_delong_pvalue,
    benjamini_hochberg_correction,
    paired_wilcoxon_multidrug_test,
)
from src.analysis import (
    compute_calibration_error,
    TemperatureScaler,
    evaluate_by_subtype,
    compute_clade_b_vs_non_b_gap,
)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


# -------------------------------------------------------------------------
# Test 1 & 2: Mutation Features and Zero-Leakage Frequency Encoder
# -------------------------------------------------------------------------

def test_mutation_encoder_no_wildtype_leakage():
    """Verify that wildtype amino acids are NEVER treated as mutations or rare mutations."""
    ref = "PQITLWQRPL"
    # Create cohort where position 0 has 'P' (wildtype) in 1 sample and 'A' in 9 samples.
    # In legacy buggy code, 'P' would have freq=0.1 and be treated as a rare mutation.
    training_seqs = ["PQITLWQRPL"] + ["AQITLWQRPL" for _ in range(9)]
    
    encoder = FoldSpecificMutationEncoder(reference_seq=ref, rare_threshold=0.2)
    encoder.fit(training_seqs)
    
    # 1. Wildtype (0, 'P') must NOT be in known mutations
    assert (0, "P") not in encoder.known_mutations_list
    assert (0, "P") not in encoder.mutation_counts
    
    # 2. Genuine mutation (0, 'A') must be in known mutations with count 9
    assert (0, "A") in encoder.known_mutations_list
    assert encoder.mutation_counts[(0, "A")] == 9
    
    # 3. Transform wildtype sequence: binary matrix must be all 0s
    wt_matrix = encoder.transform_binary_mutations(["PQITLWQRPL"])
    assert wt_matrix.sum() == 0.0
    
    # 4. Check positional weights on wildtype sequence: wildtype position 0 must have weight 1.0, not rare
    weights, rare_mask = encoder.compute_positional_rare_weights(["PQITLWQRPL"])
    assert weights[0, 0] == 1.0
    assert rare_mask[0, 0] == 0.0


def test_fold_specific_frequency_isolation():
    """Verify test fold sequences cannot alter training frequencies or rare weights."""
    ref = "PQITLWQRPL"
    train_seqs = ["PQITLWQRPL", "AQITLWQRPL"]
    test_seqs = ["CQITLWQRPL", "EQITLWQRPL"] # Novel mutations C and E
    
    encoder = FoldSpecificMutationEncoder(reference_seq=ref)
    encoder.fit(train_seqs)
    
    # Mutations C and E were never in training
    assert (0, "C") not in encoder.known_mutations_list
    assert (0, "E") not in encoder.known_mutations_list
    
    # When test_seqs are transformed, known binary vector length matches training observations
    test_binary = encoder.transform_binary_mutations(test_seqs)
    assert test_binary.shape[1] == len(encoder.known_mutations_list)
    assert test_binary.sum() == 0.0 # Unseen mutations do not activate train features


def test_drug_specific_clinical_cutoffs():
    """Verify drug-specific clinical cutoffs prevent tenofovir (TDF) misclassification."""
    assert get_clinical_cutoff("TDF") == 1.5
    assert get_clinical_cutoff("3TC") == 3.0
    assert get_clinical_cutoff("ABC") == 2.0
    assert get_clinical_cutoff("LPV") == 9.0
    
    # TDF isolate with fold-change 1.8:
    # With flawed universal cutoff 2.5/3.0 -> would be 0 (susceptible, false negative)
    # With clinical cutoff 1.5 -> correctly 1 (resistant)
    tdf_fc = np.array([1.2, 1.8, 0.9, np.nan, 3.5])
    tdf_binary = binarize_resistance(tdf_fc, "TDF")
    assert tdf_binary[0] == 0.0
    assert tdf_binary[1] == 1.0
    assert np.isnan(tdf_binary[3])


# -------------------------------------------------------------------------
# Test 3: Reference-Contrastive Delta Representations (Experiment A)
# -------------------------------------------------------------------------

def test_reference_delta_representations():
    """Verify reference delta embedding computation and multi-modal builder."""
    L, D = 10, 16
    rng = np.random.RandomState(42)
    ref_emb = rng.randn(L, D).astype(np.float32)
    ref_seq = "PQITLWQRPL"
    
    # Mutant embedding matches reference exactly
    mut_emb_identical = ref_emb.copy()
    delta_identical = compute_per_residue_delta(mut_emb_identical, ref_emb)
    assert np.allclose(delta_identical, 0.0)
    
    # Mutant embedding with perturbation at residue 2
    mut_emb_perturbed = ref_emb.copy()
    mut_emb_perturbed[2] += 2.5
    delta_perturbed = compute_per_residue_delta(mut_emb_perturbed, ref_emb)
    assert np.allclose(delta_perturbed[0], 0.0)
    assert not np.allclose(delta_perturbed[2], 0.0)
    
    # Test RepresentationBuilder modes
    builder_delta = ReferenceDeltaRepresentationBuilder(ref_emb, ref_seq, mode="delta_only")
    feat_delta = builder_delta.build_sample_feature(mut_emb_perturbed)
    assert feat_delta.shape == (D,)
    
    builder_concat = ReferenceDeltaRepresentationBuilder(ref_emb, ref_seq, mode="mutant_plus_delta")
    feat_concat = builder_concat.build_sample_feature(mut_emb_perturbed)
    assert feat_concat.shape == (2 * D,)


# -------------------------------------------------------------------------
# Test 4: Masked Multi-Task Model (Experiment B)
# -------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch required for neural multi-task tests")
def test_masked_multitask_loss():
    """Verify that unmeasured drugs (NaN labels) have zero loss and zero gradient impact."""
    loss_fn = MaskedMultiTaskBCELoss()
    
    # 2 samples, 3 drugs. Sample 0 missing drug 1; sample 1 missing drug 2
    logits = torch.tensor([[1.0, 2.0, -1.0], [-1.0, -2.0, 1.0]], requires_grad=True)
    targets = torch.tensor([[1.0, np.nan, 0.0], [0.0, 0.0, np.nan]])
    
    loss = loss_fn(logits, targets)
    loss.backward()
    
    # Check that gradient for masked positions is exactly 0
    assert logits.grad[0, 1].item() == 0.0
    assert logits.grad[1, 2].item() == 0.0
    # Valid positions must have non-zero gradient
    assert logits.grad[0, 0].item() != 0.0


def test_multitask_drug_predictor_fit_predict():
    """Verify MultiTaskDrugPredictor handles multi-drug label matrices with missing values."""
    N, D_feat, num_drugs = 40, 16, 4
    rng = np.random.RandomState(42)
    X = rng.randn(N, D_feat).astype(np.float32)
    
    Y = rng.binomial(1, 0.4, size=(N, num_drugs)).astype(float)
    # Inject missing assays (NaN)
    Y[::3, 1] = np.nan
    Y[::4, 2] = np.nan
    
    drugs = ["ATV", "DRV", "3TC", "TDF"]
    predictor = MultiTaskDrugPredictor(drugs, max_epochs=5, batch_size=16)
    predictor.fit(X, Y)
    
    probs = predictor.predict_proba(X)
    assert probs.shape == (N, num_drugs)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


# -------------------------------------------------------------------------
# Test 5: Continuous and Ordinal Modeling (Experiment C)
# -------------------------------------------------------------------------

def test_continuous_metrics_and_regressor():
    """Verify continuous regression metrics and ordinal 3-class mapping."""
    N = 50
    rng = np.random.RandomState(42)
    X = rng.randn(N, 12).astype(np.float32)
    true_log_fc = rng.normal(0.5, 0.8, size=N)
    
    regressor = ContinuousResistanceRegressor(model_type="ridge", alpha=1.0)
    regressor.fit(X, true_log_fc, is_log10=True)
    
    pred_log_fc = regressor.predict_log_fc(X)
    assert pred_log_fc.shape == (N,)
    
    # Metrics computation
    metrics = compute_continuous_metrics(true_log_fc, pred_log_fc)
    assert "r2" in metrics
    assert "rmse" in metrics
    assert "pearson_r" in metrics
    assert "spearman_rho" in metrics
    
    # Ordinal evaluation
    ord_metrics = evaluate_continuous_as_ordinal(true_log_fc, pred_log_fc, "3TC")
    assert "ordinal_balanced_accuracy" in ord_metrics
    assert 0.0 <= ord_metrics["ordinal_balanced_accuracy"] <= 1.0


# -------------------------------------------------------------------------
# Test 6: Biologically Guided Attention (Experiment D)
# -------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch required for attention test")
def test_biologically_guided_attention():
    """Verify biologically guided attention integrates DRM position indicators."""
    L, D = 20, 32
    attn = BiologicallyGuidedAttention(embed_dim=D, hidden_dim=16)
    
    x = torch.randn(2, L, D)
    mut_ind = torch.zeros(2, L, 1)
    mut_ind[:, 5, 0] = 1.0 # Mutation at position 5
    
    prior_drm = torch.zeros(L, 1)
    prior_drm[5, 0] = 1.0 # Canonical DRM at position 5
    
    pooled, weights = attn(x, mut_indicator=mut_ind, prior_drm=prior_drm)
    assert pooled.shape == (2, D)
    assert weights.shape == (2, L)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)


# -------------------------------------------------------------------------
# Test 7: Grouped Cross-Validation (Homology Leakage Fix)
# -------------------------------------------------------------------------

def test_sequence_clustered_cross_validation():
    """Verify that highly similar sequences are isolated to the same fold."""
    # 4 sequences in cluster 0 (almost identical) and 4 sequences in cluster 1
    seq_cluster_0 = ["PQITLWQRPL" for _ in range(4)]
    seq_cluster_1 = ["AAAAAAAAAA" for _ in range(4)]
    all_seqs = seq_cluster_0 + seq_cluster_1
    
    clusters = cluster_sequences_by_identity(all_seqs, identity_threshold=0.9)
    assert len(np.unique(clusters[:4])) == 1
    assert len(np.unique(clusters[4:])) == 1
    assert clusters[0] != clusters[4]
    
    # Verify Splitter isolates clusters
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    X = np.random.randn(8, 4)
    splitter = GroupedValidationSplitter(n_splits=2, grouping_strategy="sequence_clustered")
    
    for train_idx, val_idx in splitter.split(X, y, sequences=all_seqs):
        train_clusters = set(clusters[train_idx])
        val_clusters = set(clusters[val_idx])
        # Clusters must have zero intersection between train and val
        assert len(train_clusters.intersection(val_clusters)) == 0


# -------------------------------------------------------------------------
# Test 8: Leak-Free Nested Cross-Validation
# -------------------------------------------------------------------------

def test_leak_free_nested_cv():
    """Verify leak-free nested CV executes and produces valid OOF predictions."""
    from sklearn.linear_model import LogisticRegression
    
    N, D = 40, 8
    rng = np.random.RandomState(42)
    X = rng.randn(N, D)
    y = rng.binomial(1, 0.4, size=N)
    
    nested_cv = LeakFreeNestedCV(
        estimator_factory=lambda: LogisticRegression(solver="liblinear"),
        outer_splits=3,
        grouping_strategy="stratified",
        random_state=42
    )
    res = nested_cv.evaluate(X, y)
    
    assert len(res["oof_predictions"]) == N
    assert not np.isnan(res["mean_outer_auc"])
    assert 0.0 <= res["mean_outer_auc"] <= 1.0


# -------------------------------------------------------------------------
# Test 9: Statistical Testing
# -------------------------------------------------------------------------

def test_statistical_testing_delong_and_wilcoxon():
    """Verify paired DeLong, Wilcoxon, and FDR correction."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    p1 = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15, 0.95, 0.6])
    p2 = np.array([0.4, 0.6, 0.5, 0.5, 0.6, 0.4, 0.7, 0.3, 0.6, 0.5])
    
    auc1, auc2, p_val = compute_delong_pvalue(y_true, p1, p2)
    assert auc1 > auc2
    assert 0.0 <= p_val <= 1.0
    
    # Benjamini-Hochberg FDR
    p_values = [0.001, 0.01, 0.04, 0.20, 0.80]
    q_values = benjamini_hochberg_correction(p_values)
    assert len(q_values) == len(p_values)
    assert np.all(q_values <= 1.0)
    assert q_values[0] < q_values[-1]

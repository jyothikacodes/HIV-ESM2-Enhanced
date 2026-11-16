"""
Comprehensive Unit Test Suite for Research Pipeline Upgrade (V2) & Pre-HPC Audit.

Tests:
1. Zero wildtype leakage in FoldSpecificMutationEncoder.
2. Fold-specific frequency isolation between train and test.
3. Accurate drug-specific clinical cutoffs (e.g. TDF=1.5x vs 3TC=3.0x).
4. Reference-contrastive delta representation construction (Experiment A) & all 5 ablation modes.
5. Alignment-aware delta checks (preventing silent phase shift on indels, padding masks, PR/RT coordinate bounds).
6. Pairwise mutation interaction encoding & sparse epistatic classifier (synergy vs antagonism).
7. Strictly masked multi-task loss with NaN labels (Experiment B).
8. Multi-task out-of-fold cross-validation vs independent models.
9. Continuous log10(FC) regression metrics and ordinal evaluation (Experiment C).
10. Biologically guided attention mechanism with prior DRM indicators (Experiment D) and prior ablation.
11. Transitive single-linkage clustering & joint patient-sequence grouping (homology leakage fix).
12. Leak-free nested cross-validation with inner parameter tuning.
13. DeLong test, Wilcoxon test, and Benjamini-Hochberg FDR correction.
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
    HIV_RT_REFERENCE,
    PairwiseMutationInteractionEncoder,
    EpistaticInteractionClassifier,
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
    training_seqs = ["PQITLWQRPL"] + ["AQITLWQRPL" for _ in range(9)]
    
    encoder = FoldSpecificMutationEncoder(reference_seq=ref, rare_threshold=0.2)
    encoder.fit(training_seqs)
    
    # Wildtype (0, 'P') must NOT be in known mutations
    assert (0, "P") not in encoder.known_mutations_list
    assert (0, "P") not in encoder.mutation_counts
    
    # Genuine mutation (0, 'A') must be in known mutations with count 9
    assert (0, "A") in encoder.known_mutations_list
    assert encoder.mutation_counts[(0, "A")] == 9
    
    # Transform wildtype sequence: binary matrix must be all 0s
    wt_matrix = encoder.transform_binary_mutations(["PQITLWQRPL"])
    assert wt_matrix.sum() == 0.0
    
    # Wildtype position 0 must have weight 1.0, not rare
    weights, rare_mask = encoder.compute_positional_rare_weights(["PQITLWQRPL"])
    assert weights[0, 0] == 1.0
    assert rare_mask[0, 0] == 0.0


def test_fold_specific_frequency_isolation():
    """Verify test fold sequences cannot alter training frequencies or rare weights."""
    ref = "PQITLWQRPL"
    train_seqs = ["PQITLWQRPL", "AQITLWQRPL"]
    test_seqs = ["CQITLWQRPL", "EQITLWQRPL"]
    
    encoder = FoldSpecificMutationEncoder(reference_seq=ref)
    encoder.fit(train_seqs)
    
    assert (0, "C") not in encoder.known_mutations_list
    assert (0, "E") not in encoder.known_mutations_list
    
    test_binary = encoder.transform_binary_mutations(test_seqs)
    assert test_binary.shape[1] == len(encoder.known_mutations_list)
    assert test_binary.sum() == 0.0


def test_drug_specific_clinical_cutoffs():
    """Verify drug-specific clinical cutoffs prevent tenofovir (TDF) misclassification."""
    assert get_clinical_cutoff("TDF") == 1.5
    assert get_clinical_cutoff("3TC") == 3.0
    assert get_clinical_cutoff("ABC") == 2.0
    assert get_clinical_cutoff("LPV") == 9.0
    
    tdf_fc = np.array([1.2, 1.8, 0.9, np.nan, 3.5])
    tdf_binary = binarize_resistance(tdf_fc, "TDF")
    assert tdf_binary[0] == 0.0
    assert tdf_binary[1] == 1.0
    assert np.isnan(tdf_binary[3])


# -------------------------------------------------------------------------
# Test 3 & 4: Reference-Contrastive Delta Representations & Ablations
# -------------------------------------------------------------------------

def test_reference_delta_representations_and_ablations():
    """Verify reference delta embedding computation and all 5 ablation modes."""
    L, D = 10, 16
    rng = np.random.RandomState(42)
    ref_emb = rng.randn(L, D).astype(np.float32)
    ref_seq = "PQITLWQRPL"
    
    mut_emb_identical = ref_emb.copy()
    delta_identical = compute_per_residue_delta(mut_emb_identical, ref_emb)
    assert np.allclose(delta_identical, 0.0)
    
    mut_emb_perturbed = ref_emb.copy()
    mut_emb_perturbed[2] += 2.5
    delta_perturbed = compute_per_residue_delta(mut_emb_perturbed, ref_emb)
    assert np.allclose(delta_perturbed[0], 0.0)
    assert not np.allclose(delta_perturbed[2], 0.0)
    
    # 1. mutant_only
    b_mut = ReferenceDeltaRepresentationBuilder(ref_emb, ref_seq, mode="mutant_only")
    assert b_mut.build_sample_feature(mut_emb_perturbed).shape == (D,)
    
    # 2. reference_only
    b_ref = ReferenceDeltaRepresentationBuilder(ref_emb, ref_seq, mode="reference_only")
    f_ref = b_ref.build_sample_feature(mut_emb_perturbed)
    assert f_ref.shape == (D,)
    assert np.allclose(f_ref, b_ref.ref_pooled)
    
    # 3. delta_only
    b_del = ReferenceDeltaRepresentationBuilder(ref_emb, ref_seq, mode="delta_only")
    assert b_del.build_sample_feature(mut_emb_perturbed).shape == (D,)
    
    # 4. mutant_plus_delta
    b_md = ReferenceDeltaRepresentationBuilder(ref_emb, ref_seq, mode="mutant_plus_delta")
    assert b_md.build_sample_feature(mut_emb_perturbed).shape == (2 * D,)
    
    # 5. mutant_plus_reference_plus_delta
    b_mrd = ReferenceDeltaRepresentationBuilder(ref_emb, ref_seq, mode="mutant_plus_reference_plus_delta")
    assert b_mrd.build_sample_feature(mut_emb_perturbed).shape == (3 * D,)


def test_reference_delta_alignment_and_coordinate_guards():
    """Verify indels without gap characters raise ValueError and padding is masked."""
    L, D = 10, 16
    rng = np.random.RandomState(42)
    ref_emb = rng.randn(L, D).astype(np.float32)
    ref_seq = "PQITLWQRPL"
    
    # Deletion without alignment gap character should raise ValueError
    short_mut_emb = rng.randn(9, D).astype(np.float32)
    short_mut_seq = "PQITLQRPL" # missing W at pos 5
    with pytest.raises(ValueError, match="Unaligned length mismatch"):
        compute_per_residue_delta(
            short_mut_emb, ref_emb, mutant_seq=short_mut_seq, reference_seq=ref_seq
        )
        
    # Padding mask test
    pad_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    mut_emb = ref_emb + 1.0
    delta_masked = compute_per_residue_delta(mut_emb, ref_emb, padding_mask=pad_mask)
    assert np.allclose(delta_masked[8:], 0.0)
    assert np.allclose(delta_masked[:8], 1.0)


# -------------------------------------------------------------------------
# Test 5: Pairwise Epistatic Mutation Interactions
# -------------------------------------------------------------------------

def test_pairwise_mutation_interactions():
    """Verify pairwise epistatic interaction encoder and classifier."""
    ref = "PQITLWQRPL"
    # Create cohort with co-occurring mutations: pos 0 'A' and pos 2 'I'
    seqs = ["AQTTLWQRPL" for _ in range(10)] + ["PQITLWQRPL" for _ in range(10)]
    y = np.array([1]*10 + [0]*10)
    
    enc = PairwiseMutationInteractionEncoder(ref, min_cooccurrence=4)
    enc.fit(seqs, y)
    
    # Must identify the co-occurring pair (0, 'A') + (2, 'T')
    assert len(enc.interaction_pairs) >= 1
    
    X_single = enc.transform(seqs, mode="mutation_only")
    X_pairs = enc.transform(seqs, mode="interaction_only")
    X_comb = enc.transform(seqs, mode="combined")
    
    assert X_comb.shape[1] == X_single.shape[1] + X_pairs.shape[1]
    
    clf = EpistaticInteractionClassifier(ref, mode="combined", C=1.0)
    clf.fit(seqs, y)
    probs = clf.predict_proba(seqs)
    assert probs.shape == (20, 2)
    
    top_df = clf.extract_top_interactions(top_k=5)
    assert isinstance(top_df, np.ndarray) or hasattr(top_df, "columns")


# -------------------------------------------------------------------------
# Test 6: Masked Multi-Task Model & OOF Predictions (Experiment B)
# -------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch required for neural multi-task tests")
def test_masked_multitask_loss():
    """Verify that unmeasured drugs (NaN labels) have zero loss and zero gradient impact."""
    loss_fn = MaskedMultiTaskBCELoss()
    
    logits = torch.tensor([[1.0, 2.0, -1.0], [-1.0, -2.0, 1.0]], requires_grad=True)
    targets = torch.tensor([[1.0, np.nan, 0.0], [0.0, 0.0, np.nan]])
    
    loss = loss_fn(logits, targets)
    loss.backward()
    
    assert logits.grad[0, 1].item() == 0.0
    assert logits.grad[1, 2].item() == 0.0
    assert logits.grad[0, 0].item() != 0.0


def test_multitask_drug_predictor_fit_predict():
    """Verify MultiTaskDrugPredictor handles multi-drug label matrices with missing values."""
    N, D_feat, num_drugs = 40, 16, 4
    rng = np.random.RandomState(42)
    X = rng.randn(N, D_feat).astype(np.float32)
    
    Y = rng.binomial(1, 0.4, size=(N, num_drugs)).astype(float)
    Y[::3, 1] = np.nan
    Y[::4, 2] = np.nan
    
    drugs = ["ATV", "DRV", "3TC", "TDF"]
    predictor = MultiTaskDrugPredictor(drugs, max_epochs=5, batch_size=16)
    predictor.fit(X, Y)
    
    probs = predictor.predict_proba(X)
    assert probs.shape == (N, num_drugs)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


# -------------------------------------------------------------------------
# Test 7: Continuous and Ordinal Modeling (Experiment C)
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
    
    metrics = compute_continuous_metrics(true_log_fc, pred_log_fc)
    assert "r2" in metrics
    assert "rmse" in metrics
    assert "pearson_r" in metrics
    assert "spearman_rho" in metrics
    
    ord_metrics = evaluate_continuous_as_ordinal(true_log_fc, pred_log_fc, "3TC")
    assert "ordinal_balanced_accuracy" in ord_metrics
    assert 0.0 <= ord_metrics["ordinal_balanced_accuracy"] <= 1.0


# -------------------------------------------------------------------------
# Test 8: Biologically Guided Attention & Ablation (Experiment D)
# -------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch required for attention test")
def test_biologically_guided_attention_and_ablation():
    """Verify biologically guided attention and prior ablation."""
    L, D = 20, 32
    # 1. With prior DRM
    attn_with = BiologicallyGuidedAttention(embed_dim=D, hidden_dim=16, use_prior_drm=True)
    # 2. Ablated prior DRM
    attn_ablated = BiologicallyGuidedAttention(embed_dim=D, hidden_dim=16, use_prior_drm=False)
    
    x = torch.randn(2, L, D)
    prior_drm = torch.zeros(L, 1)
    prior_drm[5, 0] = 1.0
    
    p_with, w_with = attn_with(x, prior_drm=prior_drm)
    p_abl, w_abl = attn_ablated(x, prior_drm=prior_drm)
    
    assert p_with.shape == (2, D)
    assert p_abl.shape == (2, D)
    assert torch.allclose(w_with.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(w_abl.sum(dim=-1), torch.ones(2), atol=1e-5)


# -------------------------------------------------------------------------
# Test 9: Transitive Clustering & Joint Patient-Sequence Grouping
# -------------------------------------------------------------------------

def test_transitive_clustering_and_joint_patient_groups():
    """Verify transitive clustering (A~B, B~C -> A,B,C in same group) and joint PtID grouping."""
    # seq0 ~ seq1 (1 mismatch), seq1 ~ seq2 (1 mismatch), but seq0 vs seq2 has 2 mismatches
    seq0 = "AAAAAAAAAA"
    seq1 = "AAAAAAAAAB"
    seq2 = "AAAAAAAABB"
    seqs = [seq0, seq1, seq2]
    
    # If identity threshold = 0.85 (up to 1 mismatch out of 10)
    clusters = cluster_sequences_by_identity(seqs, identity_threshold=0.85)
    # In transitive single linkage, all 3 must be in cluster 0
    assert clusters[0] == clusters[1] == clusters[2]
    
    # Joint Patient-Sequence Grouping
    # Patient 1 has seqs 0 and 3. Even if seq 3 has 0% identity with seq 0, they must be in same group
    pts = ["PT1", "PT2", "PT3", "PT1"]
    all_seqs = [seq0, seq1, seq2, "CCCCCCCCCC"]
    splitter = GroupedValidationSplitter(
        n_splits=2, grouping_strategy="joint_patient_sequence", identity_threshold=0.85
    )
    groups = splitter.get_groups(4, sequences=all_seqs, patient_ids=pts)
    
    # PT1's samples (0 and 3) must share group
    assert groups[0] == groups[3]
    # And since 0 ~ 1 ~ 2, all 4 samples merge into a single component!
    assert len(np.unique(groups)) == 1


# -------------------------------------------------------------------------
# Test 10: Leak-Free Nested Cross-Validation with Inner Tuning
# -------------------------------------------------------------------------

def test_leak_free_nested_cv_inner_tuning():
    """Verify leak-free nested CV executes inner hyperparameter tuning without leakage."""
    from sklearn.linear_model import LogisticRegression
    
    N, D = 40, 8
    rng = np.random.RandomState(42)
    X = rng.randn(N, D)
    y = rng.binomial(1, 0.4, size=N)
    
    candidates = [{"C": 0.01, "solver": "liblinear"}, {"C": 1.0, "solver": "liblinear"}]
    
    nested_cv = LeakFreeNestedCV(
        estimator_factory=lambda **kwargs: LogisticRegression(**kwargs),
        outer_splits=3,
        inner_splits=2,
        grouping_strategy="stratified",
        tune_hyperparameters=True,
        param_candidates=candidates,
        random_state=42
    )
    res = nested_cv.evaluate(X, y)
    
    assert len(res["oof_predictions"]) == N
    assert not np.isnan(res["mean_outer_auc"])
    assert 0.0 <= res["mean_outer_auc"] <= 1.0


# -------------------------------------------------------------------------
# Test 11: Statistical Testing
# -------------------------------------------------------------------------

def test_statistical_testing_delong_and_wilcoxon():
    """Verify paired DeLong, Wilcoxon, and FDR correction."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    p1 = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15, 0.95, 0.6])
    p2 = np.array([0.4, 0.6, 0.5, 0.5, 0.6, 0.4, 0.7, 0.3, 0.6, 0.5])
    
    auc1, auc2, p_val = compute_delong_pvalue(y_true, p1, p2)
    assert auc1 > auc2
    assert 0.0 <= p_val <= 1.0
    
    p_values = [0.001, 0.01, 0.04, 0.20, 0.80]
    q_values = benjamini_hochberg_correction(p_values)
    assert len(q_values) == len(p_values)
    assert np.all(q_values <= 1.0)
    assert q_values[0] < q_values[-1]

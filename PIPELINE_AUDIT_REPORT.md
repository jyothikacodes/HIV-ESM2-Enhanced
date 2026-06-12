# HIV-ESM-2 Pipeline Audit Report

**Date:** 2026-06-11  
**Current performance (reported):** Mean Test AUC = 0.9487, Mean AUC Drop = 0.0508, p = 1.28e-05  
**Target performance:** Mean Test AUC > 0.968, Mean AUC Drop < 0.034

---

## 1. Repository Structure

| Component | Path | Role |
|-----------|------|------|
| Data acquisition | `notebooks/01_data_acquisition.ipynb` | HIVDB → FASTA/CSV |
| Preprocessing | `notebooks/02_data_preprocessing.ipynb` | Cleaning, optional rare-mutation prep |
| Embeddings | `notebooks/03_esm2_embedding_extraction.ipynb` | ESM-2 pooled + optional per-residue |
| Training | `notebooks/04_baseline_and_enhanced_training.ipynb` | Mutation baselines + ESM classifiers |
| Evaluation | `notebooks/05_evaluation_and_validation.ipynb` | Temporal, bootstrap, calibration |
| Explainability | `notebooks/06_explainability_framework.ipynb` | DRM enrichment, optional SHAP |
| Statistics | `notebooks/07_statistical_validation.ipynb` | Subtype/temporal/PLM, Option B |
| Experiment runner | `run_experiments.py` | Subsampled end-to-end validation |
| Core library | `src/*.py` (19 modules) | Feature engineering, models, evaluation |

**Data paths:** `data/raw/`, `data/processed/`, `data/embeddings/`, `results/`

**No centralized config** (YAML/TOML) — hyperparameters are hard-coded or notebook toggles.

---

## 2. Proposed Methods: Implementation vs Execution

### 2.1 Fully Implemented AND Executed (Default Path)

| Method | Module | Default Execution |
|--------|--------|-------------------|
| ESM-2 embeddings (layer 33, 650M) | `feature_engineering.py` | Notebook 03 |
| Mean pooling | `feature_engineering.py` | Notebook 03–07 (primary downstream input) |
| Binary classification (FC ≥ 2.5) | `data_processing.py`, `models.py` | Notebook 04 |
| Logistic Regression classifier | `models.py` | Notebook 04 (default ESM classifier) |
| XGBoost mutation baseline | `models.py` | Notebook 04 |
| Stratified 5-fold CV | `models.py`, `evaluation.py` | Notebook 04–05 |
| Bootstrap AUC CI | `evaluation.py` | Notebook 05 |
| DeLong AUC test | `evaluation.py` | Notebook 04/07 |
| Platt scaling | `evaluation.py` → `calibration.py` | Notebook 05 (basic cells) |
| Isotonic regression | `evaluation.py` → `calibration.py` | Notebook 05 (basic cells) |
| DRM enrichment | `interpretability.py` | Notebook 06 (default) |
| SHAP (KernelExplainer) | `interpretability.py`, `shap_explainability.py` | Partial — class-level in `run_experiments.py` |

### 2.2 Implemented but NOT Executed in Default Notebook Pipeline

These exist in `src/` but are gated behind `RUN_* = False` toggles or only run via `run_experiments.py` / `option_b_evaluation.py`:

| Method | Module | Toggle / Entry Point | Why Missing from Outputs |
|--------|--------|----------------------|--------------------------|
| **Rare Mutation Weighting** | `rare_mutations.py`, `models.py` | `RUN_RARE_MUTATION_PREP` (nb02), `RUN_ATTENTION_TRAINING` (nb04) | Default notebooks skip rare-weight prep and attention training |
| **Ternary Classification** | `ternary_classification.py` | `RUN_TERNARY_CLASSIFICATION` (nb05) | Off by default; only in `run_experiments.py` |
| **Calibration Framework (full)** | `calibration.py` | `RUN_CALIBRATION_COMPARISON` (nb05) | Basic Platt/isotonic cells run; full per-drug OOF comparison off |
| Max pooling (downstream) | `feature_engineering.py` | Extracted in nb03 | nb04+ loads `_pooled_mean.npy` only |
| Mean+Max pooling (downstream) | `feature_engineering.py` | Extracted in nb03 | Never used in training notebooks |
| Learned attention pooling | `models.py` (`AttentionWeightedClassifier`) | `RUN_ATTENTION_TRAINING` (nb04) | Off by default |
| ESM native attention pooling | `feature_engineering.py` (`attention_weighted_pooling`) | — | **Never called** (dead code) |
| Per-residue embeddings | `feature_engineering.py` | `RUN_PER_RESIDUE_EXTRACTION` (nb03) | Off by default |
| Dual SHAP fusion | `dual_shap_explainability.py` | `RUN_EXPLAINABILITY_EXTENSIONS` (nb06) | Off by default |
| Option B validation suite | `option_b_evaluation.py` | `RUN_STATISTICAL_VALIDATION` (nb07) | Off by default |
| Counterfactual analysis | `counterfactual_mutation_analysis.py` | Manual only | Not in notebook chain |
| OOF per-drug calibration | `calibration.calibrate_per_drug()` | `RUN_CALIBRATION_COMPARISON` | Off by default |
| Extended temporal validation | `temporal_validation.py` | `RUN_EXTENDED_TEMPORAL_VALIDATION` | Off by default |
| Rare weights in `per_drug_training(attention)` | `models.py` L578–611 | — | Attention CV path omits `rare_mutation_weights_list` |

### 2.3 Completely Absent (Before This Improvement)

| Method | Status |
|--------|--------|
| **Attention pooling comparison** (mean vs max vs mean_max vs learned) | No systematic benchmark |
| **Multi-pooling fusion** (mean + max + attention concat) | Not implemented |
| **Drug-specific Optuna HPO** | Not implemented |
| **Classifier ensemble** (XGB + LGBM + LR + RF soft voting) | Not implemented |
| **SHAP-based feature selection** | Not implemented |
| **Nested cross-validation** | Not implemented |
| **Repeated stratified CV** | Not implemented |
| **Temperature scaling** | Not implemented |
| **LightGBM** | Not in dependencies |
| **Optuna** | Not in dependencies |
| ESM-1v 5-model ensemble | Documented TODO only |
| Voting/stacking across classifiers | Not implemented |

---

## 3. Evaluation Metrics: How They Are Computed

| Metric | Function | Notes |
|--------|----------|-------|
| AUC-ROC | `evaluation.compute_auc()` | Standard `roc_auc_score` on OOF or holdout probs |
| AUC-PR | `compute_classification_metrics()` | `average_precision_score` |
| AUC Drop | `run_experiments.py`, `option_b_evaluation.py` | CV AUC − temporal holdout AUC |
| Bootstrap CI | `evaluation.bootstrap_auc()` | 1000 resamples, percentile CI |
| DeLong test | `evaluation.delong_test()` | Paired AUC comparison |
| Wilcoxon | Notebooks 04/07 | Paired per-drug AUC comparison |
| ECE / MCE / Brier | `compute_calibration_metrics()` | 10-bin calibration |
| Ternary macro AUC | `ternary_classification.py` | OvR macro average |
| DRM enrichment | `interpretability.compute_drm_enrichment()` | Fisher's exact on top-K residues |

**Gap:** No unified reporting of Train / Val / Test AUC with nested CV. AUC drop is temporal-only, not nested-holdout-based.

---

## 4. Data Leakage Risks

| Area | Risk | Mitigation in Improved Pipeline |
|------|------|--------------------------------|
| Calibration | Fitting calibrator on test set | OOF calibration within CV folds |
| Hyperparameter tuning | Tuning on test fold | Nested CV (inner = tune, outer = test) |
| Rare mutation frequencies | Computed on full cohort | Computed within training fold only |
| SHAP feature selection | Selecting on test data | Selection within inner CV only |
| Embedding extraction | ESM is frozen | No leakage (unsupervised) |

---

## 5. Root Cause: Why Proposal Claims Are Missing from Outputs

1. **Notebook toggles default to `False`** — Rare mutation, ternary, full calibration, attention training, and Option B are opt-in.
2. **Pooling mismatch** — nb03 extracts mean/max/mean_max but nb04+ only consumes mean embeddings.
3. **Script vs notebook split** — `run_experiments.py` runs more features but on **subsampled** data (default n=100/class); full-cohort notebook results don't include those analyses.
4. **No unified publication runner** — Results scattered across notebook outputs, `results/option_b_evaluation/`, and `run_experiments.py` CSVs.

---

## 6. Improvement Plan (Phases 2–7)

Implemented in `src/improved_pipeline.py` and `run_improved_pipeline.py`:

| Phase | Deliverable |
|-------|-------------|
| 2A | Pooling comparison: mean, max, mean_max, attention |
| 2B | Multi-pooling fusion classifier |
| 2C | Per-drug Optuna hyperparameter optimization |
| 2D | Weighted soft-vote ensemble (XGB, LGBM, LR, RF) |
| 2E | Optuna search over LR/XGB/RF hyperparameters |
| 2F | SHAP-based embedding dimension selection |
| 3 | Nested + repeated stratified CV, bootstrap CIs, early stopping |
| 4 | Rare mutation weighting ablation (ESM / mutation / combined / weighted) |
| 5 | Ternary vs binary comparison with full metrics |
| 6 | Platt / isotonic / temperature scaling with reliability diagrams |
| 7 | Publication tables, ROC/PR curves, summary comparison vs paper |

---

## 7. Entry Points After Improvement

```bash
# Notebook chain (recommended) — set ENABLE_IMPROVED_PIPELINE in notebooks/pipeline_config.py
jupyter notebook notebooks/01_data_acquisition.ipynb  # run 01 → 07 in order

# Standalone improved evaluation
python run_improved_pipeline.py --full_data --optuna_trials 30

# Original experiment runner (unchanged)
python run_experiments.py --subset_size 100
```

The notebook chain wires the improved pipeline via `notebooks/pipeline_config.py` (`ENABLE_IMPROVED_PIPELINE = True`). Notebook 07 runs `run_publication_evaluation()` and writes to `results/improved_pipeline/`.

---

## 8. Summary Table

| Component | Code Exists | Default Executed | In Proposal | Gap Severity |
|-----------|-------------|------------------|-------------|--------------|
| ESM-2 + mean pooling | Yes | Yes | Yes | None |
| Max / mean_max pooling | Yes | No (extract only) | Yes | High |
| Attention pooling | Yes | No | Yes | High |
| Rare mutation weighting | Yes | No | Yes | **Critical** |
| Ternary classification | Yes | No | Yes | **Critical** |
| Calibration framework | Partial | Partial | Yes | High |
| Temperature scaling | **Added** | — | Yes | Was absent |
| Ensemble learning | **Added** | — | Yes | Was absent |
| Optuna HPO | **Added** | — | Yes | Was absent |
| Nested CV | **Added** | — | Yes | Was absent |
| SHAP feature selection | **Added** | — | Yes | Was absent |

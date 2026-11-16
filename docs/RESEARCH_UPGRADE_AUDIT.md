# Comprehensive Research Audit: HIV-1 Drug Resistance Prediction Pipeline

## 1. Executive Summary

This audit critically evaluates the codebase, experimental protocols, data integrity, and statistical claims of the `HIV-ESM2-Enhanced` repository (`jyothikacodes/HIV-ESM2-Enhanced`). The primary objective was to verify whether the reported improvements over the benchmark established by Hayden Farquhar (*Bioinformatics* 2026, DOI: `10.1093/bioinformatics/btag260`) are supported by valid, reproducible, and leak-free machine learning and computational biology practices.

**Audit Verdict:** *Partially Valid; Core Claims Contradicted by Data and Compromised by Leakage.*
While per-drug resistance prediction is feasible and individual drug ROC AUCs of 0.94–0.98 are observed for canonical drugs (e.g. EFV, NVP, ATV), the headline claim that ESM-2 embeddings or rare-mutation attention mechanisms significantly outperform simple mutation-encoding baselines is **not supported** by the repository's own empirical results. Furthermore, multiple severe data leakage vectors and implementation errors were identified in the evaluation pipeline.

---

## 2. Complete Repository Architecture & Data Flow Audit

### 2.1 Repository Layout
- `src/improved_pipeline.py`: Monolithic 1,500+ line pipeline combining feature extraction, pooling, attention training, stacked ensembles, and nested cross-validation.
- `src/rare_mutations.py`: Frequency matrix calculations, rare mutation masks, and DRM feature calculations.
- `src/models/`: Multi-head attention pooling, PyTorch classifiers, and scikit-learn wrappers.
- `src/evaluation/`: Cross-validation loops, calibration functions, and DeLong test implementations.
- `src/scientific_validation.py` / `notebooks/07_scientific_validation.ipynb`: Small-subset validation script.
- `results/`: Historical output directories (`improved_pipeline`, `baseline_results.csv`, `esm2_vs_baseline.csv`).

### 2.2 End-to-End Data Flow
1. **Raw Input**: Stanford HIVDB sequence records (`SeqID`, `PtID`, `Subtype`, aligned amino acid sequences) paired with continuous phenotypic susceptibility fold-change values across 18 drugs (8 PIs, 6 NRTIs, 4 NNRTIs).
2. **Label Discretization**: Continuous fold-change values were binarized using a single indiscriminate threshold (2.5 or 3.0) across all drugs regardless of clinical breakpoints.
3. **Representation Generation**: Per-residue 1,280-dimensional embeddings extracted from ESM-2 (`esm2_t33_650M_UR50D`).
4. **Attention Pooling**: Per-residue embeddings pooled via a multi-head attention module.
5. **Feature Concatenation (Fusion)**: Mean-pooled ESM-2 + max-pooled ESM-2 + attention-pooled ESM-2 + DRM positional features + rare mutation counts.
6. **Classification & Ensembling**: XGBoost, Logistic Regression, Random Forest, or a Meta-Learner Logistic Stacking classifier.

---

## 3. Confirmed Implementation Bugs and Methodological Weaknesses

### 3.1 Critical Data Leakages

| Leakage ID | Component | Mechanism | Consequence |
|---|---|---|---|
| **LEAK-01** | `improved_pipeline.py:1101` & `rare_mutations.py` | `compute_mutation_frequencies` called globally on the entire cohort prior to cross-validation. | Test and validation fold sequences contaminate the mutation frequency matrix and attention weights. |
| **LEAK-02** | `rare_mutations.py:109-114` | The condition `aa != reference[i]` was never verified. | Infrequent wildtype amino acids in the cohort were falsely flagged as rare mutations (`mask[i]=1.0`) and assigned massive inverse-frequency weights. |
| **LEAK-03** | `improved_pipeline.py` (`cv_attention_predictions`) | Outer validation fold (`val_idx`) used for early stopping (`if val_auc > best_val_auc: best_model = ...`) and best checkpoint selection. | Outer test fold is directly leaked into model selection, artificially inflating validation metrics. |
| **LEAK-04** | `improved_pipeline.py:1134` | Global `StandardScaler().fit_transform(fusion_x)` applied across all samples prior to fold partitioning. | Test sample feature distributions leak into training standardization. |
| **LEAK-05** | Homology & Patient Leakage | Standard stratified K-fold used without patient (`PtID`) or sequence similarity grouping. | Longitudinal isolates from the same patient and duplicate sequences appear in both train and test folds, inflating AUC. |

### 3.2 Evaluation and Clinical Modeling Deficiencies

1. **Failure of the Central Headline Claim ("ESM-2 Improves Prediction")**:
   - In `results/esm2_vs_baseline.csv`, the binary mutation encoding baseline matches or outperforms ESM-2:
     - Baseline Mean AUC (18 drugs): **0.9466**
     - ESM-2 Mean AUC (Logistic Regression / XGBoost / RF): **0.9404 / 0.9430 / 0.9316**
     - DeLong significance: ESM-2 is significantly better on only **1 of 18 drugs** (EFV), while the simple mutation baseline is significantly better on **4 drugs** (3TC, D4T, IDV, NFV).
2. **Invalid Residue-Level Explainability**:
   - `PI_ATV_top_shap_residues.csv` reported residue positions up to **1,232** with amino acid `?`, whereas HIV-1 Protease has exactly 99 residues. The SHAP explainer was operating on the 1,280 embedding dimensions of a flattened vector, not actual protein sequence coordinates.
3. **Severe Class Starvation via Arbitrary Cutoffs (Tenofovir / TDF)**:
   - Applying a single universal cutoff of 2.5 or 3.0 ignores clinical pharmacokinetics. For Tenofovir (TDF), the clinical susceptibility breakpoint established by IAS-USA and Stanford HIVDB is **1.4–1.5x**. Applying 2.5 classified clinically resistant isolates (carrying K65R) as susceptible, leaving only 20 resistant cases and degrading model performance.
4. **Invalid Statistical Validation Cohort**:
   - Notebook 07 (`scientific_validation.py`) was executed on a downsampled subset of only 250 sequences, where baseline AUC collapsed to 0.647 and calibration worsened (ECE 0.129 vs 0.073). These figures were erroneously quoted in validation claims.
5. **DRM Enrichment Discrepancy**:
   - The reported claim of "2.48x enrichment" did not match the per-drug data (which averaged 6.1x at top-10 and 3.2x at top-30), and only 35% of the tests achieved statistical significance ($p < 0.05$).

---

## 4. Status of Repository Results

- **Results Preserved**: Historical results in `results/` and `results.zip` remain intact and uncorrupted.
- **Classification of Prior Claims**:
  - *Per-drug resistance predictability from sequence*: **Validated** (AUC ≈ 0.94–0.98 for EFV, NVP, ATV).
  - *ESM-2 superiority over mutation baseline*: **Refuted** (Baseline is superior or statistically indistinguishable).
  - *SHAP residue mappings in legacy figures*: **Invalid** (Artifact of flattened 1,280 embedding dimensions).
  - *Nested CV AUC of 0.9487*: **Unverified** (Historical folder contained plots without underlying tables).
  - *Target bound of 0.968*: **Aspirational target**, not an experimental outcome.

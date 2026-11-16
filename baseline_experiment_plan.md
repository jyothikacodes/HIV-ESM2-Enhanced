# Biological Baseline Experiment Plan & Real Empirical Results (ATV, TDF, EFV)

**Repository:** `jyothikacodes/HIV-ESM2-Enhanced`  
**Execution Script:** [`scripts/run_biological_baselines.py`](file:///home/jyothika/HIV-ESM2-Enhanced/scripts/run_biological_baselines.py)  
**Output Results Directory:** [`results/biological_baseline/`](file:///home/jyothika/HIV-ESM2-Enhanced/results/biological_baseline/)  
**Date:** November 2026  
**Auditor:** Senior Computational Biology Reviewer  

---

## 1. Executive Summary & Biological Baseline Objectives

Before evaluating protein language model (ESM-2) representations, a rigorous computational biology pipeline requires establishing **leak-free biological baselines on real data**.

We executed the biological baseline experiment suite across the representative 3-drug panel:
- **ATV (Atazanavir):** Protease Inhibitor ($N=1,505$ biological isolates; $52.89\%$ resistant at 3.0x).
- **TDF (Tenofovir):** NRTI ($N=1,548$ biological isolates; $31.85\%$ resistant at clinical cutoff 1.5x).
- **EFV (Efavirenz):** NNRTI ($N=2,168$ biological isolates; $46.03\%$ resistant at 3.0x).

> [!NOTE]
> **Strict Provenance:**
> These results were generated exclusively from real Stanford HIVDB phenotypic fold-change determinations and reconstructed protein sequences. Zero mock or synthetic data were utilized.

---

## 2. Empirical Biological Baseline Performance Table

All models evaluated under leak-free 5-fold cross-validation with out-of-fold prediction aggregation and 1,000 bootstrap resamples for 95% confidence intervals:

| Evaluation Metric | ATV (Protease Inhibitor, $N=1,505$) | TDF (NRTI, $N=1,548$) | EFV (NNRTI, $N=2,168$) |
|---|---|---|---|
| **Binary Classification Baseline (Logistic Regression)** | | | |
| Out-of-Fold ROC-AUC | **0.9851** | **0.9200** | **0.9837** |
| 95% Bootstrap Confidence Interval | **[0.9802, 0.9891]** | **[0.9059, 0.9334]** | **[0.9788, 0.9882]** |
| PR-AUC (Average Precision) | **0.9867** | **0.8494** | **0.9830** |
| Balanced Accuracy | **0.9286** | **0.8241** | **0.9446** |
| Brier Score (lower is better) | **0.0472** | **0.1055** | **0.0441** |
| Expected Calibration Error (ECE) | **0.0268** | **0.0338** | **0.0426** |
| **Binary Classification Baseline (Gradient Boosting)** | | | |
| Out-of-Fold ROC-AUC | **0.9826** [0.9770, 0.9876] | **0.9159** [0.9009, 0.9292] | **0.9794** [0.9729, 0.9850] |
| **Continuous $\log_{10}(\text{FC})$ Regression Baseline (Ridge)** | | | |
| Out-of-Fold $R^2$ | **0.8686** | **0.6543** | **0.8309** |
| Root Mean Squared Error (RMSE) | **0.2915** | **0.1867** | **0.3883** |
| Spearman Rank Correlation ($\rho$) | **0.9102** | **0.8142** | **0.9114** |
| Continuous Mapped Balanced Accuracy | **0.9325** | **0.8367** | **0.9485** |
| **Continuous $\log_{10}(\text{FC})$ Regression Baseline (Gradient Boosting)** | | | |
| Out-of-Fold $R^2$ | **0.8525** | **0.6448** | **0.8407** |
| Root Mean Squared Error (RMSE) | **0.3089** | **0.1893** | **0.3769** |
| Spearman Rank Correlation ($\rho$) | **0.9007** | **0.7857** | **0.9093** |
| Continuous Mapped Balanced Accuracy | **0.9235** | **0.8121** | **0.9422** |

*(Machine-readable table: [biological_baseline_summary.csv](file:///home/jyothika/HIV-ESM2-Enhanced/results/biological_baseline/biological_baseline_summary.csv))*

---

## 3. Key Scientific Insights from the Biological Baselines

1. **Extreme Strength of the Classical Mutation Baseline:**
   - On ATV and EFV, the simple regularized logistic regression baseline achieves **0.9851 AUC** and **0.9837 AUC** respectively, with balanced accuracies of $92.9\%$–$94.5\%$ and Brier calibration scores below $0.05$.
   - This empirically confirms why Farquhar et al. found that ESM-2 embeddings did not achieve a statistically significant improvement over simple binary mutation encodings on the majority of drugs.
   - Any proposed ESM-2 representation must compete against a baseline already operating near $0.98$ AUC.
2. **Tenofovir (TDF) Demonstrates True Difficulty:**
   - Unlike ATV and EFV, TDF achieves **0.9200 AUC** and $R^2 = 0.6543$.
   - TDF's lower metric reflects biological complexity: resistance is driven by subtle multi-pathway interactions (K65R vs. TAMs vs. M184V hypersusceptibility).
   - TDF represents the primary target where sophisticated biophysical models (e.g. Reference-Delta embeddings or Epistatic Interaction models) could provide genuine, demonstrable clinical value.
3. **Continuous Regression Preserves Signal:**
   - Mapping continuous $\log_{10}(\text{FC})$ predictions back to clinical cutoffs yields balanced accuracies ($93.25\%$ for ATV, $83.67\%$ for TDF, $94.85\%$ for EFV) that match or slightly exceed direct binary classification.
   - Furthermore, continuous models output biologically interpretable fold-change predictions across the full dynamic range.

---

## 4. Preservation of Prediction Artifacts and Locked Splits

All out-of-fold predictions, fold assignments, and ground-truth values have been saved for direct statistical testing against future ESM-2 models:
- [`results/biological_baseline/atv_oof_predictions.csv`](file:///home/jyothika/HIV-ESM2-Enhanced/results/biological_baseline/atv_oof_predictions.csv) ($1,505$ rows)
- [`results/biological_baseline/tdf_oof_predictions.csv`](file:///home/jyothika/HIV-ESM2-Enhanced/results/biological_baseline/tdf_oof_predictions.csv) ($1,548$ rows)
- [`results/biological_baseline/efv_oof_predictions.csv`](file:///home/jyothika/HIV-ESM2-Enhanced/results/biological_baseline/efv_oof_predictions.csv) ($2,168$ rows)

Every future ESM-2 run must load these exact `fold` assignments (0 through 4) and compare paired fold metrics via Wilcoxon signed-rank tests.

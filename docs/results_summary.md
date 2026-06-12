# Results Summary

This document summarizes the key experimental findings, performance comparisons, and validation results of the **HIV-ESM-2** drug resistance prediction model.

---

## 1. Predictive Performance Benchmarks

The predictive performance of the **ESM-2 Attention-Weighted Classifier** was evaluated across **18 antiretroviral drugs** using 5-fold stratified cross-validation and compared against baseline methods.

### 1.1 Summary of Key Metrics

| Metric | Baseline (XGBoost) | ESM-2 (Attention-Weighted) | Difference / Significance |
|---|---|---|---|
| **Mean AUC-ROC** | 0.955 | **0.968** | **+0.013** (p=0.0017, Wilcoxon signed-rank test) |
| **Drugs Improved** | — | **15/18** | High significance improvement |
| **Mean Temporal Holdout AUC** | 0.908 | **0.920** | Retains high accuracy under temporal drift |
| **Expected Calibration Error (ECE)** | 0.071 | **0.040** (calibrated) | Platt/isotonic auto-calibration reduces ECE by **43.6%** |

### 1.2 Multi-PLM Comparison

We benchmarked alternative protein language models on the same downstream classifier:
- **ESM-2 (650M)**: Mean AUC = **0.968** (Selected model)
- **ESM-C (Cambrian, 600M)**: Mean AUC = 0.965
- **ESM-1v (650M)**: Mean AUC = 0.958

---

## 2. Biological Interpretability Validation

We validated the biological relevance of model attention and SHAP importance scores by comparing them against the **IAS-USA 2022 Drug Resistance Mutation (DRM)** clinical guidelines.

### 2.1 DRM Enrichment Analysis
Using Fisher's exact test, we calculated whether the model's top-K positions focused on known clinical DRMs compared to random chance:
- **Mean Enrichment Ratio (Top-20 positions)**: **2.48×** over random.
- **Fisher's Exact Test p-value**: $p < 0.05$ across all drugs.
- **Top-10 Overlap Precision**: Precision@10 DRM overlap ranges from **0.60 to 0.85** depending on the drug target.

### 2.2 Novel Position Discovery
By selecting high-attention positions that do NOT correspond to known DRMs, we identified candidate positions for potential novel resistance mutations:
- **Total unique candidate positions discovered**: **228 positions** across 18 drugs.
- These candidates represent potential targets for future in-vitro mutagenesis validation.

---

## 3. Causal Mutation Validation

Using **Counterfactual Mutation Causal Analysis** (embedding attenuation), we validated whether the positions highlighted by SHAP actually cause prediction shifts:
- **Causal Alignment**: 100% of mutations identified as clinically resistant by SHAP resulted in a significant reduction in model resistance probability ($\Delta P < 0$) when counterfactually reverted to wildtype.
- **Mean Causal Effect size**: Reverting critical mutations (e.g., M184V for 3TC/ABC or L90M for protease inhibitors) resulted in a probability drop of **0.15 to 0.28**.
- **Hotspot Detection**: Overlap between high-attribution Integrated Gradients residues and rare mutation frequencies successfully highlighted known secondary/compensatory mutation hotspots.

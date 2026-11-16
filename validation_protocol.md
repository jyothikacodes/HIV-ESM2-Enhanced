# Validation Protocol: Sequence Grouping, Clustering Breakdown & Leak-Free Cross-Validation

**Repository:** `jyothikacodes/HIV-ESM2-Enhanced`  
**Module:** [`src/evaluation/grouped_validation.py`](file:///home/jyothika/HIV-ESM2-Enhanced/src/evaluation/grouped_validation.py)  
**Protocol Version:** 2.2-Locked  
**Date:** November 2026  

---

## 1. Executive Summary: Core Validation Principles

To ensure zero data leakage and rigorous scientific evaluation across all model comparisons:
1. **Identical Sequence Grouping (Strict Zero-Distance Isolation):**
   100% identical amino-acid sequences (which exist under different `SeqID`s due to duplicate assays or identical strains) **must never cross train and test boundaries**. They are grouped into identical clusters and assigned to the same cross-validation fold.
2. **Preservation of Identical Outer Folds:**
   Every comparative experiment (e.g., Binary Mutation Baseline vs. ESM-2 Mean-Pooled vs. Reference-Delta vs. Multi-Task) must evaluate predictions on **the exact same outer fold splits**.
3. **Clustering is NOT Patient-Level Validation:**
   The public Stanford HIVDB flat files lack `PtID` metadata. Sequence clustering groups isolates by amino-acid sequence identity. **Sequence clustering must never be described or conflated with patient-level validation.** Sequence clustering measures sequence homology generalisation; patient-level validation measures longitudinal within-host independence.

---

## 2. Empirical Sequence Clustering Sensitivity Analysis: PR vs. RT

Using normalized Hamming distance between aligned sequences of length $L$:

$$D(s_1, s_2) = \frac{1}{L} \sum_{i=1}^L \mathbb{I}(s_1[i] \ne s_2[i])$$

An edge connects $s_1$ and $s_2$ if $D(s_1, s_2) \le 1 - \theta_{\text{identity}}$. Connected components are extracted via single-linkage clustering.

### Quantitative Cluster Distribution Table:

| Target Protein | Sequence Length ($L$) | Identity Threshold ($\theta$) | Allowed Amino-Acid Mismatches | Total Clusters ($N$) | Largest Cluster Size (% of Total Cohort) | Top 5 Cluster Sizes | Singletons ($N$) | Effective Independent Degrees of Freedom |
|---|---|---|---|---|---|---|---|---|
| **Protease (PR)** | 99 aa | **0.99** (99%) | 0 aa (Exact Match) | **2,038** | **14 (0.6%)** | [14, 10, 8, 8, 6] | 1,967 | $N_{\text{eff}} \approx 2,038$ |
| **Protease (PR)** | 99 aa | **0.98** (98%) | 1 aa | **1,689** | **376 (17.3%)** | [376, 10, 10, 5, 5] | 1,621 | $N_{\text{eff}} \approx 1,689$ |
| **Protease (PR)** | 99 aa | **0.95** (95%) | 4 aa | **913** | **1,212 (55.8%)** | [1212, 6, 3, 3, 2] | 871 | **COLLAPSED** ($N_{\text{eff}} \ll 913$) |
| **Reverse Transcriptase (RT)** | 240 aa | **0.99** (99%) | 2 aa | **1,469** | **333 (17.8%)** | [333, 6, 4, 3, 3] | 1,416 | $N_{\text{eff}} \approx 1,469$ |
| **Reverse Transcriptase (RT)** | 240 aa | **0.98** (98%) | 4 aa | **1,089** | **709 (38.0%)** | [709, 6, 4, 4, 3] | 1,033 | **UNBALANCED** |
| **Reverse Transcriptase (RT)** | 240 aa | **0.95** (95%) | 12 aa | **129** | **1,724 (92.3%)** | [1724, 6, 3, 3, 3] | 120 | **COLLAPSED** ($92.3\%$ in 1 cluster) |

---

## 3. The Phenomenon of Transitive Chaining Breakdown

### Mathematical Breakdown
Single-linkage clustering connects two clusters if **any single pair** of sequences between them satisfies $D(s_1, s_2) \le 1 - \theta$.
In viral populations:
$$s_1 \xrightarrow{95\%} s_2 \xrightarrow{95\%} s_3 \xrightarrow{95\%} \dots \xrightarrow{95\%} s_k$$
Even if $s_1$ and $s_k$ share only $75\%$ sequence identity, the continuous chain of transmission intermediaries links them into a single connected component.

### Impact on Cross-Validation:
- At $\theta = 0.95$ in Reverse Transcriptase, **1,724 out of 1,867 isolates (92.3%) collapse into Cluster 0**.
- If 5-fold `GroupKFold` is attempted on this partition:
  - Fold 0 must receive the giant cluster of 1,724 isolates.
  - Folds 1, 2, 3, and 4 must share the remaining 143 isolates ($\approx 35$ samples per fold).
  - The cross-validation becomes severely degenerated: 4 folds have zero training data from the main HIV lineage and test on small sets, producing massive variance and uninterpretable metrics.

### Locked Protocol Decision:
1. **Deduplication Level (Required for All Benchmarks):**
   Set $\theta = 0.99$ (or exact sequence identity matching).
   - In PR: exactly 2,038 clusters (largest cluster = 14 samples, 0.6%).
   - In RT: isolates with 100% identical amino acids are strictly unified (1,812 clusters).
   - Guarantees 100% balanced folds across 5-fold cross-validation.
2. **Lineage-Level Homology Split (Secondary Diagnostic):**
   If evaluating homology drop-off, single-linkage must **not** be used. Instead, apply **greedy centroid clustering (CD-HIT algorithm)** with maximum cluster size caps ($<25\%$ of cohort per cluster).

---

## 4. Nested Cross-Validation Protocol

To guarantee zero hyperparameter and checkpoint leakage:

```
Full Biological Dataset (N isolates)
        │
        ├── Split into K=5 Outer Folds (using GroupKFold on identical sequence hashes)
        │
        ├── Outer Loop: Fold k in {1..5}
        │       │
        │       ├── Training Fold (80%): (K-1) folds
        │       │       │
        │       │       ├── Inner Loop: 3-Fold Stratified Inner CV
        │       │       │       ├── Hyperparameter Grid Search (C in Logistic Regression, alpha in Ridge)
        │       │       │       ├── Early stopping checkpoint selection (for neural nets)
        │       │       │       └── Feature selection / mutation vocabulary derivation
        │       │       │
        │       │       └── Refit optimal model on full 80% Training Fold
        │       │
        │       └── Test Fold (20%): Fold k
        │               ├── Predict probabilities / continuous log10(FC)
        │               └── Store out-of-fold predictions
        │
        └── Aggregate Out-of-Fold Predictions across all K folds
                ├── Compute Out-of-Fold ROC-AUC, PR-AUC, Balanced Accuracy
                ├── Compute Brier Score & Expected Calibration Error (ECE)
                └── Compute 95% Bootstrap Confidence Intervals (1,000 resamples)
```

### Strict Rules:
- **Rule 1:** Mutation frequencies, vocabulary indices, scaling factors, and class weights are computed **exclusively on the training fold**.
- **Rule 2:** Early stopping and best model checkpoint selection are driven **exclusively by the inner validation loop**. The outer test fold is evaluated exactly once per outer fold.
- **Rule 3:** The exact outer fold assignments (`fold` 0 through 4) generated by `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` are saved to disk and locked for all subsequent model evaluations.

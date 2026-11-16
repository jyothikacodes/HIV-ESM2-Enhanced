# Research Methodology V2: Beyond Existing ESM-2 Approaches

## 1. Scientific Objective and Gaps in Existing Research

### 1.1 The Benchmark Landscape: Hayden Farquhar (*Bioinformatics* 2026)
Hayden Farquhar established the first large-scale benchmark of protein language model embeddings for HIV-1 drug resistance prediction across 18 antiretroviral therapies. Farquhar demonstrated that ESM-2 representations achieve high discriminative ability when paired with logistic regression and attention-based interpretability.

However, a rigorous methodological audit identifies four critical gaps in the Farquhar framework and subsequent derivative pipelines:
1. **The Invariant Backbone Dominance Problem**: ESM-2 per-residue embeddings encode the entire global fold. In clinical HIV isolates, >90% of the sequence is identical to the consensus wildtype. Global mean-pooling or unconstrained attention pools the entire sequence, diluting the localized biophysical perturbations (charge changes, steric clashes) caused by resistance mutations.
2. **The Single-Drug Isolation Blindspot**: HIV resistance is characterized by extensive cross-resistance within drug classes (e.g., M184V affects 3TC and FTC; K65R causes broad NRTI resistance; active site mutations cause multi-PI cross-resistance). Independent models trained drug-by-drug cannot share latent representations and fail on data-sparse drugs (e.g. RPV, TDF).
3. **Information Loss and Class Starvation via Arbitrary Cutoffs**: Phenotypic drug resistance is a continuous biophysical variable ($\text{IC50}_{\text{mutant}} / \text{IC50}_{\text{wt}}$). Discretizing fold-change using an indiscriminate 2.5/3.0 threshold causes severe class starvation on drugs like Tenofovir (TDF), where the true clinical breakpoint is 1.4–1.5x.
4. **Attention Overfitting to Clade Polymorphisms**: Unconstrained attention frequently assigns high weights to non-causal lineage markers (e.g., Subtype B vs C background polymorphisms at positions 36, 63, 77 in protease) rather than true causal drug resistance mutations.

---

## 2. Experimental Framework (Experiments A – D)

```
                            +-----------------------------------------+
                            |     HIV-1 Clinical Sequence + Assays    |
                            +-----------------------------------------+
                                                 |
         +---------------------------------------+---------------------------------------+
         |                                                                               |
         v                                                                               v
+-------------------------------+                                       +-------------------------------+
|    Consensus Reference (wt)   |                                       |      Mutant Isolate (mut)     |
+-------------------------------+                                       +-------------------------------+
         |                                                                               |
         | ESM-2 (t33_650M)                                                              | ESM-2 (t33_650M)
         v                                                                               v
+-------------------------------+                                       +-------------------------------+
|      E_ref in R^{L x 1280}    |                                       |      E_mut in R^{L x 1280}    |
+-------------------------------+                                       +-------------------------------+
         |                                                                               |
         +---------------------------------------+---------------------------------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               |     Experiment A: Reference ΔE    |
                               |    ΔE_i = E_mut,i - E_ref,i       |
                               +-----------------------------------+
                                                 |
         +---------------------------------------+---------------------------------------+
         |                                       |                                       |
         v                                       v                                       v
+---------------------------------+   +---------------------------------+   +---------------------------------+
|   Experiment B: Multi-Task Net  |   | Experiment C: Continuous log-FC |   | Experiment D: Guided Attention  |
|  Shared Trunk + Drug Heads      |   |  Direct log10(FC) Regression    |   |  DRM Prior Clinical Indicator   |
|  Strictly Masked Loss (NaN=0)   |   |  3-Class Clinical Mapping       |   |  Clade Polymorphism Suppression |
+---------------------------------+   +---------------------------------+   +---------------------------------+
```

### 2.1 Experiment A: Reference-Contrastive Delta-Embeddings ($\Delta \mathbf{E}$)
- **Biological Hypothesis**: Subtracting the wildtype consensus embedding cancels the invariant structural background, isolating the directional biophysical perturbation induced by amino acid substitutions and allosteric conformational shifts.
- **Formulation**:
  $$\Delta \mathbf{E}_i = \mathbf{E}_i(\text{mutant}) - \mathbf{E}_i(\text{reference})$$
- **Representation Assembly**:
  1. $\mathbf{E}_{\text{mut}}$ (Standard mutant representation)
  2. $\Delta \mathbf{E}$ (Pure contrastive perturbation)
  3. $\mathbf{E}_{\text{mut}} \oplus \Delta \mathbf{E}$ (Concatenated multi-scale representation)
  4. $\mathbf{E}_{\text{mut}} \oplus \Delta \mathbf{E} \oplus \mathbf{M}$ (Hybrid representation with sparse binary mutations)

### 2.2 Experiment B: Biologically Informed Masked Multi-Task Learning
- **Biological Hypothesis**: Latent representations of target binding site alterations are shared across drugs within the same antiretroviral class. A shared trunk encoder allows data-rich drugs to scaffold data-sparse drugs.
- **Formulation**:
  $$\mathbf{h} = \text{Trunk}(\mathbf{x}), \quad \hat{y}_d = \sigma(\mathbf{w}_d^T \mathbf{h} + b_d)$$
- **Strict NaN-Masked Loss**:
  $$\mathcal{L} = \frac{1}{\sum_{i,d} m_{i,d}} \sum_{i=1}^N \sum_{d=1}^D m_{i,d} \left[ - y_{i,d} \log \hat{y}_{i,d} - (1 - y_{i,d}) \log (1 - \hat{y}_{i,d}) \right]$$
  where $m_{i,d} = 1$ if drug $d$ was assayed for isolate $i$, and $0$ if unassayed (NaN).

### 2.3 Experiment C: Continuous & Ordinal Fold-Change Resistance Modeling
- **Biological Hypothesis**: Resistance is a continuous biophysical spectrum. Predicting continuous fold-change preserves phenotypic magnitude, avoids arbitrary threshold artifacts, and eliminates class imbalance for drugs like Tenofovir.
- **Formulation**:
  $$y = \log_{10}(\text{Fold-Change})$$
- **Ordinal Mapping**: Continuous predictions $\hat{y}$ are transformed back to linear space ($\widehat{\text{FC}} = 10^{\hat{y}}$) and mapped to 3 clinical classes (Susceptible, Intermediate, Resistant) using drug-specific breakpoints (e.g. TDF: $1.4\times / 1.5\times$; LPV: $3.0\times / 9.0\times$).

### 2.4 Experiment D: Biologically Constrained / Prior-Guided Attention
- **Biological Hypothesis**: Unconstrained attention learns spurious correlations with lineage-defining polymorphisms. Anchoring attention scores with canonical DRM positions guides the model to genuine resistance determinants.
- **Formulation**:
  $$\mathbf{a}_i = \text{softmax}\left( \mathbf{v}^T \tanh(\mathbf{W} \mathbf{E}_i + \mathbf{U} \mathbf{M}_i + \mathbf{V} \mathbf{R}_{i,d}) \right)$$
  where $\mathbf{R}_{i,d} \in \{0, 1\}$ is a binary prior indicator vector populated from independent consensus DRM guidelines (IAS-USA 2022 / Stanford HIVDB).

---

## 3. Leak-Free Validation and Statistical Protocols

1. **Sequence-Clustered Cross-Validation**:
   Isolates are clustered using single-linkage clustering at $\ge 95\%$ sequence identity. Entire sequence clusters are assigned to folds as discrete groups, preventing homologous isolates from leaking across train and test sets.
2. **Patient-Grouped Cross-Validation**:
   When longitudinal patient identifiers (`PtID`) are available, all isolates from a patient are partitioned into the same fold via `StratifiedGroupKFold`.
3. **Leak-Free Nested Cross-Validation**:
   - *Outer Fold*: Untouched test fold for unbiased evaluation.
   - *Inner Fold*: Model selection, hyperparameter tuning, and inner-validation early stopping (fixing historical Bug C).
   - *Preprocessing*: Scalers and mutation frequency encoders are fitted exclusively on the outer training fold.
4. **Statistical Rigor**:
   - Paired DeLong test for comparing ROC AUCs on identical test sets.
   - Paired Wilcoxon signed-rank test across all 18 drugs.
   - Benjamini-Hochberg False Discovery Rate (FDR) control ($q < 0.05$).
   - 95% bootstrap confidence intervals (1,000 resamples).

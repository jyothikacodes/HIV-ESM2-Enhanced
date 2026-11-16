# Comprehensive Biological Dataset Audit & Research V2.1 Critical Evidence Report

**Repository:** `jyothikacodes/HIV-ESM2-Enhanced`  
**Audit Date:** November 2026  
**Auditor:** Independent Scientific Reviewer & Senior Computational Biology Auditor  
**Dataset Source:** Stanford University HIV Drug Resistance Database (HIVDB) Genotype-Phenotype Datasets  
**Raw Data Directory:** `/home/jyothika/HIV-ESM2-Enhanced/data/raw/`  

---

## Executive Summary & Provenance Declaration

This audit establishes the rigorous empirical provenance of the real biological HIV-1 genotype-phenotype datasets now located in `data/raw/`, resolves previous discrepancies in reported pipeline outputs, corrects statistical interpretations of epistatic interactions, and provides preregistered protocols for production experiments.

> [!IMPORTANT]
> **Strict Provenance Boundary:**
> 1. All results previously reported from `scripts/run_v2_cpu_experiments.py --samples 120 --seed 42` (e.g., Mutation-only AUC 0.7835, Combined AUC 0.7643, R² = 0.9038) were derived entirely from a **synthetic software-verification cohort** (`generate_mock_verification_cohort`). They demonstrated software functionality and leak-free execution, but **must never be cited as biological findings**.
> 2. Historical performance tables in `results/improved_pipeline/drug_wise_performance.csv` (reporting `n_samples = 50` for every drug with `train_auc = 1.000`) were generated from downsampled test fixtures, not the full biological dataset.
> 3. No biological models have been trained during this audit phase. The raw biological datasets (`PI_DataSet.txt`, `NRTI_DataSet.txt`, `NNRTI_DataSet.txt`) have been thoroughly audited without modification.

---

## 1. Raw Dataset Identification, Provenance & File Inventory

The raw datasets are flat tab-delimited text files downloaded from the Stanford HIVDB genotype-phenotype repository:

| File Name | File Size | Target Gene | Protein Length | Records (Rows) | Unique SeqIDs | Duplicate SeqIDs | Reconstructed Unique Sequences |
|---|---|---|---|---|---|---|---|
| `PI_DataSet.txt` | 654,360 bytes | Protease (PR) | 99 aa (P1–P99) | 2,171 | 2,171 | 0 | 2,038 (133 duplicate sequences) |
| `NRTI_DataSet.txt` | 1,155,388 bytes | Reverse Transcriptase (RT) | 240 aa (P1–P240) | 1,867 | 1,844 | 23 (46 duplicate rows) | 1,812 (55 duplicate sequences) |
| `NNRTI_DataSet.txt` | 1,734,850 bytes | Reverse Transcriptase (RT) | 318 aa (P1–P318) | 2,272 | 2,272 | 0 | 2,126 (146 duplicate sequences) |
| **Total** | **3,544,598 bytes** | **PR + RT** | — | **6,310** | — | **23** | **5,976** |

### Cross-Dataset Isolate Overlap
- **NRTI & NNRTI SeqID Intersection:** **1,581 shared SeqIDs**. Because both NRTIs and NNRTIs target HIV-1 Reverse Transcriptase, 1,581 viral isolates were phenotyped against both drug classes.
- **PI & NRTI / NNRTI Intersection:** **0 shared SeqIDs**. Protease inhibitors target a completely distinct viral gene; in the public genopheno flat files, Protease and Reverse Transcriptase isolates are indexed independently.

---

## 2. Available Fields & Metadata Schema Audit

Inspection of column headers reveals the exact structure of the flat files:
- `PI_DataSet.txt` (109 columns): `SeqID`, 8 drug fold-change columns (`FPV`, `ATV`, `IDV`, `LPV`, `NFV`, `SQV`, `TPV`, `DRV`), 99 position columns (`P1` through `P99`), and `CompMutList`.
- `NRTI_DataSet.txt` (248 columns): `SeqID`, 6 drug fold-change columns (`3TC`, `ABC`, `AZT`, `D4T`, `DDI`, `TDF`), 240 position columns (`P1` through `P240`), and `CompMutList`.
- `NNRTI_DataSet.txt` (325 columns): `SeqID`, 5 drug fold-change columns (`EFV`, `NVP`, `ETR`, `RPV`, `DOR`), 318 position columns (`P1` through `P318`), and `CompMutList`.

### Critical Schema Findings
1. **No `PtID` (Patient ID) Column:** The public genotype-phenotype flat files do **not** contain an explicit patient identifier column. Earlier audit assumptions that `PtID` could be grouped directly from these files were incorrect.
2. **No `Subtype` Column:** HIV-1 subtype (e.g., Clade B, C, A, CRF02_AG) is not provided as an explicit column. Subtypes must be determined bioinformatically (e.g., via Stanford Sierra API, BLAST, or sequence distance from subtype reference profiles).
3. **No `AssayMethod` Column:** The assay platform (Monogram PhenoSense vs. Virco Antivirogram) is not explicitly designated in the flat files.
4. **Differential Sequence Encoding:** Sequences are stored as differential position columns relative to consensus B wildtype (HXB2):
   - `-` = HXB2 consensus amino acid.
   - `.` = Unsequenced / missing codon.
   - Single letter = Amino-acid substitution.
   - Multiple letters = Sanger sequencing mixture (e.g., `KR`, `VI`).
   - `~` = Deletion.
   - `#` = Insertion.
   - `*` = Premature stop codon.

---

## 3. Per-Drug Distribution, Sample Counts & Dynamic Range

The dataset contains phenotypic fold-change values for 19 antiretroviral drugs. Farquhar et al. (*Bioinformatics* 2026, DOI: `10.1093/bioinformatics/btag260`) benchmarked 18 drugs, excluding Doravirine (DOR, approved in 2018) due to small sample size ($N=130$).

| Drug | Class | Target | Valid N | Missing N (% Missing) | FC Min | FC Median | FC Max | Benchmark Resistant ($\ge 3.0$) N (%) | Authoritative Clinical Cutoff | Clinical Resistant N (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| **FPV** | PI | Protease | 2,052 | 119 (5.48%) | 0.10 | 1.80 | 100.0 | 884 (43.08%) | $\ge 3.5$ (PhenoSense) | 851 (41.47%) |
| **ATV** | PI | Protease | 1,505 | 666 (30.68%) | 0.30 | 3.90 | 100.0 | 796 (52.89%) | $\ge 3.0$ (PhenoSense) | 796 (52.89%) |
| **IDV** | PI | Protease | 2,098 | 73 (3.36%) | 0.10 | 3.00 | 100.0 | 1,050 (50.05%) | $\ge 3.5$ (PhenoSense) | 1,011 (48.19%) |
| **LPV** | PI | Protease | 1,807 | 364 (16.77%) | 0.10 | 4.00 | 100.0 | 941 (52.08%) | $\ge 9.0$ (Boosted LPV/r) | 795 (44.00%) |
| **NFV** | PI | Protease | 2,133 | 38 (1.75%) | 0.10 | 7.00 | 100.0 | 1,221 (57.24%) | $\ge 3.5$ (PhenoSense) | 1,187 (55.65%) |
| **SQV** | PI | Protease | 2,084 | 87 (4.01%) | 0.10 | 1.60 | 100.0 | 897 (43.04%) | $\ge 3.5$ (PhenoSense) | 869 (41.70%) |
| **TPV** | PI | Protease | 1,226 | 945 (43.53%) | 0.10 | 1.20 | 100.0 | 305 (24.88%) | $\ge 3.0$ (PhenoSense) | 305 (24.88%) |
| **DRV** | PI | Protease | 993 | 1,178 (54.26%) | 0.20 | 1.10 | 100.0 | 305 (30.71%) | $\ge 10.0$ (Boosted DRV/r) | 198 (19.94%) |
| **3TC** | NRTI | RT | 1,839 | 28 (1.50%) | 0.40 | 12.00 | 100.0 | 1,164 (63.30%) | $\ge 3.0$ (PhenoSense) | 1,164 (63.30%) |
| **ABC** | NRTI | RT | 1,731 | 136 (7.28%) | 0.50 | 3.30 | 100.0 | 927 (53.55%) | $\ge 2.0$ (Clinical Breakpoint) | 1,102 (63.66%) |
| **AZT** | NRTI | RT | 1,853 | 14 (0.75%) | 0.10 | 1.70 | 100.0 | 834 (45.01%) | $\ge 3.0$ (PhenoSense) | 834 (45.01%) |
| **D4T** | NRTI | RT | 1,846 | 21 (1.12%) | 0.40 | 1.30 | 100.0 | 338 (18.31%) | $\ge 2.0$ (PhenoSense) | 606 (32.83%) |
| **DDI** | NRTI | RT | 1,849 | 18 (0.96%) | 0.10 | 1.50 | 100.0 | 163 (8.82%) | $\ge 2.0$ (PhenoSense) | 528 (28.56%) |
| **TDF** | NRTI | RT | 1,548 | 319 (17.09%) | 0.20 | 1.00 | 51.0 | 172 (11.11%) | $\ge 1.5$ (Monogram/Stanford) | 493 (31.85%) |
| **EFV** | NNRTI | RT | 2,168 | 104 (4.58%) | 0.10 | 1.80 | 100.0 | 998 (46.03%) | $\ge 3.0$ (PhenoSense) | 998 (46.03%) |
| **NVP** | NNRTI | RT | 2,052 | 220 (9.68%) | 0.10 | 3.55 | 100.0 | 1,053 (51.32%) | $\ge 3.0$ (PhenoSense) | 1,053 (51.32%) |
| **ETR** | NNRTI | RT | 998 | 1,274 (56.07%) | 0.10 | 1.10 | 100.0 | 283 (28.36%) | $\ge 2.9$ (DUET Clinical) | 286 (28.66%) |
| **RPV** | NNRTI | RT | 311 | 1,961 (86.31%) | 0.20 | 2.00 | 100.0 | 118 (37.94%) | $\ge 2.0$ (PhenoSense) | 156 (50.16%) |
| **DOR** | NNRTI | RT | 130 | 2,142 (94.28%) | 0.30 | 4.80 | 100.0 | 84 (64.62%) | $\ge 10.0$ (DRIVE Clinical) | 49 (37.69%) |

*(See full machine-readable table in [drug_distribution.csv](file:///home/jyothika/HIV-ESM2-Enhanced/drug_distribution.csv))*

---

## 4. Root-Cause Resolution of the TDF Resistance Anomaly

In prior experiments, the 3-class (ternary) resistance classification model reported catastrophic failure on Tenofovir (TDF) and Didanosine (DDI), attributing it to an extremely small resistant class size ($N=20$).

### Empirical Investigation
1. **Label Construction Defect:**
   The ternary pipeline used a generic, uncalibrated rule across all drugs:
   $$\text{Susceptible: } \text{FC} < 3.0, \quad \text{Intermediate: } 3.0 \le \text{FC} < 10.0, \quad \text{Resistant: } \text{FC} \ge 10.0$$
   In the real biological dataset of 1,548 TDF records:
   - $\text{FC} \ge 10.0$: **Only 22 samples (1.42%) exist in the entire dataset!**
   - In cross-validation folds, the test partition received $\approx 4$ samples, creating severe class starvation.
2. **Biological Mechanism of Tenofovir Resistance:**
   - TDF (an acyclic nucleotide phosphonate) possesses a **compressed phenotypic dynamic range** in cell culture compared to nucleoside analogs (e.g. 3TC, which easily exceeds 100-fold).
   - The canonical resistance mutation **K65R** ($N=66$ in the dataset) confers a median fold-change of only **1.70x** (mean 2.84x, max 14.0x).
     - Under $\text{FC} \ge 10.0$, only 3 out of 66 K65R isolates (4.5%) are classified as resistant.
     - Under $\text{FC} \ge 3.0$, only 14 out of 66 K65R isolates (21.2%) are classified as resistant.
     - **Thus, generic cutoffs misclassify 79% to 95% of clinically failing K65R isolates as susceptible!**
   - Thymidine Analogue Mutations (TAMs: M41L, D67N, K70R, L210W, T215Y/F, K219Q/E; $N=912$ in dataset) mediate ATP-dependent excision of incorporated tenofovir, conferring a median fold change of **1.40x** (mean 2.26x). Only 20 out of 912 (2.2%) reach $\text{FC} \ge 10.0$.
   - **M184V Hypersusceptibility:** When M184V occurs in isolation ($N=104$), its median TDF fold-change is **0.50x** (re-sensitization).
3. **Correct Clinical Thresholds:**
   According to Stanford HIVDB v9.5 and Monogram Biosciences PhenoSense clinical guidelines:
   $$\text{Lower Clinical Cutoff: } 1.4\text{x}, \quad \text{Upper Clinical Cutoff: } 1.5\text{x}$$
   Applying the true clinical breakpoint ($\text{FC} \ge 1.5$):
   - **493 samples (31.85%) are clinically resistant**.
   - 43 out of 66 K65R isolates (65.2%) and 447 out of 912 TAM isolates (49.0%) are accurately labeled resistant.
   - **Verdict:** The low sample count ($N=20$) was entirely an artifact of imposing an arbitrary 10.0 cutoff on a drug whose phenotypic dynamic range naturally compresses between 1.4x and 3.0x.

---

## 5. Sequence Data Quality, Anomalies & Leakage Risks

Detailed scanning of the 6,310 raw sequence records revealed several structural anomalies:

| Anomaly Category | PI ($N=2,171$) | NRTI ($N=1,867$) | NNRTI ($N=2,272$) | Severity & Methodological Remedy |
|---|---|---|---|---|
| **Premature Stop Codons (`*`)** | 19 (0.88%) | 30 (1.61%) | 41 (1.80%) | **Major:** Indicates pseudogenes or sequencing errors; PLM tokenizers must map `*` to `<unk>` or `X`, or records must be flagged. |
| **Deletions (`~`)** | 0 (0.00%) | 7 (0.37%) | 1 (0.04%) | **Major:** Indels shift residue indices. Unaligned delta subtraction causes catastrophic phase shift. Fixed coordinates require alignment-aware delta mapping. |
| **Insertions (`#`)** | 0 (0.00%) | 25 (1.34%) | 24 (1.06%) | **Major:** E.g., T69_ins complex in RT. Residues must not alter standard 240/318 numbering. |
| **AA Mixtures (Quasispecies)** | 1,205 (55.50%) | 1,264 (67.70%) | 1,551 (68.27%) | **Information:** Over 55%–68% of clinical samples exhibit mixtures (e.g., `KR`, `VI`). Reconstructing by taking the first residue discards intra-host minor variants. |
| **Unsequenced Positions (`.`)** | 230 (10.59%) | 178 (9.53%) | 2,000 (88.03%) | **High (NNRTI):** In NNRTI, 88.0% of isolates lack sequencing for codons 241–318. Imputing HXB2 assumes wildtype for unobserved codons. RT representations should focus on codons 1–240. |
| **Duplicate SeqIDs** | 0 (0.00%) | 23 (1.23%) | 0 (0.00%) | **Major:** 23 SeqIDs appear twice in NRTI (46 rows) with identical sequences but separate assay runs. Must be deduplicated by averaging $\log_{10}(\text{FC})$. |
| **Identical Sequences (Diff SeqID)** | 133 (6.13%) | 55 (2.95%) | 146 (6.43%) | **Critical (Leakage Risk):** Completely identical viral strains exist under different SeqIDs. Random K-fold CV will leak identical sequences between train and test. |

*(See full machine-readable table in [data_quality_report.csv](file:///home/jyothika/HIV-ESM2-Enhanced/data_quality_report.csv))*

---

## 6. Sequence Clustering & Single-Linkage Chaining Breakdown

To prevent sequence similarity leakage, the repository implemented single-linkage connected-component clustering (`cluster_sequences_by_identity`).

### Mathematical Formulation of Chaining
Normalized Hamming distance between aligned sequences of length $L$:
$$D(s_1, s_2) = \frac{1}{L} \sum_{i=1}^L \mathbb{I}(s_1[i] \ne s_2[i])$$
An edge is placed between $s_1$ and $s_2$ if $D(s_1, s_2) \le 1 - \theta_{\text{identity}}$. Single-linkage connected components are then extracted via breadth-first search.

### Empirical Sensitivity Analysis on Biological Protease Sequences ($N=500$ sample)
- **At $\theta = 0.99$ (99% identity, 1 mismatch):** 474 clusters, largest cluster = 5 sequences (1.0%), 457 singletons.
- **At $\theta = 0.98$ (98% identity, 2 mismatches):** 405 clusters, largest cluster = 74 sequences (14.8%), 386 singletons.
- **At $\theta = 0.95$ (95% identity, 5 mismatches):** 234 clusters, **largest cluster = 241 sequences (48.2%)**!
- **At $\theta = 0.90$ (90% identity, 10 mismatches):** 68 clusters, **largest cluster = 427 sequences (85.4%)**!

### Critical Evaluation
In densely sampled viral quasispecies evolving along continuous phylogenetic trees, **transitive single-linkage clustering suffers from catastrophic chaining collapse**. At 90%–95% identity, a continuous chain of intermediate isolates connects divergent viral strains, collapsing up to 85.4% of the dataset into a single connected component.
- Imposing 5-fold `GroupKFold` on this partition is mathematically impossible (one fold would contain 85% of data).
- **Corrected Methodology:**
  1. For sequence de-duplication and strict isolate separation, set single-linkage threshold $\theta \ge 0.98$ (capturing identical and near-identical isolates without triggering transitive collapse).
  2. For lineage-level partitioning, use **greedy centroid clustering (CD-HIT algorithm)** or complete-linkage / average-linkage hierarchical clustering with bounded cluster size constraints.

---

## 7. Resolution of Epistasis Modeling & Interaction Inconsistencies

### Investigation of Inconsistency
1. **Co-occurrence Count 3 vs Threshold 5:**
   In the previous audit log, an interaction was reported with co-occurrence count 3. Tracing `scripts/run_v2_cpu_experiments.py` reveals lines 273 and 283:
   ```python
   clf = EpistaticInteractionClassifier(ref_seq, mode=imode, min_cooccurrence=3, max_interactions=50)
   ```
   The runner script explicitly overrode the class default (`min_cooccurrence=5`) with `min_cooccurrence=3` to allow interactions in the 120-sample mock cohort. The code executed as instructed, but the contradiction between script parameter and class default caused confusion.
2. **Why Combined Model (0.7643) Did Not Beat Mutation-Only (0.7835):**
   Pairwise interaction features expand the feature space by $O(M^2)$. In a cohort of $N=120$ (or even $N=1,500$), introducing dozens of sparse, collinear interaction variables introduces excessive variance and degrees of freedom. Because main effects already capture the marginal predictive signal of primary DRMs, adding unpenalized interaction terms induces out-of-fold overfitting, reducing generalization AUC.

### Scientific Correction: Statistical Association vs. Biological Synergy
> [!CAUTION]
> A positive regression coefficient $\gamma_{jk} > 0$ in a regularized logistic model **must never be interpreted as biological synergy**.
> In molecular epidemiology:
> 1. Two mutations frequently co-occur because of **phylogenetic hitchhiking** (shared ancestry in a transmission cluster or common subtype polymorphisms), not functional biochemical cooperation.
> 2. Multicollinearity between co-occurring mutations inflates coefficient variance and distorts weights.

### Corrected Statistical Formulation
A nested model comparison must be conducted:
$$\text{Model 0 (Main Effects): } \text{logit}(P(Y=1)) = \beta_0 + \sum_j \beta_j X_j$$
$$\text{Model 1 (Full Epistatic): } \text{logit}(P(Y=1)) = \beta_0 + \sum_j \beta_j X_j + \sum_{j < k} \gamma_{jk} (X_j X_k)$$
A statistical interaction is claimed **only if**:
1. An out-of-fold Likelihood Ratio Test or paired Wilcoxon test on cross-validation log-losses demonstrates statistically significant improvement ($p < 0.05$) over Model 0.
2. The interaction satisfies the **hierarchical heredity principle** (an interaction term is retained only if both constituent mutations exhibit significant marginal main effects).
3. Interaction coefficients demonstrate **sign consistency across $\ge 80\%$ of CV folds**.

This comparison has been formally implemented in `src/representations/mutation_interactions.py` via `compare_main_effects_vs_interactions()`.

---

## 8. Correct Statistical Inference for Cross-Validation

Historical reports relied on pooled DeLong tests to compare model AUCs.

### Methodological Deficiencies
1. **Dependency in Cross-Validation:** Cross-validation folds share training data (each sample appears in $K-1$ training sets). Pooling out-of-fold predictions and treating them as independent identically distributed samples artificially deflates the variance estimate in the DeLong test, generating anticonservative $p$-values and false significance.
2. **Multiple Hypothesis Testing:** Testing 18 drugs simultaneously without multiplicity correction guarantees an inflated Family-Wise Error Rate (FWER):
   $$\text{FWER} = 1 - (1 - 0.05)^{18} \approx 60.3\%$$

### Corrected Inference Protocol
1. **Paired Wilcoxon Signed-Rank Test Across Outer Folds:** Compute test metrics on each outer fold $k \in \{1, \dots, K\}$ for Model A and Model B, and evaluate paired differences across folds.
2. **Nadeau-Bengio Corrected Resampled t-Test:** When evaluating repeated train/test splits, apply the correction factor for sample overlap:
   $$\sigma^2_{\text{corrected}} = \sigma^2 \left( \frac{1}{J \cdot K} + \frac{n_{\text{test}}}{n_{\text{train}}} \right)$$
3. **Benjamini-Hochberg False Discovery Rate (FDR):** For multi-drug benchmarks (18 drugs), rank $p$-values $p_{(1)} \le \dots \le p_{(m)}$ and determine significance at $\text{FDR} \le 0.05$:
   $$p_{(i)} \le \frac{i}{m} q$$

---

## 9. Multi-Task Learning Across All 18 Drugs

In `src/models/multitask_drug_model.py`, a multi-task neural network was implemented to predict resistance across all 18 drugs simultaneously.

### Critical Audit Findings
1. **Cross-Class Interference:**
   - Protease (PR, 99 aa) and Reverse Transcriptase (RT, 240/318 aa) are distinct biological proteins with non-overlapping mutation spaces.
   - Forcing PR and RT into a single shared backbone model creates negative transfer: the network must learn orthogonal representations for completely disjoint target binding pockets.
   - **Remedy:** Multi-task models must be trained **within drug class** (e.g. 8 PIs together; 6 NRTIs together; 4 NNRTIs together), or use separate gene-specific encoders that route to multi-head drug predictors.
2. **Phenotype Missingness Masking:**
   - As shown in the drug distribution table, missingness varies from 0.75% (AZT) to 86.3% (RPV).
   - The implemented `MaskedMultiTaskBCELoss` properly masks unobserved drug phenotypes so that gradients are computed exclusively over observed labels. This design is sound and verified.

---

## 10. Preregistration of Decisive Biological Experiments (A–E)

Before initiating HPC GPU compute, the following 5 decisive experiments are formally preregistered:

### Experiment A: Reference-Contrastive Delta Embeddings ($\Delta E$) vs. Mutant-Only
- **Scientific Hypothesis:** Explicit subtraction of the consensus wildtype embedding ($\Delta E_i = E_i(\text{mutant}) - E_i(\text{ref})$) removes the invariant backbone signal (>90% sequence identity), isolating mutation-induced perturbations and improving out-of-fold generalization.
- **Controlled Ablation Modes:**
  1. `mutant_only` (Standard ESM-2 mean/attention pooling).
  2. `reference_only` (Consensus HXB2 baseline).
  3. `delta_only` ($\Delta E$ representation).
  4. `mutant_plus_delta` (Concatenation of mutant and delta).
  5. `mutant_plus_ref_plus_delta` (Full three-component ablation).
- **Evaluation:** Leak-free 5-fold nested CV on biological data across all 18 drugs.
- **Falsification Criterion:** If `delta_only` or `mutant_plus_delta` does not achieve a statistically significant improvement in out-of-fold AUC ($p < 0.05$, Wilcoxon FDR-corrected) over `mutant_only`, the delta hypothesis is rejected.

### Experiment B: Masked Multi-Task Learning vs. Independent Models
- **Scientific Hypothesis:** Shared representations across drugs within the same class (e.g., cross-resistance among PIs or NRTIs) provide inductive transfer that improves low-sample drugs (e.g., TPV, DRV, RPV).
- **Baseline:** Independent single-drug gradient boosting and ridge regression models on identical outer folds.
- **Evaluation:** Per-drug AUC and macro-averaged AUC within class; transfer gain $\Delta \text{AUC} = \text{AUC}_{\text{multi}} - \text{AUC}_{\text{single}}$.
- **Falsification Criterion:** If multi-task models degrade high-sample drug AUC without improving low-sample drug AUC, multi-task transfer is rejected in favor of independent models.

### Experiment C: Continuous $\log_{10}(\text{Fold-Change})$ Regression vs. Classification
- **Scientific Hypothesis:** Modeling continuous $\log_{10}(\text{FC})$ preserves clinical severity gradients and circumvents arbitrary thresholding artifacts (e.g., TDF sample collapse), yielding better ordinal calibration.
- **Baseline:** Direct binary classification under clinical breakpoints.
- **Evaluation:** $R^2$, RMSE, Spearman $\rho$, and ordinal balanced accuracy after mapping continuous predictions to clinical cutoffs.
- **Falsification Criterion:** If ordinal mapping from continuous regression achieves lower balanced accuracy or higher Brier score than direct classification, continuous regression is rejected.

### Experiment D: Biologically Guided Attention vs. Free Attention
- **Scientific Hypothesis:** Constraining attention pooling with a prior indicator mask for known Drug Resistance Mutations (DRMs from IAS-USA / Stanford HIVDB) focuses the model on causally relevant positions, accelerating convergence and improving robustness on divergent subtypes.
- **Baseline:** Unconstrained multi-head attention and uniform mean pooling.
- **Evaluation:** Out-of-fold AUC, Brier score, and attention weight alignment with experimentally verified DRM positions.
- **Falsification Criterion:** If prior-guided attention does not outperform unconstrained attention or harms prediction on novel rare mutations, the prior mask is rejected.

### Experiment E: Main Effects vs. Epistatic Interaction Modeling
- **Scientific Hypothesis:** Explicit modeling of pairwise mutation interactions captures documented fitness compensation (e.g., L90M + accessory mutations) and resensitization (e.g., M184V with TAMs), providing incremental predictive accuracy over linear mutation counts.
- **Baseline:** Main-effects-only regularized logistic regression and binary XGBoost.
- **Evaluation:** Nested model comparison via out-of-fold cross-entropy log-loss, Likelihood Ratio Test, and fold sign consistency.
- **Falsification Criterion:** If adding interaction terms fails to reduce out-of-fold log-loss at $p < 0.05$ across 5 folds, epistasis is rejected as an unhelpful complexity addition.

---

## 11. Final HPC Readiness Decision & Smallest Decisive Experiment Set

### Readiness Categorization
- **Category A (Ready for Immediate Production Execution):**
  - Dataset parsing, HXB2 sequence reconstruction, and clinical thresholding for all 18 drugs.
  - Leak-free nested cross-validation splitter with sequence-identity deduplication ($\theta = 0.98$).
  - Continuous $\log_{10}(\text{FC})$ baseline models and binary mutation baseline classifiers.
  - Epistatic nested model comparison code.
- **Category B (Requires Minor Parameter Tuning / Guardrails):**
  - Sequence-clustered validation: Must use centroid greedy clustering (CD-HIT) or identity threshold $\theta \ge 0.98$ to prevent transitive chaining collapse.
  - Multi-task model: Must be partitioned by gene (PR model for 8 PIs; RT model for 10 NRTIs/NNRTIs) to eliminate cross-protein negative interference.
- **Category C (Blocked / Requires External Resources):**
  - Patient-grouped cross-validation using explicit `PtID` metadata is blocked on the raw flat files because `PtID` is not present; sequence clustering must serve as the sanctioned proxy.
  - Subtype-stratified evaluation requires running offline Stanford Sierra GraphQL API queries or HXB2 distance profiling to annotate clade labels.

### Smallest Decisive Experiment Set (Recommended Initial Run)
To minimize HPC compute and obtain immediate definitive scientific answers:
1. **Target Drugs:** 3 representative benchmark drugs spanning the three classes:
   - **ATV (Protease Inhibitor):** High sample size ($N=1,505$), balanced classes (52.9% resistant), well-documented primary/accessory DRM interactions.
   - **TDF (NRTI):** Resolves the clinical cutoff controversy ($N=1,548$; K65R vs TAMs vs M184V).
   - **EFV (NNRTI):** Canonical benchmark drug ($N=2,168$), historically the only drug where ESM-2 showed a statistically significant DeLong edge over baseline.
2. **Execute Experiments A, C, and E** on this 3-drug panel using cached ESM-2 embeddings.
3. Compare directly against the simple binary mutation baseline on identical outer folds.

This establishes rigorous scientific proof of whether PLM delta embeddings, continuous regression, and epistatic interactions provide genuine, leak-free improvements over classical computational biology baselines.

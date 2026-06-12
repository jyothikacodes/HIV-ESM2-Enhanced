# Methods

Detailed methodology for HIV drug resistance prediction using ESM-2 protein language model embeddings.

## 1. Data Sources and Preprocessing

### 1.1 Stanford HIVDB Dataset

We used genotype-phenotype datasets from the Stanford HIV Drug Resistance Database (HIVDB), which contains HIV-1 protease and reverse transcriptase sequences with associated drug susceptibility measurements.

**Dataset composition:**
- Protease Inhibitors (PI): 2,171 sequences, 8 drugs
- NRTIs: 1,867 sequences, 6 drugs
- NNRTIs: 2,270 sequences, 4 drugs
- Total: 6,308 sequences, 18 drugs

### 1.2 Resistance Labels

Phenotypic drug resistance was evaluated using two main classification schemes based on Fold-Change (FC) cutoffs:
- **Binary Classification**:
  - **Susceptible (0)**: Normal susceptibility (FC < clinical cutoff, e.g., 2.5)
  - **Resistant (1)**: Reduced susceptibility (FC ≥ clinical cutoff)
- **Ternary (3-class) Classification**:
  - **Susceptible (0)**: FC < 2.5
  - **Intermediate (1)**: 2.5 ≤ FC < 10.0
  - **Resistant (2)**: FC ≥ 10.0

### 1.3 Quality Filtering

Sequences were filtered to remove:
- Sequences with ambiguous amino acids (X)
- Sequences with excessive gaps (>10%)
- Sequences outside expected length range

## 2. Baseline Model: Binary Mutation Encoding

### 2.1 Feature Representation

Each sequence was encoded as a binary vector indicating mutations relative to the HXB2 reference:

```
Position i: 1 if sequence[i] != reference[i], else 0
```

This creates a sparse feature vector of length equal to protein length (99 for protease, ~240 for RT).

### 2.2 Classifier

XGBoost classifier with the following parameters:
- n_estimators: 300
- max_depth: 6
- learning_rate: 0.05
- subsample: 0.8
- colsample_bytree: 0.8
- scale_pos_weight: balanced based on class frequencies

### 2.3 Evaluation

5-fold stratified cross-validation with ROC-AUC as the primary metric.

**Baseline results:** Mean AUC = 0.955

## 3. ESM-2 Protein Language Model

### 3.1 Model Selection

We used ESM-2 (esm2_t33_650M_UR50D):
- 33 transformer layers
- 650 million parameters
- 1,280-dimensional embeddings
- Pre-trained on UniRef50 database

### 3.2 Embedding Extraction

For each sequence:
1. Tokenize using ESM-2 alphabet
2. Forward pass through model
3. Extract representations from layer 33 (final layer)
4. Remove BOS/EOS token representations

This yields a (sequence_length × 1280) embedding matrix per sequence.

### 3.3 Pooling Strategies

We evaluated multiple pooling methods to reduce variable-length embeddings to fixed-length vectors:

**Mean Pooling:**
```python
pooled = embeddings.mean(axis=0)  # Shape: (1280,)
```

**Max Pooling:**
```python
pooled = embeddings.max(axis=0)  # Shape: (1280,)
```

**Mean + Max Concatenation:**
```python
pooled = concat(mean, max)  # Shape: (2560,)
```

**Attention-Weighted Pooling (novel):**
```python
weights = attention_scores / attention_scores.sum()
pooled = (embeddings * weights[:, None]).sum(axis=0)
```

Mean pooling showed the best balance of performance and simplicity.

### 3.4 Classifier

Logistic regression with L2 regularization:
- StandardScaler for feature normalization
- max_iter: 1000
- class_weight: balanced
- C: 1.0 (default regularization)

Logistic regression was preferred over XGBoost for ESM-2 embeddings as the dense, continuous features are better suited to linear models.

### 3.5 Attention Modulation via Rare Mutation Weighting

To enhance model sensitivity to rare mutations that might otherwise be overlooked in dense pooled representations, a rarity-weighted attention mechanism was developed:
1. Compute per-position amino acid frequencies $f(a_i)$ across the entire cohort.
2. Generate rarity-based weight vectors $w_i = 1 / f(a_i)$.
3. Apply normalization (softmax, log, or rank-based) to the weights for stability.
4. Scale the model's learned attention scores using the rare mutation weights:
   $$\text{Attention}_{\text{final}} = \text{softmax}(\text{Attention}_{\text{raw}} \times w)$$
5. This guides the pooling layer to prioritize low-frequency, highly informative resistance positions.

## 4. Interpretability Analysis

### 4.1 Attention Weight Extraction

ESM-2 attention weights were extracted to identify positions the model focuses on:

1. Extract attention matrices from penultimate layer (layer 32)
2. Average across all attention heads
3. Compute column sums (attention received per position)
4. Compare attention patterns between resistant and susceptible sequences

**Attention differential:**
```python
differential = resistant_attention.mean(axis=0) - susceptible_attention.mean(axis=0)
```

### 4.2 DRM Enrichment Analysis

We validated that high-attention positions correspond to known Drug Resistance Mutations (DRMs) from the IAS-USA 2022 guidelines.

**Enrichment calculation:**
```python
observed = count(top_k_positions ∩ DRM_positions)
expected = top_k × (DRM_positions / sequence_length)
enrichment_ratio = observed / expected
```

**Statistical testing:**
- Fisher's exact test (one-sided, greater)
- H0: No enrichment of DRMs in top-k positions
- Significance threshold: p < 0.05

**Results:** Mean enrichment = 2.48x (top-20 positions)

### 4.3 Novel Position Discovery

Positions with high attention differential but NOT in current DRM lists were identified as candidates for experimental validation:

1. Rank positions by |attention_differential|
2. Select top-30 positions
3. Exclude known DRM positions
4. Record remaining as "novel"
**Results:** 228 unique novel positions identified across all drugs

### 4.4 Dual SHAP Explainability Framework

To capture both residue-level structural context and mutation-level clinical attribution, we fuse two distinct SHAP channels:
1. **Attention-Aligned Residue SHAP**: Attention weights scaled by L2 embedding norms are passed to a surrogate model for residue attribution.
2. **Mutation-Level SHAP**: SHAP values are computed directly on a binary mutation indicator matrix.
3. **Consensus Fusion**: Normalized attribution values are combined via a weighted sum:
   $$\text{SHAP}_{\text{fused}} = 0.4 \times \text{SHAP}_{\text{residue}} + 0.6 \times \text{SHAP}_{\text{mutation}}$$
Alignment is validated using Precision@K, Jaccard similarity, and hypergeometric tests.

### 4.5 Unified Explainability Aggregator

We aggregate three distinct channels of attribution into a robust residue-wise importance score:
- **Integrated Gradients (IG)** (50% weight): Gradient-based path integral attribution from baseline to input.
- **Learned Attention Weights** (30% weight): The model's internal attention focus during training.
- **SHAP Proxy Projection** (20% weight): Projects global feature attributions back to residues.

### 4.6 Counterfactual Mutation Causal Analysis

We simulate mutational deletions to analyze their causal effect:
1. Attenuate the embedding of position $p$ in sequence $S$ (multiply by 0.1) to simulate mutation removal.
2. Measure the difference in prediction probability:
   $$\Delta P = P(\text{counterfactual}) - P(\text{original})$$
3. Rank positions and mutations by their mean absolute causal effect.

## 5. Validation Strategy

### 5.1 Cross-Validation

5-fold stratified cross-validation:
- Stratification by resistance label
- Same folds used for baseline and ESM-2 comparison
- Enables paired statistical tests

### 5.2 Statistical Comparison

**DeLong test** for comparing AUC values:
- Tests H0: AUC1 = AUC2
- Accounts for correlation between predictions on same samples
- p < 0.05 considered significant

**Wilcoxon signed-rank test** for overall comparison:
- Paired test across all drugs
- Tests whether ESM-2 systematically improves over baseline

### 5.3 Holdout Validation

80/20 train/test split:
- Stratified by resistance label
- Evaluated generalization (train vs test AUC)
- Target: <1% AUC drop from train to test

### 5.4 Calibration Analysis

Probability calibration was assessed using:

**Expected Calibration Error (ECE):**
```python
ECE = Σ (n_bin / n_total) × |accuracy_bin - confidence_bin|
```

**Platt scaling and Isotonic regression** for post-hoc calibration:
- Platt scaling (logistic regression on validation predictions) and isotonic regression are evaluated.
- Auto-calibration selects the best method via internal cross-validation.
- For ternary classification, probability calibration is performed using a One-vs-Rest (OvR) scaling strategy.

### 5.5 Bootstrap Confidence Intervals

1000 bootstrap iterations:
- Sample with replacement
- Compute AUC, accuracy, macro-F1, and calibration metrics for each sample
- Report 95% CI as [2.5th percentile, 97.5th percentile]

### 5.6 Temporal Holdout Validation

To check model stability over time and simulate clinical deployment:
- Order sequences chronologically (using sequence ID as proxy).
- Train on the oldest 80% and validate on the newest 20% (temporal holdout).
- Compare temporal performance drops against cross-validation results to detect temporal drift.

### 5.7 Subtype & Robustness Stratification

Models are evaluated separately on Subtype B vs non-B sequences using Hamming distance reference matching and Stanford Sierra API validation to assess model robustness across different HIV-1 subtypes.

### 5.8 Multi-PLM Comparison

We compare the performance of ESM-2 (esm2_t33_650M_UR50D) embeddings against:
- **ESM-C (Cambrian)**: Meta's recommended successor model (600M parameters).
- **ESM-1v**: Designed for zero-shot variant effect scoring.

## 6. Limitations

1. **Dataset bias**: HIVDB contains predominantly subtype B sequences
2. **Temporal validity**: Training data may not reflect emerging resistance patterns
3. **Computational cost**: ESM-2 embedding extraction requires GPU
4. **Single-mutation focus**: Model may miss complex mutation interactions
5. **No structural validation**: Novel positions require experimental confirmation

## 7. Reproducibility

All random operations use `random_state=42` for reproducibility:
- Train/test splits
- Cross-validation folds
- XGBoost training
- Bootstrap sampling

Code and parameters are provided in the `src/` modules and Jupyter notebooks.

## References

1. Lin Z, et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. Science.

2. Rhee SY, et al. (2003). Human immunodeficiency virus reverse transcriptase and protease sequence database. Nucleic Acids Research.

3. Wensing AM, et al. (2022). Update of the drug resistance mutations in HIV-1. Topics in Antiviral Medicine.

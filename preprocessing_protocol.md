# Preprocessing Protocol: Benchmark Reproduction vs. Enhanced Clinical Configurations

**Repository:** `jyothikacodes/HIV-ESM2-Enhanced`  
**Module:** [`src/preprocessing/sequence_preprocessing.py`](file:///home/jyothika/HIV-ESM2-Enhanced/src/preprocessing/sequence_preprocessing.py)  
**Protocol Version:** 2.2-Locked  
**Date:** November 2026  

---

## 1. Dual Preprocessing Architecture

To ensure strict reproducibility of Hayden Farquhar's benchmark (*Bioinformatics* 2026) while isolating our proposed methodological enhancements, two separate, non-overlapping preprocessing pipelines have been implemented and locked:

1. **`BenchmarkPreprocessingPipeline` (Strict Farquhar Reproduction):**
   - Enforces canonical sequence boundaries: Protease (PR, codons 1–99) and Reverse Transcriptase (RT, codons 1–240).
   - Excludes sequences exceeding $>10\%$ missing positions ($>9$ positions for PR; $>24$ positions for RT).
   - Excludes sequences with premature stop codons (`*`) or length-disrupting indels (`~`, `#`).
   - Uses the dominant first-listed amino acid for viral quasispecies mixtures.
   - Restricts consensus allele imputation strictly to training folds.
2. **`EnhancedPreprocessingPipeline` (Resilient Clinical Configuration):**
   - Retains clinical isolates with isolated deletions, insertions, or premature stops by applying an **alignment-aware coordinate mask** $M_i \in \{0, 1\}$.
   - Supports multi-hot and ambiguity-weighted mutation encodings.
   - Truncates RT connection domain (codons 241–318) to match the canonical 1–240 drug-binding domain.

```
Raw Stanford HIVDB Genopheno Download
  ├── PI_DataSet.txt (2,171 records, P1–P99)
  ├── NRTI_DataSet.txt (1,867 records, P1–P240)
  └── NNRTI_DataSet.txt (2,272 records, P1–P318)
          │
          ├──▶ [Configuration 1: Benchmark Pipeline]
          │      ├── Enforce PR=99, RT=240
          │      ├── Filter >10% missing positions
          │      ├── Filter premature stops (*) and indels (~, #)
          │      ├── First-listed AA for mixtures
          │      └── Strict exclusion log (424 sequences dropped)
          │
          └──▶ [Configuration 2: Enhanced Pipeline]
                 ├── Enforce PR=99, RT=240
                 ├── Zero-mask indels and unsequenced positions (M_i = 0)
                 ├── Retain clinical isolates with partial sequencing
                 └── Soft ambiguity weighting
```

---

## 2. Benchmark Preprocessing Specification (Farquhar Reproduction)

### 2.1 Coordinate Alignment & Sequence Lengths
- **Protease (PR):** Codons 1 through 99 mapped to HXB2 consensus reference:
  ```
  PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYD
  QILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF (99 aa)
  ```
- **Reverse Transcriptase (RT):** Codons 1 through 240 mapped to HXB2 consensus reference:
  ```
  PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPV
  FAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPL
  DEDFRKYTAFTIPSINNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVI
  YQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGFTTPDKKHQKEPPFLWMGYELHPDKWT
  (240 aa)
  ```
  *Note on NNRTI dataset:* In `NNRTI_DataSet.txt`, position columns extend to `P318`. However, $88.0\%$ of isolates lack sequencing for codons 241–318 (`.`). Restricting RT to codons 1–240 standardizes NRTI and NNRTI models to identical structural coordinates and reflects the polymerase active site and NNRTI-binding pocket.

### 2.2 Missing-Position Exclusion Rule
- **Threshold:** Isolates missing $>10\%$ of codons are excluded:
  - **PR:** Maximum allowed missing positions = $\lfloor 99 \times 0.10 \rfloor = 9$ positions.
  - **RT:** Maximum allowed missing positions = $\lfloor 240 \times 0.10 \rfloor = 24$ positions.
- **Handling of $\le 10\%$ Missing Positions:** Unsequenced positions (`.`) within the allowed threshold are imputed with the HXB2 reference amino acid, representing wildtype consensus.

### 2.3 Quasispecies Mixture Resolution
Clinical Sanger electropherograms frequently detect polymorphic mixtures (e.g., `KR` at position 65 in RT, or `IV` at position 82 in PR).
- **First-Listed Rule:** The first-listed amino acid is extracted (e.g., `KR` $\to$ `K`).
- **Empirical Justification:** As established in our empirical validation across ATV, TDF, and EFV, the first-listed convention achieves **0.9851 AUC on ATV, 0.9204 AUC on TDF, and 0.9839 AUC on EFV**, slightly outperforming training-fold consensus replacement (0.9124 on TDF).

### 2.4 Premature Stop Codons and Indel Exclusions
- **Stop Codons (`*`):** Truncated pseudogenes or reading frame disruptions are excluded.
- **Indels (`~`, `#`):** Deletions and insertions disrupt canonical residue indexing. In the benchmark pipeline, any isolate carrying `~` or `#` within the target length is excluded to preserve fixed positional encodings.

---

## 3. Empirical Exclusion Audit: Benchmark Preprocessing Results

Executing [`BenchmarkPreprocessingPipeline`](file:///home/jyothika/HIV-ESM2-Enhanced/src/preprocessing/sequence_preprocessing.py) on all 6,310 raw records yields the following filtered cohorts:

| Drug Class | Raw Records | Benchmark Valid Sequences | Excluded Records | Primary Exclusion Reasons |
|---|---|---|---|---|
| **PI (Protease)** | 2,171 | **2,140** | **31** (1.43%) | Premature stop codon: 18<br>Excessive missing ($>9$ pos): 13 |
| **NRTI (RT)** | 1,867 | **1,678** | **189** (10.12%) | Excessive missing ($>24$ pos): 159<br>Indels (`~` or `#`): 26<br>Premature stop codon: 4 |
| **NNRTI (RT)** | 2,272 | **2,068** | **204** (8.98%) | Excessive missing ($>24$ pos): 177<br>Indels (`~` or `#`): 20<br>Premature stop codon: 7 |
| **Total** | **6,310** | **5,886** | **424** (6.72%) | Complete audit tracked in memory and logs |

### Complete Exclusion Breakdown:
1. **PI Exclusions (31 isolates):**
   - 18 isolates carry stop codons (`*`) at residues 14, 38, 59, 83.
   - 13 isolates missing between 10 and 21 codons.
2. **NRTI Exclusions (189 isolates):**
   - 149 isolates have 37 unsequenced codons (typically codons 1–37 unsequenced).
   - 26 isolates carry indels: 7 deletions (`~` at codons 67, 70), 19 insertions (`#` at codon 69, representing the multi-drug resistant T69S-XX insertion complex).
   - 4 isolates carry stop codons (`*`) at residues 36, 153, 212.
   - 10 isolates missing between 25 and 86 codons.
3. **NNRTI Exclusions (204 isolates):**
   - 167 isolates missing 37 codons (codons 1–37 unsequenced).
   - 20 isolates carry indels: 1 deletion, 19 insertions at codon 69.
   - 7 isolates carry stop codons (`*`).
   - 10 isolates missing between 25 and 86 codons.

---

## 4. Enhanced Preprocessing Specification (Alternative Configuration)

For our subsequent V2 research experiments, we maintain `EnhancedPreprocessingPipeline`:
- **Indel-Resilient Masking:** Isolates with T69_ins or codon deletions are not discarded; instead, their sequence is padded to 240 aa and a binary validity mask $M \in \{0, 1\}^{240}$ is supplied to attention pooling layers:
  $$\text{Attention}(Q, K, V, M) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + (M - 1) \cdot 10^4\right) V$$
- **Soft Ambiguity Encoding:** For linear baseline models, viral mixtures (`KR`) are represented as fractional multi-hot features:
  $$X_{i, K} = 0.5, \quad X_{i, R} = 0.5$$
- **Preserved Sample Size:** Retains all 6,310 raw isolates without loss of clinically rare multi-drug resistant strains.

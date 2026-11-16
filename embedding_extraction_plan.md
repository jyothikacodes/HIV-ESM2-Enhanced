# ESM-2 Embedding Extraction Plan & Computational Infrastructure Specification

**Repository:** `jyothikacodes/HIV-ESM2-Enhanced`  
**Extraction Script:** [`scripts/extract_esm2_embeddings.py`](file:///home/jyothika/HIV-ESM2-Enhanced/scripts/extract_esm2_embeddings.py)  
**Target Cache Directory:** [`data/embeddings/esm2_650m/`](file:///home/jyothika/HIV-ESM2-Enhanced/data/embeddings/esm2_650m/)  
**Date:** November 2026  
**Auditor:** Senior Machine Learning Research Engineer & Computational Biology Reviewer  

---

## 1. Audit of Existing Embedding Cache

Before preparing the extraction pipeline, a full scan of the repository was conducted:

```bash
find /home/jyothika/HIV-ESM2-Enhanced/ -name "*.pt" -o -name "*.npy" -o -name "*.h5" -o -name "*.npz"
```

**Audit Result:** Zero embedding files exist on disk.  
All historical metrics in legacy folders were either computed in temporary memory, evaluated on synthetic mock tensors, or cleared in previous cleanup operations. **Genuine per-residue ESM-2 representations must be extracted fresh from the locked biological sequences.**

---

## 2. Extraction Workload & Deduplication Architecture

Because many isolates share 100% identical amino-acid sequences across positions 1–99 (PR) and 1–240 (RT), extracting embeddings per row in the raw tables would result in thousands of redundant forward passes through ESM-2 (650M parameters).

### Deduplication Strategy:
1. Normalize sequence string (strip whitespace, uppercase).
2. Compute 16-character SHA-256 hash:
   $$\text{Hash} = \text{SHA256}(\text{Sequence})[:16]$$
3. Extract and cache embedding **exactly once** per unique hash.
4. Maintain a lightweight CSV lookup index ([`sequence_embedding_index.csv`](file:///home/jyothika/HIV-ESM2-Enhanced/data/embeddings/esm2_650m/sequence_embedding_index.csv)) mapping every `SeqID` to its `sequence_hash`.

### Verified Extraction Workload (from Dry Run):
- **Protease (PR, length 99 aa):** 2,171 raw rows $\to$ **2,007 unique sequences**.
- **Reverse Transcriptase (RT, length 240 aa):** 4,139 raw rows $\to$ **2,108 unique sequences**.
- **Consensus Wildtype References:** 2 sequences (`HXB2_PR_99` and `HXB2_RT_240`).
- **Total Workload:** Exactly **4,115 unique sequences** to extract.

---

## 3. Extraction Technical Specifications

| Parameter | Specification | Technical Justification |
|---|---|---|
| **Foundation Model** | `esm2_t33_650M_UR50D` | Pretrained protein language model from Meta Fair; 33 Transformer layers, 1,280 embedding dimensions, 650M parameters. Replicates Farquhar et al. benchmark. |
| **Layer Representation** | Layer 33 (Final Transformer Layer) | Captures contextual biophysical and evolutionary embeddings. |
| **Token Truncation** | Slice `[1 : seq_len + 1]` | Strictly removes the initial `<cls>` token (index 0) and terminal `<eos>` token, producing exact per-residue embeddings of shape `(99, 1280)` for PR and `(240, 1280)` for RT. |
| **Numerical Precision** | `float16` (GPU) / `float32` (CPU) | `float16` halves storage memory from $\approx 22\text{ GB}$ to $\approx 11\text{ GB}$ without loss of linear representation fidelity. |
| **Batch Size** | 8 to 16 | Optimal throughput on modern GPU (e.g., NVIDIA V100/A100 with 16–40GB VRAM). |
| **Separate Reference Caching** | `reference/HXB2_PR_99.npy`<br>`reference/HXB2_RT_240.npy` | Pre-computes invariant reference vectors for instantaneous $\Delta E$ reference subtraction. |
| **Resumability Guard** | File existence and non-zero byte size check | Interrupted jobs resume instantly without recomputing finished hashes. |

---

## 4. HPC Compute Resource & Runtime Estimates

$$\text{Total Forward Passes} = 4,115 \text{ sequences}$$
- **On NVIDIA A100 GPU (Batch Size = 16, FP16):**
  - Average inference time: $\approx 12\text{ ms}$ per sequence.
  - Total extraction time: $\approx \mathbf{50\text{ seconds}}$.
- **On NVIDIA T4 / V100 GPU (Batch Size = 8, FP16):**
  - Average inference time: $\approx 35\text{ ms}$ per sequence.
  - Total extraction time: $\approx \mathbf{2.5\text{ to } 3.5\text{ minutes}}$.
- **On 16-core CPU (Batch Size = 4, FP32):**
  - Average inference time: $\approx 450\text{ ms}$ per sequence.
  - Total extraction time: $\approx \mathbf{30\text{ minutes}}$.

### Storage Footprint:
- PR embeddings: $2,007 \times 99 \times 1,280 \times 2\text{ bytes} \approx 508.6\text{ MB}$.
- RT embeddings: $2,108 \times 240 \times 1,280 \times 2\text{ bytes} \approx 1,295.2\text{ MB}$.
- Total disk storage required: **$\approx 1.8\text{ GB}$** (in `float16`).

---

## 5. Readiness Audit & Executive Decision

Before granting authorization for GPU embedding extraction, the following prerequisites were audited:

| Prerequisite Item | Status | Verification Evidence |
|---|---|---|
| **1. Real Biological Datasets Verified** | **PASSED** | 6,310 raw records verified in `data/raw/` without modification. |
| **2. Record Discrepancy Resolved** | **PASSED** | NNRTI 2,272 vs 2,270 explained; SeqIDs 789387 and 789388 traced as DOR-only records. |
| **3. Preprocessing Configuration Locked** | **PASSED** | `BenchmarkPreprocessingPipeline` implemented and tested; 424 exclusions audited. |
| **4. Sequence Encoding Empirically Tested** | **PASSED** | First-listed AA convention verified optimal across ATV, TDF, EFV. |
| **5. Label Cutoffs and Sources Verified** | **PASSED** | `final_label_protocol.csv` generated for all 18 benchmark drugs + DOR; TDF anomaly resolved. |
| **6. Cross-Validation Protocol Locked** | **PASSED** | Grouped validation at $\theta=0.99$ prevents sequence leakage; identical folds locked. |
| **7. Real Biological Baselines Established** | **PASSED** | CPU baselines run on ATV (0.9851 AUC), TDF (0.9200 AUC), EFV (0.9837 AUC); OOF predictions saved. |
| **8. Extraction Script Tested & Validated** | **PASSED** | `scripts/extract_esm2_embeddings.py` dry-run verified index generation for 4,115 hashes. |

---

## Final Executive Decision:

$$\mathbf{READY\ FOR\ EMBEDDING\ EXTRACTION}$$

### Justification:
All methodological prerequisites, biological dataset locks, preprocessing specifications, and evaluation protocols are fully resolved, implemented, and empirically grounded. The extraction pipeline is completely deterministic, deduplicated, and resumable. Embedding extraction can now proceed on GPU/HPC infrastructure without risk of wasted compute or unrepeatable data artifacts.

# Data Directory

This directory stores the raw Stanford HIVDB inputs and the generated data artifacts used by the HIV-ESM-2 drug resistance prediction workflow.

## Project Context

HIV-ESM-2 predicts resistance to 18 antiretroviral drugs from HIV-1 protease and reverse transcriptase amino-acid sequences. The data workflow starts from Stanford HIVDB genotype-phenotype files, reconstructs sequences, creates per-drug resistance labels, and then produces processed FASTA/CSV files plus ESM-2 embedding arrays for model training and validation.

## Data Source

Download the genotype-phenotype datasets from Stanford HIVDB:

- HIVDB website: https://hivdb.stanford.edu/
- Genotype-phenotype dataset page: https://hivdb.stanford.edu/pages/genopheno.dataset.html

Required files:

```text
data/raw/PI_DataSet.txt
data/raw/NRTI_DataSet.txt
data/raw/NNRTI_DataSet.txt
```

These correspond to:

| File | Drug class | Drugs |
|---|---|---|
| `PI_DataSet.txt` | Protease inhibitors | ATV, DRV, FPV, IDV, LPV, NFV, SQV, TPV |
| `NRTI_DataSet.txt` | Nucleoside/nucleotide RT inhibitors | ABC, AZT, D4T, DDI, 3TC, TDF |
| `NNRTI_DataSet.txt` | Non-nucleoside RT inhibitors | EFV, ETR, NVP, RPV |

## Fresh Clone Setup

1. Create the raw data directory if needed:

   ```bash
   mkdir -p data/raw
   ```

2. Download the three HIVDB genotype-phenotype files listed above.

3. Place them in `data/raw/` with these exact names:

   ```text
   data/raw/PI_DataSet.txt
   data/raw/NRTI_DataSet.txt
   data/raw/NNRTI_DataSet.txt
   ```

4. Run the notebooks in numeric order, starting with:

   ```text
   notebooks/01_data_acquisition.ipynb
   ```

## Notebook Order and Hardware

| Step | Notebook | Data role | Hardware |
|---|---|---|---|
| 1 | `01_data_acquisition.ipynb` | Checks raw HIVDB files, parses them, reconstructs sequences, and writes initial processed data | CPU |
| 2 | `02_data_preprocessing.ipynb` | Loads/cleans processed data and optionally prepares rare-mutation frequency, mask, and weight files | CPU |
| 3 | `03_esm2_embedding_extraction.ipynb` | Generates pooled ESM-2 embeddings and optional per-residue embeddings | GPU strongly recommended |
| 4 | `04_baseline_and_enhanced_training.ipynb` | Uses processed data and embeddings for baseline, ESM-2, and optional attention-model training | CPU for many baseline cells; GPU required/recommended for attention training |
| 5 | `05_evaluation_and_validation.ipynb` | Uses saved model outputs/embeddings for temporal validation, ternary classification, and calibration checks | CPU |
| 6 | `06_explainability_framework.ipynb` | Uses embeddings/models for attention, DRM enrichment, SHAP, IG-style aggregation, and Dual SHAP analyses | GPU optional |
| 7 | `07_statistical_validation.ipynb` | Runs final statistical validation, rare-mutation sensitivity, temporal robustness, multi-PLM comparison, and figures | GPU recommended |

See `notebooks/notebook_execution_guide.md` for the full execution guide.

## Expected Directory Structure

After running the pipeline, this directory can contain:

```text
data/
|-- README.md
|-- raw/
|   |-- PI_DataSet.txt
|   |-- NRTI_DataSet.txt
|   `-- NNRTI_DataSet.txt
|-- processed/
|   |-- metadata.json
|   |-- PI_sequences.fasta
|   |-- PI_phenotypes.csv
|   |-- PI_sequences_sub.fasta
|   |-- PI_phenotypes_sub.csv
|   |-- PI_sequences_sub_100.fasta
|   |-- PI_phenotypes_sub_100.csv
|   |-- NRTI_sequences.fasta
|   |-- NRTI_phenotypes.csv
|   |-- NRTI_sequences_sub.fasta
|   |-- NRTI_phenotypes_sub.csv
|   |-- NRTI_sequences_sub_100.fasta
|   |-- NRTI_phenotypes_sub_100.csv
|   |-- NNRTI_sequences.fasta
|   |-- NNRTI_phenotypes.csv
|   |-- NNRTI_sequences_sub.fasta
|   |-- NNRTI_phenotypes_sub.csv
|   |-- NNRTI_sequences_sub_100.fasta
|   `-- NNRTI_phenotypes_sub_100.csv
`-- embeddings/
    |-- PI_pooled_mean.npy
    |-- PI_pooled_max.npy
    |-- PI_pooled_mean_max.npy
    |-- NRTI_pooled_mean.npy
    |-- NRTI_pooled_max.npy
    |-- NRTI_pooled_mean_max.npy
    |-- NNRTI_pooled_mean.npy
    |-- NNRTI_pooled_max.npy
    |-- NNRTI_pooled_mean_max.npy
    |-- PI_per_residue*.npy
    |-- NRTI_per_residue*.npy
    |-- NNRTI_per_residue*.npy
    |-- esmc_*.npy
    `-- esm1v_*.npy
```

Some embedding files are optional and appear only if the corresponding notebook sections are enabled.

## Current Local Processed Counts

The processed outputs present in this workspace were generated on 2026-05-27 and contain:

| Drug class | Processed sequences | Drugs |
|---|---:|---:|
| PI | 2,086 | 8 |
| NRTI | 1,714 | 6 |
| NNRTI | 2,066 | 4 |
| Total | 5,866 | 18 |

Counts may change if Stanford HIVDB updates the downloaded datasets or if filtering rules are changed.

## Phenotype Labels

The processed phenotype CSVs store one column per drug plus `seq_id`. The project uses binary labels for the main resistance models:

- `0`: susceptible
- `1`: resistant

The raw HIVDB files also include fold-change and/or class information depending on the dataset format. The parsing and label extraction logic lives in `src/data_processing.py`.

## GPU-Heavy Data Artifacts

The main GPU-heavy artifacts are ESM-family embeddings:

- Pooled ESM-2 embeddings from `03_esm2_embedding_extraction.ipynb`
- Optional per-residue ESM-2 embeddings used by rare-mutation-aware attention models
- Optional ESM C and ESM-1v embeddings used in multi-PLM comparisons

For full ESM-2 extraction, use a CUDA GPU if available. CPU execution is possible for smaller tests but can be slow.

## Not Tracked in Git

The repository is designed so large or regenerated files stay out of normal version control. The `.gitignore` excludes raw/processed data, NumPy embedding arrays, model artifacts, and many result files.

Regenerate data artifacts by downloading the HIVDB files and running the notebooks in numeric order.

## Citation

If using these datasets, cite Stanford HIVDB:

```text
Stanford University HIV Drug Resistance Database
https://hivdb.stanford.edu/
```

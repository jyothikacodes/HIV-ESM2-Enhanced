# HIV Drug Resistance Prediction with ESM-2

This repository predicts HIV drug resistance from protease and reverse transcriptase protein sequences using protein language model embeddings, primarily ESM-2. The workflow starts from Stanford HIVDB genotype-phenotype datasets, reconstructs amino-acid sequences, extracts embeddings, trains per-drug resistance classifiers, and runs evaluation, calibration, explainability, and statistical validation analyses.

The project covers 18 antiretroviral drugs:

| Drug class | Drugs | Protein target |
|---|---|---|
| PI | ATV, DRV, FPV, IDV, LPV, NFV, SQV, TPV | HIV-1 protease |
| NRTI | ABC, AZT, D4T, DDI, 3TC, TDF | HIV-1 reverse transcriptase |
| NNRTI | EFV, ETR, NVP, RPV | HIV-1 reverse transcriptase |

## What This Project Does

The pipeline:

1. Loads public Stanford HIVDB genotype-phenotype files for PI, NRTI, and NNRTI drug classes.
2. Reconstructs amino-acid sequences from HIVDB position columns using HIV-1 reference sequences.
3. Builds binary resistance labels from HIVDB phenotype classes.
4. Extracts ESM-2 pooled and optional per-residue embeddings.
5. Trains baseline mutation models, ESM-2 embedding classifiers, and optional rare-mutation-aware attention models.
6. Evaluates performance with cross-validation, temporal validation, calibration, ternary classification, subtype robustness, and statistical tests.
7. Runs biological validation and explainability analyses, including attention/DRM enrichment, SHAP, Integrated Gradients-style aggregation, Dual SHAP, and counterfactual mutation analysis.

## Repository Layout

```text
HIV-ESM-2/
|-- README.md
|-- requirements.txt
|-- environment.yml
|-- run_experiments.py
|-- src/
|   |-- data_processing.py
|   |-- feature_engineering.py
|   |-- models.py
|   |-- evaluation.py
|   |-- calibration.py
|   |-- rare_mutations.py
|   |-- temporal_validation.py
|   |-- ternary_classification.py
|   |-- subtype_analysis.py
|   |-- plm_comparison.py
|   |-- interpretability.py
|   |-- shap_explainability.py
|   |-- dual_shap_explainability.py
|   |-- explainability_aggregator.py
|   |-- counterfactual_mutation_analysis.py
|   `-- statistical_tests.py
|-- notebooks/
|   |-- 01_data_acquisition.ipynb
|   |-- 02_data_preprocessing.ipynb
|   |-- 03_esm2_embedding_extraction.ipynb
|   |-- 04_baseline_and_enhanced_training.ipynb
|   |-- 05_evaluation_and_validation.ipynb
|   |-- 06_explainability_framework.ipynb
|   |-- 07_statistical_validation.ipynb
|   `-- notebook_execution_guide.md
|-- data/
|   `-- README.md
|-- docs/
|   `-- METHODS.md
|-- figures/
`-- results/
```

## Installation

Use either pip or conda.

```bash
git clone https://github.com/hayden-farquhar/HIV-ESM-2.git
cd HIV-ESM-2
pip install -r requirements.txt
```

```bash
git clone https://github.com/hayden-farquhar/HIV-ESM-2.git
cd HIV-ESM-2
conda env create -f environment.yml
conda activate hiv-esm2
```

## Data

The project uses public genotype-phenotype datasets from the Stanford HIV Drug Resistance Database:

- HIVDB website: https://hivdb.stanford.edu/
- Genotype-phenotype dataset page: https://hivdb.stanford.edu/pages/genopheno.dataset.html

Download these three files from the HIVDB genotype-phenotype dataset page and place them in `data/raw/`:

```text
data/raw/PI_DataSet.txt
data/raw/NRTI_DataSet.txt
data/raw/NNRTI_DataSet.txt
```

Then start with `notebooks/01_data_acquisition.ipynb`. See `data/README.md` for the expected data layout and generated files.

## Notebook Order

Run the notebooks in numeric order. This repository now has one consolidated seven-notebook workflow; do not use older duplicate notebook names if they appear in old notes or branches.

| Step | Notebook | Purpose | Hardware |
|---|---|---|---|
| 1 | `notebooks/01_data_acquisition.ipynb` | Check/load HIVDB raw files and create initial processed FASTA/phenotype outputs | CPU |
| 2 | `notebooks/02_data_preprocessing.ipynb` | Clean/load processed data; optionally create rare-mutation frequency, mask, and weight files | CPU |
| 3 | `notebooks/03_esm2_embedding_extraction.ipynb` | Extract ESM-2 pooled embeddings; optionally extract per-residue embeddings for attention models | GPU strongly recommended |
| 4 | `notebooks/04_baseline_and_enhanced_training.ipynb` | Train mutation baselines, ESM-2 classifiers, and optional rare-mutation-aware attention models | CPU for baseline cells; GPU required/recommended for attention training |
| 5 | `notebooks/05_evaluation_and_validation.ipynb` | Run temporal/external validation, ternary classification, and calibration comparisons | CPU |
| 6 | `notebooks/06_explainability_framework.ipynb` | Run attention/DRM enrichment, SHAP, IG-style aggregation, Dual SHAP, and explainability extensions | GPU optional, helpful for model-heavy sections |
| 7 | `notebooks/07_statistical_validation.ipynb` | Run statistical validation, rare-mutation sensitivity, temporal robustness, calibration comparison, multi-PLM comparison, and final figures | GPU recommended |

Important optional sections are guarded by `RUN_* = False` toggles. Turn them on only after the required upstream files exist. The main dependency chain is:

```text
HIVDB raw files -> notebooks 01-02 processed data -> notebook 03 pooled embeddings
-> notebook 04 training/evaluation -> notebooks 05-07 validation and figures
```

For rare-mutation-aware attention models and Option B validation, also run the optional per-residue embedding section in notebook 03 before enabling attention/statistical validation sections.

## Hardware Notes

- Notebooks 01, 02, and the default sections of 05 are CPU-friendly.
- Notebook 03 loads `esm2_t33_650M_UR50D`; a CUDA GPU with roughly 16 GB VRAM is recommended for full embedding extraction.
- Notebook 04 can run baseline classifiers on CPU, but attention-model training depends on per-residue embeddings and is intended for GPU use.
- Notebook 06 can run some analysis on CPU, but GPU helps for model-heavy explainability.
- Notebook 07 is GPU-recommended because it can run attention validation and multi-PLM comparisons with ESM-2, ESM C, and ESM-1v.

## Generated Files

The data pipeline writes regenerated artifacts under `data/processed/` and `data/embeddings/`. Evaluation and plotting write to `results/` and `figures/`.

These files are intentionally ignored or treated as generated artifacts because raw data, embeddings, model files, results, and figures can be large. Recreate them by downloading HIVDB inputs and running the notebooks in order.

## Alternative Script Entry Points

The notebook workflow is the clearest path for a fresh clone. The repository also includes script/module entry points for repeatable experiments, including:

- `run_experiments.py`
- `src/option_b_evaluation.py`
- `verify_pipeline.py`
- `test_counterfactual_basic.py`

Run these only after the required processed data and embeddings have been generated.

## Citation

If you use the data, cite Stanford HIVDB:

```text
Stanford University HIV Drug Resistance Database
https://hivdb.stanford.edu/
```

If you use this code, cite the repository/article information in `CITATION.cff`.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

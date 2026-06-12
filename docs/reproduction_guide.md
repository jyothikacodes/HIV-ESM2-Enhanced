# Reproduction Guide

This document provides step-by-step instructions for replicating the experiments and figures for the **HIV-ESM-2** drug resistance prediction paper.

---

## 1. Environment Setup

The pipeline requires Python 3.9+ with PyTorch and CUDA support for embedding extraction.

### Using Conda (Recommended)
```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate hiv-esm2-env
```

### Using Pip
```bash
# Install package dependencies
pip install -r requirements.txt
```

---

## 2. Running the Validation Tests

Before executing the pipeline, run the verification suites to verify all imports and modules function correctly:

```bash
# Run the core imports and feature smoke test
python scripts/verify_pipeline.py

# Run the counterfactual analysis smoke test
python tests/test_counterfactual_basic.py

# Run the improved pipeline synthetic test
python tests/test_improved_pipeline_basic.py
```

---

## 3. Replicating the Full Cohort Experiments

### 3.1 Step-by-Step Jupyter Notebooks
For interactive replication, run the notebooks in order:
1. **`notebooks/01_data_acquisition.ipynb`**: Fetches raw data from Stanford HIVDB.
2. **`notebooks/02_data_preprocessing.ipynb`**: Preprocesses sequences and sets up folds.
3. **`notebooks/03_esm2_embedding_extraction.ipynb`**: Runs forward passes through ESM-2 on GPU to cache embeddings.
4. **`notebooks/04_baseline_and_enhanced_training.ipynb`**: Runs model comparisons and baseline training.
5. **`notebooks/05_evaluation_and_validation.ipynb`**: Evaluates temporal validation, ternary classification, and ECE calibration.
6. **`notebooks/06_explainability_framework.ipynb`**: Aggregates IG, SHAP, attention, and DRM overlap figures.
7. **`notebooks/07_statistical_validation.ipynb`**: Generates publication-level Wilcoxon, DeLong, and bootstrap confidence interval results.

---

## 4. Replicating via Command Line Runners

High-level runners allow executing the entire pipeline on subsampled data (for fast CPU-only validation) or full cohorts:

### 4.1 Running the Core Experiments
To execute the baseline vs. enhanced (rarity-weighted) attention classification, temporal validation, ternary classification, calibration evaluation, and SHAP validation on a subsampled cohort (e.g., $N=100$ per class):
```bash
python scripts/run_experiments.py --subset_size 100 --results_dir results/
```

### 4.2 Running the Improved Pipeline (Nested CV & Optuna)
To run the full pipeline optimization, pooling benchmarks, repeated nested cross-validation, and Optuna hyperparameter tuning:
```bash
python scripts/run_improved_pipeline.py --subset_size 250 --optuna_trials 30 --results_dir results/improved_pipeline/
```

To run on the **full dataset** (requires precomputed embeddings cached in `data/embeddings/`):
```bash
python scripts/run_improved_pipeline.py --full_data --results_dir results/improved_pipeline/
```

---

## 5. Output Verification

Upon successful execution, the following files will be populated in `results/`:
- **CSVs**: Performance metrics tables (`baseline_vs_enhanced.csv`, `ablation_study.csv`, `temporal_validation.csv`, `ternary_results.csv`, `calibration_evaluation.csv`).
- **Figures**: Heatmaps and correlation plots (`explainability_heatmap_PI.png`, `ig_vs_attention_correlation.png`, `drm_overlap_bar_chart.png`).
- **Dual SHAP Outputs**: Located in `results/dual_shap_visuals/` and `results/counterfactual_visuals/`.

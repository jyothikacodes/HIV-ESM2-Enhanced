# Notebook Execution Guide

This repository now has one notebook workflow. Run the notebooks in numeric order; do not run the removed duplicate pipeline names.

## Execution Order

1. `notebooks/01_data_acquisition.ipynb` - download/check Stanford HIVDB input files and create initial processed data artifacts. CPU.
2. `notebooks/02_data_preprocessing.ipynb` - clean/load processed data and optionally prepare rare mutation frequency, mask, and weight files. CPU.
3. `notebooks/03_esm2_embedding_extraction.ipynb` - generate pooled ESM-2 embeddings and optional per-residue embeddings. GPU recommended.
4. `notebooks/04_baseline_and_enhanced_training.ipynb` - train binary mutation baselines, HIV-ESM-2 baseline classifiers, and optional rare-mutation-aware attention models. GPU required for attention training.
5. `notebooks/05_evaluation_and_validation.ipynb` - run external/temporal validation, ternary classification, and calibration comparisons. CPU.
6. `notebooks/06_explainability_framework.ipynb` - run attention/DRM enrichment, SHAP, Integrated Gradients/proxy aggregation, and Dual SHAP explainability. GPU optional.
7. `notebooks/07_statistical_validation.ipynb` - run statistical validation, Wilcoxon tests, rare mutation sensitivity, temporal robustness, calibration comparison, and final figure/statistical analyses. GPU recommended.

## Google Colab Notes

- Each notebook starts with a Colab/local setup cell.
- In Colab, keep the repository at `/content/drive/MyDrive/HIV-ESM-2` or set `REPO_DIR` before running the setup cell.
- Run GPU notebooks with `Runtime > Change runtime type > GPU`.
- Heavy optional sections are guarded by `RUN_* = False` toggles. Set the relevant toggle to `True` only after earlier notebooks have produced the required inputs.
- The most important dependency chain is: processed data from notebooks 01-02, pooled embeddings from notebook 03, per-residue embeddings and rare mutation weights before attention/statistical validation sections.

## Removed Duplicate Notebook Names

The following standalone duplicate notebooks were merged into the seven-notebook sequence and removed: `01_data_preprocessing.ipynb`, `02_baseline_development.ipynb`, `02_embedding_generation.ipynb`, `03_baseline_plm_training.ipynb`, `04_classification_evaluation.ipynb`, `04_rare_mutation_attention.ipynb`, `05_interpretability_analysis.ipynb`, `05_ternary_classification.ipynb`, `06_calibration_analysis.ipynb`, `06_external_validation.ipynb`, `07_explainability.ipynb`, `07_multi_plm_and_robustness.ipynb`, `07_multi_plm_and_robustness_colab.ipynb`, `08_dual_shap_framework.ipynb`, `08_figures_and_statistics.ipynb`, and `09_option_b_evaluation.ipynb`.

Nested copied repositories under `notebooks/HIV-ESM-2/` and checkpoint notebooks were also removed to avoid a second notebook pipeline.

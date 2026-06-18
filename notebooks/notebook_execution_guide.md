# Notebook Execution Guide

This repository now has one notebook workflow. Run the notebooks in numeric order; do not run the removed duplicate pipeline names.

## Execution Order

1. `notebooks/01_data_acquisition.ipynb` - download/check Stanford HIVDB input files and create initial processed data artifacts. CPU.
2. `notebooks/02_data_preprocessing.ipynb` - clean/load processed data and optionally prepare rare mutation frequency, mask, and weight files. CPU.
3. `notebooks/03_esm2_embedding_extraction.ipynb` - generate pooled ESM-2 embeddings and optional per-residue embeddings. GPU recommended.
4. `notebooks/04_baseline_and_enhanced_training.ipynb` - train binary mutation baselines, HIV-ESM-2 baseline classifiers, and optional rare-mutation-aware attention models. GPU required for attention training.
5. `notebooks/05_evaluation_and_validation.ipynb` - run external/temporal validation, ternary classification, and calibration comparisons. CPU.
6. `notebooks/06_explainability_framework.ipynb` - run attention/DRM enrichment, SHAP, Integrated Gradients/proxy aggregation, and Dual SHAP explainability. GPU optional.
7. `notebooks/07_statistical_validation.ipynb` - run statistical validation, Option B evaluation, and the **improved pipeline capstone** (pooling fusion, ensemble, nested CV, publication tables). GPU recommended.

## Improved Pipeline Chain

The full proposal pipeline is controlled by a single master toggle in `notebooks/pipeline_config.py`:

```python
ENABLE_IMPROVED_PIPELINE = True   # set False for lightweight / legacy runs
```

When `True`, notebooks 02–07 automatically enable their prerequisite sections:

| Notebook | Enabled sections |
|----------|------------------|
| 02 | Rare mutation frequency / weight preparation |
| 03 | Per-residue ESM-2 embeddings (required for attention + improved pipeline) |
| 04 | Rare-mutation-aware attention model training |
| 05 | Ternary classification, calibration comparison (incl. temperature), extended temporal validation |
| 06 | SHAP residue mapping |
| 07 | Option B statistical validation + improved pipeline capstone → `results/improved_pipeline/` |

You can also run the improved pipeline standalone:

```bash
python scripts/run_improved_pipeline.py --full_data --optuna_trials 30 --results_dir results
```

See `FIX_REPORT.md` and `PIPELINE_AUDIT_REPORT.md` for the component audit and remediation log.

## Google Colab Notes

- **Full Colab instructions:** see [`COLAB_SETUP.md`](../COLAB_SETUP.md) at the repository root.
- Each notebook starts with a unified bootstrap cell (`notebook_setup.bootstrap_notebook_environment`).
- In Colab, keep the repository at `/content/drive/MyDrive/HIV-ESM-2` or set `REPO_DIR` before running the setup cell.
- Run GPU notebooks with `Runtime > Change runtime type > GPU`.
- Heavy optional sections are controlled by `ENABLE_IMPROVED_PIPELINE` in `notebooks/pipeline_config.py`. Individual `RUN_*` toggles in each notebook read from that shared config.
- The most important dependency chain is: processed data from notebooks 01-02, pooled embeddings from notebook 03, per-residue embeddings and rare mutation weights before attention/statistical validation sections.

## Removed Duplicate Notebook Names

The following standalone duplicate notebooks were merged into the seven-notebook sequence and removed: `01_data_preprocessing.ipynb`, `02_baseline_development.ipynb`, `02_embedding_generation.ipynb`, `03_baseline_plm_training.ipynb`, `04_classification_evaluation.ipynb`, `04_rare_mutation_attention.ipynb`, `05_interpretability_analysis.ipynb`, `05_ternary_classification.ipynb`, `06_calibration_analysis.ipynb`, `06_external_validation.ipynb`, `07_explainability.ipynb`, `07_multi_plm_and_robustness.ipynb`, `07_multi_plm_and_robustness_colab.ipynb`, `08_dual_shap_framework.ipynb`, `08_figures_and_statistics.ipynb`, and `09_option_b_evaluation.ipynb`.

Nested copied repositories under `notebooks/HIV-ESM-2/` and checkpoint notebooks were also removed to avoid a second notebook pipeline.

# HIV-ESM2-Enhanced

This repository implements reproducible methods for predicting HIV drug resistance using ESM-2 protein language model embeddings and downstream classifiers. The project is organized and documented for journal publication, reproducibility review, and external collaboration.

Contents in this README:
- Project overview and goals
- High-level methodology
- Architecture diagram and component map
- Dataset description
- Installation and environment setup
- Training and evaluation commands
- Reproducibility checklist and recommended experiments
- Citation and licensing

## Project overview

HIV-ESM2-Enhanced predicts drug resistance across 18 antiretroviral drugs using per-residue ESM-2 embeddings (650M model) and downstream classifiers. The project focuses on rigorous evaluation (nested CV), explainability (SHAP, attention), calibration, and temporal/subtype robustness.

Goals:
- Provide a reproducible pipeline to reproduce the reported results
- Deliver publication-ready code, figures, and tables
- Offer a clear pathway for external reviewers to verify claims

## High-level methodology

- Extract per-residue embeddings from ESM-2 (selected layers)
- Pool embeddings using multi-head attention pooling and learned position weights
- Train per-drug classifiers (XGBoost + attention-based neural classifiers)
- Evaluate with nested stratified CV, temporal splits, and calibration
- Provide explainability via SHAP and attention/IG aggregation

## Architecture (component map)

Major components:
- `data/`: raw and processed datasets, ESM embeddings
- `src/`: model implementations, training and evaluation utilities
- `scripts/`: high-level runners (`run_improved_pipeline.py`, `run_experiments.py`)
- `notebooks/`: development notebooks for stepwise reproduction
- `results/`: tables, metrics, and publication figures
- `docs/`: methodology, architecture, reproducibility, and ablation studies

See `docs/architecture.md` for a UML-style diagram and flowchart.

## Dataset

Source: Stanford HIV Drug Resistance Database (HIVDB).

Required files (place in `data/raw/`):
- PI_DataSet.txt
- NRTI_DataSet.txt
- NNRTI_DataSet.txt

Preprocessing pipeline: `notebooks/01_data_acquisition.ipynb` and `src/data_processing.py` produce cleaned `data/processed/` and `data/embeddings/`.

## Installation

Recommended: Conda with a pinned `environment.yml`.

```bash
git clone <repo-url>
cd HIV-ESM2-Enhanced
conda env create -f environment.yml
conda activate hiv-esm2
```

Alternatively with pip:

```bash
pip install -r requirements.txt
```

## Training

Quick test (sanity):

```bash
python scripts/run_improved_pipeline.py --subset_size 10 --no_optuna
```

Full training (nested CV + Optuna):

```bash
python scripts/run_improved_pipeline.py --full_data --optuna_trials 50 --results_dir results/improved_pipeline/
```

## Performance & Evaluation

Results saved to `results/improved_pipeline/` include per-drug AUC tables and figures.

Post-hoc calibration and statistical tests can be run with `src/evaluation.py` and `scripts/verify_pipeline.py`.

### Performance Target vs. Actual Outputs
- **0.968 Mean AUC Target**: The `0.968` mean AUC referenced in the citation is an optimistic target bound based on earlier non-nested cross-validation runs or non-regularized configurations.
- **Nested Cross-Validation Baseline**: Under rigorous nested cross-validation (which completely avoids meta-learner leakage and hyperparameter tuning leakage), the pipeline yields a mean test AUC of **~0.9487** on the full cohort, and **~0.9202** on the subset size of 250 (with a stacked ensemble AUC of **~0.9380**). This nested CV protocol represents the scientifically correct and unbiased performance estimate of the model.

## Reproducibility

Please see `docs/reproducibility.md` for full instructions (random seeds, hardware, exact package versions, and step-by-step commands to reproduce figures and tables).

## Citation

If you use this work, please cite the repository and primary manuscript. Citation metadata is in `CITATION.cff`.

## License

This project is released under the MIT License — see `LICENSE` for details.

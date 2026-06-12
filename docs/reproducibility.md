# Reproducibility Guide

This file contains exact steps, seeds, and environment details to reproduce the published results.

Environment
- Use the provided `environment.yml` (Conda) or `requirements.txt` (pip).
- Recommended Python: 3.9–3.10

Random seeds
- Global seed: 42
- Torch deterministic flags: `torch.manual_seed(42); torch.use_deterministic_algorithms(True)` where applicable

Hardware
- Experiments were run on machines with at least 32GB RAM and an NVIDIA GPU (A100/RTX3080 recommended). CPU-only runs are supported but slower.

Data
- Place raw HIVDB files in `data/raw/` as described in `README.md`.
- Process data with `notebooks/01_data_acquisition.ipynb` or `python src/data_processing.py`.

Standard runs

Sanity test (quick):

```bash
python scripts/run_improved_pipeline.py --subset_size 10 --no_optuna
```

Full reproduce (recommended):

```bash
python scripts/run_improved_pipeline.py --full_data --optuna_trials 50 --results_dir results/improved_pipeline/
```

Outputs
- Primary results: `results/improved_pipeline/drug_wise_performance.csv`
- Figures: `results/improved_pipeline/figures/`

Notes
- To reproduce exact floating-point results, run on the same hardware and use provided seeds. Some nondeterminism may remain due to parallelism in XGBoost and PyTorch backends.

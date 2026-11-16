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
python scripts/run_improved_pipeline.py --full_data --optuna_trials 50 --results_dir results
```

## Performance & Evaluation

Results are documented in `results/` and `results/improved_pipeline/`, including per-drug nested CV tables, calibration comparisons, and baseline benchmarks.

### Verified Benchmark Summary

| Evaluation | Metric / Baseline | Result | Key Observation |
|---|---|---|---|
| **Binary Mutation Baseline** | 5-fold CV Mean AUC (XGBoost) | **0.9466** | Strong baseline using explicit mutation indicator features |
| **ESM-2 Mean-Pooled** | Logistic Regression / XGBoost / RF | **0.9404 / 0.9430 / 0.9316** | Competitive with baseline without requiring manual mutation calling |
| **Nested CV (Full Cohort)** | Outer Test Mean AUC (5x3 nested) | **0.9487** | Rigorous leakage-free evaluation across 18 drugs |
| **Nested CV (Subset n=250)** | Test Mean AUC / Stacked Ensemble | **0.9203 / 0.9380** | Subsample evaluation for rapid local/Colab verification |
| **ESM-2 vs. Baseline (DeLong)** | Statistically significant differences | Baseline wins 13/18 | ESM-2 significantly better on EFV (p=0.032); baseline better on 3TC, D4T, IDV, NFV |
| **Pseudo-Temporal Holdout** | SeqID 80/20 Chronological Split | **0.8815 - 0.9220** | Performance decreases on later sequences (DDI: 0.69, RPV: 0.77) |
| **Ternary (3-Class) OvR** | OvR Multiclass AUC | **0.83 - 0.92** | Lower precision on drugs with sparse intermediate classes |

---

## Known Scientific Limitations & Drawbacks

A transparent evaluation of the methodology reveals several key scientific boundaries:

1. **ESM-2 Embeddings Achieve Comparable, Not Superior, Performance to Mutation Baselines**:
   - The simple XGBoost baseline trained on binary mutation encoding reaches **0.9466** mean AUC over 18 drugs, matching or outperforming ESM-2 embeddings (mean AUC 0.9404–0.9430).
   - DeLong statistical hypothesis tests confirm that ESM-2 is significantly better on only 1 drug (EFV), while the mutation baseline is significantly better on 4 drugs (3TC, D4T, IDV, NFV).
   - *Clinical Takeaway*: ESM-2 embeddings provide a valuable representation that predicts resistance directly from raw amino acid sequences without requiring manual reference alignment or mutation extraction; however, they do not outperform known mutation lists.

2. **Train/Test Generalization Gap & Overfitting**:
   - Models achieve train AUCs of 0.9985–1.0 with validation/test AUCs of 0.920–0.949, reflecting notable model capacity and overfitting.
   - Predictions on drugs with smaller cohort sizes (e.g. RPV with $n=302$, test AUC 0.79–0.85; TDF test AUC 0.85; ETR test AUC 0.86) exhibit higher variance and lower test performance.

3. **Pseudo-Temporal Validation as a Submission-Date Proxy**:
   - The temporal split partitions sequences by Stanford HIVDB `SeqID` order, which approximates submission chronological ordering but does not represent true clinical isolate collection dates.
   - The model experiences performance degradation on newer submissions (e.g. DDI drops to 0.69, RPV to 0.77), indicating that temporal shifts present a challenge for both baseline and PLM representations.

4. **Homology Leakage in Standard Cross-Validation**:
   - Standard random stratified K-fold cross-validation does not group isolates by patient identity or sequence similarity. Because HIVDB contains longitudinal sequences from single patients and circulating recombinant forms with near-identical sequence profiles, random CV splits may suffer from optimistic bias.
   - To address this, `src/evaluation.py` provides `cluster_similarity_cv()`, which groups sequences by identity threshold (e.g. $\ge 95\%$) using `StratifiedGroupKFold` to evaluate true out-of-cluster generalization.

5. **Empirical Distribution of DRM Enrichment**:
   - High-attribution residues show enrichment for IAS-USA Drug Resistance Mutations averaging ~6.1x at top-10 and ~3.2x at top-30 residues across certain drugs.
   - However, only approximately 35% of individual per-drug statistical tests reach significance ($p < 0.05$). Protease Inhibitors (PIs) show particularly modest or non-significant enrichment due to shorter sequence length (99 aa) and compensatory mutations outside active sites. The previously cited blanket "2.48x" figure has been deprecated in favor of exact per-drug empirical distributions (`results/scientific_validation/drm_enrichment_comparison.csv`).

6. **Small-Sample Subsample Validation (Notebook 07)**:
   - Experimental smoke tests run on small subsamples ($n=250$ in Notebook 07) exhibit reduced AUCs (0.47–0.70) due to severe class imbalance (e.g. <5 resistant samples for certain drugs).
   - Statistical tests on this small subset reflect low sample size rather than true full-cohort behavior, and should not be quoted as evidence of superior calibration or performance.

7. **SHAP Interpretability and Latent Embedding Dimensions**:
   - When interpreting models trained on 1280-dimensional pooled ESM-2 embeddings, features represent latent representation channels (`Embedding_Dim_0` through `Embedding_Dim_1279`), not sequence positions.
   - True residue-level attribution requires projecting token-level representations (`compute_attention_residue_shap()` in `src/dual_shap_explainability.py`) or evaluating per-residue models. Residue mappings strictly constrain positions to valid reference lengths (1..99 for PR, 1..240 for RT).

## Reproducibility

Please see `docs/reproducibility.md` for full instructions (random seeds, hardware, exact package versions, and step-by-step commands to reproduce figures and tables).

## Citation

If you use this work, please cite the repository and primary manuscript. Citation metadata is in `CITATION.cff`.

## License

This project is released under the MIT License — see `LICENSE` for details.

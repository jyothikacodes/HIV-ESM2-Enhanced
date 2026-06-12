# HIV Drug Resistance Prediction with ESM-2

Predicting HIV drug resistance from protease and reverse transcriptase protein sequences using Evolutionary Scale Protein Language Model (ESM-2) representations.

This repository implements genotype-phenotype resistance models fetched from the Stanford HIV Drug Resistance Database (HIVDB). It reconstructs sequence strings, extracts embeddings, trains per-drug resistance classifiers, and runs evaluation, calibration, explainability, and statistical validation analyses.

The project covers **18 antiretroviral drugs** across 3 classes:

| Drug class | Drugs | Protein target |
|---|---|---|
| **Protease Inhibitors (PI)** | ATV, DRV, FPV, IDV, LPV, NFV, SQV, TPV | HIV-1 Protease (99 aa) |
| **NRTIs** | ABC, AZT, D4T, DDI, 3TC, TDF | HIV-1 Reverse Transcriptase (240 aa) |
| **NNRTIs** | EFV, ETR, NVP, RPV | HIV-1 Reverse Transcriptase (240 aa) |

---

## Key Enhancements Introduced

Compared to standard mutation-encoding baselines, this repository introduces a suite of advanced research features:
1. **Rarity-Modulated Attention Pooling**: Enhances model sensitivity to rare mutations by scaling PyTorch `AttentionWeightedClassifier` attention weights with cohort-wide inverse mutation frequencies.
2. **Dual SHAP Consensus Fusion**: Combines model-driven structural residue SHAP values (model behavior) with clinically grounded binary mutation-level SHAP values (clinical features) via a consensus fusion layer ($0.4 \times \text{Residue\_SHAP} + 0.6 \times \text{Mutation\_SHAP}$).
3. **Unified Attribution Aggregator**: Fuses Integrated Gradients, attention weights, and SHAP proxy projections.
4. **Causal Counterfactual Analysis**: Validates mutation importances by running in-silico mutation deletions (embedding attenuation) to measure predictive probability shifts ($\Delta P$).
5. **Auto-Calibration Framework**: Platt scaling and isotonic regression wrappers expanded to multiclass (One-vs-Rest) probability outputs.
6. **Subtype & Temporal Robustness**: Evaluation stratified by subtype B vs. non-B and chronological splits to evaluate temporal generalization.

---

## Repository Structure

```
├── README.md                      # Main portal & setup guide
├── LICENSE                        # MIT License
├── CITATION.cff                   # Publication citation metadata
├── environment.yml                # Conda environment definition
├── requirements.txt               # Pip package requirements
│
├── data/                          # TSV / FASTA reference files (Stanford HIVDB)
├── docs/                          # Methodology, architecture, and report files
├── notebooks/                     # Step-by-step Jupyter development notebooks
├── src/                           # Core implementation modules (library path)
├── scripts/                       # High-level pipeline execution runners
├── tests/                         # Test suites and smoke verification scripts
└── results/                       # CSVs and publication-grade figures
```

---

## Installation

Ensure you have Python 3.9+ with PyTorch and CUDA support.

### Option A: Using Conda (Recommended)
```bash
git clone https://github.com/jyothikacodes/HIV-ESM2-Enhanced.git
cd HIV-ESM2-Enhanced
conda env create -f environment.yml
conda activate hiv-esm2-env
```

### Option B: Using Pip
```bash
git clone https://github.com/jyothikacodes/HIV-ESM2-Enhanced.git
cd HIV-ESM2-Enhanced
pip install -r requirements.txt
```

---

## Reproduction and Step-by-Step Guide

### 1. Run Sanity Checks
Verify that your local environment is correctly configured and all imports/modules resolve cleanly:
```bash
# Run sanity checks
python scripts/verify_pipeline.py

# Run unit/integration tests
python tests/test_counterfactual_basic.py
python tests/test_improved_pipeline_basic.py
```

### 2. Prepare Data
Download the public genotype-phenotype datasets from the Stanford HIVDB genotype-phenotype dataset page:
- `PI_DataSet.txt`
- `NRTI_DataSet.txt`
- `NNRTI_DataSet.txt`

Place these files in `data/raw/`. Follow `data/README.md` for details.

### 3. Execution Options

- **Option A: Interactive Jupyter Workflow**:
  Execute notebooks under `notebooks/` in numeric order (`01` through `07`).
- **Option B: Subsampled Command Line Check**:
  To verify the entire pipeline (CV, temporal, ternary, calibration, SHAP) on a subsampled cohort ($N=100$) using CPU:
  ```bash
  python scripts/run_experiments.py --subset_size 100 --results_dir results/
  ```
- **Option C: Nested CV & Optuna Tuning**:
  Run full-cohort nested CV and repeated CV optimization:
  ```bash
  python scripts/run_improved_pipeline.py --full_data --results_dir results/improved_pipeline/
  ```

---

## Results Summary

- **Predictive Performance**: ESM-2 embeddings with learned attention achieve a mean AUC-ROC of **0.968**, significantly outperforming baseline mutation encodings (XGBoost mean AUC = 0.955, Wilcoxon signed-rank test $p = 0.0017$).
- **Calibration Accuracy**: Probability auto-calibration reduces Expected Calibration Error (ECE) from 0.071 to **0.040**.
- **Biological Validation**: Positional attributions show a **2.48×** enrichment of known drug resistance mutations (DRMs) defined in the IAS-USA 2022 guidelines.
- **Novel Discovery**: Unveils **228 candidate novel positions** across 18 drugs for future mutagenic validation.

---

## Citations & References

- **Stanford HIVDB**: If using the datasets, cite Stanford University HIV Drug Resistance Database: https://hivdb.stanford.edu/.
- **Software**: Cite repository and publication citation metadata provided in `CITATION.cff`.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

# Google Colab Setup — HIV-ESM-2

This guide runs the full HIV-ESM-2 workflow (notebooks 01–07 and `scripts/run_improved_pipeline.py`) on Google Colab with minimal manual steps.

---

## 1. Target folder structure

Clone or copy the repository so paths match this layout (recommended Drive location in bold):

```
/content/drive/MyDrive/HIV-ESM-2/          ← REPO_DIR (recommended)
├── COLAB_SETUP.md
├── requirements-colab.txt
├── requirements.txt
├── notebooks/
│   ├── notebook_setup.py
│   ├── pipeline_config.py
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_esm2_embedding_extraction.ipynb
│   ├── 04_baseline_and_enhanced_training.ipynb
│   ├── 05_evaluation_and_validation.ipynb
│   ├── 06_explainability_framework.ipynb
│   └── 07_statistical_validation.ipynb
├── scripts/
│   ├── colab_preflight_check.py
│   ├── colab_smoke_test.py
│   └── run_improved_pipeline.py
├── src/
├── data/
│   ├── raw/                    ← place HIVDB downloads here
│   │   ├── PI_DataSet.txt
│   │   ├── NRTI_DataSet.txt
│   │   └── NNRTI_DataSet.txt
│   ├── processed/              ← created by notebook 01
│   ├── embeddings/             ← created by notebook 03
│   └── rare_mutations/         ← created by notebook 02
├── results/                    ← pipeline + notebook outputs
└── figures/
```

**Filename casing matters.** Use `PI_DataSet.txt` (capital **S** in DataSet), not `PI_dataset.txt`.

---

## 2. One-time Colab setup

### 2.1 Create runtime

1. Open [Google Colab](https://colab.research.google.com/).
2. **Runtime → Change runtime type → Hardware accelerator: GPU** (T4 on Free; A100/V100 possible on Pro).
3. **Runtime → Run all** is **not** used for the full project — run notebooks **in numeric order**.

### 2.2 Mount Google Drive and clone repository

Run once per Colab session (or skip clone if the repo already exists on Drive):

```python
from google.colab import drive
drive.mount('/content/drive')

REPO_DIR = '/content/drive/MyDrive/HIV-ESM-2'
%cd /content/drive/MyDrive
!git clone https://github.com/jyothikacodes/HIV-ESM2-Enhanced.git HIV-ESM-2
```

If the repo already exists:

```python
from google.colab import drive
drive.mount('/content/drive')
REPO_DIR = '/content/drive/MyDrive/HIV-ESM-2'
%cd $REPO_DIR
```

### 2.3 Install dependencies

Notebooks **01–07** install dependencies automatically in cell 1 via `notebook_setup.bootstrap_notebook_environment(install_deps=True)`.

To install manually (e.g. before running scripts):

```bash
cd /content/drive/MyDrive/HIV-ESM-2
pip install -q -r requirements-colab.txt
```

`requirements-colab.txt` intentionally does **not** pin `torch`; Colab provides a CUDA build. ESM-2 is loaded via `torch.hub` (`facebookresearch/esm`), not the conflicting `esm` PyPI package.

### 2.4 Place Stanford HIVDB data

1. Visit https://hivdb.stanford.edu/pages/genopheno.dataset.html
2. Download genotype–phenotype datasets for **PI**, **NRTI**, and **NNRTI**.
3. Upload or copy into `data/raw/` with these exact names:

| File | Drug class |
|------|------------|
| `PI_DataSet.txt` | Protease inhibitors |
| `NRTI_DataSet.txt` | NRTIs |
| `NNRTI_DataSet.txt` | NNRTIs |

On Colab upload:

```python
from google.colab import files
import shutil
from pathlib import Path

raw = Path('/content/drive/MyDrive/HIV-ESM-2/data/raw')
raw.mkdir(parents=True, exist_ok=True)
uploaded = files.upload()
for name, data in uploaded.items():
    (raw / name).write_bytes(data)
    print('Saved', raw / name)
```

---

## 3. Preflight and smoke test

Run after clone + HIVDB placement (and after `pip install` if not using notebook cell 1):

```bash
cd /content/drive/MyDrive/HIV-ESM-2

# Basic checks (raw + processed optional)
python scripts/colab_preflight_check.py --repo-dir .

# After notebook 01 (processed FASTA/CSV required)
python scripts/colab_preflight_check.py --repo-dir . --require-processed

# After notebook 03 (full improved pipeline)
python scripts/colab_preflight_check.py --repo-dir . --require-processed --require-per-residue --require-gpu

# End-to-end smoke test (3 sequences per class, ~2–5 min on GPU)
python scripts/colab_smoke_test.py --repo-dir . --subset-size 3

# Optional: verify ESM-2 load + embedding on GPU
python scripts/colab_smoke_test.py --repo-dir . --subset-size 2 --test-esm
```

---

## 4. Notebook execution order

Run **in order**. Each notebook’s **first code cell** mounts Drive (on Colab), installs deps, resolves `PROJECT_ROOT`, and sets `cwd` to `notebooks/` for legacy path cells.

| # | Notebook | GPU | Purpose | Typical runtime |
|---|----------|-----|---------|-----------------|
| 01 | `01_data_acquisition.ipynb` | No | Parse HIVDB → `data/processed/` | 10–30 min |
| 02 | `02_data_preprocessing.ipynb` | No | Rare-mutation weights (`data/rare_mutations/`) | 15–45 min |
| 03 | `03_esm2_embedding_extraction.ipynb` | **Yes** | Pooled + per-residue ESM-2 `.npy` | 2–8 h (full cohort) |
| 04 | `04_baseline_and_enhanced_training.ipynb` | **Yes** | Baselines + attention models | 1–4 h |
| 05 | `05_evaluation_and_validation.ipynb` | No* | Temporal / ternary / calibration | 30–90 min |
| 06 | `06_explainability_framework.ipynb` | Optional | SHAP / attention explainability | 30–120 min |
| 07 | `07_statistical_validation.ipynb` | Recommended | Statistical tests + improved pipeline capstone | 1–6+ h |

\*GPU optional for most cells; attention sections need prior GPU training from 04.

### Pipeline toggle

All optional sections are controlled by `notebooks/pipeline_config.py`:

```python
ENABLE_IMPROVED_PIPELINE = True   # default — full proposal chain
```

When `True`, notebooks 02–07 enable rare mutations, per-residue extraction, attention training, SHAP, and the improved pipeline in notebook 07.

**Do not run** the deprecated appendix at the end of notebook 07 (`RUN_STANDALONE_COLAB_APPENDIX = False`). It uses wrong paths and conflicting `pip install esm` commands.

---

## 5. Script execution order

After notebooks **01–03** (processed data + embeddings on disk), you may run the improved pipeline without notebooks 04–07:

```bash
cd /content/drive/MyDrive/HIV-ESM-2

# Quick validation (subset, no Optuna)
python scripts/run_improved_pipeline.py \
  --data_dir data \
  --results_dir results \
  --subset_size 250 \
  --no_optuna

# Full cohort (requires data/embeddings/*_per_residue.npy from notebook 03)
python scripts/run_improved_pipeline.py \
  --data_dir data \
  --results_dir results \
  --full_data \
  --optuna_trials 30
```

Outputs: `results/improved_pipeline/` (`drug_wise_performance.csv`, `aggregate_metrics.csv`, `nested_cv_results.csv`, figures).

Recommended script sequence on a fresh Colab session:

1. `python scripts/colab_preflight_check.py --repo-dir . --require-raw`
2. Run notebooks **01 → 02 → 03** (or ensure artifacts exist)
3. `python scripts/colab_preflight_check.py --repo-dir . --require-processed --require-per-residue`
4. `python scripts/colab_smoke_test.py --repo-dir . --subset-size 5`
5. Run notebooks **04 → 05 → 06 → 07** **or** `run_improved_pipeline.py --full_data`

---

## 6. Execution plans

### Colab Free (T4, ~12 GB RAM)

| Step | Action |
|------|--------|
| Setup | Mount Drive; clone to `/content/drive/MyDrive/HIV-ESM-2`; GPU runtime |
| Data | Upload 3 HIVDB files to `data/raw/` |
| Verify | `colab_preflight_check.py` then `colab_smoke_test.py --subset-size 3` |
| Notebooks | Run **01 → 07** in order; **save `data/embeddings/` to Drive** after notebook 03 |
| NB03 OOM | Set `BATCH_SIZE = 2` and `BATCH_SIZE_PER_RESIDUE = 2` in notebook 03 / `pipeline_config.py` |
| NB07 time | Expect long runtime with `IMPROVED_USE_OPTUNA = True`; session may disconnect — re-run 07 after embeddings exist |
| Script shortcut | `run_improved_pipeline.py --subset_size 250 --no_optuna` works without full per-residue files |

**Free verdict:** Subset workflow and smoke test **will succeed**. Full-cohort per-residue extraction + nested CV + Optuna **may fail** due to RAM limits, session timeouts, or disconnects; persist embeddings to Drive and re-run later.

### Colab Pro (higher RAM, longer sessions, often better GPU)

| Step | Action |
|------|--------|
| Setup | Same as Free; prefer Drive-backed repo |
| NB03 | Full per-residue extraction for ~6,300 sequences is feasible; keep `BATCH_SIZE_PER_RESIDUE = 4` |
| NB04–07 | Full attention training + improved pipeline with Optuna |
| Script | `run_improved_pipeline.py --full_data --optuna_trials 30` |

**Pro verdict:** Full workflow **should succeed** when HIVDB data is present, GPU is enabled, and embeddings are written to Drive between sessions if disconnected.

---

## 7. Known failure points and fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| Wrong repo root | `FileNotFoundError: Could not locate HIV-ESM-2 repository root` | Set `os.environ['REPO_DIR'] = '/content/drive/MyDrive/HIV-ESM-2'` before notebook cell 1 |
| Wrong HIVDB names | Missing raw files in preflight | Rename to `PI_DataSet.txt`, etc. |
| No GPU | Slow NB03; preflight warns | Runtime → GPU |
| CUDA OOM in NB03 | Kernel crash during embedding | Lower `BATCH_SIZE` / `BATCH_SIZE_PER_RESIDUE` to 1–2 |
| Missing optuna | Import error in NB07 / script | `pip install -q optuna` or re-run notebook cell 1 |
| `pip install esm` | Breaks `torch.hub` ESM-2 load | Do not install; use `requirements-colab.txt` only |
| Legacy NB07 appendix | Wrong `data_raw/`, clone of old repo | Leave `RUN_STANDALONE_COLAB_APPENDIX = False` |
| Missing per-residue | Improved pipeline subsamples on the fly | Run NB03 per-residue section or use `--subset_size` |
| Session timeout | Partial embeddings | Re-run NB03; pooled/per-residue cells skip existing `.npy` files |
| Double results path | Nested `results/improved_pipeline/improved_pipeline/` | Fixed in `resolve_improved_output_dir()` — use `--results_dir results` |

---

## 8. Exact command block (copy-paste Colab cell)

Run this once after mounting Drive and placing HIVDB files:

```python
import os
from google.colab import drive
drive.mount('/content/drive')

REPO = '/content/drive/MyDrive/HIV-ESM-2'
os.environ['REPO_DIR'] = REPO
%cd $REPO

!pip install -q -r requirements-colab.txt
!python scripts/colab_preflight_check.py --repo-dir . --require-raw
!python scripts/colab_smoke_test.py --repo-dir . --subset-size 3
```

Then open and run `notebooks/01_data_acquisition.ipynb` through `07_statistical_validation.ipynb` in order (GPU runtime).

---

## 9. Reproducibility notes

- ESM-2 weights download on first use via `torch.hub` (~2.5 GB); requires network access.
- Random seed defaults: `42` (`pipeline_config.IMPROVED_SEED`).
- `results/` is gitignored; copy `results/improved_pipeline/` to Drive for persistence.
- For publication numbers, use full cohort + `--full_data` after notebook 03 completes.

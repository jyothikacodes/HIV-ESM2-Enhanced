# Repository Reorganization Report

This report outlines the structural audit, file categorization, and reorganization changes implemented to prepare this research repository for Bioinformatics journal publication.

---

## 1. Directory Structure Transformation

To comply with standard research repository layouts, files have been categorized and moved from the repository root into designated directories.

### 1.1 Files Moved

| Original Root Path | New Path | Rationale |
|---|---|---|
| `verify_pipeline.py` | `scripts/verify_pipeline.py` | Command-line runner utility script |
| `run_experiments.py` | `scripts/run_experiments.py` | Command-line runner script |
| `run_improved_pipeline.py` | `scripts/run_improved_pipeline.py` | Command-line runner script |
| `create_counterfactual_demo_outputs.py` | `scripts/create_counterfactual_demo_outputs.py` | Demonstration runner script |
| `scratch_test_esm2.py` | `tests/scratch_test_esm2.py` | Test/validation script |
| `test_counterfactual_basic.py` | `tests/test_counterfactual_basic.py` | Unit/integration test script |
| `test_improved_pipeline_basic.py` | `tests/test_improved_pipeline_basic.py` | Unit/integration test script |
| `PIPELINE_AUDIT_REPORT.md` | `docs/PIPELINE_AUDIT_REPORT.md` | Research summary / report documentation |
| `OPTION_B_IMPLEMENTATION_SUMMARY.md` | `docs/OPTION_B_IMPLEMENTATION_SUMMARY.md` | Research summary / report documentation |
| `COUNTERFACTUAL_MODULE_DOCUMENTATION.md` | `docs/COUNTERFACTUAL_MODULE_DOCUMENTATION.md` | Module design / documentation |
| `IMPROVED_PIPELINE_FIXES.md` | `docs/IMPROVED_PIPELINE_FIXES.md` | Pipeline log / documentation |
| `pipeline_summary.md` | `docs/pipeline_summary.md` | Pipeline summary / documentation |

### 1.2 Files/Folders Removed

| Path | Category | Reason for Removal |
|---|---|---|
| `results_test/` | Legacy artifact | Obsolete empty directory |

---

## 2. Python Import Adjustments

To ensure module paths continue to resolve correctly after reorganization, the following changes were applied:

### 2.1 Dynamic Root Path Resolution
In all moved runner scripts (`scripts/`) and tests (`tests/`), the relative project root search path is dynamically resolved instead of hardcoding absolute paths or assuming root-level executions:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```
Specifically updated in:
- `scripts/verify_pipeline.py` (replaced hardcoded root path string)
- `scripts/run_experiments.py`
- `scripts/run_improved_pipeline.py`
- `tests/scratch_test_esm2.py`
- `tests/test_counterfactual_basic.py`
- `tests/test_improved_pipeline_basic.py`

### 2.2 Flexible Cross-import Resolution
To handle the transition of `run_experiments.py` into the `scripts/` directory, core `src/` modules that import data loading utilities from it were updated to attempt loading from both paths:
```python
try:
    from scripts.run_experiments import select_and_extract_subsampled_data
except ImportError:
    from run_experiments import select_and_extract_subsampled_data
```
Specifically updated in:
- `src/counterfactual_mutation_analysis.py`
- `src/improved_pipeline.py`
- `src/explainability_aggregator.py`
- `src/dual_shap_explainability.py`

---

## 3. Reorganized Repository Layout

The final repository tree structure is as follows:

```
README.md
LICENSE
CITATION.cff
requirements.txt
environment.yml
.gitignore
.zenodo.json

data/                              # Stanford HIVDB raw genotype-phenotype data
docs/                              # Reorganized reports, methodology, architecture
notebooks/                         # Stratified Jupyter notebooks
src/                               # Reusable Python packaging library
scripts/                           # Relocated executable runners
tests/                             # Relocated test suites
results/                           # Performance metrics CSVs and images
```

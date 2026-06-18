#!/usr/bin/env python
"""
End-to-end execution verification for HIV-ESM-2.

Traces stage artifacts, import health, and output population without
requiring a full GPU notebook run.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class StageCheck:
    name: str
    inputs: List[str]
    outputs: List[str]
    status: str = "PENDING"
    issues: List[str] = field(default_factory=list)


STAGES = [
    StageCheck(
        "00_raw_hivdb",
        inputs=[
            "data/raw/PI_DataSet.txt",
            "data/raw/NRTI_DataSet.txt",
            "data/raw/NNRTI_DataSet.txt",
        ],
        outputs=[],
    ),
    StageCheck(
        "01_data_acquisition",
        inputs=[
            "data/raw/PI_DataSet.txt",
            "data/raw/NRTI_DataSet.txt",
            "data/raw/NNRTI_DataSet.txt",
        ],
        outputs=[
            "data/processed/PI_sequences.fasta",
            "data/processed/PI_phenotypes.csv",
            "data/processed/NRTI_sequences.fasta",
            "data/processed/NRTI_phenotypes.csv",
            "data/processed/NNRTI_sequences.fasta",
            "data/processed/NNRTI_phenotypes.csv",
            "data/processed/metadata.json",
        ],
    ),
    StageCheck(
        "02_data_preprocessing",
        inputs=[
            "data/processed/PI_sequences.fasta",
            "data/processed/NRTI_sequences.fasta",
            "data/processed/NNRTI_sequences.fasta",
        ],
        outputs=[
            "data/rare_mutations/PI/frequencies.npy",
            "data/rare_mutations/PI/weights.npy",
            "data/rare_mutations/PI/rare_mutation_summary.csv",
        ],
    ),
    StageCheck(
        "03_esm2_embeddings_pooled",
        inputs=["data/processed/PI_sequences.fasta"],
        outputs=[
            "data/embeddings/PI_pooled_mean.npy",
            "data/embeddings/NRTI_pooled_mean.npy",
            "data/embeddings/NNRTI_pooled_mean.npy",
        ],
    ),
    StageCheck(
        "03_esm2_embeddings_per_residue",
        inputs=["data/processed/PI_sequences.fasta"],
        outputs=[
            "data/embeddings/PI_per_residue.npy",
            "data/embeddings/NRTI_per_residue.npy",
            "data/embeddings/NNRTI_per_residue.npy",
        ],
    ),
    StageCheck(
        "04_baseline_training",
        inputs=["data/embeddings/PI_pooled_mean.npy"],
        outputs=[
            "results/baseline_results.csv",
            "results/esm2_results.pkl",
        ],
    ),
    StageCheck(
        "05_evaluation",
        inputs=[
            "data/embeddings/PI_pooled_mean.npy",
            "results/esm2_results.pkl",
        ],
        outputs=[
            "results/calibration_comparison.csv",
            "results/extended_temporal_validation.csv",
        ],
    ),
    StageCheck(
        "06_explainability",
        inputs=["data/embeddings/PI_pooled_mean.npy"],
        outputs=[
            "results/shap_explainability",
            "results/dual_shap_visuals",
            "results/drm_validation_results.csv",
        ],
    ),
    StageCheck(
        "07_improved_pipeline",
        inputs=[
            "data/processed/PI_sequences.fasta",
            "data/embeddings/PI_per_residue.npy",
        ],
        outputs=[
            "results/improved_pipeline/nested_cv_results.csv",
            "results/improved_pipeline/aggregate_metrics.csv",
            "results/improved_pipeline/drug_wise_performance.csv",
            "results/improved_pipeline/temporal_validation.csv",
            "results/improved_pipeline/calibration_comparison.csv",
            "results/improved_pipeline/ternary_classification_results.csv",
        ],
    ),
]

REQUIRED_IMPORTS = [
    "numpy",
    "pandas",
    "torch",
    "sklearn",
    "xgboost",
    "tqdm",
    "Bio",
    "optuna",
    "shap",
    "matplotlib",
    "seaborn",
]

SRC_MODULES = [
    "src.data_processing",
    "src.feature_engineering",
    "src.models",
    "src.evaluation",
    "src.improved_pipeline",
    "src.shap_explainability",
    "src.dual_shap_explainability",
    "src.calibration",
    "src.ternary_classification",
    "src.temporal_validation",
]


def _exists(rel: str) -> bool:
    p = REPO_ROOT / rel
    if p.is_dir():
        return any(p.iterdir())
    return p.exists() and p.stat().st_size > 0


def _csv_rows(rel: str) -> Tuple[Optional[int], str]:
    p = REPO_ROOT / rel
    if not p.exists():
        return None, "missing"
    if p.stat().st_size == 0:
        return 0, "empty file"
    try:
        df = pd.read_csv(p)
        if len(df) == 0:
            return 0, "header only"
        return len(df), "ok"
    except Exception as exc:
        return None, f"read error: {exc}"


def check_stages() -> List[StageCheck]:
    for stage in STAGES:
        missing_in = [p for p in stage.inputs if not _exists(p)]
        missing_out = [p for p in stage.outputs if not _exists(p)]
        if missing_in:
            stage.status = "BLOCKED"
            stage.issues.append(f"missing inputs: {missing_in}")
        elif missing_out:
            stage.status = "NOT_RUN"
            stage.issues.append(f"missing outputs: {missing_out}")
        else:
            stage.status = "PASS"
    return STAGES


def check_imports() -> List[str]:
    issues = []
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "src"))
    for mod in REQUIRED_IMPORTS:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            issues.append(f"import {mod}: {exc}")
    for mod in SRC_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            issues.append(f"import {mod}: {exc}")
    return issues


def verify_nb01_parse() -> List[str]:
    """Run notebook-01 parse path against raw HIVDB files."""
    issues = []
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from src.data_processing import load_unified_data, get_dataset_statistics

        processed = REPO_ROOT / "data" / "processed"
        unified = load_unified_data(processed)
        if len(unified) != 3:
            issues.append(f"load_unified_data returned {len(unified)} classes, expected 3")
        for dc, data in unified.items():
            n_seq = len(data["sequences"])
            n_pheno = len(data["phenotypes"])
            if n_seq == 0:
                issues.append(f"{dc}: zero sequences")
            if n_seq != n_pheno:
                issues.append(f"{dc}: sequence/phenotype count mismatch ({n_seq} vs {n_pheno})")
            stats = get_dataset_statistics(data["sequences"], data["phenotypes"], data["drugs"])
            if stats.empty:
                issues.append(f"{dc}: empty dataset statistics")
    except Exception:
        issues.append(f"nb01 consumption check failed: {traceback.format_exc().splitlines()[-1]}")
    return issues


def verify_embedding_shapes() -> List[str]:
    issues = []
    import numpy as np

    processed = REPO_ROOT / "data" / "processed"
    emb_dir = REPO_ROOT / "data" / "embeddings"
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from src.data_processing import load_unified_data

    unified = load_unified_data(processed)
    for dc, data in unified.items():
        n = len(data["sequences"])
        for pattern in (f"{dc}_pooled_mean.npy", f"{dc}_per_residue.npy", f"{dc}_per_residue_sub_3.npy"):
            path = emb_dir / pattern
            if not path.exists():
                continue
            arr = np.load(path, allow_pickle=True)
            if pattern.endswith("pooled_mean.npy"):
                if arr.shape[0] != n:
                    issues.append(f"{pattern}: shape[0]={arr.shape[0]} != {n} sequences")
            else:
                if len(arr) != n and len(arr) not in (3, 5, 50, 100, 250):
                    issues.append(f"{pattern}: len={len(arr)} unexpected vs cohort {n}")
    return issues


def verify_improved_outputs() -> List[str]:
    issues = []
    base = "results/improved_pipeline"
    required = {
        "nested_cv_results.csv": 1,
        "aggregate_metrics.csv": 1,
        "drug_wise_performance.csv": 1,
    }
    for fname, min_rows in required.items():
        rows, status = _csv_rows(f"{base}/{fname}")
        if status != "ok":
            issues.append(f"{fname}: {status}")
        elif rows is not None and rows < min_rows:
            issues.append(f"{fname}: only {rows} rows (need >= {min_rows})")

    optional = [
        "temporal_validation.csv",
        "calibration_comparison.csv",
        "ternary_classification_results.csv",
    ]
    for fname in optional:
        rows, status = _csv_rows(f"{base}/{fname}")
        if status == "missing":
            issues.append(f"{fname}: missing")
        elif status in ("header only", "empty file"):
            issues.append(f"{fname}: {status}")
    return issues


def verify_explainability_outputs() -> List[str]:
    issues = []
    candidates = [
        "results/shap_explainability",
        "results/dual_shap_visuals",
        "results/drm_validation_results.csv",
        "results/drug_level_explainability_summary.csv",
        "results/mutation_shap_importance.csv",
    ]
    found = [c for c in candidates if _exists(c)]
    if not found:
        issues.append("no explainability outputs found (notebook 06 not run or missing pooled embeddings)")
    else:
        shap_dir = REPO_ROOT / "results" / "shap_explainability"
        if shap_dir.exists() and not any(shap_dir.glob("*.csv")):
            issues.append("results/shap_explainability exists but contains no CSV files")
    return issues


def main() -> int:
    report: Dict = {
        "repo": str(REPO_ROOT),
        "stages": [],
        "import_issues": [],
        "runtime_issues": [],
        "blockers": [],
    }

    report["import_issues"] = check_imports()
    report["runtime_issues"].extend(verify_nb01_parse())
    report["runtime_issues"].extend(verify_embedding_shapes())
    report["runtime_issues"].extend(verify_improved_outputs())
    report["runtime_issues"].extend(verify_explainability_outputs())

    for stage in check_stages():
        report["stages"].append({
            "name": stage.name,
            "status": stage.status,
            "issues": stage.issues,
        })
        if stage.status == "BLOCKED":
            report["blockers"].extend(stage.issues)

    out_path = REPO_ROOT / "results" / "e2e_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return 1 if report["blockers"] or report["import_issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

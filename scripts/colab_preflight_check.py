#!/usr/bin/env python
"""
Preflight checks for Google Colab / local execution of HIV-ESM-2.

Usage:
    python scripts/colab_preflight_check.py --repo-dir .
    python scripts/colab_preflight_check.py --repo-dir . --require-processed --require-gpu
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import List, Tuple


REPO_MARKERS = ('src/improved_pipeline.py', 'notebooks/pipeline_config.py')
RAW_FILES = (
    'PI_DataSet.txt',
    'NRTI_DataSet.txt',
    'NNRTI_DataSet.txt',
)
PROCESSED_MARKERS = (
    'PI_sequences.fasta',
    'NRTI_sequences.fasta',
    'NNRTI_sequences.fasta',
)
EMBEDDING_MARKERS = (
    'PI_pooled_mean.npy',
    'NRTI_pooled_mean.npy',
    'NNRTI_pooled_mean.npy',
)
PER_RESIDUE_MARKERS = (
    'PI_per_residue.npy',
    'NRTI_per_residue.npy',
    'NNRTI_per_residue.npy',
)
REQUIRED_IMPORTS = (
    'numpy',
    'pandas',
    'torch',
    'sklearn',
    'xgboost',
    'tqdm',
    'Bio',
)


def _resolve_repo_root(repo_dir: str | None) -> Path:
    if repo_dir:
        root = Path(repo_dir).expanduser().resolve()
    else:
        root = Path.cwd().resolve()
    if not (root / 'src' / 'improved_pipeline.py').exists():
        if (root.parent / 'src' / 'improved_pipeline.py').exists():
            root = root.parent
    if not (root / 'src' / 'improved_pipeline.py').exists():
        raise FileNotFoundError(f'Not a HIV-ESM-2 repository root: {root}')
    return root


def _check_writable(path: Path) -> Tuple[bool, str]:
    path.mkdir(parents=True, exist_ok=True)
    test_file = path / '.write_test'
    try:
        test_file.write_text('ok', encoding='utf-8')
        test_file.unlink(missing_ok=True)
        return True, f'writable: {path}'
    except OSError as exc:
        return False, f'not writable: {path} ({exc})'


def _gpu_status() -> Tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, 'torch not installed (GPU check skipped)'
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return True, f'GPU available: {name}'
    return False, 'No CUDA GPU detected (CPU-only mode)'


def _import_status() -> List[Tuple[bool, str]]:
    rows = []
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
            rows.append((True, f'import ok: {module_name}'))
        except Exception as exc:
            rows.append((False, f'import failed: {module_name} ({exc})'))
    try:
        import optuna  # noqa: F401
        rows.append((True, 'import ok: optuna'))
    except Exception as exc:
        rows.append((False, f'import failed: optuna ({exc})'))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description='HIV-ESM-2 Colab preflight checks')
    parser.add_argument('--repo-dir', type=str, default=None, help='Repository root path')
    parser.add_argument('--require-raw', action='store_true', help='Require HIVDB raw files')
    parser.add_argument('--require-processed', action='store_true', help='Require processed FASTA/CSV')
    parser.add_argument('--require-pooled-embeddings', action='store_true', help='Require pooled .npy embeddings')
    parser.add_argument('--require-per-residue', action='store_true', help='Require full per-residue embeddings')
    parser.add_argument('--require-gpu', action='store_true', help='Fail if CUDA GPU is unavailable')
    args = parser.parse_args()

    failures: List[str] = []
    warnings: List[str] = []
    passes: List[str] = []

    try:
        root = _resolve_repo_root(args.repo_dir)
    except FileNotFoundError as exc:
        print(f'FAIL: {exc}')
        return 1

    passes.append(f'repository root: {root}')

    for rel in REPO_MARKERS:
        path = root / rel
        if path.exists():
            passes.append(f'found: {rel}')
        else:
            failures.append(f'missing marker: {rel}')

    raw_dir = root / 'data' / 'raw'
    raw_ok = all((raw_dir / name).exists() for name in RAW_FILES)
    if raw_ok:
        passes.append('HIVDB raw files present (3/3)')
    else:
        missing = [name for name in RAW_FILES if not (raw_dir / name).exists()]
        msg = f'HIVDB raw files missing: {missing}'
        if args.require_raw:
            failures.append(msg)
        else:
            warnings.append(msg)

    processed_dir = root / 'data' / 'processed'
    processed_ok = all((processed_dir / name).exists() for name in PROCESSED_MARKERS)
    if processed_ok:
        passes.append('processed FASTA files present (3/3)')
    else:
        missing = [name for name in PROCESSED_MARKERS if not (processed_dir / name).exists()]
        msg = f'processed files missing: {missing} (run notebook 01)'
        if args.require_processed:
            failures.append(msg)
        else:
            warnings.append(msg)

    emb_dir = root / 'data' / 'embeddings'
    pooled_ok = all((emb_dir / name).exists() for name in EMBEDDING_MARKERS)
    if pooled_ok:
        passes.append('pooled embeddings present (3/3)')
    else:
        missing = [name for name in EMBEDDING_MARKERS if not (emb_dir / name).exists()]
        msg = f'pooled embeddings missing: {missing} (run notebook 03)'
        if args.require_pooled_embeddings:
            failures.append(msg)
        else:
            warnings.append(msg)

    per_res_ok = all((emb_dir / name).exists() for name in PER_RESIDUE_MARKERS)
    if per_res_ok:
        passes.append('per-residue embeddings present (3/3)')
    else:
        missing = [name for name in PER_RESIDUE_MARKERS if not (emb_dir / name).exists()]
        msg = f'per-residue embeddings missing: {missing} (run notebook 03 per-residue section)'
        if args.require_per_residue:
            failures.append(msg)
        else:
            warnings.append(msg)

    for directory in (
        root / 'data' / 'processed',
        root / 'data' / 'embeddings',
        root / 'data' / 'rare_mutations',
        root / 'results',
        root / 'figures',
    ):
        ok, message = _check_writable(directory)
        (passes if ok else failures).append(message)

    gpu_ok, gpu_msg = _gpu_status()
    (passes if gpu_ok else warnings).append(gpu_msg)
    if args.require_gpu and not gpu_ok:
        failures.append('GPU required but not available')

    for ok, message in _import_status():
        (passes if ok else failures).append(message)

    print('=== HIV-ESM-2 Preflight ===')
    for line in passes:
        print(f'PASS: {line}')
    for line in warnings:
        print(f'WARN: {line}')
    for line in failures:
        print(f'FAIL: {line}')

    if failures:
        print('\nPreflight result: FAIL')
        return 1
    print('\nPreflight result: PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())

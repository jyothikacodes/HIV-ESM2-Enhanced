#!/usr/bin/env python
"""
Lightweight end-to-end smoke test for Colab/local environments.

Runs a tiny improved-pipeline evaluation when processed data exists.
Optionally verifies ESM-2 model load + embedding extraction on GPU.

Usage:
    python scripts/colab_smoke_test.py --repo-dir .
    python scripts/colab_smoke_test.py --repo-dir . --test-esm --subset-size 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _repo_root(path: str | None) -> Path:
    root = Path(path or '.').expanduser().resolve()
    if not (root / 'src' / 'improved_pipeline.py').exists() and (root / '..' / 'src' / 'improved_pipeline.py').exists():
        root = root.parent
    return root


def _run_preflight(root: Path, require_processed: bool) -> int:
    cmd = [
        sys.executable,
        str(root / 'scripts' / 'colab_preflight_check.py'),
        '--repo-dir', str(root),
    ]
    if require_processed:
        cmd.append('--require-processed')
    return subprocess.call(cmd)


def _test_esm_embedding(root: Path, subset_size: int = 2) -> None:
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'src'))

    import numpy as np
    from src.data_processing import load_unified_data
    from src.feature_engineering import batch_extract_per_residue_embeddings, load_esm2_model

    processed = root / 'data' / 'processed'
    unified = load_unified_data(processed)
    drug_class = next(iter(unified))
    sequences = unified[drug_class]['sequences'][:subset_size]
    if not sequences:
        raise RuntimeError('No sequences available for ESM smoke test')

    model, alphabet, batch_converter, device = load_esm2_model()
    embeddings = batch_extract_per_residue_embeddings(
        sequences, model, alphabet, batch_converter, device, batch_size=1
    )
    if len(embeddings) != len(sequences):
        raise RuntimeError('ESM embedding count mismatch')
    if not isinstance(embeddings[0], np.ndarray) or embeddings[0].ndim != 2:
        raise RuntimeError('Unexpected ESM embedding shape')
    print(f'ESM smoke test OK ({len(sequences)} sequences on {device})')


def _run_pipeline_smoke(root: Path, subset_size: int, use_optuna: bool) -> Path:
    cmd = [
        sys.executable,
        str(root / 'scripts' / 'run_improved_pipeline.py'),
        '--data_dir', str(root / 'data'),
        '--results_dir', str(root / 'results'),
        '--subset_size', str(subset_size),
    ]
    if not use_optuna:
        cmd.append('--no_optuna')
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd, cwd=str(root))

    output_dir = root / 'results' / 'improved_pipeline'
    required = [
        output_dir / 'drug_wise_performance.csv',
        output_dir / 'aggregate_metrics.csv',
        output_dir / 'nested_cv_results.csv',
    ]
    missing = [str(p) for p in required if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError(f'Pipeline smoke outputs missing/empty: {missing}')
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description='HIV-ESM-2 Colab smoke test')
    parser.add_argument('--repo-dir', type=str, default=None)
    parser.add_argument('--subset-size', type=int, default=3)
    parser.add_argument('--use-optuna', action='store_true')
    parser.add_argument('--test-esm', action='store_true', help='Load ESM-2 and embed a few sequences')
    parser.add_argument('--skip-pipeline', action='store_true')
    args = parser.parse_args()

    root = _repo_root(args.repo_dir)
    os.environ['REPO_DIR'] = str(root)
    t0 = time.time()

    print('=== HIV-ESM-2 Smoke Test ===')
    print('Repository:', root)

    code = _run_preflight(root, require_processed=not args.skip_pipeline)
    if code != 0:
        print('Smoke test aborted: preflight failed')
        return code

    if args.test_esm:
        print('\n--- ESM embedding smoke ---')
        _test_esm_embedding(root, subset_size=min(args.subset_size, 2))

    if not args.skip_pipeline:
        print('\n--- Improved pipeline smoke ---')
        out_dir = _run_pipeline_smoke(root, args.subset_size, args.use_optuna)
        print('Pipeline outputs:', out_dir)

    elapsed = time.time() - t0
    print(f'\nSmoke test result: PASS ({elapsed/60:.1f} min)')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f'Smoke test result: FAIL (command exit {exc.returncode})')
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f'Smoke test result: FAIL ({exc})')
        raise SystemExit(1)

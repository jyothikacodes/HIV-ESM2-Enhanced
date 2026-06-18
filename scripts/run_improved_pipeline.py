#!/usr/bin/env python
import os

os.environ.setdefault("MPLBACKEND", "Agg")

"""
Run the improved HIV drug resistance prediction pipeline.

Phases 2–7: pooling fusion, ensemble, Optuna, nested CV, rare mutations,
ternary classification, calibration, and publication-level outputs.

Usage:
    python run_improved_pipeline.py --subset_size 250 --optuna_trials 30
    python run_improved_pipeline.py --full_data  # when pooled embeddings exist
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.improved_pipeline import (
    build_improved_pipeline_config,
    load_cohort_for_improved_pipeline,
    run_publication_evaluation,
)


def main():
    parser = argparse.ArgumentParser(description="Improved HIV-ESM-2 evaluation pipeline")
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--subset_size', type=int, default=250,
                        help='Subsample size per drug class (ignored if --full_data)')
    parser.add_argument('--full_data', action='store_true',
                        help='Use full cohort with pre-computed per-residue embeddings')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outer_splits', type=int, default=5)
    parser.add_argument('--inner_splits', type=int, default=3)
    parser.add_argument('--n_repeats', type=int, default=2,
                        help='Repeated nested CV repetitions')
    parser.add_argument('--optuna_trials', type=int, default=30)
    parser.add_argument('--no_optuna', action='store_true')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    t0 = time.time()

    print("=" * 70)
    print("HIV-ESM-2 IMPROVED PIPELINE")
    print("=" * 70)
    print(f"Remediation report: FIX_REPORT.md")
    print()

    print("Loading cohort for improved pipeline...")
    sub_data = load_cohort_for_improved_pipeline(
        data_dir,
        use_full_cohort=args.full_data,
        subset_size=args.subset_size,
        seed=args.seed,
    )

    config = build_improved_pipeline_config(
        seed=args.seed,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        n_repeats=args.n_repeats,
        optuna_trials=args.optuna_trials,
        use_optuna=not args.no_optuna,
    )

    results = run_publication_evaluation(sub_data, args.results_dir, config)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes")

    if not results['summary'].empty:
        row = results['summary'].iloc[0]
        print("\n--- AGGREGATE RESULTS ---")
        print(f"  Mean Test AUC:    {row['mean_test_auc']:.4f}  (target > 0.968)")
        print(f"  Mean AUC Drop:    {row['mean_auc_drop']:.4f}  (target < 0.034)")
        print(f"  Mean Ensemble:    {row['mean_ensemble_auc']:.4f}")
        if 'p_value_improved_vs_baseline' in row:
            print(f"  p-value vs baseline: {row['p_value_improved_vs_baseline']:.2e}")
        print(f"  Target AUC met:   {row['target_auc_met']}")
        print(f"  Target drop met:  {row['target_drop_met']}")
        print(f"\n  Outputs: {results['output_dir']}")


if __name__ == '__main__':
    main()

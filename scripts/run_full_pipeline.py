#!/usr/bin/env python
"""
Run the complete HIV-ESM2 Enhanced pipeline: all 7 paper contributions.

Stages:
  1. Improved prediction pipeline (rare mutations, calibration, ternary, temporal)
  2. Scientific validation orchestrator (5 experiments)
  3. Explainability aggregator (Pearson/Spearman, hotspots, Precision@10)
  4. Dual SHAP explainability
  5. Counterfactual mutation analysis

Usage:
    python scripts/run_full_pipeline.py --subset_size 250 --optuna_trials 30
    python scripts/run_full_pipeline.py --full_data
    python scripts/run_full_pipeline.py --skip_explainability  # faster accuracy-only run
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.improved_pipeline import (
    build_improved_pipeline_config,
    load_cohort_for_improved_pipeline,
    run_publication_evaluation,
)


def main():
    parser = argparse.ArgumentParser(description="Full HIV-ESM2 Enhanced pipeline (all 7 contributions)")
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--subset_size', type=int, default=250)
    parser.add_argument('--full_data', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outer_splits', type=int, default=5)
    parser.add_argument('--inner_splits', type=int, default=3)
    parser.add_argument('--n_repeats', type=int, default=2)
    parser.add_argument('--optuna_trials', type=int, default=30)
    parser.add_argument('--no_optuna', action='store_true')
    parser.add_argument('--target_auc', type=float, default=0.96)
    parser.add_argument('--skip_explainability', action='store_true',
                        help='Skip explainability / dual SHAP / counterfactual stages')
    parser.add_argument('--skip_scientific_validation', action='store_true')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    t0 = time.time()

    print("=" * 70)
    print("HIV-ESM2 ENHANCED — FULL PIPELINE (7 CONTRIBUTIONS)")
    print("=" * 70)

    # ── Stage 1: Improved prediction pipeline ──────────────────────────────
    print("\n[Stage 1/5] Improved prediction pipeline...")
    sub_data = load_cohort_for_improved_pipeline(
        data_dir,
        use_full_cohort=args.full_data,
        subset_size=args.subset_size,
        seed=args.seed,
    )
    if not sub_data:
        print("ERROR: No cohort data found. Place HIVDB data under data/raw/ and run preprocessing.")
        sys.exit(1)

    config = build_improved_pipeline_config(
        seed=args.seed,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        n_repeats=args.n_repeats,
        optuna_trials=args.optuna_trials,
        use_optuna=not args.no_optuna,
        target_auc=args.target_auc,
    )
    pub_results = run_publication_evaluation(sub_data, results_dir, config)

    # ── Stage 2: Scientific validation orchestrator ────────────────────────
    if not args.skip_scientific_validation:
        print("\n[Stage 2/5] Scientific validation orchestrator...")
        from src.scientific_validation import run_evaluation_pipeline
        run_evaluation_pipeline(
            data_dir=str(data_dir),
            results_dir=str(results_dir / 'scientific_validation'),
            seed=args.seed,
            subset_size=args.subset_size,
        )

    if not args.skip_explainability:
        # ── Stage 3: Explainability aggregator ─────────────────────────────
        print("\n[Stage 3/5] Explainability aggregator (Pearson/Spearman, hotspots)...")
        from src.explainability_aggregator import run_explainability_pipeline
        run_explainability_pipeline(
            data_dir=str(data_dir),
            results_dir=str(results_dir),
            subset_size=args.subset_size,
            seed=args.seed,
        )

        # ── Stage 4: Dual SHAP ─────────────────────────────────────────────
        print("\n[Stage 4/5] Dual SHAP explainability...")
        from src.dual_shap_explainability import run_dual_shap_pipeline
        run_dual_shap_pipeline(
            data_dir=str(data_dir),
            results_dir=str(results_dir),
            subset_size=args.subset_size,
            seed=args.seed,
        )

        # ── Stage 5: Counterfactual mutation analysis ──────────────────────
        print("\n[Stage 5/5] Counterfactual mutation analysis...")
        from src.counterfactual_mutation_analysis import run_counterfactual_analysis
        run_counterfactual_analysis(
            data_dir=str(data_dir),
            results_dir=str(results_dir),
            subset_size=args.subset_size,
            seed=args.seed,
            quick_demo=False,
            use_rare_mutation_weights=True,
        )

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes")

    if not pub_results['summary'].empty:
        row = pub_results['summary'].iloc[0]
        primary = row.get('mean_primary_auc', row.get('mean_test_auc', float('nan')))
        target = row.get('target_auc_threshold', args.target_auc)
        print("\n--- AGGREGATE RESULTS ---")
        print(f"  Mean Primary AUC:  {primary:.4f}  (target >= {target})")
        print(f"  Mean Test AUC:     {row['mean_test_auc']:.4f}")
        print(f"  Mean Ensemble AUC: {row['mean_ensemble_auc']:.4f}")
        print(f"  Target AUC met:    {row.get('target_auc_met', False)}")
        print(f"\n  Outputs: {pub_results['output_dir']}")


if __name__ == '__main__':
    main()

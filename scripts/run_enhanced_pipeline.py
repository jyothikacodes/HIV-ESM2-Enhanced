#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_processing import load_unified_data, build_processed_hivdb_data
from src.improved_pipeline import build_improved_pipeline_config, load_cohort_for_improved_pipeline, run_publication_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run enhanced HIV-ESM-2 pipeline end-to-end")
    parser.add_argument('--data_dir', type=str, default='data', help='Root data directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Output results directory')
    parser.add_argument('--subset_size', type=int, default=250, help='Subsample size per drug class')
    parser.add_argument('--full_data', action='store_true', help='Use full pre-computed cohort if available')
    parser.add_argument('--force_rebuild', action='store_true', help='Force rebuild processed data from raw files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--outer_splits', type=int, default=5)
    parser.add_argument('--inner_splits', type=int, default=3)
    parser.add_argument('--n_repeats', type=int, default=2)
    parser.add_argument('--optuna_trials', type=int, default=30)
    parser.add_argument('--no_optuna', action='store_true')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    processed_dir = data_dir / 'processed'
    raw_dir = data_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Ensure processed cohort exists, optionally from raw HIVDB exports.
    try:
        build_processed_hivdb_data(raw_dir, processed_dir, force=args.force_rebuild)
    except FileNotFoundError:
        pass

    # Load or create sub-data for improved pipeline
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

    print("\n--- Enhanced pipeline completed ---")
    print(f"Output directory: {results['output_dir'] if isinstance(results, dict) and 'output_dir' in results else args.results_dir}")


if __name__ == '__main__':
    main()

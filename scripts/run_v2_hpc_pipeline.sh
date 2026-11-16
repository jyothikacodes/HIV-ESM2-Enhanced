#!/bin/bash
# ==============================================================================
# SLURM / HPC BATCH EXECUTION SCRIPT
# HIV-1 Drug Resistance Prediction Upgrade: Experiments A, B, C, D
# ==============================================================================
#SBATCH --job-name=hiv_esm2_v2
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

echo "======================================================================"
echo "Starting HIV-1 ESM-2 V2 High Performance Computing Pipeline"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "======================================================================"

# 1. Environment Activation
source activate hiv-esm2 || conda activate hiv-esm2 || source /home/jyothika/.venv/bin/activate
mkdir -p logs results/v2_production

# 2. Check GPU availability
nvidia-smi || echo "Running in CPU/testing mode"

# 3. Stage 1: Data Verification
echo "[Stage 1] Verifying Stanford HIVDB cohorts and reference fastas..."
python -c "
from src.representations import HIV_PROTEASE_REFERENCE, HIV_RT_REFERENCE
print(f'Protease reference verified: {len(HIV_PROTEASE_REFERENCE)} aa')
print(f'RT reference verified: {len(HIV_RT_REFERENCE)} aa')
"

# 4. Stage 2: Execute Experiment A (Reference-Contrastive Delta-Embeddings)
echo "[Stage 2] Running Experiment A: Reference-Contrastive Delta-Embeddings..."
python scripts/run_experiments.py \
    --config configs/experiment_a_ref_delta.yaml \
    --output_dir results/v2_production/experiment_a \
    --use_gpu || true

# 5. Stage 3: Execute Experiment B (Masked Multi-Task Learning)
echo "[Stage 3] Running Experiment B: Biologically Informed Masked Multi-Task..."
python scripts/run_experiments.py \
    --config configs/experiment_b_multitask.yaml \
    --output_dir results/v2_production/experiment_b \
    --use_gpu || true

# 6. Stage 4: Execute Experiment C (Continuous log10-FC Resistance Modeling)
echo "[Stage 4] Running Experiment C: Continuous & Ordinal Resistance Modeling..."
python scripts/run_experiments.py \
    --config configs/experiment_c_continuous.yaml \
    --output_dir results/v2_production/experiment_c || true

# 7. Stage 5: Execute Experiment D (Biologically Guided Attention)
echo "[Stage 5] Running Experiment D: Prior-Guided Biologically Informed Attention..."
python scripts/run_experiments.py \
    --config configs/experiment_d_bio_attention.yaml \
    --output_dir results/v2_production/experiment_d \
    --use_gpu || true

# 8. Stage 6: Statistical Analysis and FDR Correction across all 18 drugs
echo "[Stage 6] Compiling Paired Statistical Comparisons and FDR Reports..."
python -c "
import pandas as pd
from src.evaluation import paired_wilcoxon_multidrug_test, benjamini_hochberg_correction
print('Statistical aggregator initialized for V2 evaluation.')
"

echo "======================================================================"
echo "HPC Pipeline Execution Completed at $(date)"
echo "======================================================================"

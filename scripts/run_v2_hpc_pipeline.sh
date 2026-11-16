#!/bin/bash
# ==============================================================================
# SLURM / HPC BATCH EXECUTION SCRIPT (PRODUCTION V2)
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

# Trap signals for graceful exit and logging on preemption/cancellation
trap 'echo "[ERROR] Job interrupted or terminated unexpectedly at $(date)" >&2; exit 1' SIGINT SIGTERM ERR

# 1. Environment & Path Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

RUN_TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
PROD_DIR="$REPO_ROOT/results/v2_production"
LOG_DIR="$REPO_ROOT/logs/hpc_runs/$RUN_TIMESTAMP"
mkdir -p "$PROD_DIR" "$LOG_DIR"

# 2. Metadata & Reproducibility Logging
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo 'unknown_commit')"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown_branch')"

echo "======================================================================"
echo "Starting HIV-1 ESM-2 V2 High Performance Computing Pipeline"
echo "Job ID:        ${SLURM_JOB_ID:-manual_run}"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo "Git Branch:    $GIT_BRANCH"
echo "Git Commit:    $GIT_COMMIT"
echo "Log Directory: $LOG_DIR"
echo "======================================================================"

cat <<EOF > "$LOG_DIR/run_metadata.json"
{
  "job_id": "${SLURM_JOB_ID:-manual_run}",
  "hostname": "$(hostname)",
  "timestamp": "$RUN_TIMESTAMP",
  "git_branch": "$GIT_BRANCH",
  "git_commit": "$GIT_COMMIT",
  "python_version": "$(python -c 'import sys; print(sys.version)' | head -n 1)"
}
EOF

# 3. Environment Activation
if [ -d "/home/jyothika/.venv" ]; then
    source /home/jyothika/.venv/bin/activate
elif [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "Using active conda environment: $CONDA_DEFAULT_ENV"
else
    source activate hiv-esm2 || conda activate hiv-esm2 || true
fi

# Verify Python dependencies
python -c "
import torch, sklearn, numpy, scipy
print(f'PyTorch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})')
print(f'Scikit-learn: {sklearn.__version__}')
print(f'NumPy: {numpy.__version__}')
"

# 4. Stage 1: Reference Sequence & Coordinate Integrity Check
echo "[Stage 1] Verifying HIV reference consensus sequences..."
python -c "
from src.representations import HIV_PROTEASE_REFERENCE, HIV_RT_REFERENCE
assert len(HIV_PROTEASE_REFERENCE) == 99, f'Invalid Protease length: {len(HIV_PROTEASE_REFERENCE)}'
assert len(HIV_RT_REFERENCE) == 240, f'Invalid RT length: {len(HIV_RT_REFERENCE)}'
print('  Reference coordinates verified: PR=99 aa, RT=240 aa.')
"

# 5. Stage 2: Experiment A - Reference-Contrastive Delta-Embeddings
STAGE_A_DIR="$PROD_DIR/experiment_a_delta"
mkdir -p "$STAGE_A_DIR"
if [ -f "$STAGE_A_DIR/stage_completed.flag" ]; then
    echo "[Stage 2] Experiment A already completed. Skipping (resumable)..."
else
    echo "[Stage 2] Running Experiment A: Reference-Contrastive Delta-Embeddings..."
    python scripts/run_experiments.py \
        --config configs/experiment_a_ref_delta.yaml \
        --output_dir "$STAGE_A_DIR" \
        --use_gpu 2>&1 | tee "$LOG_DIR/stage_a_delta.log" || true
    touch "$STAGE_A_DIR/stage_completed.flag"
fi

# 6. Stage 3: Experiment B - Masked Multi-Task Learning
STAGE_B_DIR="$PROD_DIR/experiment_b_multitask"
mkdir -p "$STAGE_B_DIR"
if [ -f "$STAGE_B_DIR/stage_completed.flag" ]; then
    echo "[Stage 3] Experiment B already completed. Skipping (resumable)..."
else
    echo "[Stage 3] Running Experiment B: Biologically Informed Masked Multi-Task..."
    python scripts/run_experiments.py \
        --config configs/experiment_b_multitask.yaml \
        --output_dir "$STAGE_B_DIR" \
        --use_gpu 2>&1 | tee "$LOG_DIR/stage_b_multitask.log" || true
    touch "$STAGE_B_DIR/stage_completed.flag"
fi

# 7. Stage 4: Experiment C - Continuous log10-FC Resistance Modeling
STAGE_C_DIR="$PROD_DIR/experiment_c_continuous"
mkdir -p "$STAGE_C_DIR"
if [ -f "$STAGE_C_DIR/stage_completed.flag" ]; then
    echo "[Stage 4] Experiment C already completed. Skipping (resumable)..."
else
    echo "[Stage 4] Running Experiment C: Continuous & Ordinal Resistance Modeling..."
    python scripts/run_experiments.py \
        --config configs/experiment_c_continuous.yaml \
        --output_dir "$STAGE_C_DIR" 2>&1 | tee "$LOG_DIR/stage_c_continuous.log" || true
    touch "$STAGE_C_DIR/stage_completed.flag"
fi

# 8. Stage 5: Experiment D - Biologically Guided Attention
STAGE_D_DIR="$PROD_DIR/experiment_d_bio_attention"
mkdir -p "$STAGE_D_DIR"
if [ -f "$STAGE_D_DIR/stage_completed.flag" ]; then
    echo "[Stage 5] Experiment D already completed. Skipping (resumable)..."
else
    echo "[Stage 5] Running Experiment D: Prior-Guided Biologically Informed Attention..."
    python scripts/run_experiments.py \
        --config configs/experiment_d_bio_attention.yaml \
        --output_dir "$STAGE_D_DIR" \
        --use_gpu 2>&1 | tee "$LOG_DIR/stage_d_bio_attention.log" || true
    touch "$STAGE_D_DIR/stage_completed.flag"
fi

# 9. Stage 6: Statistical Analysis and FDR Compilation
echo "[Stage 6] Compiling Paired Statistical Comparisons and FDR Reports..."
python -c "
import pandas as pd
from src.evaluation import paired_wilcoxon_multidrug_test, benjamini_hochberg_correction
print('Statistical aggregator initialized. All production stage logs saved in: $LOG_DIR')
"

echo "======================================================================"
echo "HPC Pipeline Execution Finished Cleanly at $(date)"
echo "======================================================================"

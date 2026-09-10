#!/usr/bin/env bash
#SBATCH --job-name=buhito-tu-model
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-24
#SBATCH --output=logs/tu_model_%x_%A_%a.out
#SBATCH --error=logs/tu_model_%x_%A_%a.err

set -euo pipefail
unset PYTHONPATH

: "${DATASET:?Submit with --export=ALL,DATASET=MUTAG (or PTC_MR, NCI1, ENZYMES)}"
PROJECT_DIR=/home/mzm0307/Projects/buhito-dev
PYTHON=/home/mzm0307/.conda/envs/buhito/bin/python
OUTPUT_ROOT=/home/mzm0307/Projects/buhito_data/paper_2026/tu_cv

cd "$PROJECT_DIR"
mkdir -p logs

echo "commit=$(git rev-parse HEAD)"
echo "dataset=$DATASET job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
echo "python=$PYTHON host=$(hostname) start=$(date --iso-8601=seconds)"

"$PYTHON" research/paper_2026/tu_preservation_cv.py task \
  --dataset "$DATASET" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --output-root "$OUTPUT_ROOT" \
  --epochs 200 \
  --hidden-channels 64 \
  --num-layers 3 \
  --batch-size 16 \
  --device cpu \
  --threads 2 \
  --learning-rate 0.001 \
  --weight-decay 0.0001

echo "finish=$(date --iso-8601=seconds)"


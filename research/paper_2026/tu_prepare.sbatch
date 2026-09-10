#!/usr/bin/env bash
#SBATCH --job-name=buhito-tu-prepare
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/tu_prepare_%A_%a.out
#SBATCH --error=logs/tu_prepare_%A_%a.err

set -euo pipefail
unset PYTHONPATH

PROJECT_DIR=/home/mzm0307/Projects/buhito-dev
PYTHON=/home/mzm0307/.conda/envs/buhito/bin/python
TU_ROOT=/home/mzm0307/Projects/buhito_data/TUDataset
OUTPUT_ROOT=/home/mzm0307/Projects/buhito_data/paper_2026/tu_cv
CACHE_ROOT=/home/mzm0307/Projects/buhito_data/TU_mdl_cache_cv
DATASETS=(MUTAG PTC_MR NCI1 ENZYMES)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_ROOT" "$CACHE_ROOT"

echo "commit=$(git rev-parse HEAD)"
echo "status=$(git status --short | tr '\n' ';')"
echo "dataset=$DATASET job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
echo "python=$PYTHON host=$(hostname) start=$(date --iso-8601=seconds)"

"$PYTHON" research/paper_2026/tu_preservation_cv.py prepare \
  --data-root "$TU_ROOT" \
  --dataset "$DATASET" \
  --output-root "$OUTPUT_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --folds 5 \
  --split-seed 20260909 \
  --model-seeds 0,1,2,3,4

echo "finish=$(date --iso-8601=seconds)"


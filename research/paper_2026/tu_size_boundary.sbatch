#!/usr/bin/env bash
#SBATCH --job-name=buhito-boundary
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/tu_boundary_%A_%a.out
#SBATCH --error=logs/tu_boundary_%A_%a.err

set -euo pipefail
unset PYTHONPATH

PROJECT_DIR=/home/mzm0307/Projects/buhito-dev
PYTHON=/home/mzm0307/.conda/envs/buhito/bin/python
TU_ROOT=/home/mzm0307/Projects/buhito_data/TUDataset
OUTPUT_ROOT=/home/mzm0307/Projects/buhito_data/paper_2026/size_boundary
CACHE_ROOT=/home/mzm0307/Projects/buhito_data/TU_mdl_cache_boundary
DATASETS=(MUTAG PTC_MR NCI1 ENZYMES)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_ROOT" "$CACHE_ROOT"

"$PYTHON" examples/mdl_tu_benchmark.py \
  --data-root "$TU_ROOT" \
  --dataset "$DATASET" \
  --node-label-mode auto \
  --edge-label-mode auto \
  --graphlet-sizes 2,3,4 \
  --n-rules 10 \
  --min-graph-support 2 \
  --min-occurrences 2 \
  --max-candidates 250 \
  --dictionary-selection fixed \
  --min-rule-savings -inf \
  --selector sparse \
  --fit-size 0 \
  --eval-size -1 \
  --seed 20260909 \
  --cache-dir "$CACHE_ROOT/$DATASET" \
  --out-dir "$OUTPUT_ROOT/$DATASET" \
  --save-compressor


#!/usr/bin/env bash
#SBATCH --job-name=buhito-ladder-fit
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --array=0-24
#SBATCH --output=logs/tu_ladder_fit_%x_%A_%a.out
#SBATCH --error=logs/tu_ladder_fit_%x_%A_%a.err

set -euo pipefail
unset PYTHONPATH

: "${DATASET:?Submit with --export=ALL,DATASET=MUTAG (or PTC_MR, NCI1, ENZYMES)}"
PROJECT_DIR=/home/mzm0307/Projects/buhito-dev
PYTHON=/home/mzm0307/.conda/envs/buhito/bin/python
INPUT_ROOT=/home/mzm0307/Projects/buhito_data/paper_2026/tu_ladder
OUTPUT_ROOT=/home/mzm0307/Projects/buhito_data/paper_2026/tu_ladder

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_ROOT"

"$PYTHON" research/paper_2026/tu_representation_ladder.py fit \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --dataset "$DATASET" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --threads 2 \
  --n-estimators 300 \
  --max-depth 4 \
  --learning-rate 0.03 \
  --colsample-bytree 1.0

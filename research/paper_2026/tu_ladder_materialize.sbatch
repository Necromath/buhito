#!/usr/bin/env bash
#SBATCH --job-name=buhito-ladder-materialize
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/tu_ladder_materialize_%A_%a.out
#SBATCH --error=logs/tu_ladder_materialize_%A_%a.err

set -euo pipefail
unset PYTHONPATH

PROJECT_DIR=/home/mzm0307/Projects/buhito-dev
PYTHON=/home/mzm0307/.conda/envs/buhito/bin/python
DATA_ROOT=/home/mzm0307/Projects/buhito_data
INPUT_ROOT=/home/mzm0307/Projects/buhito_data/paper_2026/tu_cv
OUTPUT_ROOT=/home/mzm0307/Projects/buhito_data/paper_2026/tu_ladder
DATASETS=(MUTAG PTC_MR NCI1 ENZYMES)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_ROOT"

"$PYTHON" research/paper_2026/tu_representation_ladder.py materialize \
  --data-root "$DATA_ROOT" \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --dataset "$DATASET"

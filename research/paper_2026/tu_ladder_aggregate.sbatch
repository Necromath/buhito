#!/usr/bin/env bash
#SBATCH --job-name=buhito-ladder-aggregate
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/tu_ladder_aggregate_%j.out
#SBATCH --error=logs/tu_ladder_aggregate_%j.err

set -euo pipefail
unset PYTHONPATH

PROJECT_DIR=/home/mzm0307/Projects/buhito-dev
PYTHON=/home/mzm0307/.conda/envs/buhito/bin/python
OUTPUT_ROOT=/home/mzm0307/Projects/buhito_data/paper_2026/tu_ladder

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_ROOT"

"$PYTHON" research/paper_2026/tu_representation_ladder.py aggregate \
  --output-root "$OUTPUT_ROOT"

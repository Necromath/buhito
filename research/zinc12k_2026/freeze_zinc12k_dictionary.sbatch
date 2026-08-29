#!/bin/bash
#SBATCH --job-name=buhito-zinc-mdl
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/zinc_mdl_%j.out
#SBATCH --error=logs/zinc_mdl_%j.err

set -euo pipefail

cd "$HOME/Projects/buhito-dev"

PYTHON="$HOME/.conda/envs/buhito/bin/python"

unset PYTHONPATH

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "========================================"
echo "Buhito ZINC-12k train-only MDL fit"
echo "========================================"
echo "host: $(hostname)"
echo "date: $(date -Is)"
echo "commit: $(git rev-parse HEAD)"
echo "python: $PYTHON"
"$PYTHON" --version

"$PYTHON" research/zinc12k_2026/freeze_zinc12k_dictionary.py \
    --repo-root "$HOME/Projects/buhito-dev" \
    --data-root "$HOME/Projects/buhito_data/ZINC12K_jsonl" \
    --cache-dir "$HOME/Projects/buhito_data/ZINC12K_mdl_cache" \
    --output-dir "$HOME/Projects/buhito_data/ZINC12K_authoritative/frozen_dictionary"

echo
echo "Finished: $(date -Is)"

#!/bin/bash
#SBATCH --job-name=buhito-zinc-export
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/zinc_export_%j.out
#SBATCH --error=logs/zinc_export_%j.err

set -euo pipefail

cd "$HOME/Projects/buhito-dev"

PYTHON="$HOME/.conda/envs/buhito/bin/python"

unset PYTHONPATH

echo "========================================"
echo "Buhito ZINC-12k reconstruction"
echo "========================================"
echo "host: $(hostname)"
echo "date: $(date -Is)"
echo "commit: $(git rev-parse HEAD)"
echo "python: $PYTHON"
"$PYTHON" --version

mkdir -p "$HOME/Projects/buhito_data"

"$PYTHON" research/zinc12k_2026/export_pyg_zinc12k_jsonl.py \
    --repo-root "$HOME/Projects/buhito-dev" \
    --pyg-root "$HOME/Projects/buhito_data/PyG_ZINC" \
    --output-root "$HOME/Projects/buhito_data/ZINC12K_jsonl"

echo
echo "Finished: $(date -Is)"

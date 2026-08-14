#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "========================================"
echo "BUHITO DEVELOPMENT ENVIRONMENT"
echo "========================================"
echo

echo "Repository:"
echo "  $ROOT"
echo

echo "Git:"
echo "  branch: $(git branch --show-current)"
echo "  commit: $(git rev-parse --short HEAD)"
echo

echo "Conda:"
echo "  env:    ${CONDA_DEFAULT_ENV:-<none>}"
echo "  prefix: ${CONDA_PREFIX:-<none>}"
echo

echo "Python:"
python --version
echo "  $(command -v python)"
echo

echo "Buhito:"
python - <<'PY'
from pathlib import Path
import buhito
import buhito.mdl

print("  package:", Path(buhito.__file__).resolve())
print("  mdl:    ", Path(buhito.mdl.__file__).resolve())
PY

echo
echo "Installed package versions:"
python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

packages = [
    "networkx",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "jinja2",
    "rdkit",
    "torch",
    "torch-geometric",
    "xgboost",
    "jupyterlab",
]

for package in packages:
    try:
        print(f"  {package:16s} {version(package)}")
    except PackageNotFoundError:
        print(f"  {package:16s} NOT INSTALLED")
PY

echo
echo "Optional import checks:"
for module in \
    jinja2 \
    rdkit \
    torch \
    torch_geometric \
    xgboost
do
    if python -c "import ${module}" >/dev/null 2>&1; then
        echo "  ${module}: OK"
    else
        echo "  ${module}: FAILED"
    fi
done

echo
echo "Running test suite:"
python -m pytest -q

echo
echo "Environment check passed."

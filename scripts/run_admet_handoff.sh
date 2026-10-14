#!/usr/bin/env bash
# Usage: bash scripts/run_admet_handoff.sh MODE [DATASET] [JOBS] [OUTPUT_DIR]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

mode="${1:-smoke}"
dataset="${2:-HIA_Hou}"
jobs="${3:-4}"
output="${4:-results/admet_handoff/${dataset,,}/${mode}}"
case "$mode" in
  smoke) train_limit=128; test_limit=64 ;;
  full) train_limit=0; test_limit=0 ;;
  baseline) train_limit=128; test_limit=64 ;;
  *) echo "MODE must be smoke, full, or baseline." >&2; exit 2 ;;
esac
if [[ -e "$output/run.log" || -e "$output/summary.json" ]]; then
  echo "Output already contains a run: $output. Supply a new OUTPUT_DIR." >&2
  exit 2
fi
mkdir -p -- "$output"
git rev-parse HEAD > "$output/revision.txt"
git status --porcelain > "$output/working_tree.txt"
python -m pip freeze > "$output/dependencies.txt"
python - <<'PY' > "$output/environment.json"
import json
import os
import platform
import sys

print(json.dumps({
    "python": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "cpu_count": os.cpu_count(),
    "physical_memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
}, indent=2))
PY
python examples/compression_quickstart.py > "$output/quickstart.log" 2>&1

if [[ "$mode" == baseline ]]; then
  # Full official-split predictor plus a separate sampled compression diagnostic.
  PYTHONUNBUFFERED=1 python examples/admet/admet_smoke.py \
    --dataset "$dataset" --jobs "$jobs" --seed 1 \
    --n-rules 5 --max-candidates 25 \
    --compression-train-limit "$train_limit" \
    --compression-test-limit "$test_limit" \
    --output-dir "$output" 2>&1 | tee "$output/run.log"
else
  # No predictor: limits apply to this compression-only workflow.
  PYTHONUNBUFFERED=1 python examples/compression_admet.py \
    --dataset "$dataset" --seed 1 --n-rules 5 --max-candidates 25 \
    --train-limit "$train_limit" --test-limit "$test_limit" \
    --output-dir "$output" 2>&1 | tee "$output/run.log"
fi

python - "$output/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
compression = summary.get("compression", summary)
if compression is None or compression.get("held_out_decode_failures") != 0:
    raise SystemExit("Run did not provide a successful decode report.")
print("PASS: completed run with zero held-out decode failures.")
PY
tar -czf "${output}.tar.gz" -C "$(dirname -- "$output")" "$(basename -- "$output")"
echo "Return this results archive: ${output}.tar.gz"

# ADMET benchmark driver

The current installation and run instructions are in the
[ADMET compute handoff](../../docs/compute_handoff.md) on main. That guide covers
the small compression check, full-partition compression, and the ordinary
graphlet predictor, with provenance capture and results archives.

For the predictor and its separate sampled compression diagnostic, run:

~~~bash
bash scripts/run_admet_handoff.sh baseline HIA_Hou 4
~~~

The driver uses the official TDC train_val/test boundary. It fits the graphlet
vocabulary on train_val only and asks TDC to evaluate held-out predictions.
Its compression section fits on 128 training molecules and transforms 64
held-out molecules. Those limits apply to the compression diagnostic; the
predictor uses the complete supplied split.

For custom diagnostic limits and output paths, the underlying entry point is:

~~~bash
python examples/admet/admet_smoke.py \
  --dataset HIA_Hou --jobs 4 --seed 1 \
  --n-rules 5 --max-candidates 25 \
  --compression-train-limit 128 --compression-test-limit 64 \
  --output-dir results/admet_driver/hia_hou
~~~

The first run downloads the TDC data. The driver writes summary.json, held-out
predictions, candidates/dictionary CSVs, train/held-out compression tables, and
an occurrence cache. Successful completion includes an official TDC score and
zero held-out decode failures. Zero selected rules are a valid MDL outcome.

A compressed-input predictive comparison still needs an explicit motif/port
feature adapter. This single deterministic run is not a leaderboard submission.
See the [checkpoint validation](../../docs/checkpoint_validation.md) for the
tested configuration and its results, and the [short dataset examples](../README.md)
for compression-only entry points.

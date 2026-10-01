# ADMET smoke test

This example proves that one official TDC ADMET benchmark can pass through a
small, reproducible Buhito workflow on a laptop. It uses the official scaffold
test split for `HIA_Hou`, fits the graphlet vocabulary on `train_val` only, and
asks TDC to evaluate the held-out predictions with the benchmark metric.

It also runs a separate lossless-compression diagnostic. The default diagnostic
uses 128 training molecules and 64 held-out molecules so that the first run is
quick. It verifies exact decoding and records dictionary and MDL statistics.

This is deliberately not the final compressed-feature comparison: the choice of
how motif identity and boundary-port data enter a predictor remains part of the
shared experimental contract.

## Install

PyTDC 1.1.15 still uses the legacy `pkg_resources` interface, so keep
`setuptools` below the removal release in this environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel "setuptools==80.9.0"
python -m pip install -e ".[admet]"
python -m pip install --no-deps --no-build-isolation "PyTDC==1.1.15"
```

PyTDC's published dependency list currently pulls in a large multimodal model
stack, including Torch and CUDA packages, that this loader does not use. The
`--no-deps` line avoids that multi-gigabyte installation; the `admet` extra
above installs the dependencies needed by this example explicitly.

## Run the laptop smoke test

```bash
python examples/admet/admet_smoke.py \
  --dataset HIA_Hou \
  --jobs 4 \
  --compression-train-limit 128 \
  --compression-test-limit 64
```

The first run downloads the TDC data. The script writes:

- `results/admet_smoke/hia_hou/summary.json`;
- held-out predictions;
- candidate and selected-dictionary tables;
- train and held-out compression tables;
- a reusable occurrence cache.

`summary.json` must report an official TDC score and zero held-out decode
failures. A selected rule count of zero is a valid MDL result, not a failed run.

After the smoke test passes, run the compression diagnostic over the complete
official split by setting both limits to zero:

```bash
python examples/admet/admet_smoke.py \
  --dataset HIA_Hou \
  --jobs 4 \
  --compression-train-limit 0 \
  --compression-test-limit 0
```

The script performs one deterministic run. TDC asks for at least five runs for
leaderboard-quality reporting, so neither command is a leaderboard submission.

# ADMET compute handoff

Run from a checkout of the reviewed commit on Python 3.12/Linux. The source
branch is feat/lanl-checkpoint-2026-10-19. Pin the commit displayed by the PR,
then record it in each run; the wrapper does that automatically.

## 1. Get the code and install

~~~bash
git clone https://github.com/Necromath/buhito.git
cd buhito
git fetch origin feat/lanl-checkpoint-2026-10-19
git switch --detach origin/feat/lanl-checkpoint-2026-10-19
git rev-parse HEAD
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel "setuptools==80.9.0"
python -m pip install -c requirements/admet-constraints.txt -e ".[admet]"
python -m pip install --no-deps --no-build-isolation "PyTDC==1.1.15"
~~~

Use the PR's exact commit SHA in place of the moving branch ref for a repeat
experiment. The constraints pin the core scientific packages; every run also
records the complete installed dependency list.

PyTDC is installed separately because its declared dependencies include an
unrelated multimodal stack. The admet extra provides the dependencies used by
this loader. The setuptools pin retains PyTDC's pkg_resources compatibility;
the NumPy pin matches the RDKit wheel used here.

## 2. Run the small compression check

~~~bash
bash scripts/run_admet_handoff.sh smoke
~~~

This first runs the labeled triangle quick start, then fits on 128 molecules
sampled from the official HIA_Hou train_val partition and transforms 64 sampled
test molecules. It learns no predictor. Check the printed source rows, scored
candidates, selected rules, applied contractions, costs, and decode status.

## 3. Run compression on the full split

~~~bash
bash scripts/run_admet_handoff.sh full
~~~

This uses every molecule in the supplied train_val/test partitions with the same
dictionary settings and seed. It tests whether the small sample's dictionary
behavior persists at the full HIA_Hou scale.

## 4. Run the ordinary graphlet predictor

~~~bash
bash scripts/run_admet_handoff.sh baseline HIA_Hou 4
~~~

This runs the existing official-split graphlet predictor and asks TDC to compute
the benchmark score. Its compression section is a separate 128/64 diagnostic.
The jobs argument controls the baseline featurizer. Compression does not receive
that argument and does not currently parallelize through it.

These commands do not compare predictor quality using original and compressed
inputs. That experiment still needs a motif/port feature adapter and matched
model evaluation. One seeded run is not a leaderboard-quality result.

## Other datasets and repeated runs

~~~bash
bash scripts/run_admet_handoff.sh smoke HIA_Hou 4 results/hia_repeat_2
~~~

The positional arguments are mode, TDC dataset name, baseline jobs, and output
directory. Other ADMET benchmarks use the same loader, but their runtime and
dictionary behavior must be measured independently. Existing successful output
directories are preserved; use a new directory for a repeat.

## What the run produces

Return the .tar.gz path printed at completion. It contains:

- summary.json and the mode's candidate/dictionary/per-graph tables;
- the terminal log and quick-start log;
- source revision and working-tree status;
- Python, platform, CPU/RAM, and complete dependency information;
- held-out predictions for baseline mode.

Source data remain in the local data/tdc cache and are not included in the results
archive. Compression-only summaries retain the sampled source row positions.
The existing baseline driver records its seed and sample sizes.

A successful run completes without an exception and reports zero selected-label
decode failures. Zero selected rules are valid. Inspect candidate and prefix
tables before interpreting that result as absent recurring structure. The
default empty-dictionary accounting still includes the one-bit model-choice
charge, so its reported net savings can be -1 bit.

## Hardware and scope

The current commands use CPU-based graphlet enumeration and scikit-learn.
They do not use CUDA. Choose CPU cores and RAM according to the dataset and
the log from the small check; a GPU is needed only for a separate GPU workload.
No measured memory minimum or speedup is claimed by these examples.

The standalone short examples for the remaining datasets are documented in
[examples/README.md](../examples/README.md). A missing local QMugs/ZINC/TU input
is a data-availability blocker, not an automatic download or synthetic fallback.

# QM9 benchmark

The QM9 dataset is intentionally not stored in this repository. Prepare it
locally and run the existing benchmark with an explicit data path:

```bash
python scripts/prepare_qm9.py \
  --output data/qm9/qm9_processed.csv

python examples/qm9/benchmark_graphlet_featurizers_train_test.py \
  --data data/qm9/qm9_processed.csv \
  --max-len 2 3 4 \
  --repeats 3 \
  --n-jobs -1 \
  --outdir artifacts/qm9_benchmark
```

For a quick smoke test, prepare only 1,000 molecules:

```bash
python scripts/prepare_qm9.py \
  --output data/qm9/qm9_1000.csv \
  --limit 1000
```

SMILES conversion requires the chemistry extra:

```bash
python -m pip install -e ".[chem]"
```

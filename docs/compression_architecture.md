# Compression architecture

This refactor gives the lossless MDL pipeline one dataset-independent entry
point:

```python
from buhito.compression import MDLGraphCompressor
```

The caller owns loading and cleaning. The core accepts a collection of already
loaded, undirected simple NetworkX graphs and preserves the node and edge
attributes named by `GraphSchema` or the compressor's label-key arguments.

## Stage boundaries

| Stage | Module | Responsibility |
|---|---|---|
| Recognize | `recognition.py` | Enumerate candidate connected induced graphlets |
| Count/select | `dictionary.py` | Aggregate support, score candidates, freeze a dictionary |
| Substitute | `substitution.py` | Choose non-overlapping occurrences, contract, and decode |
| Orchestrate | `pipeline.py` | Normalize input, fit on training data, transform held-out data, report |
| Share types | `models.py` | Schemas, rules, occurrences, rewrites, and results |

Dataset-specific SMILES parsing, TU-file parsing, key normalization, and target
handling stay in `examples/` or `buhito.datasets`; they are not branches in
the core compression functions.

## Estimator contract

```python
compressor.fit(train_graphs)
train_result = compressor.transform(train_graphs)
valid_result = compressor.transform(valid_graphs)
test_result = compressor.transform(test_graphs)
```

Dictionary discovery and selection happen only in `fit`. `transform` uses
the frozen dictionary, which prevents held-out graph structure from influencing
the learned representation. Targets are never passed to the compressor.

A convenience `fit_transform` method remains available, but experiment code
must not call it on a train/validation/test concatenation.

## Lossless scope

Round-trip guarantees cover undirected simple topology and the configured node
and edge attributes. The compressed representation carries motif identity and
boundary-port metadata. A carrier graph with that metadata removed is a lossy
model view, even when the encoded archive remains exactly decodable.

## Migration plan

The first commit establishes and tests the public boundary without changing
behavior. Follow-up commits will move implementation from the monolithic
`buhito.mdl` module in this order:

1. shared dataclasses and graph-schema normalization;
2. candidate recognition and occurrence counting;
3. dictionary accounting and selection;
4. substitution, boundary ports, and reconstruction;
5. the thin orchestration class.

During migration, `buhito.mdl` remains a compatibility import path. Each move
must preserve the current test suite and exact-reconstruction checks.

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

## Implemented layout

The production implementation has been moved out of the former monolithic
`buhito.mdl` module:

1. `models.py` owns shared dataclasses, graph-schema normalization, and coding
   primitives;
2. `recognition.py` owns enumeration, canonical motif recognition, and exact
   occurrence alignment;
3. `substitution.py` owns occurrence packing, contraction, boundary ports,
   validation, and decoding;
4. `dictionary.py` owns rewrite codelengths and corpus selection;
5. `pipeline.py` owns fitted dictionary construction, frozen transformation,
   caching, and reports.

`buhito.mdl` is now a compatibility import path for existing experiments. The
five implementation files total roughly 2,400 lines. This is above the initial
1,000--1,500-line estimate because it includes the explicit analytical MDL
accounting, exact boundary-port codec, validation diagnostics, caching, and
report construction that were already part of the tested implementation.

## Definition of done

The refactor is complete when all of the following are true:

- the production implementation lives in `buhito.compression`, not in
  `buhito.mdl`;
- `buhito.mdl` is a compatibility module containing imports and deprecation
  guidance rather than a second implementation;
- recognition, dictionary construction, substitution, and orchestration have
  one-way dependencies with no circular imports;
- the core accepts already-loaded NetworkX graphs and contains no
  dataset-specific loading branches;
- `fit` is the only operation that learns a dictionary and `transform` never
  changes the fitted rules;
- configured topology and labels round-trip exactly on every rewritten graph;
- the README example runs as written in the base test environment;
- the full unit-test suite passes locally;
- the combined production compression modules remain small enough to review,
  with any departure from the 1,000--1,500-line estimate explained in the PR.

Large ADMET, TU, REDDIT, ZINC, and QMugs runs are explicitly outside this
refactor. They consume the stable API after this deliverable is merged.

# Short end-to-end compression examples

Each 65–80-line entry point loads graphs, fits on training graphs only, applies
the frozen dictionary to held-out graphs, decodes, and reports the outcome.
The shared _compression_common.py helper handles sampling, selected-label
validation, and reports. Each script calls the public compression API directly.

## Install and run

Topology and syntax examples use the base package:

~~~bash
python -m pip install -e .
python examples/compression_quickstart.py
python examples/compression_syntax.py --force-prefix 5
~~~

Molecular CSVs additionally need: python -m pip install -e ".[chem]".
For ADMET, use the [compute handoff](../docs/compute_handoff.md) dependency recipe.

## TU datasets

Place the existing TU files below data/TUDataset. Use the same entry point for
REDDIT-MULTI-5K, MUTAG, PTC_MR, NCI1, and ENZYMES:

~~~bash
python examples/compression_tu.py \
  --data-root data/TUDataset --dataset MUTAG \
  --fit-size 64 --test-size 32 --force-prefix 1 \
  --output-dir artifacts/examples/mutag
~~~

Change --dataset and --output-dir for each dataset. Available discrete node and
edge labels are coded; absent labels give topology-only compression. The seeded,
disjoint sample is an illustrative compression split, not an accuracy benchmark
split. Large, high-degree REDDIT graphs can be expensive even in small samples.

## Molecular CSVs

~~~bash
python scripts/prepare_qm9.py \
  --limit 1000 --output data/qm9/qm9_1000.csv
python examples/compression_molecules.py \
  --dataset QM9 --data data/qm9/qm9_1000.csv \
  --output-dir artifacts/examples/qm9
python examples/compression_molecules.py \
  --dataset QMugs --data data/qmugs/molecules.csv --smiles-column smiles \
  --output-dir artifacts/examples/qmugs
~~~

Targets and other CSV columns stay outside the coding schema. Repeated exact
SMILES are deduplicated before a seeded sample of 64 fitting/32 held-out graphs.
Use --fit-size/--test-size for smaller inputs. Original CSV row positions are
reported. This adapter does not canonicalize equivalent SMILES or reproduce the
historical QMugs conformer split and predictive experiment.

## ADMET and ZINC

~~~bash
python examples/compression_admet.py \
  --dataset HIA_Hou --train-limit 128 --test-limit 64 \
  --output-dir artifacts/examples/hia_hou
python examples/compression_zinc.py \
  --data-root data/zinc12k --train-limit 64 --test-limit 32 \
  --output-dir artifacts/examples/zinc
~~~

ADMET retains the official train_val/test boundary, sampling within each
partition. ZINC retains the supplied train.jsonl/test.jsonl partitions and reuses
the existing audited node_feat/edge_index/edge_attr/y/num_nodes adapter.
Zero limits mean the complete partitions. Targets remain outside the schema.
These examples evaluate compression; they do not train compressed predictors.

## Syntax and historical limitations

The syntax example uses handcrafted dependency graphs, identified as a fixture.
Repeated fixtures can compress under current settings; that is independent of
the historical negative predictive comparison. The original syntax corpus and
QMugs split are not bundled. OMOL-25 appears in the project proposal but has no
verified loader here. These limitations are not hidden behind synthetic data.

## Outputs and negative results

| File | Contents |
| --- | --- |
| summary.json | Dataset/split, source rows, configuration, source revision, decode status, applied size changes, and corpus costs. |
| candidates.csv | Scored candidates and their costs/support. |
| dictionary.csv | Rules used by this result. |
| dictionary_path.csv | Evaluated training prefixes, including the empty dictionary. |
| held_out.csv | Per-graph occurrences, selections, proposed template sizes, and costs. |
| selector_curve.csv | Evaluated corpus-selector choices. |

Zero rules and unfavorable savings are valid outcomes when decoding passes.
Empty candidate/dictionary CSVs still contain headers. --force-prefix N writes
a separate forced/ diagnostic, preserving the fitted dictionary and the main
MDL-selected result. N must not exceed the number of scored candidates.

Summary node/edge counts describe the selected model views. In held_out.csv,
inspect use_rewrite before treating a proposed template reduction as an applied
reduction. code.n_rewritten/code.n_occurrences count applied rewrites/contractions.
Bad inputs or decoding mismatches raise errors instead of successful summaries.

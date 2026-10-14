# Checkpoint validation, 14 October 2026

The implementation checked here is preserved at
[commit 4de6ba8](https://github.com/Necromath/buhito/commit/4de6ba881aca2daf8a69871d2b1be4c6ad513793),
merged into `main` through [PR #5](https://github.com/Necromath/buhito/pull/5).
For current instructions, use the [compute handoff](compute_handoff.md),
[dataset examples](../examples/README.md), and [paper companion](paper/README.md).
The results below are a record of that tested implementation.

## Checks

- Full unit suite: 105 passed, four optional PyTorch checks skipped.
- Eight focused example checks passed, including deliberate selected-label
  corruption, disjoint sampling, input adapters, and an official-split mock.
- New example/helper/test files pass Ruff. Repository-wide Ruff reports 278
  pre-existing findings on both the starting revision and this working tree;
  comparison found no new lint findings.
- Documented Python 3.12 ADMET installation succeeded in an isolated venv.
- Handoff smoke, full-compression, and baseline modes completed successfully,
  produced result archives, and reported zero held-out decode failures.
- Chemistry and syntax notebook code cells executed with their declared fixtures.
- Methods/algorithms companion compiled; the PDF was visually checked.

## Real-data and fixture checks

| Input | Fit graphs | Held out | Scored candidates | Selected rules | Decode failures |
| --- | --- | --- | --- | --- | --- |
| HIA_Hou sample | 128 | 64 | 25 | 0 | 0 |
| HIA_Hou full | 461 | 117 | 25 | 0 | 0 |
| MUTAG | 64 | 32 | 24 | 0 | 0 |
| QM9 sample | 64 | 32 | 18 | 0 | 0 |
| syntax fixture | 12 | 4 | 7 | 1 | 0 |

HIA_Hou used the supplied TDC train_val/test partitions (461/117 molecules).
The full run used size-three graphlets, a cap of 25 scored candidates, at most
five selected rules, and the default MDL selection. Zero rules persisted on the
full split under these settings. This does not show that every compression
configuration fails. The -1-bit reported savings are the model-choice charge
for an unchanged graph corpus.

The ordinary graphlet predictor used LogisticRegression with 178 features.
Its single seed-1 official HIA_Hou test ROC-AUC was 0.959. This is a smoke
baseline, without tuning or repeated-seed leaderboard reporting, and does not
measure prediction using compressed features.

The MUTAG forced one-rule diagnostic rewrote one of 32 held-out graphs and
reported -44.97 bits of net savings, with exact selected-label decoding. Its
MDL-selected result retained zero rules. The syntax fixture selected one rule
and reported +7.77 bits; it does not reproduce the historical negative corpus
experiment.

## Remaining external inputs and scope

The ZINC adapter was checked with format-correct input fixtures; an original
ZINC export was not present. The QMugs-style CSV adapter was checked with
molecular fixtures; the historical conformer dataset/split was not present.
REDDIT, PTC_MR, NCI1, and ENZYMES use the common TU entry point but were not
re-run against their original datasets here. No OMOL adapter is claimed.

Opening Overleaf timed out; its live project was not edited. The compiled
methods.tex companion and inclusion instructions are provided instead.

Initial run receipts name starting commit
5c9814cb1b28aab7a138986efed8ea5e19ef4ed4 and correctly mark the working tree as
modified: those checks preceded publication. A further smoke check passed at
committed revision 4de6ba881aca2daf8a69871d2b1be4c6ad513793 with a clean working
tree. New experiments use `main` and retain their own exact revision receipts.
Generated data, logs, caches, and results are excluded from Git.

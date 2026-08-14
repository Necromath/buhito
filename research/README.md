# Buhito research layer

This directory contains experiment-specific and retrospective research code.

Reusable graph-compression functionality belongs under `src/buhito/`.
Research code should only move into the package when it represents a
dataset-independent reusable abstraction with corresponding tests.

## Local development layout

- clean development repository:
  `~/Projects/buhito-dev`

- frozen Darwin source/reference tree:
  `~/Projects/buhito`

- frozen historical result artifacts:
  `~/Projects/buhito_frozen_results`

- local datasets:
  `~/Projects/buhito_data`

- immutable Darwin transfer snapshot:
  `~/Projects/buhito_freeze`

The restored Darwin checkout is historical reference material and should not
be cleaned or refactored in place.

## Research areas

- `bit_accounting/`
  Reconciliation of the tracked analytical MDL accountant and the later
  explicit ZINC enumerative codec.

- `tu_2026/`
  Retrospective TU experiments and reproducibility code.

- `zinc12k_2026/`
  ZINC-12k compression, prediction, runtime, and representation studies.

- `archive/`
  Historical research scaffolding retained only when useful for provenance.

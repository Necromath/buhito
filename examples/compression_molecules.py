"""Compress QM9/QMugs-style molecular CSVs with an illustrative graph-only split."""

import argparse
import time
from pathlib import Path

import pandas as pd
from _compression_common import add_run_arguments, finish_run, split_indices

from buhito.compression import MDLGraphCompressor
from buhito.converters import smiles_to_nx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", default="QM9")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--fit-size", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=32)
    add_run_arguments(parser, "artifacts/examples/molecules")
    args = parser.parse_args()
    started = time.perf_counter()

    frame = pd.read_csv(args.data)
    if args.smiles_column not in frame:
        raise ValueError(
            f"Missing SMILES column {args.smiles_column!r} in {args.data}."
        )
    if frame[args.smiles_column].isna().any():
        raise ValueError("SMILES column contains missing values.")
    # Repeated QMugs conformer rows with the same SMILES share a single graph.
    frame = frame.drop_duplicates(subset=[args.smiles_column])
    source_rows = frame.index.tolist()
    frame = frame.reset_index(drop=True)  # Targets remain in the CSV frame.
    fit_ids, test_ids = split_indices(
        len(frame),
        args.fit_size,
        args.test_size,
        args.seed,
    )

    def graphs(indices):
        return [smiles_to_nx(str(frame.at[i, args.smiles_column]))[0] for i in indices]

    fitting, held_out = graphs(fit_ids), graphs(test_ids)

    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=args.n_rules,
        max_candidates=args.max_candidates,
        node_label_keys="atom_key",
        edge_label_keys="bond_key",
        validate=True,
    )
    compressor.fit(fitting)
    result = compressor.transform(held_out)
    finish_run(
        compressor,
        result,
        held_out,
        args,
        {
            "dataset": args.dataset,
            "input_kind": "local molecular CSV",
            "split": "illustrative seeded sample after exact-SMILES deduplication",
            "fit_source_rows": [source_rows[i] for i in fit_ids],
            "held_out_source_rows": [source_rows[i] for i in test_ids],
            "fit_graphs": len(fitting),
            "held_out_graphs": len(held_out),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()

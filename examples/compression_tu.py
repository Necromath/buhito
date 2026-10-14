"""Fit, transform, and decode a bounded sample of any local TU dataset."""

import argparse
import time
from pathlib import Path

from _compression_common import add_run_arguments, finish_run, split_indices

from buhito.compression import MDLGraphCompressor
from buhito.datasets import load_tu_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", default="REDDIT-MULTI-5K")
    parser.add_argument("--fit-size", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=32)
    add_run_arguments(parser, "artifacts/examples/tu")
    args = parser.parse_args()
    started = time.perf_counter()

    # Discrete TU labels are selected when present; otherwise use topology only.
    dataset = load_tu_dataset(args.data_root, args.dataset)
    fit_ids, test_ids = split_indices(
        len(dataset.graphs),
        args.fit_size,
        args.test_size,
        args.seed,
    )
    fitting = [dataset.graphs[i] for i in fit_ids]
    held_out = [dataset.graphs[i] for i in test_ids]
    targets = dataset.graph_labels  # Retained outside the compression schema.

    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=args.n_rules,
        max_candidates=args.max_candidates,
        node_label_keys=dataset.node_label_key,
        edge_label_keys=dataset.edge_label_key,
        validate=True,
    )
    compressor.fit(fitting)
    result = compressor.transform(held_out)
    # Validation includes the selected labels, not only unlabeled topology.
    finish_run(
        compressor,
        result,
        held_out,
        args,
        {
            "dataset": args.dataset,
            "input_kind": "local TU dataset",
            "split": "illustrative seeded sample; not a benchmark accuracy split",
            "fit_indices": fit_ids,
            "held_out_indices": test_ids,
            "fit_graphs": len(fitting),
            "held_out_graphs": len(held_out),
            "target_count": len(targets),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()

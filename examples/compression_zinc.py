"""Compress prepared ZINC JSONL train/test graphs using the audited adapter."""

import argparse
import importlib.util
import time
from itertools import islice
from pathlib import Path

from _compression_common import add_run_arguments, finish_run

from buhito.compression import MDLGraphCompressor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--test-limit", type=int, default=32)
    add_run_arguments(parser, "artifacts/examples/zinc")
    args = parser.parse_args()
    if args.train_limit < 0 or args.test_limit < 0:
        parser.error("Limits must be nonnegative; zero means the full split.")
    started = time.perf_counter()

    # Reuse the existing adapter, including bidirectional-edge consistency checks.
    adapter_path = (
        Path(__file__).resolve().parents[1] / "research/zinc12k_2026/zinc12k_jsonl.py"
    )
    spec = importlib.util.spec_from_file_location("zinc_adapter", adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    def graphs(split, limit):
        records = adapter.iter_records(args.data_root, split)
        selected = islice(records, limit) if limit else records
        return [
            adapter.record_to_networkx(record, split=split, dataset_index=index)[0]
            for index, record in selected
        ]

    fitting = graphs("train", args.train_limit)
    held_out = graphs("test", args.test_limit)
    if not fitting or not held_out:
        raise ValueError("Both train.jsonl and test.jsonl must contain graphs.")
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=args.n_rules,
        max_candidates=args.max_candidates,
        node_label_keys="tu_node_label",
        edge_label_keys="tu_edge_label",
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
            "dataset": "ZINC-12k",
            "input_kind": "prepared JSONL files",
            "split": "supplied train/test partitions; leading rows if limited",
            "fit_indices": [g.graph["dataset_index"] for g in fitting],
            "held_out_indices": [g.graph["dataset_index"] for g in held_out],
            "fit_graphs": len(fitting),
            "held_out_graphs": len(held_out),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()

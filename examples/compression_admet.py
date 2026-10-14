"""Compression-only example retaining the official TDC train_val/test boundary."""

import argparse
import time
from pathlib import Path

from _compression_common import add_run_arguments, finish_run
from admet.admet_smoke import _graphs, _limited, _load_benchmark

from buhito.compression import MDLGraphCompressor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="HIA_Hou")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tdc"))
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--test-limit", type=int, default=64)
    add_run_arguments(parser, "artifacts/examples/admet")
    args = parser.parse_args()
    if args.train_limit < 0 or args.test_limit < 0:
        parser.error("Limits must be nonnegative; zero means the full split.")
    started = time.perf_counter()

    _, name, train, test = _load_benchmark(args.dataset, args.data_dir)
    # Preserve original split positions before deterministic sampling.
    train = train.assign(source_row=train.index)
    test = test.assign(source_row=test.index)
    train_sample = _limited(train, args.train_limit, seed=args.seed)
    test_sample = _limited(test, args.test_limit, seed=args.seed)
    fitting, held_out = _graphs(train_sample), _graphs(test_sample)
    # Prediction targets stay in the source frames and are never passed to fit.
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=args.n_rules,
        max_candidates=args.max_candidates,
        node_label_keys="atom_key",
        edge_label_keys="bond_key",
        validate=True,
        progress=True,
    )
    compressor.fit(fitting)
    result = compressor.transform(held_out)
    finish_run(
        compressor,
        result,
        held_out,
        args,
        {
            "dataset": name,
            "input_kind": "official TDC benchmark",
            "split": "fit on train_val; apply frozen dictionary to test",
            "fit_source_rows": train_sample.source_row.tolist(),
            "held_out_source_rows": test_sample.source_row.tolist(),
            "fit_graphs": len(fitting),
            "held_out_graphs": len(held_out),
            "official_train_val_rows": len(train),
            "official_test_rows": len(test),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()

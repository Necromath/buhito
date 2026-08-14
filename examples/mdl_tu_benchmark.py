"""Run Buhito's MDL graphlet compressor on any local TU dataset.

Examples
--------
Honest REDDIT-5K screen (empty dictionary is allowed)::

    python examples/mdl_tu_benchmark.py \
        --data-root data \
        --dataset REDDIT-MULTI-5K \
        --node-label-mode none \
        --edge-label-mode none \
        --graphlet-sizes 3 \
        --fit-size 500 \
        --eval-size 500

Force the best ranked rule into the dictionary even when its MDL gain is
negative. This is useful for diagnosing *why* compression fails::

    python examples/mdl_tu_benchmark.py \
        --data-root data \
        --dataset REDDIT-MULTI-5K \
        --node-label-mode none \
        --edge-label-mode none \
        --graphlet-sizes 3 \
        --n-rules 1 \
        --dictionary-selection fixed \
        --min-rule-savings -inf

The same script works for labeled molecular/protein datasets such as MUTAG,
PROTEINS, NCI1, and ENZYMES when their TU text files are present locally.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from buhito.datasets import load_tu_dataset
from buhito.mdl import (
    BuhitoGraphletEnumerator,
    ExhaustiveGraphletEnumerator,
    MDLGraphCompressor,
)


def _parse_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "graphlet sizes must be comma-separated integers"
        ) from exc
    if not sizes or min(sizes) < 2:
        raise argparse.ArgumentTypeError("graphlet sizes must be >= 2")
    return sizes


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _split_indices(
    labels: np.ndarray,
    *,
    fit_size: int,
    eval_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_graphs = len(labels)
    all_indices = np.arange(n_graphs, dtype=int)
    if fit_size <= 0 or fit_size >= n_graphs:
        return all_indices, np.empty(0, dtype=int)

    unique, counts = np.unique(labels, return_counts=True)
    can_stratify = (
        len(unique) > 1
        and counts.min() >= 2
        and fit_size >= len(unique)
        and (n_graphs - fit_size) >= len(unique)
    )
    fit_indices, remaining = train_test_split(
        all_indices,
        train_size=fit_size,
        random_state=seed,
        shuffle=True,
        stratify=labels if can_stratify else None,
    )
    fit_indices = np.asarray(sorted(fit_indices), dtype=int)
    remaining = np.asarray(remaining, dtype=int)

    if eval_size == 0 or eval_size >= len(remaining):
        return fit_indices, np.asarray(sorted(remaining), dtype=int)
    if eval_size < 0:
        return fit_indices, np.empty(0, dtype=int)

    remaining_labels = labels[remaining]
    unique_eval, counts_eval = np.unique(remaining_labels, return_counts=True)
    can_stratify_eval = (
        len(unique_eval) > 1
        and counts_eval.min() >= 2
        and eval_size >= len(unique_eval)
        and (len(remaining) - eval_size) >= len(unique_eval)
    )
    eval_indices, _ = train_test_split(
        remaining,
        train_size=eval_size,
        random_state=seed + 1,
        shuffle=True,
        stratify=remaining_labels if can_stratify_eval else None,
    )
    return fit_indices, np.asarray(sorted(eval_indices), dtype=int)


def _attach_dataset_columns(
    frame: pd.DataFrame,
    dataset_indices: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    output = frame.copy()
    output.insert(1, "dataset_index", dataset_indices)
    output.insert(2, "graph_label", labels[dataset_indices])
    return output


def _write_result_tables(
    output_directory: Path,
    prefix: str,
    result,
    dataset_indices: np.ndarray,
    labels: np.ndarray,
) -> None:
    _attach_dataset_columns(result.per_graph, dataset_indices, labels).to_csv(
        output_directory / f"{prefix}_per_graph.csv", index=False
    )
    result.selector_curve.to_csv(
        output_directory / f"{prefix}_selector_curve.csv", index=False
    )
    (output_directory / f"{prefix}_report.json").write_text(
        json.dumps(_jsonable(asdict(result.report)), indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run model-agnostic lossless MDL graphlet compression on a TU dataset."
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default="REDDIT-MULTI-5K")
    parser.add_argument(
        "--node-label-mode",
        choices=["auto", "none", "file", "degree"],
        default="auto",
    )
    parser.add_argument(
        "--edge-label-mode",
        choices=["auto", "none", "file", "constant"],
        default="auto",
    )
    parser.add_argument("--graphlet-sizes", type=_parse_sizes, default=(3,))
    parser.add_argument("--n-rules", type=int, default=2)
    parser.add_argument("--min-graph-support", type=int, default=5)
    parser.add_argument("--min-occurrences", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=25)
    parser.add_argument(
        "--dictionary-selection",
        choices=["best", "best_nonempty", "fixed"],
        default="best",
    )
    parser.add_argument(
        "--min-rule-savings",
        type=float,
        default=None,
        help=(
            "Minimum single-rule gain. Defaults to 0 for 'best' and -inf for "
            "diagnostic non-empty/fixed selection."
        ),
    )
    parser.add_argument(
        "--selector",
        choices=["sparse", "per_graph", "all_eligible"],
        default="sparse",
    )
    parser.add_argument("--model-choice-bits", type=float, default=1.0)
    parser.add_argument("--fit-size", type=int, default=500)
    parser.add_argument(
        "--eval-size",
        type=int,
        default=500,
        help="0 means all remaining graphs; a negative value disables evaluation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enumerator", choices=["buhito", "exhaustive"], default="buhito"
    )
    parser.add_argument("--cache-dir", default="artifacts/mdl_cache")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--save-compressor", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = load_tu_dataset(
        args.data_root,
        args.dataset,
        node_label_mode=args.node_label_mode,
        edge_label_mode=args.edge_label_mode,
    )
    fit_indices, eval_indices = _split_indices(
        dataset.graph_labels,
        fit_size=args.fit_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )
    fit_graphs = [dataset.graphs[index] for index in fit_indices]
    eval_graphs = [dataset.graphs[index] for index in eval_indices]

    output_directory = Path(
        args.out_dir
        or Path("artifacts")
        / "mdl_tu"
        / args.dataset
        / f"nodes-{args.node_label_mode}_edges-{args.edge_label_mode}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    min_rule_savings = args.min_rule_savings
    if min_rule_savings is None:
        min_rule_savings = (
            0.0 if args.dictionary_selection == "best" else -math.inf
        )
    enumerator = (
        BuhitoGraphletEnumerator()
        if args.enumerator == "buhito"
        else ExhaustiveGraphletEnumerator()
    )

    print(
        f"Loaded {len(dataset.graphs)} {dataset.name} graphs; "
        f"fit={len(fit_indices)}, eval={len(eval_indices)}, "
        f"node_label={dataset.node_label_key!r}, "
        f"edge_label={dataset.edge_label_key!r}."
    )
    compressor = MDLGraphCompressor(
        graphlet_sizes=args.graphlet_sizes,
        n_rules=args.n_rules,
        min_graph_support=args.min_graph_support,
        min_occurrences=args.min_occurrences,
        max_candidates=args.max_candidates,
        node_label_keys=dataset.node_label_key,
        edge_label_keys=dataset.edge_label_key,
        selector=args.selector,
        model_choice_bits=args.model_choice_bits,
        min_rule_savings_bits=min_rule_savings,
        dictionary_selection=args.dictionary_selection,
        enumerator=enumerator,
        cache_dir=args.cache_dir,
        validate=not args.no_validate,
        progress=True,
    )
    compressor.fit(fit_graphs)
    fit_result = compressor.training_result_
    assert fit_result is not None

    print("\nSelected dictionary")
    dictionary = compressor.dictionary_frame()
    print(dictionary if not dictionary.empty else "<empty dictionary>")
    print("\nFit report")
    print(fit_result.report)

    dictionary.to_csv(output_directory / "dictionary.csv", index=False)
    if compressor.candidate_table_ is not None:
        compressor.candidate_table_.to_csv(
            output_directory / "candidates.csv", index=False
        )
    if compressor.dictionary_path_ is not None:
        compressor.dictionary_path_.to_csv(
            output_directory / "dictionary_path.csv", index=False
        )
    _write_result_tables(
        output_directory,
        "fit",
        fit_result,
        fit_indices,
        dataset.graph_labels,
    )

    eval_result = None
    if len(eval_graphs):
        eval_result = compressor.transform(eval_graphs)
        print("\nEvaluation report")
        print(eval_result.report)
        _write_result_tables(
            output_directory,
            "eval",
            eval_result,
            eval_indices,
            dataset.graph_labels,
        )

    config = {
        **vars(args),
        "graphlet_sizes": args.graphlet_sizes,
        "resolved_min_rule_savings": min_rule_savings,
        "source_directory": dataset.source_directory,
        "node_label_key": dataset.node_label_key,
        "edge_label_key": dataset.edge_label_key,
        "fit_indices": fit_indices.tolist(),
        "eval_indices": eval_indices.tolist(),
        "fit_report": asdict(fit_result.report),
        "eval_report": asdict(eval_result.report) if eval_result is not None else None,
    }
    (output_directory / "run_summary.json").write_text(
        json.dumps(_jsonable(config), indent=2), encoding="utf-8"
    )

    if args.save_compressor:
        with gzip.open(output_directory / "compressor.pkl.gz", "wb") as handle:
            pickle.dump(compressor, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved artifacts to {output_directory.resolve()}")


if __name__ == "__main__":
    main()

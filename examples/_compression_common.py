"""Reporting and validation shared by the short compression examples."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import pandas as pd

from buhito.compression import GraphSchema, labeled_isomorphic


def add_run_arguments(parser: argparse.ArgumentParser, output: str) -> None:
    parser.add_argument("--output-dir", type=Path, default=Path(output))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-rules", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=25)
    parser.add_argument(
        "--force-prefix",
        type=int,
        default=None,
        help="Also evaluate a ranked prefix as a separate forced diagnostic.",
    )


def split_indices(
    count: int,
    fit_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    if fit_size < 1 or test_size < 1 or fit_size + test_size > count:
        raise ValueError(
            f"Need {fit_size} fitting and {test_size} held-out graphs; "
            f"received {count}. Use positive sizes whose sum fits the input."
        )
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    return indices[:fit_size], indices[fit_size : fit_size + test_size]


def assert_exact(
    originals: list[nx.Graph],
    decoded: list[nx.Graph],
    schema: GraphSchema,
) -> None:
    """Normalize both sides so selected labels and missing attributes are checked."""
    for index, (original, restored) in enumerate(zip(originals, decoded, strict=True)):
        if not labeled_isomorphic(
            schema.normalize(original), schema.normalize(restored)
        ):
            raise RuntimeError(
                f"Selected topology/labels failed decoding at graph {index}."
            )


def _provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "revision": revision,
        "working_tree_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }


def _save_result(compressor, result, graphs, directory, metadata, args, mode):
    assert_exact(graphs, result.decoded_graphs(), compressor.schema)
    views = result.model_graphs()
    summary = {
        **metadata,
        "mode": mode,
        "configuration": vars(args),
        "provenance": _provenance(),
        "compressor_configuration": {
            key: getattr(compressor, key)
            for key in (
                "graphlet_sizes",
                "n_rules",
                "min_graph_support",
                "min_occurrences",
                "max_candidates",
                "selector",
                "dictionary_selection",
                "model_choice_bits",
                "min_rule_savings_bits",
            )
        },
        "selected_rule_count": len(result.rules),
        "scored_candidate_count": len(compressor.candidate_frame()),
        "held_out_decode_failures": 0,
        "code": asdict(result.report),
        "original_nodes": sum(g.number_of_nodes() for g in graphs),
        "model_nodes": sum(g.number_of_nodes() for g in views),
        "original_edges": sum(g.number_of_edges() for g in graphs),
        "model_edges": sum(g.number_of_edges() for g in views),
        "node_label_keys": compressor.schema.node_label_keys,
        "edge_label_keys": compressor.schema.edge_label_keys,
    }
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "candidates.csv": (
            compressor.candidate_frame(),
            ["key", "forced_net_savings_bits"],
        ),
        "dictionary.csv": (
            pd.DataFrame([{"rank": r.rank, "key": r.key} for r in result.rules]),
            ["rank", "key"],
        ),
        "dictionary_path.csv": (compressor.dictionary_path_frame(), ["n_rules"]),
        "held_out.csv": (result.per_graph, ["graph_index", "use_rewrite"]),
        "selector_curve.csv": (result.selector_curve, ["k", "net_savings"]),
    }
    for filename, (table, empty_columns) in tables.items():
        if not len(table.columns):
            table = pd.DataFrame(columns=empty_columns)
        table.to_csv(directory / filename, index=False)
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str, allow_nan=False) + "\n",
    )
    print(json.dumps(summary, indent=2, default=str, allow_nan=False))
    return summary


def finish_run(compressor, result, held_out, args, metadata):
    """Save honest selection first; keep any forced-prefix result separate."""
    frozen = tuple(r.key for r in compressor.rules_ or ())
    summary = _save_result(
        compressor,
        result,
        held_out,
        args.output_dir,
        metadata,
        args,
        "mdl_selected",
    )
    if args.force_prefix is not None:
        forced = compressor.transform_rule_prefix(
            held_out,
            args.force_prefix,
            require_nonempty_rewrite=True,
        )
        _save_result(
            compressor,
            forced,
            held_out,
            args.output_dir / "forced",
            metadata,
            args,
            "forced_prefix_diagnostic",
        )
    if tuple(r.key for r in compressor.rules_ or ()) != frozen:
        raise RuntimeError("Transform changed the fitted dictionary.")
    return summary

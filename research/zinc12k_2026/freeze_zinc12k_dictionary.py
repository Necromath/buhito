#!/usr/bin/env python3
"""Fit and freeze the authoritative train-only Buhito dictionary for ZINC-12k.

This script:

1. loads canonical ZINC-12k JSONL;
2. fits MDLGraphCompressor on TRAIN ONLY;
3. freezes the selected dictionary and the full ranked candidate path;
4. applies the frozen compressor to train/val/test;
5. verifies exact labeled-graph reconstruction for every available rewrite;
6. records both MDL-selected and forced-covered structural contractions;
7. stores transformation objects for downstream prediction experiments.

No validation/test graph participates in dictionary discovery.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import networkx as nx
import pandas as pd

from buhito.mdl import MDLGraphCompressor, decode_rewrite


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from zinc12k_jsonl import EXPECTED_SPLIT_COUNTS, load_graphs


SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def labeled_user_isomorphic(left: nx.Graph, right: nx.Graph) -> bool:
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        "tu_node_label",
        None,
    )
    edge_match = nx.algorithms.isomorphism.categorical_edge_match(
        "tu_edge_label",
        None,
    )
    return nx.is_isomorphic(
        left,
        right,
        node_match=node_match,
        edge_match=edge_match,
    )


def write_pickle_gz(path: Path, value: Any) -> None:
    with gzip.open(path, "wb", compresslevel=6) as handle:
        pickle.dump(
            value,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def audit_result(
    *,
    split: str,
    originals: list[nx.Graph],
    result,
    compressor: MDLGraphCompressor,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(originals) != len(result.records):
        raise RuntimeError(
            f"{split}: original/result length mismatch: "
            f"{len(originals)} vs {len(result.records)}"
        )

    frame = result.per_graph.copy()

    frame.insert(0, "split", split)
    frame.insert(
        1,
        "dataset_index",
        [int(graph.graph["dataset_index"]) for graph in originals],
    )
    frame.insert(
        2,
        "target",
        [float(graph.graph["target"]) for graph in originals],
    )

    has_rewrite = []
    forced_nodes = []
    forced_edges = []
    selected_nodes = []
    selected_edges = []
    reconstruction_ok = []

    rewrites_checked = 0

    for original, record in zip(
        originals,
        result.records,
        strict=True,
    ):
        baseline_nodes = record.baseline_graph.number_of_nodes()
        baseline_edges = record.baseline_graph.number_of_edges()

        if record.rewrite is None:
            has_rewrite.append(False)

            forced_nodes.append(baseline_nodes)
            forced_edges.append(baseline_edges)

            selected_nodes.append(baseline_nodes)
            selected_edges.append(baseline_edges)

            reconstruction_ok.append(True)
            continue

        has_rewrite.append(True)

        template = record.rewrite.template

        forced_nodes.append(template.number_of_nodes())
        forced_edges.append(template.number_of_edges())

        if record.use_rewrite:
            selected_nodes.append(template.number_of_nodes())
            selected_edges.append(template.number_of_edges())
        else:
            selected_nodes.append(baseline_nodes)
            selected_edges.append(baseline_edges)

        # Public decoder audit, independent of internal validation.
        decoded_internal = decode_rewrite(record.rewrite)
        decoded = compressor.schema.restore(decoded_internal)

        topology_labels_ok = labeled_user_isomorphic(
            original,
            decoded,
        )

        metadata_ok = (
            float(decoded.graph["target"])
            == float(original.graph["target"])
            and decoded.graph["split"]
            == original.graph["split"]
            and int(decoded.graph["dataset_index"])
            == int(original.graph["dataset_index"])
        )

        ok = bool(topology_labels_ok and metadata_ok)
        reconstruction_ok.append(ok)
        rewrites_checked += 1

        if not ok:
            raise RuntimeError(
                f"{split}[{original.graph['dataset_index']}]: "
                "round-trip reconstruction failed."
            )

    frame["has_rewrite"] = has_rewrite
    frame["reconstruction_ok"] = reconstruction_ok

    frame["forced_nodes"] = forced_nodes
    frame["forced_edges"] = forced_edges

    frame["mdl_selected_nodes"] = selected_nodes
    frame["mdl_selected_edges"] = selected_edges

    frame["forced_node_reduction_fraction"] = (
        frame["original_nodes"] - frame["forced_nodes"]
    ) / frame["original_nodes"].clip(lower=1)

    frame["forced_edge_reduction_fraction"] = (
        frame["original_edges"] - frame["forced_edges"]
    ) / frame["original_edges"].clip(lower=1)

    frame["mdl_selected_node_reduction_fraction"] = (
        frame["original_nodes"] - frame["mdl_selected_nodes"]
    ) / frame["original_nodes"].clip(lower=1)

    frame["mdl_selected_edge_reduction_fraction"] = (
        frame["original_edges"] - frame["mdl_selected_edges"]
    ) / frame["original_edges"].clip(lower=1)

    original_nodes = int(frame["original_nodes"].sum())
    original_edges = int(frame["original_edges"].sum())

    forced_node_total = int(frame["forced_nodes"].sum())
    forced_edge_total = int(frame["forced_edges"].sum())

    selected_node_total = int(frame["mdl_selected_nodes"].sum())
    selected_edge_total = int(frame["mdl_selected_edges"].sum())

    summary = {
        "split": split,
        "graphs": len(frame),
        "graphs_with_available_rewrite": int(frame["has_rewrite"].sum()),
        "fraction_with_available_rewrite": float(frame["has_rewrite"].mean()),
        "graphs_mdl_selected_for_rewrite": int(
            frame["use_rewrite"].sum()
        ),
        "fraction_mdl_selected_for_rewrite": float(
            frame["use_rewrite"].mean()
        ),
        "rewrites_round_trip_checked": rewrites_checked,
        "reconstruction_failures": int(
            (~frame["reconstruction_ok"]).sum()
        ),
        "motif_occurrences_available": int(
            frame.loc[
                frame["has_rewrite"],
                "selected_occurrences",
            ].sum()
        ),
        "original_nodes": original_nodes,
        "original_edges": original_edges,
        "forced_nodes": forced_node_total,
        "forced_edges": forced_edge_total,
        "mdl_selected_nodes": selected_node_total,
        "mdl_selected_edges": selected_edge_total,
        "forced_node_reduction_fraction": (
            1.0 - forced_node_total / max(original_nodes, 1)
        ),
        "forced_edge_reduction_fraction": (
            1.0 - forced_edge_total / max(original_edges, 1)
        ),
        "mdl_selected_node_reduction_fraction": (
            1.0 - selected_node_total / max(original_nodes, 1)
        ),
        "mdl_selected_edge_reduction_fraction": (
            1.0 - selected_edge_total / max(original_edges, 1)
        ),
        "analytical_mdl": asdict(result.report),
    }

    return frame, summary


def motif_json_rows(compressor: MDLGraphCompressor) -> list[dict[str, Any]]:
    motifs = compressor.candidate_motif_graphs(restored=True)

    rows = []

    for key, motif in motifs.items():
        rows.append(
            {
                "key": key,
                "nodes": [
                    {
                        "node": repr(node),
                        "attributes": dict(attrs),
                    }
                    for node, attrs in motif.nodes(data=True)
                ],
                "edges": [
                    {
                        "source": repr(source),
                        "target": repr(target),
                        "attributes": dict(attrs),
                    }
                    for source, target, attrs in motif.edges(data=True)
                ],
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    cache = args.cache_dir.expanduser().resolve()
    repo = args.repo_root.expanduser().resolve()

    if output.exists():
        if not args.force:
            raise SystemExit(
                f"Output already exists: {output}. "
                "Use --force to replace it."
            )
        shutil.rmtree(output)

    output.mkdir(parents=True)
    cache.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()

    print("==========================================")
    print("Buhito ZINC-12k train-only dictionary fit")
    print("==========================================")
    print("Data:", data_root)
    print("Output:", output)
    print("Cache:", cache)
    print("Commit:", git_output(repo, "rev-parse", "HEAD"))
    print("Python:", sys.executable)
    print()

    datasets: dict[str, list[nx.Graph]] = {}

    for split in SPLITS:
        print(f"Loading {split}...", flush=True)

        graphs = load_graphs(data_root, split)

        expected = EXPECTED_SPLIT_COUNTS[split]
        if len(graphs) != expected:
            raise RuntimeError(
                f"{split}: expected {expected}, got {len(graphs)}"
            )

        datasets[split] = graphs

        print(
            f"{split}: {len(graphs)} graphs loaded",
            flush=True,
        )

    # ------------------------------------------------------------
    # IMPORTANT:
    # Only the TRAIN split enters fit().
    # ------------------------------------------------------------

    config = {
        "graphlet_sizes": (3,),
        "n_rules": 5,
        "min_graph_support": 2,
        "min_occurrences": 2,
        "max_candidates": 100,
        "node_label_keys": "tu_node_label",
        "edge_label_keys": "tu_edge_label",
        "selector": "sparse",
        "model_choice_bits": 1.0,
        "min_rule_savings_bits": 0.0,
        "dictionary_selection": "best",
        "cache_dir": cache,
        "validate": True,
        "progress": True,
    }

    compressor = MDLGraphCompressor(**config)

    print()
    print("FIT: TRAIN ONLY")
    print("----------------", flush=True)

    fit_started = time.perf_counter()

    compressor.fit(datasets["train"])

    fit_seconds = time.perf_counter() - fit_started

    print()
    print(f"Fit completed in {fit_seconds:.3f} s")
    print(
        "Selected rules:",
        len(compressor.rules_ or ()),
    )

    dictionary = compressor.dictionary_frame()
    candidates = compressor.candidate_frame()
    dictionary_path = compressor.dictionary_path_frame()

    dictionary.to_csv(
        output / "dictionary.csv",
        index=False,
    )

    candidates.to_csv(
        output / "candidate_ranking.csv",
        index=False,
    )

    dictionary_path.to_csv(
        output / "dictionary_path.csv",
        index=False,
    )

    motif_rows = motif_json_rows(compressor)

    with (output / "candidate_motifs.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in motif_rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    default=repr,
                )
                + "\n"
            )

    print()
    print("Selected dictionary:")
    if dictionary.empty:
        print("  <EMPTY DICTIONARY>")
    else:
        print(dictionary.to_string(index=False))

    print()
    print("Dictionary path:")
    print(dictionary_path.to_string(index=False))

    # ------------------------------------------------------------
    # Apply exactly the same frozen compressor to all three splits.
    # No refitting is allowed here.
    # ------------------------------------------------------------

    results = {
        "train": compressor.training_result_,
        "val": compressor.transform(datasets["val"]),
        "test": compressor.transform(datasets["test"]),
    }

    if results["train"] is None:
        raise RuntimeError("Training result unexpectedly unavailable.")

    summaries = []
    per_graph_frames = []

    result_dir = output / "transforms"
    result_dir.mkdir()

    for split in SPLITS:
        print()
        print(f"AUDIT: {split}")
        print("-" * (7 + len(split)), flush=True)

        result = results[split]

        frame, summary = audit_result(
            split=split,
            originals=datasets[split],
            result=result,
            compressor=compressor,
        )

        frame.to_csv(
            output / f"{split}_per_graph.csv",
            index=False,
        )

        result.selector_curve.to_csv(
            output / f"{split}_selector_curve.csv",
            index=False,
        )

        write_pickle_gz(
            result_dir / f"{split}_compression_result.pkl.gz",
            result,
        )

        summaries.append(summary)
        per_graph_frames.append(frame)

        print(
            f"{split}: "
            f"available rewrites="
            f"{summary['graphs_with_available_rewrite']}/"
            f"{summary['graphs']}; "
            f"MDL selected="
            f"{summary['graphs_mdl_selected_for_rewrite']}; "
            f"round-trip failures="
            f"{summary['reconstruction_failures']}",
            flush=True,
        )

    combined = pd.concat(
        per_graph_frames,
        ignore_index=True,
    )

    combined.to_csv(
        output / "all_splits_per_graph.csv",
        index=False,
    )

    # Freeze only the learned state, not the very large training result.
    training_result = compressor.training_result_
    compressor.training_result_ = None

    write_pickle_gz(
        output / "frozen_compressor.pkl.gz",
        compressor,
    )

    compressor.training_result_ = training_result

    provenance_path = data_root / "provenance.json"

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "canonical PyG ZINC subset=True",
        "task": "penalized logP regression",
        "fit_population": "train only",
        "fit_graph_count": len(datasets["train"]),
        "validation_graph_count": len(datasets["val"]),
        "test_graph_count": len(datasets["test"]),
        "dictionary_config": {
            key: value
            for key, value in config.items()
            if key not in {
                "cache_dir",
                "validate",
                "progress",
            }
        },
        "cache_dir": str(cache),
        "validate_rewrites": True,
        "buhito_git_commit": git_output(
            repo,
            "rev-parse",
            "HEAD",
        ),
        "buhito_git_status_short": git_output(
            repo,
            "status",
            "--short",
        ),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "selected_rule_count": len(compressor.rules_ or ()),
        "selected_rule_keys": [
            rule.key
            for rule in (compressor.rules_ or ())
        ],
        "fit_seconds": fit_seconds,
        "total_seconds": time.perf_counter() - started,
        "dataset_provenance_sha256": (
            sha256_file(provenance_path)
            if provenance_path.is_file()
            else None
        ),
        "split_summaries": summaries,
        "important_semantics": {
            "has_rewrite": (
                "Frozen dictionary occurs in the molecule and a "
                "reversible contraction exists."
            ),
            "use_rewrite": (
                "Corpus-level analytical MDL selector actually "
                "chooses the rewrite."
            ),
            "forced_structural_contraction": (
                "Apply the frozen dictionary wherever a rewrite "
                "exists, independent of corpus selector."
            ),
            "analytical_mdl": (
                "Analytical MDL objective from buhito.mdl; "
                "not serialized bytes and not enumerative-v1."
            ),
        },
    }

    (output / "provenance.json").write_text(
        json.dumps(
            provenance,
            indent=2,
            sort_keys=True,
            default=repr,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_rows = []

    for summary in summaries:
        analytical = summary.pop("analytical_mdl")

        row = dict(summary)

        for key, value in analytical.items():
            row[f"analytical_mdl_{key}"] = value

        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        output / "split_summary.csv",
        index=False,
    )

    readme = """# ZINC-12k frozen train-only dictionary

This artifact was produced by fitting `MDLGraphCompressor` on the 10,000
training molecules only.

Validation and test molecules were never used for candidate discovery,
candidate ranking, dictionary-prefix selection, or label vocabulary fitting.

Two contraction notions are retained:

1. `has_rewrite`: a reversible contraction exists under the frozen dictionary.
   This is the forced frozen-dictionary representation used to study
   representation effects.

2. `use_rewrite`: the analytical corpus-level MDL selector chose to encode that
   graph through the rewrite-capable code.

Structural node/edge reduction, analytical MDL codelength, serialized storage
size, enumerative-v1 description length, runtime, and prediction quality are
different quantities and must not be conflated.

Every available rewrite is decoded and checked for labeled-graph isomorphism
against its original molecule.
"""

    (output / "README.md").write_text(
        readme,
        encoding="utf-8",
    )

    # Hash final human-readable artifacts.
    artifact_hashes = {}

    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.json":
            artifact_hashes[
                str(path.relative_to(output))
            ] = sha256_file(path)

    (output / "artifact_hashes.json").write_text(
        json.dumps(
            artifact_hashes,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("==========================================")
    print("FREEZE COMPLETE")
    print("==========================================")
    print(f"Selected rule count: {len(compressor.rules_ or ())}")
    print(f"Fit seconds: {fit_seconds:.3f}")
    print(
        f"Total seconds: "
        f"{time.perf_counter() - started:.3f}"
    )
    print()
    print(
        pd.read_csv(
            output / "split_summary.csv"
        ).to_string(index=False)
    )
    print()
    print("Output:", output)
    print("PASS: all available rewrites round-trip exactly.")


if __name__ == "__main__":
    main()

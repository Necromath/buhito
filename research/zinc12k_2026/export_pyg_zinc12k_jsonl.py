#!/usr/bin/env python3
"""Export canonical PyG ZINC-12k into Buhito's JSONL format."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch
import torch_geometric
from torch_geometric.datasets import ZINC


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from zinc12k_jsonl import EXPECTED_SPLIT_COUNTS, record_to_networkx


SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
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


def read_subset_indices(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    return [
        int(value.strip())
        for value in text.rstrip(",").split(",")
        if value.strip()
    ]


def to_json_record(data) -> dict:
    return {
        "node_feat": data.x.detach().cpu().tolist(),
        "edge_index": data.edge_index.detach().cpu().tolist(),
        "edge_attr": data.edge_attr.detach().cpu().tolist(),
        "y": data.y.detach().cpu().tolist(),
        "num_nodes": int(data.num_nodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pyg-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    pyg_root = args.pyg_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()

    pyg_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("Canonical PyG ZINC-12k -> Buhito JSONL")
    print("========================================")
    print("Python:", sys.executable)
    print("Python version:", platform.python_version())
    print("Torch:", torch.__version__)
    print("PyG:", torch_geometric.__version__)
    print("PyG root:", pyg_root)
    print("Output root:", output_root)
    print("Buhito commit:", git_output(repo_root, "rev-parse", "HEAD"))
    print()

    manifest_rows = []
    split_summaries = {}

    for split in SPLITS:
        print(f"[{split}] loading PyG ZINC subset...", flush=True)

        dataset = ZINC(
            root=str(pyg_root),
            subset=True,
            split=split,
        )

        expected = EXPECTED_SPLIT_COUNTS[split]

        if len(dataset) != expected:
            raise RuntimeError(
                f"{split}: expected {expected} graphs, "
                f"but PyG returned {len(dataset)}"
            )

        index_path = Path(dataset.raw_dir) / f"{split}.index"

        if not index_path.exists():
            raise FileNotFoundError(
                f"Could not find canonical subset index file: {index_path}"
            )

        source_indices = read_subset_indices(index_path)

        if len(source_indices) != len(dataset):
            raise RuntimeError(
                f"{split}: dataset contains {len(dataset)} graphs, "
                f"but {index_path} contains {len(source_indices)} indices"
            )

        jsonl_path = output_root / f"{split}.jsonl"

        total_nodes = 0
        total_edges = 0

        target_min = float("inf")
        target_max = float("-inf")
        target_sum = 0.0

        node_labels = set()
        edge_labels = set()

        with jsonl_path.open("w", encoding="utf-8") as handle:
            for dataset_index, data in enumerate(dataset):
                record = to_json_record(data)

                # Validate using the same conversion used by later experiments.
                graph, audit = record_to_networkx(
                    record,
                    split=split,
                    dataset_index=dataset_index,
                    require_bidirectional=True,
                )

                handle.write(
                    json.dumps(
                        record,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )

                target = float(graph.graph["target"])
                nodes = graph.number_of_nodes()
                edges = graph.number_of_edges()

                total_nodes += nodes
                total_edges += edges

                target_sum += target
                target_min = min(target_min, target)
                target_max = max(target_max, target)

                node_labels.update(
                    int(attrs["tu_node_label"])
                    for _, attrs in graph.nodes(data=True)
                )

                edge_labels.update(
                    int(attrs["tu_edge_label"])
                    for _, _, attrs in graph.edges(data=True)
                )

                manifest_rows.append(
                    {
                        "split": split,
                        "dataset_index": dataset_index,
                        "source_split_index": source_indices[dataset_index],
                        "target": target,
                        "num_nodes": nodes,
                        "num_edges": edges,
                        "directed_edges": audit["directed_edges"],
                        "missing_reverse_arcs": audit[
                            "missing_reverse_arcs"
                        ],
                        "non_double_edges": audit[
                            "non_double_edges"
                        ],
                    }
                )

                if (dataset_index + 1) % 1000 == 0:
                    print(
                        f"[{split}] "
                        f"{dataset_index + 1}/{len(dataset)} validated",
                        flush=True,
                    )

        split_summaries[split] = {
            "graphs": len(dataset),
            "nodes": total_nodes,
            "edges": total_edges,
            "mean_nodes": total_nodes / len(dataset),
            "mean_edges": total_edges / len(dataset),
            "target_min": target_min,
            "target_max": target_max,
            "target_mean": target_sum / len(dataset),
            "node_labels": sorted(node_labels),
            "edge_labels": sorted(edge_labels),
            "jsonl_sha256": sha256_file(jsonl_path),
            "subset_index_path": str(index_path),
            "subset_index_sha256": sha256_file(index_path),
        }

        print(
            f"[{split}] PASS: "
            f"{len(dataset)} graphs, "
            f"sha256={split_summaries[split]['jsonl_sha256']}",
            flush=True,
        )

    manifest_path = output_root / "split_manifest.csv"

    columns = [
        "split",
        "dataset_index",
        "source_split_index",
        "target",
        "num_nodes",
        "num_edges",
        "directed_edges",
        "missing_reverse_arcs",
        "non_double_edges",
    ]

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "PyTorch Geometric ZINC",
        "subset": True,
        "task": "penalized logP regression",
        "expected_split_counts": EXPECTED_SPLIT_COUNTS,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "buhito_git_commit": git_output(
            repo_root,
            "rev-parse",
            "HEAD",
        ),
        "buhito_git_status": git_output(
            repo_root,
            "status",
            "--short",
        ),
        "pyg_root": str(pyg_root),
        "output_root": str(output_root),
        "splits": split_summaries,
        "split_manifest_sha256": sha256_file(manifest_path),
        "notes": [
            (
                "Fresh reconstruction from "
                "torch_geometric.datasets.ZINC(subset=True)."
            ),
            (
                "Historical LANL JSONL files were unavailable on Easley; "
                "byte identity with those files is therefore not claimed."
            ),
            (
                "Every exported record was parsed and validated using "
                "research/zinc12k_2026/zinc12k_jsonl.py."
            ),
        ],
    }

    provenance_path = output_root / "provenance.json"

    provenance_path.write_text(
        json.dumps(
            provenance,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("========================================")
    print("EXPORT COMPLETE")
    print("========================================")

    for split in SPLITS:
        info = split_summaries[split]
        print(
            f"{split:5s}: "
            f"{info['graphs']:5d} graphs | "
            f"{info['nodes']:7d} nodes | "
            f"{info['edges']:7d} edges"
        )

    print()
    print("Manifest:", manifest_path)
    print("Provenance:", provenance_path)
    print("PASS: all 12,000 molecules validated.")


if __name__ == "__main__":
    main()

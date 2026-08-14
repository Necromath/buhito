#!/usr/bin/env python3
"""Load the transferred ZINC-12k JSONL files as labeled NetworkX graphs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import networkx as nx


EXPECTED_SPLIT_COUNTS = {
    "train": 10_000,
    "val": 1_000,
    "test": 1_000,
}


def scalar(value: Any) -> Any:
    """Unwrap singleton feature lists such as [3] or [[3]]."""
    while isinstance(value, list) and len(value) == 1:
        value = value[0]

    if isinstance(value, list):
        raise ValueError(
            f"Expected a scalar or singleton list, got {value!r}"
        )

    return value


def parse_edge_index(edge_index: Any) -> list[tuple[int, int]]:
    """Accept either PyG's 2xE layout or an Ex2 edge list."""
    if not isinstance(edge_index, list):
        raise TypeError(
            f"edge_index must be a list, got {type(edge_index)}"
        )

    if (
        len(edge_index) == 2
        and all(isinstance(row, list) for row in edge_index)
        and len(edge_index[0]) == len(edge_index[1])
    ):
        return [
            (int(left), int(right))
            for left, right in zip(
                edge_index[0],
                edge_index[1],
            )
        ]

    if all(
        isinstance(edge, list) and len(edge) == 2
        for edge in edge_index
    ):
        return [
            (int(edge[0]), int(edge[1]))
            for edge in edge_index
        ]

    raise ValueError(
        "Could not interpret edge_index as 2xE or Ex2."
    )


def record_to_networkx(
    record: dict[str, Any],
    *,
    split: str,
    dataset_index: int,
    require_bidirectional: bool = True,
) -> tuple[nx.Graph, dict[str, int]]:
    """Convert one JSON record into the schema used by Buhito.

    Node features:
        tu_node_label

    Edge features:
        tu_edge_label

    Graph metadata:
        target
        split
        dataset_index
    """
    required = {
        "node_feat",
        "edge_index",
        "edge_attr",
        "y",
        "num_nodes",
    }

    missing = required.difference(record)

    if missing:
        raise ValueError(
            f"{split}[{dataset_index}] missing fields: "
            f"{sorted(missing)}"
        )

    num_nodes = int(scalar(record["num_nodes"]))
    node_features = [
        int(scalar(value))
        for value in record["node_feat"]
    ]

    if len(node_features) != num_nodes:
        raise ValueError(
            f"{split}[{dataset_index}] has num_nodes={num_nodes}, "
            f"but {len(node_features)} node features."
        )

    directed_edges = parse_edge_index(record["edge_index"])
    edge_features = [
        int(scalar(value))
        for value in record["edge_attr"]
    ]

    if len(directed_edges) != len(edge_features):
        raise ValueError(
            f"{split}[{dataset_index}] has "
            f"{len(directed_edges)} directed edges but "
            f"{len(edge_features)} edge attributes."
        )

    graph = nx.Graph()

    for node, node_label in enumerate(node_features):
        graph.add_node(
            node,
            tu_node_label=node_label,
        )

    directed_labeled_edges = Counter()
    undirected_labels: dict[tuple[int, int], int] = {}
    undirected_multiplicities = Counter()

    for (left, right), edge_label in zip(
        directed_edges,
        edge_features,
    ):
        if not 0 <= left < num_nodes:
            raise ValueError(
                f"{split}[{dataset_index}] invalid node {left}."
            )

        if not 0 <= right < num_nodes:
            raise ValueError(
                f"{split}[{dataset_index}] invalid node {right}."
            )

        if left == right:
            raise ValueError(
                f"{split}[{dataset_index}] contains a self-loop "
                f"at node {left}."
            )

        directed_labeled_edges[(left, right, edge_label)] += 1

        key = tuple(sorted((left, right)))
        undirected_multiplicities[key] += 1

        previous = undirected_labels.get(key)

        if previous is not None and previous != edge_label:
            raise ValueError(
                f"{split}[{dataset_index}] inconsistent labels "
                f"for edge {key}: {previous} versus {edge_label}."
            )

        undirected_labels[key] = edge_label

    missing_reverse_arcs = 0

    for left, right, edge_label in directed_labeled_edges:
        if (right, left, edge_label) not in directed_labeled_edges:
            missing_reverse_arcs += 1

    non_double_edges = sum(
        multiplicity != 2
        for multiplicity in undirected_multiplicities.values()
    )

    if require_bidirectional:
        if missing_reverse_arcs:
            raise ValueError(
                f"{split}[{dataset_index}] has "
                f"{missing_reverse_arcs} directed arcs without "
                "matching reverse arcs."
            )

        if non_double_edges:
            raise ValueError(
                f"{split}[{dataset_index}] has "
                f"{non_double_edges} undirected edges that do not "
                "appear exactly twice."
            )

    for (left, right), edge_label in undirected_labels.items():
        graph.add_edge(
            left,
            right,
            tu_edge_label=edge_label,
        )

    target = float(scalar(record["y"]))

    graph.graph.update({
        "target": target,
        "split": split,
        "dataset_index": dataset_index,
    })

    audit = {
        "num_nodes": graph.number_of_nodes(),
        "directed_edges": len(directed_edges),
        "undirected_edges": graph.number_of_edges(),
        "missing_reverse_arcs": missing_reverse_arcs,
        "non_double_edges": non_double_edges,
    }

    return graph, audit


def iter_records(
    root: Path,
    split: str,
) -> Iterator[tuple[int, dict[str, Any]]]:
    path = root / f"{split}.jsonl"

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        for dataset_index, line in enumerate(handle):
            line = line.strip()

            if not line:
                raise ValueError(
                    f"Blank line in {path} at index {dataset_index}."
                )

            yield dataset_index, json.loads(line)


def load_graphs(
    root: Path,
    split: str,
    indices: Iterable[int] | None = None,
) -> list[nx.Graph]:
    selected = None if indices is None else set(indices)
    graphs = []

    for dataset_index, record in iter_records(root, split):
        if selected is not None and dataset_index not in selected:
            continue

        graph, _ = record_to_networkx(
            record,
            split=split,
            dataset_index=dataset_index,
        )
        graphs.append(graph)

    if selected is not None and len(graphs) != len(selected):
        raise RuntimeError(
            f"Requested {len(selected)} {split} graphs, "
            f"but loaded {len(graphs)}."
        )

    return graphs

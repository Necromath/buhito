"""Load graph-classification datasets in TU Dortmund text formats.

The loader supports both the canonical TU names (``DATASET_A.txt``,
``DATASET_graph_indicator.txt``) and the legacy REDDIT names already used by
Buhito (``DATASET.edges``, ``DATASET.graph_idx``).

Only NetworkX is exposed to the MDL module. Graph-level labels are returned as
metadata and never enter the compression objective unless the caller explicitly
copies them onto nodes or edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np


NodeLabelMode = Literal["auto", "none", "file", "degree"]
EdgeLabelMode = Literal["auto", "none", "file", "constant"]


@dataclass(frozen=True)
class TUDataset:
    """A TU graph dataset represented as ordinary NetworkX graphs."""

    name: str
    graphs: list[nx.Graph]
    graph_labels: np.ndarray
    node_label_key: str | None
    edge_label_key: str | None
    source_directory: Path


def _first_existing(directory: Path, names: list[str]) -> Path | None:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    return None


def _read_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            values = stripped.replace(",", " ").split()
            if not values:
                continue
            rows.append(values)
    if not rows:
        raise ValueError(f"No data rows found in {path}.")
    return rows


def _read_int_vector(path: Path) -> np.ndarray:
    rows = _read_rows(path)
    try:
        return np.asarray([int(row[0]) for row in rows], dtype=int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected one integer per row in {path}.") from exc


def _read_label_vector(path: Path) -> list[object]:
    rows = _read_rows(path)
    values: list[object] = []
    for row in rows:
        try:
            parsed = tuple(int(value) for value in row)
        except ValueError:
            parsed = tuple(row)
        values.append(parsed[0] if len(parsed) == 1 else parsed)
    return values


def _read_float_vector(path: Path) -> list[tuple[float, ...]]:
    rows = _read_rows(path)
    try:
        return [tuple(float(value) for value in row) for row in rows]
    except ValueError as exc:
        raise ValueError(f"Expected numeric attributes in {path}.") from exc


def _looks_like_dataset_directory(directory: Path, dataset_name: str) -> bool:
    """Return whether ``directory`` contains the three required TU files."""
    if not directory.is_dir():
        return False
    edge_path = _first_existing(
        directory,
        [f"{dataset_name}_A.txt", f"{dataset_name}.edges"],
    )
    indicator_path = _first_existing(
        directory,
        [f"{dataset_name}_graph_indicator.txt", f"{dataset_name}.graph_idx"],
    )
    graph_label_path = _first_existing(
        directory,
        [f"{dataset_name}_graph_labels.txt", f"{dataset_name}.graph_labels"],
    )
    return edge_path is not None and indicator_path is not None and graph_label_path is not None


def _resolve_dataset_directory(data_root: str | Path, dataset_name: str) -> Path:
    """Resolve canonical, legacy, and PyG-style TU dataset layouts.

    Accepted examples include::

        data/REDDIT-MULTI-5K/<files>
        data/REDDIT-MULTI-5K/raw/<files>
        data/some-cache/REDDIT-MULTI-5K/raw/<files>
        /absolute/path/to/a-directory-containing-the-files

    The directory itself does not need to be named after the dataset.
    """
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {root}")

    candidates = [
        root,
        root / dataset_name,
        root / dataset_name / "raw",
        root / "raw",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _looks_like_dataset_directory(candidate, dataset_name):
            return candidate

    # Search by a required edge filename rather than traversing every file.
    # This covers nested caches while keeping the search targeted.
    edge_names = [f"{dataset_name}_A.txt", f"{dataset_name}.edges"]
    matches: list[Path] = []
    for edge_name in edge_names:
        for edge_path in root.rglob(edge_name):
            directory = edge_path.parent.resolve()
            if directory not in seen and _looks_like_dataset_directory(
                directory, dataset_name
            ):
                matches.append(directory)
                seen.add(directory)

    if matches:
        # Prefer the shallowest path, then a deterministic lexical tie-break.
        matches.sort(key=lambda item: (len(item.relative_to(root).parts), str(item)))
        return matches[0]

    expected = [
        root / dataset_name,
        root / dataset_name / "raw",
        root,
    ]
    expected_text = "\n  - ".join(str(path) for path in expected)
    raise FileNotFoundError(
        f"Could not find the TU files for dataset {dataset_name!r} below {root}.\n"
        "Looked for either the canonical files "
        f"{dataset_name}_A.txt, {dataset_name}_graph_indicator.txt, and "
        f"{dataset_name}_graph_labels.txt, or the legacy files "
        f"{dataset_name}.edges, {dataset_name}.graph_idx, and "
        f"{dataset_name}.graph_labels.\n"
        f"Common locations are:\n  - {expected_text}"
    )


def load_tu_dataset(
    data_root: str | Path,
    dataset_name: str,
    *,
    node_label_mode: NodeLabelMode = "auto",
    edge_label_mode: EdgeLabelMode = "auto",
    keep_attributes: bool = True,
) -> TUDataset:
    """Load a TU Dortmund graph-classification dataset.

    Parameters
    ----------
    data_root:
        Directory containing a dataset subdirectory, or the dataset directory
        itself.
    dataset_name:
        TU dataset stem, for example ``"REDDIT-MULTI-5K"`` or ``"MUTAG"``.
    node_label_mode:
        ``"auto"`` uses discrete node labels when a file is present and is
        topology-only otherwise. ``"degree"`` reproduces the degree-label
        variant used in the exploratory REDDIT notebooks.
    edge_label_mode:
        ``"auto"`` uses discrete edge labels when present and is topology-only
        otherwise. ``"constant"`` adds one shared edge label.
    keep_attributes:
        Load optional continuous TU node/edge attribute files as uncoded
        ``tu_node_attributes`` and ``tu_edge_attributes`` metadata. These are
        not automatically passed to the MDL schema.
    """
    if node_label_mode not in {"auto", "none", "file", "degree"}:
        raise ValueError(f"Unknown node_label_mode {node_label_mode!r}.")
    if edge_label_mode not in {"auto", "none", "file", "constant"}:
        raise ValueError(f"Unknown edge_label_mode {edge_label_mode!r}.")

    directory = _resolve_dataset_directory(data_root, dataset_name)
    stem = dataset_name

    edge_path = _first_existing(
        directory,
        [f"{stem}_A.txt", f"{stem}.edges"],
    )
    indicator_path = _first_existing(
        directory,
        [f"{stem}_graph_indicator.txt", f"{stem}.graph_idx"],
    )
    graph_label_path = _first_existing(
        directory,
        [f"{stem}_graph_labels.txt", f"{stem}.graph_labels"],
    )
    if edge_path is None or indicator_path is None or graph_label_path is None:
        raise FileNotFoundError(
            "A TU dataset requires edge, graph-indicator, and graph-label files. "
            f"Resolved edge={edge_path}, indicator={indicator_path}, "
            f"graph_labels={graph_label_path}."
        )

    node_label_path = _first_existing(
        directory,
        [f"{stem}_node_labels.txt", f"{stem}.node_labels"],
    )
    edge_label_path = _first_existing(
        directory,
        [f"{stem}_edge_labels.txt", f"{stem}.edge_labels"],
    )
    node_attribute_path = _first_existing(
        directory,
        [f"{stem}_node_attributes.txt", f"{stem}.node_attributes"],
    )
    edge_attribute_path = _first_existing(
        directory,
        [f"{stem}_edge_attributes.txt", f"{stem}.edge_attributes"],
    )

    graph_indicator = _read_int_vector(indicator_path)
    graph_labels = _read_int_vector(graph_label_path)
    edge_rows = _read_rows(edge_path)
    try:
        raw_edges = [(int(row[0]), int(row[1])) for row in edge_rows]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Expected integer edge pairs in {edge_path}.") from exc

    n_nodes = len(graph_indicator)
    if not raw_edges:
        raise ValueError(f"No edges found in {edge_path}.")
    minimum_node_id = min(min(source, target) for source, target in raw_edges)
    if minimum_node_id == 0:
        node_offset = 0
    elif minimum_node_id == 1:
        node_offset = 1
    else:
        raise ValueError(
            f"Could not infer 0- or 1-based TU node indexing from {edge_path}."
        )

    edge_pairs = [
        (source - node_offset, target - node_offset)
        for source, target in raw_edges
    ]
    for source, target in edge_pairs:
        if source < 0 or target < 0 or source >= n_nodes or target >= n_nodes:
            raise ValueError(
                f"Edge {(source, target)} is outside the node range 0..{n_nodes - 1}."
            )

    node_labels = _read_label_vector(node_label_path) if node_label_path else None
    edge_labels = _read_label_vector(edge_label_path) if edge_label_path else None
    node_attributes = (
        _read_float_vector(node_attribute_path)
        if keep_attributes and node_attribute_path
        else None
    )
    edge_attributes = (
        _read_float_vector(edge_attribute_path)
        if keep_attributes and edge_attribute_path
        else None
    )

    if node_labels is not None and len(node_labels) != n_nodes:
        raise ValueError(
            f"Node-label count {len(node_labels)} does not match {n_nodes} nodes."
        )
    if node_attributes is not None and len(node_attributes) != n_nodes:
        raise ValueError(
            f"Node-attribute count {len(node_attributes)} does not match {n_nodes} nodes."
        )
    if edge_labels is not None and len(edge_labels) != len(edge_pairs):
        raise ValueError(
            f"Edge-label count {len(edge_labels)} does not match {len(edge_pairs)} rows."
        )
    if edge_attributes is not None and len(edge_attributes) != len(edge_pairs):
        raise ValueError(
            f"Edge-attribute count {len(edge_attributes)} does not match {len(edge_pairs)} rows."
        )

    graph_ids = sorted(set(int(value) for value in graph_indicator))
    if len(graph_ids) != len(graph_labels):
        raise ValueError(
            f"Found {len(graph_ids)} graph IDs but {len(graph_labels)} graph labels."
        )
    graph_id_to_position = {graph_id: index for index, graph_id in enumerate(graph_ids)}
    graphs = [nx.Graph() for _ in graph_ids]
    local_maps: list[dict[int, int]] = [dict() for _ in graph_ids]

    for global_node, graph_id_value in enumerate(graph_indicator):
        graph_id = int(graph_id_value)
        position = graph_id_to_position[graph_id]
        graph = graphs[position]
        local_node = len(local_maps[position])
        local_maps[position][global_node] = local_node
        attrs: dict[str, object] = {"tu_global_node_id": global_node + node_offset}
        if node_labels is not None:
            attrs["tu_node_label"] = node_labels[global_node]
        if node_attributes is not None:
            attrs["tu_node_attributes"] = node_attributes[global_node]
        graph.add_node(local_node, **attrs)

    for edge_index, (global_source, global_target) in enumerate(edge_pairs):
        source_graph_id = int(graph_indicator[global_source])
        target_graph_id = int(graph_indicator[global_target])
        if source_graph_id != target_graph_id:
            raise ValueError(
                f"Cross-graph edge found at row {edge_index}: "
                f"{(global_source, global_target)}."
            )
        position = graph_id_to_position[source_graph_id]
        source = local_maps[position][global_source]
        target = local_maps[position][global_target]
        if source == target:
            # The MDL module rejects self loops; fail at load time with context.
            raise ValueError(f"Self-loop found at edge row {edge_index}.")
        attrs: dict[str, object] = {}
        if edge_labels is not None:
            attrs["tu_edge_label"] = edge_labels[edge_index]
        if edge_attributes is not None:
            attrs["tu_edge_attributes"] = edge_attributes[edge_index]
        graph = graphs[position]
        if graph.has_edge(source, target):
            existing = graph.edges[source, target]
            for key, value in attrs.items():
                if key in existing and existing[key] != value:
                    raise ValueError(
                        f"Conflicting duplicate edge metadata for {(source, target)}: "
                        f"{existing[key]!r} versus {value!r}."
                    )
                existing.setdefault(key, value)
        else:
            graph.add_edge(source, target, **attrs)

    node_label_key: str | None
    if node_label_mode == "none":
        node_label_key = None
    elif node_label_mode == "degree":
        for graph in graphs:
            nx.set_node_attributes(
                graph,
                {node: int(graph.degree[node]) for node in graph.nodes()},
                "tu_degree_label",
            )
        node_label_key = "tu_degree_label"
    elif node_label_mode == "file":
        if node_labels is None:
            raise FileNotFoundError(
                f"node_label_mode='file' but no node-label file exists in {directory}."
            )
        node_label_key = "tu_node_label"
    else:  # auto
        node_label_key = "tu_node_label" if node_labels is not None else None

    edge_label_key: str | None
    if edge_label_mode == "none":
        edge_label_key = None
    elif edge_label_mode == "constant":
        for graph in graphs:
            nx.set_edge_attributes(graph, 0, "tu_constant_edge_label")
        edge_label_key = "tu_constant_edge_label"
    elif edge_label_mode == "file":
        if edge_labels is None:
            raise FileNotFoundError(
                f"edge_label_mode='file' but no edge-label file exists in {directory}."
            )
        edge_label_key = "tu_edge_label"
    else:  # auto
        edge_label_key = "tu_edge_label" if edge_labels is not None else None

    for position, graph in enumerate(graphs):
        graph.graph["tu_dataset"] = dataset_name
        graph.graph["tu_graph_id"] = graph_ids[position]
        graph.graph["tu_graph_label"] = int(graph_labels[position])

    return TUDataset(
        name=dataset_name,
        graphs=graphs,
        graph_labels=graph_labels,
        node_label_key=node_label_key,
        edge_label_key=edge_label_key,
        source_directory=directory,
    )

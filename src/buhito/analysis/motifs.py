"""Human-readable motif descriptions, cost tables, and portable exports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import math
import re

import networkx as nx
import numpy as np
import pandas as pd


_ASSET_COLUMNS = (
    "motif_id",
    "key",
    "human_name",
    "json_path",
    "graphml_path",
)


def enrich_motif_table(
    motifs: pd.DataFrame,
    motif_graphs: Mapping[str, nx.Graph],
) -> pd.DataFrame:
    """Add stable IDs and human-readable structural descriptions.

    The graph key remains the exact machine identifier used by the compressor.
    ``motif_id`` is a short run-local identifier intended for tables and plots.
    """

    table = motifs.copy()
    if table.empty:
        return table

    if "motif_id" not in table:
        ranks = pd.to_numeric(
            table.get("rank", pd.Series(range(1, len(table) + 1))),
            errors="coerce",
        )
        fallback = np.arange(1, len(table) + 1, dtype=int)
        ranks = ranks.fillna(pd.Series(fallback, index=table.index)).astype(int)
        table.insert(1 if "rank" in table else 0, "motif_id", [
            f"M{rank:03d}" for rank in ranks
        ])

    descriptors: list[dict[str, Any]] = []
    for row in table.itertuples(index=False):
        key = str(getattr(row, "key", ""))
        motif_id = str(getattr(row, "motif_id", ""))
        graph = motif_graphs.get(key)
        if graph is None:
            descriptors.append(_empty_descriptor(motif_id, key))
            continue
        descriptors.append(
            describe_motif_graph(
                graph,
                motif_id=motif_id,
                key=key,
            )
        )

    descriptor_frame = pd.DataFrame(descriptors, index=table.index)
    for column in descriptor_frame:
        if column in table:
            table[column] = table[column].where(
                table[column].notna(), descriptor_frame[column]
            )
        else:
            table[column] = descriptor_frame[column]
    return table


def describe_motif_graph(
    graph: nx.Graph,
    *,
    motif_id: str,
    key: str,
) -> dict[str, Any]:
    """Return a compact, deterministic description of one candidate motif."""

    nodes = sorted(graph.nodes(), key=_stable_token)
    degrees = tuple(sorted((int(graph.degree(node)) for node in nodes), reverse=True))
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    topology = topology_name(graph)
    node_rows = [
        {
            "port": _jsonable(node),
            "attributes": _jsonable(dict(sorted(graph.nodes[node].items()))),
        }
        for node in nodes
    ]
    edge_rows = []
    for source, target, data in graph.edges(data=True):
        left, right = sorted((source, target), key=_stable_token)
        edge_rows.append(
            {
                "source": _jsonable(left),
                "target": _jsonable(right),
                "attributes": _jsonable(dict(sorted(data.items()))),
            }
        )
    edge_rows.sort(
        key=lambda row: (
            _stable_token(row["source"]),
            _stable_token(row["target"]),
            _stable_token(row["attributes"]),
        )
    )

    node_pattern = _attribute_pattern(
        [dict(graph.nodes[node]) for node in nodes],
        empty_label="unlabeled nodes",
    )
    edge_pattern = _attribute_pattern(
        [dict(data) for _, _, data in graph.edges(data=True)],
        empty_label="unlabeled edges",
    )
    label_pattern = f"{node_pattern}; {edge_pattern}"
    fingerprint_payload = {
        "nodes": node_rows,
        "edges": edge_rows,
    }
    fingerprint = sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]

    human_name = f"{motif_id} {topology} ({n_nodes} nodes, {n_edges} edges)"
    return {
        "motif_id": motif_id,
        "key": key,
        "human_name": human_name,
        "topology_name": topology,
        "degree_sequence": degrees,
        "density": float(nx.density(graph)) if n_nodes > 1 else 0.0,
        "is_tree": bool(n_nodes > 0 and nx.is_tree(graph)),
        "is_cycle": bool(
            n_nodes >= 3
            and n_edges == n_nodes
            and all(graph.degree(node) == 2 for node in nodes)
        ),
        "is_clique": bool(
            n_nodes > 0 and n_edges == n_nodes * (n_nodes - 1) // 2
        ),
        "is_bipartite": bool(nx.is_bipartite(graph)),
        "node_label_pattern": node_pattern,
        "edge_label_pattern": edge_pattern,
        "label_pattern": label_pattern,
        "labeled_edge_list": _edge_list_text(edge_rows),
        "motif_fingerprint": fingerprint,
    }


def topology_name(graph: nx.Graph) -> str:
    """Return a concise topology class for a connected simple motif."""

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    if n_nodes == 0:
        return "empty"
    if n_nodes == 1:
        return "singleton"
    if n_nodes == 2 and n_edges == 1:
        return "edge"
    if n_edges == n_nodes * (n_nodes - 1) // 2:
        if n_nodes == 3:
            return "triangle"
        return f"clique-K{n_nodes}"
    if nx.is_tree(graph):
        degrees = [graph.degree(node) for node in graph]
        if max(degrees, default=0) <= 2:
            return f"path-P{n_nodes}"
        if degrees.count(1) == n_nodes - 1:
            return f"star-S{n_nodes - 1}"
        return "tree"
    if n_nodes >= 3 and n_edges == n_nodes and all(
        graph.degree(node) == 2 for node in graph
    ):
        return f"cycle-C{n_nodes}"
    if nx.is_bipartite(graph):
        return "bipartite motif"
    return "connected motif"


def build_motif_cost_table(motifs: pd.DataFrame) -> pd.DataFrame:
    """Convert forced candidate accounting into a long-form cost table.

    The components sum to ``forced_encoded_bits``. ``passthrough_baseline`` is
    the baseline code retained for graphs that were not rewritten by the forced
    candidate selector.
    """

    columns = [
        "motif_id",
        "key",
        "human_name",
        "component",
        "bits",
        "fraction_of_encoded",
        "forced_baseline_bits",
        "forced_encoded_bits",
        "forced_net_savings_bits",
        "is_selected",
        "accounting_residual_bits",
    ]
    required = {
        "forced_encoded_bits",
        "forced_net_savings_bits",
    }
    if motifs.empty or not required.issubset(motifs.columns):
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for row in motifs.to_dict(orient="records"):
        encoded = _number(row.get("forced_encoded_bits"))
        baseline = _number(row.get("forced_baseline_bits"))
        dictionary = _number(row.get("forced_dictionary_bits"), default=0.0)
        template = _number(row.get("forced_template_bits"), default=0.0)
        boundary = _number(row.get("forced_boundary_bits"), default=0.0)
        selector = _number(row.get("forced_selector_bits"), default=0.0)
        model_choice = _number(
            row.get("forced_model_choice_bits"), default=0.0
        )

        known = dictionary + template + boundary + selector + model_choice
        passthrough = encoded - known if math.isfinite(encoded) else math.nan
        if math.isfinite(passthrough) and abs(passthrough) < 1e-10:
            passthrough = 0.0
        components = {
            "passthrough_baseline": passthrough,
            "dictionary": dictionary,
            "rewrite_template": template,
            "boundary_ports": boundary,
            "selector": selector,
            "model_choice": model_choice,
        }
        reconstructed = sum(
            value for value in components.values() if math.isfinite(value)
        )
        residual = encoded - reconstructed if math.isfinite(encoded) else math.nan
        for component, bits in components.items():
            rows.append(
                {
                    "motif_id": row.get("motif_id"),
                    "key": row.get("key"),
                    "human_name": row.get("human_name"),
                    "component": component,
                    "bits": bits,
                    "fraction_of_encoded": (
                        bits / encoded
                        if math.isfinite(bits) and encoded > 0
                        else math.nan
                    ),
                    "forced_baseline_bits": baseline,
                    "forced_encoded_bits": encoded,
                    "forced_net_savings_bits": _number(
                        row.get("forced_net_savings_bits")
                    ),
                    "is_selected": bool(row.get("is_selected", False)),
                    "accounting_residual_bits": residual,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def export_motif_assets(
    motif_graphs: Mapping[str, nx.Graph],
    motifs: pd.DataFrame,
    output_dir: str | Path,
    *,
    max_motifs: int | None = None,
) -> pd.DataFrame:
    """Write JSON and GraphML assets for scored motifs.

    JSON preserves a readable record of node and edge attributes. GraphML uses
    string-safe labels so the files open in common graph visualization tools.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if motifs.empty or not motif_graphs:
        return pd.DataFrame(columns=_ASSET_COLUMNS)

    table = motifs if max_motifs is None else motifs.head(max_motifs)
    manifest: list[dict[str, Any]] = []
    for row in table.to_dict(orient="records"):
        key = str(row.get("key", ""))
        graph = motif_graphs.get(key)
        if graph is None:
            continue
        motif_id = str(row.get("motif_id", "motif"))
        topology = str(row.get("topology_name", "motif"))
        stem = _safe_stem(f"{motif_id}_{topology}")
        json_path = destination / f"{stem}.json"
        graphml_path = destination / f"{stem}.graphml"

        payload = {
            "motif": {
                name: _jsonable(value)
                for name, value in row.items()
                if name not in {"base_key"}
            },
            "nodes": [
                {
                    "port": _jsonable(node),
                    "attributes": _jsonable(dict(data)),
                }
                for node, data in sorted(
                    graph.nodes(data=True), key=lambda item: _stable_token(item[0])
                )
            ],
            "edges": [
                {
                    "source": _jsonable(source),
                    "target": _jsonable(target),
                    "attributes": _jsonable(dict(data)),
                }
                for source, target, data in sorted(
                    graph.edges(data=True),
                    key=lambda item: (
                        _stable_token(item[0]),
                        _stable_token(item[1]),
                    ),
                )
            ],
        }
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        nx.write_graphml(
            _graphml_graph(graph, row),
            graphml_path,
            infer_numeric_types=True,
        )
        manifest.append(
            {
                "motif_id": motif_id,
                "key": key,
                "human_name": row.get("human_name"),
                "json_path": str(json_path.name),
                "graphml_path": str(graphml_path.name),
            }
        )
    return pd.DataFrame(manifest, columns=_ASSET_COLUMNS)


def _graphml_graph(graph: nx.Graph, row: Mapping[str, Any]) -> nx.Graph:
    exported = nx.Graph()
    exported.graph.update(
        {
            "motif_id": str(row.get("motif_id", "")),
            "key": str(row.get("key", "")),
            "human_name": str(row.get("human_name", "")),
            "topology_name": str(row.get("topology_name", "")),
        }
    )
    for node, data in graph.nodes(data=True):
        exported.add_node(
            str(node),
            port=str(node),
            label=_attribute_text(dict(data)),
        )
    for source, target, data in graph.edges(data=True):
        exported.add_edge(
            str(source),
            str(target),
            label=_attribute_text(dict(data)),
        )
    return exported


def _empty_descriptor(motif_id: str, key: str) -> dict[str, Any]:
    return {
        "motif_id": motif_id,
        "key": key,
        "human_name": f"{motif_id} unavailable motif",
        "topology_name": "unavailable",
        "degree_sequence": (),
        "density": math.nan,
        "is_tree": False,
        "is_cycle": False,
        "is_clique": False,
        "is_bipartite": False,
        "node_label_pattern": "unavailable",
        "edge_label_pattern": "unavailable",
        "label_pattern": "unavailable",
        "labeled_edge_list": "unavailable",
        "motif_fingerprint": sha256(key.encode("utf-8")).hexdigest()[:16],
    }


def _attribute_pattern(
    records: list[dict[str, Any]],
    *,
    empty_label: str,
) -> str:
    if not records or not any(records):
        return empty_label
    counts = Counter(_attribute_text(record) for record in records)
    return ", ".join(
        f"{label} x{count}"
        for label, count in sorted(counts.items())
    )


def _attribute_text(attributes: Mapping[str, Any]) -> str:
    if not attributes:
        return "unlabeled"
    return ", ".join(
        f"{key}={_short_value(value)}"
        for key, value in sorted(attributes.items())
    )


def _edge_list_text(edge_rows: list[dict[str, Any]]) -> str:
    if not edge_rows:
        return "no edges"
    return "; ".join(
        f"{row['source']}--{row['target']}[{_attribute_text(row['attributes'])}]"
        for row in edge_rows
    )


def _short_value(value: Any, *, limit: int = 80) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned or "motif"


def _stable_token(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def _number(value: Any, *, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)

"""Minimum-description-length graphlet compression for graph datasets.

This module turns the exploratory MDL code used in the Buhito notebooks into a
model-agnostic, reusable API.  It learns a frozen dictionary of repeated
connected induced graphlets from a training corpus, contracts node-disjoint
occurrences into motif supernodes, records boundary-port metadata needed for
exact reconstruction, and compares the resulting code to a labeled edgelist
baseline.

The implementation is intentionally independent of any downstream predictor.
The compressed NetworkX graphs can be adapted to a GNN, a graph kernel, a
linear model, or any other learner.  Reconstruction is lossless with respect to
undirected simple topology and the node/edge attributes selected by
``GraphSchema``.  NetworkX node identifiers and unselected attributes are not
part of the code and are therefore not guaranteed to survive round-trip.

The codelengths are analytical MDL codelengths.  They are not the byte size of
``pickle`` or another general-purpose serializer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import cached_property
from itertools import combinations
from math import lgamma
from pathlib import Path
from typing import Any, Literal, Protocol
import gzip
import hashlib
import math
import pickle

import networkx as nx
import numpy as np
import pandas as pd


__all__ = [
    "Alphabets",
    "BuhitoGraphletEnumerator",
    "CompressionResult",
    "DatasetCode",
    "DictionarySelection",
    "EncodedGraph",
    "ExhaustiveGraphletEnumerator",
    "GraphSchema",
    "GraphletEnumerator",
    "MDLGraphCompressor",
    "MotifRule",
    "Occurrence",
    "Rewrite",
    "decode_rewrite",
    "labeled_isomorphic",
]


# Internal attribute names are deliberately unlikely to collide with user data.
_NODE_LABEL = "__buhito_mdl_node_label__"
_EDGE_LABEL = "__buhito_mdl_edge_label__"
_NODE_KIND = "__buhito_mdl_kind__"
_PORTS = "__buhito_mdl_ports__"
_DATA_TAG = "data"
_MOTIF_TAG = "motif"
_EDGE_NODE_TAG = "edge"
_INCIDENCE_LABEL = ("buhito-mdl", "incidence")
_TOPOLOGY_ONLY = ("buhito-mdl", "unlabeled")
_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Public schemas and result containers
# ---------------------------------------------------------------------------


def _as_key_tuple(keys: str | Sequence[str] | None) -> tuple[str, ...]:
    if keys is None:
        return ()
    if isinstance(keys, str):
        return (keys,)
    result = tuple(str(key) for key in keys)
    if len(set(result)) != len(result):
        raise ValueError("Attribute keys must be unique.")
    return result


def _require_hashable(value: Any, *, context: str) -> Hashable:
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(
            f"{context} must be hashable for Buhito graphlet hashing; "
            f"received {type(value).__name__}. Convert lists/dicts/arrays "
            "to tuples or provide a preprocessed attribute."
        ) from exc
    return value


@dataclass(frozen=True)
class GraphSchema:
    """Attributes preserved by the graph code.

    Parameters
    ----------
    node_label_keys:
        One or more node attribute names.  An empty tuple means topology-only.
    edge_label_keys:
        One or more edge attribute names.  An empty tuple means topology-only.

    Notes
    -----
    Only these attributes and undirected simple topology are included in the
    MDL code.  Targets, coordinates, graph metadata, and other attributes should
    remain in the surrounding dataset object.
    """

    node_label_keys: tuple[str, ...] = ()
    edge_label_keys: tuple[str, ...] = ()

    @classmethod
    def from_keys(
        cls,
        *,
        node_label_keys: str | Sequence[str] | None = None,
        edge_label_keys: str | Sequence[str] | None = None,
    ) -> "GraphSchema":
        return cls(
            node_label_keys=_as_key_tuple(node_label_keys),
            edge_label_keys=_as_key_tuple(edge_label_keys),
        )

    def _node_value(self, data: Mapping[str, Any], node: Hashable) -> Hashable:
        if not self.node_label_keys:
            return _TOPOLOGY_ONLY
        missing = [key for key in self.node_label_keys if key not in data]
        if missing:
            raise ValueError(f"Node {node!r} is missing attributes {missing!r}.")
        values = tuple(data[key] for key in self.node_label_keys)
        return _require_hashable(values, context=f"Label for node {node!r}")

    def _edge_value(
        self,
        data: Mapping[str, Any],
        edge: tuple[Hashable, Hashable],
    ) -> Hashable:
        if not self.edge_label_keys:
            return _TOPOLOGY_ONLY
        missing = [key for key in self.edge_label_keys if key not in data]
        if missing:
            raise ValueError(f"Edge {edge!r} is missing attributes {missing!r}.")
        values = tuple(data[key] for key in self.edge_label_keys)
        return _require_hashable(values, context=f"Label for edge {edge!r}")

    def normalize(self, graph: nx.Graph) -> nx.Graph:
        """Return an integer-labeled graph containing only coded attributes."""
        _assert_supported_graph(graph)
        original_nodes = tuple(graph.nodes())
        old_to_new = {node: index for index, node in enumerate(original_nodes)}

        normalized = nx.Graph()
        normalized.graph.update(graph.graph)
        normalized.graph["__buhito_mdl_original_node_order__"] = original_nodes

        for node in original_nodes:
            value = self._node_value(graph.nodes[node], node)
            normalized.add_node(
                old_to_new[node],
                **{
                    _NODE_LABEL: (_DATA_TAG, value),
                    _NODE_KIND: "data",
                },
            )

        for source, target, data in graph.edges(data=True):
            value = self._edge_value(data, (source, target))
            normalized.add_edge(
                old_to_new[source],
                old_to_new[target],
                **{_EDGE_LABEL: (_DATA_TAG, value)},
            )

        return normalized

    def restore(self, graph: nx.Graph) -> nx.Graph:
        """Expand internal labels back into the selected user attribute keys.

        A rewrite template is a ``MultiGraph`` because contraction can create
        parallel boundary edges.  The restored model view therefore preserves
        the input graph class instead of silently collapsing those edges.
        """
        restored: nx.Graph
        restored = nx.MultiGraph() if graph.is_multigraph() else nx.Graph()
        restored.graph.update(
            {
                key: value
                for key, value in graph.graph.items()
                if key != "__buhito_mdl_original_node_order__"
            }
        )

        for node, data in graph.nodes(data=True):
            label = data[_NODE_LABEL]
            attrs: dict[str, Any] = {}
            if label[0] == _DATA_TAG:
                if self.node_label_keys:
                    values = label[1]
                    attrs.update(dict(zip(self.node_label_keys, values)))
            else:
                # A compressed motif node has no direct counterpart in the
                # original schema.  Preserve its internal role for model use.
                attrs["mdl_label"] = label
                attrs["mdl_kind"] = data.get(_NODE_KIND, "unknown")
            restored.add_node(node, **attrs)

        edge_iter = (
            graph.edges(keys=True, data=True)
            if graph.is_multigraph()
            else ((u, v, None, data) for u, v, data in graph.edges(data=True))
        )
        for source, target, edge_key, data in edge_iter:
            label = data[_EDGE_LABEL]
            attrs: dict[str, Any] = {}
            if label[0] == _DATA_TAG:
                if self.edge_label_keys:
                    values = label[1]
                    attrs.update(dict(zip(self.edge_label_keys, values)))
            else:
                attrs["mdl_label"] = label
            if _PORTS in data:
                attrs["mdl_ports"] = dict(data[_PORTS])
            if restored.is_multigraph():
                restored.add_edge(source, target, key=edge_key, **attrs)
            else:
                restored.add_edge(source, target, **attrs)

        return restored


@dataclass(frozen=True)
class Occurrence:
    """Host nodes in motif-port order."""

    mapping: tuple[Hashable, ...]

    @property
    def nodes(self) -> frozenset[Hashable]:
        return frozenset(self.mapping)


@dataclass
class MotifRule:
    """A frozen dictionary entry learned from the fit corpus."""

    key: str
    base_key: Hashable
    motif: nx.Graph
    rank: int = 0
    _orbit_cache: dict[int, int] = field(default_factory=dict, repr=False)

    @cached_property
    def automorphisms(self) -> tuple[tuple[int, ...], ...]:
        node_match = nx.algorithms.isomorphism.categorical_node_match(
            _NODE_LABEL, None
        )
        edge_match = nx.algorithms.isomorphism.categorical_edge_match(
            _EDGE_LABEL, None
        )
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            self.motif,
            self.motif,
            node_match=node_match,
            edge_match=edge_match,
        )
        n_nodes = self.motif.number_of_nodes()
        return tuple(
            tuple(mapping[port] for port in range(n_nodes))
            for mapping in matcher.isomorphisms_iter()
        )

    def ordered_port_orbit_count(self, boundary_edges: int) -> int:
        """Number of ordered port strings modulo label-aware automorphisms."""
        boundary_edges = int(boundary_edges)
        if boundary_edges <= 0:
            return 1
        cached = self._orbit_cache.get(boundary_edges)
        if cached is not None:
            return cached
        fixed_sum = 0
        for automorphism in self.automorphisms:
            fixed_vertices = sum(
                1 for port, image in enumerate(automorphism) if port == image
            )
            fixed_sum += fixed_vertices**boundary_edges
        count = fixed_sum // max(len(self.automorphisms), 1)
        self._orbit_cache[boundary_edges] = count
        return count

    @property
    def automorphism_order(self) -> int:
        return len(self.automorphisms)

    @property
    def orbit_sizes(self) -> tuple[int, ...]:
        unseen = set(self.motif.nodes())
        sizes: list[int] = []
        while unseen:
            vertex = min(unseen)
            orbit = {mapping[vertex] for mapping in self.automorphisms}
            unseen.difference_update(orbit)
            sizes.append(len(orbit))
        return tuple(sorted(sizes))


@dataclass
class Rewrite:
    """A reversible multi-rule graphlet rewrite."""

    rules: tuple[MotifRule, ...]
    template: nx.MultiGraph
    supernodes: tuple[Hashable, ...]
    supernode_rule: dict[Hashable, int]
    selected: tuple[tuple[int, Occurrence], ...]
    raw_signatures: dict[Hashable, tuple]
    canonical_signatures: dict[Hashable, tuple]
    boundary_contexts: dict[Hashable, tuple]
    canonical_port_to_host: dict[Hashable, tuple[Hashable, ...]] = field(
        default_factory=dict
    )
    graph_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EncodedGraph:
    """Per-graph output of a fitted MDL dictionary."""

    graph_index: int
    baseline_graph: nx.Graph
    rewrite: Rewrite | None
    use_rewrite: bool
    baseline_bits: float
    template_bits: float | None
    boundary_bits: float | None
    rewrite_bits: float | None
    candidate_occurrences: int
    selected_occurrences: int
    rules_used: int
    total_graphlet_occurrences: int
    unseen_graphlet_occurrences: int

    @property
    def gross_gain_bits(self) -> float:
        if self.rewrite_bits is None:
            return 0.0
        return self.baseline_bits - self.rewrite_bits

    @property
    def unseen_graphlet_fraction(self) -> float:
        if self.total_graphlet_occurrences == 0:
            return 0.0
        return self.unseen_graphlet_occurrences / self.total_graphlet_occurrences

    def normalized_model_graph(self, *, force_rewrite: bool = False) -> nx.Graph:
        """Return the graph representation presented to a downstream model."""
        if self.rewrite is not None and (force_rewrite or self.use_rewrite):
            return self.rewrite.template.copy()
        return self.baseline_graph.copy()

    def normalized_decoded_graph(self) -> nx.Graph:
        if self.rewrite is not None and self.use_rewrite:
            return decode_rewrite(self.rewrite)
        return self.baseline_graph.copy()


@dataclass(frozen=True)
class DatasetCode:
    selector: str
    n_graphs: int
    n_eligible: int
    n_rewritten: int
    n_occurrences: int
    baseline_bits: float
    dictionary_bits: float
    selector_bits: float
    model_choice_bits: float
    encoded_bits: float
    net_savings_bits: float
    bits_per_graph: float
    fraction_rewritten: float


@dataclass
class CompressionResult:
    """A complete corpus result independent of a downstream ML model."""

    schema: GraphSchema
    rules: tuple[MotifRule, ...]
    records: list[EncodedGraph]
    report: DatasetCode
    per_graph: pd.DataFrame
    selector_curve: pd.DataFrame

    def model_graphs(self, *, force_rewrite: bool = False) -> list[nx.Graph]:
        """Return restored NetworkX graphs for any downstream model adapter."""
        return [
            self.schema.restore(
                record.normalized_model_graph(force_rewrite=force_rewrite)
            )
            for record in self.records
        ]

    def decoded_graphs(self) -> list[nx.Graph]:
        """Decode all selected records and restore the selected attributes."""
        return [
            self.schema.restore(record.normalized_decoded_graph())
            for record in self.records
        ]


# ---------------------------------------------------------------------------
# Graphlet enumeration backends
# ---------------------------------------------------------------------------


class GraphletEnumerator(Protocol):
    """Protocol for connected induced graphlet occurrence enumeration."""

    name: str

    def enumerate(
        self,
        graph: nx.Graph,
        sizes: tuple[int, ...],
    ) -> dict[Hashable, list[frozenset[int]]]:
        """Map a backend key to occurrence node sets."""


class BuhitoGraphletEnumerator:
    """Production backend using Buhito's breadth-first graphlet enumerator."""

    name = "buhito-bfs"

    def enumerate(
        self,
        graph: nx.Graph,
        sizes: tuple[int, ...],
    ) -> dict[Hashable, list[frozenset[int]]]:
        try:
            from .featurizers.bfs_graphlet_featurizer import (
                generate_subgraphs_breadthwise,
            )
        except ImportError:  # pragma: no cover - only relevant outside package
            from . import generate_subgraphs_breadthwise  # type: ignore

        _, bitinfo = generate_subgraphs_breadthwise(
            graph,
            depth=max(sizes),
            return_nodewise=False,
            full_hash=True,
            node_key=_NODE_LABEL,
            edge_key=_EDGE_LABEL,
        )
        wanted = set(sizes)
        return {
            key: [frozenset(nodes) for nodes in occurrence_sets]
            for key, occurrence_sets in bitinfo.items()
            if int(key[0]) in wanted
        }


class ExhaustiveGraphletEnumerator:
    """Slow exact backend for tests and very small graphs.

    All connected subsets of the requested sizes are returned in a size bucket;
    the compressor then splits that bucket into exact labeled isomorphism
    classes.  This backend is intentionally exponential and should not replace
    Buhito on real datasets.
    """

    name = "exhaustive"

    def enumerate(
        self,
        graph: nx.Graph,
        sizes: tuple[int, ...],
    ) -> dict[Hashable, list[frozenset[int]]]:
        result: dict[Hashable, list[frozenset[int]]] = defaultdict(list)
        nodes = tuple(sorted(graph.nodes()))
        for size in sizes:
            for subset in combinations(nodes, size):
                induced = graph.subgraph(subset)
                if nx.is_connected(induced):
                    result[(size, "exact")].append(frozenset(subset))
        return dict(result)


# ---------------------------------------------------------------------------
# MDL primitives
# ---------------------------------------------------------------------------


def _assert_supported_graph(graph: nx.Graph) -> None:
    if graph.is_directed():
        raise TypeError("MDLGraphCompressor currently supports undirected graphs only.")
    if graph.is_multigraph():
        raise TypeError("Input graphs must be simple; the rewrite template may be a MultiGraph.")
    if nx.number_of_selfloops(graph):
        raise ValueError("Input graphs with self-loops are not supported.")


def _stable_token(value: Hashable) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def _log2_factorial(n: int) -> float:
    return 0.0 if n < 2 else lgamma(n + 1) / _LN2


def _positive_integer_bits(n: int) -> float:
    """Notebook-compatible universal code for a positive integer."""
    n = max(int(n), 1)
    return math.log2(n) + math.log2(n + 1)


def _log2_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return math.inf
    return _log2_factorial(n) - _log2_factorial(k) - _log2_factorial(n - k)


def _log_gamma_difference(log_x: float, n: int) -> float:
    """Compute ``lgamma(x) - lgamma(x + n)`` given only ``log(x)``.

    ``x`` (typically a Dirichlet concentration derived from a boundary-port
    alphabet size) can be astronomically large -- large enough that it
    cannot be represented as a finite float at all, or large enough that
    float64 can no longer distinguish ``x`` from ``x + n``. In either case,
    computing ``x`` directly and subtracting is either impossible or
    silently wrong (it collapses to a constant, alphabet-size-independent
    value once precision is lost).

    Instead: ``log(Gamma(x) / Gamma(x + n)) = -sum_{k=0}^{n-1} log(x + k)``,
    which for ``x`` this large relative to ``n`` is accurately approximated
    by ``-n * log(x)`` (the next-order correction term, of order
    ``n * (n - 1) / (2 * x)``, is negligible whenever the approximation is
    actually needed). We use the exact computation whenever float64 can
    represent it meaningfully, and fall back to the asymptotic form
    otherwise -- decided dynamically per call, not via a fixed threshold.
    """
    if n <= 0:
        return 0.0
    if log_x < 700.0:  # exp(log_x) is representable as a finite float
        x = math.exp(log_x)
        if (x + n) != x:
            # ``n`` is large enough, relative to ``x``, that the exact
            # subtraction retains real precision.
            return lgamma(x) - lgamma(x + n)
    # ``x`` is either too large to represent as a finite float, or so much
    # larger than ``n`` that float64 cannot distinguish ``x`` from ``x + n``.
    # Both regimes collapse to the same safe asymptotic form.
    return -n * log_x


def _dirichlet_multinomial_bits(
    counts: Counter[Hashable],
    alphabet_size: int,
    alpha: float = 0.5,
) -> float:
    if alphabet_size <= 0:
        return 0.0
    total = sum(counts.values())
    # ``alphabet_size`` can be an astronomically large Python int (Burnside's
    # lemma over many boundary edges for a high-degree motif). Computing
    # ``alphabet_size * alpha`` as a float can overflow outright, and even
    # when it doesn't, ``lgamma(concentration) - lgamma(total + concentration)``
    # silently loses all information about ``total`` once ``concentration``
    # is too large for float64 to represent the difference -- well before
    # the point where it would actually overflow. ``math.log`` safely
    # handles arbitrary-precision ints (unlike float conversion), so we work
    # in log-space throughout instead of ever materializing ``concentration``
    # as a literal float.
    log_concentration = math.log(alphabet_size) + math.log(alpha)
    log_probability = _log_gamma_difference(log_concentration, total)
    for count in counts.values():
        log_probability += lgamma(count + alpha) - lgamma(alpha)
    return -log_probability / _LN2


@dataclass(frozen=True)
class Alphabets:
    node_labels: tuple[Hashable, ...]
    edge_labels: tuple[Hashable, ...]

    @property
    def n_node(self) -> int:
        return len(self.node_labels)

    @property
    def n_edge(self) -> int:
        return len(self.edge_labels)


def _build_alphabets(graphs: Iterable[nx.Graph]) -> Alphabets:
    node_labels: set[Hashable] = set()
    edge_labels: set[Hashable] = set()
    for graph in graphs:
        node_labels.update(
            data[_NODE_LABEL] for _, data in graph.nodes(data=True)
        )
        edge_labels.update(
            data[_EDGE_LABEL] for _, _, data in graph.edges(data=True)
        )
    return Alphabets(
        tuple(sorted(node_labels, key=_stable_token)),
        tuple(sorted(edge_labels, key=_stable_token)),
    )


def _edgelist_topology_bits(graph: nx.Graph) -> float:
    edge_count = graph.number_of_edges()
    if edge_count == 0:
        return 0.0
    return (
        _log2_factorial(2 * edge_count)
        - _log2_factorial(edge_count)
        - edge_count
        - sum(_log2_factorial(degree) for _, degree in graph.degree())
    )


def _label_bits(graph: nx.Graph, alphabets: Alphabets) -> float:
    node_counts = Counter(
        data[_NODE_LABEL] for _, data in graph.nodes(data=True)
    )
    edge_counts = Counter(
        data[_EDGE_LABEL] for _, _, data in graph.edges(data=True)
    )
    return _dirichlet_multinomial_bits(
        node_counts, alphabets.n_node
    ) + _dirichlet_multinomial_bits(edge_counts, alphabets.n_edge)


def _base_bits(
    graph: nx.Graph,
    alphabets: Alphabets,
    *,
    complete: bool = True,
) -> float:
    bits = _edgelist_topology_bits(graph) + _label_bits(graph, alphabets)
    if complete:
        degrees = [degree for _, degree in graph.degree()]
        max_degree = max(degrees, default=0)
        bits += _positive_integer_bits(max(graph.number_of_nodes(), 1))
        bits += _positive_integer_bits(max(max_degree, 1))
        bits += _dirichlet_multinomial_bits(
            Counter(degrees), max_degree + 1
        )
    return bits


def labeled_isomorphic(left: nx.Graph, right: nx.Graph) -> bool:
    """Check topology and the internal node/edge labels."""
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        _NODE_LABEL, None
    )
    edge_match = nx.algorithms.isomorphism.categorical_edge_match(
        _EDGE_LABEL, None
    )
    return nx.is_isomorphic(
        left, right, node_match=node_match, edge_match=edge_match
    )


def _assert_internal_labels(graph: nx.Graph) -> None:
    missing_nodes = [
        node
        for node, data in graph.nodes(data=True)
        if _NODE_LABEL not in data
    ]
    missing_edges = [
        (source, target)
        for source, target, data in graph.edges(data=True)
        if _EDGE_LABEL not in data
    ]
    if missing_nodes or missing_edges:
        raise ValueError(
            f"Missing internal labels: nodes={missing_nodes[:5]!r}, "
            f"edges={missing_edges[:5]!r}."
        )


# ---------------------------------------------------------------------------
# Candidate discovery and exact occurrence alignment
# ---------------------------------------------------------------------------


@dataclass
class _Candidate:
    key: str
    base_key: Hashable
    motif: nx.Graph


@dataclass
class _FoundRecord:
    motif: nx.Graph
    instances: list[Occurrence]
    base_key: Hashable


def _matcher(motif: nx.Graph, subgraph: nx.Graph) -> nx.GraphMatcher:
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        _NODE_LABEL, None
    )
    edge_match = nx.algorithms.isomorphism.categorical_edge_match(
        _EDGE_LABEL, None
    )
    return nx.algorithms.isomorphism.GraphMatcher(
        motif,
        subgraph,
        node_match=node_match,
        edge_match=edge_match,
    )


def _make_motif(subgraph: nx.Graph) -> nx.Graph:
    ordered_nodes = tuple(sorted(subgraph.nodes()))
    mapping = {node: index for index, node in enumerate(ordered_nodes)}
    motif = nx.relabel_nodes(subgraph, mapping, copy=True)
    _assert_internal_labels(motif)
    return motif


def _occurrence_for_motif(
    motif: nx.Graph,
    graph: nx.Graph,
    nodes: frozenset[int],
) -> Occurrence | None:
    subgraph = graph.subgraph(nodes).copy()
    matcher = _matcher(motif, subgraph)
    if not matcher.is_isomorphic():
        return None
    return Occurrence(
        tuple(matcher.mapping[port] for port in range(motif.number_of_nodes()))
    )


def _discover_candidates(
    graphs: list[nx.Graph],
    raw_occurrences: list[dict[Hashable, list[frozenset[int]]]],
) -> tuple[dict[str, _Candidate], list[dict[str, _FoundRecord]]]:
    catalog_by_base: dict[Hashable, list[_Candidate]] = defaultdict(list)
    candidates: dict[str, _Candidate] = {}
    found_list: list[dict[str, _FoundRecord]] = []

    for graph, raw in zip(graphs, raw_occurrences):
        found: dict[str, _FoundRecord] = {}
        for base_key in sorted(raw, key=_stable_token):
            occurrence_sets = sorted(
                raw[base_key], key=lambda nodes: tuple(sorted(nodes))
            )
            for nodes in occurrence_sets:
                subgraph = graph.subgraph(nodes).copy()
                candidate: _Candidate | None = None
                occurrence: Occurrence | None = None

                for existing in catalog_by_base[base_key]:
                    maybe = _occurrence_for_motif(existing.motif, graph, nodes)
                    if maybe is not None:
                        candidate = existing
                        occurrence = maybe
                        break

                if candidate is None:
                    motif = _make_motif(subgraph)
                    collision_index = len(catalog_by_base[base_key])
                    key = (
                        f"{_stable_token(base_key)}::iso{collision_index}"
                    )
                    candidate = _Candidate(
                        key=key,
                        base_key=base_key,
                        motif=motif,
                    )
                    catalog_by_base[base_key].append(candidate)
                    candidates[key] = candidate
                    occurrence = _occurrence_for_motif(motif, graph, nodes)
                    if occurrence is None:  # pragma: no cover - defensive
                        raise AssertionError("New motif failed to match its source occurrence.")

                record = found.setdefault(
                    candidate.key,
                    _FoundRecord(
                        motif=candidate.motif,
                        instances=[],
                        base_key=base_key,
                    ),
                )
                record.instances.append(occurrence)
        found_list.append(found)

    return candidates, found_list


def _find_frozen_occurrences(
    graph: nx.Graph,
    raw: dict[Hashable, list[frozenset[int]]],
    rules: tuple[MotifRule, ...],
) -> dict[int, list[Occurrence]]:
    rules_by_base: dict[Hashable, list[MotifRule]] = defaultdict(list)
    for rule in rules:
        rules_by_base[rule.base_key].append(rule)

    result: dict[int, list[Occurrence]] = {rule.rank: [] for rule in rules}
    for base_key, occurrence_sets in raw.items():
        matching_rules = rules_by_base.get(base_key, [])
        if not matching_rules:
            continue
        for nodes in sorted(occurrence_sets, key=lambda x: tuple(sorted(x))):
            for rule in matching_rules:
                occurrence = _occurrence_for_motif(rule.motif, graph, nodes)
                if occurrence is not None:
                    result[rule.rank].append(occurrence)
                    break
    return result


# ---------------------------------------------------------------------------
# Reversible multi-rule rewrite
# ---------------------------------------------------------------------------


def _external_degree(graph: nx.Graph, occurrence: Occurrence) -> int:
    inside = occurrence.nodes
    return sum(
        1
        for vertex in inside
        for neighbor in graph.neighbors(vertex)
        if neighbor not in inside
    )


def _select_nonoverlapping_multi(
    graph: nx.Graph,
    rules: tuple[MotifRule, ...],
    occurrences_by_rule: Mapping[int, Iterable[Occurrence]],
) -> tuple[tuple[int, Occurrence], ...]:
    rule_map = {rule.rank: rule for rule in rules}
    candidates: list[tuple[tuple[Any, ...], int, Occurrence]] = []
    for rule_rank, occurrences in occurrences_by_rule.items():
        rule = rule_map[rule_rank]
        for occurrence in occurrences:
            priority = (
                _external_degree(graph, occurrence),
                -rule.motif.number_of_edges(),
                -rule.motif.number_of_nodes(),
                rule.rank,
                tuple(_stable_token(node) for node in occurrence.mapping),
            )
            candidates.append((priority, rule_rank, occurrence))
    candidates.sort(key=lambda item: item[0])

    selected: list[tuple[int, Occurrence]] = []
    occupied: set[Hashable] = set()
    for _, rule_rank, occurrence in candidates:
        if occurrence.nodes & occupied:
            continue
        selected.append((rule_rank, occurrence))
        occupied.update(occurrence.nodes)
    return tuple(selected)


def _canonical_boundary_signature(
    rule: MotifRule,
    groups: list[tuple[tuple[Any, ...], list[tuple[Any, ...]]]],
) -> tuple[tuple, tuple[int, ...]]:
    best_signature: tuple | None = None
    best_automorphism: tuple[int, ...] | None = None
    for automorphism in rule.automorphisms:
        signature = tuple(
            tuple(sorted(automorphism[reference[-1]] for reference in refs))
            for _, refs in groups
        )
        if best_signature is None or signature < best_signature:
            best_signature = signature
            best_automorphism = automorphism
    if best_signature is None or best_automorphism is None:
        # The empty boundary is fixed by every automorphism.
        return (), tuple(range(rule.motif.number_of_nodes()))
    return best_signature, best_automorphism


def _rewrite_graph(
    graph: nx.Graph,
    rules: tuple[MotifRule, ...],
    occurrences_by_rule: Mapping[int, Iterable[Occurrence]],
) -> Rewrite:
    _assert_internal_labels(graph)
    rule_map = {rule.rank: rule for rule in rules}
    selected = _select_nonoverlapping_multi(graph, rules, occurrences_by_rule)

    host_to_port: dict[Hashable, tuple[Hashable, int]] = {}
    supernodes: list[Hashable] = []
    supernode_rule: dict[Hashable, int] = {}

    for occurrence_index, (rule_rank, occurrence) in enumerate(selected):
        rule = rule_map[rule_rank]
        if len(occurrence.mapping) != rule.motif.number_of_nodes():
            raise ValueError(f"Occurrence for rule {rule.key!r} has the wrong size.")
        supernode = ("__buhito_mdl_motif__", occurrence_index)
        supernodes.append(supernode)
        supernode_rule[supernode] = rule_rank
        for port, host_node in enumerate(occurrence.mapping):
            if host_node in host_to_port:
                raise ValueError("Selected motif occurrences overlap.")
            host_to_port[host_node] = (supernode, port)

    template = nx.MultiGraph()
    template.graph.update(graph.graph)
    for vertex, data in graph.nodes(data=True):
        if vertex not in host_to_port:
            template.add_node(
                vertex,
                **{
                    _NODE_LABEL: data[_NODE_LABEL],
                    _NODE_KIND: "outside",
                },
            )
    for supernode in supernodes:
        rule_rank = supernode_rule[supernode]
        template.add_node(
            supernode,
            **{
                _NODE_LABEL: (_MOTIF_TAG, rule_rank),
                _NODE_KIND: "motif",
            },
        )

    references: dict[Hashable, list[tuple[Any, ...]]] = defaultdict(list)
    for source, target, data in graph.edges(data=True):
        source_info = host_to_port.get(source)
        target_info = host_to_port.get(target)
        if (
            source_info is not None
            and target_info is not None
            and source_info[0] == target_info[0]
        ):
            continue

        template_source = source_info[0] if source_info is not None else source
        template_target = target_info[0] if target_info is not None else target
        edge_key = template.add_edge(
            template_source,
            template_target,
            **{
                _EDGE_LABEL: data[_EDGE_LABEL],
                _PORTS: {},
            },
        )
        edge_reference = (template_source, template_target, edge_key)

        if source_info is not None:
            references[source_info[0]].append(
                (
                    _stable_token(template_target),
                    _stable_token(data[_EDGE_LABEL]),
                    _stable_token(edge_reference),
                    edge_reference,
                    source_info[1],
                )
            )
        if target_info is not None:
            references[target_info[0]].append(
                (
                    _stable_token(template_source),
                    _stable_token(data[_EDGE_LABEL]),
                    _stable_token(edge_reference),
                    edge_reference,
                    target_info[1],
                )
            )

    raw_signatures: dict[Hashable, tuple] = {}
    canonical_signatures: dict[Hashable, tuple] = {}
    boundary_contexts: dict[Hashable, tuple] = {}
    canonical_port_to_host: dict[Hashable, tuple[Hashable, ...]] = {}
    occurrence_by_supernode = {
        supernode: occurrence
        for supernode, (_, occurrence) in zip(
            supernodes, selected, strict=True
        )
    }

    for supernode in supernodes:
        rule = rule_map[supernode_rule[supernode]]
        grouped: dict[tuple[str, str], list[tuple[Any, ...]]] = defaultdict(list)
        for reference in references[supernode]:
            grouped[(reference[0], reference[1])].append(reference)

        groups: list[tuple[tuple[Any, ...], list[tuple[Any, ...]]]] = []
        contexts: list[tuple[Any, ...]] = []
        for group_key in sorted(grouped):
            group_references = sorted(grouped[group_key], key=lambda item: item[2])
            groups.append((group_key, group_references))
            other_token, edge_token = group_key
            other_node = next(
                (node for node in template.nodes if _stable_token(node) == other_token),
                None,
            )
            if other_node is None:  # pragma: no cover - defensive
                other_kind: Hashable = "unknown"
                other_label: Hashable = "unknown"
            else:
                other_kind = template.nodes[other_node][_NODE_KIND]
                other_label = template.nodes[other_node][_NODE_LABEL]
            contexts.append(
                (other_kind, other_label, edge_token, len(group_references))
            )

        raw_signature = tuple(
            tuple(sorted(reference[-1] for reference in group_references))
            for _, group_references in groups
        )
        canonical_signature, automorphism = _canonical_boundary_signature(
            rule, groups
        )
        raw_signatures[supernode] = raw_signature
        canonical_signatures[supernode] = canonical_signature
        boundary_contexts[supernode] = tuple(contexts)

        occurrence = occurrence_by_supernode[supernode]
        canonical_hosts: list[Hashable | None] = [
            None
        ] * rule.motif.number_of_nodes()
        for raw_port, host_node in enumerate(occurrence.mapping):
            canonical_hosts[automorphism[raw_port]] = host_node
        if any(host is None for host in canonical_hosts):
            raise AssertionError(
                "Canonical motif ports did not map to every host node."
            )
        canonical_port_to_host[supernode] = tuple(
            host for host in canonical_hosts if host is not None
        )

        for _, group_references in groups:
            for reference in group_references:
                edge_source, edge_target, edge_key = reference[3]
                template[edge_source][edge_target][edge_key][_PORTS][supernode] = (
                    automorphism[reference[-1]]
                )

    return Rewrite(
        rules=rules,
        template=template,
        supernodes=tuple(supernodes),
        supernode_rule=supernode_rule,
        selected=selected,
        raw_signatures=raw_signatures,
        canonical_signatures=canonical_signatures,
        boundary_contexts=boundary_contexts,
        canonical_port_to_host=canonical_port_to_host,
        graph_metadata=dict(graph.graph),
    )


def _decode_rewrite_original_ids(rewrite: Rewrite) -> nx.Graph:
    """Decode a rewrite using the normalized host-node IDs.

    Unlike :func:`decode_rewrite`, this internal decoder uses the witness
    retained during contraction.  It is intended for exact linear-time
    validation and not as a replacement for the public isomorphism-invariant
    decoder.
    """
    rule_map = {rule.rank: rule for rule in rewrite.rules}
    decoded = nx.Graph()
    decoded.graph.update(rewrite.graph_metadata)

    for vertex, data in rewrite.template.nodes(data=True):
        if vertex in rewrite.supernodes:
            continue
        node_data = dict(data)
        node_data[_NODE_KIND] = "data"
        decoded.add_node(vertex, **node_data)

    for supernode in rewrite.supernodes:
        rule = rule_map[rewrite.supernode_rule[supernode]]
        host_by_port = rewrite.canonical_port_to_host.get(supernode)
        if host_by_port is None:
            raise ValueError(
                f"Rewrite is missing the host-node witness for {supernode!r}."
            )
        if len(host_by_port) != rule.motif.number_of_nodes():
            raise ValueError(
                f"Host-node witness for {supernode!r} has the wrong size."
            )

        for port, data in rule.motif.nodes(data=True):
            host_node = host_by_port[int(port)]
            decoded.add_node(host_node, **dict(data))
        for source, target, data in rule.motif.edges(data=True):
            decoded.add_edge(
                host_by_port[int(source)],
                host_by_port[int(target)],
                **dict(data),
            )

    for source, target, _, data in rewrite.template.edges(keys=True, data=True):
        ports = data.get(_PORTS, {})
        decoded_source = (
            rewrite.canonical_port_to_host[source][int(ports[source])]
            if source in rewrite.supernodes
            else source
        )
        decoded_target = (
            rewrite.canonical_port_to_host[target][int(ports[target])]
            if target in rewrite.supernodes
            else target
        )
        edge_data = {
            key: value for key, value in data.items() if key != _PORTS
        }
        if decoded.has_edge(decoded_source, decoded_target):
            existing = dict(decoded.edges[decoded_source, decoded_target])
            if existing != edge_data:
                raise ValueError(
                    "Rewrite decoded two conflicting edges between "
                    f"{decoded_source!r} and {decoded_target!r}."
                )
        else:
            decoded.add_edge(decoded_source, decoded_target, **edge_data)

    return decoded


def _undirected_edge_map(graph: nx.Graph) -> dict[frozenset[Hashable], dict[str, Any]]:
    return {
        frozenset((source, target)): dict(data)
        for source, target, data in graph.edges(data=True)
    }


def _rewrite_validation_error(original: nx.Graph, rewrite: Rewrite) -> str | None:
    """Return the first exact reconstruction mismatch, if any."""
    try:
        decoded = _decode_rewrite_original_ids(rewrite)
    except (KeyError, TypeError, ValueError) as exc:
        return f"decoder error: {exc}"

    if original.is_directed() != decoded.is_directed():
        return "graph-class mismatch: directedness differs"
    if original.is_multigraph() != decoded.is_multigraph():
        return "graph-class mismatch: multigraph status differs"
    if dict(original.graph) != dict(decoded.graph):
        return (
            "graph-metadata mismatch: "
            f"expected={dict(original.graph)!r}, got={dict(decoded.graph)!r}"
        )

    original_nodes = set(original.nodes())
    decoded_nodes = set(decoded.nodes())
    missing_nodes = original_nodes - decoded_nodes
    if missing_nodes:
        return f"missing node: {next(iter(missing_nodes))!r}"
    extra_nodes = decoded_nodes - original_nodes
    if extra_nodes:
        return f"extra node: {next(iter(extra_nodes))!r}"
    for node in original.nodes():
        expected = dict(original.nodes[node])
        actual = dict(decoded.nodes[node])
        if expected != actual:
            return (
                f"node-attribute mismatch at {node!r}: "
                f"expected={expected!r}, got={actual!r}"
            )

    original_edges = _undirected_edge_map(original)
    decoded_edges = _undirected_edge_map(decoded)
    missing_edges = original_edges.keys() - decoded_edges.keys()
    if missing_edges:
        return f"missing edge: {tuple(next(iter(missing_edges)))!r}"
    extra_edges = decoded_edges.keys() - original_edges.keys()
    if extra_edges:
        return f"extra edge: {tuple(next(iter(extra_edges)))!r}"
    for edge_key, expected in original_edges.items():
        actual = decoded_edges[edge_key]
        if expected != actual:
            return (
                f"edge-attribute mismatch at {tuple(edge_key)!r}: "
                f"expected={expected!r}, got={actual!r}"
            )
    return None


def _validate_rewrite_exact(original: nx.Graph, rewrite: Rewrite) -> bool:
    """Validate exact reconstruction in O(|V| + |E|) expected time."""
    return _rewrite_validation_error(original, rewrite) is None


def decode_rewrite(rewrite: Rewrite) -> nx.Graph:
    """Decode a rewrite to a graph isomorphic to the normalized input."""
    rule_map = {rule.rank: rule for rule in rewrite.rules}
    decoded = nx.Graph()
    decoded.graph.update(rewrite.graph_metadata)

    for vertex, data in rewrite.template.nodes(data=True):
        if vertex not in rewrite.supernodes:
            decoded.add_node(
                ("outside", _stable_token(vertex)),
                **{
                    _NODE_LABEL: data[_NODE_LABEL],
                    _NODE_KIND: "data",
                },
            )

    for supernode in rewrite.supernodes:
        rule = rule_map[rewrite.supernode_rule[supernode]]
        for port, data in rule.motif.nodes(data=True):
            decoded.add_node(
                (supernode, port),
                **{
                    _NODE_LABEL: data[_NODE_LABEL],
                    _NODE_KIND: "data",
                },
            )
        for source, target, data in rule.motif.edges(data=True):
            decoded.add_edge(
                (supernode, source),
                (supernode, target),
                **{_EDGE_LABEL: data[_EDGE_LABEL]},
            )

    for source, target, _, data in rewrite.template.edges(keys=True, data=True):
        decoded_source = (
            (source, data[_PORTS][source])
            if source in rewrite.supernodes
            else ("outside", _stable_token(source))
        )
        decoded_target = (
            (target, data[_PORTS][target])
            if target in rewrite.supernodes
            else ("outside", _stable_token(target))
        )
        decoded.add_edge(
            decoded_source,
            decoded_target,
            **{_EDGE_LABEL: data[_EDGE_LABEL]},
        )
    return decoded


# ---------------------------------------------------------------------------
# Rewrite codelength
# ---------------------------------------------------------------------------


def _template_incidence_graph(template: nx.MultiGraph) -> nx.Graph:
    incidence = nx.Graph()
    endpoint_nodes: dict[Hashable, Hashable] = {}
    for vertex, data in template.nodes(data=True):
        endpoint = ("vertex", _stable_token(vertex))
        endpoint_nodes[vertex] = endpoint
        incidence.add_node(endpoint, **{_NODE_LABEL: data[_NODE_LABEL]})

    for edge_index, (source, target, _, data) in enumerate(
        template.edges(keys=True, data=True)
    ):
        edge_node = ("edge", edge_index)
        incidence.add_node(
            edge_node,
            **{_NODE_LABEL: (_EDGE_NODE_TAG, data[_EDGE_LABEL])},
        )
        incidence.add_edge(
            edge_node,
            endpoint_nodes[source],
            **{_EDGE_LABEL: _INCIDENCE_LABEL},
        )
        incidence.add_edge(
            edge_node,
            endpoint_nodes[target],
            **{_EDGE_LABEL: _INCIDENCE_LABEL},
        )
    return incidence


def _template_alphabets(
    original: Alphabets, rules: tuple[MotifRule, ...]
) -> Alphabets:
    node_labels: set[Hashable] = set(original.node_labels)
    node_labels.update((_MOTIF_TAG, rule.rank) for rule in rules)
    node_labels.update((_EDGE_NODE_TAG, label) for label in original.edge_labels)
    return Alphabets(
        tuple(sorted(node_labels, key=_stable_token)),
        (_INCIDENCE_LABEL,),
    )


def _flatten_signature(signature: tuple) -> tuple[int, ...]:
    return tuple(port for group in signature for port in group)


def _boundary_bits(rewrite: Rewrite, *, quotient_by_automorphisms: bool) -> float:
    rule_map = {rule.rank: rule for rule in rewrite.rules}
    signatures = (
        rewrite.canonical_signatures
        if quotient_by_automorphisms
        else rewrite.raw_signatures
    )
    grouped_symbols: dict[tuple[Any, ...], list[tuple]] = defaultdict(list)
    for supernode, signature in signatures.items():
        rule_rank = rewrite.supernode_rule[supernode]
        boundary_edges = len(_flatten_signature(signature))
        context_key = (
            rule_rank,
            boundary_edges,
            rewrite.boundary_contexts[supernode],
        )
        grouped_symbols[context_key].append(signature)

    bits = 0.0
    for (rule_rank, boundary_edges, _), symbols in grouped_symbols.items():
        rule = rule_map[rule_rank]
        alphabet_size = (
            rule.ordered_port_orbit_count(boundary_edges)
            if quotient_by_automorphisms
            else rule.motif.number_of_nodes() ** boundary_edges
        )
        bits += _dirichlet_multinomial_bits(
            Counter(symbols), max(alphabet_size, 1)
        )
    return bits


def _dictionary_bits(rules: tuple[MotifRule, ...], alphabets: Alphabets) -> float:
    if not rules:
        return 0.0
    return _positive_integer_bits(len(rules)) + sum(
        _base_bits(rule.motif, alphabets, complete=True) for rule in rules
    )


# ---------------------------------------------------------------------------
# Dataset selector and compressor
# ---------------------------------------------------------------------------


Selector = Literal["sparse", "per_graph", "all_eligible"]
DictionarySelection = Literal["best", "best_nonempty", "fixed"]


def _apply_selector(
    records: list[EncodedGraph],
    *,
    selector: Selector,
    dictionary_bits: float,
    model_choice_bits: float,
    require_nonempty_rewrite: bool = False,
) -> tuple[DatasetCode, pd.DataFrame]:
    n_graphs = len(records)
    baseline_total = float(sum(record.baseline_bits for record in records))
    eligible = [record for record in records if record.rewrite_bits is not None]

    if selector == "sparse":
        ordered = sorted(
            eligible,
            key=lambda record: (-record.gross_gain_bits, record.graph_index),
        )
        cumulative = np.concatenate(
            ([0.0], np.cumsum([record.gross_gain_bits for record in ordered]))
        )
        rows = [
            {
                "k": 0,
                "gross_rewrite_savings": 0.0,
                "selector_bits": 0.0,
                "dictionary_bits_charged": 0.0,
                "model_choice_bits": model_choice_bits,
                "net_savings": -model_choice_bits,
            }
        ]
        for k in range(1, len(ordered) + 1):
            selector_bits = _positive_integer_bits(k) + _log2_choose(n_graphs, k)
            rows.append(
                {
                    "k": k,
                    "gross_rewrite_savings": float(cumulative[k]),
                    "selector_bits": selector_bits,
                    "dictionary_bits_charged": dictionary_bits,
                    "model_choice_bits": model_choice_bits,
                    "net_savings": float(
                        cumulative[k]
                        - dictionary_bits
                        - selector_bits
                        - model_choice_bits
                    ),
                }
            )
        curve = pd.DataFrame(rows)
        selection_curve = curve
        if require_nonempty_rewrite and len(curve) > 1:
            selection_curve = curve.loc[curve["k"] > 0]
        best = selection_curve.loc[selection_curve["net_savings"].idxmax()]

        if require_nonempty_rewrite and len(curve) == 1:
            best = best.copy()
            best["dictionary_bits_charged"] = dictionary_bits
            best["net_savings"] = -dictionary_bits - model_choice_bits
            curve.loc[0, "dictionary_bits_charged"] = dictionary_bits
            curve.loc[0, "net_savings"] = best["net_savings"]
        selected_ids = {
            record.graph_index for record in ordered[: int(best["k"])]
        }
        for record in records:
            record.use_rewrite = record.graph_index in selected_ids
        selector_bits = float(best["selector_bits"])
        charged_dictionary = float(best["dictionary_bits_charged"])
        net_savings = float(best["net_savings"])

    elif selector == "per_graph":
        selected_ids = {
            record.graph_index
            for record in eligible
            if record.gross_gain_bits > 0.0
        }
        if require_nonempty_rewrite and eligible and not selected_ids:
            best_record = max(
                eligible,
                key=lambda record: (record.gross_gain_bits, -record.graph_index),
            )
            selected_ids = {best_record.graph_index}
        for record in records:
            record.use_rewrite = record.graph_index in selected_ids
        selector_bits = float(n_graphs)  # one escape/rewrite flag per graph
        charged_dictionary = (
            dictionary_bits
            if selected_ids or require_nonempty_rewrite
            else 0.0
        )
        gross = sum(
            record.gross_gain_bits for record in records if record.use_rewrite
        )
        net_savings = gross - selector_bits - charged_dictionary - model_choice_bits
        curve = pd.DataFrame(
            [
                {
                    "k": len(selected_ids),
                    "gross_rewrite_savings": gross,
                    "selector_bits": selector_bits,
                    "dictionary_bits_charged": charged_dictionary,
                    "model_choice_bits": model_choice_bits,
                    "net_savings": net_savings,
                }
            ]
        )

    elif selector == "all_eligible":
        selected_ids = {record.graph_index for record in eligible}
        for record in records:
            record.use_rewrite = record.graph_index in selected_ids
        selector_bits = 0.0
        charged_dictionary = (
            dictionary_bits
            if selected_ids or require_nonempty_rewrite
            else 0.0
        )
        gross = sum(record.gross_gain_bits for record in eligible)
        net_savings = gross - charged_dictionary - model_choice_bits
        curve = pd.DataFrame(
            [
                {
                    "k": len(selected_ids),
                    "gross_rewrite_savings": gross,
                    "selector_bits": selector_bits,
                    "dictionary_bits_charged": charged_dictionary,
                    "model_choice_bits": model_choice_bits,
                    "net_savings": net_savings,
                }
            ]
        )
    else:  # pragma: no cover - Literal plus constructor validation
        raise ValueError(f"Unknown selector {selector!r}.")

    encoded_bits = baseline_total - net_savings
    n_rewritten = sum(record.use_rewrite for record in records)
    n_occurrences = sum(
        record.selected_occurrences for record in records if record.use_rewrite
    )
    report = DatasetCode(
        selector=selector,
        n_graphs=n_graphs,
        n_eligible=len(eligible),
        n_rewritten=n_rewritten,
        n_occurrences=n_occurrences,
        baseline_bits=baseline_total,
        dictionary_bits=charged_dictionary,
        selector_bits=selector_bits,
        model_choice_bits=model_choice_bits,
        encoded_bits=encoded_bits,
        net_savings_bits=net_savings,
        bits_per_graph=(net_savings / n_graphs if n_graphs else 0.0),
        fraction_rewritten=(n_rewritten / n_graphs if n_graphs else 0.0),
    )
    return report, curve


def _selected_rewrite_components(
    records: list[EncodedGraph],
) -> dict[str, float]:
    selected = [record for record in records if record.use_rewrite]
    return {
        "gross_rewrite_savings_bits": float(
            sum(record.gross_gain_bits for record in selected)
        ),
        "template_bits": float(
            sum(record.template_bits or 0.0 for record in selected)
        ),
        "boundary_bits": float(
            sum(record.boundary_bits or 0.0 for record in selected)
        ),
        "rewrite_bits": float(
            sum(record.rewrite_bits or 0.0 for record in selected)
        ),
    }


def _records_frame(records: list[EncodedGraph]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "graph_index": record.graph_index,
                "use_rewrite": record.use_rewrite,
                "original_nodes": record.baseline_graph.number_of_nodes(),
                "original_edges": record.baseline_graph.number_of_edges(),
                "template_nodes": (
                    record.rewrite.template.number_of_nodes()
                    if record.rewrite is not None
                    else record.baseline_graph.number_of_nodes()
                ),
                "template_edges": (
                    record.rewrite.template.number_of_edges()
                    if record.rewrite is not None
                    else record.baseline_graph.number_of_edges()
                ),
                "node_reduction_fraction": (
                    1.0
                    - (
                        record.rewrite.template.number_of_nodes()
                        / max(record.baseline_graph.number_of_nodes(), 1)
                    )
                    if record.rewrite is not None
                    else 0.0
                ),
                "edge_reduction_fraction": (
                    1.0
                    - (
                        record.rewrite.template.number_of_edges()
                        / max(record.baseline_graph.number_of_edges(), 1)
                    )
                    if record.rewrite is not None
                    else 0.0
                ),
                "baseline_bits": record.baseline_bits,
                "template_bits": record.template_bits,
                "boundary_bits": record.boundary_bits,
                "rewrite_bits": record.rewrite_bits,
                "gross_gain_bits": record.gross_gain_bits,
                "candidate_occurrences": record.candidate_occurrences,
                "selected_occurrences": record.selected_occurrences,
                "rules_used": record.rules_used,
                "total_graphlet_occurrences": record.total_graphlet_occurrences,
                "unseen_graphlet_occurrences": record.unseen_graphlet_occurrences,
                "unseen_graphlet_fraction": record.unseen_graphlet_fraction,
            }
            for record in records
        ]
    )


class MDLGraphCompressor:
    """Learn and apply a lossless graphlet MDL dictionary.

    The estimator follows a train/transform contract so dictionary discovery is
    performed only on training graphs.  Candidate graphlets are ranked by their
    single-rule MDL gain, and the best joint prefix of at most ``n_rules`` is
    retained.  The corpus selector is then applied independently to each
    transformed dataset.

    Parameters
    ----------
    graphlet_sizes:
        Connected induced graphlet sizes considered by the enumerator.
    n_rules:
        Maximum dictionary size.
    min_graph_support, min_occurrences:
        Candidate filters on the fit corpus.
    max_candidates:
        Maximum number of support-ranked candidates that receive the expensive
        exact MDL evaluation.
    node_label_keys, edge_label_keys:
        Attributes preserved by the code.  ``None`` means topology-only.
    selector:
        ``"sparse"`` uses the corpus-level enumerative selector from the final
        notebooks; ``"per_graph"`` uses one flag per graph; ``"all_eligible"``
        rewrites every graph containing a selected motif.
    model_choice_bits:
        Optional cost for choosing baseline-only versus rewrite-capable corpus
        code.
    min_rule_savings_bits:
        Minimum single-rule training gain before a candidate is allowed into the
        joint-prefix search. Use ``-math.inf`` for diagnostic experiments.
    dictionary_selection:
        ``"best"`` selects the best prefix including the empty dictionary.
        ``"best_nonempty"`` selects the best non-empty prefix when any candidate
        passes the threshold. ``"fixed"`` selects the longest evaluated prefix.
        The latter two modes are useful when a scientifically informative negative
        MDL result should still produce compressed graphs for downstream analysis.
    enumerator:
        Graphlet occurrence backend.  Defaults to Buhito BFS.
    cache_dir:
        Optional directory for corpus-specific occurrence caches.
    validate:
        Decode and check every candidate rewrite during fit/transform.
    progress:
        Print coarse progress messages.
    """

    def __init__(
        self,
        *,
        graphlet_sizes: Sequence[int] = (3,),
        n_rules: int = 5,
        min_graph_support: int = 2,
        min_occurrences: int = 2,
        max_candidates: int = 100,
        node_label_keys: str | Sequence[str] | None = None,
        edge_label_keys: str | Sequence[str] | None = None,
        selector: Selector = "sparse",
        model_choice_bits: float = 1.0,
        min_rule_savings_bits: float = 0.0,
        dictionary_selection: DictionarySelection = "best",
        enumerator: GraphletEnumerator | None = None,
        cache_dir: str | Path | None = None,
        validate: bool = True,
        progress: bool = False,
    ) -> None:
        sizes = tuple(sorted({int(size) for size in graphlet_sizes}))
        if not sizes or min(sizes) < 2:
            raise ValueError("graphlet_sizes must contain integers >= 2.")
        if n_rules < 0 or max_candidates < 1:
            raise ValueError("n_rules must be >= 0 and max_candidates must be >= 1.")
        if selector not in {"sparse", "per_graph", "all_eligible"}:
            raise ValueError(f"Unknown selector {selector!r}.")
        if dictionary_selection not in {"best", "best_nonempty", "fixed"}:
            raise ValueError(
                f"Unknown dictionary_selection {dictionary_selection!r}."
            )

        self.graphlet_sizes = sizes
        self.n_rules = int(n_rules)
        self.min_graph_support = int(min_graph_support)
        self.min_occurrences = int(min_occurrences)
        self.max_candidates = int(max_candidates)
        self.schema = GraphSchema.from_keys(
            node_label_keys=node_label_keys,
            edge_label_keys=edge_label_keys,
        )
        self.selector = selector
        self.model_choice_bits = float(model_choice_bits)
        self.min_rule_savings_bits = float(min_rule_savings_bits)
        self.dictionary_selection = dictionary_selection
        self.enumerator = enumerator or BuhitoGraphletEnumerator()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.validate = bool(validate)
        self.progress = bool(progress)

        self.rules_: tuple[MotifRule, ...] | None = None
        self.alphabets_: Alphabets | None = None
        self.seen_base_keys_: set[Hashable] | None = None
        self.candidate_table_: pd.DataFrame | None = None
        self.candidate_motifs_: dict[str, nx.Graph] | None = None
        self.dictionary_path_: pd.DataFrame | None = None
        self.training_result_: CompressionResult | None = None

    def _check_fitted(self) -> None:
        if self.rules_ is None or self.alphabets_ is None:
            raise RuntimeError("Call fit before transform.")

    def _normalize_many(self, graphs: Iterable[nx.Graph]) -> list[nx.Graph]:
        return [self.schema.normalize(graph) for graph in graphs]

    def _corpus_digest(self, graphs: list[nx.Graph]) -> str:
        digest = hashlib.sha256()
        digest.update(repr(self.graphlet_sizes).encode("utf-8"))
        digest.update(getattr(self.enumerator, "name", type(self.enumerator).__name__).encode("utf-8"))
        for graph in graphs:
            node_rows = tuple(
                (node, _stable_token(data[_NODE_LABEL]))
                for node, data in sorted(graph.nodes(data=True))
            )
            edge_rows = tuple(
                sorted(
                    (
                        min(source, target),
                        max(source, target),
                        _stable_token(data[_EDGE_LABEL]),
                    )
                    for source, target, data in graph.edges(data=True)
                )
            )
            digest.update(repr((node_rows, edge_rows)).encode("utf-8"))
        return digest.hexdigest()

    def _enumerate_many(
        self, graphs: list[nx.Graph]
    ) -> list[dict[Hashable, list[frozenset[int]]]]:
        cache_path: Path | None = None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / f"mdl_occurrences_{self._corpus_digest(graphs)}.pkl.gz"
            if cache_path.exists():
                if self.progress:
                    print(f"Loading MDL occurrence cache: {cache_path}")
                with gzip.open(cache_path, "rb") as handle:
                    return pickle.load(handle)

        raw: list[dict[Hashable, list[frozenset[int]]]] = []
        for index, graph in enumerate(graphs):
            if self.progress and (index == 0 or index % 100 == 0):
                print(f"Enumerating graphlets {index}/{len(graphs)}")
            raw.append(self.enumerator.enumerate(graph, self.graphlet_sizes))

        if cache_path is not None:
            with gzip.open(cache_path, "wb") as handle:
                pickle.dump(raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if self.progress:
                print(f"Saved MDL occurrence cache: {cache_path}")
        return raw

    def _support_table(
        self,
        candidates: dict[str, _Candidate],
        found_list: list[dict[str, _FoundRecord]],
    ) -> pd.DataFrame:
        graph_support: Counter[str] = Counter()
        occurrences: Counter[str] = Counter()
        for found in found_list:
            for key, record in found.items():
                graph_support[key] += 1
                occurrences[key] += len(record.instances)

        rows: list[dict[str, Any]] = []
        for key, candidate in candidates.items():
            temporary_rule = MotifRule(
                key=key,
                base_key=candidate.base_key,
                motif=candidate.motif,
                rank=0,
            )
            rows.append(
                {
                    "key": key,
                    "base_key": candidate.base_key,
                    "graph_support": graph_support[key],
                    "occurrences": occurrences[key],
                    "motif_nodes": candidate.motif.number_of_nodes(),
                    "motif_edges": candidate.motif.number_of_edges(),
                    "automorphism_order": temporary_rule.automorphism_order,
                    "orbit_sizes": temporary_rule.orbit_sizes,
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "key",
                    "base_key",
                    "graph_support",
                    "occurrences",
                    "motif_nodes",
                    "motif_edges",
                    "automorphism_order",
                    "orbit_sizes",
                ]
            )
        return (
            pd.DataFrame(rows)
            .sort_values(
                ["graph_support", "occurrences", "key"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )

    @staticmethod
    def _occurrences_from_found(
        found_list: list[dict[str, _FoundRecord]],
        rules: tuple[MotifRule, ...],
    ) -> list[dict[int, list[Occurrence]]]:
        result: list[dict[int, list[Occurrence]]] = []
        for found in found_list:
            per_rule: dict[int, list[Occurrence]] = {}
            for rule in rules:
                record = found.get(rule.key)
                per_rule[rule.rank] = list(record.instances) if record else []
            result.append(per_rule)
        return result

    def _build_records(
        self,
        graphs: list[nx.Graph],
        raw_occurrences: list[dict[Hashable, list[frozenset[int]]]],
        rules: tuple[MotifRule, ...],
        occurrences_by_graph: list[dict[int, list[Occurrence]]],
        alphabets: Alphabets,
        seen_base_keys: set[Hashable],
    ) -> list[EncodedGraph]:
        template_alphabets = _template_alphabets(alphabets, rules)
        records: list[EncodedGraph] = []

        for graph_index, (graph, raw, occurrences_by_rule) in enumerate(
            zip(graphs, raw_occurrences, occurrences_by_graph)
        ):
            baseline_bits = _base_bits(graph, alphabets, complete=True)
            total_graphlet_occurrences = sum(len(value) for value in raw.values())
            unseen_graphlet_occurrences = sum(
                len(value)
                for key, value in raw.items()
                if key not in seen_base_keys
            )
            candidate_occurrences = sum(
                len(occurrences) for occurrences in occurrences_by_rule.values()
            )

            rewrite: Rewrite | None = None
            template_bits: float | None = None
            boundary_bits: float | None = None
            rewrite_bits: float | None = None
            selected_occurrences = 0
            rules_used = 0

            if candidate_occurrences:
                rewrite = _rewrite_graph(graph, rules, occurrences_by_rule)
                if rewrite.selected:
                    if self.validate:
                        validation_error = _rewrite_validation_error(
                            graph, rewrite
                        )
                        if validation_error is not None:
                            raise AssertionError(
                                "MDL rewrite failed reconstruction on graph "
                                f"{graph_index}: {validation_error}."
                            )
                    incidence = _template_incidence_graph(rewrite.template)
                    template_bits = _base_bits(
                        incidence, template_alphabets, complete=True
                    )
                    boundary_bits = _boundary_bits(
                        rewrite, quotient_by_automorphisms=True
                    )
                    rewrite_bits = template_bits + boundary_bits
                    selected_occurrences = len(rewrite.selected)
                    rules_used = len({rank for rank, _ in rewrite.selected})
                else:
                    rewrite = None

            records.append(
                EncodedGraph(
                    graph_index=graph_index,
                    baseline_graph=graph,
                    rewrite=rewrite,
                    use_rewrite=False,
                    baseline_bits=baseline_bits,
                    template_bits=template_bits,
                    boundary_bits=boundary_bits,
                    rewrite_bits=rewrite_bits,
                    candidate_occurrences=candidate_occurrences,
                    selected_occurrences=selected_occurrences,
                    rules_used=rules_used,
                    total_graphlet_occurrences=total_graphlet_occurrences,
                    unseen_graphlet_occurrences=unseen_graphlet_occurrences,
                )
            )
        return records

    def _evaluate_normalized(
        self,
        graphs: list[nx.Graph],
        raw_occurrences: list[dict[Hashable, list[frozenset[int]]]],
        rules: tuple[MotifRule, ...],
        occurrences_by_graph: list[dict[int, list[Occurrence]]],
        alphabets: Alphabets,
        seen_base_keys: set[Hashable],
        *,
        require_nonempty_rewrite: bool = False,
    ) -> CompressionResult:
        records = self._build_records(
            graphs,
            raw_occurrences,
            rules,
            occurrences_by_graph,
            alphabets,
            seen_base_keys,
        )
        dictionary_bits = _dictionary_bits(rules, alphabets)
        report, selector_curve = _apply_selector(
            records,
            selector=self.selector,
            dictionary_bits=dictionary_bits,
            model_choice_bits=self.model_choice_bits,
            require_nonempty_rewrite=require_nonempty_rewrite,
        )
        return CompressionResult(
            schema=self.schema,
            rules=rules,
            records=records,
            report=report,
            per_graph=_records_frame(records),
            selector_curve=selector_curve,
        )

    def fit(self, graphs: Iterable[nx.Graph]) -> "MDLGraphCompressor":
        normalized = self._normalize_many(graphs)
        if not normalized:
            raise ValueError("fit requires at least one graph.")
        alphabets = _build_alphabets(normalized)
        raw = self._enumerate_many(normalized)
        candidates, found_list = _discover_candidates(normalized, raw)
        support = self._support_table(candidates, found_list)
        filtered = support[
            (support["graph_support"] >= self.min_graph_support)
            & (support["occurrences"] >= self.min_occurrences)
        ].head(self.max_candidates)

        seen_base_keys = {key for graph_raw in raw for key in graph_raw}
        candidate_rows: list[dict[str, Any]] = []
        for position, row in enumerate(filtered.itertuples(index=False), start=1):
            if self.progress:
                print(f"Scoring candidate {position}/{len(filtered)}: {row.key}")
            candidate = candidates[row.key]
            rule = MotifRule(
                key=candidate.key,
                base_key=candidate.base_key,
                motif=candidate.motif.copy(),
                rank=0,
            )
            occurrences = self._occurrences_from_found(found_list, (rule,))
            forced_result = self._evaluate_normalized(
                normalized,
                raw,
                (rule,),
                occurrences,
                alphabets,
                seen_base_keys,
                require_nonempty_rewrite=True,
            )

            forced_components = _selected_rewrite_components(
                forced_result.records
            )
            best_model_report, _ = _apply_selector(
                forced_result.records,
                selector=self.selector,
                dictionary_bits=_dictionary_bits((rule,), alphabets),
                model_choice_bits=self.model_choice_bits,
            )
            candidate_rows.append(
                {
                    **row._asdict(),
                    "total_occurrences": row.occurrences,
                    **{
                        f"forced_{name}": value
                        for name, value in asdict(forced_result.report).items()
                    },
                    **{
                        f"forced_{name}": value
                        for name, value in forced_components.items()
                    },
                    **{
                        f"best_model_{name}": value
                        for name, value in asdict(best_model_report).items()
                    },
                    # Backwards-compatible alias. New analysis code should use
                    # ``forced_net_savings_bits`` explicitly.
                    "single_rule_net_savings_bits": (
                        forced_result.report.net_savings_bits
                    ),
                }
            )

        candidate_table = pd.DataFrame(candidate_rows)
        if not candidate_table.empty:
            candidate_table = candidate_table.sort_values(
                ["forced_net_savings_bits", "graph_support", "occurrences"],
                ascending=[False, False, False],
            ).reset_index(drop=True)
            candidate_table.insert(
                0,
                "rank",
                np.arange(1, len(candidate_table) + 1, dtype=int),
            )
            candidate_table.insert(
                1,
                "motif_id",
                [f"M{rank:03d}" for rank in candidate_table["rank"]],
            )
            candidate_table["passes_min_rule_savings"] = (
                candidate_table["forced_net_savings_bits"]
                >= self.min_rule_savings_bits
            )

        ranked_keys: list[str] = []
        allowed_keys: list[str] = []
        if not candidate_table.empty and self.n_rules:
            ranked_keys = candidate_table["key"].head(self.n_rules).tolist()
            allowed_keys = candidate_table.loc[
                candidate_table["passes_min_rule_savings"],
                "key",
            ].head(self.n_rules).tolist()

        path_rows: list[dict[str, Any]] = []
        prefix_results: list[tuple[tuple[MotifRule, ...], CompressionResult]] = []

        # Empty dictionary is a valid and important null result.
        empty_result = self._evaluate_normalized(
            normalized,
            raw,
            (),
            [dict() for _ in normalized],
            alphabets,
            seen_base_keys,
        )
        prefix_results.append(((), empty_result))
        path_rows.append(
            {
                "n_rules": 0,
                "keys": (),
                "rule_keys": (),
                "is_empty": True,
                "eligible_for_selection": True,
                "dictionary_selection": self.dictionary_selection,
                **_selected_rewrite_components(empty_result.records),
                **asdict(empty_result.report),
            }
        )

        for prefix_size in range(1, len(ranked_keys) + 1):
            keys = ranked_keys[:prefix_size]
            rules = tuple(
                MotifRule(
                    key=key,
                    base_key=candidates[key].base_key,
                    motif=candidates[key].motif.copy(),
                    rank=rank,
                )
                for rank, key in enumerate(keys)
            )
            occurrences = self._occurrences_from_found(found_list, rules)
            result = self._evaluate_normalized(
                normalized,
                raw,
                rules,
                occurrences,
                alphabets,
                seen_base_keys,
                require_nonempty_rewrite=True,
            )
            prefix_results.append((rules, result))
            path_rows.append(
                {
                    "n_rules": prefix_size,
                    "keys": tuple(keys),
                    "rule_keys": tuple(keys),
                    "is_empty": False,
                    "eligible_for_selection": prefix_size <= len(allowed_keys),
                    "dictionary_selection": self.dictionary_selection,
                    **_selected_rewrite_components(result.records),
                    **asdict(result.report),
                }
            )

        eligible_prefix_results = prefix_results[: len(allowed_keys) + 1]
        if self.dictionary_selection == "best":
            selection_pool = eligible_prefix_results
            best_rules, best_result = max(
                selection_pool,
                key=lambda item: item[1].report.net_savings_bits,
            )
        elif self.dictionary_selection == "best_nonempty":
            selection_pool = [
                item for item in eligible_prefix_results if item[0]
            ]
            if selection_pool:
                best_rules, best_result = max(
                    selection_pool,
                    key=lambda item: item[1].report.net_savings_bits,
                )
            else:
                best_rules, best_result = prefix_results[0]
        else:  # fixed
            best_rules, best_result = eligible_prefix_results[-1]

        selected_keys = tuple(rule.key for rule in best_rules)
        if not candidate_table.empty:
            candidate_table["is_selected"] = candidate_table["key"].isin(
                selected_keys
            )
        for path_row in path_rows:
            path_row["is_best"] = tuple(path_row["keys"]) == selected_keys

        self.rules_ = best_rules
        self.alphabets_ = alphabets
        self.seen_base_keys_ = seen_base_keys
        self.candidate_table_ = candidate_table
        self.candidate_motifs_ = {
            key: candidates[key].motif.copy()
            for key in candidate_table.get("key", pd.Series(dtype=str)).tolist()
        }
        self.dictionary_path_ = pd.DataFrame(path_rows)
        self.training_result_ = best_result
        return self

    def transform(self, graphs: Iterable[nx.Graph]) -> CompressionResult:
        self._check_fitted()
        assert self.rules_ is not None
        assert self.alphabets_ is not None
        assert self.seen_base_keys_ is not None

        normalized = self._normalize_many(graphs)
        raw = self._enumerate_many(normalized)
        occurrences_by_graph = [
            _find_frozen_occurrences(graph, graph_raw, self.rules_)
            for graph, graph_raw in zip(normalized, raw)
        ]
        return self._evaluate_normalized(
            normalized,
            raw,
            self.rules_,
            occurrences_by_graph,
            self.alphabets_,
            self.seen_base_keys_,
        )

    def rule_prefix(self, n_rules: int) -> tuple[MotifRule, ...]:
        """Return the first ``n_rules`` scored motifs as a fresh rule prefix.

        The prefix order is the deterministic forced-MDL candidate ranking used
        during :meth:`fit`.  Returned rules and motif graphs are independent
        copies, so callers may safely use them for diagnostic prefix sweeps
        without mutating the fitted estimator or its selected dictionary.
        """

        self._check_fitted()
        if n_rules < 0:
            raise ValueError("n_rules must be nonnegative.")
        assert self.candidate_table_ is not None
        assert self.candidate_motifs_ is not None
        available = len(self.candidate_table_)
        if n_rules > available:
            raise ValueError(
                f"Requested {n_rules} rules, but only {available} scored "
                "candidates are available."
            )
        if n_rules == 0:
            return ()

        ranked = (
            self.candidate_table_
            .sort_values("rank")
            .head(n_rules)
        )
        return tuple(
            MotifRule(
                key=str(row.key),
                base_key=row.base_key,
                motif=self.candidate_motifs_[str(row.key)].copy(),
                rank=rank,
            )
            for rank, row in enumerate(ranked.itertuples(index=False))
        )

    def transform_rule_prefix(
        self,
        graphs: Iterable[nx.Graph],
        n_rules: int,
        *,
        require_nonempty_rewrite: bool = False,
    ) -> CompressionResult:
        """Transform graphs with a ranked dictionary prefix.

        This method is intended for controlled compression--speed--quality
        sweeps.  It reuses the fitted alphabets, candidate ranking, and
        occurrence cache while leaving :attr:`rules_` unchanged.

        Parameters
        ----------
        graphs:
            Graphs to transform.
        n_rules:
            Number of top-ranked candidate rules in the diagnostic prefix.
            Zero is the identity/baseline representation.
        require_nonempty_rewrite:
            When true, the corpus selector must encode at least one available
            rewrite.  This reports the actual forced-representation MDL cost
            rather than silently falling back to the empty model.
        """

        self._check_fitted()
        assert self.alphabets_ is not None
        assert self.seen_base_keys_ is not None
        rules = self.rule_prefix(n_rules)
        normalized = self._normalize_many(graphs)
        raw = self._enumerate_many(normalized)
        occurrences_by_graph = [
            _find_frozen_occurrences(graph, graph_raw, rules)
            for graph, graph_raw in zip(normalized, raw)
        ]
        return self._evaluate_normalized(
            normalized,
            raw,
            rules,
            occurrences_by_graph,
            self.alphabets_,
            self.seen_base_keys_,
            require_nonempty_rewrite=require_nonempty_rewrite,
        )

    def fit_transform(self, graphs: Iterable[nx.Graph]) -> CompressionResult:
        graph_list = list(graphs)
        self.fit(graph_list)
        assert self.training_result_ is not None
        return self.training_result_

    def dictionary_frame(self) -> pd.DataFrame:
        self._check_fitted()
        assert self.rules_ is not None
        candidate_ids: dict[str, str] = {}
        if self.candidate_table_ is not None and not self.candidate_table_.empty:
            candidate_ids = dict(
                zip(
                    self.candidate_table_["key"].astype(str),
                    self.candidate_table_["motif_id"].astype(str),
                    strict=True,
                )
            )
        return pd.DataFrame(
            [
                {
                    "rank": rule.rank,
                    "motif_id": candidate_ids.get(rule.key),
                    "key": rule.key,
                    "base_key": rule.base_key,
                    "motif_nodes": rule.motif.number_of_nodes(),
                    "motif_edges": rule.motif.number_of_edges(),
                    "automorphism_order": rule.automorphism_order,
                    "orbit_sizes": rule.orbit_sizes,
                }
                for rule in self.rules_
            ]
        )

    def candidate_frame(self) -> pd.DataFrame:
        """Return exact forced and best-model accounting for scored motifs."""

        self._check_fitted()
        assert self.candidate_table_ is not None
        return self.candidate_table_.copy()

    def candidate_motif_graphs(
        self,
        *,
        restored: bool = True,
    ) -> dict[str, nx.Graph]:
        """Return copies of every scored candidate motif keyed by candidate key.

        Parameters
        ----------
        restored:
            When true, expose the user-facing node and edge attributes selected
            by :class:`GraphSchema`. When false, retain Buhito's normalized
            internal labels. Returned graphs are always independent copies.
        """

        self._check_fitted()
        assert self.candidate_motifs_ is not None
        if restored:
            return {
                key: self.schema.restore(motif)
                for key, motif in self.candidate_motifs_.items()
            }
        return {
            key: motif.copy()
            for key, motif in self.candidate_motifs_.items()
        }

    def candidate_motif(
        self,
        key: str,
        *,
        restored: bool = True,
    ) -> nx.Graph:
        """Return one scored candidate motif as an independent graph copy."""

        motifs = self.candidate_motif_graphs(restored=restored)
        try:
            return motifs[str(key)]
        except KeyError as exc:
            available = ", ".join(motifs) or "<none>"
            raise KeyError(
                f"Unknown motif key {key!r}. Available keys: {available}"
            ) from exc

    def dictionary_path_frame(self) -> pd.DataFrame:
        """Return every evaluated dictionary prefix, including the null model."""

        self._check_fitted()
        assert self.dictionary_path_ is not None
        return self.dictionary_path_.copy()

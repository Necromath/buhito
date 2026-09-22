"""Shared schemas, value objects, and coding primitives."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from math import lgamma
from typing import Any

import networkx as nx
import pandas as pd

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
    ) -> GraphSchema:
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
            from .substitution import decode_rewrite

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
# MDL primitives
# ---------------------------------------------------------------------------


def _assert_supported_graph(graph: nx.Graph) -> None:
    if graph.is_directed():
        raise TypeError("MDLGraphCompressor currently supports undirected graphs only.")
    if graph.is_multigraph():
        raise TypeError(
            "Input graphs must be simple; the rewrite template may be a MultiGraph."
        )
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

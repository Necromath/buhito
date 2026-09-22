"""Candidate graphlet recognition and exact occurrence alignment."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass
from itertools import combinations
from typing import Protocol

import networkx as nx

from .models import (
    _EDGE_LABEL,
    _NODE_LABEL,
    MotifRule,
    Occurrence,
    _assert_internal_labels,
    _stable_token,
)

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
                        raise AssertionError(
                            "New motif failed to match its source occurrence."
                        )

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

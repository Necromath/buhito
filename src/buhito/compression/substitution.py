"""Reversible motif substitution, ports, and reconstruction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from typing import Any

import networkx as nx

from .models import (
    _EDGE_LABEL,
    _MOTIF_TAG,
    _NODE_KIND,
    _NODE_LABEL,
    _PORTS,
    MotifRule,
    Occurrence,
    Rewrite,
    _assert_internal_labels,
    _stable_token,
)
from .models import (
    labeled_isomorphic as labeled_isomorphic,
)

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

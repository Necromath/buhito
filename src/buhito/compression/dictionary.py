"""MDL accounting, corpus selectors, and dictionary scoring helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable
from typing import Any, Literal

import networkx as nx
import numpy as np
import pandas as pd

from .models import (
    _EDGE_LABEL,
    _EDGE_NODE_TAG,
    _INCIDENCE_LABEL,
    _MOTIF_TAG,
    _NODE_LABEL,
    Alphabets,
    DatasetCode,
    EncodedGraph,
    MotifRule,
    Rewrite,
    _base_bits,
    _dirichlet_multinomial_bits,
    _log2_choose,
    _positive_integer_bits,
    _stable_token,
)

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

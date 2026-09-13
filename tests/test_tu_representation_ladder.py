from __future__ import annotations

import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "research"
    / "paper_2026"
    / "tu_representation_ladder.py"
)
SPEC = importlib.util.spec_from_file_location("tu_representation_ladder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
LADDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LADDER)


def _carrier() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, __buhito_mdl_node_label__=("data", (6,)))
    graph.add_node(
        1,
        __buhito_mdl_node_label__=("motif", 2),
        __buhito_mdl_kind__="motif",
    )
    graph.add_edge(0, 1, __buhito_mdl_edge_label__=("data", (1,)))
    return graph


def test_union_schema_gates_motif_and_port_bundles():
    graph = _carrier()
    vocabulary = LADDER.build_vocabulary([graph.subgraph([0]).copy()])
    names = LADDER.feature_names(vocabulary, maximum_rule_count=5, maximum_ports=3)
    ports = {(2, 1): 4}
    bare = LADDER.graph_features(
        graph, vocabulary, expose_motif=False, expose_ports=False,
        port_histogram=ports, maximum_rule_count=5, maximum_ports=3,
    )
    motif = LADDER.graph_features(
        graph, vocabulary, expose_motif=True, expose_ports=False,
        port_histogram=ports, maximum_rule_count=5, maximum_ports=3,
    )
    full = LADDER.graph_features(
        graph, vocabulary, expose_motif=True, expose_ports=True,
        port_histogram=ports, maximum_rule_count=5, maximum_ports=3,
    )
    assert bare.shape == motif.shape == full.shape == (len(names),)
    assert bare[names.index("motif_rank::2")] == 0
    assert motif[names.index("motif_rank::2")] == 1
    assert motif[names.index("port_rank::2::port::1")] == 0
    assert full[names.index("port_rank::2::port::1")] == 4
    structural = [names.index(name) for name in ("nodes", "edges", "degree_mean")]
    np.testing.assert_array_equal(bare[structural], full[structural])


class _Rewrite:
    def __init__(self):
        self.template = nx.MultiGraph()
        self.supernode_rule = {"left": 1, "right": 3}
        self.template.add_edge(
            "left", "outside", __buhito_mdl_ports__={"left": 2}
        )
        self.template.add_edge(
            "left", "right", __buhito_mdl_ports__={"left": 0, "right": 1}
        )


def test_port_histogram_counts_boundary_incidences():
    counts, unknown = LADDER.port_histogram(
        _Rewrite(), maximum_rule_count=5, maximum_ports=3
    )
    assert counts == {(1, 2): 1, (1, 0): 1, (3, 1): 1}
    assert unknown == 0


def test_feature_names_are_unique():
    vocabulary = {"node_labels": ["C", "N"], "edge_labels": ["single"]}
    names = LADDER.feature_names(vocabulary, maximum_rule_count=5, maximum_ports=3)
    assert len(names) == len(set(names))

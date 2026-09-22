import networkx as nx

import buhito
import buhito.mdl
from buhito.compression import ExhaustiveGraphletEnumerator, MDLGraphCompressor


def _triangle_corpus() -> list[nx.Graph]:
    graphs = []
    for repeats in (3, 4, 5):
        graph = nx.disjoint_union_all([nx.cycle_graph(3) for _ in range(repeats)])
        nx.set_node_attributes(graph, "C", "atom")
        nx.set_edge_attributes(graph, "single", "bond")
        graphs.append(graph)
    return graphs


def test_public_compression_api_round_trips_selected_labels():
    graphs = _triangle_corpus()
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=1,
        min_graph_support=1,
        min_occurrences=1,
        node_label_keys="atom",
        edge_label_keys="bond",
        selector="all_eligible",
        model_choice_bits=0.0,
        min_rule_savings_bits=float("-inf"),
        enumerator=ExhaustiveGraphletEnumerator(),
        validate=True,
    ).fit(graphs[:2])

    result = compressor.transform(graphs[2:])
    node_match = nx.algorithms.isomorphism.categorical_node_match("atom", None)
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("bond", None)
    assert nx.is_isomorphic(
        graphs[2],
        result.decoded_graphs()[0],
        node_match=node_match,
        edge_match=edge_match,
    )


def test_fit_and_transform_are_separate_public_operations():
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=1,
        min_graph_support=1,
        min_occurrences=1,
        enumerator=ExhaustiveGraphletEnumerator(),
    )
    assert hasattr(compressor, "fit")
    assert hasattr(compressor, "transform")
    assert hasattr(compressor, "fit_transform")


def test_legacy_and_top_level_imports_share_the_canonical_objects():
    assert buhito.MDLGraphCompressor is MDLGraphCompressor
    assert buhito.mdl.MDLGraphCompressor is MDLGraphCompressor
    assert (
        buhito.ExhaustiveGraphletEnumerator
        is ExhaustiveGraphletEnumerator
    )

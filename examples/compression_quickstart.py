"""Runnable dataset-independent compression example."""

import networkx as nx

from buhito.compression import ExhaustiveGraphletEnumerator, MDLGraphCompressor


def make_graph(repeats: int) -> nx.Graph:
    graph = nx.disjoint_union_all([nx.cycle_graph(3) for _ in range(repeats)])
    nx.set_node_attributes(graph, "C", "atom")
    nx.set_edge_attributes(graph, "single", "bond")
    return graph


train_graphs = [make_graph(repeats) for repeats in (3, 4, 5)]
test_graphs = [make_graph(repeats) for repeats in (6, 7)]

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
)

# Learn motif identities and selection decisions from training graphs only.
compressor.fit(train_graphs)

# Apply the frozen dictionary without inspecting held-out targets.
result = compressor.transform(test_graphs)
compressed_graphs = result.model_graphs(force_rewrite=True)
decoded_graphs = result.decoded_graphs()

node_match = nx.algorithms.isomorphism.categorical_node_match("atom", None)
edge_match = nx.algorithms.isomorphism.categorical_edge_match("bond", None)
for original, decoded in zip(test_graphs, decoded_graphs, strict=True):
    assert nx.is_isomorphic(
        original,
        decoded,
        node_match=node_match,
        edge_match=edge_match,
    )

print(compressor.dictionary_frame())
print(result.report)
print(f"compressed {len(compressed_graphs)} held-out graphs")

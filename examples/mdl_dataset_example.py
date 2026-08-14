"""Minimal model-agnostic MDL graph compression example."""

import networkx as nx

from buhito.mdl import MDLGraphCompressor


def make_dataset() -> list[nx.Graph]:
    graphs = []
    for repeats in range(3, 9):
        graph = nx.disjoint_union_all([nx.cycle_graph(3) for _ in range(repeats)])
        nx.set_node_attributes(graph, "C", "atom_key")
        nx.set_edge_attributes(graph, "1", "bond_key")
        graphs.append(graph)
    return graphs


graphs = make_dataset()
train_graphs = graphs[:4]
test_graphs = graphs[4:]

compressor = MDLGraphCompressor(
    graphlet_sizes=(3,),
    n_rules=5,
    min_graph_support=2,
    min_occurrences=4,
    node_label_keys="atom_key",
    edge_label_keys="bond_key",
    selector="sparse",
    cache_dir="artifacts/mdl_cache",
    progress=True,
)
compressor.fit(train_graphs)
result = compressor.transform(test_graphs)

print(compressor.dictionary_frame())
print(result.report)
print(result.per_graph)

# These are ordinary NetworkX graphs.  Pass them to any model-specific adapter.
compressed_graphs = result.model_graphs()

# The selected rewrites decode exactly to the selected topology/attributes.
decoded_graphs = result.decoded_graphs()
assert len(decoded_graphs) == len(test_graphs)

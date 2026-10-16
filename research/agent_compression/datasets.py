"""Small topology-only corpora for the agent-compression pilot.

These synthetic graphs exercise the harness; success here is not evidence of
compression gains on a scientific dataset.  All randomness belongs to a local
``random.Random`` instance, so calls do not change global random state.
"""

from __future__ import annotations

import random

import networkx as nx


def _chain_blocks(blocks: list[nx.Graph], rng: random.Random) -> nx.Graph:
    """Join small blocks by bridges with randomly chosen boundary ports."""
    graph = nx.disjoint_union_all(blocks)
    offset = 0
    previous: list[int] | None = None
    for block in blocks:
        current = list(range(offset, offset + len(block)))
        if previous is not None:
            graph.add_edge(rng.choice(previous), rng.choice(current))
        previous = current
        offset += len(block)
    return graph


def _packing_gadget() -> nx.Graph:
    """Three overlapping triangles where choosing the center blocks two others."""
    graph = nx.Graph()
    for triangle in ((0, 1, 2), (0, 3, 4), (1, 5, 6)):
        graph.add_edges_from(
            ((triangle[0], triangle[1]), (triangle[1], triangle[2]),
             (triangle[2], triangle[0]))
        )
    graph.add_edges_from(((3, 7), (4, 8), (5, 9), (6, 10)))
    return graph


def _make_graph(family: int, rng: random.Random) -> nx.Graph:
    if family == 0:
        # Disjoint motif interiors with varied boundary ports and competing
        # triangle occurrences inside complete graphs.
        blocks = [
            rng.choice((nx.cycle_graph(3), nx.cycle_graph(4), nx.complete_graph(4)))
            for _ in range(rng.randint(9, 10))
        ]
        return _chain_blocks(blocks, rng)
    if family == 1:
        blocks = [_packing_gadget(), _packing_gadget()]
        blocks.extend(nx.cycle_graph(rng.choice((3, 4))) for _ in range(3))
        return _chain_blocks(blocks, rng)
    if family == 2:
        columns = rng.randint(14, 18)
        graph = nx.ladder_graph(columns)
        for column in range(columns - 1):
            if rng.random() < 0.5:
                graph.add_edge(column, columns + column + 1)
        return graph
    if family == 3:
        blocks = [nx.cycle_graph(rng.choice((3, 4))) for _ in range(10)]
        graph = _chain_blocks(blocks, rng)
        # A small number of nonlocal edges changes boundary costs while keeping
        # the small induced-subgraph enumeration tractable.
        for _ in range(rng.randint(1, 4)):
            graph.add_edge(*rng.sample(list(graph), 2))
        return graph
    if family == 4:
        blocks = [nx.complete_graph(rng.choice((3, 4))) for _ in range(9)]
        return _chain_blocks(blocks, rng)
    if family == 5:
        # Sparse tree control: no triangle or square motifs, but many paths.
        count = rng.randint(29, 38)
        graph = nx.path_graph(count)
        for leaf in range(count - rng.randint(5, 9), count):
            graph.remove_edge(leaf - 1, leaf)
            graph.add_edge(leaf, rng.randrange(0, leaf - 1))
        return graph
    count = rng.randint(29, 38)
    graph_seed = rng.randrange(2**32)
    if family == 6:
        return nx.gnp_random_graph(
            count, rng.uniform(2.0, 3.5) / count, seed=graph_seed
        )
    return nx.gnm_random_graph(
        count, rng.randint(count + 4, 2 * count), seed=graph_seed
    )


def make_corpus(seed: int, n_graphs: int = 8) -> list[nx.Graph]:
    """Return a reproducible mixture of small, unlabeled simple graphs.

    The default has one graph from each of eight families.  Corpus seeds vary
    topology and node order, rather than merely relabeling one fixed corpus.
    Different seeds are not guaranteed to produce pairwise nonisomorphic
    graphs; a future scientific benchmark must also check dataset overlap.
    """
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(n_graphs, int) or n_graphs < 1:
        raise ValueError("n_graphs must be a positive integer")
    rng = random.Random(seed)
    graphs: list[nx.Graph] = []
    for index in range(n_graphs):
        graph = _make_graph(index % 8, rng)
        order = list(graph)
        rng.shuffle(order)
        graph = nx.relabel_nodes(graph, dict(zip(graph, order, strict=True)))
        # Rebuild in a randomized insertion order too: normalization maps node
        # insertion order to integers, and tie-breaking should see varied IDs.
        insertion_order = list(graph)
        rng.shuffle(insertion_order)
        shuffled = nx.Graph()
        shuffled.add_nodes_from(insertion_order)
        shuffled.add_edges_from(graph.edges())
        graphs.append(shuffled)
    return graphs


def positive_control() -> tuple[list[nx.Graph], list[nx.Graph]]:
    """Known positive MDL control from ``tests/test_mdl.py`` plus new sizes.

    Use topology-only schema, graphlet_sizes=(3,), n_rules=1, sparse selector,
    positive-gain threshold, and the exhaustive enumerator.  These are a
    control for nonvacuous rewrites, not a search objective or final test set.
    """
    def triangles(repeats: int) -> nx.Graph:
        return nx.disjoint_union_all([nx.cycle_graph(3) for _ in range(repeats)])

    training = [triangles(repeats) for repeats in (10, 12, 14)]
    validation = [triangles(repeats) for repeats in (11, 13, 15)]
    return training, validation

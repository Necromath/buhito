"""Handcrafted dependency-graph example with optional forced-prefix diagnostics."""

import argparse
import time

import networkx as nx
from _compression_common import add_run_arguments, finish_run

from buhito.compression import MDLGraphCompressor


def dependency_graph(heads, pos, relations):
    graph = nx.Graph()
    for index, label in enumerate(pos):
        graph.add_node(index, pos=label)
    for child, head in enumerate(heads):
        if child != head:
            graph.add_edge(child, head, dep=relations[child])
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser, "artifacts/examples/syntax")
    args = parser.parse_args()
    started = time.perf_counter()

    # These are illustrative dependency structures, not a downloaded AST corpus.
    fixtures = [
        dependency_graph(
            [1, 1, 1],
            ["NOUN", "VERB", "NOUN"],
            ["nsubj", "ROOT", "obj"],
        ),
        dependency_graph(
            [1, 1, 1, 1],
            ["NOUN", "VERB", "NOUN", "ADV"],
            ["nsubj", "ROOT", "obj", "advmod"],
        ),
        dependency_graph(
            [1, 2, 2, 2],
            ["DET", "NOUN", "VERB", "NOUN"],
            ["det", "nsubj", "ROOT", "obj"],
        ),
        dependency_graph(
            [3, 3, 3, 3],
            ["NOUN", "AUX", "PART", "VERB"],
            ["nsubj", "aux", "neg", "ROOT"],
        ),
    ]
    fitting = [g.copy() for g in fixtures for _ in range(3)]
    held_out = [g.copy() for g in fixtures]
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=args.n_rules,
        max_candidates=args.max_candidates,
        node_label_keys="pos",
        edge_label_keys="dep",
        validate=True,
    )
    compressor.fit(fitting)
    result = compressor.transform(held_out)
    finish_run(
        compressor,
        result,
        held_out,
        args,
        {
            "dataset": "syntax fixture",
            "input_kind": "handcrafted dependency graphs",
            "split": "independent copies of the same fixtures; no generalization claim",
            "fit_graphs": len(fitting),
            "held_out_graphs": len(held_out),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()

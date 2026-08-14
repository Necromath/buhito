import math

import networkx as nx
import pytest

from buhito.mdl import (
    ExhaustiveGraphletEnumerator,
    GraphSchema,
    MDLGraphCompressor,
    labeled_isomorphic,
)


def _labeled_cycle(n: int) -> nx.Graph:
    graph = nx.cycle_graph(n)
    nx.set_node_attributes(graph, "C", "atom")
    nx.set_edge_attributes(graph, "single", "bond")
    return graph


def _triangle_chain(repeats: int) -> nx.Graph:
    graph = nx.Graph()
    previous = None
    cursor = 0
    for _ in range(repeats):
        nodes = (cursor, cursor + 1, cursor + 2)
        graph.add_edges_from(
            [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])]
        )
        if previous is not None:
            graph.add_edge(previous, nodes[0])
        previous = nodes[2]
        cursor += 3
    nx.set_node_attributes(graph, "C", "atom")
    nx.set_edge_attributes(graph, "single", "bond")
    return graph


def _disjoint_triangles(repeats: int) -> nx.Graph:
    graph = nx.Graph()
    for index in range(repeats):
        first = 3 * index
        graph.add_edges_from(
            [
                (first, first + 1),
                (first + 1, first + 2),
                (first + 2, first),
            ]
        )
    return graph


def test_schema_roundtrip_selected_attributes():
    graph = _labeled_cycle(5)
    schema = GraphSchema.from_keys(
        node_label_keys="atom", edge_label_keys="bond"
    )
    normalized = schema.normalize(graph)
    restored = schema.restore(normalized)
    assert nx.is_isomorphic(
        graph,
        restored,
        node_match=lambda left, right: left["atom"] == right["atom"],
        edge_match=lambda left, right: left["bond"] == right["bond"],
    )


def test_rewrite_decodes_exactly_for_selected_labels():
    graphs = [_triangle_chain(4), _triangle_chain(5), _triangle_chain(6)]
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=1,
        min_graph_support=1,
        min_occurrences=1,
        max_candidates=10,
        node_label_keys="atom",
        edge_label_keys="bond",
        selector="all_eligible",
        model_choice_bits=0.0,
        min_rule_savings_bits=-math.inf,
        enumerator=ExhaustiveGraphletEnumerator(),
        validate=True,
    )
    result = compressor.fit_transform(graphs)
    assert compressor.rules_ is not None
    assert len(compressor.rules_) <= 1

    for original, record in zip(graphs, result.records):
        normalized_original = compressor.schema.normalize(original)
        if record.rewrite is not None:
            decoded = record.normalized_decoded_graph()
            assert labeled_isomorphic(normalized_original, decoded)
            assert isinstance(record.normalized_model_graph(force_rewrite=True), nx.MultiGraph)

    model_graphs = result.model_graphs(force_rewrite=True)
    assert any(isinstance(graph, nx.MultiGraph) for graph in model_graphs)


def test_fit_transform_is_model_agnostic_and_exposes_features():
    train = [_triangle_chain(4), _triangle_chain(5), _triangle_chain(6)]
    test = [_triangle_chain(3), _labeled_cycle(7)]
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=2,
        min_graph_support=1,
        min_occurrences=1,
        max_candidates=10,
        node_label_keys="atom",
        edge_label_keys="bond",
        selector="per_graph",
        model_choice_bits=0.0,
        min_rule_savings_bits=-math.inf,
        enumerator=ExhaustiveGraphletEnumerator(),
    ).fit(train)

    result = compressor.transform(test)
    assert len(result.model_graphs()) == len(test)
    assert len(result.decoded_graphs()) == len(test)
    assert {
        "baseline_bits",
        "rewrite_bits",
        "gross_gain_bits",
        "unseen_graphlet_fraction",
    }.issubset(result.per_graph.columns)
    assert result.report.n_graphs == len(test)


def test_rejects_directed_and_multigraph_inputs():
    compressor = MDLGraphCompressor(
        enumerator=ExhaustiveGraphletEnumerator()
    )
    with pytest.raises(TypeError):
        compressor.fit([nx.DiGraph([(0, 1)])])
    with pytest.raises(TypeError):
        compressor.fit([nx.MultiGraph([(0, 1)])])


def test_fixed_dictionary_selection_keeps_a_nonempty_diagnostic_rule():
    graphs = [nx.path_graph(6), nx.path_graph(7), nx.path_graph(8)]
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=1,
        min_graph_support=1,
        min_occurrences=1,
        max_candidates=5,
        selector="sparse",
        min_rule_savings_bits=-math.inf,
        dictionary_selection="fixed",
        enumerator=ExhaustiveGraphletEnumerator(),
        validate=True,
    ).fit(graphs)
    assert compressor.rules_ is not None
    assert len(compressor.rules_) == 1
    assert compressor.dictionary_path_ is not None
    assert set(compressor.dictionary_path_["n_rules"]) == {0, 1}


def test_forced_candidate_cannot_fall_back_to_empty_dictionary():
    graphs = [nx.path_graph(6), nx.path_graph(7), nx.path_graph(8)]
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=1,
        min_graph_support=1,
        min_occurrences=1,
        max_candidates=5,
        selector="sparse",
        min_rule_savings_bits=-math.inf,
        dictionary_selection="best",
        enumerator=ExhaustiveGraphletEnumerator(),
        validate=True,
    ).fit(graphs)

    row = compressor.candidate_frame().iloc[0]
    assert row["forced_dictionary_bits"] > 0.0
    assert row["forced_n_rewritten"] >= 1
    assert row["forced_n_occurrences"] >= 1
    assert row["forced_net_savings_bits"] != row["best_model_net_savings_bits"]


def test_forced_candidate_scores_reflect_candidate_costs():
    graphs = [_triangle_chain(4), _triangle_chain(5), _triangle_chain(6)]
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=2,
        min_graph_support=1,
        min_occurrences=1,
        max_candidates=10,
        selector="sparse",
        min_rule_savings_bits=-math.inf,
        dictionary_selection="best",
        enumerator=ExhaustiveGraphletEnumerator(),
        validate=True,
    ).fit(graphs)

    candidates = compressor.candidate_frame()
    assert len(candidates) >= 2
    assert candidates["forced_net_savings_bits"].nunique() > 1
    assert candidates["forced_dictionary_bits"].gt(0.0).all()


def test_negative_forced_candidate_remains_in_candidate_table():
    graphs = [nx.path_graph(6), nx.path_graph(7), nx.path_graph(8)]
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=1,
        min_graph_support=1,
        min_occurrences=1,
        max_candidates=5,
        selector="sparse",
        min_rule_savings_bits=0.0,
        dictionary_selection="best",
        enumerator=ExhaustiveGraphletEnumerator(),
        validate=True,
    ).fit(graphs)

    candidates = compressor.candidate_frame()
    assert len(candidates) == 1
    assert candidates.iloc[0]["forced_net_savings_bits"] < 0.0
    assert not bool(candidates.iloc[0]["passes_min_rule_savings"])
    assert compressor.rules_ == ()


def test_positive_low_boundary_candidate_is_selected():
    graphs = [
        _disjoint_triangles(10),
        _disjoint_triangles(12),
        _disjoint_triangles(14),
    ]
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=1,
        min_graph_support=1,
        min_occurrences=1,
        max_candidates=5,
        selector="sparse",
        min_rule_savings_bits=0.0,
        dictionary_selection="best",
        enumerator=ExhaustiveGraphletEnumerator(),
        validate=True,
    ).fit(graphs)

    row = compressor.candidate_frame().iloc[0]
    assert row["forced_net_savings_bits"] > 0.0
    assert bool(row["is_selected"])
    assert compressor.rules_ is not None
    assert len(compressor.rules_) == 1


def test_empty_dictionary_can_remain_optimal_with_complete_path():
    graphs = [nx.path_graph(6), nx.path_graph(7), nx.path_graph(8)]
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=1,
        min_graph_support=1,
        min_occurrences=1,
        max_candidates=5,
        selector="sparse",
        min_rule_savings_bits=-math.inf,
        dictionary_selection="best",
        enumerator=ExhaustiveGraphletEnumerator(),
        validate=True,
    ).fit(graphs)

    path = compressor.dictionary_path_frame()
    assert set(path["n_rules"]) == {0, 1}
    assert int(path["is_best"].sum()) == 1
    assert bool(path.loc[path["n_rules"] == 0, "is_best"].iloc[0])
    assert path.loc[path["n_rules"] == 1, "dictionary_bits"].iloc[0] > 0.0
    assert compressor.rules_ == ()


def test_dictionary_path_retains_every_evaluated_prefix():
    graphs = [_triangle_chain(4), _triangle_chain(5), _triangle_chain(6)]
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=2,
        min_graph_support=1,
        min_occurrences=1,
        max_candidates=10,
        selector="sparse",
        min_rule_savings_bits=-math.inf,
        dictionary_selection="best",
        enumerator=ExhaustiveGraphletEnumerator(),
        validate=True,
    ).fit(graphs)

    path = compressor.dictionary_path_frame()
    assert set(path["n_rules"]) == {0, 1, 2}
    assert int(path["is_best"].sum()) == 1
    assert path.loc[path["n_rules"] > 0, "dictionary_bits"].gt(0.0).all()
    assert path.loc[path["n_rules"] > 0, "is_empty"].eq(False).all()


def test_candidate_accounting_preserves_exact_decoding():
    graphs = [_triangle_chain(4), _triangle_chain(5), _triangle_chain(6)]
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=2,
        min_graph_support=1,
        min_occurrences=1,
        max_candidates=10,
        selector="sparse",
        min_rule_savings_bits=-math.inf,
        dictionary_selection="best",
        enumerator=ExhaustiveGraphletEnumerator(),
        validate=True,
    )
    result = compressor.fit_transform(graphs)

    for original, decoded in zip(graphs, result.decoded_graphs(), strict=True):
        assert nx.is_isomorphic(original, decoded)


def test_candidate_motif_graphs_are_restored_and_independent_copies():
    graphs = [_triangle_chain(4), _triangle_chain(5), _triangle_chain(6)]
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=2,
        min_graph_support=1,
        min_occurrences=1,
        max_candidates=10,
        node_label_keys="atom",
        edge_label_keys="bond",
        selector="sparse",
        min_rule_savings_bits=-math.inf,
        dictionary_selection="best",
        enumerator=ExhaustiveGraphletEnumerator(),
        validate=True,
    ).fit(graphs)

    candidates = compressor.candidate_frame()
    assert "motif_id" in candidates
    assert candidates["motif_id"].str.match(r"M\d{3}").all()

    motifs = compressor.candidate_motif_graphs()
    assert set(motifs) == set(candidates["key"])
    key = str(candidates.iloc[0]["key"])
    motif = motifs[key]
    assert all(data["atom"] == "C" for _, data in motif.nodes(data=True))
    assert all(data["bond"] == "single" for _, _, data in motif.edges(data=True))

    motif.add_node("local mutation", atom="X")
    fresh = compressor.candidate_motif(key)
    assert "local mutation" not in fresh

    with pytest.raises(KeyError, match="Unknown motif key"):
        compressor.candidate_motif("not-a-real-candidate")


def test_candidate_identity_is_invariant_to_node_relabeling():
    """Node IDs and insertion order must not change labeled motif identity."""
    import networkx as nx

    from buhito.mdl import (
        ExhaustiveGraphletEnumerator,
        MDLGraphCompressor,
    )

    first = nx.Graph()
    first.add_edge(0, 1, bond="single")
    first.add_edge(1, 2, bond="double")

    first.nodes[0]["atom"] = "N"
    first.nodes[1]["atom"] = "C"
    first.nodes[2]["atom"] = "O"

    # Same attributed graph with unrelated node IDs and reversed insertion
    # order. The endpoint labels and incident edge labels remain attached to
    # the corresponding structural positions.
    second = nx.Graph()
    second.add_edge(41, 17, bond="double")
    second.add_edge(93, 41, bond="single")

    second.nodes[93]["atom"] = "N"
    second.nodes[41]["atom"] = "C"
    second.nodes[17]["atom"] = "O"

    assert nx.is_isomorphic(
        first,
        second,
        node_match=nx.algorithms.isomorphism.categorical_node_match(
            "atom",
            None,
        ),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match(
            "bond",
            None,
        ),
    )

    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=1,
        min_graph_support=1,
        min_occurrences=1,
        max_candidates=10,
        node_label_keys="atom",
        edge_label_keys="bond",
        enumerator=ExhaustiveGraphletEnumerator(),
    ).fit([first, second])

    candidates = compressor.candidate_frame()

    # Both input graphs contain the same labeled P3 candidate. They must
    # contribute support to one canonical candidate rather than creating two.
    assert len(candidates) == 1
    assert int(candidates.iloc[0]["graph_support"]) == 2
    assert int(candidates.iloc[0]["total_occurrences"]) == 2

    motif_graphs = compressor.candidate_motif_graphs()
    assert len(motif_graphs) == 1

    motif = next(iter(motif_graphs.values()))

    assert nx.is_isomorphic(
        first,
        motif,
        node_match=nx.algorithms.isomorphism.categorical_node_match(
            "atom",
            None,
        ),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match(
            "bond",
            None,
        ),
    )


def _normalized_star_forest(component_count: int = 2, leaves: int = 25):
    import networkx as nx

    from buhito.mdl import _EDGE_LABEL, _NODE_KIND, _NODE_LABEL

    graph = nx.Graph()
    next_node = 0
    centers = []
    for _ in range(component_count):
        center = next_node
        centers.append(center)
        next_node += 1
        graph.add_node(
            center,
            **{
                _NODE_LABEL: ("data", ("center",)),
                _NODE_KIND: "data",
            },
        )
        for _ in range(leaves):
            leaf = next_node
            next_node += 1
            graph.add_node(
                leaf,
                **{
                    _NODE_LABEL: ("data", ("leaf",)),
                    _NODE_KIND: "data",
                },
            )
            graph.add_edge(
                center,
                leaf,
                **{_EDGE_LABEL: ("data", ("edge",))},
            )
    graph.graph["name"] = "symmetric-star-forest"
    return graph, centers


def _star_forest_rewrite():
    from buhito.mdl import (
        MotifRule,
        Occurrence,
        _make_motif,
        _rewrite_graph,
    )

    graph, centers = _normalized_star_forest()
    first_center = centers[0]
    first_leaves = sorted(graph.neighbors(first_center))[:2]
    motif = _make_motif(
        graph.subgraph([first_leaves[0], first_center, first_leaves[1]]).copy()
    )
    rule = MotifRule(key="star-path", base_key=(3, "exact"), motif=motif, rank=0)

    occurrences = []
    for center in centers:
        leaves = sorted(graph.neighbors(center))[:2]
        # The motif's center is port 1 because host IDs are sorted leaf,center,leaf
        # only for the first component. Resolve the mapping label-wise instead of
        # assuming that ordering for every component.
        mapping = [None, None, None]
        for port, data in motif.nodes(data=True):
            label = data["__buhito_mdl_node_label__"]
            if label[1] == ("center",):
                mapping[port] = center
            else:
                mapping[port] = leaves.pop(0)
        occurrences.append(Occurrence(tuple(mapping)))

    rewrite = _rewrite_graph(graph, (rule,), {0: occurrences})
    return graph, rewrite


def test_exact_rewrite_validator_handles_many_symmetric_leaves(monkeypatch):
    import networkx as nx

    from buhito.mdl import (
        _decode_rewrite_original_ids,
        _validate_rewrite_exact,
    )

    graph, rewrite = _star_forest_rewrite()

    def fail_isomorphism(*args, **kwargs):
        raise AssertionError("generic graph isomorphism entered validation hot path")

    monkeypatch.setattr(nx, "is_isomorphic", fail_isomorphism)
    decoded = _decode_rewrite_original_ids(rewrite)
    assert set(decoded.nodes()) == set(graph.nodes())
    assert dict(decoded.graph) == dict(graph.graph)
    assert _validate_rewrite_exact(graph, rewrite)


def test_exact_rewrite_validator_detects_corruption():
    import copy

    from buhito.mdl import (
        _EDGE_LABEL,
        _NODE_LABEL,
        _PORTS,
        _rewrite_validation_error,
    )

    graph, rewrite = _star_forest_rewrite()

    corrupted = copy.deepcopy(rewrite)
    supernode = corrupted.supernodes[0]
    rule = corrupted.rules[corrupted.supernode_rule[supernode]]
    rule.motif.nodes[0][_NODE_LABEL] = ("data", ("wrong",))
    assert "node-attribute mismatch" in _rewrite_validation_error(graph, corrupted)

    corrupted = copy.deepcopy(rewrite)
    supernode = corrupted.supernodes[0]
    rule = corrupted.rules[corrupted.supernode_rule[supernode]]
    source, target = next(iter(rule.motif.edges()))
    rule.motif.edges[source, target][_EDGE_LABEL] = ("data", ("wrong",))
    assert "edge-attribute mismatch" in _rewrite_validation_error(graph, corrupted)

    corrupted = copy.deepcopy(rewrite)
    supernode = corrupted.supernodes[0]
    rule = corrupted.rules[corrupted.supernode_rule[supernode]]
    source, target = next(iter(rule.motif.edges()))
    rule.motif.remove_edge(source, target)
    assert "missing edge" in _rewrite_validation_error(graph, corrupted)

    corrupted = copy.deepcopy(rewrite)
    boundary = next(
        data
        for source, target, _, data in corrupted.template.edges(keys=True, data=True)
        if source in corrupted.supernodes or target in corrupted.supernodes
    )
    boundary_supernode = next(iter(boundary[_PORTS]))
    port_count = len(corrupted.canonical_port_to_host[boundary_supernode])
    boundary[_PORTS][boundary_supernode] = (
        int(boundary[_PORTS][boundary_supernode]) + 1
    ) % port_count
    assert _rewrite_validation_error(graph, corrupted) is not None


def test_rule_prefix_transform_is_nested_and_does_not_mutate_selected_rules():
    graphs = [_triangle_chain(4), _triangle_chain(5), _triangle_chain(6)]
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=2,
        min_graph_support=1,
        min_occurrences=1,
        max_candidates=10,
        selector="sparse",
        min_rule_savings_bits=-math.inf,
        dictionary_selection="fixed",
        enumerator=ExhaustiveGraphletEnumerator(),
        validate=True,
    ).fit(graphs)

    selected_before = tuple(rule.key for rule in compressor.rules_ or ())
    available = len(compressor.candidate_frame())
    assert available >= 2
    assert compressor.rule_prefix(0) == ()
    assert [rule.rank for rule in compressor.rule_prefix(2)] == [0, 1]

    baseline = compressor.transform_rule_prefix(graphs, 0)
    one_rule = compressor.transform_rule_prefix(
        graphs,
        1,
        require_nonempty_rewrite=True,
    )
    two_rules = compressor.transform_rule_prefix(
        graphs,
        2,
        require_nonempty_rewrite=True,
    )

    assert all(record.rewrite is None for record in baseline.records)
    assert one_rule.report.n_rewritten >= 1
    assert two_rules.report.n_rewritten >= 1
    assert tuple(rule.key for rule in compressor.rules_ or ()) == selected_before

    with pytest.raises(ValueError, match="only .* scored candidates"):
        compressor.rule_prefix(available + 1)

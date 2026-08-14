"""Compatibility tests for the original Buhito graphlet APIs."""

import networkx as nx
import pytest

from buhito.converters import smiles_to_nx
from buhito.featurizers.bfs_graphlet_featurizer import BFSGraphletFeaturizer
from buhito.featurizers.dfs_graphlet_featurizer import DFSGraphletFeaturizer
from buhito.transformers import GraphletTransformer


def labeled_petersen_graph() -> nx.Graph:
    graph = nx.petersen_graph()
    nx.set_node_attributes(graph, "C", "atom_key")
    nx.set_edge_attributes(graph, "1", "bond_key")
    return graph


class TestBFSGraphletFeaturizer:
    def test_init(self) -> None:
        featurizer = BFSGraphletFeaturizer(max_len=3)
        assert featurizer.size == 3
        assert featurizer.return_nodewise is True

    def test_call_petersen_graph(self) -> None:
        fingerprint, bitinfo = BFSGraphletFeaturizer(
            max_len=3,
            return_nodewise=False,
        )(labeled_petersen_graph())
        assert isinstance(fingerprint, dict)
        assert isinstance(bitinfo, dict)
        assert fingerprint
        assert bitinfo

    def test_call_nodewise(self) -> None:
        graph = labeled_petersen_graph()
        fingerprint, bitinfo, node_fps, node_bitinfo = BFSGraphletFeaturizer(
            max_len=3,
            return_nodewise=True,
        )(graph)
        assert isinstance(fingerprint, dict)
        assert isinstance(bitinfo, dict)
        assert isinstance(node_fps, dict)
        assert isinstance(node_bitinfo, dict)
        assert len(node_fps) == graph.number_of_nodes()
        assert len(node_bitinfo) == graph.number_of_nodes()


class TestDFSGraphletFeaturizer:
    def test_init(self) -> None:
        featurizer = DFSGraphletFeaturizer(max_len=3)
        assert featurizer.size == 3

    def test_call_petersen_graph(self) -> None:
        fingerprint, bitinfo = DFSGraphletFeaturizer(max_len=3)(
            labeled_petersen_graph()
        )
        assert isinstance(fingerprint, dict)
        assert isinstance(bitinfo, dict)
        assert fingerprint
        assert bitinfo


class TestFeaturizerConsistency:
    def test_bfs_dfs_consistency(self) -> None:
        graph = labeled_petersen_graph()
        bfs_fingerprint, _ = BFSGraphletFeaturizer(
            max_len=3,
            return_nodewise=False,
        )(graph)
        dfs_fingerprint, _ = DFSGraphletFeaturizer(max_len=3)(graph)
        assert set(bfs_fingerprint) == set(dfs_fingerprint)


class TestConverters:
    def test_smiles_to_nx(self) -> None:
        pytest.importorskip("rdkit")
        graph, positions = smiles_to_nx("CCO")
        assert isinstance(graph, nx.Graph)
        assert positions is None
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 2
        assert all("atom_key" in graph.nodes[node] for node in graph)
        assert all("bond_key" in graph.edges[edge] for edge in graph.edges)


class TestGraphletTransformer:
    def test_init(self) -> None:
        featurizer = BFSGraphletFeaturizer(max_len=3)
        transformer = GraphletTransformer(featurizer=featurizer)
        assert transformer.featurizer == featurizer

    def test_fit_transform(self) -> None:
        graph = labeled_petersen_graph()
        featurizer = BFSGraphletFeaturizer(max_len=3, return_nodewise=False)
        transformer = GraphletTransformer(featurizer=featurizer, n_jobs=1)
        features = transformer.fit_transform([graph])
        assert hasattr(transformer, "n_bits_")
        assert hasattr(transformer, "bit_ids_")
        assert features.shape == (1, transformer.n_bits_)

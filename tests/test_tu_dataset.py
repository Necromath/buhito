from pathlib import Path

import networkx as nx

from buhito.datasets import load_tu_dataset


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_loads_canonical_tu_files_with_discrete_labels(tmp_path):
    name = "TINY"
    directory = tmp_path / name
    directory.mkdir()
    _write(directory / f"{name}_A.txt", "1, 2\n2, 1\n3, 4\n4, 5\n")
    _write(directory / f"{name}_graph_indicator.txt", "1\n1\n2\n2\n2\n")
    _write(directory / f"{name}_graph_labels.txt", "-1\n1\n")
    _write(directory / f"{name}_node_labels.txt", "1\n2\n1\n1\n2\n")
    _write(directory / f"{name}_edge_labels.txt", "7\n7\n8\n9\n")

    dataset = load_tu_dataset(tmp_path, name)
    assert len(dataset.graphs) == 2
    assert dataset.graph_labels.tolist() == [-1, 1]
    assert dataset.node_label_key == "tu_node_label"
    assert dataset.edge_label_key == "tu_edge_label"
    assert dataset.graphs[0].number_of_edges() == 1
    assert nx.get_node_attributes(dataset.graphs[1], "tu_node_label") == {
        0: 1,
        1: 1,
        2: 2,
    }


def test_loads_legacy_reddit_names_and_degree_mode(tmp_path):
    name = "REDDIT-MULTI-5K"
    directory = tmp_path / name
    directory.mkdir()
    _write(directory / f"{name}.edges", "1,2\n2,3\n4,5\n")
    _write(directory / f"{name}.graph_idx", "1\n1\n1\n2\n2\n")
    _write(directory / f"{name}.graph_labels", "1\n2\n")

    dataset = load_tu_dataset(
        tmp_path,
        name,
        node_label_mode="degree",
        edge_label_mode="constant",
    )
    assert dataset.node_label_key == "tu_degree_label"
    assert dataset.edge_label_key == "tu_constant_edge_label"
    assert nx.get_node_attributes(dataset.graphs[0], "tu_degree_label") == {
        0: 1,
        1: 2,
        2: 1,
    }


def test_finds_pyg_raw_subdirectory(tmp_path):
    name = "REDDIT-MULTI-5K"
    directory = tmp_path / name / "raw"
    directory.mkdir(parents=True)
    _write(directory / f"{name}.edges", "1,2\n2,3\n4,5\n")
    _write(directory / f"{name}.graph_idx", "1\n1\n1\n2\n2\n")
    _write(directory / f"{name}.graph_labels", "1\n2\n")

    dataset = load_tu_dataset(tmp_path, name, node_label_mode="none")
    assert dataset.source_directory == directory.resolve()
    assert len(dataset.graphs) == 2


def test_finds_nested_dataset_directory_with_unrelated_name(tmp_path):
    name = "REDDIT-MULTI-5K"
    directory = tmp_path / "download-cache" / "tu" / "raw"
    directory.mkdir(parents=True)
    _write(directory / f"{name}_A.txt", "1,2\n2,3\n4,5\n")
    _write(directory / f"{name}_graph_indicator.txt", "1\n1\n1\n2\n2\n")
    _write(directory / f"{name}_graph_labels.txt", "1\n2\n")

    dataset = load_tu_dataset(tmp_path, name, node_label_mode="none")
    assert dataset.source_directory == directory.resolve()
    assert len(dataset.graphs) == 2

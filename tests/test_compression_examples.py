"""Check label-aware validation, disjoint sampling, and external input adapters."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from buhito.compression import GraphSchema

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def helpers(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    return importlib.import_module("_compression_common")


@pytest.mark.parametrize("corruption", ["node", "edge", "missing"])
def test_selected_attribute_corruption_is_rejected(helpers, corruption):
    original = nx.cycle_graph(3)
    nx.set_node_attributes(original, ("C", 0), "atom")
    nx.set_edge_attributes(original, "single", "bond")
    decoded = nx.relabel_nodes(original, {0: 10, 1: 11, 2: 12})
    if corruption == "node":
        decoded.nodes[10]["atom"] = ("N", 0)
    elif corruption == "edge":
        decoded.edges[10, 11]["bond"] = "double"
    else:
        del decoded.nodes[10]["atom"]
    schema = GraphSchema.from_keys(node_label_keys="atom", edge_label_keys="bond")
    with pytest.raises((ValueError, RuntimeError)):
        helpers.assert_exact([original], [decoded], schema)


def test_sampling_is_disjoint_and_reproducible(helpers):
    fitting, held_out = helpers.split_indices(20, 8, 5, seed=4)
    assert not set(fitting) & set(held_out)
    assert (fitting, held_out) == helpers.split_indices(20, 8, 5, seed=4)
    with pytest.raises(ValueError, match="received 3"):
        helpers.split_indices(3, 2, 2, seed=4)


def _run(script, output, *arguments):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / script),
            "--output-dir",
            str(output),
            *arguments,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["held_out_decode_failures"] == 0
    # Empty scientific results still produce readable tables.
    pd.read_csv(output / "candidates.csv")
    pd.read_csv(output / "dictionary.csv")
    return summary


def test_tu_cli_preserves_labels_and_disjoint_source_indices(tmp_path):
    name = "TEST"
    raw = tmp_path / name
    raw.mkdir()
    edges = []
    for offset in range(0, 24, 3):
        edges.extend([f"{offset + 1},{offset + 2}", f"{offset + 2},{offset + 3}"])
    (raw / f"{name}_A.txt").write_text("\n".join(edges) + "\n")
    (raw / f"{name}_graph_indicator.txt").write_text(
        "".join(f"{i // 3 + 1}\n" for i in range(24)),
    )
    (raw / f"{name}_graph_labels.txt").write_text("1\n-1\n" * 4)
    (raw / f"{name}_node_labels.txt").write_text("6\n" * 24)
    (raw / f"{name}_edge_labels.txt").write_text("1\n" * len(edges))
    summary = _run(
        "compression_tu.py",
        tmp_path / "output",
        "--data-root",
        str(tmp_path),
        "--dataset",
        name,
        "--fit-size",
        "6",
        "--test-size",
        "2",
    )
    assert not set(summary["fit_indices"]) & set(summary["held_out_indices"])
    assert summary["node_label_keys"] == ["tu_node_label"]
    assert summary["edge_label_keys"] == ["tu_edge_label"]


def test_molecular_csv_cli_retains_source_rows_after_deduplication(tmp_path):
    pytest.importorskip("rdkit")
    path = tmp_path / "molecules.csv"
    pd.DataFrame(
        {
            "smiles": ["CCC", "CCC", "CCCC", "CCO", "CCN", "C1CC1", "c1ccccc1"],
            "target": range(7),
        }
    ).to_csv(path, index=False)
    summary = _run(
        "compression_molecules.py",
        tmp_path / "output",
        "--data",
        str(path),
        "--dataset",
        "QMugs input fixture",
        "--fit-size",
        "4",
        "--test-size",
        "2",
    )
    fitting, held_out = summary["fit_source_rows"], summary["held_out_source_rows"]
    assert not set(fitting) & set(held_out)
    assert 1 not in fitting + held_out
    assert summary["node_label_keys"] == ["atom_key"]


def test_zinc_cli_uses_supplied_splits_and_separates_forced_result(tmp_path):
    record = {
        "num_nodes": 3,
        "node_feat": [[6], [6], [6]],
        "edge_index": [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]],
        "edge_attr": [[1]] * 6,
        "y": [0.2],
    }
    for split, count in [("train", 8), ("test", 2)]:
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for _ in range(count)),
        )
    output = tmp_path / "output"
    summary = _run(
        "compression_zinc.py",
        output,
        "--data-root",
        str(tmp_path),
        "--train-limit",
        "8",
        "--test-limit",
        "2",
        "--force-prefix",
        "1",
    )
    forced = json.loads((output / "forced" / "summary.json").read_text())
    assert summary["mode"] == "mdl_selected"
    assert forced["mode"] == "forced_prefix_diagnostic"
    assert forced["selected_rule_count"] == 1
    assert summary["fit_indices"] == list(range(8))
    assert summary["held_out_indices"] == [0, 1]


def test_admet_example_keeps_official_partition_boundary(
    helpers, monkeypatch, tmp_path
):
    pytest.importorskip("rdkit")
    module = importlib.import_module("compression_admet")
    train = pd.DataFrame({"Drug": ["C1CC1"] * 8, "Y": [0, 1] * 4})
    test = pd.DataFrame({"Drug": ["N1NN1"] * 3, "Y": [1, 0, 1]})
    monkeypatch.setattr(
        module, "_load_benchmark", lambda *args: (None, "test", train, test)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compression_admet.py",
            "--train-limit",
            "4",
            "--test-limit",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
    )
    module.main()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["official_train_val_rows"] == 8
    assert summary["official_test_rows"] == 3
    assert len(summary["fit_source_rows"]) == 4
    assert len(summary["held_out_source_rows"]) == 2
    assert summary["original_nodes"] == 6
    assert summary["code"]["n_occurrences"] == 0

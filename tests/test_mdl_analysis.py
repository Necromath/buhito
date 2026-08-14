from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

from buhito.analysis import (
    AnalysisConfig,
    build_motif_cost_table,
    build_motif_table,
    run_mdl_analysis,
)


@dataclass
class FakeReport:
    baseline_bits: float
    encoded_bits: float
    net_savings_bits: float


class FakeTransformResult:
    def __init__(self, graphs):
        self._graphs = [graph.copy() for graph in graphs]
        self.report = FakeReport(
            baseline_bits=100.0 * len(graphs),
            encoded_bits=80.0 * len(graphs),
            net_savings_bits=20.0 * len(graphs),
        )
        self.per_graph = pd.DataFrame(
            {
                "graph_index": range(len(graphs)),
                "use_rewrite": [True] * len(graphs),
                "net_savings_bits": [20.0] * len(graphs),
                "node_reduction_fraction": [0.25] * len(graphs),
            }
        )

    def decoded_graphs(self):
        return [graph.copy() for graph in self._graphs]


class FakeCompressor:
    def fit(self, graphs):
        self.fit_count = len(graphs)
        return self

    def transform(self, graphs):
        return FakeTransformResult(graphs)

    def candidate_frame(self):
        return pd.DataFrame(
            {
                "rank": [0, 1],
                "key": ["triangle", "path"],
                "graph_support": [8, 10],
                "total_occurrences": [20, 30],
                "single_rule_savings_bits": [50.0, -10.0],
            }
        )

    def dictionary_frame(self):
        return pd.DataFrame({"key": ["triangle"]})

    def dictionary_path_frame(self):
        return pd.DataFrame(
            {
                "n_rules": [0, 1],
                "net_savings_bits": [0.0, 50.0],
            }
        )


def test_build_motif_table_marks_selection_and_derived_statistics():
    motifs = build_motif_table(
        candidates=FakeCompressor().candidate_frame(),
        dictionary=FakeCompressor().dictionary_frame(),
        n_fit_graphs=10,
    )
    triangle = motifs.loc[motifs["key"] == "triangle"].iloc[0]
    assert bool(triangle["is_selected"])
    assert triangle["support_fraction"] == 0.8
    assert triangle["savings_per_occurrence_bits"] == 2.5


def test_run_analysis_is_reproducible_and_writes_artifacts(tmp_path: Path):
    graphs = [nx.cycle_graph(size) for size in range(4, 16)]
    config = AnalysisConfig(
        fit_size=8,
        eval_size=4,
        seed=7,
        bootstrap_replicates=50,
    )
    result = run_mdl_analysis(
        graphs,
        FakeCompressor(),
        config=config,
        output_dir=tmp_path,
    )

    assert result.summary["n_fit_graphs"] == 8
    assert result.summary["n_eval_graphs"] == 4
    assert result.summary["fit_decode_failures"] == 0
    assert result.summary["eval_decode_failures"] == 0
    assert not result.bootstrap.empty
    assert (tmp_path / "motifs.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "README.md").exists()


class PrivateCandidateCompressor(FakeCompressor):
    def __init__(self):
        self._candidate_rows = [
            {
                "rank": 0,
                "candidate_key": "triangle",
                "support": 8,
                "occurrences": 20,
                "bits_saved": 50.0,
            },
            {
                "rank": 1,
                "candidate_key": "path",
                "support": 10,
                "occurrences": 30,
                "bits_saved": -10.0,
            },
        ]

    candidate_frame = None


def test_run_analysis_discovers_legacy_private_candidate_rows():
    graphs = [nx.cycle_graph(size) for size in range(4, 16)]
    result = run_mdl_analysis(
        graphs,
        PrivateCandidateCompressor(),
        config=AnalysisConfig(
            fit_size=8,
            eval_size=4,
            seed=7,
            bootstrap_replicates=10,
        ),
    )

    assert result.summary["n_candidate_motifs"] == 2
    assert set(result.motifs["key"]) == {"triangle", "path"}


def test_build_motif_table_prefers_forced_candidate_accounting():
    candidates = pd.DataFrame(
        {
            "rank": [1],
            "key": ["triangle"],
            "graph_support": [4],
            "total_occurrences": [8],
            "forced_dictionary_bits": [12.0],
            "forced_net_savings_bits": [24.0],
            "single_rule_savings_bits": [-1.0],
        }
    )
    motifs = build_motif_table(
        candidates=candidates,
        dictionary=pd.DataFrame({"key": ["triangle"]}),
        n_fit_graphs=4,
    )

    row = motifs.iloc[0]
    assert row["forced_net_savings_bits"] == 24.0
    assert row["savings_per_occurrence_bits"] == 3.0
    assert bool(row["is_selected"])


class ForcedAccountingCompressor(FakeCompressor):
    def candidate_frame(self):
        return pd.DataFrame(
            {
                "rank": [1, 2],
                "key": ["triangle", "path"],
                "graph_support": [8, 10],
                "total_occurrences": [20, 30],
                "forced_baseline_bits": [800.0, 1_000.0],
                "forced_dictionary_bits": [12.0, 14.0],
                "forced_template_bits": [100.0, 120.0],
                "forced_boundary_bits": [30.0, 80.0],
                "forced_rewrite_bits": [130.0, 200.0],
                "forced_selector_bits": [7.0, 5.0],
                "forced_model_choice_bits": [1.0, 1.0],
                "forced_encoded_bits": [750.0, 1_040.0],
                "forced_net_savings_bits": [50.0, -40.0],
                "forced_n_rewritten": [6, 1],
                "forced_n_occurrences": [15, 2],
            }
        )

    def dictionary_path_frame(self):
        return pd.DataFrame(
            {
                "n_rules": [0, 1, 2],
                "rule_keys": [(), ("triangle",), ("triangle", "path")],
                "dictionary_bits": [0.0, 12.0, 26.0],
                "encoded_bits": [801.0, 750.0, 790.0],
                "net_savings_bits": [-1.0, 50.0, 10.0],
                "is_empty": [True, False, False],
                "is_best": [False, True, False],
            }
        )


def test_run_analysis_exports_complete_candidate_accounting(tmp_path: Path):
    graphs = [nx.cycle_graph(size) for size in range(4, 16)]
    result = run_mdl_analysis(
        graphs,
        ForcedAccountingCompressor(),
        config=AnalysisConfig(
            fit_size=8,
            eval_size=4,
            seed=7,
            bootstrap_replicates=10,
        ),
        output_dir=tmp_path,
    )

    assert result.summary["n_candidate_motifs"] == 2
    assert result.motifs["forced_dictionary_bits"].gt(0.0).all()
    assert result.motifs["forced_net_savings_bits"].nunique() == 2
    assert set(result.dictionary_path["n_rules"]) == {0, 1, 2}
    assert int(result.dictionary_path["is_best"].sum()) == 1
    exported = pd.read_csv(tmp_path / "motifs.csv")
    assert "forced_net_savings_bits" in exported.columns


def test_build_motif_table_coalesces_occurrence_alias_with_canonical_column():
    candidates = pd.DataFrame(
        {
            "rank": [1],
            "key": ["triangle"],
            "graph_support": [4],
            # Current MDL candidate rows intentionally retain this legacy
            # field alongside the canonical field.
            "occurrences": [8],
            "total_occurrences": [8],
            "forced_dictionary_bits": [12.0],
            "forced_net_savings_bits": [24.0],
        }
    )

    motifs = build_motif_table(
        candidates=candidates,
        dictionary=pd.DataFrame({"key": ["triangle"]}),
        n_fit_graphs=4,
    )

    assert list(motifs.columns).count("total_occurrences") == 1
    assert "occurrences" not in motifs.columns
    assert motifs.loc[0, "total_occurrences"] == 8
    assert motifs.loc[0, "occurrences_per_supported_graph"] == 2.0
    assert motifs.loc[0, "savings_per_occurrence_bits"] == 3.0


class HumanReadableMotifCompressor(ForcedAccountingCompressor):
    def candidate_motif_graphs(self):
        triangle = nx.cycle_graph(3)
        nx.set_node_attributes(triangle, "C", "atom")
        nx.set_edge_attributes(triangle, "single", "bond")

        path = nx.path_graph(3)
        nx.set_node_attributes(path, {0: "C", 1: "N", 2: "C"}, "atom")
        nx.set_edge_attributes(path, "single", "bond")
        return {"triangle": triangle, "path": path}


def test_motif_cost_table_balances_forced_encoded_bits():
    motifs = build_motif_table(
        candidates=ForcedAccountingCompressor().candidate_frame(),
        dictionary=ForcedAccountingCompressor().dictionary_frame(),
        n_fit_graphs=10,
    )
    costs = build_motif_cost_table(motifs)

    assert not costs.empty
    assert costs["accounting_residual_bits"].abs().max() < 1e-9
    totals = costs.groupby("key")["bits"].sum().sort_index()
    encoded = (
        motifs.set_index("key")["forced_encoded_bits"].sort_index()
    )
    pd.testing.assert_series_equal(
        totals,
        encoded,
        check_names=False,
    )


def test_run_analysis_writes_human_readable_motif_assets_and_costs(
    tmp_path: Path,
):
    graphs = [nx.cycle_graph(size) for size in range(4, 16)]
    result = run_mdl_analysis(
        graphs,
        HumanReadableMotifCompressor(),
        config=AnalysisConfig(
            fit_size=8,
            eval_size=4,
            seed=7,
            bootstrap_replicates=10,
        ),
        output_dir=tmp_path,
    )

    assert set(result.motifs["motif_id"]) == {"M001", "M002"}
    assert set(result.motifs["topology_name"]) == {"triangle", "path-P3"}
    assert result.motifs["human_name"].str.contains("nodes").all()
    assert result.motifs["label_pattern"].str.contains("atom").all()
    assert result.motifs["motif_fingerprint"].str.len().eq(16).all()

    assert result.motif_costs["accounting_residual_bits"].abs().max() < 1e-9
    assert set(result.motif_costs["component"]) == {
        "passthrough_baseline",
        "dictionary",
        "rewrite_template",
        "boundary_ports",
        "selector",
        "model_choice",
    }

    assert (tmp_path / "motif_costs.csv").exists()
    assert (tmp_path / "motif_assets.csv").exists()
    assert len(result.motif_assets) == 2
    for asset in result.motif_assets.itertuples(index=False):
        json_path = tmp_path / "motif_assets" / asset.json_path
        graphml_path = tmp_path / "motif_assets" / asset.graphml_path
        assert json_path.exists()
        assert graphml_path.exists()
        exported = nx.read_graphml(graphml_path)
        assert exported.number_of_nodes() == 3

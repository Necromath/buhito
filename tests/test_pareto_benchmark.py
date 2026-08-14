import json
import math
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from buhito.benchmarks.gnn import GNNBenchmarkConfig
from buhito.benchmarks.pareto import (
    ParetoStatisticsConfig,
    aggregate_gnn_pareto_sweep,
    bootstrap_median_interval,
    bootstrap_median_lower_bound,
    create_pareto_task_manifest,
    exact_sign_test_pvalue,
    noninferiority_sign_test_pvalue,
    prepare_gnn_pareto_sweep,
    validate_pareto_artifacts,
    write_pareto_slurm_array_script,
)


def _triangle_chain(repeats: int) -> nx.Graph:
    graph = nx.Graph()
    previous = None
    cursor = 0
    for _ in range(repeats):
        a, b, c = cursor, cursor + 1, cursor + 2
        graph.add_edges_from([(a, b), (b, c), (c, a)])
        if previous is not None:
            graph.add_edge(previous, a)
        previous = c
        cursor += 3
    return graph


def _prepare(tmp_path: Path) -> Path:
    fit_graphs = [_triangle_chain(4), _triangle_chain(5), _triangle_chain(6)]
    eval_graphs = [
        _triangle_chain(3),
        _triangle_chain(4),
        _triangle_chain(5),
        nx.path_graph(9),
        nx.path_graph(10),
        nx.path_graph(11),
    ]
    return prepare_gnn_pareto_sweep(
        fit_graphs,
        eval_graphs,
        labels=[0, 1, 0, 1, 0, 1],
        prepared_dir=tmp_path / "pareto",
        rule_counts=(0, 1, 2),
        compressor_kwargs={
            "graphlet_sizes": (3,),
            "n_rules": 2,
            "min_graph_support": 1,
            "min_occurrences": 1,
            "max_candidates": 10,
            "selector": "sparse",
            "model_choice_bits": 1.0,
            "cache_dir": str(tmp_path / "cache"),
            "validate": True,
            "progress": False,
        },
        compressor_backend="exhaustive",
        token_projection="simple",
        gnn_config=GNNBenchmarkConfig(
            mode="training",
            repeats=1,
            warmup_steps=0,
            epochs=1,
            hidden_channels=4,
            num_layers=2,
            batch_size=3,
            device="cpu",
            threads=1,
            quality_eval_fraction=1.0 / 3.0,
            phase_timeout_seconds=60,
        ),
        statistics_config=ParetoStatisticsConfig(
            bootstrap_samples=200,
            confidence_level=0.95,
            statistics_seed=4,
        ),
        metadata={"dataset": "synthetic"},
    )


def _write_fake_results(prepared: Path) -> None:
    tasks = pd.read_csv(prepared / "task_manifest.csv")
    state = json.loads((prepared / "prepared_state.json").read_text())
    result_dir = prepared / "task_results"
    result_dir.mkdir(exist_ok=True)
    for task in tasks.itertuples(index=False):
        rule_count = int(task.rule_count)
        original = task.representation == "original"
        seconds = 2.0 if original else 2.0 / (1.0 + 0.25 * rule_count)
        accuracy = 0.50 if original else 0.50 - 0.01 * rule_count
        macro_f1 = 0.40 if original else 0.40 - 0.015 * rule_count
        loss = 1.20 if original else 1.20 + 0.02 * rule_count
        row = {
            "task_id": int(task.task_id),
            "rule_count": rule_count,
            "representation": task.representation,
            "repeat": int(task.repeat),
            "model_seed": int(task.model_seed),
            "order_position": int(task.order_position),
            "prepared_fingerprint": state["prepared_fingerprint"],
            "gnn_mode": "training",
            "workload_seconds": seconds,
            "peak_rss_mb": 100.0 - 2.0 * rule_count,
            "cuda_peak_memory_mb": 0.0,
            "final_train_loss": loss,
            "final_train_accuracy": accuracy,
            "quality_eval_loss": loss,
            "quality_eval_accuracy": accuracy,
            "quality_eval_macro_f1": macro_f1,
            "quality_eval_confusion_matrix": [[1, 0], [0, 1]],
            "quality_eval_per_class_metrics": [
                {
                    "class_index": 0,
                    "support": 1,
                    "predicted_count": 1,
                    "true_positive": 1,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                },
                {
                    "class_index": 1,
                    "support": 1,
                    "predicted_count": 1,
                    "true_positive": 1,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                },
            ],
            "quality_metrics_available": True,
        }
        (result_dir / f"task_{int(task.task_id):05d}.json").write_text(
            json.dumps(row) + "\n"
        )


def test_pareto_task_manifest_reuses_one_original_baseline_per_repeat():
    first = create_pareto_task_manifest(
        rule_counts=(0, 1, 2, 3),
        repeats=2,
        seed=10,
        prepared_fingerprint="abc",
    )
    second = create_pareto_task_manifest(
        rule_counts=(0, 1, 2, 3),
        repeats=2,
        seed=10,
        prepared_fingerprint="abc",
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 8
    assert len(first.loc[first["representation"] == "original"]) == 2
    assert not (
        (first["representation"] == "tokenized")
        & (first["rule_count"] == 0)
    ).any()
    assert first["task_id"].tolist() == list(range(8))
    assert set(first["model_seed"]) == {10, 11}


def test_exact_sign_test_and_bootstrap_are_deterministic():
    assert exact_sign_test_pvalue(5, 5) == pytest.approx(0.03125)
    assert exact_sign_test_pvalue(0, 0) is None

    first = bootstrap_median_interval(
        [1.1, 1.2, 1.3, 1.4, 1.5],
        samples=500,
        confidence_level=0.95,
        seed=7,
    )
    second = bootstrap_median_interval(
        [1.1, 1.2, 1.3, 1.4, 1.5],
        samples=500,
        confidence_level=0.95,
        seed=7,
    )
    assert first == second
    assert first[0] == pytest.approx(1.3)
    assert first[1] <= first[0] <= first[2]


def test_noninferiority_uses_one_sided_bound_not_only_the_median():
    ambiguous = [0.01, 0.01, 0.01, -0.10, -0.10]
    estimate = float(pd.Series(ambiguous).median())
    lower = bootstrap_median_lower_bound(
        ambiguous,
        samples=5000,
        confidence_level=0.95,
        seed=13,
    )
    assert estimate >= -0.02
    assert lower is not None
    assert lower < -0.02

    clearly_noninferior = [-0.01] * 5
    lower = bootstrap_median_lower_bound(
        clearly_noninferior,
        samples=500,
        confidence_level=0.95,
        seed=2,
    )
    assert lower == pytest.approx(-0.01)
    assert noninferiority_sign_test_pvalue(
        clearly_noninferior, margin=0.02
    ) == pytest.approx(0.03125)


def test_ci_config_runs_gnn_and_pareto_with_torch_extra():
    text = Path(".gitlab-ci.yml").read_text()
    assert "gnn-pareto-cpu:" in text
    assert 'python -m pip install -e ".[tests,gnn]"' in text
    assert "tests/test_zz_gnn_benchmark.py" in text
    assert "tests/test_pareto_benchmark.py" in text


def test_pareto_prepare_materializes_nested_prefixes_once(tmp_path):
    prepared = _prepare(tmp_path)

    compression = pd.read_csv(prepared / "compression_prefixes.csv")
    assert compression["rule_count"].astype(int).tolist() == [0, 1, 2]
    assert compression["decode_failures"].eq(0).all()
    assert compression.loc[
        compression["rule_count"] == 0, "standalone_preparation_seconds"
    ].iloc[0] == 0.0
    assert compression.loc[
        compression["rule_count"] > 0, "shared_dictionary_fit_seconds"
    ].gt(0.0).all()
    assert compression.loc[
        compression["rule_count"] > 0, "forced_dictionary_bits"
    ].gt(0.0).all()

    sizes = pd.read_csv(prepared / "graph_sizes_by_prefix.csv")
    baseline = sizes.loc[sizes["rule_count"] == 0]
    assert baseline["raw_nodes"].equals(baseline["token_nodes"])
    assert baseline["raw_edges"].equals(baseline["token_edges"])

    tasks = pd.read_csv(prepared / "task_manifest.csv")
    assert set(tasks["rule_count"].astype(int)) == {0, 1, 2}
    assert len(tasks) == 3
    assert (prepared / "candidate_table.csv").is_file()
    assert (prepared / "dictionary_path.csv").is_file()


def test_pareto_tasks_aggregate_to_paper_artifacts(tmp_path):
    prepared = _prepare(tmp_path)
    _write_fake_results(prepared)
    result = aggregate_gnn_pareto_sweep(prepared)
    assert result.points["rule_count"].astype(int).tolist() == [0, 1, 2]
    assert result.points.loc[
        result.points["rule_count"] == 0, "median_paired_speedup"
    ].iloc[0] == pytest.approx(1.0)
    assert result.points["decode_failures"].eq(0).all()
    assert not result.per_class_metrics.empty
    assert not result.confusion_matrices.empty
    assert {
        "pareto_speed_accuracy",
        "pareto_speed_macro_f1",
        "quality_within_tolerance",
        "quality_noninferior",
        "accuracy_noninferiority_lower_bound",
        "macro_f1_noninferiority_lower_bound",
        "recommended_noninferior",
        "timing_sign_test_pvalue",
    }.issubset(result.points.columns)
    points = result.points.set_index("rule_count")
    assert bool(points.loc[1, "quality_noninferior"])
    assert not bool(points.loc[2, "quality_noninferior"])
    assert bool(points.loc[1, "recommended_noninferior"])

    output = result.save(tmp_path / "output")
    expected = {
        "pareto_points.csv",
        "pareto_paired_runs.csv",
        "pareto_statistics.csv",
        "pareto_frontier.csv",
        "paper_table.csv",
        "paper_table.md",
        "paper_table.tex",
        "summary.json",
        "README.md",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    summary = json.loads((output / "summary.json").read_text())
    assert summary["point_count"] == 3
    validated = validate_pareto_artifacts(output)
    assert validated["point_count"] == 3


def test_pareto_aggregation_rejects_missing_task(tmp_path):
    prepared = _prepare(tmp_path)
    _write_fake_results(prepared)
    one_result = sorted((prepared / "task_results").glob("task_*.json"))[0]
    one_result.unlink()
    with pytest.raises(Exception, match="Missing Pareto task results"):
        aggregate_gnn_pareto_sweep(prepared)


def test_pareto_slurm_script_has_stable_array(tmp_path):
    prepared = _prepare(tmp_path)
    script = write_pareto_slurm_array_script(
        prepared,
        tmp_path / "pareto.slurm",
    )
    text = script.read_text()
    assert "#SBATCH --array=0-2" in text
    assert "SITE_GPU_COUNT" in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert script.stat().st_mode & 0o111

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace

import networkx as nx
import pandas as pd
import pytest

from buhito.benchmarks import (
    InsufficientEligibleGraphsError,
    RuntimeBenchmarkConfig,
    SAMPLE_MANIFEST_COLUMNS,
    WorkerTimeoutError,
    benchmark_scalar_summary,
    collapse_parallel_edges,
    filter_graph_complexity,
    graph_complexity_frame,
    relabel_for_enumeration,
    resolve_benchmark_options,
    run_runtime_benchmark,
    select_sample_manifest,
    size_stratified_sample_indices,
)
from buhito.benchmarks.runtime import _run_worker, _write_pickle
from buhito.mdl import _EDGE_LABEL, _NODE_LABEL, MDLGraphCompressor


def _disjoint_triangles(repeats: int) -> nx.Graph:
    graph = nx.Graph()
    for index in range(repeats):
        start = 3 * index
        graph.add_edges_from(
            [
                (start, start + 1),
                (start + 1, start + 2),
                (start + 2, start),
            ]
        )
    return graph


def _load_cli_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "benchmark_reddit_tokenization.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_reddit_tokenization_test_module",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_size_stratified_sample_is_deterministic_and_covers_size_range():
    graphs = [nx.path_graph(size) for size in range(3, 23)]
    first = size_stratified_sample_indices(graphs, 9, seed=7, n_bins=3)
    second = size_stratified_sample_indices(graphs, 9, seed=7, n_bins=3)

    assert first == second
    assert len(first) == 9
    assert len(set(first)) == 9

    sampled_sizes = [graphs[index].number_of_nodes() for index in first]
    assert min(sampled_sizes) <= 7
    assert max(sampled_sizes) >= 18


def test_size_caps_exclude_oversized_graphs():
    graphs = [nx.path_graph(5), nx.path_graph(12), nx.path_graph(20)]
    complexity = graph_complexity_frame(graphs)

    eligible = filter_graph_complexity(
        complexity,
        max_nodes=12,
        max_edges=11,
    )

    assert eligible["dataset_index"].tolist() == [0, 1]
    assert eligible["nodes"].max() <= 12
    assert eligible["edges"].max() <= 11


def test_degree_cap_excludes_hub_heavy_graphs():
    graphs = [nx.path_graph(20), nx.star_graph(20)]
    complexity = graph_complexity_frame(graphs)

    eligible = filter_graph_complexity(complexity, max_degree=5)

    assert eligible["dataset_index"].tolist() == [0]
    assert int(complexity.loc[1, "max_degree"]) == 20


def test_wedge_cap_excludes_computationally_expensive_graphs():
    graphs = [nx.path_graph(20), nx.star_graph(20)]
    complexity = graph_complexity_frame(graphs)
    path_wedges = int(complexity.loc[0, "wedge_proxy"])
    hub_wedges = int(complexity.loc[1, "wedge_proxy"])
    assert hub_wedges > path_wedges

    eligible = filter_graph_complexity(
        complexity,
        max_wedges=path_wedges,
    )

    assert eligible["dataset_index"].tolist() == [0]


def test_sampling_remains_deterministic_after_filtering():
    graphs = [nx.path_graph(size) for size in range(3, 33)]

    first, first_fit, first_eval = select_sample_manifest(
        graphs,
        dataset="synthetic",
        fit_size=4,
        eval_size=3,
        seed=11,
        n_bins=3,
        max_nodes=25,
        max_edges=24,
        max_degree=4,
        max_wedges=100,
    )
    second, second_fit, second_eval = select_sample_manifest(
        graphs,
        dataset="synthetic",
        fit_size=4,
        eval_size=3,
        seed=11,
        n_bins=3,
        max_nodes=25,
        max_edges=24,
        max_degree=4,
        max_wedges=100,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first_fit == second_fit
    assert first_eval == second_eval


def test_insufficient_eligible_graphs_raises_clear_exception():
    graphs = [nx.path_graph(5), nx.path_graph(50), nx.star_graph(30)]

    with pytest.raises(InsufficientEligibleGraphsError) as error:
        select_sample_manifest(
            graphs,
            dataset="synthetic",
            fit_size=2,
            eval_size=1,
            seed=0,
            n_bins=1,
            max_nodes=10,
            max_edges=10,
            max_degree=5,
            max_wedges=20,
        )

    message = str(error.value)
    assert "Requested sample size: 3" in message
    assert "Eligible graph count: 1" in message
    assert "max_nodes=10" in message
    assert "max_edges=10" in message
    assert "max_max_degree=5" not in message
    assert "max_degree=5" in message
    assert "max_wedge_proxy=20" not in message
    assert "max_wedges=20" in message
    assert "Dataset-wide ranges:" in message
    assert "Eligible ranges:" in message


def test_sample_manifest_contains_all_required_fields():
    graphs = [nx.path_graph(size) for size in range(4, 14)]
    manifest, _, _ = select_sample_manifest(
        graphs,
        dataset="synthetic",
        fit_size=3,
        eval_size=2,
        seed=3,
        n_bins=2,
        max_nodes=20,
    )

    assert tuple(manifest.columns) == SAMPLE_MANIFEST_COLUMNS
    assert len(manifest) == 5
    assert set(manifest["split"]) == {"fit", "eval"}
    assert manifest["eligible_graph_count"].nunique() == 1
    assert manifest["dataset_graph_count"].nunique() == 1


def test_smoke_preset_resolves_to_safe_values():
    values = {name: None for name in (
        "fit_size",
        "eval_size",
        "size_bins",
        "max_nodes",
        "max_edges",
        "max_degree",
        "max_wedges",
        "n_rules",
        "max_candidates",
        "repeats",
        "warmup_repeats",
        "threads",
        "phase_timeout_seconds",
    )}

    resolved = resolve_benchmark_options(values, smoke=True)

    assert resolved["fit_size"] == 2
    assert resolved["eval_size"] == 2
    assert resolved["size_bins"] == 1
    assert resolved["max_nodes"] == 500
    assert resolved["max_edges"] == 750
    assert resolved["max_degree"] == 200
    assert resolved["max_wedges"] == 50000
    assert resolved["n_rules"] == 1
    assert resolved["max_candidates"] == 2
    assert resolved["repeats"] == 1
    assert resolved["warmup_repeats"] == 0
    assert resolved["threads"] == 1
    assert resolved["phase_timeout_seconds"] == 600.0


def test_explicit_cli_values_override_smoke_defaults():
    resolved = resolve_benchmark_options(
        {
            "fit_size": 7,
            "eval_size": None,
            "size_bins": 2,
            "max_nodes": 300,
            "max_edges": None,
            "max_degree": None,
            "max_wedges": 1234,
            "n_rules": 4,
            "max_candidates": None,
            "repeats": 2,
            "warmup_repeats": 1,
            "threads": 3,
            "phase_timeout_seconds": 45.0,
        },
        smoke=True,
    )

    assert resolved["fit_size"] == 7
    assert resolved["eval_size"] == 2
    assert resolved["size_bins"] == 2
    assert resolved["max_nodes"] == 300
    assert resolved["max_edges"] == 750
    assert resolved["max_wedges"] == 1234
    assert resolved["n_rules"] == 4
    assert resolved["repeats"] == 2
    assert resolved["warmup_repeats"] == 1
    assert resolved["threads"] == 3
    assert resolved["phase_timeout_seconds"] == 45.0


def test_print_sample_only_does_not_fit_or_launch_worker(tmp_path, monkeypatch):
    module = _load_cli_module()
    graphs = [nx.path_graph(size) for size in (8, 9, 10, 11, 12, 13)]
    dataset = SimpleNamespace(
        graphs=graphs,
        node_label_key=None,
        edge_label_key=None,
    )
    monkeypatch.setattr(module, "load_tu_dataset", lambda *args, **kwargs: dataset)

    def fail_benchmark(*args, **kwargs):
        raise AssertionError("preview launched a benchmark worker")

    def fail_fit(*args, **kwargs):
        raise AssertionError("preview called compressor.fit")

    monkeypatch.setattr(module, "run_runtime_benchmark", fail_benchmark)
    monkeypatch.setattr(MDLGraphCompressor, "fit", fail_fit)
    args = module.parse_args(
        [
            "--data-root",
            str(tmp_path),
            "--smoke",
            "--print-sample-only",
            "--output-dir",
            str(tmp_path / "preview"),
        ]
    )

    assert module.run(args) == 0
    assert (tmp_path / "preview" / "sample_manifest.csv").is_file()
    assert (tmp_path / "preview" / "sample_manifest.json").is_file()


def test_simple_projection_preserves_parallel_label_multiset():
    graph = nx.MultiGraph()
    graph.add_node(0, **{_NODE_LABEL: ("data", (0,))})
    graph.add_node(1, **{_NODE_LABEL: ("motif", "M001")})
    graph.add_edge(0, 1, **{_EDGE_LABEL: ("data", ("single",))})
    graph.add_edge(0, 1, **{_EDGE_LABEL: ("data", ("double",))})

    projected = collapse_parallel_edges(graph)

    assert not projected.is_multigraph()
    assert projected.number_of_nodes() == 2
    assert projected.number_of_edges() == 1
    label = projected.edges[0, 1][_EDGE_LABEL]
    assert label[0] == "buhito-token-edge-multiset"
    assert len(label[1]) == 2
    assert set(label[1]) == {
        ("data", ("single",)),
        ("data", ("double",)),
    }


def test_relabel_for_enumeration_handles_mixed_node_identifier_types():
    graph = nx.Graph()
    graph.add_edge(0, ("__motif__", 1))
    nx.set_node_attributes(graph, ("data", (0,)), _NODE_LABEL)
    nx.set_edge_attributes(graph, ("data", (0,)), _EDGE_LABEL)

    relabeled = relabel_for_enumeration(graph)

    assert set(relabeled.nodes()) == {0, 1}
    assert relabeled.number_of_edges() == 1
    assert all(_NODE_LABEL in data for _, data in relabeled.nodes(data=True))


def test_runtime_benchmark_config_rejects_invalid_repeat_count():
    with pytest.raises(ValueError, match="repeats"):
        RuntimeBenchmarkConfig(repeats=0).validate()


def test_timeout_kills_intentionally_sleeping_worker(tmp_path):
    payload = tmp_path / "sleep.pkl"
    result = tmp_path / "sleep.json"
    pid_path = tmp_path / "sleep.pid"
    _write_pickle(
        payload,
        {
            "sleep_seconds": 30.0,
            "pid_path": str(pid_path),
        },
    )

    with pytest.raises(WorkerTimeoutError, match="sleep timeout phase"):
        _run_worker(
            action="sleep",
            payload_path=payload,
            result_path=result,
            graph_output_path=None,
            threads=1,
            phase_timeout_seconds=2.0,
            phase_name="sleep timeout phase",
        )

    assert pid_path.is_file()
    pid = int(pid_path.read_text().strip())
    deadline = time.time() + 3.0
    still_alive = True
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            still_alive = False
            break
        time.sleep(0.05)
    assert not still_alive, f"timed-out worker PID {pid} is still alive"


def test_timeout_subprocess_exits_nonzero_and_names_failed_phase(tmp_path):
    payload = tmp_path / "sleep.pkl"
    result = tmp_path / "sleep.json"
    with payload.open("wb") as handle:
        pickle.dump({"sleep_seconds": 30.0}, handle)

    code = f"""
from pathlib import Path
from buhito.benchmarks.runtime import _run_worker
_run_worker(
    action='sleep',
    payload_path=Path({str(payload)!r}),
    result_path=Path({str(result)!r}),
    graph_output_path=None,
    threads=1,
    phase_timeout_seconds=0.2,
    phase_name='named failed phase',
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode != 0
    assert "named failed phase" in completed.stderr


def test_isolated_runtime_benchmark_writes_complete_artifacts(tmp_path):
    fit_graphs = [
        _disjoint_triangles(5),
        _disjoint_triangles(6),
        _disjoint_triangles(7),
    ]
    eval_graphs = [
        _disjoint_triangles(4),
        _disjoint_triangles(8),
    ]
    config = RuntimeBenchmarkConfig(
        graphlet_sizes=(3,),
        backend="exhaustive",
        compressor_backend="exhaustive",
        repeats=1,
        warmup_repeats=0,
        token_projection="simple",
        force_rewrite=True,
        threads=1,
        phase_timeout_seconds=30,
        progress=False,
    )
    result = run_runtime_benchmark(
        fit_graphs,
        eval_graphs,
        compressor_kwargs={
            "graphlet_sizes": (3,),
            "n_rules": 1,
            "min_graph_support": 1,
            "min_occurrences": 1,
            "max_candidates": 5,
            "selector": "sparse",
            "model_choice_bits": 0.0,
            "min_rule_savings_bits": -math.inf,
            "dictionary_selection": "fixed",
            "validate": True,
        },
        config=config,
        metadata={"dataset": "synthetic-triangles"},
    )

    assert set(result.runs["representation"]) == {"original", "tokenized"}
    assert len(result.runs) == 2
    assert result.runs["enumeration_seconds"].gt(0.0).all()
    assert result.runs["peak_rss_mb"].gt(0.0).all()
    assert result.compression["selected_rule_count"] == 1
    assert result.compression["tokenized_graph_count"] == 2
    assert result.compression["decode_failures"] == 0
    assert result.graph_sizes["token_nodes"].lt(
        result.graph_sizes["raw_nodes"]
    ).all()

    output = result.save(tmp_path / "runtime")
    expected = {
        "benchmark_runs.csv",
        "benchmark_summary.csv",
        "graph_size_comparison.csv",
        "compression_summary.json",
        "summary.json",
        "metadata.json",
        "README.md",
        "sample_manifest.csv",
        "sample_manifest.json",
    }
    assert expected.issubset({path.name for path in output.iterdir()})

    summary = json.loads((output / "summary.json").read_text())
    assert summary["decode_failures"] == 0
    assert summary["selected_rule_count"] == 1
    assert summary["node_reduction_fraction"] > 0.0

    runs = pd.read_csv(output / "benchmark_runs.csv")
    assert set(runs["representation"]) == {"original", "tokenized"}
    manifest = pd.read_csv(output / "sample_manifest.csv")
    assert set(SAMPLE_MANIFEST_COLUMNS).issubset(manifest.columns)

    headline = benchmark_scalar_summary(result)
    assert headline["force_rewrite"] is True
    assert headline["token_projection"] == "simple"


def test_selected_mode_can_legitimately_leave_graphs_unchanged():
    fit_graphs = [nx.path_graph(6), nx.path_graph(7), nx.path_graph(8)]
    eval_graphs = [nx.path_graph(5)]
    result = run_runtime_benchmark(
        fit_graphs,
        eval_graphs,
        compressor_kwargs={
            "graphlet_sizes": (3,),
            "n_rules": 1,
            "min_graph_support": 1,
            "min_occurrences": 1,
            "max_candidates": 5,
            "selector": "sparse",
            "min_rule_savings_bits": 0.0,
            "dictionary_selection": "best",
            "validate": True,
        },
        config=RuntimeBenchmarkConfig(
            graphlet_sizes=(3,),
            backend="exhaustive",
            compressor_backend="exhaustive",
            repeats=1,
            warmup_repeats=0,
            token_projection="simple",
            force_rewrite=False,
            phase_timeout_seconds=30,
            progress=False,
        ),
    )

    assert result.compression["decode_failures"] == 0
    assert result.compression["selected_rule_count"] == 0
    assert result.compression["tokenized_graph_count"] == 0
    assert (
        result.graph_sizes["raw_nodes"].tolist()
        == result.graph_sizes["token_nodes"].tolist()
    )


def _prepare_small_runtime_state(tmp_path, *, repeats=2):
    from buhito.benchmarks import prepare_runtime_benchmark

    fit_graphs = [_disjoint_triangles(4), _disjoint_triangles(5)]
    eval_graphs = [_disjoint_triangles(3), _disjoint_triangles(6)]
    prepared = tmp_path / "prepared"
    prepare_runtime_benchmark(
        fit_graphs,
        eval_graphs,
        prepared_dir=prepared,
        compressor_kwargs={
            "graphlet_sizes": (3,),
            "n_rules": 1,
            "min_graph_support": 1,
            "min_occurrences": 1,
            "max_candidates": 3,
            "selector": "sparse",
            "model_choice_bits": 0.0,
            "min_rule_savings_bits": -math.inf,
            "dictionary_selection": "fixed",
            "validate": True,
        },
        config=RuntimeBenchmarkConfig(
            graphlet_sizes=(3,),
            backend="exhaustive",
            compressor_backend="exhaustive",
            repeats=repeats,
            warmup_repeats=0,
            token_projection="simple",
            force_rewrite=True,
            threads=1,
            phase_timeout_seconds=30,
            progress=False,
        ),
        metadata={"dataset": "synthetic-hpc"},
    )
    return prepared


def test_task_manifest_ids_are_stable_and_slurm_id_resolves(monkeypatch):
    from buhito.benchmarks import create_task_manifest, resolve_task_id

    first = create_task_manifest(repeats=3, prepared_fingerprint="abc")
    second = create_task_manifest(repeats=3, prepared_fingerprint="abc")
    pd.testing.assert_frame_equal(first, second)
    assert first["task_id"].tolist() == list(range(6))
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "4")
    assert resolve_task_id() == 4
    assert resolve_task_id(2) == 2


def test_local_jobs_one_and_two_produce_compatible_results(tmp_path):
    from buhito.benchmarks import (
        aggregate_prepared_results,
        run_prepared_tasks,
    )

    prepared = _prepare_small_runtime_state(tmp_path)
    serial_dir = tmp_path / "serial-results"
    parallel_dir = tmp_path / "parallel-results"
    run_prepared_tasks(prepared, jobs=1, results_dir=serial_dir)
    run_prepared_tasks(prepared, jobs=2, results_dir=parallel_dir)

    serial = aggregate_prepared_results(prepared, results_dir=serial_dir)
    parallel = aggregate_prepared_results(prepared, results_dir=parallel_dir)
    stable = [
        "task_id",
        "representation",
        "repeat",
        "order_position",
        "total_nodes",
        "total_edges",
        "total_occurrences",
        "unique_graphlet_keys",
    ]
    pd.testing.assert_frame_equal(
        serial.runs[stable].reset_index(drop=True),
        parallel.runs[stable].reset_index(drop=True),
    )
    assert not list(parallel_dir.glob("*.tmp-*"))


def test_aggregation_detects_missing_duplicate_and_incompatible_results(tmp_path):
    import shutil

    from buhito.benchmarks import (
        PreparedStateError,
        aggregate_prepared_results,
        run_prepared_tasks,
    )

    prepared = _prepare_small_runtime_state(tmp_path, repeats=1)
    results = tmp_path / "results"
    run_prepared_tasks(prepared, jobs=1, results_dir=results)

    missing_file = results / "task_00001.json"
    saved = missing_file.read_text()
    missing_file.unlink()
    with pytest.raises(PreparedStateError, match="Missing task results"):
        aggregate_prepared_results(prepared, results_dir=results)
    missing_file.write_text(saved)

    shutil.copy2(results / "task_00000.json", results / "task_duplicate.json")
    with pytest.raises(PreparedStateError, match="Duplicate task result"):
        aggregate_prepared_results(prepared, results_dir=results)
    (results / "task_duplicate.json").unlink()

    row = json.loads((results / "task_00000.json").read_text())
    row["prepared_fingerprint"] = "wrong"
    (results / "task_00000.json").write_text(json.dumps(row))
    with pytest.raises(PreparedStateError, match="incompatible"):
        aggregate_prepared_results(prepared, results_dir=results)


def test_slurm_script_has_stable_array_and_site_placeholders(tmp_path):
    from buhito.benchmarks import write_slurm_array_script

    prepared = _prepare_small_runtime_state(tmp_path, repeats=2)
    script = write_slurm_array_script(prepared, tmp_path / "submit.slurm")
    text = script.read_text()
    assert "#SBATCH --array=0-3" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "SITE_WALLTIME" in text
    assert "SITE_MEMORY" in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert script.stat().st_mode & 0o111

import importlib.util
import json
import math
from pathlib import Path
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from buhito.benchmarks.gnn import (
    GNNBenchmarkConfig,
    aggregate_gnn_results,
    gnn_scalar_summary,
    prepare_gnn_benchmark,
    run_gnn_prepared_tasks,
    structural_node_features,
    workload_tradeoff_summary,
    write_gnn_slurm_array_script,
)
from buhito.benchmarks.runtime import _sha256_file


def _write_pickle(path: Path, value):
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _fake_runtime_state(tmp_path: Path) -> Path:
    root = tmp_path / "runtime-prepared"
    root.mkdir()
    original = [nx.path_graph(8), nx.cycle_graph(10), nx.star_graph(7)]
    tokenized = [nx.path_graph(5), nx.cycle_graph(6), nx.star_graph(4)]
    tokenized[0].nodes[0]["__buhito_mdl_kind__"] = "motif"
    tokenized[0].nodes[0]["__buhito_mdl_node_label__"] = ("motif", 2)
    _write_pickle(
        root / "original_payload.pkl",
        {"graphs": original, "graphlet_sizes": (3,), "backend": "exhaustive"},
    )
    _write_pickle(
        root / "tokenized_payload.pkl",
        {"graphs": tokenized, "graphlet_sizes": (3,), "backend": "exhaustive"},
    )
    state = {
        "prepared_fingerprint": "runtime-fingerprint",
        "original_payload_sha256": _sha256_file(root / "original_payload.pkl"),
        "tokenized_payload_sha256": _sha256_file(root / "tokenized_payload.pkl"),
    }
    (root / "prepared_state.json").write_text(json.dumps(state))
    (root / "compression_summary.json").write_text(
        json.dumps(
            {
                "compression_total_seconds": 0.25,
                "mdl_net_savings_bits": -10.0,
                "selected_rule_count": 1,
                "tokenized_graph_count": 3,
                "decode_failures": 0,
            }
        )
    )
    pd.DataFrame(
        {
            "graph_index": [0, 1, 2],
            "raw_nodes": [8, 10, 8],
            "raw_edges": [7, 10, 7],
            "token_nodes": [5, 6, 5],
            "token_edges": [4, 6, 4],
        }
    ).to_csv(root / "graph_size_comparison.csv", index=False)
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic",
                "eval_indices": [0, 1, 2],
                "compressor_kwargs": {
                    "min_rule_savings_bits": -math.inf,
                },
                "legacy_unavailable_metric": math.nan,
            }
        )
    )
    return root


def test_structural_features_have_fixed_width_and_motif_indicator():
    graph = nx.path_graph(4)
    graph.nodes[1]["__buhito_mdl_kind__"] = "motif"
    graph.nodes[1]["__buhito_mdl_node_label__"] = ("motif", 3)

    features = structural_node_features(graph)

    assert features.shape == (4, 5)
    assert np.all(features[:, 0] == 1.0)
    assert features[1, 3] == 1.0
    assert features[0, 3] == 0.0
    assert features[1, 4] > 0.0


def test_framework_neutral_tradeoff_summary():
    summary = workload_tradeoff_summary(
        original_seconds=10.0,
        tokenized_seconds=6.0,
        compression_seconds=20.0,
    )

    assert summary["speedup"] == pytest.approx(10.0 / 6.0)
    assert summary["time_saved_seconds_per_use"] == pytest.approx(4.0)
    assert summary["break_even_reuses"] == pytest.approx(5.0)


def test_gnn_inference_tasks_aggregate_and_save(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("optional PyTorch dependency is unavailable")
    runtime = _fake_runtime_state(tmp_path)
    prepared = tmp_path / "gnn-prepared"
    prepare_gnn_benchmark(
        runtime,
        prepared_dir=prepared,
        config=GNNBenchmarkConfig(
            mode="inference",
            repeats=1,
            warmup_steps=0,
            steps_per_repeat=1,
            hidden_channels=8,
            num_layers=2,
            batch_size=2,
            device="cpu",
            threads=1,
            phase_timeout_seconds=30,
        ),
    )

    outputs = run_gnn_prepared_tasks(prepared, jobs=1)
    assert len(outputs) == 2
    result = aggregate_gnn_results(prepared)
    assert set(result.runs["representation"]) == {"original", "tokenized"}
    assert result.runs["workload_seconds"].gt(0).all()
    assert result.runs["peak_rss_mb"].gt(0).all()
    headline = gnn_scalar_summary(result)
    assert headline["node_reduction_fraction"] > 0
    assert headline["mdl_net_savings_bits"] < 0
    assert "gnn_speedup" in headline
    assert headline["quality_metrics_available"] is False
    assert headline["raw_median_accuracy"] is None
    assert headline["tokenized_median_accuracy"] is None
    assert headline["accuracy_delta"] is None

    output = result.save(tmp_path / "gnn-output")
    expected = {
        "gnn_runs.csv",
        "gnn_summary.csv",
        "summary.json",
        "metadata.json",
        "compression_summary.json",
        "README.md",
    }
    assert expected.issubset({path.name for path in output.iterdir()})

    summary_text = (output / "summary.json").read_text()
    assert "NaN" not in summary_text
    strict_summary = json.loads(
        summary_text,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ),
    )
    assert strict_summary["accuracy_delta"] is None

    metadata_text = (output / "metadata.json").read_text()
    assert "-Infinity" in metadata_text
    assert "NaN" not in metadata_text
    strict_metadata = json.loads(
        metadata_text,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ),
    )
    assert (
        strict_metadata["compressor_kwargs"]["min_rule_savings_bits"]
        == "-Infinity"
    )
    assert strict_metadata["legacy_unavailable_metric"] is None


def test_training_requires_labels_and_runs_with_labels(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("optional PyTorch dependency is unavailable")
    runtime = _fake_runtime_state(tmp_path)
    with pytest.raises(ValueError, match="requires graph labels"):
        prepare_gnn_benchmark(
            runtime,
            prepared_dir=tmp_path / "missing-labels",
            config=GNNBenchmarkConfig(
                mode="training",
                repeats=1,
                warmup_steps=0,
                epochs=1,
                hidden_channels=4,
                batch_size=3,
                device="cpu",
                phase_timeout_seconds=30,
            ),
        )

    prepared = tmp_path / "training"
    prepare_gnn_benchmark(
        runtime,
        prepared_dir=prepared,
        labels=[0, 1, 0],
        config=GNNBenchmarkConfig(
            mode="training",
            repeats=1,
            warmup_steps=0,
            epochs=1,
            hidden_channels=4,
            batch_size=3,
            device="cpu",
            threads=1,
            phase_timeout_seconds=30,
        ),
    )
    run_gnn_prepared_tasks(prepared, jobs=1)
    result = aggregate_gnn_results(prepared)
    assert result.runs["final_loss"].notna().all()
    assert result.runs["accuracy"].between(0.0, 1.0).all()
    assert result.runs["quality_eval_loss"].notna().all()
    assert result.runs["quality_eval_accuracy"].between(0.0, 1.0).all()
    assert result.runs["quality_eval_macro_f1"].between(0.0, 1.0).all()
    assert result.runs["quality_metrics_available"].all()

    headline = gnn_scalar_summary(result)
    assert headline["quality_metrics_available"] is True
    assert headline["quality_protocol"] == (
        "paired-representation-specific-held-out"
    )
    assert headline["train_graph_count"] == 2
    assert headline["quality_eval_graph_count"] == 1
    assert headline["raw_median_quality_eval_accuracy"] is not None
    assert headline["tokenized_median_quality_eval_accuracy"] is not None
    assert headline["quality_eval_accuracy_delta"] is not None

    output = result.save(tmp_path / "training-output")
    summary_text = (output / "summary.json").read_text()
    assert "NaN" not in summary_text
    json.loads(
        summary_text,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ),
    )


def test_gnn_slurm_script_has_stable_array_and_gpu_placeholder(tmp_path):
    runtime = _fake_runtime_state(tmp_path)
    prepared = tmp_path / "gnn-prepared"
    prepare_gnn_benchmark(
        runtime,
        prepared_dir=prepared,
        config=GNNBenchmarkConfig(
            repeats=2,
            warmup_steps=0,
            steps_per_repeat=1,
            hidden_channels=4,
            batch_size=2,
            device="cpu",
        ),
    )

    script = write_gnn_slurm_array_script(
        prepared,
        tmp_path / "submit_gnn.slurm",
    )
    text = script.read_text()
    assert "#SBATCH --array=0-3" in text
    assert "SITE_GPU_COUNT" in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert script.stat().st_mode & 0o111


def test_training_uses_paired_deterministic_holdout_indices(tmp_path):
    runtime = _fake_runtime_state(tmp_path)
    prepared = tmp_path / "paired-holdout"
    prepare_gnn_benchmark(
        runtime,
        prepared_dir=prepared,
        labels=[0, 1, 0],
        config=GNNBenchmarkConfig(
            mode="training",
            repeats=1,
            warmup_steps=0,
            epochs=1,
            hidden_channels=4,
            batch_size=2,
            device="cpu",
            threads=1,
            quality_eval_fraction=1.0 / 3.0,
            phase_timeout_seconds=30,
        ),
    )

    metadata = json.loads((prepared / "metadata.json").read_text())
    assert metadata["quality_protocol"] == (
        "paired-representation-specific-held-out"
    )
    assert len(metadata["train_indices"]) == 2
    assert len(metadata["quality_eval_indices"]) == 1
    assert set(metadata["train_indices"]).isdisjoint(
        metadata["quality_eval_indices"]
    )

    with (prepared / "original_gnn_payload.pkl").open("rb") as handle:
        original = pickle.load(handle)
    with (prepared / "tokenized_gnn_payload.pkl").open("rb") as handle:
        tokenized = pickle.load(handle)

    assert original["train_indices"] == tokenized["train_indices"]
    assert original["quality_eval_indices"] == tokenized[
        "quality_eval_indices"
    ]
    assert sum(batch["graphs"] for batch in original["train_batches"]) == 2
    assert sum(batch["graphs"] for batch in original["quality_batches"]) == 1


def test_inference_checksum_is_finite_for_extreme_finite_logits():
    import math

    import torch

    from buhito.benchmarks._gnn_worker import _stable_output_checksum

    logits = torch.tensor(
        [[3.0e38, -3.0e38], [3.0e38, -3.0e38]],
        dtype=torch.float32,
    )

    # A direct float32 reduction can overflow, but the diagnostic checksum
    # must remain finite because all model outputs are themselves finite.
    checksum = _stable_output_checksum(torch, logits)

    assert math.isfinite(checksum)


def test_inference_checksum_rejects_nonfinite_logits():
    import pytest
    import torch

    from buhito.benchmarks._gnn_worker import _stable_output_checksum

    logits = torch.tensor([[0.0, float("-inf")]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="non-finite logits"):
        _stable_output_checksum(torch, logits)


def test_classification_diagnostics_report_confusion_and_per_class_metrics():
    from buhito.benchmarks._gnn_worker import _classification_diagnostics

    matrix, rows = _classification_diagnostics(
        predictions=[0, 1, 1],
        targets=[0, 0, 1],
        num_classes=2,
    )

    assert matrix == [[1, 1], [0, 1]]
    assert rows[0]["support"] == 2
    assert rows[0]["predicted_count"] == 1
    assert rows[0]["precision"] == pytest.approx(1.0)
    assert rows[0]["recall"] == pytest.approx(0.5)
    assert rows[1]["support"] == 1
    assert rows[1]["predicted_count"] == 2
    assert rows[1]["precision"] == pytest.approx(0.5)
    assert rows[1]["recall"] == pytest.approx(1.0)

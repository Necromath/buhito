"""Compression--speed--quality Pareto sweeps for Buhito rule prefixes.

The sweep fits the motif dictionary once, materializes deterministic prefixes,
and compares each tokenized representation with one shared original-graph GNN
baseline under paired seeds and a shared held-out split.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from .gnn import (
    GNNBenchmarkConfig,
    _contiguous_labels,
    _quality_split_indices,
    _run_gnn_worker,
    _strict_json_text,
    prepare_gnn_batches,
)
from .runtime import (
    PreparedStateError,
    WorkerTimeoutError,
    _atomic_pickle,
    _read_pickle,
    _sha256_file,
    _subprocess_environment,
    _terminate_process_group,
    resolve_task_id,
)


PARETO_TASK_COLUMNS = (
    "task_id",
    "rule_count",
    "representation",
    "repeat",
    "model_seed",
    "order_position",
    "payload_file",
    "prepared_fingerprint",
)


@dataclass(frozen=True)
class ParetoStatisticsConfig:
    """Statistical and reporting controls for a Pareto sweep."""

    bootstrap_samples: int = 2000
    confidence_level: float = 0.95
    statistics_seed: int = 0
    accuracy_drop_tolerance: float = 0.02
    macro_f1_drop_tolerance: float = 0.02

    def validate(self) -> None:
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be strictly between 0 and 1.")
        if self.accuracy_drop_tolerance < 0:
            raise ValueError("accuracy_drop_tolerance cannot be negative.")
        if self.macro_f1_drop_tolerance < 0:
            raise ValueError("macro_f1_drop_tolerance cannot be negative.")


@dataclass
class ParetoSweepResult:
    """Aggregated rule-prefix sweep artifacts."""

    paired_runs: pd.DataFrame
    points: pd.DataFrame
    statistics: pd.DataFrame
    frontier: pd.DataFrame
    compression_points: pd.DataFrame
    graph_sizes: pd.DataFrame
    per_class_metrics: pd.DataFrame
    confusion_matrices: pd.DataFrame
    metadata: dict[str, Any]

    def save(self, output_dir: str | Path, *, plots: bool = False) -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.paired_runs.to_csv(output / "pareto_paired_runs.csv", index=False)
        self.points.to_csv(output / "pareto_points.csv", index=False)
        self.statistics.to_csv(output / "pareto_statistics.csv", index=False)
        self.frontier.to_csv(output / "pareto_frontier.csv", index=False)
        self.compression_points.to_csv(
            output / "compression_prefixes.csv", index=False
        )
        self.graph_sizes.to_csv(output / "graph_sizes_by_prefix.csv", index=False)
        self.per_class_metrics.to_csv(
            output / "pareto_per_class_metrics.csv", index=False
        )
        self.confusion_matrices.to_csv(
            output / "pareto_confusion_matrices.csv", index=False
        )
        (output / "metadata.json").write_text(_strict_json_text(self.metadata))
        scalar = pareto_scalar_summary(self)
        (output / "summary.json").write_text(_strict_json_text(scalar))
        (output / "README.md").write_text(render_pareto_summary(self))
        _write_paper_tables(self.points, output)
        if plots:
            save_pareto_plots(self, output)
        return output


def _paths(prepared_dir: str | Path) -> dict[str, Path]:
    root = Path(prepared_dir)
    return {
        "root": root,
        "state": root / "prepared_state.json",
        "config": root / "gnn_config.json",
        "stats_config": root / "statistics_config.json",
        "metadata": root / "metadata.json",
        "compression_csv": root / "compression_prefixes.csv",
        "compression_json": root / "compression_prefixes.json",
        "candidates": root / "candidate_table.csv",
        "dictionary_path": root / "dictionary_path.csv",
        "graph_sizes": root / "graph_sizes_by_prefix.csv",
        "original": root / "original_gnn_payload.pkl",
        "tasks_csv": root / "task_manifest.csv",
        "tasks_json": root / "task_manifest.json",
        "results": root / "task_results",
    }


def _token_payload_path(root: Path, rule_count: int) -> Path:
    return root / f"tokenized_rule_{rule_count:03d}_gnn_payload.pkl"


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_strict_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(_strict_json_text(value))
    os.replace(temporary, path)


def create_pareto_task_manifest(
    *,
    rule_counts: Sequence[int],
    repeats: int,
    seed: int,
    prepared_fingerprint: str,
) -> pd.DataFrame:
    """Create stable tasks with one shared original baseline per repeat."""

    counts = tuple(sorted({int(value) for value in rule_counts}))
    if not counts or counts[0] != 0:
        raise ValueError("rule_counts must include zero.")
    if repeats < 1:
        raise ValueError("repeats must be at least one.")
    positive = [value for value in counts if value > 0]
    rows: list[dict[str, Any]] = []
    task_id = 0
    for repeat in range(repeats):
        entries: list[tuple[int, str, str]] = [
            (0, "original", "original_gnn_payload.pkl")
        ] + [
            (
                count,
                "tokenized",
                f"tokenized_rule_{count:03d}_gnn_payload.pkl",
            )
            for count in positive
        ]
        if repeat % 2:
            entries = list(reversed(entries))
        for order_position, (count, representation, payload_file) in enumerate(
            entries
        ):
            rows.append(
                {
                    "task_id": task_id,
                    "rule_count": int(count),
                    "representation": representation,
                    "repeat": int(repeat),
                    "model_seed": int(seed) + int(repeat),
                    "order_position": int(order_position),
                    "payload_file": payload_file,
                    "prepared_fingerprint": prepared_fingerprint,
                }
            )
            task_id += 1
    return pd.DataFrame(rows, columns=PARETO_TASK_COLUMNS)


def load_pareto_task_manifest(
    prepared_dir: str | Path,
    task_manifest: str | Path | None = None,
) -> pd.DataFrame:
    path = Path(task_manifest) if task_manifest else _paths(prepared_dir)["tasks_csv"]
    if not path.is_file():
        raise PreparedStateError(f"Pareto task manifest is missing: {path}")
    tasks = pd.read_csv(path)
    missing = [name for name in PARETO_TASK_COLUMNS if name not in tasks]
    if missing:
        raise PreparedStateError(
            "Pareto task manifest is missing columns: " + ", ".join(missing)
        )
    if tasks["task_id"].duplicated().any():
        duplicates = tasks.loc[tasks["task_id"].duplicated(), "task_id"].tolist()
        raise PreparedStateError(f"Duplicate Pareto task IDs: {duplicates}")
    return tasks.loc[:, PARETO_TASK_COLUMNS].copy()


def _run_prefix_worker(
    *,
    payload_path: Path,
    result_path: Path,
    graph_output_path: Path,
    threads: int,
    timeout: float | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "buhito.benchmarks._pareto_worker",
        "--payload",
        str(payload_path),
        "--result",
        str(result_path),
        "--graph-output",
        str(graph_output_path),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_subprocess_environment(threads),
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        stdout, stderr = process.communicate()
        raise WorkerTimeoutError(
            "Pareto dictionary-prefix preparation exceeded its configured "
            f"timeout ({timeout} seconds). The worker group was terminated."
        ) from exc
    if process.returncode != 0:
        raise RuntimeError(
            "Pareto dictionary-prefix preparation failed with exit code "
            f"{process.returncode}.\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    if not result_path.is_file() or not graph_output_path.is_file():
        raise RuntimeError("Pareto prefix worker did not write all outputs.")
    return json.loads(result_path.read_text())


def prepare_gnn_pareto_sweep(
    fit_graphs: Iterable[nx.Graph],
    eval_graphs: Iterable[nx.Graph],
    labels: Sequence[Any],
    *,
    prepared_dir: str | Path,
    rule_counts: Sequence[int],
    compressor_kwargs: Mapping[str, Any],
    compressor_backend: str = "buhito",
    token_projection: str = "simple",
    gnn_config: GNNBenchmarkConfig | None = None,
    statistics_config: ParetoStatisticsConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Fit once, materialize exact prefixes, and prepare paired GNN tasks."""

    active_gnn = gnn_config or GNNBenchmarkConfig(mode="training")
    active_gnn.validate()
    active_stats = statistics_config or ParetoStatisticsConfig()
    active_stats.validate()
    counts = tuple(sorted({int(value) for value in rule_counts}))
    if not counts or counts[0] != 0 or any(value < 0 for value in counts):
        raise ValueError("rule_counts must be nonnegative and include zero.")
    fit_list = list(fit_graphs)
    eval_list = list(eval_graphs)
    if not fit_list or not eval_list:
        raise ValueError("fit_graphs and eval_graphs must both be nonempty.")
    if len(labels) != len(eval_list):
        raise ValueError("labels must match eval_graphs.")

    encoded_labels, label_mapping = _contiguous_labels(labels)
    if active_gnn.mode == "training":
        train_indices, quality_indices = _quality_split_indices(
            encoded_labels,
            evaluation_fraction=active_gnn.quality_eval_fraction,
            seed=active_gnn.seed,
        )
        quality_available = True
        quality_reason = "paired deterministic held-out evaluation"
    else:
        train_indices = list(range(len(eval_list)))
        quality_indices = []
        quality_available = False
        quality_reason = "inference timing only"

    paths = _paths(prepared_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)

    kwargs = dict(compressor_kwargs)
    kwargs.pop("enumerator", None)
    with tempfile.TemporaryDirectory(prefix="buhito-pareto-prepare-") as temporary:
        temp = Path(temporary)
        payload_path = temp / "prefix_payload.pkl"
        result_path = temp / "prefix_result.json"
        graph_output = temp / "prefix_graphs.pkl"
        _atomic_pickle(
            payload_path,
            {
                "fit_graphs": fit_list,
                "eval_graphs": eval_list,
                "rule_counts": counts,
                "compressor_kwargs": kwargs,
                "compressor_backend": compressor_backend,
                "token_projection": token_projection,
            },
        )
        prefix_metrics = _run_prefix_worker(
            payload_path=payload_path,
            result_path=result_path,
            graph_output_path=graph_output,
            threads=active_gnn.threads,
            timeout=active_gnn.phase_timeout_seconds,
        )
        graph_bundle = _read_pickle(graph_output)

    compression_points = pd.DataFrame(prefix_metrics["points"]).sort_values(
        "rule_count"
    )
    candidate_table = pd.DataFrame(prefix_metrics["candidate_table"])
    dictionary_path = pd.DataFrame(prefix_metrics["dictionary_path"])
    graph_sizes = pd.concat(
        [
            pd.DataFrame(graph_bundle["graph_sizes_by_rule"][count])
            for count in counts
        ],
        ignore_index=True,
    )
    _atomic_csv(compression_points, paths["compression_csv"])
    _atomic_strict_json(
        paths["compression_json"], compression_points.to_dict(orient="records")
    )
    _atomic_csv(candidate_table, paths["candidates"])
    _atomic_csv(dictionary_path, paths["dictionary_path"])
    _atomic_csv(graph_sizes, paths["graph_sizes"])

    common = {
        "config": asdict(active_gnn),
        "num_classes": max(int(encoded_labels.max()) + 1, 2),
        "labels_available": True,
        "feature_dim": 5,
        "quality_metrics_available": quality_available,
        "quality_metrics_reason": quality_reason,
        "train_indices": train_indices,
        "quality_eval_indices": quality_indices,
    }

    def make_payload(graphs: Sequence[nx.Graph], representation: str, count: int):
        return {
            **common,
            "representation": representation,
            "rule_count": int(count),
            "batches": prepare_gnn_batches(
                graphs, encoded_labels, batch_size=active_gnn.batch_size
            ),
            "train_batches": prepare_gnn_batches(
                [graphs[index] for index in train_indices],
                encoded_labels[train_indices],
                batch_size=active_gnn.batch_size,
            ),
            "quality_batches": (
                prepare_gnn_batches(
                    [graphs[index] for index in quality_indices],
                    encoded_labels[quality_indices],
                    batch_size=active_gnn.batch_size,
                )
                if quality_indices
                else []
            ),
        }

    raw_graphs = graph_bundle["raw_graphs"]
    _atomic_pickle(paths["original"], make_payload(raw_graphs, "original", 0))
    payload_hashes = {paths["original"].name: _sha256_file(paths["original"])}
    for count in counts:
        if count == 0:
            continue
        target = _token_payload_path(paths["root"], count)
        graphs = graph_bundle["token_graphs_by_rule"][count]
        _atomic_pickle(target, make_payload(graphs, "tokenized", count))
        payload_hashes[target.name] = _sha256_file(target)

    _atomic_strict_json(paths["config"], asdict(active_gnn))
    _atomic_strict_json(paths["stats_config"], asdict(active_stats))
    quality_label_values = encoded_labels[quality_indices].tolist()
    class_count = max(int(encoded_labels.max()) + 1, 2)
    quality_class_counts = {
        str(class_index): int(quality_label_values.count(class_index))
        for class_index in range(class_count)
    }
    majority_baseline_accuracy = (
        max(quality_class_counts.values()) / len(quality_label_values)
        if quality_label_values
        else None
    )
    metadata_value = {
        **dict(metadata or {}),
        "rule_counts": list(counts),
        "label_mapping": label_mapping,
        "train_indices": train_indices,
        "quality_eval_indices": quality_indices,
        "train_graph_count": len(train_indices),
        "quality_eval_graph_count": len(quality_indices),
        "quality_eval_class_counts": quality_class_counts,
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "quality_metrics_available": quality_available,
        "quality_metrics_reason": quality_reason,
        "compressor_kwargs": kwargs,
        "compressor_backend": compressor_backend,
        "token_projection": token_projection,
        "shared_dictionary_fit_seconds": prefix_metrics[
            "shared_dictionary_fit_seconds"
        ],
        "sweep_total_preparation_seconds": prefix_metrics[
            "sweep_total_preparation_seconds"
        ],
        "interpretation_boundary": (
            "All positive rule counts are forced diagnostic prefixes. Negative "
            "MDL savings can coexist with downstream speedup."
        ),
    }
    _atomic_strict_json(paths["metadata"], metadata_value)

    fingerprint_payload = {
        "format_version": 1,
        "rule_counts": list(counts),
        "gnn_config": asdict(active_gnn),
        "statistics_config": asdict(active_stats),
        "payload_hashes": payload_hashes,
        "compression_sha256": _sha256_file(paths["compression_csv"]),
        "graph_sizes_sha256": _sha256_file(paths["graph_sizes"]),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    tasks = create_pareto_task_manifest(
        rule_counts=counts,
        repeats=active_gnn.repeats,
        seed=active_gnn.seed,
        prepared_fingerprint=fingerprint,
    )
    _atomic_csv(tasks, paths["tasks_csv"])
    _atomic_strict_json(paths["tasks_json"], tasks.to_dict(orient="records"))
    state = {
        **fingerprint_payload,
        "prepared_fingerprint": fingerprint,
        "task_count": len(tasks),
    }
    _atomic_strict_json(paths["state"], state)
    return paths["root"]


def _validate_state(paths: Mapping[str, Path]) -> dict[str, Any]:
    required = (
        "state",
        "config",
        "stats_config",
        "metadata",
        "compression_csv",
        "graph_sizes",
        "original",
        "tasks_csv",
    )
    missing = [str(paths[name]) for name in required if not paths[name].is_file()]
    if missing:
        raise PreparedStateError(
            "Pareto prepared files are missing: " + ", ".join(missing)
        )
    state = json.loads(paths["state"].read_text())
    for filename, expected in state["payload_hashes"].items():
        path = paths["root"] / filename
        if not path.is_file() or _sha256_file(path) != expected:
            raise PreparedStateError(
                f"Pareto payload fingerprint mismatch: {filename}"
            )
    return state


def run_pareto_task(
    prepared_dir: str | Path,
    *,
    task_id: int | None = None,
    task_manifest: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> Path:
    paths = _paths(prepared_dir)
    state = _validate_state(paths)
    tasks = load_pareto_task_manifest(prepared_dir, task_manifest)
    resolved = resolve_task_id(task_id)
    matches = tasks.loc[tasks["task_id"].astype(int) == resolved]
    if len(matches) != 1:
        raise PreparedStateError(f"Pareto task ID {resolved} is not present once.")
    task = matches.iloc[0].to_dict()
    if task["prepared_fingerprint"] != state["prepared_fingerprint"]:
        raise PreparedStateError("Pareto task fingerprint mismatch.")
    payload_path = paths["root"] / str(task["payload_file"])
    destination_root = Path(results_dir) if results_dir else paths["results"]
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"task_{resolved:05d}.json"
    config = GNNBenchmarkConfig(**json.loads(paths["config"].read_text()))
    with tempfile.TemporaryDirectory(prefix=f"buhito-pareto-task-{resolved}-") as tmp:
        temporary = Path(tmp)
        payload = _read_pickle(payload_path)
        payload["config"] = dict(payload["config"])
        payload["config"]["seed"] = int(task["model_seed"])
        task_payload = temporary / "payload.pkl"
        _atomic_pickle(task_payload, payload)
        measured = _run_gnn_worker(
            payload_path=task_payload,
            result_path=temporary / "result.json",
            threads=config.threads,
            timeout=config.phase_timeout_seconds,
            phase_name=(
                f"Pareto task {resolved} rules={int(task['rule_count'])} "
                f"repeat={int(task['repeat'])}"
            ),
        )
    measured.update(
        {
            "task_id": int(resolved),
            "rule_count": int(task["rule_count"]),
            "representation": str(task["representation"]),
            "repeat": int(task["repeat"]),
            "model_seed": int(task["model_seed"]),
            "order_position": int(task["order_position"]),
            "prepared_fingerprint": state["prepared_fingerprint"],
        }
    )
    _atomic_strict_json(destination, measured)
    return destination


def run_pareto_tasks(
    prepared_dir: str | Path,
    *,
    jobs: int = 1,
    task_manifest: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> list[Path]:
    if jobs < 1:
        raise ValueError("jobs must be at least one.")
    tasks = load_pareto_task_manifest(prepared_dir, task_manifest)
    identifiers = tasks["task_id"].astype(int).tolist()
    if jobs == 1:
        return [
            run_pareto_task(
                prepared_dir,
                task_id=identifier,
                task_manifest=task_manifest,
                results_dir=results_dir,
            )
            for identifier in identifiers
        ]
    outputs: list[Path] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                run_pareto_task,
                prepared_dir,
                task_id=identifier,
                task_manifest=task_manifest,
                results_dir=results_dir,
            ): identifier
            for identifier in identifiers
        }
        for future in as_completed(futures):
            outputs.append(future.result())
    return sorted(outputs)


def exact_sign_test_pvalue(successes: int, trials: int) -> float | None:
    """One-sided exact sign-test probability for more positive than negative."""

    if trials <= 0:
        return None
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie between zero and trials.")
    return sum(math.comb(trials, value) for value in range(successes, trials + 1)) / (
        2**trials
    )


def bootstrap_median_interval(
    values: Sequence[float],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float | None, float | None, float | None]:
    finite = np.asarray(
        [value for value in values if math.isfinite(value)],
        dtype=float,
    )
    if finite.size == 0:
        return None, None, None
    estimate = float(np.median(finite))
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, finite.size, size=(samples, finite.size))
    medians = np.median(finite[indices], axis=1)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(medians, [alpha / 2.0, 1.0 - alpha / 2.0])
    return estimate, float(lower), float(upper)


def bootstrap_median_lower_bound(
    values: Sequence[float],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> float | None:
    """One-sided lower confidence bound for a paired median.

    The bound uses the percentile bootstrap.  For a configured confidence
    level of 0.95, the returned value is the fifth percentile of the
    bootstrap median distribution.  This is the bound used for formal
    non-inferiority decisions.
    """

    finite = np.asarray(
        [value for value in values if math.isfinite(value)],
        dtype=float,
    )
    if finite.size == 0:
        return None
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, finite.size, size=(samples, finite.size))
    medians = np.median(finite[indices], axis=1)
    alpha = 1.0 - confidence_level
    return float(np.quantile(medians, alpha))


def _meets_noninferiority_bound(
    lower_bound: float,
    margin: float,
    *,
    absolute_tolerance: float = 1e-12,
) -> bool:
    """Return whether a lower bound clears the non-inferiority margin."""

    boundary = -float(margin)
    value = float(lower_bound)
    return value > boundary or math.isclose(
        value, boundary, rel_tol=0.0, abs_tol=absolute_tolerance
    )


def noninferiority_sign_test_pvalue(
    values: Sequence[float],
    *,
    margin: float,
) -> float | None:
    """Exact one-sided sign test for median delta above ``-margin``.

    Ties exactly on the non-inferiority boundary are excluded, matching the
    ordinary exact sign test.  This p-value is supplementary; the declared
    non-inferiority decision is based on the one-sided bootstrap lower bound.
    """

    if margin < 0:
        raise ValueError("margin cannot be negative.")
    finite = [float(value) for value in values if math.isfinite(float(value))]
    shifted = [value + margin for value in finite]
    non_ties = [
        value
        for value in shifted
        if not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12)
    ]
    successes = sum(value > 0.0 for value in non_ties)
    return exact_sign_test_pvalue(successes, len(non_ties))


def _optional_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _paired_rows(runs: pd.DataFrame, rule_counts: Sequence[int]) -> pd.DataFrame:
    original = runs.loc[runs["representation"] == "original"].set_index("repeat")
    rows: list[dict[str, Any]] = []
    quality_columns = (
        "quality_eval_loss",
        "quality_eval_accuracy",
        "quality_eval_macro_f1",
        "final_train_loss",
        "final_train_accuracy",
    )
    for count in rule_counts:
        if count == 0:
            token = original
        else:
            token = runs.loc[
                (runs["representation"] == "tokenized")
                & (runs["rule_count"].astype(int) == int(count))
            ].set_index("repeat")
        if set(original.index) != set(token.index):
            raise PreparedStateError(
                f"Original/tokenized repeat mismatch for rule count {count}."
            )
        for repeat in sorted(original.index):
            raw = original.loc[repeat]
            compressed = token.loc[repeat]
            raw_seconds = float(raw["workload_seconds"])
            token_seconds = float(compressed["workload_seconds"])
            row = {
                "rule_count": int(count),
                "repeat": int(repeat),
                "model_seed": int(raw["model_seed"]),
                "original_seconds": raw_seconds,
                "tokenized_seconds": token_seconds,
                "paired_speedup": raw_seconds / token_seconds,
                "time_saved_seconds": raw_seconds - token_seconds,
                "time_reduction_fraction": 1.0 - token_seconds / raw_seconds,
                "original_peak_rss_mb": float(raw["peak_rss_mb"]),
                "tokenized_peak_rss_mb": float(compressed["peak_rss_mb"]),
                "peak_rss_delta_mb": float(compressed["peak_rss_mb"])
                - float(raw["peak_rss_mb"]),
            }
            for name in quality_columns:
                left = _optional_number(raw.get(name))
                right = _optional_number(compressed.get(name))
                row[f"original_{name}"] = left
                row[f"tokenized_{name}"] = right
                row[f"{name}_delta"] = (
                    None if left is None or right is None else right - left
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _pareto_flags(points: pd.DataFrame, quality_column: str) -> list[bool]:
    flags: list[bool] = []
    for index, row in points.iterrows():
        row_quality = _optional_number(row.get(quality_column))
        row_quality = 0.0 if row_quality is None else row_quality
        dominated = False
        for other_index, other in points.iterrows():
            if index == other_index:
                continue
            other_quality = _optional_number(other.get(quality_column))
            other_quality = 0.0 if other_quality is None else other_quality
            at_least = (
                other["median_paired_speedup"] >= row["median_paired_speedup"]
                and other_quality >= row_quality
            )
            strict = (
                other["median_paired_speedup"] > row["median_paired_speedup"]
                or other_quality > row_quality
            )
            if at_least and strict:
                dominated = True
                break
        flags.append(not dominated)
    return flags


def _aggregate_points(
    paired: pd.DataFrame,
    compression: pd.DataFrame,
    graph_sizes: pd.DataFrame,
    stats: ParetoStatisticsConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    metric_columns = {
        "paired_speedup": "median_paired_speedup",
        "time_reduction_fraction": "median_time_reduction_fraction",
        "quality_eval_accuracy_delta": "median_accuracy_delta",
        "quality_eval_macro_f1_delta": "median_macro_f1_delta",
        "quality_eval_loss_delta": "median_loss_delta",
    }
    for count in sorted(paired["rule_count"].unique()):
        group = paired.loc[paired["rule_count"] == count].copy()
        compression_row = compression.loc[
            compression["rule_count"].astype(int) == int(count)
        ].iloc[0]
        size_group = graph_sizes.loc[
            graph_sizes["rule_count"].astype(int) == int(count)
        ]
        raw_nodes = int(size_group["raw_nodes"].sum())
        token_nodes = int(size_group["token_nodes"].sum())
        raw_edges = int(size_group["raw_edges"].sum())
        token_edges = int(size_group["token_edges"].sum())
        row: dict[str, Any] = {
            **compression_row.to_dict(),
            "repeats": len(group),
            "raw_nodes": raw_nodes,
            "tokenized_nodes": token_nodes,
            "raw_edges": raw_edges,
            "tokenized_edges": token_edges,
            "node_reduction_fraction": 1.0 - token_nodes / max(raw_nodes, 1),
            "edge_reduction_fraction": 1.0 - token_edges / max(raw_edges, 1),
            "original_median_seconds": float(
                group["original_seconds"].median()
            ),
            "tokenized_median_seconds": float(
                group["tokenized_seconds"].median()
            ),
            "median_time_saved_seconds": float(
                group["time_saved_seconds"].median()
            ),
            "faster_repeat_count": int((group["paired_speedup"] > 1.0).sum()),
            "slower_repeat_count": int((group["paired_speedup"] < 1.0).sum()),
            "minimum_paired_speedup": float(group["paired_speedup"].min()),
            "maximum_paired_speedup": float(group["paired_speedup"].max()),
            "ratio_of_medians_speedup": float(
                group["original_seconds"].median()
                / group["tokenized_seconds"].median()
            ),
            "original_median_accuracy": _optional_number(
                group["original_quality_eval_accuracy"].median()
            ),
            "tokenized_median_accuracy": _optional_number(
                group["tokenized_quality_eval_accuracy"].median()
            ),
            "original_median_macro_f1": _optional_number(
                group["original_quality_eval_macro_f1"].median()
            ),
            "tokenized_median_macro_f1": _optional_number(
                group["tokenized_quality_eval_macro_f1"].median()
            ),
            "original_median_loss": _optional_number(
                group["original_quality_eval_loss"].median()
            ),
            "tokenized_median_loss": _optional_number(
                group["tokenized_quality_eval_loss"].median()
            ),
        }
        non_ties = group.loc[group["paired_speedup"] != 1.0]
        row["timing_sign_test_pvalue"] = exact_sign_test_pvalue(
            int((non_ties["paired_speedup"] > 1.0).sum()), len(non_ties)
        )
        for metric, prefix in (
            ("quality_eval_accuracy_delta", "accuracy"),
            ("quality_eval_macro_f1_delta", "macro_f1"),
        ):
            finite_quality = group.loc[group[metric].notna()]
            row[f"{prefix}_better_repeat_count"] = int(
                (finite_quality[metric] > 0.0).sum()
            )
            row[f"{prefix}_equal_repeat_count"] = int(
                (finite_quality[metric] == 0.0).sum()
            )
            row[f"{prefix}_worse_repeat_count"] = int(
                (finite_quality[metric] < 0.0).sum()
            )
            non_tied_quality = finite_quality.loc[finite_quality[metric] != 0.0]
            row[f"{prefix}_worse_sign_test_pvalue"] = exact_sign_test_pvalue(
                int((non_tied_quality[metric] < 0.0).sum()),
                len(non_tied_quality),
            )
        for offset, (source, destination) in enumerate(metric_columns.items()):
            values = [
                float(value)
                for value in group[source].dropna().tolist()
                if math.isfinite(float(value))
            ]
            interval_seed = stats.statistics_seed + 1000 * int(count) + offset
            estimate, lower, upper = bootstrap_median_interval(
                values,
                samples=stats.bootstrap_samples,
                confidence_level=stats.confidence_level,
                seed=interval_seed,
            )
            one_sided_lower = bootstrap_median_lower_bound(
                values,
                samples=stats.bootstrap_samples,
                confidence_level=stats.confidence_level,
                seed=interval_seed,
            )
            row[destination] = estimate
            row[f"{destination}_ci_lower"] = lower
            row[f"{destination}_ci_upper"] = upper
            row[f"{destination}_one_sided_ci_lower"] = one_sided_lower
            stat_rows.append(
                {
                    "rule_count": int(count),
                    "metric": destination,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "one_sided_ci_lower": one_sided_lower,
                    "confidence_level": stats.confidence_level,
                    "bootstrap_samples": stats.bootstrap_samples,
                }
            )
        preparation = float(row["standalone_preparation_seconds"])
        saved = float(row["median_time_saved_seconds"])
        row["break_even_reuses"] = preparation / saved if saved > 0 else None

        noninferiority_specs = (
            (
                "accuracy",
                "quality_eval_accuracy_delta",
                "median_accuracy_delta",
                stats.accuracy_drop_tolerance,
            ),
            (
                "macro_f1",
                "quality_eval_macro_f1_delta",
                "median_macro_f1_delta",
                stats.macro_f1_drop_tolerance,
            ),
        )
        noninferiority_flags: list[bool] = []
        availability_flags: list[bool] = []
        median_tolerance_flags: list[bool] = []
        for prefix, source, destination, margin in noninferiority_specs:
            estimate = row.get(destination)
            lower_bound = row.get(f"{destination}_one_sided_ci_lower")
            values = [
                float(value)
                for value in group[source].dropna().tolist()
                if math.isfinite(float(value))
            ]
            available = estimate is not None and lower_bound is not None
            noninferior = bool(
                available
                and _meets_noninferiority_bound(
                    float(lower_bound), float(margin)
                )
            )
            median_within = bool(
                estimate is not None and float(estimate) >= -float(margin)
            )
            row[f"{prefix}_noninferiority_margin"] = float(margin)
            row[f"{prefix}_noninferiority_lower_bound"] = lower_bound
            row[f"{prefix}_noninferiority_sign_test_pvalue"] = (
                noninferiority_sign_test_pvalue(values, margin=float(margin))
            )
            row[f"{prefix}_noninferior"] = noninferior if available else None
            availability_flags.append(available)
            noninferiority_flags.append(noninferior)
            median_tolerance_flags.append(median_within)

        row["median_quality_within_tolerance"] = bool(
            all(median_tolerance_flags)
        )
        if all(availability_flags):
            quality_noninferior: bool | None = bool(all(noninferiority_flags))
            status = "noninferior" if quality_noninferior else "inconclusive"
        else:
            quality_noninferior = None
            status = "unavailable"
        row["quality_noninferior"] = quality_noninferior
        row["quality_noninferiority_status"] = status
        # Backward-compatible column name.  It now carries the formal
        # confidence-bound decision, not the old median-only tolerance check.
        row["quality_within_tolerance"] = bool(quality_noninferior)
        point_rows.append(row)

    points = pd.DataFrame(point_rows).sort_values("rule_count").reset_index(drop=True)
    points["pareto_speed_accuracy"] = _pareto_flags(
        points, "median_accuracy_delta"
    )
    points["pareto_speed_macro_f1"] = _pareto_flags(
        points, "median_macro_f1_delta"
    )
    points["recommended_noninferior"] = False
    points["recommended_under_tolerance"] = False
    eligible = points.loc[points["quality_noninferior"].fillna(False).astype(bool)]
    if not eligible.empty:
        recommended_index = eligible["median_paired_speedup"].idxmax()
        points.loc[recommended_index, "recommended_noninferior"] = True
        points.loc[recommended_index, "recommended_under_tolerance"] = True
    return points, pd.DataFrame(stat_rows)


def _quality_detail_frames(
    runs: pd.DataFrame,
    rule_counts: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    originals = runs.loc[runs["representation"] == "original"].set_index("repeat")
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for count in rule_counts:
        tokenized = (
            originals
            if int(count) == 0
            else runs.loc[
                (runs["representation"] == "tokenized")
                & (runs["rule_count"].astype(int) == int(count))
            ].set_index("repeat")
        )
        for repeat in sorted(originals.index):
            for representation, row in (
                ("original", originals.loc[repeat]),
                ("tokenized", tokenized.loc[repeat]),
            ):
                metrics = row.get("quality_eval_per_class_metrics")
                matrix = row.get("quality_eval_confusion_matrix")
                if isinstance(metrics, str):
                    metrics = json.loads(metrics)
                if isinstance(matrix, str):
                    matrix = json.loads(matrix)
                for metric in metrics or []:
                    per_class_rows.append(
                        {
                            "rule_count": int(count),
                            "repeat": int(repeat),
                            "model_seed": int(row["model_seed"]),
                            "representation": representation,
                            **dict(metric),
                        }
                    )
                for actual, matrix_row in enumerate(matrix or []):
                    for predicted, value in enumerate(matrix_row):
                        confusion_rows.append(
                            {
                                "rule_count": int(count),
                                "repeat": int(repeat),
                                "model_seed": int(row["model_seed"]),
                                "representation": representation,
                                "actual_class": int(actual),
                                "predicted_class": int(predicted),
                                "count": int(value),
                            }
                        )
    per_class_columns = [
        "rule_count",
        "repeat",
        "model_seed",
        "representation",
        "class_index",
        "precision",
        "recall",
        "f1",
        "support",
        "predicted_count",
    ]
    confusion_columns = [
        "rule_count",
        "repeat",
        "model_seed",
        "representation",
        "actual_class",
        "predicted_class",
        "count",
    ]
    return (
        pd.DataFrame(per_class_rows, columns=per_class_columns),
        pd.DataFrame(confusion_rows, columns=confusion_columns),
    )


def aggregate_gnn_pareto_sweep(
    prepared_dir: str | Path,
    *,
    task_manifest: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> ParetoSweepResult:
    paths = _paths(prepared_dir)
    state = _validate_state(paths)
    tasks = load_pareto_task_manifest(prepared_dir, task_manifest)
    root = Path(results_dir) if results_dir else paths["results"]
    expected = set(tasks["task_id"].astype(int))
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for result_file in sorted(root.glob("task_*.json")):
        row = json.loads(result_file.read_text())
        task_id = int(row.get("task_id", -1))
        if task_id in seen:
            raise PreparedStateError(f"Duplicate Pareto task result {task_id}.")
        seen.add(task_id)
        if row.get("prepared_fingerprint") != state["prepared_fingerprint"]:
            raise PreparedStateError(
                f"Pareto task {task_id} has an incompatible fingerprint."
            )
        rows.append(row)
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    if missing:
        raise PreparedStateError(f"Missing Pareto task results: {missing}")
    if unexpected:
        raise PreparedStateError(f"Unexpected Pareto task results: {unexpected}")
    runs = pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)
    compression = pd.read_csv(paths["compression_csv"])
    graph_sizes = pd.read_csv(paths["graph_sizes"])
    rule_counts = compression["rule_count"].astype(int).tolist()
    paired = _paired_rows(runs, rule_counts)
    stats_config = ParetoStatisticsConfig(
        **json.loads(paths["stats_config"].read_text())
    )
    points, statistics = _aggregate_points(
        paired, compression, graph_sizes, stats_config
    )
    per_class_metrics, confusion_matrices = _quality_detail_frames(
        runs, rule_counts
    )
    metadata = json.loads(paths["metadata"].read_text())
    points["majority_baseline_accuracy"] = metadata.get(
        "majority_baseline_accuracy"
    )
    frontier = points.loc[
        points["pareto_speed_accuracy"] | points["pareto_speed_macro_f1"]
    ].copy()
    metadata.update(
        {
            "prepared_fingerprint": state["prepared_fingerprint"],
            "statistics_config": asdict(stats_config),
        }
    )
    return ParetoSweepResult(
        paired_runs=paired,
        points=points,
        statistics=statistics,
        frontier=frontier,
        compression_points=compression,
        graph_sizes=graph_sizes,
        per_class_metrics=per_class_metrics,
        confusion_matrices=confusion_matrices,
        metadata=metadata,
    )


def pareto_scalar_summary(result: ParetoSweepResult) -> dict[str, Any]:
    points = result.points
    recommended = points.loc[points["recommended_noninferior"]]
    best_speed = points.loc[points["median_paired_speedup"].idxmax()]
    return {
        "rule_counts": points["rule_count"].astype(int).tolist(),
        "point_count": len(points),
        "best_speed_rule_count": int(best_speed["rule_count"]),
        "best_median_paired_speedup": float(best_speed["median_paired_speedup"]),
        "best_speed_accuracy_delta": _optional_number(
            best_speed["median_accuracy_delta"]
        ),
        "best_speed_macro_f1_delta": _optional_number(
            best_speed["median_macro_f1_delta"]
        ),
        "recommended_rule_count": (
            None if recommended.empty else int(recommended.iloc[0]["rule_count"])
        ),
        "recommended_noninferior_rule_count": (
            None if recommended.empty else int(recommended.iloc[0]["rule_count"])
        ),
        "pareto_rule_counts_speed_accuracy": points.loc[
            points["pareto_speed_accuracy"], "rule_count"
        ].astype(int).tolist(),
        "pareto_rule_counts_speed_macro_f1": points.loc[
            points["pareto_speed_macro_f1"], "rule_count"
        ].astype(int).tolist(),
    }


def _write_paper_tables(points: pd.DataFrame, output: Path) -> None:
    columns = [
        "rule_count",
        "node_reduction_fraction",
        "edge_reduction_fraction",
        "median_paired_speedup",
        "median_time_reduction_fraction",
        "original_median_accuracy",
        "tokenized_median_accuracy",
        "median_accuracy_delta",
        "median_accuracy_delta_ci_lower",
        "median_accuracy_delta_ci_upper",
        "accuracy_noninferiority_margin",
        "accuracy_noninferiority_lower_bound",
        "accuracy_noninferiority_sign_test_pvalue",
        "accuracy_noninferior",
        "original_median_macro_f1",
        "tokenized_median_macro_f1",
        "median_macro_f1_delta",
        "median_macro_f1_delta_ci_lower",
        "median_macro_f1_delta_ci_upper",
        "macro_f1_noninferiority_margin",
        "macro_f1_noninferiority_lower_bound",
        "macro_f1_noninferiority_sign_test_pvalue",
        "macro_f1_noninferior",
        "quality_noninferiority_status",
        "recommended_noninferior",
        "original_median_loss",
        "tokenized_median_loss",
        "median_loss_delta",
        "forced_net_savings_bits",
        "standalone_preparation_seconds",
        "break_even_reuses",
        "timing_sign_test_pvalue",
        "pareto_speed_accuracy",
        "pareto_speed_macro_f1",
    ]
    table = points.loc[:, [name for name in columns if name in points]].copy()
    table.to_csv(output / "paper_table.csv", index=False)
    headers = [str(column) for column in table.columns]
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        markdown_rows.append(
            "| "
            + " | ".join(
                "" if pd.isna(value) else str(value) for value in row
            )
            + " |"
        )
    (output / "paper_table.md").write_text("\n".join(markdown_rows) + "\n")
    latex = table.to_latex(index=False, float_format=lambda value: f"{value:.4g}")
    (output / "paper_table.tex").write_text(latex)


def save_pareto_plots(result: ParetoSweepResult, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Pareto plots require matplotlib.") from exc
    points = result.points.sort_values("rule_count")

    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    quality = pd.to_numeric(points["median_accuracy_delta"], errors="coerce")
    if quality.notna().any():
        x_values = quality
        x_label = "Held-out accuracy delta (tokenized - original)"
        axis.axvline(0.0, linestyle="--", linewidth=1)
    else:
        x_values = points["rule_count"]
        x_label = "Applied rule count (quality unavailable)"
    axis.plot(x_values, points["median_paired_speedup"], marker="o")
    for x_value, row in zip(x_values, points.itertuples(index=False), strict=True):
        axis.annotate(
            f"k={int(row.rule_count)}",
            (x_value, row.median_paired_speedup),
        )
    axis.axhline(1.0, linestyle="--", linewidth=1)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Median paired training speedup")
    axis.set_title("Compression--speed--quality Pareto sweep")
    figure.tight_layout()
    figure.savefig(output / "pareto_speed_accuracy.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.plot(
        points["node_reduction_fraction"],
        points["median_paired_speedup"],
        marker="o",
    )
    for row in points.itertuples(index=False):
        axis.annotate(
            f"k={int(row.rule_count)}",
            (row.node_reduction_fraction, row.median_paired_speedup),
        )
    axis.set_xlabel("Node reduction fraction")
    axis.set_ylabel("Median paired training speedup")
    axis.set_title("Structural reduction versus speedup")
    figure.tight_layout()
    figure.savefig(output / "pareto_reduction_speed.png", dpi=200)
    plt.close(figure)


def render_pareto_summary(result: ParetoSweepResult) -> str:
    scalar = pareto_scalar_summary(result)
    recommended = scalar["recommended_rule_count"]
    recommended_text = (
        "none under configured tolerance"
        if recommended is None
        else str(recommended)
    )
    return f"""# Buhito compression--speed--quality Pareto sweep

This sweep fits the motif dictionary once and evaluates nested forced rule
prefixes on the same graphs, split, architecture, and paired random seeds.

## Headline

- Rule counts: `{scalar['rule_counts']}`
- Best-speed rule count: `{scalar['best_speed_rule_count']}`
- Best median paired speedup: `{scalar['best_median_paired_speedup']:.6f}x`
- Recommended rule count under configured quality tolerances: `{recommended_text}`
- Speed/accuracy Pareto rule counts: `{scalar['pareto_rule_counts_speed_accuracy']}`
- Speed/macro-F1 Pareto rule counts: `{scalar['pareto_rule_counts_speed_macro_f1']}`

## Interpretation

Positive rule counts are forced diagnostic tokenizations. Their forced MDL
savings may be negative even when training becomes faster. Report structural
reduction, runtime, preparation cost, held-out quality, and uncertainty
together. The exact sign test uses paired repeats; bootstrap intervals estimate
median paired effects and are descriptive when the number of seeds is small.

## Paper artifacts

- `pareto_points.csv`: one aggregate row per rule count.
- `pareto_paired_runs.csv`: paired original/tokenized results per seed.
- `pareto_statistics.csv`: bootstrap median intervals.
- `pareto_frontier.csv`: nondominated speed/quality points.
- `pareto_per_class_metrics.csv`: per-seed precision, recall, F1, and support.
- `pareto_confusion_matrices.csv`: long-form held-out confusion matrices.
- `paper_table.csv`, `.md`, `.tex`: publication-oriented tables.
"""


def validate_pareto_artifacts(
    output_dir: str | Path,
    *,
    require_plots: bool = False,
) -> dict[str, Any]:
    """Validate a saved sweep and return its strict-JSON summary."""

    root = Path(output_dir)
    required = {
        "pareto_points.csv",
        "pareto_paired_runs.csv",
        "pareto_statistics.csv",
        "pareto_frontier.csv",
        "compression_prefixes.csv",
        "graph_sizes_by_prefix.csv",
        "pareto_per_class_metrics.csv",
        "pareto_confusion_matrices.csv",
        "paper_table.csv",
        "paper_table.md",
        "paper_table.tex",
        "summary.json",
        "metadata.json",
        "README.md",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise PreparedStateError(
            "Pareto artifacts are missing: " + ", ".join(missing)
        )
    if require_plots:
        plot_names = {
            "pareto_speed_accuracy.png",
            "pareto_reduction_speed.png",
        }
        missing_plots = sorted(
            name for name in plot_names if not (root / name).is_file()
        )
        if missing_plots:
            raise PreparedStateError(
                "Pareto plots are missing: " + ", ".join(missing_plots)
            )

    def strict_load(path: Path) -> Any:
        return json.loads(
            path.read_text(),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-standard JSON constant: {value}")
            ),
        )

    summary = strict_load(root / "summary.json")
    strict_load(root / "metadata.json")
    points = pd.read_csv(root / "pareto_points.csv")
    paired = pd.read_csv(root / "pareto_paired_runs.csv")
    compression = pd.read_csv(root / "compression_prefixes.csv")
    frontier = pd.read_csv(root / "pareto_frontier.csv")

    if points.empty or paired.empty:
        raise PreparedStateError("Pareto points and paired runs must be nonempty.")
    counts = points["rule_count"].astype(int).tolist()
    if counts != sorted(set(counts)) or not counts or counts[0] != 0:
        raise PreparedStateError(
            "Pareto rule counts must be unique, sorted, and include zero."
        )
    baseline = points.loc[points["rule_count"].astype(int) == 0].iloc[0]
    for name, expected in (
        ("node_reduction_fraction", 0.0),
        ("edge_reduction_fraction", 0.0),
        ("median_paired_speedup", 1.0),
        ("median_accuracy_delta", 0.0),
        ("median_macro_f1_delta", 0.0),
    ):
        value = _optional_number(baseline.get(name))
        if value is not None and not math.isclose(value, expected, abs_tol=1e-12):
            raise PreparedStateError(
                f"Baseline Pareto point has invalid {name}: {value}."
            )

    noninferiority_columns = {
        "accuracy_noninferiority_margin",
        "accuracy_noninferiority_lower_bound",
        "accuracy_noninferior",
        "macro_f1_noninferiority_margin",
        "macro_f1_noninferiority_lower_bound",
        "macro_f1_noninferior",
        "quality_noninferior",
        "quality_noninferiority_status",
        "recommended_noninferior",
    }
    missing_noninferiority = sorted(noninferiority_columns - set(points.columns))
    if missing_noninferiority:
        raise PreparedStateError(
            "Pareto non-inferiority fields are missing: "
            + ", ".join(missing_noninferiority)
        )
    for row in points.itertuples(index=False):
        accuracy_expected = _meets_noninferiority_bound(
            float(row.accuracy_noninferiority_lower_bound),
            float(row.accuracy_noninferiority_margin),
        )
        f1_expected = _meets_noninferiority_bound(
            float(row.macro_f1_noninferiority_lower_bound),
            float(row.macro_f1_noninferiority_margin),
        )
        if bool(row.accuracy_noninferior) != accuracy_expected:
            raise PreparedStateError(
                "Accuracy non-inferiority decision is inconsistent with its "
                "one-sided confidence bound."
            )
        if bool(row.macro_f1_noninferior) != f1_expected:
            raise PreparedStateError(
                "Macro-F1 non-inferiority decision is inconsistent with its "
                "one-sided confidence bound."
            )
        if bool(row.quality_noninferior) != (accuracy_expected and f1_expected):
            raise PreparedStateError(
                "Combined quality non-inferiority decision is inconsistent."
            )

    positive = points.loc[points["rule_count"].astype(int) > 0]
    if not positive.empty:
        if not (
            positive["applied_rule_count"].astype(int)
            == positive["rule_count"].astype(int)
        ).all():
            raise PreparedStateError(
                "Applied rule counts do not match requested prefixes."
            )
        if not positive["forced_dictionary_bits"].gt(0.0).all():
            raise PreparedStateError(
                "Positive Pareto prefixes must pay dictionary cost."
            )
    if not compression["decode_failures"].astype(int).eq(0).all():
        raise PreparedStateError("One or more Pareto prefixes failed decoding.")

    repeat_counts = paired.groupby("rule_count")["repeat"].nunique()
    if repeat_counts.nunique() != 1:
        raise PreparedStateError(
            "Pareto rule counts do not have equal paired-repeat counts."
        )
    original_by_repeat = paired.pivot_table(
        index="repeat",
        columns="rule_count",
        values="original_seconds",
        aggfunc="first",
    )
    for _, row in original_by_repeat.iterrows():
        values = row.dropna().to_numpy(dtype=float)
        if values.size and not np.allclose(values, values[0], rtol=0, atol=0):
            raise PreparedStateError(
                "The shared original baseline differs across rule counts."
            )

    finite_columns = [
        "median_paired_speedup",
        "median_time_reduction_fraction",
        "node_reduction_fraction",
        "edge_reduction_fraction",
    ]
    for name in finite_columns:
        values = pd.to_numeric(points[name], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise PreparedStateError(f"Non-finite Pareto metric: {name}.")
    if not set(frontier["rule_count"].astype(int)).issubset(set(counts)):
        raise PreparedStateError("Pareto frontier contains an unknown point.")
    if int(summary["point_count"]) != len(points):
        raise PreparedStateError("Pareto summary point count is inconsistent.")
    return summary


def write_pareto_slurm_array_script(
    prepared_dir: str | Path,
    script_path: str | Path,
    *,
    cli_path: str = "examples/benchmark_gnn_pareto.py",
) -> Path:
    tasks = load_pareto_task_manifest(prepared_dir)
    maximum = int(tasks["task_id"].max())
    prepared = Path(prepared_dir).resolve()
    script = Path(script_path)
    text = (
        "#!/usr/bin/env bash\n"
        "#SBATCH --job-name=buhito-pareto\n"
        f"#SBATCH --array=0-{maximum}\n"
        "#SBATCH --cpus-per-task=1\n"
        "##SBATCH --gres=gpu:SITE_GPU_COUNT\n"
        "#SBATCH --time=SITE_WALLTIME\n"
        "#SBATCH --mem=SITE_MEMORY\n"
        "##SBATCH --partition=SITE_PARTITION\n"
        "##SBATCH --account=SITE_ACCOUNT\n\n"
        "set -euo pipefail\n\n"
        f"python {cli_path} \\\n"
        f"  --prepared-dir {prepared} \\\n"
        "  --task-id \"$SLURM_ARRAY_TASK_ID\"\n"
    )
    script.parent.mkdir(parents=True, exist_ok=True)
    temporary = script.with_name(script.name + f".tmp-{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, script)
    script.chmod(0o755)
    return script

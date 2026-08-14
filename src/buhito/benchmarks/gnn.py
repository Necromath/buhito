"""Downstream GNN timing for original and motif-tokenized graphs.

The runtime benchmark in :mod:`buhito.benchmarks.runtime` measures Buhito
 graphlet enumeration.  This module measures a separate downstream workload:
 a small structural graph-convolutional network implemented with optional
 PyTorch and no PyTorch Geometric dependency.

The built-in model is a systems reference, not a claim that tokenized and
 original graphs are prediction-equivalent.  Practitioners should report both
 runtime and task quality when replacing it with an application model.
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
from typing import Any, Literal, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from .runtime import (
    PreparedStateError,
    WorkerTimeoutError,
    _atomic_json,
    _atomic_pickle,
    _read_pickle,
    _sha256_file,
    _subprocess_environment,
    _terminate_process_group,
    create_task_manifest,
    load_task_manifest,
    resolve_task_id,
)


GNNMode = Literal["inference", "training"]


def _strict_json_value(value: Any) -> Any:
    """Convert nested benchmark metadata to standards-compliant JSON values.

    Runtime metadata from older forced-tokenization experiments may contain
    infinite configuration sentinels such as ``min_rule_savings_bits=-inf``.
    Those values are meaningful configuration, not missing statistics. Encode
    them as the explicit strings ``"-Infinity"`` and ``"Infinity"``. Encode
    unavailable or undefined floating-point statistics (NaN) as JSON ``null``.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _strict_json_value(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _strict_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, set):
        return [
            _strict_json_value(item)
            for item in sorted(value, key=repr)
        ]
    return str(value)


def _strict_json_text(value: Any) -> str:
    """Serialize benchmark data as strict RFC-compatible JSON text."""

    return (
        json.dumps(
            _strict_json_value(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


@dataclass(frozen=True)
class GNNBenchmarkConfig:
    """Configuration for the built-in structural GCN benchmark."""

    mode: GNNMode = "inference"
    repeats: int = 5
    warmup_steps: int = 2
    steps_per_repeat: int = 10
    epochs: int = 10
    hidden_channels: int = 64
    num_layers: int = 2
    batch_size: int = 32
    device: str = "auto"
    threads: int = 1
    seed: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    quality_eval_fraction: float = 0.2
    phase_timeout_seconds: float | None = None

    def validate(self) -> None:
        if self.mode not in {"inference", "training"}:
            raise ValueError(f"Unsupported GNN mode: {self.mode!r}.")
        if self.repeats < 1:
            raise ValueError("repeats must be at least one.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative.")
        if self.steps_per_repeat < 1:
            raise ValueError("steps_per_repeat must be at least one.")
        if self.epochs < 1:
            raise ValueError("epochs must be at least one.")
        if self.hidden_channels < 1:
            raise ValueError("hidden_channels must be positive.")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.threads < 1:
            raise ValueError("threads must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if not 0.0 < self.quality_eval_fraction < 1.0:
            raise ValueError(
                "quality_eval_fraction must be strictly between zero and one."
            )
        if (
            self.phase_timeout_seconds is not None
            and self.phase_timeout_seconds <= 0
        ):
            raise ValueError("phase_timeout_seconds must be positive.")


@dataclass
class GNNBenchmarkResult:
    """Aggregated original-versus-tokenized downstream GNN measurements."""

    config: GNNBenchmarkConfig
    runs: pd.DataFrame
    summary: pd.DataFrame
    compression: dict[str, Any]
    graph_sizes: pd.DataFrame
    metadata: dict[str, Any]

    def save(self, output_dir: str | Path) -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.runs.to_csv(output / "gnn_runs.csv", index=False)
        self.summary.to_csv(output / "gnn_summary.csv", index=False)
        (output / "compression_summary.json").write_text(
            _strict_json_text(self.compression)
        )
        (output / "metadata.json").write_text(
            _strict_json_text(self.metadata)
        )
        (output / "summary.json").write_text(
            _strict_json_text(gnn_scalar_summary(self))
        )
        (output / "README.md").write_text(render_gnn_summary(self))
        return output


def _node_is_motif(data: Mapping[str, Any]) -> tuple[float, float]:
    kind = data.get("__buhito_mdl_kind__", data.get("mdl_kind"))
    label = data.get("__buhito_mdl_node_label__", data.get("mdl_label"))
    rank = 0.0
    motif = kind == "motif"
    if isinstance(label, tuple) and label:
        motif = motif or label[0] == "motif"
        if motif and len(label) > 1:
            try:
                rank = float(label[1])
            except (TypeError, ValueError):
                rank = 0.0
    return float(motif), math.log1p(max(rank, 0.0))


def structural_node_features(graph: nx.Graph) -> np.ndarray:
    """Return fixed-width topology/token features for the reference GCN.

    Columns are: constant one, log-degree, graph-normalized degree, motif-node
    indicator, and log motif-rule rank.  The fixed width makes the same model
    architecture valid for original and tokenized representations.
    """

    nodes = list(graph.nodes())
    if not nodes:
        return np.zeros((0, 5), dtype=np.float32)
    degree = np.asarray([graph.degree(node) for node in nodes], dtype=np.float32)
    scale = float(max(float(degree.max()), 1.0))
    features = np.zeros((len(nodes), 5), dtype=np.float32)
    features[:, 0] = 1.0
    features[:, 1] = np.log1p(degree)
    features[:, 2] = degree / scale
    for index, node in enumerate(nodes):
        motif, rank = _node_is_motif(graph.nodes[node])
        features[index, 3] = motif
        features[index, 4] = rank
    return features


def _graph_arrays(graph: nx.Graph) -> dict[str, Any]:
    nodes = list(graph.nodes())
    mapping = {node: index for index, node in enumerate(nodes)}
    sources: list[int] = []
    targets: list[int] = []
    for source, target in graph.edges():
        left = mapping[source]
        right = mapping[target]
        sources.append(left)
        targets.append(right)
        if left != right:
            sources.append(right)
            targets.append(left)
    for index in range(len(nodes)):
        sources.append(index)
        targets.append(index)
    edge_index = np.asarray([sources, targets], dtype=np.int64)
    return {
        "x": structural_node_features(graph),
        "edge_index": edge_index,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "message_edges": edge_index.shape[1],
    }


def prepare_gnn_batches(
    graphs: Sequence[nx.Graph],
    labels: Sequence[int],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Convert graphs into framework-neutral NumPy mini-batches."""

    if len(graphs) != len(labels):
        raise ValueError("graphs and labels must have the same length.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    arrays = [_graph_arrays(graph) for graph in graphs]
    batches: list[dict[str, Any]] = []
    for start in range(0, len(arrays), batch_size):
        subset = arrays[start : start + batch_size]
        subset_labels = labels[start : start + batch_size]
        xs: list[np.ndarray] = []
        edges: list[np.ndarray] = []
        graph_index: list[np.ndarray] = []
        node_offset = 0
        total_edges = 0
        total_message_edges = 0
        for local_graph, record in enumerate(subset):
            xs.append(record["x"])
            edges.append(record["edge_index"] + node_offset)
            graph_index.append(
                np.full(record["nodes"], local_graph, dtype=np.int64)
            )
            node_offset += int(record["nodes"])
            total_edges += int(record["edges"])
            total_message_edges += int(record["message_edges"])
        batches.append(
            {
                "x": np.concatenate(xs, axis=0),
                "edge_index": np.concatenate(edges, axis=1),
                "batch": np.concatenate(graph_index, axis=0),
                "labels": np.asarray(subset_labels, dtype=np.int64),
                "graphs": len(subset),
                "nodes": node_offset,
                "edges": total_edges,
                "message_edges": total_message_edges,
            }
        )
    return batches


def _contiguous_labels(labels: Sequence[Any]) -> tuple[np.ndarray, dict[str, int]]:
    values = list(labels)
    unique = sorted(set(values), key=repr)
    mapping = {repr(value): index for index, value in enumerate(unique)}
    encoded = np.asarray([mapping[repr(value)] for value in values], dtype=np.int64)
    return encoded, mapping


def _quality_split_indices(
    labels: np.ndarray,
    *,
    evaluation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return deterministic paired train/evaluation indices.

    The split is shared by the original and tokenized representations.  It
    preserves at least one training graph and, whenever possible, keeps one
    member of each repeated class in the training set.
    """

    count = int(labels.shape[0])
    if count < 2:
        raise ValueError(
            "Held-out GNN quality evaluation requires at least two graphs."
        )
    evaluation_count = min(
        max(1, int(round(count * float(evaluation_fraction)))),
        count - 1,
    )
    generator = np.random.default_rng(int(seed))
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        groups.setdefault(int(label), []).append(index)
    for members in groups.values():
        generator.shuffle(members)

    evaluation: list[int] = []
    repeated = [
        (label, members)
        for label, members in sorted(groups.items())
        if len(members) >= 2
    ]
    if evaluation_count >= len(repeated):
        for _, members in repeated:
            evaluation.append(members.pop())

    candidates: list[int] = []
    protected: list[int] = []
    for _, members in sorted(groups.items()):
        if len(members) > 1:
            candidates.extend(members[1:])
            protected.append(members[0])
        else:
            protected.extend(members)
    generator.shuffle(candidates)
    evaluation.extend(candidates[: max(0, evaluation_count - len(evaluation))])

    if len(evaluation) < evaluation_count:
        remaining = [index for index in protected if index not in evaluation]
        generator.shuffle(remaining)
        evaluation.extend(remaining[: evaluation_count - len(evaluation)])

    evaluation_set = set(evaluation[:evaluation_count])
    training = [index for index in range(count) if index not in evaluation_set]
    evaluation_sorted = sorted(evaluation_set)
    if not training or not evaluation_sorted:
        raise RuntimeError("Could not construct a nonempty train/evaluation split.")
    return training, evaluation_sorted


def _gnn_paths(prepared_dir: str | Path) -> dict[str, Path]:
    root = Path(prepared_dir)
    return {
        "root": root,
        "state": root / "prepared_state.json",
        "config": root / "gnn_config.json",
        "metadata": root / "metadata.json",
        "compression": root / "compression_summary.json",
        "graph_sizes": root / "graph_size_comparison.csv",
        "original": root / "original_gnn_payload.pkl",
        "tokenized": root / "tokenized_gnn_payload.pkl",
        "tasks_csv": root / "task_manifest.csv",
        "tasks_json": root / "task_manifest.json",
        "results": root / "task_results",
    }


def _runtime_paths(runtime_prepared_dir: str | Path) -> dict[str, Path]:
    root = Path(runtime_prepared_dir)
    return {
        "root": root,
        "state": root / "prepared_state.json",
        "metadata": root / "metadata.json",
        "compression": root / "compression_summary.json",
        "graph_sizes": root / "graph_size_comparison.csv",
        "original": root / "original_payload.pkl",
        "tokenized": root / "tokenized_payload.pkl",
    }


def _validate_runtime_source(paths: Mapping[str, Path]) -> dict[str, Any]:
    required = (
        "state",
        "metadata",
        "compression",
        "graph_sizes",
        "original",
        "tokenized",
    )
    missing = [str(paths[name]) for name in required if not paths[name].is_file()]
    if missing:
        raise PreparedStateError(
            "Runtime prepared files are missing: " + ", ".join(missing)
        )
    state = json.loads(paths["state"].read_text())
    if _sha256_file(paths["original"]) != state["original_payload_sha256"]:
        raise PreparedStateError("Original runtime payload fingerprint mismatch.")
    if _sha256_file(paths["tokenized"]) != state["tokenized_payload_sha256"]:
        raise PreparedStateError("Tokenized runtime payload fingerprint mismatch.")
    return state


def prepare_gnn_benchmark(
    runtime_prepared_dir: str | Path,
    *,
    prepared_dir: str | Path,
    config: GNNBenchmarkConfig | None = None,
    labels: Sequence[Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Prepare immutable GNN inputs from a runtime benchmark state."""

    active = config or GNNBenchmarkConfig()
    active.validate()
    runtime_paths = _runtime_paths(runtime_prepared_dir)
    runtime_state = _validate_runtime_source(runtime_paths)
    original = _read_pickle(runtime_paths["original"])["graphs"]
    tokenized = _read_pickle(runtime_paths["tokenized"])["graphs"]
    if len(original) != len(tokenized):
        raise PreparedStateError(
            "Original and tokenized runtime payloads contain different graph counts."
        )
    if not original:
        raise ValueError("The runtime prepared state contains no evaluation graphs.")

    labels_available = labels is not None
    if labels is None:
        if active.mode == "training":
            raise ValueError(
                "Training timing requires graph labels. Supply TU labels or an "
                "explicit label sequence."
            )
        encoded_labels = np.zeros(len(original), dtype=np.int64)
        label_mapping = {"timing-only": 0}
    else:
        if len(labels) != len(original):
            raise ValueError(
                "Label count does not match the prepared evaluation graph count."
            )
        encoded_labels, label_mapping = _contiguous_labels(labels)

    paths = _gnn_paths(prepared_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)

    if active.mode == "training":
        train_indices, quality_indices = _quality_split_indices(
            encoded_labels,
            evaluation_fraction=active.quality_eval_fraction,
            seed=active.seed,
        )
        quality_metrics_available = True
        quality_metrics_reason = "paired deterministic held-out evaluation"
    else:
        train_indices = list(range(len(original)))
        quality_indices = []
        quality_metrics_available = False
        quality_metrics_reason = (
            "Inference mode measures an untrained timing workload; predictive "
            "quality is intentionally unavailable. Use --gnn-mode training "
            "with labels for held-out quality metrics."
        )

    import time

    started = time.perf_counter()
    original_batches = prepare_gnn_batches(
        original, encoded_labels, batch_size=active.batch_size
    )
    original_train_batches = prepare_gnn_batches(
        [original[index] for index in train_indices],
        encoded_labels[train_indices],
        batch_size=active.batch_size,
    )
    original_quality_batches = (
        prepare_gnn_batches(
            [original[index] for index in quality_indices],
            encoded_labels[quality_indices],
            batch_size=active.batch_size,
        )
        if quality_indices
        else []
    )
    original_preprocessing = time.perf_counter() - started

    started = time.perf_counter()
    tokenized_batches = prepare_gnn_batches(
        tokenized, encoded_labels, batch_size=active.batch_size
    )
    tokenized_train_batches = prepare_gnn_batches(
        [tokenized[index] for index in train_indices],
        encoded_labels[train_indices],
        batch_size=active.batch_size,
    )
    tokenized_quality_batches = (
        prepare_gnn_batches(
            [tokenized[index] for index in quality_indices],
            encoded_labels[quality_indices],
            batch_size=active.batch_size,
        )
        if quality_indices
        else []
    )
    tokenized_preprocessing = time.perf_counter() - started

    common = {
        "config": asdict(active),
        "num_classes": max(int(encoded_labels.max()) + 1, 2),
        "labels_available": labels_available,
        "feature_dim": 5,
        "quality_metrics_available": quality_metrics_available,
        "quality_metrics_reason": quality_metrics_reason,
        "train_indices": train_indices,
        "quality_eval_indices": quality_indices,
    }
    _atomic_pickle(
        paths["original"],
        {
            **common,
            "representation": "original",
            "batches": original_batches,
            "train_batches": original_train_batches,
            "quality_batches": original_quality_batches,
        },
    )
    _atomic_pickle(
        paths["tokenized"],
        {
            **common,
            "representation": "tokenized",
            "batches": tokenized_batches,
            "train_batches": tokenized_train_batches,
            "quality_batches": tokenized_quality_batches,
        },
    )
    _atomic_json(paths["config"], asdict(active))
    compression = _strict_json_value(
        json.loads(runtime_paths["compression"].read_text())
    )
    _atomic_json(paths["compression"], compression)
    paths["graph_sizes"].write_bytes(runtime_paths["graph_sizes"].read_bytes())

    source_metadata = _strict_json_value(
        json.loads(runtime_paths["metadata"].read_text())
    )
    metadata_value = {
        **source_metadata,
        **dict(metadata or {}),
        "runtime_prepared_dir": str(Path(runtime_prepared_dir).resolve()),
        "runtime_prepared_fingerprint": runtime_state["prepared_fingerprint"],
        "gnn_config": asdict(active),
        "label_mapping": label_mapping,
        "labels_available": labels_available,
        "quality_metrics_available": quality_metrics_available,
        "quality_metrics_reason": quality_metrics_reason,
        "quality_protocol": (
            "paired-representation-specific-held-out"
            if active.mode == "training"
            else "timing-only"
        ),
        "quality_eval_fraction": active.quality_eval_fraction,
        "train_indices": train_indices,
        "quality_eval_indices": quality_indices,
        "train_graph_count": len(train_indices),
        "quality_eval_graph_count": len(quality_indices),
        "original_preprocessing_seconds": original_preprocessing,
        "tokenized_preprocessing_seconds": tokenized_preprocessing,
        "interpretation_boundary": (
            "The built-in structural GCN measures computational cost. "
            "Tokenized and original graphs are different model inputs; runtime "
            "speedup alone does not establish predictive equivalence."
        ),
    }
    metadata_value = _strict_json_value(metadata_value)
    _atomic_json(paths["metadata"], metadata_value)

    fingerprint_payload = {
        "format_version": 2,
        "runtime_prepared_fingerprint": runtime_state["prepared_fingerprint"],
        "config": asdict(active),
        "original_sha256": _sha256_file(paths["original"]),
        "tokenized_sha256": _sha256_file(paths["tokenized"]),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    state = {
        **fingerprint_payload,
        "prepared_fingerprint": fingerprint,
        "task_count": 2 * active.repeats,
    }
    _atomic_json(paths["state"], state)
    tasks = create_task_manifest(
        repeats=active.repeats,
        prepared_fingerprint=fingerprint,
    )
    tasks.to_csv(paths["tasks_csv"], index=False)
    _atomic_json(paths["tasks_json"], tasks.to_dict(orient="records"))
    return paths["root"]


def _validate_gnn_state(paths: Mapping[str, Path]) -> dict[str, Any]:
    required = (
        "state",
        "config",
        "metadata",
        "compression",
        "graph_sizes",
        "original",
        "tokenized",
        "tasks_csv",
    )
    missing = [str(paths[name]) for name in required if not paths[name].is_file()]
    if missing:
        raise PreparedStateError(
            "GNN prepared files are missing: " + ", ".join(missing)
        )
    state = json.loads(paths["state"].read_text())
    if _sha256_file(paths["original"]) != state["original_sha256"]:
        raise PreparedStateError("Original GNN payload fingerprint mismatch.")
    if _sha256_file(paths["tokenized"]) != state["tokenized_sha256"]:
        raise PreparedStateError("Tokenized GNN payload fingerprint mismatch.")
    return state


def _run_gnn_worker(
    *,
    payload_path: Path,
    result_path: Path,
    threads: int,
    timeout: float | None,
    phase_name: str,
) -> dict[str, Any]:
    """Run one GNN worker without background output-capture threads."""

    command = [
        sys.executable,
        "-m",
        "buhito.benchmarks._gnn_worker",
        "--payload",
        str(payload_path),
        "--result",
        str(result_path),
    ]
    result_path.unlink(missing_ok=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_subprocess_environment(threads),
        start_new_session=True,
    )
    try:
        stdout_text, stderr_text = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        stdout_text, stderr_text = process.communicate()
        timeout_text = (
            "without a configured limit"
            if timeout is None
            else f"after {timeout:g} seconds"
        )
        raise WorkerTimeoutError(
            f"Phase '{phase_name}' exceeded its timeout {timeout_text}. "
            "The GNN worker process group was terminated."
        ) from exc
    if process.returncode != 0:
        raise RuntimeError(
            f"Phase '{phase_name}' failed with exit code {process.returncode}.\n"
            f"stdout:\n{stdout_text}\n"
            f"stderr:\n{stderr_text}"
        )
    if not result_path.is_file():
        raise RuntimeError(
            f"Phase '{phase_name}' completed without writing {result_path}."
        )
    return json.loads(result_path.read_text())


def run_gnn_prepared_task(
    prepared_dir: str | Path,
    *,
    task_id: int | None = None,
    task_manifest: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> Path:
    """Run one isolated GNN measurement and atomically save the result."""

    paths = _gnn_paths(prepared_dir)
    state = _validate_gnn_state(paths)
    tasks = load_task_manifest(prepared_dir, task_manifest)
    resolved = resolve_task_id(task_id)
    matches = tasks.loc[tasks["task_id"].astype(int) == resolved]
    if len(matches) != 1:
        raise PreparedStateError(f"Task ID {resolved} is not present exactly once.")
    task = matches.iloc[0].to_dict()
    if task["prepared_fingerprint"] != state["prepared_fingerprint"]:
        raise PreparedStateError(
            "GNN task fingerprint is incompatible with prepared state."
        )
    config = GNNBenchmarkConfig(**json.loads(paths["config"].read_text()))
    payload_path = paths[str(task["representation"])]
    repeat = int(task["repeat"])
    model_seed = int(config.seed) + repeat
    destination_root = Path(results_dir) if results_dir else paths["results"]
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"task_{resolved:05d}.json"
    with tempfile.TemporaryDirectory(prefix=f"buhito-gnn-{resolved}-") as temporary:
        temporary_root = Path(temporary)
        task_payload = _read_pickle(payload_path)
        task_payload["config"] = dict(task_payload["config"])
        task_payload["config"]["seed"] = model_seed
        task_payload_path = temporary_root / "task_payload.pkl"
        _atomic_pickle(task_payload_path, task_payload)
        measured = _run_gnn_worker(
            payload_path=task_payload_path,
            result_path=temporary_root / "measured.json",
            threads=config.threads,
            timeout=config.phase_timeout_seconds,
            phase_name=(
                f"GNN task {resolved} {task['representation']} "
                f"repeat {repeat + 1}"
            ),
        )
    measured.update(
        {
            "task_id": resolved,
            "representation": task["representation"],
            "repeat": repeat,
            "model_seed": model_seed,
            "order_position": int(task["order_position"]),
            "prepared_fingerprint": state["prepared_fingerprint"],
        }
    )
    _atomic_json(destination, measured)
    return destination


def run_gnn_prepared_tasks(
    prepared_dir: str | Path,
    *,
    jobs: int = 1,
    task_manifest: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> list[Path]:
    """Run all GNN tasks with explicitly bounded local concurrency."""

    if jobs < 1:
        raise ValueError("jobs must be at least one.")
    tasks = load_task_manifest(prepared_dir, task_manifest)
    task_ids = [int(value) for value in tasks["task_id"]]
    if jobs == 1:
        return [
            run_gnn_prepared_task(
                prepared_dir,
                task_id=task_id,
                task_manifest=task_manifest,
                results_dir=results_dir,
            )
            for task_id in task_ids
        ]
    outputs: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=min(jobs, len(task_ids))) as executor:
        future_to_id = {
            executor.submit(
                run_gnn_prepared_task,
                prepared_dir,
                task_id=task_id,
                task_manifest=task_manifest,
                results_dir=results_dir,
            ): task_id
            for task_id in task_ids
        }
        for future in as_completed(future_to_id):
            task_id = future_to_id[future]
            outputs[task_id] = future.result()
    return [outputs[task_id] for task_id in task_ids]


def _quartile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    return float(np.quantile(np.asarray(ordered), fraction))


def _finite_median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.median()) if not numeric.empty else None


def _column_median(group: pd.DataFrame, name: str) -> float | None:
    if name not in group.columns:
        return None
    return _finite_median(group[name])


def summarize_gnn_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for representation, group in runs.groupby("representation", sort=False):
        times = group["workload_seconds"].astype(float).tolist()
        quality_available = bool(
            "quality_metrics_available" in group.columns
            and group["quality_metrics_available"].astype(bool).all()
        )
        quality_reason = (
            str(group["quality_metrics_reason"].iloc[0])
            if "quality_metrics_reason" in group.columns
            else "quality metadata unavailable"
        )
        rows.append(
            {
                "representation": representation,
                "repeats": len(group),
                "median_workload_seconds": float(np.median(times)),
                "q1_workload_seconds": _quartile(times, 0.25),
                "q3_workload_seconds": _quartile(times, 0.75),
                "median_peak_rss_mb": float(
                    group["peak_rss_mb"].astype(float).median()
                ),
                "median_cuda_peak_memory_mb": float(
                    group["cuda_peak_memory_mb"].astype(float).median()
                ),
                "median_graphs_per_second": float(
                    group["graphs_per_second"].astype(float).median()
                ),
                "median_nodes_per_second": float(
                    group["nodes_per_second"].astype(float).median()
                ),
                "median_edges_per_second": float(
                    group["edges_per_second"].astype(float).median()
                ),
                "median_final_loss": _column_median(group, "final_loss"),
                "median_accuracy": _column_median(group, "accuracy"),
                "median_final_train_loss": _column_median(
                    group, "final_train_loss"
                ),
                "median_final_train_accuracy": _column_median(
                    group, "final_train_accuracy"
                ),
                "median_quality_eval_loss": _column_median(
                    group, "quality_eval_loss"
                ),
                "median_quality_eval_accuracy": _column_median(
                    group, "quality_eval_accuracy"
                ),
                "median_quality_eval_macro_f1": _column_median(
                    group, "quality_eval_macro_f1"
                ),
                "median_quality_evaluation_seconds": _column_median(
                    group, "quality_evaluation_seconds"
                ),
                "quality_metrics_available": quality_available,
                "quality_metrics_reason": quality_reason,
                "quality_eval_graphs": int(
                    group.get("quality_eval_graphs", pd.Series([0])).iloc[0]
                ),
                "total_graphs": int(group["total_graphs"].iloc[0]),
                "total_nodes": int(group["total_nodes"].iloc[0]),
                "total_edges": int(group["total_edges"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def workload_tradeoff_summary(
    *,
    original_seconds: float,
    tokenized_seconds: float,
    compression_seconds: float,
) -> dict[str, float]:
    """Summarize speedup and amortization for any downstream workload.

    This framework-neutral helper is useful when practitioners time their own
    GNN, graph kernel, solver, or database query outside the built-in GCN.
    Inputs should be comparable robust statistics, normally medians from the
    same number of isolated repeats on the same hardware.
    """

    original = float(original_seconds)
    tokenized = float(tokenized_seconds)
    compression = float(compression_seconds)
    if original < 0 or tokenized < 0 or compression < 0:
        raise ValueError("Timing values cannot be negative.")
    speedup = original / tokenized if tokenized > 0 else math.inf
    saved = original - tokenized
    break_even = compression / saved if saved > 0 else math.inf
    return {
        "original_seconds": original,
        "tokenized_seconds": tokenized,
        "speedup": speedup,
        "time_saved_seconds_per_use": saved,
        "compression_seconds": compression,
        "first_tokenized_use_seconds": compression + tokenized,
        "break_even_reuses": break_even,
    }


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _optional_delta(left: Any, right: Any) -> float | None:
    left_value = _optional_float(left)
    right_value = _optional_float(right)
    if left_value is None or right_value is None:
        return None
    return right_value - left_value


def gnn_scalar_summary(result: GNNBenchmarkResult) -> dict[str, Any]:
    indexed = result.summary.set_index("representation")
    raw = float(indexed.loc["original", "median_workload_seconds"])
    token = float(indexed.loc["tokenized", "median_workload_seconds"])
    compression_seconds = float(result.compression["compression_total_seconds"])
    tradeoff = workload_tradeoff_summary(
        original_seconds=raw,
        tokenized_seconds=token,
        compression_seconds=compression_seconds,
    )
    raw_nodes = int(result.graph_sizes["raw_nodes"].sum())
    token_nodes = int(result.graph_sizes["token_nodes"].sum())
    raw_edges = int(result.graph_sizes["raw_edges"].sum())
    token_edges = int(result.graph_sizes["token_edges"].sum())

    quality_available = bool(
        indexed.loc["original", "quality_metrics_available"]
        and indexed.loc["tokenized", "quality_metrics_available"]
    )
    quality_reason = str(
        indexed.loc["original", "quality_metrics_reason"]
    )
    raw_quality_accuracy = _optional_float(
        indexed.loc["original", "median_quality_eval_accuracy"]
    )
    token_quality_accuracy = _optional_float(
        indexed.loc["tokenized", "median_quality_eval_accuracy"]
    )
    raw_quality_f1 = _optional_float(
        indexed.loc["original", "median_quality_eval_macro_f1"]
    )
    token_quality_f1 = _optional_float(
        indexed.loc["tokenized", "median_quality_eval_macro_f1"]
    )
    raw_quality_loss = _optional_float(
        indexed.loc["original", "median_quality_eval_loss"]
    )
    token_quality_loss = _optional_float(
        indexed.loc["tokenized", "median_quality_eval_loss"]
    )

    return {
        "gnn_mode": result.config.mode,
        "raw_median_workload_seconds": raw,
        "tokenized_median_workload_seconds": token,
        "gnn_speedup": _optional_float(tradeoff["speedup"]),
        "gnn_time_saved_seconds_per_repeat": tradeoff[
            "time_saved_seconds_per_use"
        ],
        "compression_total_seconds": compression_seconds,
        "first_tokenized_pass_total_seconds": compression_seconds + token,
        "gnn_break_even_reuses": _optional_float(
            tradeoff["break_even_reuses"]
        ),
        "raw_median_peak_rss_mb": float(
            indexed.loc["original", "median_peak_rss_mb"]
        ),
        "tokenized_median_peak_rss_mb": float(
            indexed.loc["tokenized", "median_peak_rss_mb"]
        ),
        "raw_median_cuda_peak_memory_mb": float(
            indexed.loc["original", "median_cuda_peak_memory_mb"]
        ),
        "tokenized_median_cuda_peak_memory_mb": float(
            indexed.loc["tokenized", "median_cuda_peak_memory_mb"]
        ),
        "total_raw_nodes": raw_nodes,
        "total_token_nodes": token_nodes,
        "total_raw_edges": raw_edges,
        "total_token_edges": token_edges,
        "node_reduction_fraction": 1.0 - token_nodes / max(raw_nodes, 1),
        "edge_reduction_fraction": 1.0 - token_edges / max(raw_edges, 1),
        "mdl_net_savings_bits": float(
            result.compression["mdl_net_savings_bits"]
        ),
        "selected_rule_count": int(result.compression["selected_rule_count"]),
        "tokenized_graph_count": int(
            result.compression["tokenized_graph_count"]
        ),
        "quality_metrics_available": quality_available,
        "quality_metrics_reason": quality_reason,
        "quality_protocol": result.metadata.get(
            "quality_protocol", "unknown"
        ),
        "quality_eval_graph_count": int(
            result.metadata.get("quality_eval_graph_count", 0)
        ),
        "train_graph_count": int(result.metadata.get("train_graph_count", 0)),
        "raw_median_final_train_loss": _optional_float(
            indexed.loc["original", "median_final_train_loss"]
        ),
        "tokenized_median_final_train_loss": _optional_float(
            indexed.loc["tokenized", "median_final_train_loss"]
        ),
        "raw_median_final_train_accuracy": _optional_float(
            indexed.loc["original", "median_final_train_accuracy"]
        ),
        "tokenized_median_final_train_accuracy": _optional_float(
            indexed.loc["tokenized", "median_final_train_accuracy"]
        ),
        "raw_median_quality_eval_loss": raw_quality_loss,
        "tokenized_median_quality_eval_loss": token_quality_loss,
        "quality_eval_loss_delta": _optional_delta(
            raw_quality_loss, token_quality_loss
        ),
        "raw_median_quality_eval_accuracy": raw_quality_accuracy,
        "tokenized_median_quality_eval_accuracy": token_quality_accuracy,
        "quality_eval_accuracy_delta": _optional_delta(
            raw_quality_accuracy, token_quality_accuracy
        ),
        "raw_median_quality_eval_macro_f1": raw_quality_f1,
        "tokenized_median_quality_eval_macro_f1": token_quality_f1,
        "quality_eval_macro_f1_delta": _optional_delta(
            raw_quality_f1, token_quality_f1
        ),
        # Backward-compatible names now refer specifically to held-out quality.
        "raw_median_accuracy": raw_quality_accuracy,
        "tokenized_median_accuracy": token_quality_accuracy,
        "accuracy_delta": _optional_delta(
            raw_quality_accuracy, token_quality_accuracy
        ),
    }


def _metric_text(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "not available"
    return f"{value:.2%}" if percent else f"{value:.6f}"


def render_gnn_summary(result: GNNBenchmarkResult) -> str:
    summary = gnn_scalar_summary(result)
    break_even = summary["gnn_break_even_reuses"]
    break_even_text = (
        "not reached because the tokenized GNN workload was not faster"
        if break_even is None
        else f"{break_even:.3f} repeated GNN workloads"
    )
    if summary["quality_metrics_available"]:
        quality_section = f"""## Held-out predictive quality

The original and tokenized representations used the same deterministic train
and evaluation indices. Separate models with the same architecture and seed
were trained for each representation. Quality evaluation was performed after
the timed training region and therefore is not included in `workload_seconds`.

- Training graphs: `{summary['train_graph_count']}`
- Held-out evaluation graphs: `{summary['quality_eval_graph_count']}`
- Original held-out accuracy: `{_metric_text(summary['raw_median_quality_eval_accuracy'], percent=True)}`
- Tokenized held-out accuracy: `{_metric_text(summary['tokenized_median_quality_eval_accuracy'], percent=True)}`
- Accuracy delta (tokenized - original): `{_metric_text(summary['quality_eval_accuracy_delta'], percent=True)}`
- Original held-out macro-F1: `{_metric_text(summary['raw_median_quality_eval_macro_f1'])}`
- Tokenized held-out macro-F1: `{_metric_text(summary['tokenized_median_quality_eval_macro_f1'])}`
- Macro-F1 delta (tokenized - original): `{_metric_text(summary['quality_eval_macro_f1_delta'])}`
- Original held-out loss: `{_metric_text(summary['raw_median_quality_eval_loss'])}`
- Tokenized held-out loss: `{_metric_text(summary['tokenized_median_quality_eval_loss'])}`

For smoke runs, these quality values only verify that the evaluation pipeline
works. They are not statistically meaningful when the held-out set contains
only one or two graphs.
"""
    else:
        quality_section = f"""## Predictive quality

Predictive quality is not available for this run.

Reason: `{summary['quality_metrics_reason']}`

Inference mode uses an untrained reference network solely to measure systems
cost. Accuracy from an untrained model would be misleading, so the JSON value
is `null` rather than `NaN`.
"""

    return f"""# Buhito downstream GNN benchmark

## Interpretation boundary

This benchmark compares the same reference structural GCN architecture on the
original and reversible motif-tokenized graph representations. It measures
computational cost. It does **not** assert that the two representations have the
same predictive semantics. For scientific model claims, report held-out task
quality together with runtime.

## Headline results

- GNN mode: `{summary['gnn_mode']}`
- Original median time: `{summary['raw_median_workload_seconds']:.6f}` seconds
- Tokenized median time: `{summary['tokenized_median_workload_seconds']:.6f}` seconds
- GNN speedup: `{summary['gnn_speedup']:.6f}x`
- Time saved per repeated workload: `{summary['gnn_time_saved_seconds_per_repeat']:.6f}` seconds
- One-time compression cost: `{summary['compression_total_seconds']:.6f}` seconds
- Compression break-even: {break_even_text}
- Node reduction: `{summary['node_reduction_fraction']:.6%}`
- Edge reduction: `{summary['edge_reduction_fraction']:.6%}`
- Analytical MDL savings: `{summary['mdl_net_savings_bits']:.6f}` bits

Negative MDL savings and positive GNN speedup can coexist. In that case the
representation is computationally useful but is not a smaller complete MDL
code.

{quality_section}
"""


def aggregate_gnn_results(
    prepared_dir: str | Path,
    *,
    task_manifest: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> GNNBenchmarkResult:
    paths = _gnn_paths(prepared_dir)
    state = _validate_gnn_state(paths)
    tasks = load_task_manifest(prepared_dir, task_manifest)
    result_root = Path(results_dir) if results_dir else paths["results"]
    expected = set(tasks["task_id"].astype(int))
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for result_file in sorted(result_root.glob("task_*.json")):
        row = json.loads(result_file.read_text())
        task_id = int(row.get("task_id", -1))
        if task_id in seen:
            raise PreparedStateError(f"Duplicate GNN task result {task_id}.")
        seen.add(task_id)
        if row.get("prepared_fingerprint") != state["prepared_fingerprint"]:
            raise PreparedStateError(
                f"GNN task {task_id} has an incompatible fingerprint."
            )
        rows.append(row)
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    if missing:
        raise PreparedStateError(f"Missing GNN task results: {missing}")
    if unexpected:
        raise PreparedStateError(f"Unexpected GNN task results: {unexpected}")
    runs = pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)
    config = GNNBenchmarkConfig(**json.loads(paths["config"].read_text()))
    return GNNBenchmarkResult(
        config=config,
        runs=runs,
        summary=summarize_gnn_runs(runs),
        compression=json.loads(paths["compression"].read_text()),
        graph_sizes=pd.read_csv(paths["graph_sizes"]),
        metadata=json.loads(paths["metadata"].read_text()),
    )


def write_gnn_slurm_array_script(
    prepared_dir: str | Path,
    script_path: str | Path,
    *,
    cli_path: str = "examples/benchmark_gnn_tokenization.py",
) -> Path:
    tasks = load_task_manifest(prepared_dir)
    maximum = int(tasks["task_id"].max())
    prepared = Path(prepared_dir).resolve()
    script = Path(script_path)
    text = (
        "#!/usr/bin/env bash\n"
        "#SBATCH --job-name=buhito-gnn\n"
        f"#SBATCH --array=0-{maximum}\n"
        "#SBATCH --cpus-per-task=1\n"
        "#SBATCH --gres=gpu:SITE_GPU_COUNT\n"
        "#SBATCH --time=SITE_WALLTIME\n"
        "#SBATCH --mem=SITE_MEMORY\n"
        "##SBATCH --partition=SITE_PARTITION\n"
        "##SBATCH --account=SITE_ACCOUNT\n\n"
        "set -euo pipefail\n\n"
        "# Activate the site-specific environment before this command.\n"
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

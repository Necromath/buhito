"""Isolated runtime and memory benchmarks for MDL-tokenized graph datasets.

The benchmark deliberately separates three costs:

1. fitting and applying an MDL dictionary;
2. enumerating graphlets on the original evaluation graphs; and
3. enumerating graphlets on the tokenized evaluation representation.

The tokenized graphlet representation is not claimed to reproduce the original
raw graphlet counts. It is a different, reversible graph representation whose
computational cost and downstream utility must be evaluated separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable, Literal, Mapping, Sequence

import networkx as nx
import pandas as pd

from buhito.mdl import _EDGE_LABEL


BackendName = Literal["buhito", "exhaustive"]
TokenProjection = Literal["native", "simple"]
WorkerAction = Literal["compress", "enumerate", "sleep"]

SAMPLE_MANIFEST_COLUMNS = (
    "dataset",
    "seed",
    "split",
    "sample_position",
    "dataset_index",
    "size_bin",
    "nodes",
    "edges",
    "max_degree",
    "wedge_proxy",
    "eligible_graph_count",
    "dataset_graph_count",
)

STANDARD_BENCHMARK_DEFAULTS: dict[str, Any] = {
    "fit_size": 20,
    "eval_size": 20,
    "size_bins": 3,
    "max_nodes": None,
    "max_edges": None,
    "max_degree": None,
    "max_wedges": None,
    "n_rules": 3,
    "max_candidates": 20,
    "repeats": 3,
    "warmup_repeats": 1,
    "threads": 1,
    "phase_timeout_seconds": None,
}

SMOKE_BENCHMARK_DEFAULTS: dict[str, Any] = {
    "fit_size": 2,
    "eval_size": 2,
    "size_bins": 1,
    "max_nodes": 500,
    "max_edges": 750,
    "max_degree": 200,
    "max_wedges": 50000,
    "n_rules": 1,
    "max_candidates": 2,
    "repeats": 1,
    "warmup_repeats": 0,
    "threads": 1,
    "phase_timeout_seconds": 600.0,
}


class InsufficientEligibleGraphsError(ValueError):
    """Raised when safety caps leave too few graphs for a requested sample."""


class WorkerTimeoutError(RuntimeError):
    """Raised when an isolated benchmark worker exceeds its phase timeout."""


@dataclass(frozen=True)
class RuntimeBenchmarkConfig:
    """Configuration for an original-versus-tokenized enumeration benchmark."""

    graphlet_sizes: tuple[int, ...] = (3,)
    backend: BackendName = "buhito"
    compressor_backend: BackendName = "buhito"
    repeats: int = 3
    warmup_repeats: int = 1
    token_projection: TokenProjection = "simple"
    force_rewrite: bool = True
    threads: int = 1
    phase_timeout_seconds: float | None = None
    progress: bool = True

    def validate(self) -> None:
        if not self.graphlet_sizes:
            raise ValueError("graphlet_sizes cannot be empty.")
        if any(int(size) < 2 for size in self.graphlet_sizes):
            raise ValueError("Every graphlet size must be at least two.")
        if self.repeats < 1:
            raise ValueError("repeats must be at least one.")
        if self.warmup_repeats < 0:
            raise ValueError("warmup_repeats cannot be negative.")
        if self.threads < 1:
            raise ValueError("threads must be at least one.")
        if (
            self.phase_timeout_seconds is not None
            and self.phase_timeout_seconds <= 0
        ):
            raise ValueError("phase_timeout_seconds must be positive.")
        if self.backend not in {"buhito", "exhaustive"}:
            raise ValueError(f"Unsupported backend: {self.backend!r}.")
        if self.compressor_backend not in {"buhito", "exhaustive"}:
            raise ValueError(
                f"Unsupported compressor backend: {self.compressor_backend!r}."
            )
        if self.token_projection not in {"native", "simple"}:
            raise ValueError(
                f"Unsupported token_projection: {self.token_projection!r}."
            )


@dataclass
class RuntimeBenchmarkResult:
    """Complete benchmark result and publication-oriented derived summaries."""

    config: RuntimeBenchmarkConfig
    compression: dict[str, Any]
    runs: pd.DataFrame
    summary: pd.DataFrame
    graph_sizes: pd.DataFrame
    metadata: dict[str, Any]
    sample_manifest: pd.DataFrame = field(default_factory=pd.DataFrame)

    def save(self, output_dir: str | Path, *, plots: bool = False) -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        self.runs.to_csv(output / "benchmark_runs.csv", index=False)
        self.summary.to_csv(output / "benchmark_summary.csv", index=False)
        self.graph_sizes.to_csv(output / "graph_size_comparison.csv", index=False)
        save_sample_manifest(self.sample_manifest, output)
        (output / "compression_summary.json").write_text(
            json.dumps(self.compression, indent=2, sort_keys=True, default=str)
            + "\n"
        )
        (output / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True, default=str)
            + "\n"
        )
        (output / "summary.json").write_text(
            json.dumps(
                benchmark_scalar_summary(self),
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )
        (output / "README.md").write_text(render_runtime_summary(self))

        if plots:
            save_runtime_plots(self, output)

        return output


def resolve_benchmark_options(
    values: Mapping[str, Any],
    *,
    smoke: bool,
) -> dict[str, Any]:
    """Resolve smoke or standard defaults without overriding explicit values.

    CLI options controlled by the smoke preset are parsed with ``None`` as the
    sentinel for "not supplied". Any non-``None`` value is treated as explicit
    and preserved.
    """

    resolved = dict(values)
    defaults = (
        {**STANDARD_BENCHMARK_DEFAULTS, **SMOKE_BENCHMARK_DEFAULTS}
        if smoke
        else STANDARD_BENCHMARK_DEFAULTS
    )
    for name, default in defaults.items():
        if resolved.get(name) is None:
            resolved[name] = default
    return resolved


def _canonical_pair(source: Any, target: Any) -> tuple[Any, Any]:
    return (
        (source, target)
        if repr(source) <= repr(target)
        else (target, source)
    )


def relabel_for_enumeration(graph: nx.Graph) -> nx.Graph:
    """Copy a graph with consecutive integer node identifiers."""

    mapping = {node: index for index, node in enumerate(graph.nodes())}
    return nx.relabel_nodes(graph, mapping, copy=True)


def collapse_parallel_edges(graph: nx.Graph) -> nx.Graph:
    """Return a simple labeled projection of a tokenized graph."""

    if not graph.is_multigraph():
        return nx.Graph(graph)

    projected = nx.Graph()
    projected.graph.update(graph.graph)
    projected.add_nodes_from(
        (node, dict(data)) for node, data in graph.nodes(data=True)
    )

    grouped: dict[tuple[Any, Any], list[Any]] = {}
    for source, target, data in graph.edges(data=True):
        pair = _canonical_pair(source, target)
        grouped.setdefault(pair, []).append(data.get(_EDGE_LABEL))

    for (source, target), labels in grouped.items():
        ordered = tuple(sorted(labels, key=repr))
        projected.add_edge(
            source,
            target,
            **{
                _EDGE_LABEL: (
                    "buhito-token-edge-multiset",
                    ordered,
                )
            },
        )

    return projected


def graph_complexity_metrics(graph: nx.Graph) -> dict[str, int]:
    """Return inexpensive graph statistics used by sample safety filters."""

    degrees = [int(degree) for _, degree in graph.degree()]
    return {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "max_degree": max(degrees, default=0),
        "wedge_proxy": int(
            sum(degree * (degree - 1) // 2 for degree in degrees)
        ),
    }


def graph_complexity_frame(graphs: Sequence[nx.Graph]) -> pd.DataFrame:
    rows = []
    for dataset_index, graph in enumerate(graphs):
        rows.append(
            {
                "dataset_index": dataset_index,
                **graph_complexity_metrics(graph),
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "dataset_index",
            "nodes",
            "edges",
            "max_degree",
            "wedge_proxy",
        ),
    )


def active_sample_caps(
    *,
    max_nodes: int | None = None,
    max_edges: int | None = None,
    max_degree: int | None = None,
    max_wedges: int | None = None,
) -> dict[str, int]:
    caps = {
        "nodes": max_nodes,
        "edges": max_edges,
        "max_degree": max_degree,
        "wedge_proxy": max_wedges,
    }
    for name, value in caps.items():
        if value is not None and int(value) < 0:
            raise ValueError(f"{name} cap cannot be negative.")
    return {
        name: int(value)
        for name, value in caps.items()
        if value is not None
    }


def filter_graph_complexity(
    complexity: pd.DataFrame,
    *,
    max_nodes: int | None = None,
    max_edges: int | None = None,
    max_degree: int | None = None,
    max_wedges: int | None = None,
) -> pd.DataFrame:
    """Filter a graph-complexity table by all active safety caps."""

    caps = active_sample_caps(
        max_nodes=max_nodes,
        max_edges=max_edges,
        max_degree=max_degree,
        max_wedges=max_wedges,
    )
    eligible = complexity.copy()
    for column, cap in caps.items():
        eligible = eligible.loc[eligible[column].astype(int) <= cap]
    return eligible.reset_index(drop=True)


def _range_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "none"
    fragments = []
    for column in ("nodes", "edges", "max_degree", "wedge_proxy"):
        fragments.append(
            f"{column}={int(frame[column].min())}..{int(frame[column].max())}"
        )
    return ", ".join(fragments)


def insufficient_eligible_message(
    *,
    requested: int,
    eligible: pd.DataFrame,
    complexity: pd.DataFrame,
    caps: Mapping[str, int],
) -> str:
    cap_names = {
        "nodes": "max_nodes",
        "edges": "max_edges",
        "max_degree": "max_degree",
        "wedge_proxy": "max_wedges",
    }
    active = (
        ", ".join(
            f"{cap_names[name]}={value}" for name, value in caps.items()
        )
        if caps
        else "none"
    )
    return (
        "Too few eligible graphs after safety filtering.\n"
        f"Requested sample size: {requested}\n"
        f"Eligible graph count: {len(eligible)}\n"
        f"Active caps: {active}\n"
        f"Dataset-wide ranges: {_range_text(complexity)}\n"
        f"Eligible ranges: {_range_text(eligible)}"
    )


def size_bin_assignments(
    graphs: Sequence[nx.Graph],
    indices: Sequence[int],
    *,
    n_bins: int,
) -> dict[int, int]:
    if n_bins < 1:
        raise ValueError("n_bins must be at least one.")
    if not indices:
        return {}
    ordered = sorted(
        (int(index) for index in indices),
        key=lambda index: (
            graphs[index].number_of_nodes(),
            graphs[index].number_of_edges(),
            graph_complexity_metrics(graphs[index])["max_degree"],
            graph_complexity_metrics(graphs[index])["wedge_proxy"],
            index,
        ),
    )
    count = len(ordered)
    bin_count = min(n_bins, count)
    return {
        index: min(position * bin_count // count, bin_count - 1)
        for position, index in enumerate(ordered)
    }


def size_stratified_sample_indices(
    graphs: Sequence[nx.Graph],
    sample_size: int,
    *,
    seed: int = 0,
    n_bins: int = 3,
    candidate_indices: Sequence[int] | None = None,
) -> list[int]:
    """Sample approximately equally across eligible graph-size bins."""

    import random

    candidates = (
        list(range(len(graphs)))
        if candidate_indices is None
        else [int(index) for index in candidate_indices]
    )
    if sample_size < 1:
        raise ValueError("sample_size must be at least one.")
    if n_bins < 1:
        raise ValueError("n_bins must be at least one.")
    if sample_size > len(candidates):
        raise ValueError(
            f"sample_size={sample_size} exceeds candidate count "
            f"{len(candidates)}."
        )
    if sample_size == len(candidates):
        selected = list(candidates)
        random.Random(seed).shuffle(selected)
        return selected

    bin_map = size_bin_assignments(graphs, candidates, n_bins=n_bins)
    bins: dict[int, list[int]] = {}
    for index in candidates:
        bins.setdefault(bin_map[index], []).append(index)

    rng = random.Random(seed)
    ordered_bins = [bins[index] for index in sorted(bins)]
    base, remainder = divmod(sample_size, len(ordered_bins))
    selected: list[int] = []
    for bin_index, members in enumerate(ordered_bins):
        quota = base + (1 if bin_index < remainder else 0)
        quota = min(quota, len(members))
        selected.extend(rng.sample(members, quota))

    if len(selected) < sample_size:
        remaining = [index for index in candidates if index not in selected]
        selected.extend(rng.sample(remaining, sample_size - len(selected)))

    rng.shuffle(selected)
    return selected


def select_sample_manifest(
    graphs: Sequence[nx.Graph],
    *,
    dataset: str,
    fit_size: int,
    eval_size: int,
    seed: int,
    n_bins: int,
    max_nodes: int | None = None,
    max_edges: int | None = None,
    max_degree: int | None = None,
    max_wedges: int | None = None,
    complexity: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[int], list[int]]:
    """Filter, sample, split, and describe a deterministic benchmark sample."""

    if fit_size < 1 or eval_size < 1:
        raise ValueError("fit_size and eval_size must be positive.")
    full = graph_complexity_frame(graphs) if complexity is None else complexity
    eligible = filter_graph_complexity(
        full,
        max_nodes=max_nodes,
        max_edges=max_edges,
        max_degree=max_degree,
        max_wedges=max_wedges,
    )
    requested = int(fit_size + eval_size)
    caps = active_sample_caps(
        max_nodes=max_nodes,
        max_edges=max_edges,
        max_degree=max_degree,
        max_wedges=max_wedges,
    )
    if len(eligible) < requested:
        raise InsufficientEligibleGraphsError(
            insufficient_eligible_message(
                requested=requested,
                eligible=eligible,
                complexity=full,
                caps=caps,
            )
        )

    eligible_indices = eligible["dataset_index"].astype(int).tolist()
    selected = size_stratified_sample_indices(
        graphs,
        requested,
        seed=seed,
        n_bins=n_bins,
        candidate_indices=eligible_indices,
    )
    fit_indices = selected[:fit_size]
    eval_indices = selected[fit_size:]
    bin_map = size_bin_assignments(graphs, eligible_indices, n_bins=n_bins)
    metrics = full.set_index("dataset_index")

    rows: list[dict[str, Any]] = []
    for split, indices in (("fit", fit_indices), ("eval", eval_indices)):
        for position, dataset_index in enumerate(indices):
            record = metrics.loc[dataset_index]
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "split": split,
                    "sample_position": position,
                    "dataset_index": int(dataset_index),
                    "size_bin": int(bin_map[dataset_index]),
                    "nodes": int(record["nodes"]),
                    "edges": int(record["edges"]),
                    "max_degree": int(record["max_degree"]),
                    "wedge_proxy": int(record["wedge_proxy"]),
                    "eligible_graph_count": len(eligible),
                    "dataset_graph_count": len(graphs),
                }
            )
    manifest = pd.DataFrame(rows, columns=SAMPLE_MANIFEST_COLUMNS)
    return manifest, fit_indices, eval_indices


def default_sample_manifest(
    fit_graphs: Sequence[nx.Graph],
    eval_graphs: Sequence[nx.Graph],
    *,
    dataset: str = "unspecified",
    seed: int = 0,
) -> pd.DataFrame:
    combined = list(fit_graphs) + list(eval_graphs)
    rows: list[dict[str, Any]] = []
    for split, graphs, offset in (
        ("fit", fit_graphs, 0),
        ("eval", eval_graphs, len(fit_graphs)),
    ):
        for position, graph in enumerate(graphs):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "split": split,
                    "sample_position": position,
                    "dataset_index": offset + position,
                    "size_bin": 0,
                    **graph_complexity_metrics(graph),
                    "eligible_graph_count": len(combined),
                    "dataset_graph_count": len(combined),
                }
            )
    return pd.DataFrame(rows, columns=SAMPLE_MANIFEST_COLUMNS)


def save_sample_manifest(
    manifest: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    normalized = manifest.copy()
    missing = [column for column in SAMPLE_MANIFEST_COLUMNS if column not in normalized]
    if missing:
        raise ValueError(
            "Sample manifest is missing required fields: " + ", ".join(missing)
        )
    normalized = normalized.loc[:, SAMPLE_MANIFEST_COLUMNS]
    csv_path = output / "sample_manifest.csv"
    json_path = output / "sample_manifest.json"
    normalized.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            normalized.to_dict(orient="records"),
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    return csv_path, json_path


def print_sample_manifest(manifest: pd.DataFrame) -> None:
    columns = [
        "split",
        "sample_position",
        "dataset_index",
        "size_bin",
        "nodes",
        "edges",
        "max_degree",
        "wedge_proxy",
    ]
    print(manifest.loc[:, columns].to_string(index=False), flush=True)


def graph_size_frame(
    raw_graphs: Sequence[nx.Graph],
    token_graphs: Sequence[nx.Graph],
) -> pd.DataFrame:
    if len(raw_graphs) != len(token_graphs):
        raise ValueError("raw_graphs and token_graphs must have equal length.")

    rows: list[dict[str, Any]] = []
    for index, (raw, token) in enumerate(
        zip(raw_graphs, token_graphs, strict=True)
    ):
        raw_nodes = raw.number_of_nodes()
        raw_edges = raw.number_of_edges()
        token_nodes = token.number_of_nodes()
        token_edges = token.number_of_edges()
        rows.append(
            {
                "graph_index": index,
                "raw_nodes": raw_nodes,
                "raw_edges": raw_edges,
                "token_nodes": token_nodes,
                "token_edges": token_edges,
                "node_reduction": raw_nodes - token_nodes,
                "edge_reduction": raw_edges - token_edges,
                "node_reduction_fraction": (
                    (raw_nodes - token_nodes) / raw_nodes if raw_nodes else 0.0
                ),
                "edge_reduction_fraction": (
                    (raw_edges - token_edges) / raw_edges if raw_edges else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _subprocess_environment(threads: int) -> dict[str, str]:
    environment = dict(os.environ)
    source_root = str(_source_root())
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not existing else source_root + os.pathsep + existing
    )
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = str(threads)
    environment.setdefault("PYTHONHASHSEED", "0")
    return environment


def _write_pickle(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _stream_worker_output(
    pipe: Any,
    *,
    target: Any,
    buffer: list[str],
) -> None:
    try:
        for line in iter(pipe.readline, ""):
            buffer.append(line)
            print(line, end="", file=target, flush=True)
    finally:
        pipe.close()


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - Windows fallback
            process.terminate()
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=2.0)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - Windows fallback
            process.kill()
    except ProcessLookupError:
        return
    process.wait(timeout=2.0)


def _run_worker(
    *,
    action: WorkerAction,
    payload_path: Path,
    result_path: Path,
    graph_output_path: Path | None,
    threads: int,
    phase_timeout_seconds: float | None = None,
    phase_name: str | None = None,
) -> dict[str, Any]:
    """Run one isolated worker with live output and process-group timeout."""

    phase = phase_name or f"{action} worker"
    command = [
        sys.executable,
        "-m",
        "buhito.benchmarks._worker",
        action,
        "--payload",
        str(payload_path),
        "--result",
        str(result_path),
    ]
    if graph_output_path is not None:
        command.extend(["--graph-output", str(graph_output_path)])

    result_path.unlink(missing_ok=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=_subprocess_environment(threads),
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    stdout_thread = threading.Thread(
        target=_stream_worker_output,
        kwargs={
            "pipe": process.stdout,
            "target": sys.stdout,
            "buffer": stdout_lines,
        },
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_worker_output,
        kwargs={
            "pipe": process.stderr,
            "target": sys.stderr,
            "buffer": stderr_lines,
        },
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    try:
        return_code = process.wait(timeout=phase_timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        stdout_thread.join(timeout=2.0)
        stderr_thread.join(timeout=2.0)
        timeout_text = (
            "without a configured limit"
            if phase_timeout_seconds is None
            else f"after {phase_timeout_seconds:g} seconds"
        )
        raise WorkerTimeoutError(
            f"Phase '{phase}' exceeded its timeout {timeout_text}. "
            "The worker process group was terminated."
        ) from exc

    stdout_thread.join(timeout=2.0)
    stderr_thread.join(timeout=2.0)
    if return_code != 0:
        raise RuntimeError(
            f"Phase '{phase}' worker failed with exit code {return_code}.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{''.join(stdout_lines)}\n"
            f"stderr:\n{''.join(stderr_lines)}"
        )
    if not result_path.is_file():
        raise RuntimeError(
            f"Phase '{phase}' completed without writing {result_path}."
        )
    return json.loads(result_path.read_text())


def _phase_start(name: str, *, enabled: bool = True) -> float:
    if enabled:
        print(f"[buhito-runtime] START {name}", flush=True)
    return time.perf_counter()


def _phase_end(name: str, started: float, *, enabled: bool = True) -> None:
    if enabled:
        elapsed = time.perf_counter() - started
        print(
            f"[buhito-runtime] END {name} elapsed_seconds={elapsed:.6f}",
            flush=True,
        )


def _quartile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_benchmark_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for representation, group in runs.groupby("representation", sort=False):
        wall = group["enumeration_seconds"].astype(float).tolist()
        rss = group["peak_rss_mb"].astype(float).tolist()
        python_peak = group["python_peak_mb"].astype(float).tolist()
        rows.append(
            {
                "representation": representation,
                "repeats": len(group),
                "median_enumeration_seconds": statistics.median(wall),
                "q1_enumeration_seconds": _quartile(wall, 0.25),
                "q3_enumeration_seconds": _quartile(wall, 0.75),
                "iqr_enumeration_seconds": (
                    _quartile(wall, 0.75) - _quartile(wall, 0.25)
                ),
                "median_peak_rss_mb": statistics.median(rss),
                "median_python_peak_mb": statistics.median(python_peak),
                "total_occurrences": int(group["total_occurrences"].iloc[0]),
                "unique_graphlet_keys": int(
                    group["unique_graphlet_keys"].iloc[0]
                ),
                "total_nodes": int(group["total_nodes"].iloc[0]),
                "total_edges": int(group["total_edges"].iloc[0]),
                "max_nodes": int(group["max_nodes"].iloc[0]),
                "max_edges": int(group["max_edges"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def hardware_metadata() -> dict[str, Any]:
    total_memory: int | None = None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        total_memory = page_size * pages
    except (AttributeError, OSError, ValueError):
        pass

    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "total_memory_bytes": total_memory,
        "networkx_version": nx.__version__,
    }


def benchmark_scalar_summary(result: RuntimeBenchmarkResult) -> dict[str, Any]:
    indexed = result.summary.set_index("representation")
    raw_seconds = float(indexed.loc["original", "median_enumeration_seconds"])
    token_seconds = float(
        indexed.loc["tokenized", "median_enumeration_seconds"]
    )
    raw_memory = float(indexed.loc["original", "median_peak_rss_mb"])
    token_memory = float(indexed.loc["tokenized", "median_peak_rss_mb"])

    speedup = raw_seconds / token_seconds if token_seconds > 0 else math.inf
    memory_ratio = token_memory / raw_memory if raw_memory > 0 else math.nan
    time_saved = raw_seconds - token_seconds
    compression_seconds = float(result.compression["compression_total_seconds"])
    break_even = (
        compression_seconds / time_saved if time_saved > 0 else math.inf
    )

    return {
        "raw_median_enumeration_seconds": raw_seconds,
        "tokenized_median_enumeration_seconds": token_seconds,
        "enumeration_speedup": speedup,
        "raw_median_peak_rss_mb": raw_memory,
        "tokenized_median_peak_rss_mb": token_memory,
        "peak_rss_ratio": memory_ratio,
        "compression_total_seconds": compression_seconds,
        "compression_peak_rss_mb": float(result.compression["peak_rss_mb"]),
        "compression_python_peak_mb": float(result.compression["python_peak_mb"]),
        "first_pass_total_seconds": compression_seconds + token_seconds,
        "break_even_reuses": break_even,
        "total_raw_nodes": int(result.graph_sizes["raw_nodes"].sum()),
        "total_token_nodes": int(result.graph_sizes["token_nodes"].sum()),
        "total_raw_edges": int(result.graph_sizes["raw_edges"].sum()),
        "total_token_edges": int(result.graph_sizes["token_edges"].sum()),
        "node_reduction_fraction": float(
            1.0
            - result.graph_sizes["token_nodes"].sum()
            / max(result.graph_sizes["raw_nodes"].sum(), 1)
        ),
        "edge_reduction_fraction": float(
            1.0
            - result.graph_sizes["token_edges"].sum()
            / max(result.graph_sizes["raw_edges"].sum(), 1)
        ),
        "selected_rule_count": int(result.compression["selected_rule_count"]),
        "tokenized_graph_count": int(
            result.compression["tokenized_graph_count"]
        ),
        "decode_failures": int(result.compression["decode_failures"]),
        "mdl_net_savings_bits": float(
            result.compression["mdl_net_savings_bits"]
        ),
        "force_rewrite": bool(result.config.force_rewrite),
        "token_projection": result.config.token_projection,
    }


def render_runtime_summary(result: RuntimeBenchmarkResult) -> str:
    summary = benchmark_scalar_summary(result)
    mode = (
        "forced tokenization"
        if result.config.force_rewrite
        else "MDL-selected tokenization"
    )
    break_even = summary["break_even_reuses"]
    token_seconds = summary["tokenized_median_enumeration_seconds"]
    break_even_text = (
        "not reached because tokenized enumeration was not faster"
        if not math.isfinite(break_even)
        else f"{break_even:.3f} repeated enumeration runs"
    )
    forced_warning = (
        "Forced tokenization is a computational representation experiment. "
        "Negative MDL savings are not positive compression."
        if result.config.force_rewrite
        else "Only rewrites selected by the complete MDL objective were used."
    )
    return f"""# Buhito runtime benchmark

## Interpretation boundary

This run compares graphlet enumeration on the original graphs with enumeration
on a reversible motif-tokenized representation. Tokenized graphlet counts are
not asserted to equal the original graphlet counts. The tokenization mode was
**{mode}** and the token projection was `{result.config.token_projection}`.

{forced_warning}

## Main results

- Original median enumeration time: {summary['raw_median_enumeration_seconds']:.6f} s
- Tokenized median enumeration time: {token_seconds:.6f} s
- Enumeration speedup: {summary['enumeration_speedup']:.6f}x
- Original median peak RSS: {summary['raw_median_peak_rss_mb']:.3f} MB
- Tokenized median peak RSS: {summary['tokenized_median_peak_rss_mb']:.3f} MB
- Peak RSS ratio: {summary['peak_rss_ratio']:.6f}
- Node reduction fraction: {summary['node_reduction_fraction']:.6f}
- Edge reduction fraction: {summary['edge_reduction_fraction']:.6f}
- Compression overhead: {summary['compression_total_seconds']:.6f} s
- Compression peak RSS: {summary['compression_peak_rss_mb']:.3f} MB
- Estimated break-even: {break_even_text}
- Selected dictionary rules: {summary['selected_rule_count']}
- Graphs represented with rewrites: {summary['tokenized_graph_count']}
- Exact decode failures: {summary['decode_failures']}
- MDL net savings: {summary['mdl_net_savings_bits']:.6f} bits

## Files

- `sample_manifest.csv` / `sample_manifest.json`: exact sampled graphs and safety metrics.
- `benchmark_runs.csv`: one isolated process measurement per repeat.
- `benchmark_summary.csv`: median and interquartile summaries.
- `graph_size_comparison.csv`: per-graph raw/tokenized sizes.
- `compression_summary.json`: dictionary and tokenization accounting.
- `summary.json`: headline systems metrics.
- `metadata.json`: hardware, software, configuration, and safety caps.
"""


def save_runtime_plots(result: RuntimeBenchmarkResult, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Runtime plots require matplotlib. Install buhito[viz]."
        ) from exc

    summary = result.summary.set_index("representation")
    representations = ["original", "tokenized"]

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(
        representations,
        [summary.loc[name, "median_enumeration_seconds"] for name in representations],
    )
    axis.set_ylabel("Median enumeration time (seconds)")
    axis.set_title("Buhito enumeration runtime")
    figure.tight_layout()
    figure.savefig(output_dir / "runtime_comparison.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(
        representations,
        [summary.loc[name, "median_peak_rss_mb"] for name in representations],
    )
    axis.set_ylabel("Median process peak RSS (MB)")
    axis.set_title("Buhito enumeration memory")
    figure.tight_layout()
    figure.savefig(output_dir / "memory_comparison.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(
        result.graph_sizes["raw_nodes"],
        result.graph_sizes["token_nodes"],
    )
    maximum = max(
        result.graph_sizes["raw_nodes"].max(),
        result.graph_sizes["token_nodes"].max(),
        1,
    )
    axis.plot([0, maximum], [0, maximum], linestyle="--")
    axis.set_xlabel("Original nodes")
    axis.set_ylabel("Tokenized nodes")
    axis.set_title("Per-graph node reduction")
    figure.tight_layout()
    figure.savefig(output_dir / "node_reduction.png", dpi=180)
    plt.close(figure)


def run_runtime_benchmark(
    fit_graphs: Iterable[nx.Graph],
    eval_graphs: Iterable[nx.Graph],
    *,
    compressor_kwargs: Mapping[str, Any] | None = None,
    config: RuntimeBenchmarkConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
    sample_manifest: pd.DataFrame | None = None,
) -> RuntimeBenchmarkResult:
    """Run isolated compression and enumeration measurements."""

    active_config = config or RuntimeBenchmarkConfig()
    active_config.validate()
    fit_list = list(fit_graphs)
    eval_list = list(eval_graphs)
    if not fit_list:
        raise ValueError("fit_graphs cannot be empty.")
    if not eval_list:
        raise ValueError("eval_graphs cannot be empty.")

    kwargs = dict(compressor_kwargs or {})
    kwargs.pop("enumerator", None)

    with tempfile.TemporaryDirectory(prefix="buhito-runtime-") as temporary:
        temp = Path(temporary)
        compression_payload = temp / "compression_payload.pkl"
        compression_result = temp / "compression_result.json"
        graph_output = temp / "graphs.pkl"
        _write_pickle(
            compression_payload,
            {
                "fit_graphs": fit_list,
                "eval_graphs": eval_list,
                "compressor_kwargs": kwargs,
                "compressor_backend": active_config.compressor_backend,
                "force_rewrite": active_config.force_rewrite,
                "token_projection": active_config.token_projection,
            },
        )
        phase = "compression worker"
        started = _phase_start(phase, enabled=active_config.progress)
        compression = _run_worker(
            action="compress",
            payload_path=compression_payload,
            result_path=compression_result,
            graph_output_path=graph_output,
            threads=active_config.threads,
            phase_timeout_seconds=active_config.phase_timeout_seconds,
            phase_name=phase,
        )
        _phase_end(phase, started, enabled=active_config.progress)
        graph_payload = _read_pickle(graph_output)
        raw_graphs = graph_payload["raw_graphs"]
        token_graphs = graph_payload["token_graphs"]
        graph_sizes = pd.DataFrame(graph_payload["graph_sizes"])

        graph_payload_paths: dict[str, Path] = {}
        for representation, graphs in (
            ("original", raw_graphs),
            ("tokenized", token_graphs),
        ):
            enumeration_payload = temp / f"{representation}_payload.pkl"
            _write_pickle(
                enumeration_payload,
                {
                    "graphs": graphs,
                    "graphlet_sizes": active_config.graphlet_sizes,
                    "backend": active_config.backend,
                },
            )
            graph_payload_paths[representation] = enumeration_payload

        for representation in ("original", "tokenized"):
            display = "raw" if representation == "original" else "tokenized"
            for warmup in range(active_config.warmup_repeats):
                phase = f"{display} enumeration warmup {warmup + 1}"
                started = _phase_start(phase, enabled=active_config.progress)
                _run_worker(
                    action="enumerate",
                    payload_path=graph_payload_paths[representation],
                    result_path=temp / f"{representation}_warmup_{warmup}.json",
                    graph_output_path=None,
                    threads=active_config.threads,
                    phase_timeout_seconds=active_config.phase_timeout_seconds,
                    phase_name=phase,
                )
                _phase_end(phase, started, enabled=active_config.progress)

        rows: list[dict[str, Any]] = []
        for repeat in range(active_config.repeats):
            order = (
                ("original", "tokenized")
                if repeat % 2 == 0
                else ("tokenized", "original")
            )
            for order_position, representation in enumerate(order):
                display = "raw" if representation == "original" else "tokenized"
                phase = f"{display} enumeration repeat {repeat + 1}"
                started = _phase_start(phase, enabled=active_config.progress)
                row = _run_worker(
                    action="enumerate",
                    payload_path=graph_payload_paths[representation],
                    result_path=temp / f"{representation}_{repeat}.json",
                    graph_output_path=None,
                    threads=active_config.threads,
                    phase_timeout_seconds=active_config.phase_timeout_seconds,
                    phase_name=phase,
                )
                _phase_end(phase, started, enabled=active_config.progress)
                row["representation"] = representation
                row["repeat"] = repeat
                row["order_position"] = order_position
                rows.append(row)

    runs = pd.DataFrame(rows)
    summary = summarize_benchmark_runs(runs)
    metadata_dict = dict(metadata or {})
    manifest = (
        default_sample_manifest(
            fit_list,
            eval_list,
            dataset=str(metadata_dict.get("dataset", "unspecified")),
            seed=int(metadata_dict.get("sample_seed", 0)),
        )
        if sample_manifest is None
        else sample_manifest.copy()
    )
    full_metadata = {
        **hardware_metadata(),
        "benchmark_config": asdict(active_config),
        "compressor_kwargs": kwargs,
        "fit_graph_count": len(fit_list),
        "eval_graph_count": len(eval_list),
        **metadata_dict,
    }
    return RuntimeBenchmarkResult(
        config=active_config,
        compression=compression,
        runs=runs,
        summary=summary,
        graph_sizes=graph_sizes,
        metadata=full_metadata,
        sample_manifest=manifest,
    )

# ---------------------------------------------------------------------------
# Prepared, parallel, and HPC task execution
# ---------------------------------------------------------------------------

TASK_MANIFEST_COLUMNS = (
    "task_id",
    "representation",
    "repeat",
    "order_position",
    "prepared_fingerprint",
)


class PreparedStateError(RuntimeError):
    """Raised when prepared benchmark files are incomplete or incompatible."""


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        path.name + f".tmp-{os.getpid()}-{threading.get_ident()}"
    )
    temporary.write_text(text)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
    )


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        path.name + f".tmp-{os.getpid()}-{threading.get_ident()}"
    )
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_task_manifest(
    *, repeats: int, prepared_fingerprint: str
) -> pd.DataFrame:
    if repeats < 1:
        raise ValueError("repeats must be at least one.")
    rows: list[dict[str, Any]] = []
    task_id = 0
    for repeat in range(repeats):
        order = (
            ("original", "tokenized")
            if repeat % 2 == 0
            else ("tokenized", "original")
        )
        for order_position, representation in enumerate(order):
            rows.append(
                {
                    "task_id": task_id,
                    "representation": representation,
                    "repeat": repeat,
                    "order_position": order_position,
                    "prepared_fingerprint": prepared_fingerprint,
                }
            )
            task_id += 1
    return pd.DataFrame(rows, columns=TASK_MANIFEST_COLUMNS)


def _prepared_paths(prepared_dir: str | Path) -> dict[str, Path]:
    root = Path(prepared_dir)
    return {
        "root": root,
        "state": root / "prepared_state.json",
        "config": root / "benchmark_config.json",
        "compression": root / "compression_summary.json",
        "graph_sizes": root / "graph_size_comparison.csv",
        "metadata": root / "metadata.json",
        "sample_csv": root / "sample_manifest.csv",
        "sample_json": root / "sample_manifest.json",
        "original": root / "original_payload.pkl",
        "tokenized": root / "tokenized_payload.pkl",
        "tasks_csv": root / "task_manifest.csv",
        "tasks_json": root / "task_manifest.json",
        "results": root / "task_results",
    }


def prepare_runtime_benchmark(
    fit_graphs: Iterable[nx.Graph],
    eval_graphs: Iterable[nx.Graph],
    *,
    prepared_dir: str | Path,
    compressor_kwargs: Mapping[str, Any] | None = None,
    config: RuntimeBenchmarkConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
    sample_manifest: pd.DataFrame | None = None,
) -> Path:
    """Fit/tokenize once and write immutable inputs for independent tasks."""
    active_config = config or RuntimeBenchmarkConfig()
    active_config.validate()
    fit_list = list(fit_graphs)
    eval_list = list(eval_graphs)
    if not fit_list or not eval_list:
        raise ValueError("fit_graphs and eval_graphs must both be nonempty.")

    paths = _prepared_paths(prepared_dir)
    root = paths["root"]
    root.mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)
    kwargs = dict(compressor_kwargs or {})
    kwargs.pop("enumerator", None)

    with tempfile.TemporaryDirectory(prefix="buhito-prepare-") as temporary:
        temp = Path(temporary)
        payload = temp / "compression_payload.pkl"
        result_path = temp / "compression_result.json"
        graph_output = temp / "graphs.pkl"
        _write_pickle(
            payload,
            {
                "fit_graphs": fit_list,
                "eval_graphs": eval_list,
                "compressor_kwargs": kwargs,
                "compressor_backend": active_config.compressor_backend,
                "force_rewrite": active_config.force_rewrite,
                "token_projection": active_config.token_projection,
            },
        )
        compression = _run_worker(
            action="compress",
            payload_path=payload,
            result_path=result_path,
            graph_output_path=graph_output,
            threads=active_config.threads,
            phase_timeout_seconds=active_config.phase_timeout_seconds,
            phase_name="compression worker",
        )
        graph_payload = _read_pickle(graph_output)

    _atomic_pickle(
        paths["original"],
        {
            "graphs": graph_payload["raw_graphs"],
            "graphlet_sizes": active_config.graphlet_sizes,
            "backend": active_config.backend,
        },
    )
    _atomic_pickle(
        paths["tokenized"],
        {
            "graphs": graph_payload["token_graphs"],
            "graphlet_sizes": active_config.graphlet_sizes,
            "backend": active_config.backend,
        },
    )
    _atomic_json(paths["config"], asdict(active_config))
    _atomic_json(paths["compression"], compression)
    pd.DataFrame(graph_payload["graph_sizes"]).to_csv(
        paths["graph_sizes"], index=False
    )

    metadata_dict = {
        **hardware_metadata(),
        "benchmark_config": asdict(active_config),
        "compressor_kwargs": kwargs,
        "fit_graph_count": len(fit_list),
        "eval_graph_count": len(eval_list),
        **dict(metadata or {}),
    }
    _atomic_json(paths["metadata"], metadata_dict)
    manifest = (
        default_sample_manifest(
            fit_list,
            eval_list,
            dataset=str(metadata_dict.get("dataset", "unspecified")),
            seed=int(metadata_dict.get("sample_seed", 0)),
        )
        if sample_manifest is None
        else sample_manifest.copy()
    )
    save_sample_manifest(manifest, root)

    fingerprint_payload = {
        "config": asdict(active_config),
        "compression": compression,
        "metadata": metadata_dict,
        "original_sha256": _sha256_file(paths["original"]),
        "tokenized_sha256": _sha256_file(paths["tokenized"]),
        "sample_sha256": _sha256_file(paths["sample_csv"]),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    state = {
        "format_version": 1,
        "prepared_fingerprint": fingerprint,
        "original_payload_sha256": fingerprint_payload["original_sha256"],
        "tokenized_payload_sha256": fingerprint_payload["tokenized_sha256"],
        "task_count": 2 * active_config.repeats,
    }
    _atomic_json(paths["state"], state)
    tasks = create_task_manifest(
        repeats=active_config.repeats,
        prepared_fingerprint=fingerprint,
    )
    tasks.to_csv(paths["tasks_csv"], index=False)
    _atomic_json(paths["tasks_json"], tasks.to_dict(orient="records"))
    return root


def load_task_manifest(
    prepared_dir: str | Path,
    task_manifest: str | Path | None = None,
) -> pd.DataFrame:
    path = (
        Path(task_manifest)
        if task_manifest is not None
        else _prepared_paths(prepared_dir)["tasks_csv"]
    )
    if not path.is_file():
        raise PreparedStateError(f"Task manifest is missing: {path}")
    tasks = pd.read_csv(path)
    missing = [name for name in TASK_MANIFEST_COLUMNS if name not in tasks]
    if missing:
        raise PreparedStateError(
            "Task manifest is missing columns: " + ", ".join(missing)
        )
    if tasks["task_id"].duplicated().any():
        duplicates = tasks.loc[
            tasks["task_id"].duplicated(), "task_id"
        ].tolist()
        raise PreparedStateError(f"Duplicate task IDs in manifest: {duplicates}")
    return tasks.loc[:, TASK_MANIFEST_COLUMNS].copy()


def resolve_task_id(task_id: int | None = None) -> int:
    if task_id is not None:
        return int(task_id)
    environment_value = os.environ.get("SLURM_ARRAY_TASK_ID")
    if environment_value is None:
        raise ValueError(
            "No task ID was supplied and SLURM_ARRAY_TASK_ID is not set."
        )
    return int(environment_value)


def _validate_prepared_state(paths: Mapping[str, Path]) -> dict[str, Any]:
    required = (
        "state",
        "config",
        "compression",
        "graph_sizes",
        "metadata",
        "sample_csv",
        "original",
        "tokenized",
        "tasks_csv",
    )
    missing = [
        str(paths[name]) for name in required if not paths[name].is_file()
    ]
    if missing:
        raise PreparedStateError(
            "Prepared files are missing: " + ", ".join(missing)
        )
    state = json.loads(paths["state"].read_text())
    if _sha256_file(paths["original"]) != state["original_payload_sha256"]:
        raise PreparedStateError(
            "Original payload fingerprint does not match prepared state."
        )
    if _sha256_file(paths["tokenized"]) != state["tokenized_payload_sha256"]:
        raise PreparedStateError(
            "Tokenized payload fingerprint does not match prepared state."
        )
    return state


def run_prepared_task(
    prepared_dir: str | Path,
    *,
    task_id: int | None = None,
    task_manifest: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> Path:
    """Run one stable benchmark task and atomically write its result."""
    paths = _prepared_paths(prepared_dir)
    state = _validate_prepared_state(paths)
    tasks = load_task_manifest(prepared_dir, task_manifest)
    resolved = resolve_task_id(task_id)
    matches = tasks.loc[tasks["task_id"].astype(int) == resolved]
    if len(matches) != 1:
        raise PreparedStateError(f"Task ID {resolved} is not present exactly once.")
    task = matches.iloc[0].to_dict()
    if task["prepared_fingerprint"] != state["prepared_fingerprint"]:
        raise PreparedStateError(
            "Task fingerprint is incompatible with prepared state."
        )

    config = RuntimeBenchmarkConfig(**json.loads(paths["config"].read_text()))
    payload_path = paths[str(task["representation"])]
    destination_root = Path(results_dir) if results_dir else paths["results"]
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"task_{resolved:05d}.json"

    with tempfile.TemporaryDirectory(
        prefix=f"buhito-task-{resolved}-"
    ) as temporary:
        temp = Path(temporary)
        for warmup in range(config.warmup_repeats):
            _run_worker(
                action="enumerate",
                payload_path=payload_path,
                result_path=temp / f"warmup_{warmup}.json",
                graph_output_path=None,
                threads=config.threads,
                phase_timeout_seconds=config.phase_timeout_seconds,
                phase_name=f"task {resolved} warmup {warmup + 1}",
            )
        measured = _run_worker(
            action="enumerate",
            payload_path=payload_path,
            result_path=temp / "measured.json",
            graph_output_path=None,
            threads=config.threads,
            phase_timeout_seconds=config.phase_timeout_seconds,
            phase_name=(
                f"task {resolved} {task['representation']} "
                f"repeat {int(task['repeat']) + 1}"
            ),
        )

    measured.update(
        {
            "task_id": resolved,
            "representation": task["representation"],
            "repeat": int(task["repeat"]),
            "order_position": int(task["order_position"]),
            "prepared_fingerprint": state["prepared_fingerprint"],
        }
    )
    _atomic_json(destination, measured)
    return destination


def run_prepared_tasks(
    prepared_dir: str | Path,
    *,
    jobs: int = 1,
    task_manifest: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> list[Path]:
    """Run all prepared tasks with explicitly bounded local concurrency."""
    if jobs < 1:
        raise ValueError("jobs must be at least one.")
    tasks = load_task_manifest(prepared_dir, task_manifest)
    task_ids = [int(value) for value in tasks["task_id"]]
    if jobs == 1:
        return [
            run_prepared_task(
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
                run_prepared_task,
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


def aggregate_prepared_results(
    prepared_dir: str | Path,
    *,
    task_manifest: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> RuntimeBenchmarkResult:
    """Verify and aggregate all expected prepared-task results."""
    paths = _prepared_paths(prepared_dir)
    state = _validate_prepared_state(paths)
    tasks = load_task_manifest(prepared_dir, task_manifest)
    result_root = Path(results_dir) if results_dir else paths["results"]
    expected_ids = set(tasks["task_id"].astype(int))
    result_files = sorted(result_root.glob("task_*.json"))
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for result_file in result_files:
        row = json.loads(result_file.read_text())
        task_id = int(row.get("task_id", -1))
        if task_id in seen:
            raise PreparedStateError(
                f"Duplicate task result for task ID {task_id}."
            )
        seen.add(task_id)
        if row.get("prepared_fingerprint") != state["prepared_fingerprint"]:
            raise PreparedStateError(
                f"Task {task_id} has an incompatible prepared-state fingerprint."
            )
        rows.append(row)
    missing = sorted(expected_ids - seen)
    unexpected = sorted(seen - expected_ids)
    if missing:
        raise PreparedStateError(f"Missing task results: {missing}")
    if unexpected:
        raise PreparedStateError(f"Unexpected task results: {unexpected}")

    runs = pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)
    config = RuntimeBenchmarkConfig(**json.loads(paths["config"].read_text()))
    return RuntimeBenchmarkResult(
        config=config,
        compression=json.loads(paths["compression"].read_text()),
        runs=runs,
        summary=summarize_benchmark_runs(runs),
        graph_sizes=pd.read_csv(paths["graph_sizes"]),
        metadata=json.loads(paths["metadata"].read_text()),
        sample_manifest=pd.read_csv(paths["sample_csv"]),
    )


def write_slurm_array_script(
    prepared_dir: str | Path,
    script_path: str | Path,
    *,
    cli_path: str = "examples/benchmark_reddit_tokenization.py",
) -> Path:
    """Write a portable SLURM array template with site placeholders."""
    tasks = load_task_manifest(prepared_dir)
    maximum = int(tasks["task_id"].max())
    prepared = Path(prepared_dir).resolve()
    script = Path(script_path)
    text = (
        "#!/usr/bin/env bash\n"
        "#SBATCH --job-name=buhito-runtime\n"
        f"#SBATCH --array=0-{maximum}\n"
        "#SBATCH --cpus-per-task=1\n"
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
    _atomic_text(script, text)
    script.chmod(0o755)
    return script

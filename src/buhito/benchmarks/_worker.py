"""Private subprocess worker for isolated Buhito runtime measurements."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time
import tracemalloc
from typing import Any

import networkx as nx

from buhito.benchmarks.runtime import (
    collapse_parallel_edges,
    relabel_for_enumeration,
)
from buhito.mdl import (
    BuhitoGraphletEnumerator,
    ExhaustiveGraphletEnumerator,
    MDLGraphCompressor,
    _validate_rewrite_exact,
)


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_pickle(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0**2) if sys.platform == "darwin" else value / 1024.0


def _enumerator(name: str):
    if name == "buhito":
        return BuhitoGraphletEnumerator()
    if name == "exhaustive":
        return ExhaustiveGraphletEnumerator()
    raise ValueError(f"Unknown graphlet backend: {name!r}.")


def _finish_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics["python_peak_mb"] = python_peak / (1024.0**2)
    metrics["peak_rss_mb"] = _peak_rss_mb()
    return metrics


def _phase_start(name: str) -> float:
    print(f"[buhito-worker] START {name}", flush=True)
    return time.perf_counter()


def _phase_end(name: str, started: float) -> float:
    elapsed = time.perf_counter() - started
    print(
        f"[buhito-worker] END {name} elapsed_seconds={elapsed:.6f}",
        flush=True,
    )
    return elapsed


def _decode_failure_count(result: Any, *, force_rewrite: bool) -> int:
    failures = 0
    for record in result.records:
        raw = record.baseline_graph.copy()
        use_token = record.rewrite is not None and (
            force_rewrite or record.use_rewrite
        )
        if use_token and not _validate_rewrite_exact(raw, record.rewrite):
            failures += 1
    return failures


def run_enumeration(payload: dict[str, Any]) -> dict[str, Any]:
    graphs = payload["graphs"]
    sizes = tuple(int(size) for size in payload["graphlet_sizes"])
    enumerator = _enumerator(payload["backend"])

    total_occurrences = 0
    unique_keys: set[Any] = set()
    maximum_occurrences = 0

    start = time.perf_counter()
    for graph in graphs:
        prepared = relabel_for_enumeration(graph)
        found = enumerator.enumerate(prepared, sizes)
        graph_occurrences = sum(len(values) for values in found.values())
        total_occurrences += graph_occurrences
        maximum_occurrences = max(maximum_occurrences, graph_occurrences)
        unique_keys.update(found)
    elapsed = time.perf_counter() - start

    return {
        "backend": payload["backend"],
        "enumeration_seconds": elapsed,
        "n_graphs": len(graphs),
        "total_nodes": sum(graph.number_of_nodes() for graph in graphs),
        "total_edges": sum(graph.number_of_edges() for graph in graphs),
        "max_nodes": max((graph.number_of_nodes() for graph in graphs), default=0),
        "max_edges": max((graph.number_of_edges() for graph in graphs), default=0),
        "total_occurrences": total_occurrences,
        "unique_graphlet_keys": len(unique_keys),
        "max_occurrences_per_graph": maximum_occurrences,
    }


def run_compression(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit_graphs = payload["fit_graphs"]
    eval_graphs = payload["eval_graphs"]
    kwargs = dict(payload["compressor_kwargs"])
    kwargs["enumerator"] = _enumerator(payload["compressor_backend"])
    force_rewrite = bool(payload["force_rewrite"])
    projection = payload["token_projection"]

    compressor = MDLGraphCompressor(**kwargs)

    phase = "dictionary fitting"
    started = _phase_start(phase)
    compressor.fit(fit_graphs)
    fit_seconds = _phase_end(phase, started)

    phase = "fit transformation"
    started = _phase_start(phase)
    fit_result = compressor.transform(fit_graphs)
    fit_transform_seconds = _phase_end(phase, started)

    phase = "evaluation transformation"
    started = _phase_start(phase)
    result = compressor.transform(eval_graphs)
    eval_transform_seconds = _phase_end(phase, started)

    fit_decode_failures = _decode_failure_count(
        fit_result,
        force_rewrite=force_rewrite,
    )

    raw_graphs: list[nx.Graph] = []
    token_graphs: list[nx.Graph] = []
    graph_sizes: list[dict[str, Any]] = []
    eval_decode_failures = 0
    tokenized_count = 0

    for index, record in enumerate(result.records):
        raw = record.baseline_graph.copy()
        use_token = record.rewrite is not None and (
            force_rewrite or record.use_rewrite
        )
        token = record.normalized_model_graph(force_rewrite=force_rewrite)
        if use_token:
            tokenized_count += 1
            if not _validate_rewrite_exact(raw, record.rewrite):
                eval_decode_failures += 1

        if projection == "simple":
            token = collapse_parallel_edges(token)
        elif projection != "native":
            raise ValueError(f"Unknown token projection: {projection!r}.")

        raw_graphs.append(raw)
        token_graphs.append(token)
        raw_nodes = raw.number_of_nodes()
        raw_edges = raw.number_of_edges()
        token_nodes = token.number_of_nodes()
        token_edges = token.number_of_edges()
        graph_sizes.append(
            {
                "graph_index": index,
                "raw_nodes": raw_nodes,
                "raw_edges": raw_edges,
                "token_nodes": token_nodes,
                "token_edges": token_edges,
                "node_reduction": raw_nodes - token_nodes,
                "edge_reduction": raw_edges - token_edges,
                "node_reduction_fraction": (
                    (raw_nodes - token_nodes) / raw_nodes
                    if raw_nodes
                    else 0.0
                ),
                "edge_reduction_fraction": (
                    (raw_edges - token_edges) / raw_edges
                    if raw_edges
                    else 0.0
                ),
                "rewrite_available": record.rewrite is not None,
                "mdl_selected_rewrite": record.use_rewrite,
                "tokenized_for_benchmark": use_token,
                "selected_occurrences": record.selected_occurrences,
            }
        )

    transform_seconds = fit_transform_seconds + eval_transform_seconds
    decode_failures = fit_decode_failures + eval_decode_failures
    compression = {
        "compressor_backend": payload["compressor_backend"],
        "token_projection": projection,
        "force_rewrite": force_rewrite,
        "fit_seconds": fit_seconds,
        "fit_transform_seconds": fit_transform_seconds,
        "eval_transform_seconds": eval_transform_seconds,
        "transform_seconds": transform_seconds,
        "compression_total_seconds": fit_seconds + transform_seconds,
        "selected_rule_count": len(compressor.rules_ or ()),
        "tokenized_graph_count": tokenized_count,
        "eval_graph_count": len(eval_graphs),
        "fit_decode_failures": fit_decode_failures,
        "eval_decode_failures": eval_decode_failures,
        "decode_failures": decode_failures,
        "mdl_n_rewritten": result.report.n_rewritten,
        "mdl_n_occurrences": result.report.n_occurrences,
        "mdl_baseline_bits": result.report.baseline_bits,
        "mdl_encoded_bits": result.report.encoded_bits,
        "mdl_net_savings_bits": result.report.net_savings_bits,
        "mdl_fraction_rewritten": result.report.fraction_rewritten,
    }
    graph_output = {
        "raw_graphs": raw_graphs,
        "token_graphs": token_graphs,
        "graph_sizes": graph_sizes,
    }
    return compression, graph_output


def run_sleep(payload: dict[str, Any]) -> dict[str, Any]:
    """Test-only worker action used to verify timeout process cleanup."""

    pid_path_value = payload.get("pid_path")
    if pid_path_value:
        Path(pid_path_value).write_text(f"{os.getpid()}\n")
    seconds = float(payload.get("sleep_seconds", 60.0))
    print(
        f"[buhito-worker] START intentional sleep seconds={seconds:g}",
        flush=True,
    )
    time.sleep(seconds)
    return {"slept_seconds": seconds}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("compress", "enumerate", "sleep"))
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--graph-output", type=Path)
    args = parser.parse_args()

    tracemalloc.start()
    load_start = time.perf_counter()
    payload = _read_pickle(args.payload)
    load_seconds = time.perf_counter() - load_start

    if args.action == "enumerate":
        metrics = run_enumeration(payload)
    elif args.action == "sleep":
        metrics = run_sleep(payload)
    else:
        if args.graph_output is None:
            parser.error("--graph-output is required for compression.")
        metrics, graphs = run_compression(payload)
        _write_pickle(args.graph_output, graphs)

    metrics["payload_load_seconds"] = load_seconds
    metrics = _finish_metrics(metrics)
    args.result.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n"
    )


if __name__ == "__main__":
    main()

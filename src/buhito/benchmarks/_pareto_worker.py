"""Private worker that fits one MDL dictionary and materializes rule prefixes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import pickle
import resource
import sys
import time
import tracemalloc
from typing import Any, Sequence

import networkx as nx

from buhito.benchmarks._worker import _decode_failure_count, _enumerator
from buhito.benchmarks.runtime import collapse_parallel_edges
from buhito.mdl import MDLGraphCompressor, _validate_rewrite_exact


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_pickle(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0**2) if sys.platform == "darwin" else value / 1024.0


def _phase(name: str, function):
    print(f"[buhito-pareto-worker] START {name}", flush=True)
    started = time.perf_counter()
    value = function()
    elapsed = time.perf_counter() - started
    print(
        f"[buhito-pareto-worker] END {name} elapsed_seconds={elapsed:.6f}",
        flush=True,
    )
    return value, elapsed


def _project(graph: nx.Graph, projection: str) -> nx.Graph:
    if projection == "simple":
        return collapse_parallel_edges(graph)
    if projection == "native":
        return graph.copy()
    raise ValueError(f"Unknown token projection: {projection!r}.")


def _graph_point(
    result: Any,
    *,
    rule_count: int,
    projection: str,
) -> tuple[list[nx.Graph], list[dict[str, Any]], int]:
    token_graphs: list[nx.Graph] = []
    sizes: list[dict[str, Any]] = []
    decode_failures = 0
    for graph_index, record in enumerate(result.records):
        raw = record.baseline_graph.copy()
        use_token = rule_count > 0 and record.rewrite is not None
        token = (
            record.normalized_model_graph(force_rewrite=True)
            if use_token
            else raw.copy()
        )
        if use_token and not _validate_rewrite_exact(raw, record.rewrite):
            decode_failures += 1
        token = _project(token, projection)
        token_graphs.append(token)
        raw_nodes = raw.number_of_nodes()
        raw_edges = raw.number_of_edges()
        token_nodes = token.number_of_nodes()
        token_edges = token.number_of_edges()
        sizes.append(
            {
                "rule_count": int(rule_count),
                "graph_index": int(graph_index),
                "raw_nodes": int(raw_nodes),
                "raw_edges": int(raw_edges),
                "token_nodes": int(token_nodes),
                "token_edges": int(token_edges),
                "node_reduction": int(raw_nodes - token_nodes),
                "edge_reduction": int(raw_edges - token_edges),
                "node_reduction_fraction": (
                    (raw_nodes - token_nodes) / raw_nodes if raw_nodes else 0.0
                ),
                "edge_reduction_fraction": (
                    (raw_edges - token_edges) / raw_edges if raw_edges else 0.0
                ),
                "rewrite_available": bool(record.rewrite is not None),
                "tokenized_for_benchmark": bool(use_token),
                "selected_occurrences": int(record.selected_occurrences),
            }
        )
    return token_graphs, sizes, decode_failures


def _candidate_identity(compressor: MDLGraphCompressor, count: int) -> dict[str, Any]:
    if count == 0:
        return {"motif_ids": [], "rule_keys": []}
    table = compressor.candidate_frame().sort_values("rank").head(count)
    return {
        "motif_ids": table["motif_id"].astype(str).tolist(),
        "rule_keys": table["key"].astype(str).tolist(),
    }


def run(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    fit_graphs = list(payload["fit_graphs"])
    eval_graphs = list(payload["eval_graphs"])
    rule_counts = tuple(sorted({int(value) for value in payload["rule_counts"]}))
    if not rule_counts or rule_counts[0] != 0:
        raise ValueError("Pareto rule_counts must include zero.")
    if any(value < 0 for value in rule_counts):
        raise ValueError("Pareto rule counts must be nonnegative.")
    maximum = max(rule_counts)
    kwargs = dict(payload["compressor_kwargs"])
    kwargs["n_rules"] = maximum
    kwargs["dictionary_selection"] = "fixed"
    kwargs["min_rule_savings_bits"] = -math.inf
    kwargs["enumerator"] = _enumerator(payload["compressor_backend"])
    projection = str(payload["token_projection"])

    compressor = MDLGraphCompressor(**kwargs)
    _, fit_seconds = _phase(
        "shared dictionary fitting",
        lambda: compressor.fit(fit_graphs),
    )
    available = len(compressor.candidate_frame())
    if maximum > available:
        raise ValueError(
            f"Requested rule prefix {maximum}, but only {available} scored "
            "candidates are available."
        )

    points: list[dict[str, Any]] = []
    token_graphs_by_rule: dict[int, list[nx.Graph]] = {}
    graph_sizes_by_rule: dict[int, list[dict[str, Any]]] = {}
    raw_graphs: list[nx.Graph] | None = None

    for rule_count in rule_counts:
        if rule_count == 0:
            baseline_result, eval_seconds = _phase(
                "rule prefix 0 evaluation",
                lambda: compressor.transform_rule_prefix(eval_graphs, 0),
            )
            raw_graphs = [
                record.baseline_graph.copy()
                for record in baseline_result.records
            ]
            token_graphs, graph_sizes, eval_failures = _graph_point(
                baseline_result,
                rule_count=0,
                projection=projection,
            )
            fit_transform_seconds = 0.0
            forced_report = baseline_result.report
            best_report = baseline_result.report
            fit_failures = 0
            tokenized_count = 0
            standalone_seconds = 0.0
            best_accounting_seconds = 0.0
        else:
            fit_result, fit_transform_seconds = _phase(
                f"rule prefix {rule_count} fit transformation",
                lambda count=rule_count: compressor.transform_rule_prefix(
                    fit_graphs,
                    count,
                    require_nonempty_rewrite=True,
                ),
            )
            forced_result, eval_seconds = _phase(
                f"rule prefix {rule_count} evaluation transformation",
                lambda count=rule_count: compressor.transform_rule_prefix(
                    eval_graphs,
                    count,
                    require_nonempty_rewrite=True,
                ),
            )
            best_result, best_accounting_seconds = _phase(
                f"rule prefix {rule_count} best-model accounting",
                lambda count=rule_count: compressor.transform_rule_prefix(
                    eval_graphs,
                    count,
                    require_nonempty_rewrite=False,
                ),
            )
            if raw_graphs is None:
                raw_graphs = [
                    record.baseline_graph.copy() for record in forced_result.records
                ]
            token_graphs, graph_sizes, eval_failures = _graph_point(
                forced_result,
                rule_count=rule_count,
                projection=projection,
            )
            fit_failures = _decode_failure_count(
                fit_result,
                force_rewrite=True,
            )
            forced_report = forced_result.report
            best_report = best_result.report
            tokenized_count = sum(
                record.rewrite is not None for record in forced_result.records
            )
            standalone_seconds = (
                fit_seconds + fit_transform_seconds + eval_seconds
            )

        best_used_rule_ranks = {
            rank
            for record in (best_result.records if rule_count > 0 else [])
            if record.use_rewrite and record.rewrite is not None
            for rank, _ in record.rewrite.selected
        }
        identity = _candidate_identity(compressor, rule_count)
        token_graphs_by_rule[rule_count] = token_graphs
        graph_sizes_by_rule[rule_count] = graph_sizes
        points.append(
            {
                "rule_count": int(rule_count),
                "applied_rule_count": int(rule_count),
                "mdl_selected_rule_count": int(len(best_used_rule_ranks)),
                **identity,
                "shared_dictionary_fit_seconds": float(fit_seconds),
                "fit_transform_seconds": float(fit_transform_seconds),
                "eval_transform_seconds": float(eval_seconds),
                "diagnostic_accounting_seconds": float(
                    best_accounting_seconds
                ),
                "incremental_transform_seconds": float(
                    fit_transform_seconds + eval_seconds
                ),
                "sweep_incremental_seconds": float(
                    fit_transform_seconds
                    + eval_seconds
                    + best_accounting_seconds
                ),
                "standalone_preparation_seconds": float(standalone_seconds),
                "tokenized_graph_count": int(tokenized_count),
                "eval_graph_count": len(eval_graphs),
                "fit_decode_failures": int(fit_failures),
                "eval_decode_failures": int(eval_failures),
                "decode_failures": int(fit_failures + eval_failures),
                "forced_n_rewritten": int(forced_report.n_rewritten),
                "forced_n_occurrences": int(forced_report.n_occurrences),
                "forced_baseline_bits": float(forced_report.baseline_bits),
                "forced_dictionary_bits": float(forced_report.dictionary_bits),
                "forced_encoded_bits": float(forced_report.encoded_bits),
                "forced_net_savings_bits": (
                    0.0
                    if rule_count == 0
                    else float(forced_report.net_savings_bits)
                ),
                "best_model_n_rewritten": int(best_report.n_rewritten),
                "best_model_encoded_bits": float(best_report.encoded_bits),
                "best_model_net_savings_bits": float(
                    best_report.net_savings_bits
                ),
            }
        )

    assert raw_graphs is not None
    candidate_table = compressor.candidate_frame().sort_values("rank").to_dict(
        orient="records"
    )
    dictionary_path = compressor.dictionary_path_frame().to_dict(orient="records")
    sweep_total_seconds = float(fit_seconds) + sum(
        float(point["sweep_incremental_seconds"])
        for point in points
    )
    metrics = {
        "rule_counts": list(rule_counts),
        "sweep_total_preparation_seconds": sweep_total_seconds,
        "available_candidate_count": int(available),
        "shared_dictionary_fit_seconds": float(fit_seconds),
        "points": points,
        "candidate_table": candidate_table,
        "dictionary_path": dictionary_path,
    }
    graphs = {
        "raw_graphs": raw_graphs,
        "token_graphs_by_rule": token_graphs_by_rule,
        "graph_sizes_by_rule": graph_sizes_by_rule,
    }
    return metrics, graphs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--graph-output", type=Path, required=True)
    args = parser.parse_args()

    tracemalloc.start()
    payload = _read_pickle(args.payload)
    metrics, graphs = run(payload)
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics["python_peak_mb"] = python_peak / (1024.0**2)
    metrics["peak_rss_mb"] = _peak_rss_mb()
    _write_pickle(args.graph_output, graphs)
    args.result.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n"
    )


if __name__ == "__main__":
    main()

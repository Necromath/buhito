#!/usr/bin/env python3
"""Benchmark graphlet enumeration on original and MDL-tokenized Reddit graphs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

from buhito.benchmarks import (
    InsufficientEligibleGraphsError,
    PreparedStateError,
    RuntimeBenchmarkConfig,
    WorkerTimeoutError,
    active_sample_caps,
    aggregate_prepared_results,
    benchmark_scalar_summary,
    filter_graph_complexity,
    graph_complexity_frame,
    prepare_runtime_benchmark,
    print_sample_manifest,
    resolve_benchmark_options,
    run_prepared_task,
    run_prepared_tasks,
    run_runtime_benchmark,
    save_sample_manifest,
    select_sample_manifest,
    write_slurm_array_script,
)
from buhito.datasets import load_tu_dataset


def parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(sorted({int(item) for item in value.split(",") if item}))
    if not sizes:
        raise argparse.ArgumentTypeError("Provide at least one graphlet size.")
    return sizes


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be nonnegative.")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, run, and aggregate isolated raw-versus-tokenized "
            "Buhito graphlet benchmarks locally or through an HPC array."
        )
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset", default="REDDIT-MULTI-5K")
    parser.add_argument("--node-label-mode", default="auto")
    parser.add_argument("--edge-label-mode", default="auto")
    parser.add_argument("--fit-size", type=int, default=None)
    parser.add_argument("--eval-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size-bins", type=int, default=None)
    parser.add_argument("--graphlet-sizes", type=parse_sizes, default=(3,))
    parser.add_argument("--n-rules", type=int, default=None)
    parser.add_argument("--min-graph-support", type=int, default=2)
    parser.add_argument("--min-occurrences", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-nodes", type=nonnegative_int, default=None)
    parser.add_argument("--max-edges", type=nonnegative_int, default=None)
    parser.add_argument("--max-degree", type=nonnegative_int, default=None)
    parser.add_argument("--max-wedges", type=nonnegative_int, default=None)
    parser.add_argument(
        "--mode",
        choices=("selected", "forced"),
        default="forced",
    )
    parser.add_argument(
        "--backend", choices=("buhito", "exhaustive"), default="buhito"
    )
    parser.add_argument(
        "--compressor-backend",
        choices=("buhito", "exhaustive"),
        default="buhito",
    )
    parser.add_argument(
        "--token-projection", choices=("simple", "native"), default="simple"
    )
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup-repeats", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--jobs", type=positive_int, default=1)
    parser.add_argument(
        "--phase-timeout-seconds", type=positive_float, default=None
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runtime/reddit_tokenization"),
    )
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--task-id", type=nonnegative_int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--write-slurm-script", type=Path)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--print-sample-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        **resolve_benchmark_options(vars(args), smoke=bool(args.smoke))
    )


def _serializable_configuration(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in sorted(vars(args).items()):
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def _phase_start(name: str) -> float:
    print(f"[buhito-cli] START {name}", flush=True)
    return time.perf_counter()


def _phase_end(name: str, started: float, *, detail: str = "") -> None:
    elapsed = time.perf_counter() - started
    suffix = f" {detail}" if detail else ""
    print(
        f"[buhito-cli] END {name} elapsed_seconds={elapsed:.6f}{suffix}",
        flush=True,
    )


def _prepared_dir(args: argparse.Namespace) -> Path:
    return args.prepared_dir or (args.output_dir / "prepared")


def _save_and_report(args: argparse.Namespace, result: Any) -> int:
    phase = "artifact writing"
    started = _phase_start(phase)
    output = result.save(args.output_dir, plots=args.plots)
    _phase_end(phase, started, detail=f"output_dir={output}")
    print("Headline results", flush=True)
    for key, value in benchmark_scalar_summary(result).items():
        print(f"{key}: {value}", flush=True)
    if result.config.force_rewrite:
        print(
            "Forced mode is a computational tokenization experiment; "
            "negative MDL savings are not positive compression.",
            flush=True,
        )
    return 0


def run(args: argparse.Namespace) -> int:
    args = resolve_args(args)
    prepared_dir = _prepared_dir(args)
    task_requested = args.task_id is not None or (
        args.prepared_dir is not None
        and os.environ.get("SLURM_ARRAY_TASK_ID") is not None
    )

    if task_requested:
        output = run_prepared_task(
            prepared_dir,
            task_id=args.task_id,
            task_manifest=args.task_manifest,
            results_dir=args.results_dir,
        )
        print(f"Task result: {output}", flush=True)
        return 0

    if args.aggregate_only:
        result = aggregate_prepared_results(
            prepared_dir,
            task_manifest=args.task_manifest,
            results_dir=args.results_dir,
        )
        return _save_and_report(args, result)

    if args.data_root is None:
        raise ValueError(
            "--data-root is required for preview and prepare stages."
        )
    if args.fit_size < 1 or args.eval_size < 1:
        raise ValueError("--fit-size and --eval-size must be positive.")
    if args.size_bins < 1:
        raise ValueError("--size-bins must be positive.")

    print("Resolved configuration", flush=True)
    print(
        json.dumps(_serializable_configuration(args), indent=2, sort_keys=True),
        flush=True,
    )

    started = _phase_start("dataset loading")
    dataset = load_tu_dataset(
        args.data_root,
        args.dataset,
        node_label_mode=args.node_label_mode,
        edge_label_mode=args.edge_label_mode,
    )
    graphs = list(dataset.graphs)
    _phase_end(
        "dataset loading", started, detail=f"dataset_graph_count={len(graphs)}"
    )

    started = _phase_start("eligibility filtering")
    complexity = graph_complexity_frame(graphs)
    eligible = filter_graph_complexity(
        complexity,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_degree=args.max_degree,
        max_wedges=args.max_wedges,
    )
    _phase_end(
        "eligibility filtering",
        started,
        detail=f"eligible_graph_count={len(eligible)}",
    )

    started = _phase_start("sample selection")
    manifest, fit_indices, eval_indices = select_sample_manifest(
        graphs,
        dataset=args.dataset,
        fit_size=args.fit_size,
        eval_size=args.eval_size,
        seed=args.seed,
        n_bins=args.size_bins,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_degree=args.max_degree,
        max_wedges=args.max_wedges,
        complexity=complexity,
    )
    _phase_end("sample selection", started)
    print("Selected sample", flush=True)
    print_sample_manifest(manifest)
    save_sample_manifest(manifest, args.output_dir)

    if args.print_sample_only:
        print(
            "Sample preview complete. No compressor or worker process was created.",
            flush=True,
        )
        return 0

    forced = args.mode == "forced"
    cache_dir = args.cache_dir or (
        args.output_dir.parent / "cache" / f"{args.dataset}_{args.mode}"
    )
    compressor_kwargs = {
        "graphlet_sizes": args.graphlet_sizes,
        "n_rules": args.n_rules,
        "min_graph_support": args.min_graph_support,
        "min_occurrences": args.min_occurrences,
        "max_candidates": args.max_candidates,
        "node_label_keys": dataset.node_label_key,
        "edge_label_keys": dataset.edge_label_key,
        "selector": "sparse",
        "model_choice_bits": 1.0,
        "min_rule_savings_bits": -math.inf if forced else 0.0,
        "dictionary_selection": "fixed" if forced else "best",
        "cache_dir": str(cache_dir),
        "validate": True,
        "progress": True,
    }
    config = RuntimeBenchmarkConfig(
        graphlet_sizes=args.graphlet_sizes,
        backend=args.backend,
        compressor_backend=args.compressor_backend,
        repeats=args.repeats,
        warmup_repeats=args.warmup_repeats,
        token_projection=args.token_projection,
        force_rewrite=forced,
        threads=args.threads,
        phase_timeout_seconds=args.phase_timeout_seconds,
        progress=True,
    )
    caps = active_sample_caps(
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_degree=args.max_degree,
        max_wedges=args.max_wedges,
    )

    started = _phase_start("benchmark preparation")
    prepare_runtime_benchmark(
        [graphs[index] for index in fit_indices],
        [graphs[index] for index in eval_indices],
        prepared_dir=prepared_dir,
        compressor_kwargs=compressor_kwargs,
        config=config,
        sample_manifest=manifest,
        metadata={
            "dataset": args.dataset,
            "dataset_size": len(graphs),
            "sample_seed": args.seed,
            "fit_indices": fit_indices,
            "eval_indices": eval_indices,
            "eligible_graph_count": len(eligible),
            "sampling": "safety-filtered size stratified",
            "safety_caps": caps,
            "mode": args.mode,
            "smoke": bool(args.smoke),
        },
    )
    _phase_end("benchmark preparation", started, detail=str(prepared_dir))

    if args.write_slurm_script is not None:
        script = write_slurm_array_script(
            prepared_dir, args.write_slurm_script
        )
        print(f"SLURM array script: {script}", flush=True)

    if args.prepare_only:
        print(
            "Preparation complete. Submit the task manifest locally or as an "
            "HPC array, then aggregate separately.",
            flush=True,
        )
        return 0

    started = _phase_start("benchmark tasks")
    run_prepared_tasks(
        prepared_dir,
        jobs=args.jobs,
        task_manifest=args.task_manifest,
        results_dir=args.results_dir,
    )
    _phase_end("benchmark tasks", started, detail=f"jobs={args.jobs}")

    result = aggregate_prepared_results(
        prepared_dir,
        task_manifest=args.task_manifest,
        results_dir=args.results_dir,
    )
    return _save_and_report(args, result)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        code = run(args)
    except (
        InsufficientEligibleGraphsError,
        PreparedStateError,
        WorkerTimeoutError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()

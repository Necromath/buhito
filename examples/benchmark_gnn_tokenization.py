#!/usr/bin/env python3
"""Benchmark a reference structural GCN on original and tokenized graphs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from buhito.benchmarks import PreparedStateError, WorkerTimeoutError
from buhito.benchmarks.gnn import (
    GNNBenchmarkConfig,
    aggregate_gnn_results,
    gnn_scalar_summary,
    prepare_gnn_benchmark,
    run_gnn_prepared_task,
    run_gnn_prepared_tasks,
    write_gnn_slurm_array_script,
)
from buhito.datasets import load_tu_dataset


SMOKE_DEFAULTS = {
    "repeats": 1,
    "warmup_steps": 1,
    "steps_per_repeat": 2,
    "epochs": 2,
    "hidden_channels": 16,
    "num_layers": 2,
    "batch_size": 4,
    "threads": 1,
    "quality_eval_fraction": 0.5,
    "phase_timeout_seconds": 600.0,
}

STANDARD_DEFAULTS = {
    "repeats": 5,
    "warmup_steps": 2,
    "steps_per_repeat": 10,
    "epochs": 10,
    "hidden_channels": 64,
    "num_layers": 2,
    "batch_size": 32,
    "threads": 1,
    "quality_eval_fraction": 0.2,
    "phase_timeout_seconds": None,
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be nonnegative.")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a reference structural GCN on the immutable original and "
            "motif-tokenized graphs produced by the Buhito runtime benchmark."
        )
    )
    parser.add_argument("--runtime-prepared-dir", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/gnn/reddit_tokenization"),
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset", default="REDDIT-MULTI-5K")
    parser.add_argument("--node-label-mode", default="auto")
    parser.add_argument("--edge-label-mode", default="auto")
    parser.add_argument(
        "--gnn-mode",
        choices=("inference", "training"),
        default="inference",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repeats", type=positive_int)
    parser.add_argument("--warmup-steps", type=nonnegative_int)
    parser.add_argument("--steps-per-repeat", type=positive_int)
    parser.add_argument("--epochs", type=positive_int)
    parser.add_argument("--hidden-channels", type=positive_int)
    parser.add_argument("--num-layers", type=positive_int)
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--threads", type=positive_int)
    parser.add_argument("--jobs", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=positive_float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--quality-eval-fraction",
        type=float,
        help=(
            "Fraction of paired graphs reserved for held-out predictive "
            "quality in training mode. The same indices are used for the "
            "original and tokenized representations."
        ),
    )
    parser.add_argument("--phase-timeout-seconds", type=positive_float)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--task-id", type=nonnegative_int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--write-slurm-script", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = SMOKE_DEFAULTS if args.smoke else STANDARD_DEFAULTS
    values = vars(args).copy()
    for name, value in defaults.items():
        if values.get(name) is None:
            values[name] = value
    if values["weight_decay"] < 0:
        raise ValueError("--weight-decay cannot be negative.")
    if values["prepared_dir"] is None:
        values["prepared_dir"] = values["output_dir"] / "prepared"
    return argparse.Namespace(**values)


def _labels(args: argparse.Namespace) -> list[Any] | None:
    if args.data_root is None:
        if args.gnn_mode == "training":
            raise ValueError(
                "--data-root is required for training so graph labels can be "
                "matched to the prepared evaluation sample."
            )
        return None
    if args.runtime_prepared_dir is None:
        raise ValueError("--runtime-prepared-dir is required when loading labels.")
    metadata_path = args.runtime_prepared_dir / "metadata.json"
    if not metadata_path.is_file():
        raise PreparedStateError(f"Runtime metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    eval_indices = metadata.get("eval_indices")
    if eval_indices is None:
        raise PreparedStateError(
            "Runtime metadata does not contain eval_indices needed for labels."
        )
    dataset = load_tu_dataset(
        args.data_root,
        args.dataset,
        node_label_mode=args.node_label_mode,
        edge_label_mode=args.edge_label_mode,
    )
    return [dataset.graph_labels[int(index)] for index in eval_indices]


def _configuration(args: argparse.Namespace) -> GNNBenchmarkConfig:
    return GNNBenchmarkConfig(
        mode=args.gnn_mode,
        repeats=args.repeats,
        warmup_steps=args.warmup_steps,
        steps_per_repeat=args.steps_per_repeat,
        epochs=args.epochs,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        device=args.device,
        threads=args.threads,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        quality_eval_fraction=args.quality_eval_fraction,
        phase_timeout_seconds=args.phase_timeout_seconds,
    )


def run(args: argparse.Namespace) -> int:
    args = resolve_args(args)
    task_requested = args.task_id is not None or (
        args.prepared_dir is not None
        and os.environ.get("SLURM_ARRAY_TASK_ID") is not None
    )
    if task_requested:
        output = run_gnn_prepared_task(
            args.prepared_dir,
            task_id=args.task_id,
            task_manifest=args.task_manifest,
            results_dir=args.results_dir,
        )
        print(f"GNN task result: {output}", flush=True)
        return 0
    if args.aggregate_only:
        result = aggregate_gnn_results(
            args.prepared_dir,
            task_manifest=args.task_manifest,
            results_dir=args.results_dir,
        )
        output = result.save(args.output_dir)
        print(f"GNN artifacts: {output}", flush=True)
        for key, value in gnn_scalar_summary(result).items():
            print(f"{key}: {value}", flush=True)
        return 0
    if args.runtime_prepared_dir is None:
        raise ValueError(
            "--runtime-prepared-dir is required when preparing a GNN benchmark."
        )
    config = _configuration(args)
    config.validate()
    print("Resolved GNN configuration", flush=True)
    print(json.dumps(config.__dict__, indent=2, sort_keys=True), flush=True)
    labels = _labels(args)
    prepare_gnn_benchmark(
        args.runtime_prepared_dir,
        prepared_dir=args.prepared_dir,
        config=config,
        labels=labels,
        metadata={
            "dataset": args.dataset,
            "data_root": str(args.data_root) if args.data_root else None,
        },
    )
    print(f"GNN prepared state: {args.prepared_dir}", flush=True)
    if args.write_slurm_script is not None:
        script = write_gnn_slurm_array_script(
            args.prepared_dir,
            args.write_slurm_script,
        )
        print(f"GNN SLURM array script: {script}", flush=True)
    if args.prepare_only:
        return 0
    run_gnn_prepared_tasks(
        args.prepared_dir,
        jobs=args.jobs,
        task_manifest=args.task_manifest,
        results_dir=args.results_dir,
    )
    result = aggregate_gnn_results(
        args.prepared_dir,
        task_manifest=args.task_manifest,
        results_dir=args.results_dir,
    )
    output = result.save(args.output_dir)
    print(f"GNN artifacts: {output}", flush=True)
    for key, value in gnn_scalar_summary(result).items():
        print(f"{key}: {value}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        code = run(args)
    except (
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

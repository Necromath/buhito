#!/usr/bin/env python3
"""Run a nested Buhito compression--speed--quality Pareto sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from buhito.benchmarks.gnn import GNNBenchmarkConfig
from buhito.benchmarks.pareto import (
    ParetoStatisticsConfig,
    aggregate_gnn_pareto_sweep,
    pareto_scalar_summary,
    prepare_gnn_pareto_sweep,
    run_pareto_task,
    run_pareto_tasks,
    write_pareto_slurm_array_script,
)
from buhito.benchmarks.runtime import PreparedStateError
from buhito.datasets import load_tu_dataset


def parse_rule_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--rule-counts must be comma-separated integers."
        ) from exc
    if not counts or counts[0] != 0 or any(item < 0 for item in counts):
        raise argparse.ArgumentTypeError(
            "--rule-counts must be nonnegative and include zero."
        )
    return counts


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-runtime-prepared-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset", default="REDDIT-MULTI-5K")
    parser.add_argument("--node-label-mode", default="auto")
    parser.add_argument("--edge-label-mode", default="auto")
    parser.add_argument(
        "--rule-counts",
        type=parse_rule_counts,
        default=(0, 1, 2),
    )
    parser.add_argument("--compressor-backend", choices=("buhito", "exhaustive"))
    parser.add_argument("--token-projection", choices=("simple", "native"))
    parser.add_argument("--cache-dir", type=Path)

    parser.add_argument(
        "--gnn-mode",
        choices=("training", "inference"),
        default="training",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=positive_int, default=200)
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--warmup-steps", type=nonnegative_int, default=2)
    parser.add_argument("--steps-per-repeat", type=positive_int, default=10)
    parser.add_argument("--hidden-channels", type=positive_int, default=64)
    parser.add_argument("--num-layers", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=positive_float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--quality-eval-fraction", type=float, default=0.2)
    parser.add_argument("--phase-timeout-seconds", type=positive_float, default=1800.0)

    parser.add_argument("--bootstrap-samples", type=positive_int, default=5000)
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help=(
            "Confidence level for two-sided bootstrap intervals and the "
            "one-sided non-inferiority lower bound."
        ),
    )
    parser.add_argument("--statistics-seed", type=int, default=0)
    parser.add_argument(
        "--accuracy-drop-tolerance",
        "--accuracy-noninferiority-margin",
        dest="accuracy_drop_tolerance",
        type=float,
        default=0.02,
        help="Largest acceptable tokenized-minus-original accuracy loss.",
    )
    parser.add_argument(
        "--macro-f1-drop-tolerance",
        "--macro-f1-noninferiority-margin",
        dest="macro_f1_drop_tolerance",
        type=float,
        default=0.02,
        help="Largest acceptable tokenized-minus-original macro-F1 loss.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/gnn/reddit5k_pareto"),
    )
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--task-id", type=nonnegative_int)
    parser.add_argument("--jobs", type=positive_int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-tasks-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--write-slurm-script", type=Path)
    parser.add_argument("--plots", action="store_true")
    return parser


def _prepared_dir(args: argparse.Namespace) -> Path:
    return args.prepared_dir or args.output_dir / "prepared"


def _reference_metadata(reference: Path) -> dict[str, Any]:
    path = reference / "metadata.json"
    if not path.is_file():
        raise PreparedStateError(f"Reference runtime metadata is missing: {path}")
    metadata = json.loads(path.read_text())
    if "fit_indices" not in metadata or "eval_indices" not in metadata:
        raise PreparedStateError(
            "Reference runtime metadata must contain fit_indices and eval_indices."
        )
    return metadata


def _compressor_kwargs(
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    kwargs = dict(metadata.get("compressor_kwargs", {}))
    kwargs.pop("enumerator", None)
    kwargs["n_rules"] = max(args.rule_counts)
    kwargs["dictionary_selection"] = "fixed"
    kwargs["min_rule_savings_bits"] = -math.inf
    kwargs["validate"] = True
    kwargs["progress"] = True
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)
    return kwargs


def _gnn_config(args: argparse.Namespace) -> GNNBenchmarkConfig:
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
    prepared = _prepared_dir(args)
    task_requested = args.task_id is not None or (
        args.prepared_dir is not None
        and os.environ.get("SLURM_ARRAY_TASK_ID") is not None
    )
    if task_requested:
        output = run_pareto_task(
            prepared,
            task_id=args.task_id,
            task_manifest=args.task_manifest,
            results_dir=args.results_dir,
        )
        print(f"Pareto task result: {output}", flush=True)
        return 0

    if args.run_tasks_only:
        outputs = run_pareto_tasks(
            prepared,
            jobs=args.jobs,
            task_manifest=args.task_manifest,
            results_dir=args.results_dir,
        )
        print(f"Completed Pareto tasks: {len(outputs)}", flush=True)
        return 0

    if args.aggregate_only:
        result = aggregate_gnn_pareto_sweep(
            prepared,
            task_manifest=args.task_manifest,
            results_dir=args.results_dir,
        )
        output = result.save(args.output_dir, plots=args.plots)
        print(f"Pareto artifacts: {output}", flush=True)
        for key, value in pareto_scalar_summary(result).items():
            print(f"{key}: {value}", flush=True)
        return 0

    if args.reference_runtime_prepared_dir is None or args.data_root is None:
        raise ValueError(
            "Preparation requires --reference-runtime-prepared-dir and --data-root."
        )
    metadata = _reference_metadata(args.reference_runtime_prepared_dir)
    dataset_name = str(metadata.get("dataset", args.dataset))
    print(f"Loading dataset {dataset_name}", flush=True)
    dataset = load_tu_dataset(
        args.data_root,
        dataset_name,
        node_label_mode=args.node_label_mode,
        edge_label_mode=args.edge_label_mode,
    )
    fit_indices = [int(value) for value in metadata["fit_indices"]]
    eval_indices = [int(value) for value in metadata["eval_indices"]]
    fit_graphs = [dataset.graphs[index] for index in fit_indices]
    eval_graphs = [dataset.graphs[index] for index in eval_indices]
    labels = [dataset.graph_labels[index] for index in eval_indices]

    gnn_config = _gnn_config(args)
    stats_config = ParetoStatisticsConfig(
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        statistics_seed=args.statistics_seed,
        accuracy_drop_tolerance=args.accuracy_drop_tolerance,
        macro_f1_drop_tolerance=args.macro_f1_drop_tolerance,
    )
    compressor_backend = args.compressor_backend or str(
        metadata.get("benchmark_config", {}).get("compressor_backend", "buhito")
    )
    token_projection = args.token_projection or str(
        metadata.get("benchmark_config", {}).get("token_projection", "simple")
    )
    print("Resolved Pareto configuration", flush=True)
    print(
        json.dumps(
            {
                "rule_counts": list(args.rule_counts),
                "fit_graph_count": len(fit_graphs),
                "eval_graph_count": len(eval_graphs),
                "gnn_config": gnn_config.__dict__,
                "statistics_config": stats_config.__dict__,
                "compressor_backend": compressor_backend,
                "token_projection": token_projection,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    prepare_gnn_pareto_sweep(
        fit_graphs,
        eval_graphs,
        labels,
        prepared_dir=prepared,
        rule_counts=args.rule_counts,
        compressor_kwargs=_compressor_kwargs(metadata, args),
        compressor_backend=compressor_backend,
        token_projection=token_projection,
        gnn_config=gnn_config,
        statistics_config=stats_config,
        metadata={
            "dataset": dataset_name,
            "data_root": str(args.data_root),
            "reference_runtime_prepared_dir": str(
                args.reference_runtime_prepared_dir.resolve()
            ),
            "fit_indices": fit_indices,
            "eval_indices": eval_indices,
        },
    )
    print(f"Pareto prepared state: {prepared}", flush=True)
    if args.write_slurm_script is not None:
        script = write_pareto_slurm_array_script(
            prepared, args.write_slurm_script
        )
        print(f"Pareto SLURM array script: {script}", flush=True)
    if args.prepare_only:
        return 0
    run_pareto_tasks(
        prepared,
        jobs=args.jobs,
        task_manifest=args.task_manifest,
        results_dir=args.results_dir,
    )
    result = aggregate_gnn_pareto_sweep(
        prepared,
        task_manifest=args.task_manifest,
        results_dir=args.results_dir,
    )
    output = result.save(args.output_dir, plots=args.plots)
    print(f"Pareto artifacts: {output}", flush=True)
    for key, value in pareto_scalar_summary(result).items():
        print(f"{key}: {value}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    exclusive_modes = sum(
        bool(value)
        for value in (
            args.prepare_only,
            args.run_tasks_only,
            args.aggregate_only,
        )
    )
    if exclusive_modes > 1:
        parser.error(
            "--prepare-only, --run-tasks-only, and --aggregate-only "
            "are mutually exclusive."
        )
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative.")
    if not 0.0 < args.quality_eval_fraction < 1.0:
        parser.error("--quality-eval-fraction must lie between zero and one.")
    if not 0.0 < args.confidence_level < 1.0:
        parser.error("--confidence-level must lie between zero and one.")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()

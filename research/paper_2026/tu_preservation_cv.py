#!/usr/bin/env python3
"""Prepare, run, and aggregate the paper TU preservation experiment.

One preparation task fits a single frozen candidate ranking per training fold.
One model task trains every representation for one dataset/fold/model seed,
which keeps original and contracted runs paired. Aggregation refuses incomplete
or decode-invalid results.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import random
import subprocess
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from buhito.benchmarks._gnn_worker import run as run_gnn
from buhito.benchmarks.gnn import GNNBenchmarkConfig, prepare_gnn_batches
from buhito.benchmarks.runtime import collapse_parallel_edges
from buhito.datasets import load_tu_dataset
from buhito.mdl import MDLGraphCompressor, _validate_rewrite_exact


DATASETS = ("MUTAG", "PTC_MR", "NCI1", "ENZYMES")
PREFIXES = (1, 2, 3, 4, 5)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n")


def _git_state() -> dict[str, Any]:
    def command(*args: str) -> str:
        return subprocess.check_output(args, text=True).strip()

    return {
        "commit": command("git", "rev-parse", "HEAD"),
        "branch": command("git", "branch", "--show-current"),
        "status_short": command("git", "status", "--short"),
    }


def _dataset_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*")):
        if not path.is_file():
            continue
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def _compressor(
    cache: Path,
    *,
    node_label_key: str | None,
    edge_label_key: str | None,
) -> MDLGraphCompressor:
    return MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=max(PREFIXES),
        min_graph_support=2,
        min_occurrences=2,
        max_candidates=100,
        node_label_keys=node_label_key,
        edge_label_keys=edge_label_key,
        selector="sparse",
        model_choice_bits=1.0,
        min_rule_savings_bits=-math.inf,
        dictionary_selection="fixed",
        cache_dir=cache,
        validate=True,
        progress=True,
    )


def _strict_rule_count(compressor: MDLGraphCompressor) -> int:
    candidates = compressor.candidate_frame().sort_values("rank")
    admitted = int((candidates["forced_net_savings_bits"] >= 0.0).sum())
    path = compressor.dictionary_path_frame()
    eligible = path[path["n_rules"].astype(int) <= admitted]
    return int(eligible.loc[eligible["net_savings_bits"].idxmax(), "n_rules"])


def _materialize(result: Any, rule_count: int) -> tuple[list[Any], list[dict[str, Any]]]:
    graphs = []
    rows = []
    for index, record in enumerate(result.records):
        use_rewrite = rule_count > 0 and record.rewrite is not None
        if use_rewrite and not _validate_rewrite_exact(
            record.baseline_graph, record.rewrite
        ):
            raise RuntimeError(f"Decode failure at graph {index}, prefix {rule_count}.")
        graph = (
            record.normalized_model_graph(force_rewrite=True)
            if use_rewrite
            else record.baseline_graph.copy()
        )
        graph = collapse_parallel_edges(graph)
        graphs.append(graph)
        raw_nodes = record.baseline_graph.number_of_nodes()
        raw_edges = record.baseline_graph.number_of_edges()
        rows.append({
            "local_index": index,
            "rule_count": rule_count,
            "raw_nodes": raw_nodes,
            "raw_edges": raw_edges,
            "model_nodes": graph.number_of_nodes(),
            "model_edges": graph.number_of_edges(),
            "node_reduction": raw_nodes - graph.number_of_nodes(),
            "edge_reduction": raw_edges - graph.number_of_edges(),
            "rewrite_available": bool(record.rewrite is not None),
            "selected_occurrences": int(record.selected_occurrences),
        })
    return graphs, rows


def prepare_dataset(args: argparse.Namespace) -> None:
    dataset = load_tu_dataset(
        args.data_root, args.dataset,
        node_label_mode="auto", edge_label_mode="auto",
    )
    labels_raw = np.asarray(dataset.graph_labels)
    label_values = sorted(set(labels_raw.tolist()))
    label_map = {value: index for index, value in enumerate(label_values)}
    labels = np.asarray([label_map[value] for value in labels_raw], dtype=int)
    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.split_seed
    )
    root = args.output_root / "prepared" / args.dataset
    root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels)
    ):
        fold_root = root / f"fold_{fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        compressor = _compressor(
            args.cache_root / args.dataset / f"fold_{fold}",
            node_label_key=dataset.node_label_key,
            edge_label_key=dataset.edge_label_key,
        )
        fit_graphs = [dataset.graphs[int(i)] for i in train_idx]
        test_graphs = [dataset.graphs[int(i)] for i in test_idx]
        started = time.perf_counter()
        compressor.fit(fit_graphs)
        fit_seconds = time.perf_counter() - started
        candidates = compressor.candidate_frame().sort_values("rank")
        path = compressor.dictionary_path_frame().sort_values("n_rules")
        candidates.to_csv(fold_root / "candidates.csv", index=False)
        path.to_csv(fold_root / "dictionary_path.csv", index=False)
        with gzip.open(fold_root / "compressor.pkl.gz", "wb") as handle:
            pickle.dump(compressor, handle, protocol=pickle.HIGHEST_PROTOCOL)
        strict_k = _strict_rule_count(compressor)
        available = min(len(candidates), max(PREFIXES))

        conditions: dict[str, dict[str, Any]] = {}
        for condition, count in [("original", 0), ("strict_mdl", strict_k)] + [
            (f"prefix_{count}", count) for count in PREFIXES if count <= available
        ]:
            train_result = compressor.transform_rule_prefix(
                fit_graphs, count, require_nonempty_rewrite=count > 0
            )
            test_result = compressor.transform_rule_prefix(
                test_graphs, count, require_nonempty_rewrite=count > 0
            )
            train_models, train_sizes = _materialize(train_result, count)
            test_models, test_sizes = _materialize(test_result, count)
            conditions[condition] = {
                "rule_count": count,
                "train_graphs": train_models,
                "test_graphs": test_models,
                "train_sizes": train_sizes,
                "test_sizes": test_sizes,
                "train_report": asdict(train_result.report),
                "test_report": asdict(test_result.report),
            }

        bundle = {
            "dataset": args.dataset,
            "fold": fold,
            "split_seed": args.split_seed,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "train_labels": labels[train_idx].tolist(),
            "test_labels": labels[test_idx].tolist(),
            "label_mapping": {str(key): value for key, value in label_map.items()},
            "num_classes": len(label_values),
            "strict_rule_count": strict_k,
            "conditions": conditions,
        }
        bundle_path = fold_root / "representations.pkl"
        with bundle_path.open("wb") as handle:
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        fingerprint = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        metadata = {
            "dataset": args.dataset,
            "dataset_directory": dataset.source_directory,
            "dataset_sha256": _dataset_digest(dataset.source_directory),
            "fold": fold,
            "split_seed": args.split_seed,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "strict_rule_count": strict_k,
            "available_prefixes": list(range(1, available + 1)),
            "fit_seconds": fit_seconds,
            "node_label_key": dataset.node_label_key,
            "edge_label_key": dataset.edge_label_key,
            "compressor": {
                "graphlet_sizes": [3], "n_rules": 5,
                "min_graph_support": 2, "min_occurrences": 2,
                "max_candidates": 100, "selector": "sparse",
                "model_choice_bits": 1.0,
                "ranking_min_rule_savings_bits": "-Infinity",
                "enumerator": compressor.enumerator.name,
            },
            "git": _git_state(),
            "python": {"executable": sys.executable, "version": sys.version},
            "platform": platform.platform(),
            "slurm": {
                key: os.environ.get(key)
                for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")
            },
            "representations_sha256": fingerprint,
        }
        _write_json(fold_root / "manifest.json", metadata)
        for model_seed in args.model_seeds:
            manifest_rows.append({
                "dataset": args.dataset, "fold": fold,
                "model_seed": model_seed,
                "bundle": str(bundle_path), "fingerprint": fingerprint,
            })

    pd.DataFrame(manifest_rows).to_csv(root / "task_manifest.csv", index=False)
    print(f"Prepared {args.dataset}: {len(manifest_rows)} paired model tasks")


def run_task(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(
        args.output_root / "prepared" / args.dataset / "task_manifest.csv"
    )
    row = manifest.iloc[args.task_id]
    bundle_path = Path(row["bundle"])
    if hashlib.sha256(bundle_path.read_bytes()).hexdigest() != row["fingerprint"]:
        raise RuntimeError("Prepared representation fingerprint mismatch.")
    with bundle_path.open("rb") as handle:
        bundle = pickle.load(handle)
    seed = int(row["model_seed"])
    config = GNNBenchmarkConfig(
        mode="training", epochs=args.epochs, repeats=1, warmup_steps=0,
        hidden_channels=args.hidden_channels, num_layers=args.num_layers,
        batch_size=args.batch_size, device=args.device, threads=args.threads,
        seed=seed, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    names = list(bundle["conditions"])
    random.Random(seed).shuffle(names)
    result_rows = []
    prediction_rows = []
    for order, name in enumerate(names):
        condition = bundle["conditions"][name]
        train_graphs = condition["train_graphs"]
        test_graphs = condition["test_graphs"]
        payload = {
            "config": asdict(config),
            "num_classes": bundle["num_classes"],
            "feature_dim": 5,
            "labels_available": True,
            "quality_metrics_available": True,
            "quality_metrics_reason": "outer stratified CV test fold",
            "representation": name,
            "batches": prepare_gnn_batches(
                train_graphs, bundle["train_labels"], batch_size=config.batch_size
            ),
            "train_batches": prepare_gnn_batches(
                train_graphs, bundle["train_labels"], batch_size=config.batch_size
            ),
            "quality_batches": prepare_gnn_batches(
                test_graphs, bundle["test_labels"], batch_size=config.batch_size
            ),
        }
        measured = run_gnn(payload)
        result_rows.append({
            "dataset": bundle["dataset"], "fold": bundle["fold"],
            "split_seed": bundle["split_seed"], "model_seed": seed,
            "condition": name, "rule_count": condition["rule_count"],
            "order": order,
            "accuracy": measured["quality_eval_accuracy"],
            "macro_f1": measured["quality_eval_macro_f1"],
            "loss": measured["quality_eval_loss"],
            "training_seconds": measured["workload_seconds"],
            "inference_seconds": measured["quality_evaluation_seconds"],
        })
        for local, graph_id in enumerate(bundle["test_indices"]):
            probabilities = measured["quality_eval_probabilities"][local]
            size = condition["test_sizes"][local]
            prediction_rows.append({
                "dataset": bundle["dataset"], "fold": bundle["fold"],
                "split_seed": bundle["split_seed"], "model_seed": seed,
                "graph_id": graph_id, "condition": name,
                "rule_count": condition["rule_count"],
                "true_label": measured["quality_eval_targets"][local],
                "predicted_label": measured["quality_eval_predictions"][local],
                "probabilities": json.dumps(probabilities),
                **{
                    f"probability_{class_index}": probability
                    for class_index, probability in enumerate(probabilities)
                },
                **size,
            })
    target = args.output_root / "task_results" / args.dataset
    target.mkdir(parents=True, exist_ok=True)
    stem = f"fold_{int(bundle['fold'])}_seed_{seed}"
    pd.DataFrame(result_rows).to_csv(target / f"{stem}_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        target / f"{stem}_predictions.csv", index=False
    )
    print(f"Completed {stem}: {len(names)} paired representations")


def aggregate(args: argparse.Namespace) -> None:
    metric_files = sorted((args.output_root / "task_results").glob("**/*_metrics.csv"))
    prediction_files = sorted(
        (args.output_root / "task_results").glob("**/*_predictions.csv")
    )
    if not metric_files or not prediction_files:
        raise RuntimeError("No model task outputs found.")
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    predictions = pd.concat(
        [pd.read_csv(path) for path in prediction_files], ignore_index=True
    )
    expected = 0
    for dataset in DATASETS:
        path = args.output_root / "prepared" / dataset / "task_manifest.csv"
        if path.exists():
            expected += len(pd.read_csv(path))
    completed = metrics[["dataset", "fold", "model_seed"]].drop_duplicates()
    if len(completed) != expected:
        raise RuntimeError(f"Incomplete tasks: expected {expected}, found {len(completed)}.")

    original = metrics[metrics.condition == "original"].rename(columns={
        "accuracy": "original_accuracy", "macro_f1": "original_macro_f1",
        "training_seconds": "original_training_seconds",
    })[["dataset", "fold", "model_seed", "original_accuracy",
        "original_macro_f1", "original_training_seconds"]]
    paired = metrics.merge(original, on=["dataset", "fold", "model_seed"])
    paired["accuracy_delta"] = paired.accuracy - paired.original_accuracy
    paired["macro_f1_delta"] = paired.macro_f1 - paired.original_macro_f1
    paired["training_speedup"] = (
        paired.original_training_seconds / paired.training_seconds
    )
    sizes = predictions.groupby(
        ["dataset", "fold", "model_seed", "condition", "rule_count"], as_index=False
    ).agg(raw_nodes=("raw_nodes", "sum"), model_nodes=("model_nodes", "sum"),
          raw_edges=("raw_edges", "sum"), model_edges=("model_edges", "sum"))
    sizes["node_reduction_fraction"] = 1 - sizes.model_nodes / sizes.raw_nodes
    sizes["edge_reduction_fraction"] = 1 - sizes.model_edges / sizes.raw_edges
    paired = paired.merge(
        sizes, on=["dataset", "fold", "model_seed", "condition", "rule_count"]
    )
    summary = paired.groupby(["dataset", "condition", "rule_count"], as_index=False).agg(
        n_pairs=("accuracy_delta", "size"),
        mean_node_reduction=("node_reduction_fraction", "mean"),
        mean_edge_reduction=("edge_reduction_fraction", "mean"),
        mean_accuracy_delta=("accuracy_delta", "mean"),
        se_accuracy_delta=("accuracy_delta", "sem"),
        mean_macro_f1_delta=("macro_f1_delta", "mean"),
        se_macro_f1_delta=("macro_f1_delta", "sem"),
        median_training_speedup=("training_speedup", "median"),
    )
    output = args.output_root / "aggregated"
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "model_runs.csv", index=False)
    predictions.to_csv(output / "graph_predictions.csv", index=False)
    paired.to_csv(output / "paired_effects.csv", index=False)
    summary.to_csv(output / "tu_preservation_summary.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    for axis, dataset in zip(axes.flat, DATASETS, strict=True):
        frame = summary[(summary.dataset == dataset) & (summary.condition != "strict_mdl")]
        axis.errorbar(
            frame.mean_node_reduction, frame.mean_accuracy_delta,
            yerr=1.96 * frame.se_accuracy_delta, marker="o", linewidth=1.5,
        )
        strict = summary[(summary.dataset == dataset) & (summary.condition == "strict_mdl")]
        axis.scatter(strict.mean_node_reduction, strict.mean_accuracy_delta,
                     marker="*", s=100, color="#d1495b", label="strict MDL")
        axis.axhline(0, color="0.5", linestyle="--", linewidth=1)
        axis.set_title(dataset.replace("_", "-"))
        axis.set_xlabel("Node reduction fraction")
        axis.set_ylabel("Accuracy delta")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"tu_preservation_map.{suffix}", dpi=300)
    plt.close(figure)
    print(f"Aggregated {len(completed)} paired tasks to {output}")


def parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("mode", choices=("prepare", "task", "aggregate"))
    result.add_argument("--data-root", type=Path)
    result.add_argument("--dataset", choices=DATASETS)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--cache-root", type=Path)
    result.add_argument("--folds", type=int, default=5)
    result.add_argument("--split-seed", type=int, default=20260909)
    result.add_argument("--model-seeds", type=parse_seeds, default=(0, 1, 2, 3, 4))
    result.add_argument("--task-id", type=int)
    result.add_argument("--epochs", type=int, default=200)
    result.add_argument("--hidden-channels", type=int, default=64)
    result.add_argument("--num-layers", type=int, default=3)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--device", default="cpu")
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--learning-rate", type=float, default=0.001)
    result.add_argument("--weight-decay", type=float, default=0.0001)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.mode == "prepare":
        if args.data_root is None or args.dataset is None or args.cache_root is None:
            raise ValueError("prepare requires --data-root, --dataset, and --cache-root")
        prepare_dataset(args)
    elif args.mode == "task":
        if args.dataset is None or args.task_id is None:
            raise ValueError("task requires --dataset and --task-id")
        run_task(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()

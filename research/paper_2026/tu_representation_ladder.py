#!/usr/bin/env python3
"""Materialize and fit the frozen-dictionary TU representation ladder.

The materialization stage reloads the fold compressors produced by
``tu_preservation_cv.py``.  It never fits a compressor.  Instead it verifies
the original bundle and dataset fingerprints, rematerializes each frozen rule
prefix, checks that its carrier equals the previously serialized carrier, and
extracts canonical boundary-port histograms before parallel-edge projection
discards the port maps.

The fitting stage holds the feature dimension fixed and gates two feature
bundles to expose the following learner views:

* ``carrier``: anonymous contracted carrier;
* ``carrier_motif``: carrier plus motif-rank counts;
* ``carrier_motif_ports``: carrier plus motif-rank and canonical-port counts.

The original graph is fitted once per fold/seed as the paired baseline.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import subprocess
from typing import Any, Iterable, Mapping

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from buhito.benchmarks.runtime import collapse_parallel_edges
from buhito.datasets import load_tu_dataset
from buhito.mdl import _validate_rewrite_exact


DATASETS = ("MUTAG", "PTC_MR", "NCI1", "ENZYMES")
PREFIXES = (1, 2, 3, 4, 5)
VIEWS = ("carrier", "carrier_motif", "carrier_motif_ports")
NODE_LABEL = "__buhito_mdl_node_label__"
EDGE_LABEL = "__buhito_mdl_edge_label__"
NODE_KIND = "__buhito_mdl_kind__"
PORTS = "__buhito_mdl_ports__"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_digest(directory: Path) -> str:
    """Reproduce the digest used by ``tu_preservation_cv.py`` exactly."""

    digest = hashlib.sha256()
    for path in sorted(directory.glob("*")):
        if not path.is_file():
            continue
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--short")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def stable_token(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def motif_node(data: Mapping[str, Any]) -> tuple[bool, int]:
    label = data.get(NODE_LABEL, data.get("mdl_label"))
    kind = data.get(NODE_KIND, data.get("mdl_kind"))
    is_motif = kind == "motif" or (
        isinstance(label, tuple) and bool(label) and label[0] == "motif"
    )
    rank = -1
    if is_motif and isinstance(label, tuple) and len(label) > 1:
        try:
            rank = int(label[1])
        except (TypeError, ValueError):
            pass
    return is_motif, rank


def build_vocabulary(graphs: Iterable[nx.Graph]) -> dict[str, list[str]]:
    """Fit categorical vocabularies on original training graphs only."""

    node_labels: set[str] = set()
    edge_labels: set[str] = set()
    for graph in graphs:
        for _, data in graph.nodes(data=True):
            is_motif, _ = motif_node(data)
            if not is_motif:
                node_labels.add(stable_token(data.get(NODE_LABEL, "<missing>")))
        for _, _, data in graph.edges(data=True):
            edge_labels.add(stable_token(data.get(EDGE_LABEL, "<missing>")))
    return {"node_labels": sorted(node_labels), "edge_labels": sorted(edge_labels)}


def feature_names(
    vocabulary: Mapping[str, list[str]],
    *,
    maximum_rule_count: int = 5,
    maximum_ports: int = 3,
) -> list[str]:
    result = [
        "nodes", "edges", "density", "degree_mean", "degree_std",
        "degree_max", "components", "average_clustering", "transitivity",
        "triangles",
    ]
    result.extend(f"node_label::{label}" for label in vocabulary["node_labels"])
    result.append("node_label::<unknown>")
    result.extend(f"motif_rank::{rank}" for rank in range(maximum_rule_count))
    result.append("motif_rank::<unknown>")
    result.extend(f"edge_label::{label}" for label in vocabulary["edge_labels"])
    result.append("edge_label::<unknown>")
    result.extend(
        f"port_rank::{rank}::port::{port}"
        for rank in range(maximum_rule_count)
        for port in range(maximum_ports)
    )
    result.append("port_rank::<unknown>")
    return result


def graph_features(
    graph: nx.Graph,
    vocabulary: Mapping[str, list[str]],
    *,
    expose_motif: bool,
    expose_ports: bool,
    port_histogram: Mapping[tuple[int, int], int] | None = None,
    unknown_ports: int = 0,
    maximum_rule_count: int = 5,
    maximum_ports: int = 3,
) -> np.ndarray:
    """Return one union-schema feature vector with gated side information."""

    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degree = np.asarray([value for _, value in graph.degree()], dtype=float)
    values = [
        float(n),
        float(m),
        nx.density(graph) if n > 1 else 0.0,
        float(degree.mean()) if degree.size else 0.0,
        float(degree.std()) if degree.size else 0.0,
        float(degree.max()) if degree.size else 0.0,
        float(nx.number_connected_components(graph)) if n else 0.0,
        nx.average_clustering(graph) if n > 1 else 0.0,
        nx.transitivity(graph) if n > 2 else 0.0,
        sum(nx.triangles(graph).values()) / 3.0 if n else 0.0,
    ]

    node_counts = Counter()
    motif_counts = [0] * maximum_rule_count
    unknown_nodes = 0
    unknown_motifs = 0
    known_nodes = set(vocabulary["node_labels"])
    for _, data in graph.nodes(data=True):
        is_motif, rank = motif_node(data)
        if is_motif:
            if 0 <= rank < maximum_rule_count:
                motif_counts[rank] += 1
            else:
                unknown_motifs += 1
            continue
        label = stable_token(data.get(NODE_LABEL, "<missing>"))
        if label in known_nodes:
            node_counts[label] += 1
        else:
            unknown_nodes += 1
    values.extend(float(node_counts[label]) for label in vocabulary["node_labels"])
    values.append(float(unknown_nodes))
    values.extend(float(count if expose_motif else 0) for count in motif_counts)
    values.append(float(unknown_motifs if expose_motif else 0))

    edge_counts = Counter()
    unknown_edges = 0
    known_edges = set(vocabulary["edge_labels"])
    for _, _, data in graph.edges(data=True):
        label = stable_token(data.get(EDGE_LABEL, "<missing>"))
        if label in known_edges:
            edge_counts[label] += 1
        else:
            unknown_edges += 1
    values.extend(float(edge_counts[label]) for label in vocabulary["edge_labels"])
    values.append(float(unknown_edges))

    ports = port_histogram or {}
    values.extend(
        float(ports.get((rank, port), 0) if expose_ports else 0)
        for rank in range(maximum_rule_count)
        for port in range(maximum_ports)
    )
    values.append(float(unknown_ports if expose_ports else 0))
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite ladder feature encountered.")
    return result


def port_histogram(
    rewrite: Any,
    *,
    maximum_rule_count: int,
    maximum_ports: int,
) -> tuple[dict[tuple[int, int], int], int]:
    """Count canonical boundary incidences by rule rank and motif port."""

    counts: Counter[tuple[int, int]] = Counter()
    unknown = 0
    if rewrite is None:
        return {}, 0
    for _, _, _, data in rewrite.template.edges(keys=True, data=True):
        assignments = data.get(PORTS, {})
        for supernode, raw_port in assignments.items():
            try:
                rank = int(rewrite.supernode_rule[supernode])
                port = int(raw_port)
            except (KeyError, TypeError, ValueError):
                unknown += 1
                continue
            if 0 <= rank < maximum_rule_count and 0 <= port < maximum_ports:
                counts[(rank, port)] += 1
            else:
                unknown += 1
    return dict(counts), unknown


def graphs_equal(left: nx.Graph, right: nx.Graph) -> bool:
    """Strict equality for a previously frozen normalized carrier."""

    return nx.utils.graphs_equal(left, right)


def _rematerialize(
    compressor: Any,
    graphs: list[nx.Graph],
    rule_count: int,
    frozen_graphs: list[nx.Graph],
    *,
    maximum_rule_count: int,
    maximum_ports: int,
) -> tuple[list[dict[tuple[int, int], int]], list[int]]:
    if rule_count == 0:
        return [{} for _ in graphs], [0 for _ in graphs]
    result = compressor.transform_rule_prefix(
        graphs, rule_count, require_nonempty_rewrite=True
    )
    histograms: list[dict[tuple[int, int], int]] = []
    unknowns: list[int] = []
    for index, (record, frozen) in enumerate(zip(result.records, frozen_graphs, strict=True)):
        if record.rewrite is not None and not _validate_rewrite_exact(
            record.baseline_graph, record.rewrite
        ):
            raise RuntimeError(
                f"Decode failure while rematerializing graph {index}, prefix {rule_count}."
            )
        projected = (
            record.normalized_model_graph(force_rewrite=True)
            if record.rewrite is not None
            else record.baseline_graph.copy()
        )
        projected = collapse_parallel_edges(projected)
        if not graphs_equal(projected, frozen):
            raise RuntimeError(
                f"Frozen carrier mismatch at graph {index}, prefix {rule_count}."
            )
        histogram, unknown = port_histogram(
            record.rewrite,
            maximum_rule_count=maximum_rule_count,
            maximum_ports=maximum_ports,
        )
        histograms.append(histogram)
        unknowns.append(unknown)
    return histograms, unknowns


def materialize(args: argparse.Namespace) -> None:
    prepared = args.input_root / "prepared" / args.dataset
    dataset = load_tu_dataset(
        args.data_root, args.dataset, node_label_mode="auto", edge_label_mode="auto"
    )
    dataset_directory = Path(dataset.source_directory)
    output_dataset = args.output_root / "prepared" / args.dataset
    output_dataset.mkdir(parents=True, exist_ok=True)
    task_rows: list[dict[str, Any]] = []

    for fold in range(args.folds):
        fold_root = prepared / f"fold_{fold}"
        manifest_path = fold_root / "manifest.json"
        bundle_path = fold_root / "representations.pkl"
        compressor_path = fold_root / "compressor.pkl.gz"
        manifest = json.loads(manifest_path.read_text())
        if manifest["dataset"] != args.dataset or int(manifest["fold"]) != fold:
            raise RuntimeError(f"Manifest identity mismatch in {manifest_path}.")
        actual_bundle_hash = sha256_file(bundle_path)
        if actual_bundle_hash != manifest["representations_sha256"]:
            raise RuntimeError(f"Representation fingerprint mismatch in fold {fold}.")
        actual_dataset_hash = dataset_digest(dataset_directory)
        if actual_dataset_hash != manifest["dataset_sha256"]:
            raise RuntimeError(f"Dataset fingerprint mismatch in fold {fold}.")
        with bundle_path.open("rb") as handle:
            bundle = pickle.load(handle)
        with gzip.open(compressor_path, "rb") as handle:
            compressor = pickle.load(handle)
        if bundle["train_indices"] != manifest["train_indices"]:
            raise RuntimeError(f"Training indices disagree in fold {fold}.")
        if bundle["test_indices"] != manifest["test_indices"]:
            raise RuntimeError(f"Test indices disagree in fold {fold}.")

        train_raw = [dataset.graphs[int(index)] for index in bundle["train_indices"]]
        test_raw = [dataset.graphs[int(index)] for index in bundle["test_indices"]]
        original_train = bundle["conditions"]["original"]["train_graphs"]
        vocabulary = build_vocabulary(original_train)
        max_rule_count = max(PREFIXES)
        frozen_prefix = compressor.rule_prefix(max_rule_count)
        max_ports = max(
            (rule.motif.number_of_nodes() for rule in frozen_prefix), default=0
        )
        if max_ports < 1:
            max_ports = max(manifest["compressor"]["graphlet_sizes"])
        names = feature_names(
            vocabulary,
            maximum_rule_count=max_rule_count,
            maximum_ports=max_ports,
        )

        materialized: dict[str, dict[str, Any]] = {}
        rematerialized: dict[int, tuple[list[Any], list[int], list[Any], list[int]]] = {}
        for condition_name, condition in bundle["conditions"].items():
            count = int(condition["rule_count"])
            if count not in rematerialized:
                train_ports, train_unknown = _rematerialize(
                    compressor, train_raw, count, condition["train_graphs"],
                    maximum_rule_count=max_rule_count, maximum_ports=max_ports,
                )
                test_ports, test_unknown = _rematerialize(
                    compressor, test_raw, count, condition["test_graphs"],
                    maximum_rule_count=max_rule_count, maximum_ports=max_ports,
                )
                rematerialized[count] = (
                    train_ports, train_unknown, test_ports, test_unknown
                )
            train_ports, train_unknown, test_ports, test_unknown = rematerialized[count]

            view_names = ("original",) if condition_name == "original" else VIEWS
            for view in view_names:
                expose_motif = view in ("carrier_motif", "carrier_motif_ports")
                expose_ports = view == "carrier_motif_ports"
                train_x = np.stack([
                    graph_features(
                        graph, vocabulary, expose_motif=expose_motif,
                        expose_ports=expose_ports, port_histogram=train_ports[index],
                        unknown_ports=train_unknown[index],
                        maximum_rule_count=max_rule_count, maximum_ports=max_ports,
                    )
                    for index, graph in enumerate(condition["train_graphs"])
                ])
                test_x = np.stack([
                    graph_features(
                        graph, vocabulary, expose_motif=expose_motif,
                        expose_ports=expose_ports, port_histogram=test_ports[index],
                        unknown_ports=test_unknown[index],
                        maximum_rule_count=max_rule_count, maximum_ports=max_ports,
                    )
                    for index, graph in enumerate(condition["test_graphs"])
                ])
                key = "original" if view == "original" else f"{condition_name}__{view}"
                materialized[key] = {
                    "condition": condition_name,
                    "view": view,
                    "rule_count": count,
                    "train_x": train_x,
                    "test_x": test_x,
                    "train_sizes": condition["train_sizes"],
                    "test_sizes": condition["test_sizes"],
                }

        result = {
            "dataset": args.dataset,
            "fold": fold,
            "split_seed": bundle["split_seed"],
            "train_indices": bundle["train_indices"],
            "test_indices": bundle["test_indices"],
            "train_labels": bundle["train_labels"],
            "test_labels": bundle["test_labels"],
            "num_classes": bundle["num_classes"],
            "strict_rule_count": bundle["strict_rule_count"],
            "vocabulary": vocabulary,
            "feature_names": names,
            "maximum_rule_count": max_rule_count,
            "maximum_ports": max_ports,
            "representations": materialized,
        }
        destination = output_dataset / f"fold_{fold}"
        destination.mkdir(parents=True, exist_ok=True)
        result_path = destination / "ladder_features.pkl"
        with result_path.open("wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        fingerprint = sha256_file(result_path)
        metadata = {
            "dataset": args.dataset,
            "fold": fold,
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_representations_sha256": actual_bundle_hash,
            "source_compressor_sha256": sha256_file(compressor_path),
            "dataset_sha256": actual_dataset_hash,
            "ladder_features_sha256": fingerprint,
            "feature_count": len(names),
            "feature_names": names,
            "views": list(VIEWS),
            "git": git_state(),
            "slurm": {
                key: os.environ.get(key)
                for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")
            },
        }
        (destination / "manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        for seed in args.model_seeds:
            task_rows.append({
                "dataset": args.dataset,
                "fold": fold,
                "model_seed": seed,
                "bundle": str(result_path),
                "fingerprint": fingerprint,
            })

    pd.DataFrame(task_rows).to_csv(output_dataset / "task_manifest.csv", index=False)
    print(f"Materialized {args.dataset}: {len(task_rows)} paired ladder tasks")


def fit_task(args: argparse.Namespace) -> None:
    manifest_path = args.input_root / "prepared" / args.dataset / "task_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    row = manifest.iloc[args.task_id]
    bundle_path = Path(row["bundle"])
    actual = sha256_file(bundle_path)
    if actual != str(row["fingerprint"]):
        raise RuntimeError("Ladder feature fingerprint mismatch.")
    with bundle_path.open("rb") as handle:
        bundle = pickle.load(handle)
    seed = int(row["model_seed"])

    from xgboost import XGBClassifier

    keys = list(bundle["representations"])
    random.Random(seed).shuffle(keys)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    train_y = np.asarray(bundle["train_labels"], dtype=int)
    test_y = np.asarray(bundle["test_labels"], dtype=int)
    for order, key in enumerate(keys):
        representation = bundle["representations"][key]
        model = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            min_child_weight=1.0,
            subsample=0.85,
            # Every view uses the same union schema, with hidden bundles set
            # to zero.  Full column sampling prevents zero-only gated columns
            # from reducing the effective feature budget of a view.
            colsample_bytree=args.colsample_bytree,
            reg_lambda=1.0,
            objective=("binary:logistic" if bundle["num_classes"] == 2 else "multi:softprob"),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=args.threads,
        )
        model.fit(representation["train_x"], train_y)
        probabilities = model.predict_proba(representation["test_x"])
        predictions = probabilities.argmax(axis=1)
        metric_rows.append({
            "dataset": bundle["dataset"],
            "fold": bundle["fold"],
            "split_seed": bundle["split_seed"],
            "model_seed": seed,
            "representation": key,
            "condition": representation["condition"],
            "view": representation["view"],
            "rule_count": int(representation["rule_count"]),
            "order": order,
            "accuracy": accuracy_score(test_y, predictions),
            "macro_f1": f1_score(test_y, predictions, average="macro"),
            "log_loss": log_loss(
                test_y, probabilities, labels=list(range(bundle["num_classes"]))
            ),
            "feature_count": len(bundle["feature_names"]),
        })
        for local_index, graph_id in enumerate(bundle["test_indices"]):
            size = representation["test_sizes"][local_index]
            values = probabilities[local_index].tolist()
            prediction_rows.append({
                "dataset": bundle["dataset"],
                "fold": bundle["fold"],
                "split_seed": bundle["split_seed"],
                "model_seed": seed,
                "graph_id": graph_id,
                "representation": key,
                "condition": representation["condition"],
                "view": representation["view"],
                "rule_count": int(representation["rule_count"]),
                "true_label": int(test_y[local_index]),
                "predicted_label": int(predictions[local_index]),
                "probabilities": json.dumps(values),
                **{f"probability_{index}": value for index, value in enumerate(values)},
                **size,
            })

    destination = args.output_root / "task_results" / args.dataset
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"fold_{int(bundle['fold'])}_seed_{seed}"
    pd.DataFrame(metric_rows).to_csv(destination / f"{stem}_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        destination / f"{stem}_predictions.csv", index=False
    )
    metadata = {
        "dataset": bundle["dataset"],
        "fold": bundle["fold"],
        "model_seed": seed,
        "ladder_fingerprint": actual,
        "feature_names": bundle["feature_names"],
        "views": list(VIEWS),
        "estimator": "xgboost.XGBClassifier",
        "parameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "colsample_bytree": args.colsample_bytree,
            "threads": args.threads,
        },
        "git": git_state(),
    }
    (destination / f"{stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Completed {args.dataset} {stem}: {len(keys)} representations")


def aggregate(args: argparse.Namespace) -> None:
    metric_files = sorted(args.output_root.glob("task_results/**/*_metrics.csv"))
    prediction_files = sorted(args.output_root.glob("task_results/**/*_predictions.csv"))
    metadata_files = sorted(args.output_root.glob("task_results/**/*_metadata.json"))
    if (len(metric_files), len(prediction_files), len(metadata_files)) != (100, 100, 100):
        raise RuntimeError(
            "Expected 100 metric, prediction, and metadata files; found "
            f"{len(metric_files)}, {len(prediction_files)}, and {len(metadata_files)}."
        )
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    predictions = pd.concat(
        [pd.read_csv(path) for path in prediction_files], ignore_index=True
    )
    original = metrics[metrics.representation == "original"].rename(columns={
        "accuracy": "original_accuracy",
        "macro_f1": "original_macro_f1",
        "log_loss": "original_log_loss",
    })[[
        "dataset", "fold", "model_seed", "original_accuracy",
        "original_macro_f1", "original_log_loss",
    ]]
    paired = metrics.merge(original, on=["dataset", "fold", "model_seed"])
    paired["accuracy_delta"] = paired.accuracy - paired.original_accuracy
    paired["macro_f1_delta"] = paired.macro_f1 - paired.original_macro_f1
    paired["log_loss_delta"] = paired.log_loss - paired.original_log_loss
    sizes = predictions.groupby(
        ["dataset", "fold", "model_seed", "representation", "condition", "view", "rule_count"],
        as_index=False,
    ).agg(
        raw_nodes=("raw_nodes", "sum"), model_nodes=("model_nodes", "sum"),
        raw_edges=("raw_edges", "sum"), model_edges=("model_edges", "sum"),
    )
    sizes["node_reduction_fraction"] = 1 - sizes.model_nodes / sizes.raw_nodes
    sizes["edge_reduction_fraction"] = 1 - sizes.model_edges / sizes.raw_edges
    paired = paired.merge(
        sizes,
        on=["dataset", "fold", "model_seed", "representation", "condition", "view", "rule_count"],
    )
    fold_effects = paired.groupby(
        ["dataset", "representation", "condition", "view", "rule_count", "fold"],
        as_index=False,
    ).agg(
        accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
        accuracy_delta=("accuracy_delta", "mean"),
        macro_f1_delta=("macro_f1_delta", "mean"),
        log_loss_delta=("log_loss_delta", "mean"),
        node_reduction_fraction=("node_reduction_fraction", "mean"),
        edge_reduction_fraction=("edge_reduction_fraction", "mean"),
    )
    summary = fold_effects.groupby(
        ["dataset", "representation", "condition", "view", "rule_count"],
        as_index=False,
    ).agg(
        folds=("fold", "size"), accuracy=("accuracy", "mean"),
        macro_f1=("macro_f1", "mean"),
        accuracy_delta=("accuracy_delta", "mean"),
        accuracy_delta_se=("accuracy_delta", "sem"),
        macro_f1_delta=("macro_f1_delta", "mean"),
        macro_f1_delta_se=("macro_f1_delta", "sem"),
        log_loss_delta=("log_loss_delta", "mean"),
        node_reduction_fraction=("node_reduction_fraction", "mean"),
        edge_reduction_fraction=("edge_reduction_fraction", "mean"),
    )
    destination = args.output_root / "aggregated"
    destination.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(destination / "ladder_model_runs.csv", index=False)
    predictions.to_csv(destination / "ladder_graph_predictions.csv", index=False)
    paired.to_csv(destination / "ladder_paired_effects.csv", index=False)
    fold_effects.to_csv(destination / "ladder_fold_effects.csv", index=False)
    summary.to_csv(destination / "ladder_summary.csv", index=False)
    print(f"Aggregated representation ladder to {destination}")


def parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("mode", choices=("materialize", "fit", "aggregate"))
    result.add_argument("--data-root", type=Path)
    result.add_argument("--input-root", type=Path)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--dataset", choices=DATASETS)
    result.add_argument("--folds", type=int, default=5)
    result.add_argument("--model-seeds", type=parse_seeds, default=(0, 1, 2, 3, 4))
    result.add_argument("--task-id", type=int)
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--n-estimators", type=int, default=300)
    result.add_argument("--max-depth", type=int, default=4)
    result.add_argument("--learning-rate", type=float, default=0.03)
    result.add_argument("--colsample-bytree", type=float, default=1.0)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.mode == "materialize":
        if args.data_root is None or args.input_root is None or args.dataset is None:
            raise ValueError(
                "materialize requires --data-root, --input-root, and --dataset"
            )
        materialize(args)
    elif args.mode == "fit":
        if args.input_root is None or args.dataset is None or args.task_id is None:
            raise ValueError("fit requires --input-root, --dataset, and --task-id")
        fit_task(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()

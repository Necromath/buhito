#!/usr/bin/env python3
"""Label-aware TU preservation experiment using prepared fold representations.

This reuses the immutable fold-local graph bundles produced by
``tu_preservation_cv.py``. It does not refit dictionaries. For every graph it
computes a fixed-width representation containing structural statistics,
training-vocabulary node/edge-label histograms, and motif-token counts, then
fits an XGBoost classifier independently for every contraction condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
from typing import Any, Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from xgboost import XGBClassifier


DATASETS = ("MUTAG", "PTC_MR", "NCI1", "ENZYMES")
NODE_LABEL = "__buhito_mdl_node_label__"
EDGE_LABEL = "__buhito_mdl_edge_label__"
NODE_KIND = "__buhito_mdl_kind__"


def token(value: Any) -> str:
    """Stable text token for nested categorical labels."""
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def is_motif_node(data: dict[str, Any]) -> tuple[bool, int]:
    label = data.get(NODE_LABEL, data.get("mdl_label"))
    kind = data.get(NODE_KIND, data.get("mdl_kind"))
    motif = kind == "motif" or (
        isinstance(label, tuple) and len(label) > 0 and label[0] == "motif"
    )
    rank = -1
    if motif and isinstance(label, tuple) and len(label) > 1:
        try:
            rank = int(label[1])
        except (TypeError, ValueError):
            rank = -1
    return motif, rank


def build_vocabulary(graphs: Iterable[nx.Graph]) -> dict[str, list[str]]:
    node_labels: set[str] = set()
    edge_labels: set[str] = set()
    for graph in graphs:
        for _, data in graph.nodes(data=True):
            motif, _ = is_motif_node(data)
            if not motif:
                node_labels.add(token(data.get(NODE_LABEL, "<missing>")))
        for _, _, data in graph.edges(data=True):
            edge_labels.add(token(data.get(EDGE_LABEL, "<missing>")))
    return {
        "node_labels": sorted(node_labels),
        "edge_labels": sorted(edge_labels),
    }


def graph_features(
    graph: nx.Graph,
    vocabulary: dict[str, list[str]],
    *,
    maximum_rule_count: int = 5,
) -> tuple[np.ndarray, list[str]]:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degree = np.asarray([value for _, value in graph.degree()], dtype=float)
    degree_mean = float(degree.mean()) if degree.size else 0.0
    degree_std = float(degree.std()) if degree.size else 0.0
    degree_max = float(degree.max()) if degree.size else 0.0
    components = nx.number_connected_components(graph) if n else 0
    density = nx.density(graph) if n > 1 else 0.0
    clustering = nx.average_clustering(graph) if n > 1 else 0.0
    transitivity = nx.transitivity(graph) if n > 2 else 0.0
    triangles = sum(nx.triangles(graph).values()) / 3.0 if n else 0.0

    values = [
        float(n), float(m), density, degree_mean, degree_std, degree_max,
        float(components), clustering, transitivity, float(triangles),
    ]
    names = [
        "nodes", "edges", "density", "degree_mean", "degree_std",
        "degree_max", "components", "average_clustering", "transitivity",
        "triangles",
    ]

    node_counts = {name: 0 for name in vocabulary["node_labels"]}
    unknown_nodes = 0
    motif_counts = [0] * maximum_rule_count
    unknown_motifs = 0
    for _, data in graph.nodes(data=True):
        motif, rank = is_motif_node(data)
        if motif:
            if 0 <= rank < maximum_rule_count:
                motif_counts[rank] += 1
            else:
                unknown_motifs += 1
            continue
        value = token(data.get(NODE_LABEL, "<missing>"))
        if value in node_counts:
            node_counts[value] += 1
        else:
            unknown_nodes += 1

    for label in vocabulary["node_labels"]:
        values.append(float(node_counts[label]))
        names.append(f"node_label::{label}")
    values.append(float(unknown_nodes))
    names.append("node_label::<unknown>")
    for rank, count in enumerate(motif_counts):
        values.append(float(count))
        names.append(f"motif_rank::{rank}")
    values.append(float(unknown_motifs))
    names.append("motif_rank::<unknown>")

    edge_counts = {name: 0 for name in vocabulary["edge_labels"]}
    unknown_edges = 0
    for _, _, data in graph.edges(data=True):
        value = token(data.get(EDGE_LABEL, "<missing>"))
        if value in edge_counts:
            edge_counts[value] += 1
        else:
            unknown_edges += 1
    for label in vocabulary["edge_labels"]:
        values.append(float(edge_counts[label]))
        names.append(f"edge_label::{label}")
    values.append(float(unknown_edges))
    names.append("edge_label::<unknown>")

    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite graph feature encountered.")
    return result, names


def featurize(
    graphs: list[nx.Graph], vocabulary: dict[str, list[str]]
) -> tuple[np.ndarray, list[str]]:
    rows = [graph_features(graph, vocabulary) for graph in graphs]
    names = rows[0][1]
    if any(row_names != names for _, row_names in rows):
        raise RuntimeError("Feature columns changed within one fold.")
    return np.stack([values for values, _ in rows]), names


def run_task(args: argparse.Namespace) -> None:
    prepared = args.input_root / "prepared" / args.dataset
    manifest = pd.read_csv(prepared / "task_manifest.csv")
    row = manifest.iloc[args.task_id]
    bundle_path = Path(row["bundle"])
    actual = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    if actual != str(row["fingerprint"]):
        raise RuntimeError("Prepared representation fingerprint mismatch.")
    with bundle_path.open("rb") as handle:
        bundle = pickle.load(handle)

    seed = int(row["model_seed"])
    original_train = bundle["conditions"]["original"]["train_graphs"]
    vocabulary = build_vocabulary(original_train)
    conditions = list(bundle["conditions"])
    random.Random(seed).shuffle(conditions)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for order, condition_name in enumerate(conditions):
        condition = bundle["conditions"][condition_name]
        train_x, feature_names = featurize(condition["train_graphs"], vocabulary)
        test_x, test_names = featurize(condition["test_graphs"], vocabulary)
        if feature_names != test_names:
            raise RuntimeError("Train/test feature columns differ.")
        train_y = np.asarray(bundle["train_labels"], dtype=int)
        test_y = np.asarray(bundle["test_labels"], dtype=int)
        model = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            min_child_weight=1.0,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective=("binary:logistic" if bundle["num_classes"] == 2
                       else "multi:softprob"),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=args.threads,
        )
        model.fit(train_x, train_y)
        probabilities = model.predict_proba(test_x)
        predictions = probabilities.argmax(axis=1)
        metric_rows.append({
            "dataset": bundle["dataset"], "fold": bundle["fold"],
            "split_seed": bundle["split_seed"], "model_seed": seed,
            "condition": condition_name,
            "rule_count": int(condition["rule_count"]), "order": order,
            "accuracy": accuracy_score(test_y, predictions),
            "macro_f1": f1_score(test_y, predictions, average="macro"),
            "log_loss": log_loss(
                test_y, probabilities, labels=list(range(bundle["num_classes"]))
            ),
            "feature_count": len(feature_names),
        })
        for local_index, graph_id in enumerate(bundle["test_indices"]):
            size = condition["test_sizes"][local_index]
            values = probabilities[local_index].tolist()
            prediction_rows.append({
                "dataset": bundle["dataset"], "fold": bundle["fold"],
                "split_seed": bundle["split_seed"], "model_seed": seed,
                "graph_id": graph_id, "condition": condition_name,
                "rule_count": int(condition["rule_count"]),
                "true_label": int(test_y[local_index]),
                "predicted_label": int(predictions[local_index]),
                "probabilities": json.dumps(values),
                **{f"probability_{i}": value for i, value in enumerate(values)},
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
        "dataset": bundle["dataset"], "fold": bundle["fold"], "model_seed": seed,
        "prepared_fingerprint": actual, "vocabulary": vocabulary,
        "feature_names": feature_names,
        "estimator": "xgboost.XGBClassifier",
        "parameters": {
            "n_estimators": args.n_estimators, "max_depth": args.max_depth,
            "learning_rate": args.learning_rate, "threads": args.threads,
        },
    }
    (destination / f"{stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Completed {args.dataset} {stem}: {len(conditions)} conditions")


def aggregate(args: argparse.Namespace) -> None:
    metric_files = sorted(args.output_root.glob("task_results/**/*_metrics.csv"))
    prediction_files = sorted(args.output_root.glob("task_results/**/*_predictions.csv"))
    if len(metric_files) != 100 or len(prediction_files) != 100:
        raise RuntimeError(
            f"Expected 100 metric and prediction files; found "
            f"{len(metric_files)} and {len(prediction_files)}."
        )
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    predictions = pd.concat(
        [pd.read_csv(path) for path in prediction_files], ignore_index=True
    )
    original = metrics[metrics.condition == "original"].rename(columns={
        "accuracy": "original_accuracy", "macro_f1": "original_macro_f1",
        "log_loss": "original_log_loss",
    })[["dataset", "fold", "model_seed", "original_accuracy",
        "original_macro_f1", "original_log_loss"]]
    paired = metrics.merge(original, on=["dataset", "fold", "model_seed"])
    paired["accuracy_delta"] = paired.accuracy - paired.original_accuracy
    paired["macro_f1_delta"] = paired.macro_f1 - paired.original_macro_f1
    paired["log_loss_delta"] = paired.log_loss - paired.original_log_loss
    sizes = predictions.groupby(
        ["dataset", "fold", "model_seed", "condition", "rule_count"], as_index=False
    ).agg(raw_nodes=("raw_nodes", "sum"), model_nodes=("model_nodes", "sum"),
          raw_edges=("raw_edges", "sum"), model_edges=("model_edges", "sum"))
    sizes["node_reduction_fraction"] = 1 - sizes.model_nodes / sizes.raw_nodes
    sizes["edge_reduction_fraction"] = 1 - sizes.model_edges / sizes.raw_edges
    paired = paired.merge(
        sizes, on=["dataset", "fold", "model_seed", "condition", "rule_count"]
    )
    fold_means = paired.groupby(
        ["dataset", "condition", "rule_count", "fold"], as_index=False
    ).agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
          accuracy_delta=("accuracy_delta", "mean"),
          macro_f1_delta=("macro_f1_delta", "mean"),
          log_loss_delta=("log_loss_delta", "mean"),
          node_reduction_fraction=("node_reduction_fraction", "mean"),
          edge_reduction_fraction=("edge_reduction_fraction", "mean"))
    summary = fold_means.groupby(
        ["dataset", "condition", "rule_count"], as_index=False
    ).agg(folds=("fold", "size"), accuracy=("accuracy", "mean"),
          macro_f1=("macro_f1", "mean"),
          accuracy_delta=("accuracy_delta", "mean"),
          accuracy_delta_se=("accuracy_delta", "sem"),
          macro_f1_delta=("macro_f1_delta", "mean"),
          macro_f1_delta_se=("macro_f1_delta", "sem"),
          log_loss_delta=("log_loss_delta", "mean"),
          node_reduction_fraction=("node_reduction_fraction", "mean"),
          edge_reduction_fraction=("edge_reduction_fraction", "mean"))

    destination = args.output_root / "aggregated"
    destination.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(destination / "xgb_model_runs.csv", index=False)
    predictions.to_csv(destination / "xgb_graph_predictions.csv", index=False)
    paired.to_csv(destination / "xgb_paired_effects.csv", index=False)
    fold_means.to_csv(destination / "xgb_fold_effects.csv", index=False)
    summary.to_csv(destination / "xgb_preservation_summary.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    for axis, dataset in zip(axes.flat, DATASETS, strict=True):
        frame = summary[
            (summary.dataset == dataset) &
            (~summary.condition.isin(["original", "strict_mdl"]))
        ].sort_values("rule_count")
        axis.errorbar(
            frame.node_reduction_fraction, frame.accuracy_delta,
            yerr=1.96 * frame.accuracy_delta_se,
            marker="o", linewidth=1.6, capsize=3,
        )
        strict = summary[
            (summary.dataset == dataset) & (summary.condition == "strict_mdl")
        ]
        axis.scatter(
            strict.node_reduction_fraction, strict.accuracy_delta,
            marker="*", s=110, color="#d1495b", label="strict MDL", zorder=4,
        )
        axis.axhline(0, color="0.5", linestyle="--", linewidth=1)
        axis.set_title(dataset.replace("_", "-"))
        axis.set_xlabel("Node reduction fraction")
        axis.set_ylabel("Accuracy delta")
        axis.grid(alpha=0.18)
    figure.suptitle("Label-aware TU contraction–preservation map")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"xgb_tu_preservation_map.{suffix}", dpi=300)
    plt.close(figure)
    print(f"Aggregated label-aware results to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("task", "aggregate"))
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    args = parser.parse_args()
    if args.mode == "task":
        if args.input_root is None or args.dataset is None or args.task_id is None:
            raise ValueError("task requires --input-root, --dataset, and --task-id")
        run_task(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()

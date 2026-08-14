"""Private isolated worker for the optional PyTorch structural GCN benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
import resource
import sys
import time
from typing import Any


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0**2) if sys.platform == "darwin" else value / 1024.0


def _load(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "The GNN benchmark requires optional PyTorch support. Install it "
            "with `python -m pip install -e '.[gnn]'`."
        ) from exc
    return torch


def _device(torch: Any, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_class(torch: Any):
    class StructuralGCN(torch.nn.Module):
        def __init__(
            self,
            feature_dim: int,
            hidden_channels: int,
            num_layers: int,
            num_classes: int,
        ) -> None:
            super().__init__()
            widths = [feature_dim] + [hidden_channels] * num_layers
            self.layers = torch.nn.ModuleList(
                torch.nn.Linear(widths[index], widths[index + 1])
                for index in range(num_layers)
            )
            self.head = torch.nn.Linear(hidden_channels, num_classes)

        @staticmethod
        def aggregate(x: Any, edge_index: Any) -> Any:
            source, target = edge_index
            degree = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            degree.index_add_(
                0,
                target,
                torch.ones(target.shape[0], dtype=x.dtype, device=x.device),
            )
            inverse = degree.clamp(min=1.0).pow(-0.5)
            weights = inverse[source] * inverse[target]
            messages = x[source] * weights.unsqueeze(1)
            output = torch.zeros_like(x)
            output.index_add_(0, target, messages)
            return output

        def forward(self, x: Any, edge_index: Any, batch: Any, graphs: int) -> Any:
            for layer in self.layers:
                x = layer(self.aggregate(x, edge_index))
                x = torch.relu(x)
            pooled = torch.zeros(
                (graphs, x.shape[1]), dtype=x.dtype, device=x.device
            )
            pooled.index_add_(0, batch, x)
            counts = torch.zeros(graphs, dtype=x.dtype, device=x.device)
            counts.index_add_(
                0,
                batch,
                torch.ones(batch.shape[0], dtype=x.dtype, device=x.device),
            )
            pooled = pooled / counts.clamp(min=1.0).unsqueeze(1)
            return self.head(pooled)

    return StructuralGCN


def _tensor_batches(
    torch: Any,
    batches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for batch in batches:
        result.append(
            {
                "x": torch.from_numpy(batch["x"]),
                "edge_index": torch.from_numpy(batch["edge_index"]),
                "batch": torch.from_numpy(batch["batch"]),
                "labels": torch.from_numpy(batch["labels"]),
                "graphs": int(batch["graphs"]),
                "nodes": int(batch["nodes"]),
                "edges": int(batch["edges"]),
                "message_edges": int(batch["message_edges"]),
            }
        )
    return result


def _move(batch: dict[str, Any], device: Any) -> tuple[Any, Any, Any, Any]:
    return (
        batch["x"].to(device, non_blocking=False),
        batch["edge_index"].to(device, non_blocking=False),
        batch["batch"].to(device, non_blocking=False),
        batch["labels"].to(device, non_blocking=False),
    )


def _build_model(torch: Any, payload: dict[str, Any], device: Any):
    config = payload["config"]
    torch.manual_seed(int(config["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seed"]))
    model_class = _model_class(torch)
    return model_class(
        int(payload["feature_dim"]),
        int(config["hidden_channels"]),
        int(config["num_layers"]),
        int(payload["num_classes"]),
    ).to(device)


def _stable_output_checksum(torch: Any, logits: Any) -> float:
    """Return a bounded finite checksum for diagnostic reproducibility.

    A raw float32 sum can overflow even when every logit is finite.  The
    checksum is not a model-quality metric, so use a float64 bounded transform
    after first rejecting genuinely non-finite model outputs.
    """

    detached = logits.detach()
    finite = torch.isfinite(detached)
    if not bool(finite.all().cpu()):
        nonfinite = int((~finite).sum().cpu())
        raise RuntimeError(
            "The reference GCN produced non-finite logits "
            f"({nonfinite} values). Check node features and model numerical "
            "stability before interpreting timing or quality results."
        )
    value = torch.tanh(detached.to(dtype=torch.float64)).sum()
    checksum = float(value.cpu())
    if not math.isfinite(checksum):
        raise RuntimeError(
            "The bounded GNN output checksum was unexpectedly non-finite."
        )
    return checksum


def _inference_pass(
    torch: Any,
    model: Any,
    batches: list[dict[str, Any]],
    device: Any,
) -> float:
    checksum = 0.0
    with torch.inference_mode():
        for batch in batches:
            x, edge_index, graph_index, _ = _move(batch, device)
            if not bool(torch.isfinite(x).all().cpu()):
                nonfinite = int((~torch.isfinite(x)).sum().cpu())
                raise RuntimeError(
                    "The prepared GNN batch contains non-finite node "
                    f"features ({nonfinite} values)."
                )
            logits = model(x, edge_index, graph_index, batch["graphs"])
            checksum += _stable_output_checksum(torch, logits)
    if not math.isfinite(checksum):
        raise RuntimeError(
            "The accumulated GNN output checksum was unexpectedly non-finite."
        )
    return checksum


def _training_pass(
    torch: Any,
    model: Any,
    optimizer: Any,
    batches: list[dict[str, Any]],
    device: Any,
) -> tuple[float, int, int]:
    model.train()
    loss_total = 0.0
    correct = 0
    total = 0
    for batch in batches:
        x, edge_index, graph_index, labels = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, edge_index, graph_index, batch["graphs"])
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        count = int(labels.numel())
        loss_total += float(loss.detach().cpu()) * count
        predictions = logits.detach().argmax(dim=1)
        correct += int((predictions == labels).sum().cpu())
        total += count
    return loss_total / max(total, 1), correct, total


def _evaluation_pass(
    torch: Any,
    model: Any,
    batches: list[dict[str, Any]],
    device: Any,
) -> tuple[float, int, int, list[int], list[int]]:
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0
    predictions_all: list[int] = []
    targets_all: list[int] = []
    with torch.inference_mode():
        for batch in batches:
            x, edge_index, graph_index, labels = _move(batch, device)
            logits = model(x, edge_index, graph_index, batch["graphs"])
            loss = torch.nn.functional.cross_entropy(logits, labels)
            predictions = logits.argmax(dim=1)
            count = int(labels.numel())
            loss_total += float(loss.detach().cpu()) * count
            correct += int((predictions == labels).sum().cpu())
            total += count
            predictions_all.extend(int(value) for value in predictions.cpu())
            targets_all.extend(int(value) for value in labels.cpu())
    return (
        loss_total / max(total, 1),
        correct,
        total,
        predictions_all,
        targets_all,
    )


def _macro_f1(
    predictions: list[int],
    targets: list[int],
    num_classes: int,
) -> float | None:
    if not targets:
        return None
    scores: list[float] = []
    for label in range(num_classes):
        true_positive = sum(
            predicted == label and target == label
            for predicted, target in zip(predictions, targets, strict=True)
        )
        false_positive = sum(
            predicted == label and target != label
            for predicted, target in zip(predictions, targets, strict=True)
        )
        false_negative = sum(
            predicted != label and target == label
            for predicted, target in zip(predictions, targets, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores)


def _classification_diagnostics(
    predictions: list[int],
    targets: list[int],
    num_classes: int,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Return a confusion matrix and per-class held-out diagnostics."""

    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for predicted, target in zip(predictions, targets, strict=True):
        matrix[int(target)][int(predicted)] += 1
    rows: list[dict[str, Any]] = []
    for label in range(num_classes):
        true_positive = matrix[label][label]
        support = sum(matrix[label])
        predicted_count = sum(matrix[actual][label] for actual in range(num_classes))
        false_positive = predicted_count - true_positive
        false_negative = support - true_positive
        precision = (
            true_positive / predicted_count if predicted_count else 0.0
        )
        recall = true_positive / support if support else 0.0
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else 2 * true_positive / denominator
        rows.append(
            {
                "class_index": label,
                "support": support,
                "predicted_count": predicted_count,
                "true_positive": true_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return matrix, rows


def _totals(batches: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "graphs": sum(batch["graphs"] for batch in batches),
        "nodes": sum(batch["nodes"] for batch in batches),
        "edges": sum(batch["edges"] for batch in batches),
        "message_edges": sum(batch["message_edges"] for batch in batches),
    }


def run(payload: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    config = payload["config"]
    torch.set_num_threads(int(config["threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    device = _device(torch, str(config["device"]))
    all_batches = _tensor_batches(torch, payload["batches"])
    train_batches = _tensor_batches(
        torch,
        payload.get("train_batches", payload["batches"]),
    )
    quality_batches = _tensor_batches(
        torch,
        payload.get("quality_batches", []),
    )
    timed_batches = train_batches if config["mode"] == "training" else all_batches
    totals = _totals(timed_batches)

    def fresh_model():
        return _build_model(torch, payload, device)

    for _ in range(int(config["warmup_steps"])):
        warm_model = fresh_model()
        warm_optimizer = None
        if config["mode"] == "training":
            warm_optimizer = torch.optim.Adam(
                warm_model.parameters(),
                lr=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
            )
            _training_pass(
                torch,
                warm_model,
                warm_optimizer,
                train_batches,
                device,
            )
        else:
            warm_model.eval()
            _inference_pass(torch, warm_model, all_batches, device)
        del warm_model
        if warm_optimizer is not None:
            del warm_optimizer
    _synchronize(torch, device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = fresh_model()
    final_train_loss: float | None = None
    final_train_accuracy: float | None = None
    quality_eval_loss: float | None = None
    quality_eval_accuracy: float | None = None
    quality_eval_macro_f1: float | None = None
    quality_evaluation_seconds: float | None = None
    quality_confusion_matrix: list[list[int]] | None = None
    quality_per_class_metrics: list[dict[str, Any]] | None = None
    quality_predictions: list[int] | None = None
    quality_targets: list[int] | None = None
    checksum = 0.0

    started = time.perf_counter()
    if config["mode"] == "training":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        train_correct = 0
        train_total = 0
        for _ in range(int(config["epochs"])):
            final_train_loss, train_correct, train_total = _training_pass(
                torch,
                model,
                optimizer,
                train_batches,
                device,
            )
        final_train_accuracy = train_correct / max(train_total, 1)
        passes = int(config["epochs"])
    else:
        model.eval()
        for _ in range(int(config["steps_per_repeat"])):
            checksum = _inference_pass(torch, model, all_batches, device)
        passes = int(config["steps_per_repeat"])
    _synchronize(torch, device)
    elapsed = time.perf_counter() - started

    quality_available = bool(
        config["mode"] == "training"
        and payload.get("quality_metrics_available", False)
        and quality_batches
    )
    if quality_available:
        quality_started = time.perf_counter()
        (
            quality_eval_loss,
            quality_correct,
            quality_total,
            quality_predictions,
            quality_targets,
        ) = _evaluation_pass(torch, model, quality_batches, device)
        _synchronize(torch, device)
        quality_evaluation_seconds = time.perf_counter() - quality_started
        quality_eval_accuracy = quality_correct / max(quality_total, 1)
        quality_eval_macro_f1 = _macro_f1(
            quality_predictions,
            quality_targets,
            int(payload["num_classes"]),
        )
        quality_confusion_matrix, quality_per_class_metrics = (
            _classification_diagnostics(
                quality_predictions,
                quality_targets,
                int(payload["num_classes"]),
            )
        )

    processed_graphs = totals["graphs"] * passes
    processed_nodes = totals["nodes"] * passes
    processed_edges = totals["edges"] * passes
    processed_message_edges = totals["message_edges"] * passes
    cuda_peak = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
        if device.type == "cuda"
        else 0.0
    )
    quality_reason = payload.get("quality_metrics_reason")
    if quality_available:
        quality_reason = "held-out evaluation completed"

    return {
        "gnn_mode": config["mode"],
        "workload_seconds": elapsed,
        "total_graphs": totals["graphs"],
        "total_nodes": totals["nodes"],
        "total_edges": totals["edges"],
        "total_message_edges": totals["message_edges"],
        "passes": passes,
        "graphs_processed": processed_graphs,
        "nodes_processed": processed_nodes,
        "edges_processed": processed_edges,
        "message_edges_processed": processed_message_edges,
        "graphs_per_second": processed_graphs / elapsed,
        "nodes_per_second": processed_nodes / elapsed,
        "edges_per_second": processed_edges / elapsed,
        "message_edges_per_second": processed_message_edges / elapsed,
        "peak_rss_mb": _peak_rss_mb(),
        "cuda_peak_memory_mb": cuda_peak,
        "device": str(device),
        "torch_version": torch.__version__,
        "final_loss": final_train_loss,
        "accuracy": quality_eval_accuracy,
        "final_train_loss": final_train_loss,
        "final_train_accuracy": final_train_accuracy,
        "quality_eval_loss": quality_eval_loss,
        "quality_eval_accuracy": quality_eval_accuracy,
        "quality_eval_macro_f1": quality_eval_macro_f1,
        "quality_eval_confusion_matrix": quality_confusion_matrix,
        "quality_eval_per_class_metrics": quality_per_class_metrics,
        "quality_eval_predictions": quality_predictions,
        "quality_eval_targets": quality_targets,
        "quality_evaluation_seconds": quality_evaluation_seconds,
        "quality_eval_graphs": sum(
            batch["graphs"] for batch in quality_batches
        ),
        "quality_metrics_available": quality_available,
        "quality_metrics_reason": quality_reason,
        "output_checksum": checksum,
        "labels_available": bool(payload["labels_available"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    payload = _load(args.payload)
    print(
        f"[buhito-gnn-worker] START representation={payload['representation']} "
        f"mode={payload['config']['mode']}",
        flush=True,
    )
    result = run(payload)
    args.result.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    print(
        f"[buhito-gnn-worker] END elapsed_seconds="
        f"{result['workload_seconds']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

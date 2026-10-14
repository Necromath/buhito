#!/usr/bin/env python3
"""Run one reproducible, laptop-sized TDC ADMET smoke test.

This script intentionally answers two questions without conflating them:

1. Can Buhito train an ordinary graphlet predictor on an official ADMET split?
2. Can the lossless compressor fit on training molecules, transform held-out
   molecules, and decode the selected representation exactly?

It does not yet claim that a particular compressed-graph featurization is the
right downstream representation. Motif identity and boundary-port features are
left for the experiment contract shared by the collaborators.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from buhito.compression import MDLGraphCompressor
from buhito.converters import smiles_to_nx
from buhito.featurizers.bfs_graphlet_featurizer import BFSGraphletFeaturizer
from buhito.transformers import GraphletTransformer

SMILES_COLUMN = "Drug"
TARGET_COLUMN = "Y"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="HIA_Hou")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tdc"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/admet_smoke/hia_hou"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--graphlet-size", type=int, default=3)
    parser.add_argument("--n-rules", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument(
        "--compression-train-limit",
        type=int,
        default=128,
        help="Training molecules used for the compression diagnostic; 0 means all.",
    )
    parser.add_argument(
        "--compression-test-limit",
        type=int,
        default=64,
        help="Test molecules used for the compression diagnostic; 0 means all.",
    )
    parser.add_argument(
        "--skip-compression",
        action="store_true",
        help="Run only the official-split graphlet baseline.",
    )
    return parser.parse_args()


def _load_benchmark(
    dataset: str, data_dir: Path
) -> tuple[Any, str, pd.DataFrame, pd.DataFrame]:
    try:
        from tdc.benchmark_group import admet_group
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "The ADMET demo requires PyTDC and RDKit. Install with "
            "`python -m pip install -e '.[admet]'`."
        ) from exc

    data_dir.mkdir(parents=True, exist_ok=True)
    group = admet_group(path=str(data_dir))
    benchmark = group.get(dataset)
    name = str(benchmark["name"])
    train_val = benchmark["train_val"].reset_index(drop=True)
    test = benchmark["test"].reset_index(drop=True)
    _validate_frame(train_val, split="train_val")
    _validate_frame(test, split="test")
    return group, name, train_val, test


def _validate_frame(frame: pd.DataFrame, *, split: str) -> None:
    missing = {SMILES_COLUMN, TARGET_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(f"TDC {split} split is missing columns: {sorted(missing)}")
    if frame[[SMILES_COLUMN, TARGET_COLUMN]].isna().any().any():
        raise ValueError(f"TDC {split} split contains missing SMILES or targets.")


def _graphs(frame: pd.DataFrame) -> list[nx.Graph]:
    graphs = []
    for index, smiles in enumerate(frame[SMILES_COLUMN].astype(str)):
        try:
            graph, _ = smiles_to_nx(smiles)
        except ValueError as exc:
            raise ValueError(f"Invalid SMILES at row {index}: {smiles!r}") from exc
        graphs.append(graph)
    return graphs


def _is_binary_target(values: np.ndarray) -> bool:
    unique = set(np.unique(values).tolist())
    return bool(unique) and unique <= {0, 1}


def _fit_graphlet_baseline(
    train_graphs: list[nx.Graph],
    test_graphs: list[nx.Graph],
    y_train: np.ndarray,
    *,
    graphlet_size: int,
    jobs: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    featurizer = BFSGraphletFeaturizer(
        max_len=graphlet_size,
        return_nodewise=False,
    )
    transformer = GraphletTransformer(
        featurizer=featurizer,
        return_float=True,
        n_jobs=jobs,
        verbose=1,
    )

    started = time.perf_counter()
    x_train = transformer.fit_transform(train_graphs)
    x_test = transformer.transform(test_graphs)
    feature_seconds = time.perf_counter() - started

    started = time.perf_counter()
    if _is_binary_target(y_train):
        task = "binary_classification"
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=seed,
            solver="liblinear",
        )
        model.fit(x_train, y_train.astype(int))
        predictions = model.predict_proba(x_test)[:, 1]
    else:
        task = "regression"
        model = Ridge(alpha=1.0)
        model.fit(x_train, y_train.astype(float))
        predictions = model.predict(x_test)
    model_seconds = time.perf_counter() - started

    return predictions, {
        "task": task,
        "graphlet_size": graphlet_size,
        "feature_count": int(transformer.n_bits_),
        "unseen_test_feature_count": int(transformer.n_unseen_ or 0),
        "feature_seconds": feature_seconds,
        "model_seconds": model_seconds,
        "model": type(model).__name__,
    }


def _limited(frame: pd.DataFrame, limit: int, *, seed: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(frame):
        return frame.reset_index(drop=True).copy()
    return frame.sample(n=limit, random_state=seed).reset_index(drop=True)


def _assert_exact(original: list[nx.Graph], decoded: list[nx.Graph]) -> None:
    node_match = nx.algorithms.isomorphism.categorical_node_match("atom_key", None)
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("bond_key", None)
    for index, (left, right) in enumerate(zip(original, decoded, strict=True)):
        if not nx.is_isomorphic(
            left,
            right,
            node_match=node_match,
            edge_match=edge_match,
        ):
            raise RuntimeError(f"Lossless decode failed for held-out graph {index}.")


def _compression_diagnostic(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    output_dir: Path,
    seed: int,
    train_limit: int,
    test_limit: int,
    n_rules: int,
    max_candidates: int,
) -> dict[str, Any]:
    sampled_train = _limited(train_frame, train_limit, seed=seed)
    sampled_test = _limited(test_frame, test_limit, seed=seed)
    train_graphs = _graphs(sampled_train)
    test_graphs = _graphs(sampled_test)

    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=n_rules,
        min_graph_support=2,
        min_occurrences=2,
        max_candidates=max_candidates,
        node_label_keys="atom_key",
        edge_label_keys="bond_key",
        cache_dir=output_dir / "cache",
        validate=True,
        progress=True,
    )

    started = time.perf_counter()
    compressor.fit(train_graphs)
    fit_seconds = time.perf_counter() - started
    train_result = compressor.training_result_
    assert train_result is not None

    started = time.perf_counter()
    test_result = compressor.transform(test_graphs)
    transform_seconds = time.perf_counter() - started
    _assert_exact(test_graphs, test_result.decoded_graphs())

    compressor.candidate_frame().to_csv(output_dir / "candidates.csv", index=False)
    compressor.dictionary_frame().to_csv(output_dir / "dictionary.csv", index=False)
    train_result.per_graph.to_csv(output_dir / "compression_train.csv", index=False)
    test_result.per_graph.to_csv(output_dir / "compression_test.csv", index=False)

    return {
        "scope": "sampled diagnostic"
        if (
            len(sampled_train) < len(train_frame) or len(sampled_test) < len(test_frame)
        )
        else "full benchmark split",
        "fit_graphs": len(train_graphs),
        "test_graphs": len(test_graphs),
        "candidate_count": len(compressor.candidate_frame()),
        "selected_rule_count": len(compressor.rules_ or ()),
        "fit_seconds": fit_seconds,
        "transform_seconds": transform_seconds,
        "held_out_decode_failures": 0,
        "train_code": asdict(train_result.report),
        "test_code": asdict(test_result.report),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    args = parse_args()
    if args.jobs == 0:
        raise SystemExit("--jobs must be nonzero; use -1 for all cores.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading official TDC benchmark: {args.dataset}")
    group, name, train_val, test = _load_benchmark(args.dataset, args.data_dir)
    print(f"Loaded {len(train_val)} train/validation and {len(test)} test molecules")

    print("Converting the official split to labeled NetworkX molecular graphs")
    train_graphs = _graphs(train_val)
    test_graphs = _graphs(test)

    print("Fitting train-only Buhito graphlet vocabulary and linear baseline")
    predictions, baseline = _fit_graphlet_baseline(
        train_graphs,
        test_graphs,
        train_val[TARGET_COLUMN].to_numpy(),
        graphlet_size=args.graphlet_size,
        jobs=args.jobs,
        seed=args.seed,
    )
    official_score = group.evaluate({name: predictions})

    prediction_frame = test.copy()
    prediction_frame["prediction"] = predictions
    prediction_frame.to_csv(args.output_dir / "predictions.csv", index=False)

    compression = None
    if not args.skip_compression:
        print("Running train-only lossless compression diagnostic")
        compression = _compression_diagnostic(
            train_val,
            test,
            output_dir=args.output_dir,
            seed=args.seed,
            train_limit=args.compression_train_limit,
            test_limit=args.compression_test_limit,
            n_rules=args.n_rules,
            max_candidates=args.max_candidates,
        )

    summary = _json_ready(
        {
            "dataset_requested": args.dataset,
            "benchmark_name": name,
            "seed": args.seed,
            "train_val_rows": len(train_val),
            "test_rows": len(test),
            "baseline": baseline,
            "official_tdc_score": official_score,
            "compression": compression,
            "interpretation": (
                "Smoke test only: one seed and no hyperparameter tuning. The "
                "compression section is a codec/MDL diagnostic, not yet a "
                "compressed-feature predictive comparison."
            ),
        }
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("\nADMET smoke test complete")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

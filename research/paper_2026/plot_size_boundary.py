#!/usr/bin/env python3
"""Aggregate candidate cost/boundary diagnostics across TU datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ("MUTAG", "PTC_MR", "NCI1", "ENZYMES")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frames = []
    for dataset in DATASETS:
        frame = pd.read_csv(args.input_root / dataset / "candidates.csv")
        frame.insert(0, "dataset", dataset)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    selected = data["forced_n_occurrences"].clip(lower=1)
    data["boundary_bits_per_contraction"] = data["forced_boundary_bits"] / selected
    data["gross_bits_per_contraction"] = (
        data["forced_gross_rewrite_savings_bits"] / selected
    )
    data["overlap_retention"] = (
        data["forced_n_occurrences"] / data["total_occurrences"].clip(lower=1)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "size_boundary_candidates.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    markers = {2: "o", 3: "s", 4: "^"}
    for size, group in data.groupby("motif_nodes"):
        axes[0].scatter(
            group.boundary_bits_per_contraction,
            group.forced_net_savings_bits,
            s=22 + 35 * group.overlap_retention,
            alpha=0.62, marker=markers.get(int(size), "o"), label=f"size {int(size)}",
        )
        axes[1].scatter(
            group.gross_bits_per_contraction,
            group.boundary_bits_per_contraction,
            c=np.log1p(group.graph_support), cmap="viridis",
            s=22 + 35 * group.overlap_retention,
            alpha=0.62, marker=markers.get(int(size), "o"), label=f"size {int(size)}",
        )
    axes[0].axhline(0, color="0.4", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Boundary bits per selected contraction")
    axes[0].set_ylabel("Forced single-rule net savings (bits)")
    axes[0].set_title("A. Boundary burden versus profitability")
    axes[1].set_xlabel("Gross rewrite gain per selected contraction")
    axes[1].set_ylabel("Boundary bits per selected contraction")
    axes[1].set_title("B. Benefit–boundary tradeoff")
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(args.output_dir / f"size_boundary_diagnostic.{suffix}", dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()

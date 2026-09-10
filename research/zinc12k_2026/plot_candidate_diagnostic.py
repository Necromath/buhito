#!/usr/bin/env python3
"""Plot why frequent ZINC motif candidates fail the analytical MDL test.

This is a diagnostic for a fitted candidate table, not evidence that a motif
dictionary was selected.  It separates motif prevalence from the costs that
must be amortized by a forced nonempty single-rule rewrite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "rank",
    "graph_support",
    "forced_net_savings_bits",
    "forced_gross_rewrite_savings_bits",
    "forced_dictionary_bits",
    "forced_selector_bits",
    "forced_model_choice_bits",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_ranking", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--title",
        default="Frequent ZINC motifs need not improve the MDL code",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.candidate_ranking)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing candidate columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Candidate table is empty.")

    frame = frame.sort_values("rank").copy()
    top = frame.head(max(1, min(args.top, len(frame)))).copy()
    top["candidate"] = top["rank"].map(lambda value: f"M{int(value):03d}")
    top["selection_cost_bits"] = (
        top["forced_selector_bits"] + top["forced_model_choice_bits"]
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    scatter = axes[0].scatter(
        frame["graph_support"],
        frame["forced_net_savings_bits"],
        c=frame["forced_gross_rewrite_savings_bits"],
        cmap="coolwarm",
        s=34,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.4,
    )
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_xlabel("Training graphs containing candidate")
    axes[0].set_ylabel("Forced single-rule net MDL gain (bits)")
    axes[0].set_title("A. Frequency is insufficient")
    colorbar = figure.colorbar(scatter, ax=axes[0], pad=0.02)
    colorbar.set_label("Gross rewrite gain (bits)")

    best = frame.loc[frame["forced_net_savings_bits"].idxmax()]
    axes[0].annotate(
        f"best: M{int(best['rank']):03d}\n"
        f"support={int(best['graph_support']):,}\n"
        f"net={best['forced_net_savings_bits']:.1f} bits",
        xy=(best["graph_support"], best["forced_net_savings_bits"]),
        xytext=(0.04, 0.94),
        textcoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )

    positions = np.arange(len(top))
    width = 0.25
    axes[1].bar(
        positions - width,
        top["forced_gross_rewrite_savings_bits"],
        width,
        label="Gross rewrite gain",
    )
    axes[1].bar(
        positions,
        -top["forced_dictionary_bits"],
        width,
        label="Dictionary cost",
    )
    axes[1].bar(
        positions + width,
        -top["selection_cost_bits"],
        width,
        label="Selector + model cost",
    )
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_xticks(positions, top["candidate"], rotation=55, ha="right")
    axes[1].set_ylabel("Contribution (bits; costs shown negative)")
    axes[1].set_title(f"B. Top {len(top)} forced candidates")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")

    figure.suptitle(args.title, fontsize=13, fontweight="semibold")
    figure.text(
        0.5,
        0.01,
        "Strict MDL selected the empty dictionary; all points are diagnostic forced candidates.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot strict selection against the complete scored prefix path.

This is a diagnostic/provenance figure, not the downstream preservation map.
It makes the distinction between the strict eligible choice and the best
evaluated controlled prefix explicit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATASETS = ("MUTAG", "PTC_MR", "NCI1", "ENZYMES")


def build(root: Path, output: Path) -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        path = root / f"{dataset}_strict_mdl" / "dictionary_path.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "dataset", dataset)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)

    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "dictionary_screening_paths.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), sharex=True)
    for axis, dataset in zip(axes.flat, DATASETS, strict=True):
        frame = data[data["dataset"] == dataset].sort_values("n_rules")
        axis.axhline(0.0, color="0.55", linewidth=1, linestyle="--")
        axis.plot(
            frame["n_rules"], frame["net_savings_bits"],
            color="#365c8d", marker="o", linewidth=1.8,
            label="evaluated prefix",
        )
        selected = frame[frame["is_best"].astype(bool)]
        axis.scatter(
            selected["n_rules"], selected["net_savings_bits"],
            s=85, marker="*", color="#d1495b", zorder=4,
            label="strict selection",
        )
        ineligible = frame[~frame["eligible_for_selection"].astype(bool)]
        axis.scatter(
            ineligible["n_rules"], ineligible["net_savings_bits"],
            s=42, facecolors="none", edgecolors="#111111", zorder=3,
            label="controlled only",
        )
        axis.set_title(dataset.replace("_", "-"))
        axis.set_xlabel("Ranked-prefix rule count")
        axis.set_ylabel("Training-corpus net savings (bits)")
        axis.grid(alpha=0.18)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle(
        "Strict MDL selection versus evaluated ranked prefixes", y=0.985
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"dictionary_screening.{suffix}", dpi=300)
    plt.close(figure)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screening-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = build(args.screening_root, args.output_dir)
    print(f"Wrote {len(frame)} prefix rows to {args.output_dir}")


if __name__ == "__main__":
    main()

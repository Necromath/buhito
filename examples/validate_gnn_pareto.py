#!/usr/bin/env python3
"""Validate Buhito compression--speed--quality Pareto artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from buhito.benchmarks.pareto import validate_pareto_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--require-plots", action="store_true")
    args = parser.parse_args()
    summary = validate_pareto_artifacts(
        args.output_dir,
        require_plots=args.require_plots,
    )
    print("PASS: Pareto artifacts are complete and internally consistent.")
    print("Rule counts:", summary["rule_counts"])
    print("Best speed rule count:", summary["best_speed_rule_count"])
    print("Best median paired speedup:", summary["best_median_paired_speedup"])
    print("Recommended rule count:", summary["recommended_rule_count"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail CI when generated data or unexpectedly large files are tracked."""

from __future__ import annotations

import argparse
from pathlib import PurePosixPath
import subprocess
import sys

FORBIDDEN_PARTS = {
    ".ipynb_checkpoints",
    "__pycache__",
    "artifacts",
    "artifacts_syntax_treelets",
    "data",
    "regression_artifacts",
}
FORBIDDEN_SUFFIXES = {".joblib", ".pkl", ".pyc"}
FORBIDDEN_EXACT = {
    "summary.csv",
    "examples/qm9/qm9data/qm9_processed.csv",
}


def tracked_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"],
        text=False,
    )
    return [item.decode("utf-8") for item in output.split(b"\0") if item]


def blob_size(path: str) -> int:
    return int(
        subprocess.check_output(
            ["git", "cat-file", "-s", f":{path}"],
            text=True,
        ).strip()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=10 * 1024 * 1024,
        help="Maximum permitted tracked blob size (default: 10 MiB).",
    )
    args = parser.parse_args()

    problems: list[str] = []
    for path in tracked_files():
        pure = PurePosixPath(path)
        if path in FORBIDDEN_EXACT:
            problems.append(f"forbidden tracked file: {path}")
        if any(part.endswith(".egg-info") for part in pure.parts):
            problems.append(f"generated package metadata is tracked: {path}")
        if FORBIDDEN_PARTS.intersection(pure.parts):
            problems.append(f"generated/data directory is tracked: {path}")
        if pure.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"generated binary/cache is tracked: {path}")
        size = blob_size(path)
        if size > args.max_bytes:
            problems.append(
                f"tracked file exceeds {args.max_bytes} bytes: {path} ({size} bytes)"
            )

    if problems:
        print("Repository hygiene check failed:", file=sys.stderr)
        for problem in sorted(set(problems)):
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

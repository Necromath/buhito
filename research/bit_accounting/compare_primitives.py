#!/usr/bin/env python
"""Compare tracked MDL primitives with the legacy ZINC accountant.

The legacy file is parsed with AST so that its experiment-level imports are
never executed. This lets us compare the exact historical function bodies
without copying unrelated ZINC dependencies.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import buhito.mdl as core


ROOT = Path(__file__).resolve().parents[2]

LEGACY_PATH = (
    ROOT
    / "research"
    / "zinc12k_2026"
    / "run_zinc12k_bit_accounting_legacy.py"
)

WANTED = {
    "delta_pos",
    "delta_nonneg",
    "width",
    "subset_bits",
}


def load_legacy_primitives():
    source = LEGACY_PATH.read_text()
    tree = ast.parse(source, filename=str(LEGACY_PATH))

    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in WANTED
    ]

    found = {node.name for node in selected}
    missing = WANTED - found

    if missing:
        raise RuntimeError(
            f"Missing expected legacy functions: {sorted(missing)}"
        )

    module = ast.Module(
        body=selected,
        type_ignores=[],
    )
    ast.fix_missing_locations(module)

    namespace = {
        "math": math,
    }

    exec(
        compile(
            module,
            filename=str(LEGACY_PATH),
            mode="exec",
        ),
        namespace,
    )

    return namespace


def main() -> None:
    legacy = load_legacy_primitives()

    delta_pos = legacy["delta_pos"]
    delta_nonneg = legacy["delta_nonneg"]
    subset_bits = legacy["subset_bits"]
    width = legacy["width"]

    print("POSITIVE INTEGER CODE")
    print("=====================")
    print(
        f"{'n':>8s} "
        f"{'core_bits':>14s} "
        f"{'zinc_delta':>14s} "
        f"{'difference':>14s}"
    )

    for n in [
        1, 2, 3, 4, 5, 6, 7, 8,
        15, 16, 17, 31, 32, 33,
        100, 1000, 10000,
    ]:
        c = float(core._positive_integer_bits(n))
        z = float(delta_pos(n))

        print(
            f"{n:8d} "
            f"{c:14.6f} "
            f"{z:14.6f} "
            f"{z-c:14.6f}"
        )

    print()
    print("SUBSET / ENUMERATIVE CODE")
    print("=========================")
    print(
        f"{'N':>8s} "
        f"{'k':>8s} "
        f"{'core_log2C':>14s} "
        f"{'zinc_bits':>14s} "
        f"{'difference':>14s}"
    )

    for n, k in [
        (3, 1),
        (6, 3),
        (10, 2),
        (10, 5),
        (45, 10),
        (100, 5),
        (1000, 10),
    ]:
        c = float(core._log2_choose(n, k))
        z = float(subset_bits(n, k))

        print(
            f"{n:8d} "
            f"{k:8d} "
            f"{c:14.6f} "
            f"{z:14.6f} "
            f"{z-c:14.6f}"
        )

    print()
    print("LEGACY NONNEGATIVE INTEGER CODE")
    print("===============================")

    for x in range(10):
        print(
            f"x={x:2d} "
            f"delta_nonneg={delta_nonneg(x):2d}"
        )

    print()
    print("LEGACY SYMBOL WIDTH")
    print("===================")

    for cardinality in range(0, 10):
        print(
            f"A={cardinality:2d} "
            f"width={width(cardinality):2d}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Inventory Buhito's two graph bit-accounting conventions.

This intentionally does not assume the tracked analytical MDL accountant and
the later ZINC explicit enumerative codec are mathematically equivalent.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import buhito.mdl as mdl


ROOT = Path(__file__).resolve().parents[2]

LEGACY = (
    ROOT
    / "research"
    / "zinc12k_2026"
    / "run_zinc12k_bit_accounting_legacy.py"
)

CORE_NAMES = [
    "_positive_integer_bits",
    "_log2_choose",
    "_dirichlet_multinomial_bits",
    "_edgelist_topology_bits",
    "_label_bits",
    "_base_bits",
    "_boundary_bits",
    "_dictionary_bits",
]

LEGACY_NAMES = [
    "delta_pos",
    "delta_nonneg",
    "width",
    "subset_bits",
    "graph_bits",
    "encode_covered",
    "decode",
    "compressed_bits",
    "build_vocabs",
    "summarize",
]


def heading(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def core_inventory() -> None:
    heading("TRACKED CORE ACCOUNTANT")

    print("module:", Path(mdl.__file__).resolve())

    for name in CORE_NAMES:
        obj = getattr(mdl, name, None)

        print()
        print(f"[{name}]")

        if obj is None:
            print("NOT FOUND")
            continue

        try:
            print("signature:", inspect.signature(obj))
        except (TypeError, ValueError):
            print("signature: unavailable")

        doc = inspect.getdoc(obj)

        if doc:
            print("doc:")
            print(doc)


def legacy_inventory() -> None:
    heading("DARWIN ZINC EXPLICIT CODEC")

    print("source:", LEGACY)

    if not LEGACY.exists():
        raise SystemExit(f"Missing legacy accountant: {LEGACY}")

    source = LEGACY.read_text()
    tree = ast.parse(source)

    found = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for name in LEGACY_NAMES:
        print()
        print(f"[{name}]")

        node = found.get(name)

        if node is None:
            print("NOT FOUND")
            continue

        first_line = source.splitlines()[node.lineno - 1]
        print(first_line)


def checklist() -> None:
    heading("ACCOUNTING COMPARISON CHECKLIST")

    topics = [
        "integer code",
        "node-count code",
        "edge-count code",
        "topology code",
        "node-label code",
        "edge-label code",
        "dictionary code",
        "coverage-map code",
        "occurrence-count code",
        "rule-ID code",
        "attachment endpoint code",
        "attachment port code",
        "attachment edge-label code",
        "selector/model-choice code",
        "dictionary amortization",
        "decoder target",
        "coded graph properties",
        "node-identifier treatment",
    ]

    for topic in topics:
        print(f"[ ] {topic}")


def main() -> None:
    core_inventory()
    legacy_inventory()
    checklist()


if __name__ == "__main__":
    main()

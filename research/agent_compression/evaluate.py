"""Fixed, CPU-only development evaluator for agent-written packing priorities.

This is a synthetic ranking pilot, not a byte codec or a benchmark claim.
Core coding, contraction, decoding, and candidate discovery stay unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType

from candidates import CANDIDATES
from datasets import make_corpus, positive_control

from buhito.compression import (
    ExhaustiveGraphletEnumerator,
    MDLGraphCompressor,
    decode_rewrite,
    labeled_isomorphic,
)
from buhito.compression import substitution
from buhito.compression.models import _stable_token

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SEED = 20261016
DEV_SEED = 20261017


def candidate_hash(name):
    path = Path(name)
    if name in CANDIDATES:
        path = Path(__file__).with_name("candidates.py")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_priority(name):
    """Load a built-in name or an agent file exposing priority(features)."""
    if name in CANDIDATES:
        return CANDIDATES[name]
    path = Path(name).resolve()
    spec = importlib.util.spec_from_file_location("agent_priority", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.priority


def packing_selector(priority):
    """Keep the occurrence pool and disjoint-packing loop under evaluator control."""
    def select(graph, rules, occurrences_by_rule):
        rule_map = {rule.rank: rule for rule in rules}
        items = [
            (rule_map[rank], occurrence)
            for rank, occurrences in occurrences_by_rule.items()
            for occurrence in occurrences
        ]
        memberships = defaultdict(set)
        for index, (_, occurrence) in enumerate(items):
            for node in occurrence.nodes:
                memberships[node].add(index)
        ranked = []
        for index, (rule, occurrence) in enumerate(items):
            conflicts = set().union(*(memberships[n] for n in occurrence.nodes))
            features = MappingProxyType({
                "motif_nodes": rule.motif.number_of_nodes(),
                "motif_edges": rule.motif.number_of_edges(),
                "boundary_edges": substitution._external_degree(graph, occurrence),
                "conflicts": len(conflicts - {index}),
            })
            key = tuple(float(value) for value in priority(features))
            if not key or not all(math.isfinite(value) for value in key):
                raise ValueError("Priority must be a nonempty tuple of finite numbers")
            tie = (rule.rank, tuple(_stable_token(n) for n in occurrence.mapping))
            ranked.append((key, tie, rule.rank, occurrence))
        ranked.sort(key=lambda row: (row[0], row[1]))
        occupied = set()
        selected = []
        for _, _, rank, occurrence in ranked:
            if not (occurrence.nodes & occupied):
                selected.append((rank, occurrence))
                occupied.update(occurrence.nodes)
        return tuple(selected)
    return select


@contextmanager
def use_priority(priority):
    # Process-local adapter: never edit the checked-in compressor or decoder.
    original = substitution._select_nonoverlapping_multi
    substitution._select_nonoverlapping_multi = packing_selector(priority)
    try:
        yield
    finally:
        substitution._select_nonoverlapping_multi = original


def verify(result):
    """Independently decode proposed rewrites, including ones the selector rejects."""
    proposed = 0
    for record in result.records:
        if record.rewrite is not None:
            proposed += 1
            if not labeled_isomorphic(
                record.baseline_graph, decode_rewrite(record.rewrite)
            ):
                raise AssertionError("Public decoder failed labeled isomorphism")
    report = result.report
    graph_bits = sum(
        (record.template_bits + record.boundary_bits)
        if record.use_rewrite else record.baseline_bits
        for record in result.records
    )
    total = graph_bits + report.dictionary_bits + report.selector_bits
    total += report.model_choice_bits
    if not math.isclose(total, report.encoded_bits, abs_tol=1e-7):
        raise AssertionError("Codelength components do not sum to total")
    if not math.isclose(
        report.baseline_bits - report.encoded_bits,
        report.net_savings_bits,
        abs_tol=1e-7,
    ):
        raise AssertionError("Savings accounting does not reconcile")
    return {"proposed_rewrites_decoded": proposed, "decode_failures": 0}


def check_positive_control():
    train, evaluation = positive_control()
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,), n_rules=1, min_graph_support=1,
        min_occurrences=1, max_candidates=5,
        enumerator=ExhaustiveGraphletEnumerator(), validate=True,
    ).fit(train)
    result = compressor.transform(evaluation)
    checked = verify(result)
    if result.report.n_rewritten == 0 or result.report.net_savings_bits <= 0:
        raise AssertionError("Positive control did not actually compress")
    return {**checked, **asdict(result.report)}


def provenance():
    paths = sorted((ROOT / "src/buhito/compression").glob("*.py"))
    paths += [
        Path(__file__), Path(__file__).with_name("datasets.py"),
        Path(__file__).with_name("candidates.py"),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    return {"git_revision": revision, "evaluator_sha256": digest.hexdigest()}


def run(names, *, phase="dev", evaluation_seed=DEV_SEED, n_graphs=8):
    control = check_positive_control()
    train = make_corpus(TRAIN_SEED, n_graphs)
    evaluation = make_corpus(evaluation_seed, n_graphs)
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3, 4), n_rules=3, min_graph_support=1,
        min_occurrences=1, max_candidates=10,
        enumerator=ExhaustiveGraphletEnumerator(), validate=True,
    ).fit(train)
    # Fix the same training-derived dictionary for every candidate. A ranked
    # prefix exposes packing differences even when normal MDL chooses no rules.
    n_rules = min(3, len(compressor.candidate_frame()))
    if n_rules == 0:
        raise AssertionError("No motifs discovered; the pilot would be vacuous")
    default_result = compressor.transform(evaluation)
    default_checked = verify(default_result)
    results = []
    for name in names:
        started = time.perf_counter()
        row = {"candidate": name}
        try:
            row["candidate_sha256"] = candidate_hash(name)
            priority = load_priority(name)
            with use_priority(priority):
                compressor.selector = "sparse"
                sparse = compressor.transform_rule_prefix(evaluation, n_rules)
                sparse_checked = verify(sparse)
                compressor.selector = "all_eligible"
                forced = compressor.transform_rule_prefix(evaluation, n_rules)
                forced_checked = verify(forced)
            if forced_checked["proposed_rewrites_decoded"] == 0:
                raise AssertionError("No proposed rewrites; ranking was not exercised")
            row.update({
                "status": "ok",
                "forced_dictionary_diagnostic": {
                    **asdict(forced.report), **forced_checked,
                },
                "sparse_selector": {**asdict(sparse.report), **sparse_checked},
            })
        except Exception as error:
            row.update({"status": "invalid", "error": str(error)})
        row["elapsed_seconds"] = time.perf_counter() - started
        results.append(row)
    return {
        "schema_version": 1,
        "metric": "analytical MDL codelength estimate; not archive bytes",
        "scope": "synthetic occurrence-packing pilot; dictionary and coder frozen",
        "phase": phase, "train_seed": TRAIN_SEED,
        "evaluation_seed": evaluation_seed, "graphs_per_split": n_graphs,
        "dictionary_rule_keys": [r.key for r in compressor.rule_prefix(n_rules)],
        "training_selected_rules": len(compressor.rules_),
        "unmodified_pipeline": {
            **asdict(default_result.report), **default_checked,
        },
        "positive_control": control, **provenance(), "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", help="Name or Python file")
    parser.add_argument("--phase", choices=("dev", "test"), default="dev")
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--freeze-from", type=Path, help="Development result JSON")
    parser.add_argument("--graphs", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.graphs < 2:
        parser.error("--graphs must be at least 2")
    if args.phase == "test" and args.evaluation_seed is None:
        parser.error("Operator must supply a fresh --evaluation-seed for final test")
    seed = DEV_SEED if args.evaluation_seed is None else args.evaluation_seed
    if seed == TRAIN_SEED or (args.phase == "test" and seed == DEV_SEED):
        parser.error("Evaluation seed must differ from training and development")
    names = args.candidate or list(CANDIDATES)
    if args.phase == "test" and len(names) != 1:
        parser.error("Final test accepts exactly one previously frozen candidate")
    if args.phase == "test":
        if args.freeze_from is None:
            parser.error("Final test requires --freeze-from DEV_RESULT.json")
        previous = json.loads(args.freeze_from.read_text())
        if previous.get("phase") != "dev":
            parser.error("Freeze manifest must come from development")
        if args.graphs != previous["graphs_per_split"]:
            parser.error("Keep --graphs fixed to the development setting")
        if seed == previous["evaluation_seed"]:
            parser.error("Final seed must differ from the development seed")
        if previous["evaluator_sha256"] != provenance()["evaluator_sha256"]:
            parser.error("Evaluator or dataset source changed after development")
        frozen = next((row for row in previous["results"] if (
            row["candidate"] == names[0] and row["status"] == "ok"
        )), None)
        if frozen is None:
            parser.error("Selected candidate has no valid development result")
        if frozen["candidate_sha256"] != candidate_hash(names[0]):
            parser.error("Candidate source changed after development")
        # One frozen proposal plus the original packing policy, never a test sweep.
        names = list(dict.fromkeys(["baseline", names[0]]))
    payload = run(names, phase=args.phase, evaluation_seed=seed, n_graphs=args.graphs)
    rendered = json.dumps(payload, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    if any(row["status"] != "ok" for row in payload["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

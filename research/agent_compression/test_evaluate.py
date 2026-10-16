"""Bounded correctness checks; these never generate or consume final-test data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate as evaluator  # noqa: E402
from candidates import baseline  # noqa: E402

from buhito.compression import (  # noqa: E402
    ExhaustiveGraphletEnumerator,
    GraphSchema,
    MDLGraphCompressor,
)
from buhito.compression.models import MotifRule, _NODE_LABEL  # noqa: E402
from buhito.compression.recognition import _discover_candidates  # noqa: E402
from buhito.compression.substitution import (  # noqa: E402
    _select_nonoverlapping_multi,
)


@pytest.fixture
def packing_inputs():
    graph = GraphSchema.from_keys().normalize(
        nx.disjoint_union(nx.ladder_graph(3), nx.complete_graph(4))
    )
    raw = ExhaustiveGraphletEnumerator().enumerate(graph, (3, 4))
    candidates, found = _discover_candidates([graph], [raw])
    rules = tuple(
        MotifRule(key=key, base_key=c.base_key, motif=c.motif.copy(), rank=rank)
        for rank, (key, c) in enumerate(candidates.items())
    )
    occurrences = MDLGraphCompressor._occurrences_from_found(found, rules)[0]
    assert len(rules) > 1
    assert sum(map(len, occurrences.values())) > 2
    return graph, rules, occurrences


def test_baseline_adapter_matches_original_packing(packing_inputs):
    expected = _select_nonoverlapping_multi(*packing_inputs)
    actual = evaluator.packing_selector(baseline)(*packing_inputs)
    assert expected
    assert actual == expected
    occupied = set()
    for _, occurrence in actual:
        assert not occupied.intersection(occurrence.nodes)
        occupied.update(occurrence.nodes)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_priority_is_rejected(packing_inputs, score):
    with pytest.raises(ValueError, match="finite numbers"):
        evaluator.packing_selector(lambda features: (score,))(*packing_inputs)


def test_verify_checks_corrupt_rewrites_even_when_unselected():
    compressor = MDLGraphCompressor(
        graphlet_sizes=(3,),
        n_rules=1,
        min_graph_support=1,
        min_occurrences=1,
        max_candidates=2,
        min_rule_savings_bits=float("-inf"),
        dictionary_selection="fixed",
        enumerator=ExhaustiveGraphletEnumerator(),
        validate=True,
    ).fit([nx.path_graph(n) for n in (6, 7, 8)])
    result = compressor.transform_rule_prefix([nx.path_graph(7)], 1)
    record = result.records[0]
    assert record.rewrite is not None
    assert record.use_rewrite is False
    assert evaluator.verify(result)["proposed_rewrites_decoded"] == 1

    motif = record.rewrite.rules[0].motif
    motif.nodes[next(iter(motif))][_NODE_LABEL] = ("data", ("corrupt",))
    with pytest.raises(AssertionError, match="Public decoder failed"):
        evaluator.verify(result)


@pytest.fixture
def frozen_cli(tmp_path, monkeypatch):
    candidate = tmp_path / "proposal.py"
    candidate.write_text("def priority(features):\n    return (0,)\n")
    monkeypatch.setattr(
        evaluator,
        "provenance",
        lambda: {"git_revision": "test-only", "evaluator_sha256": "fixed-coder"},
    )
    manifest = tmp_path / "dev.json"
    manifest.write_text(json.dumps({
        "phase": "dev",
        "graphs_per_split": 2,
        "evaluation_seed": evaluator.DEV_SEED,
        "evaluator_sha256": "fixed-coder",
        "results": [{
            "candidate": str(candidate),
            "candidate_sha256": evaluator.candidate_hash(str(candidate)),
            "status": "ok",
        }],
    }))
    argv = [
        "evaluate.py", "--phase", "test", "--graphs", "2",
        "--evaluation-seed", "20261018", "--freeze-from", str(manifest),
        "--candidate", str(candidate),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return candidate, argv


def test_final_manifest_rejects_changed_candidate_before_evaluation(
    frozen_cli, monkeypatch, capsys,
):
    candidate, _ = frozen_cli
    candidate.write_text(candidate.read_text() + "# Changed after development.\n")

    def never_evaluate(*args, **kwargs):
        pytest.fail("Invalid frozen source must not reach final evaluation")

    monkeypatch.setattr(evaluator, "run", never_evaluate)
    with pytest.raises(SystemExit) as error:
        evaluator.main()
    assert error.value.code == 2
    assert "Candidate source changed after development" in capsys.readouterr().err


def test_final_manifest_adds_fixed_baseline_without_consuming_final_data(
    frozen_cli, monkeypatch, capsys,
):
    candidate, _ = frozen_cli
    calls = []

    def fake_run(names, **kwargs):
        calls.append((names, kwargs))
        return {"results": [{"candidate": name, "status": "ok"} for name in names]}

    def never_generate(*args, **kwargs):
        pytest.fail("This manifest check must not generate final-test data")

    monkeypatch.setattr(evaluator, "run", fake_run)
    monkeypatch.setattr(evaluator, "make_corpus", never_generate)
    evaluator.main()
    assert calls == [(
        ["baseline", str(candidate)],
        {"phase": "test", "evaluation_seed": 20261018, "n_graphs": 2},
    )]
    rendered = json.loads(capsys.readouterr().out)
    assert [row["candidate"] for row in rendered["results"]] == [
        "baseline", str(candidate),
    ]


def test_final_manifest_refuses_multiple_selected_candidates(
    frozen_cli, monkeypatch, capsys,
):
    _, argv = frozen_cli
    monkeypatch.setattr(sys, "argv", argv + ["--candidate", "largest_first"])

    def never_evaluate(*args, **kwargs):
        pytest.fail("A final-test proposal sweep must not run")

    monkeypatch.setattr(evaluator, "run", never_evaluate)
    with pytest.raises(SystemExit) as error:
        evaluator.main()
    assert error.value.code == 2
    assert "exactly one previously frozen candidate" in capsys.readouterr().err

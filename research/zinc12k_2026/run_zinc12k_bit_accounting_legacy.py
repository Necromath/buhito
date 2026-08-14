#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_zinc12k_tabular_regression import (
    edge_bond_label,
    is_motif_node,
    load_dataset,
    motif_rule_index,
    node_atom_label,
    parse_literal,
)

SPLITS = ("train", "val", "test")


def delta_pos(x: int) -> int:
    if x <= 0:
        raise ValueError(x)
    length = x.bit_length()
    return length + 2 * length.bit_length() - 2


def delta_nonneg(x: int) -> int:
    return delta_pos(x + 1)


def width(cardinality: int) -> int:
    return 0 if cardinality <= 1 else int(math.ceil(math.log2(cardinality)))


def subset_bits(universe: int, chosen: int) -> int:
    if chosen < 0 or chosen > universe:
        raise ValueError((universe, chosen))
    if chosen in (0, universe):
        return 0
    value = (
        math.lgamma(universe + 1)
        - math.lgamma(chosen + 1)
        - math.lgamma(universe - chosen + 1)
    ) / math.log(2.0)
    return int(math.ceil(value))


def atom(attrs: Mapping[str, Any]) -> str:
    if "__label__" in attrs:
        return str(attrs["__label__"])
    value = node_atom_label(attrs)
    return "_" if value is None else str(value)


def bond(attrs: Mapping[str, Any]) -> str:
    if "__label__" in attrs:
        return str(attrs["__label__"])
    value = edge_bond_label(attrs)
    return "_" if value is None else str(value)


def require_simple(graph: nx.Graph, name: str) -> None:
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise RuntimeError(f"{name} must be simple, undirected, and loop-free.")


def graph_bits(graph: nx.Graph, n_atom: int, n_bond: int) -> dict[str, int]:
    require_simple(graph, "graph")
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    out = {
        "header": delta_nonneg(n) + delta_nonneg(m),
        "topology": subset_bits(n * (n - 1) // 2, m),
        "node_labels": n * width(n_atom),
        "edge_labels": m * width(n_bond),
        "flag": 0,
        "occurrences": 0,
        "attachments": 0,
    }
    out["total"] = sum(out.values())
    return out


def parse_attrs(value: Any) -> dict[str, Any]:
    parsed = parse_literal(value)
    return dict(parsed)


def load_single_rule(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path).sort_values("rule_index")
    if len(frame) != 1 or int(frame.iloc[0]["rule_index"]) != 0:
        raise RuntimeError("Expected exactly one frozen ZINC rule with rule_index=0.")
    row = frame.iloc[0]
    graph = nx.Graph()
    for node, attrs in parse_literal(row["nodes_with_attributes"]):
        graph.add_node(node, **parse_attrs(attrs))
    for u, v, attrs in parse_literal(row["edges_with_attributes"]):
        graph.add_edge(u, v, **parse_attrs(attrs))
    require_simple(graph, "dictionary rule")
    return {
        "index": 0,
        "graph": graph,
        "ports": tuple(sorted(graph.nodes(), key=repr)),
    }


def candidates(graph: nx.Graph, rule: dict[str, Any]) -> list[tuple[Any, ...]]:
    rank = {node: i for i, node in enumerate(graph.nodes())}
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph,
        rule["graph"],
        node_match=lambda a, b: atom(a) == atom(b),
        edge_match=lambda a, b: bond(a) == bond(b),
    )
    best: dict[frozenset[Any], tuple[Any, ...]] = {}
    for mapping in matcher.subgraph_isomorphisms_iter():
        inverse = {rule_node: graph_node for graph_node, rule_node in mapping.items()}
        ordered = tuple(inverse[node] for node in rule["ports"])
        key = frozenset(ordered)
        if key not in best or tuple(rank[x] for x in ordered) < tuple(rank[x] for x in best[key]):
            best[key] = ordered
    out = list(best.values())
    out.sort(key=lambda occ: tuple(rank[x] for x in occ))
    return out


def quotient_graph(graph: nx.Graph, selected: Sequence[tuple[Any, ...]]) -> nx.Graph:
    membership = {}
    for occ_id, occurrence in enumerate(selected):
        for node in occurrence:
            if node in membership:
                raise RuntimeError("Overlapping selected occurrences.")
            membership[node] = occ_id

    out = nx.Graph()
    residual_nodes = [node for node in graph.nodes() if node not in membership]
    residual_id = {node: i for i, node in enumerate(residual_nodes)}

    for node in residual_nodes:
        out.add_node(("r", residual_id[node]), __label__=atom(graph.nodes[node]))
    for occ_id in range(len(selected)):
        out.add_node(("m", occ_id), __label__="motif:0")

    for u, v, attrs in graph.edges(data=True):
        u_occ = membership.get(u)
        v_occ = membership.get(v)
        if u_occ is not None and v_occ is not None and u_occ == v_occ:
            continue
        left = ("m", u_occ) if u_occ is not None else ("r", residual_id[u])
        right = ("m", v_occ) if v_occ is not None else ("r", residual_id[v])
        label = bond(attrs)
        if out.has_edge(left, right):
            if out.edges[left, right]["__label__"] != label:
                raise RuntimeError("Boundary edge merge with conflicting labels.")
        else:
            out.add_edge(left, right, __label__=label)
    return out


def saved_token_graph(graph: nx.Graph) -> nx.Graph:
    out = nx.Graph()
    for node, attrs in graph.nodes(data=True):
        if is_motif_node(node, attrs):
            label = f"motif:{motif_rule_index(node, attrs, 1)}"
        else:
            label = atom(attrs)
        out.add_node(node, __label__=label)
    for u, v, attrs in graph.edges(data=True):
        out.add_edge(u, v, __label__=bond(attrs))
    return out


def quotient_matches(original: nx.Graph, selected: Sequence[tuple[Any, ...]], token_graph: nx.Graph) -> bool:
    node_match = nx.algorithms.isomorphism.categorical_node_match("__label__", None)
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("__label__", None)
    return nx.is_isomorphic(
        quotient_graph(original, selected),
        saved_token_graph(token_graph),
        node_match=node_match,
        edge_match=edge_match,
    )


def choose_disjoint(options, target, original, token_graph):
    if target == 0:
        if not quotient_matches(original, [], token_graph):
            raise RuntimeError("Uncovered tokenized graph differs from original.")
        return []

    greedy = []
    used = set()
    for option in options:
        if set(option).isdisjoint(used):
            greedy.append(option)
            used.update(option)
            if len(greedy) == target and quotient_matches(original, greedy, token_graph):
                return greedy

    answer = None

    def search(i, picked, occupied):
        nonlocal answer
        if len(picked) == target:
            selected = [options[j] for j in picked]
            if quotient_matches(original, selected, token_graph):
                answer = list(picked)
                return True
            return False
        if i == len(options) or len(picked) + len(options) - i < target:
            return False
        nodes = set(options[i])
        if nodes.isdisjoint(occupied):
            picked.append(i)
            if search(i + 1, picked, occupied | nodes):
                return True
            picked.pop()
        return search(i + 1, picked, occupied)

    if not search(0, [], set()) or answer is None:
        raise RuntimeError(f"Could not reproduce quotient with {target} occurrences.")
    return [options[i] for i in answer]


def endpoint_key(endpoint):
    return (0, int(endpoint[1]), 0) if endpoint[0] == "r" else (1, int(endpoint[1]), int(endpoint[2]))


def encode_covered(graph, selected, rule, atom_index, bond_index):
    membership = {}
    for occ_id, occurrence in enumerate(selected):
        for port_id, node in enumerate(occurrence):
            if node in membership:
                raise RuntimeError("Overlapping selected occurrences.")
            membership[node] = (occ_id, port_id)

    residual_nodes = [node for node in graph.nodes() if node not in membership]
    residual_id = {node: i for i, node in enumerate(residual_nodes)}
    residual = nx.Graph()

    for node in residual_nodes:
        residual.add_node(residual_id[node], __label__=atom(graph.nodes[node]))

    attachments = []
    ports = rule["ports"]
    rule_graph = rule["graph"]

    for u, v, attrs in graph.edges(data=True):
        u_mem = membership.get(u)
        v_mem = membership.get(v)
        label = bond(attrs)

        if u_mem is None and v_mem is None:
            residual.add_edge(residual_id[u], residual_id[v], __label__=label)
            continue

        if u_mem is not None and v_mem is not None and u_mem[0] == v_mem[0]:
            ru, rv = ports[u_mem[1]], ports[v_mem[1]]
            if not rule_graph.has_edge(ru, rv) or bond(rule_graph.edges[ru, rv]) != label:
                raise RuntimeError("Selected occurrence does not match rule internally.")
            continue

        left = ("r", residual_id[u]) if u_mem is None else ("p", u_mem[0], u_mem[1])
        right = ("r", residual_id[v]) if v_mem is None else ("p", v_mem[0], v_mem[1])
        if endpoint_key(right) < endpoint_key(left):
            left, right = right, left
        attachments.append((left, right, label))

    attachments.sort(key=lambda row: (endpoint_key(row[0]), endpoint_key(row[1]), row[2]))

    residual_edges = []
    for u, v, attrs in residual.edges(data=True):
        if u > v:
            u, v = v, u
        residual_edges.append([u, v, bond_index[bond(attrs)]])
    residual_edges.sort()

    record = {
        "res": {
            "n": len(residual_nodes),
            "x": [atom_index[atom(graph.nodes[node])] for node in residual_nodes],
            "e": residual_edges,
        },
        "occ": [0] * len(selected),
        "att": [[list(a), list(b), bond_index[label]] for a, b, label in attachments],
    }
    return record, residual


def decode(record, rule, atoms, bonds):
    graph = nx.Graph()
    residual = record["res"]

    for i, label in enumerate(residual["x"]):
        graph.add_node(i, __label__=atoms[int(label)])

    next_id = len(residual["x"])
    port_nodes = {}

    for occ_id, _rule_id in enumerate(record["occ"]):
        for port_id, rule_node in enumerate(rule["ports"]):
            node_id = next_id
            next_id += 1
            port_nodes[(occ_id, port_id)] = node_id
            graph.add_node(node_id, __label__=atom(rule["graph"].nodes[rule_node]))

        for u, v, attrs in rule["graph"].edges(data=True):
            pu = rule["ports"].index(u)
            pv = rule["ports"].index(v)
            graph.add_edge(port_nodes[(occ_id, pu)], port_nodes[(occ_id, pv)], __label__=bond(attrs))

    for u, v, label in residual["e"]:
        graph.add_edge(int(u), int(v), __label__=bonds[int(label)])

    for left, right, label in record["att"]:
        def endpoint(obj):
            if obj[0] == "r":
                return int(obj[1])
            return port_nodes[(int(obj[1]), int(obj[2]))]

        graph.add_edge(endpoint(left), endpoint(right), __label__=bonds[int(label)])

    return graph


def labeled_copy(graph):
    out = nx.Graph()
    for node, attrs in graph.nodes(data=True):
        out.add_node(node, __label__=atom(attrs))
    for u, v, attrs in graph.edges(data=True):
        out.add_edge(u, v, __label__=bond(attrs))
    return out


def verify(original, reconstructed):
    node_match = nx.algorithms.isomorphism.categorical_node_match("__label__", None)
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("__label__", None)
    if not nx.is_isomorphic(labeled_copy(original), reconstructed, node_match=node_match, edge_match=edge_match):
        raise RuntimeError("Round-trip labeled-graph isomorphism failed.")


def compressed_bits(original, residual, record, n_atom, n_bond, ports_per_rule):
    residual_bits = graph_bits(residual, n_atom, n_bond)
    k = len(record["occ"])
    r = residual.number_of_nodes()
    endpoint_universe = r + k * ports_per_rule
    attachment_count = len(record["att"])

    out = {
        **{f"residual_{key}": value for key, value in residual_bits.items() if key != "total"},
        "flag": 1,
        "occurrences": delta_nonneg(k),
        "rule_ids": k * width(1),
        "attachment_count": delta_nonneg(attachment_count),
        "attachment_endpoints": attachment_count * subset_bits(endpoint_universe, 2),
        "attachment_bond_labels": attachment_count * width(n_bond),
    }

    out["attachments"] = (
        out["attachment_count"]
        + out["attachment_endpoints"]
        + out["attachment_bond_labels"]
    )
    out["total"] = (
        residual_bits["total"]
        + out["flag"]
        + out["occurrences"]
        + out["rule_ids"]
        + out["attachments"]
    )
    return out


def prefixed(d, prefix):
    return {f"{prefix}_{key}_bits": value for key, value in d.items()}


def build_vocabs(original, rule):
    train_atoms = {
        atom(attrs)
        for graph in original.graphs["train"]
        for _, attrs in graph.nodes(data=True)
    }
    train_bonds = {
        bond(attrs)
        for graph in original.graphs["train"]
        for _, _, attrs in graph.edges(data=True)
    }

    rule_atoms = {atom(attrs) for _, attrs in rule["graph"].nodes(data=True)}
    rule_bonds = {bond(attrs) for _, _, attrs in rule["graph"].edges(data=True)}

    atom_vocab = sorted(train_atoms | rule_atoms)
    bond_vocab = sorted(train_bonds | rule_bonds)

    atom_set = set(atom_vocab)
    bond_set = set(bond_vocab)

    for split in SPLITS:
        unseen_atoms = {
            atom(attrs)
            for graph in original.graphs[split]
            for _, attrs in graph.nodes(data=True)
            if atom(attrs) not in atom_set
        }
        unseen_bonds = {
            bond(attrs)
            for graph in original.graphs[split]
            for _, _, attrs in graph.edges(data=True)
            if bond(attrs) not in bond_set
        }
        if unseen_atoms or unseen_bonds:
            raise RuntimeError(
                f"{split} has labels outside the train-frozen vocabulary: "
                f"atoms={sorted(unseen_atoms)}, bonds={sorted(unseen_bonds)}"
            )

    return atom_vocab, bond_vocab


def summarize(per_graph, dictionary_bits):
    rows = []

    def one(scope, frame, charge_dictionary):
        original = int(frame["original_total_bits"].sum())
        payload = int(frame["compressed_total_bits"].sum())
        covered = frame[frame["covered"]]
        covered_original = int(covered["original_total_bits"].sum())
        covered_payload = int(covered["compressed_total_bits"].sum())
        coverage_map = subset_bits(len(frame), int(frame["covered"].sum()))
        dictionary = dictionary_bits if charge_dictionary else 0
        total = payload + coverage_map + dictionary
        saved = original - total
        return {
            "scope": scope,
            "n_graphs": int(len(frame)),
            "covered_graphs": int(frame["covered"].sum()),
            "motif_occurrences": int(frame["motif_occurrences"].sum()),
            "coverage_fraction": float(frame["covered"].mean()),
            "coverage_map_bits": int(coverage_map),
            "original_bits": original,
            "compressed_payload_bits": payload,
            "dictionary_bits_charged": int(dictionary),
            "compressed_total_bits": int(total),
            "net_bits_saved": int(saved),
            "net_savings_fraction": saved / original if original else math.nan,
            "compression_ratio_original_over_compressed": original / total if total else math.inf,
            "covered_original_bits": covered_original,
            "covered_payload_bits": covered_payload,
            "covered_payload_bits_saved": covered_original - covered_payload,
            "covered_payload_savings_fraction": (
                (covered_original - covered_payload) / covered_original if covered_original else math.nan
            ),
        }

    for split in SPLITS:
        part = per_graph[per_graph["split"] == split]
        rows.append(one(f"{split}_frozen_reuse", part, False))
        rows.append(one(f"{split}_standalone", part, True))

    rows.append(one("all_splits_dictionary_once", per_graph, True))
    return pd.DataFrame(rows)


def component_table(per_graph, summary, dictionary_bits):
    rows = []
    for split in SPLITS:
        frame = per_graph[per_graph["split"] == split]
        rows.append({
            "scope": split,
            "original_total": int(frame["original_total_bits"].sum()),
            "compressed_residual_graphs": int(
                frame["compressed_residual_header_bits"].sum()
                + frame["compressed_residual_topology_bits"].sum()
                + frame["compressed_residual_node_labels_bits"].sum()
                + frame["compressed_residual_edge_labels_bits"].sum()
            ),
            "compressed_flags": int(frame["compressed_flag_bits"].sum()),
            "compressed_occurrences": int(frame["compressed_occurrences_bits"].sum() + frame["compressed_rule_ids_bits"].sum()),
            "compressed_attachments": int(frame["compressed_attachments_bits"].sum()),
            "coverage_map": int(subset_bits(len(frame), int(frame["covered"].sum()))),
            "dictionary_if_charged": int(dictionary_bits),
        })
    return pd.DataFrame(rows)


def write_outputs(output, presentation, backup_presentation, per_graph, summary, components, dictionary_bits):
    per_graph.to_csv(output / "bit_accounting_per_graph.csv.gz", index=False, compression="gzip")
    summary.to_csv(output / "bit_accounting_summary.csv", index=False)
    components.to_csv(output / "bit_accounting_components.csv", index=False)

    test_reuse = summary[summary["scope"] == "test_frozen_reuse"].iloc[0]
    test_standalone = summary[summary["scope"] == "test_standalone"].iloc[0]
    all_row = summary[summary["scope"] == "all_splits_dictionary_once"].iloc[0]

    report = "# ZINC-12k bit-level compression accounting\n\n"
    report += "This is a full graph-code accountant, not a node/edge-count proxy and not guaranteed to match the legacy TU local-MDL delta exactly.\n\n"
    report += "| Scope | Graphs | Covered | Occurrences | Original bits | Compressed bits | Net saved | Savings |\n"
    report += "|---|---:|---:|---:|---:|---:|---:|---:|\n"

    for scope in ("train_standalone", "val_standalone", "test_standalone", "all_splits_dictionary_once"):
        row = summary[summary["scope"] == scope].iloc[0]
        report += (
            f"| {scope} | {int(row.n_graphs):,} | {int(row.covered_graphs):,} | "
            f"{int(row.motif_occurrences):,} | {int(row.original_bits):,} | "
            f"{int(row.compressed_total_bits):,} | {int(row.net_bits_saved):,} | "
            f"{100 * row.net_savings_fraction:.4f}% |\n"
        )

    report += (
        f"\nDictionary cost: **{dictionary_bits:,} bits**.\n\n"
        f"Test with pre-shared frozen dictionary: **{int(test_reuse.net_bits_saved):,} bits** "
        f"saved ({100 * test_reuse.net_savings_fraction:.4f}%).\n\n"
        f"Test with dictionary charged to test alone: **{int(test_standalone.net_bits_saved):,} bits** "
        f"saved ({100 * test_standalone.net_savings_fraction:.4f}%).\n\n"
        f"Covered test payload savings: **{100 * test_reuse.covered_payload_savings_fraction:.2f}%** before dictionary amortization.\n"
    )

    (output / "ZINC12K_BIT_ACCOUNTING.md").write_text(report)

    presentation.parent.mkdir(parents=True, exist_ok=True)
    presentation.write_text(f"""% Auto-generated by tools/run_zinc12k_bit_accounting.py
\\begin{{frame}}{{ZINC-12k: Bit-Level Compression}}
\\small
\\begin{{center}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Scope & Original & Compressed & Net saved & Savings \\\\
\\midrule
Test, frozen dict. & {int(test_reuse.original_bits):,} & {int(test_reuse.compressed_total_bits):,} & {int(test_reuse.net_bits_saved):,} & {100 * test_reuse.net_savings_fraction:.3f}\\% \\\\
Test, dict. charged & {int(test_standalone.original_bits):,} & {int(test_standalone.compressed_total_bits):,} & {int(test_standalone.net_bits_saved):,} & {100 * test_standalone.net_savings_fraction:.3f}\\% \\\\
All 12k, dict. once & {int(all_row.original_bits):,} & {int(all_row.compressed_total_bits):,} & {int(all_row.net_bits_saved):,} & {100 * all_row.net_savings_fraction:.3f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{center}}

\\vspace{{0.6em}}
\\begin{{columns}}[T,onlytextwidth]
\\column{{0.33\\textwidth}}
\\card{{bluecard}}{{\\centering\\textbf{{Coverage}}\\par\\vspace{{0.25em}}{int(test_reuse.covered_graphs)}/{int(test_reuse.n_graphs)} test graphs\\par {int(test_reuse.motif_occurrences)} occurrences}}
\\column{{0.33\\textwidth}}
\\card{{greencard}}{{\\centering\\textbf{{Covered payload}}\\par\\vspace{{0.25em}}{100 * test_reuse.covered_payload_savings_fraction:.2f}\\% saved before dictionary}}
\\column{{0.33\\textwidth}}
\\card{{orangecard}}{{\\centering\\textbf{{Bottleneck}}\\par\\vspace{{0.25em}}Global savings are limited by rare motif coverage.}}
\\end{{columns}}

\\vspace{{0.4em}}
\\muted{{\\footnotesize This is a full graph-code accountant: topology, labels, coverage map, rule occurrences, exact attachment ports, and dictionary cost.}}
\\end{{frame}}
""")

    backup_presentation.parent.mkdir(parents=True, exist_ok=True)
    backup_presentation.write_text("""% Auto-generated by tools/run_zinc12k_bit_accounting.py
\\begin{frame}{Backup: ZINC bit-accounting charge}
\\small
\\begin{itemize}
\\item Original graph: Elias-delta graph size, enumerative topology, fixed-width atom labels, fixed-width bond labels.
\\item Compressed graph: residual graph plus coverage map, rule count, rule IDs, exact attachment endpoints, attachment bond labels.
\\item Dictionary: encoded once using the same graph code.
\\item Targets and arbitrary node IDs are excluded from both sides.
\\item This is stricter whole-corpus accounting and should not be numerically merged with the older TU local-MDL cards unless the TU datasets are rerun through the same accountant.
\\end{itemize}
\\[
L(G)=L(n)+L(m)+\\left\\lceil\\log_2 {N \\choose m}\\right\\rceil
+n\\left\\lceil\\log_2 |\\mathcal A|\\right\\rceil
+m\\left\\lceil\\log_2 |\\mathcal B|\\right\\rceil .
\\]
\\end{frame}
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/vast/home/mmontes/data/ZINC12K_jsonl"))
    parser.add_argument("--frozen-dir", type=Path, default=Path("artifacts/darwin/zinc12k/frozen_train_dictionary"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/darwin/zinc12k/bit_accounting"))
    parser.add_argument("--presentation-path", type=Path, default=Path("presentation/zinc12k_bit_accounting_frame.tex"))
    parser.add_argument("--backup-presentation-path", type=Path, default=Path("presentation/zinc12k_bit_accounting_frame_backup.tex"))
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        if not args.force:
            raise SystemExit(f"Output exists: {args.output_dir}; use --force.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    rule = load_single_rule(args.frozen_dir / "rules_detailed.csv")

    original, tokenized = load_dataset(
        args.data_root,
        args.frozen_dir,
        {"train": args.limit_train, "val": args.limit_val, "test": args.limit_test},
    )

    atom_vocab, bond_vocab = build_vocabs(original, rule)
    atom_index = {label: i for i, label in enumerate(atom_vocab)}
    bond_index = {label: i for i, label in enumerate(bond_vocab)}

    dictionary = graph_bits(rule["graph"], len(atom_vocab), len(bond_vocab))
    dictionary_bits = delta_nonneg(1) + dictionary["total"]

    rows = []
    start = time.perf_counter()

    for split in SPLITS:
        print(f"Processing {split}...", flush=True)
        for row_index, (graph, expected, dataset_index, token_graph) in enumerate(zip(
            original.graphs[split],
            original.motif_occurrences[split],
            original.dataset_indices[split],
            tokenized.graphs[split],
        )):
            expected = int(expected)
            options = candidates(graph, rule) if expected else []
            selected = choose_disjoint(options, expected, graph, token_graph)

            record, residual = encode_covered(graph, selected, rule, atom_index, bond_index)
            reconstructed = decode(record, rule, atom_vocab, bond_vocab) if expected else labeled_copy(graph)
            verify(graph, reconstructed)

            removed_nodes = graph.number_of_nodes() - token_graph.number_of_nodes()
            removed_edges = graph.number_of_edges() - token_graph.number_of_edges()

            if removed_nodes != expected * (rule["graph"].number_of_nodes() - 1):
                raise RuntimeError(f"{split}[{row_index}] node reduction mismatch.")
            if removed_edges != expected * rule["graph"].number_of_edges():
                raise RuntimeError(f"{split}[{row_index}] edge reduction mismatch.")

            original_code = graph_bits(graph, len(atom_vocab), len(bond_vocab))
            compressed_code = compressed_bits(
                graph,
                residual,
                record,
                len(atom_vocab),
                len(bond_vocab),
                rule["graph"].number_of_nodes(),
            )

            row = {
                "split": split,
                "row_index": row_index,
                "dataset_index": int(dataset_index),
                "covered": bool(expected),
                "motif_occurrences": expected,
                "candidate_occurrences": len(options),
                "original_nodes": graph.number_of_nodes(),
                "original_edges": graph.number_of_edges(),
                "residual_nodes": residual.number_of_nodes(),
                "residual_edges": residual.number_of_edges(),
                "attachment_edges": len(record["att"]),
                **prefixed(original_code, "original"),
                **prefixed(compressed_code, "compressed"),
            }
            row["payload_bits_saved"] = row["original_total_bits"] - row["compressed_total_bits"]
            row["payload_savings_fraction"] = row["payload_bits_saved"] / row["original_total_bits"]
            rows.append(row)

    per_graph = pd.DataFrame(rows)
    summary = summarize(per_graph, dictionary_bits)
    components = component_table(per_graph, summary, dictionary_bits)

    write_outputs(
        args.output_dir,
        args.presentation_path,
        args.backup_presentation_path,
        per_graph,
        summary,
        components,
        dictionary_bits,
    )

    manifest = {
        "dataset": "ZINC-12k",
        "code_model": "Elias-delta headers + enumerative topology + fixed-width labels + exact motif ports",
        "vocabulary_policy": "train split plus frozen rule; validation/test must have no unseen labels",
        "atom_vocab": atom_vocab,
        "bond_vocab": bond_vocab,
        "atom_width_bits": width(len(atom_vocab)),
        "bond_width_bits": width(len(bond_vocab)),
        "dictionary_bits": dictionary_bits,
        "round_trip_graphs": len(per_graph),
        "reconstruction": "labeled-graph isomorphism; arbitrary node IDs not preserved",
        "targets_encoded": False,
        "comparison_note": "Not guaranteed numerically identical to legacy TU local-MDL bit deltas unless TU datasets are rerun with this exact accountant.",
        "elapsed_seconds": time.perf_counter() - start,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print()
    print("BIT ACCOUNTING SUMMARY")
    print(summary.to_string(index=False))
    print()
    print("BIT ACCOUNTING COMPONENTS")
    print(components.to_string(index=False))
    print()
    print(f"Dictionary bits: {dictionary_bits}")
    print(f"Round-trip validated graphs: {len(per_graph)}")
    print(f"Wrote: {args.output_dir}")
    print(f"Presentation frame: {args.presentation_path}")
    print(f"Backup frame: {args.backup_presentation_path}")


if __name__ == "__main__":
    main()

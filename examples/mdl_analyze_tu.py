"""Run the standard Buhito MDL analysis on a TU-format graph dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from buhito import MDLGraphCompressor
from buhito.analysis import AnalysisConfig, run_mdl_analysis, save_standard_plots
from buhito.datasets import load_tu_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the Buhito MDL compressor on a small dataset slice and write "
            "human-readable motif assets, MDL cost decompositions, "
            "uncertainty, and per-graph analysis tables."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/TUDataset"))
    parser.add_argument("--dataset", default="MUTAG")
    parser.add_argument("--node-label-mode", default="auto")
    parser.add_argument("--edge-label-mode", default="auto")
    parser.add_argument("--graphlet-sizes", default="3")
    parser.add_argument("--fit-size", type=int, default=100)
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-rules", type=int, default=5)
    parser.add_argument("--min-graph-support", type=int, default=3)
    parser.add_argument("--min-occurrences", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=25)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--selector", default="sparse")
    parser.add_argument("--dictionary-selection", default="best")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/mdl_cache"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--skip-decode-check", action="store_true")
    return parser.parse_args()


def _dataset_label_key(dataset, *, kind: str, mode: str):
    if mode == "none":
        return None

    declared = getattr(dataset, f"{kind}_label_key", None)
    if declared:
        return declared

    graphs = getattr(dataset, "graphs", [])
    if not graphs:
        return None

    if kind == "node":
        records = (attributes for _, attributes in graphs[0].nodes(data=True))
        preferred = ("tu_node_label", "node_label", "label")
    else:
        records = (attributes for _, _, attributes in graphs[0].edges(data=True))
        preferred = ("tu_edge_label", "edge_label", "label")

    keys = set()
    for attributes in records:
        keys.update(attributes)
    for key in preferred:
        if key in keys:
            return key
    return None




def _dataset_graph_labels(dataset):
    """Return graph-level labels across supported TU loader versions."""

    for name in ("labels", "graph_labels", "targets", "y"):
        value = getattr(dataset, name, None)
        if value is not None:
            try:
                if len(value) == len(dataset.graphs):
                    return value
            except TypeError:
                continue
    return None


def main() -> None:
    args = parse_args()
    graphlet_sizes = tuple(
        int(value.strip())
        for value in args.graphlet_sizes.split(",")
        if value.strip()
    )

    dataset = load_tu_dataset(
        args.data_root,
        args.dataset,
        node_label_mode=args.node_label_mode,
        edge_label_mode=args.edge_label_mode,
    )
    output_dir = args.output_dir or (
        Path("artifacts") / "mdl_analysis" / args.dataset.lower()
    )

    node_label_keys = _dataset_label_key(
        dataset,
        kind="node",
        mode=args.node_label_mode,
    )
    edge_label_keys = _dataset_label_key(
        dataset,
        kind="edge",
        mode=args.edge_label_mode,
    )

    compressor = MDLGraphCompressor(
        graphlet_sizes=graphlet_sizes,
        n_rules=args.n_rules,
        min_graph_support=args.min_graph_support,
        min_occurrences=args.min_occurrences,
        max_candidates=args.max_candidates,
        node_label_keys=node_label_keys,
        edge_label_keys=edge_label_keys,
        selector=args.selector,
        dictionary_selection=args.dictionary_selection,
        cache_dir=args.cache_dir,
        validate=True,
        progress=True,
    )
    config = AnalysisConfig(
        fit_size=args.fit_size,
        eval_size=args.eval_size,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap,
        decode_check=not args.skip_decode_check,
    )
    result = run_mdl_analysis(
        dataset.graphs,
        compressor,
        labels=_dataset_graph_labels(dataset),
        config=config,
        output_dir=output_dir,
    )
    if args.plots:
        plot_paths = save_standard_plots(result, output_dir)
        print(f"Plots: {len(plot_paths)} files")

    print(f"Artifacts: {output_dir}")
    print("\nTop motifs")
    display_columns = [
        column
        for column in (
            "rank",
            "motif_id",
            "topology_name",
            "label_pattern",
            "graph_support",
            "total_occurrences",
            "forced_dictionary_bits",
            "forced_template_bits",
            "forced_boundary_bits",
            "forced_net_savings_bits",
            "is_selected",
        )
        if column in result.motifs
    ]
    print(result.motifs[display_columns].head(20).to_string(index=False))
    print("\nSummary")
    print(result.summary)


if __name__ == "__main__":
    main()

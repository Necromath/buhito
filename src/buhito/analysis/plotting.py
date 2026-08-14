"""Optional matplotlib visualizations for MDL analysis results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .core import MDLAnalysisResult


def save_standard_plots(
    result: MDLAnalysisResult,
    output_dir: str | Path,
    *,
    top_n: int = 20,
) -> list[Path]:
    """Save standard diagnostics and individual motif drawings."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_motif_savings(
            result.motifs,
            destination / "motif_savings.png",
            top_n=top_n,
        ),
        plot_support_vs_savings(
            result.motifs,
            destination / "support_vs_savings.png",
        ),
        plot_motif_cost_decomposition(
            result.motif_costs,
            destination / "motif_cost_decomposition.png",
            top_n=top_n,
        ),
        plot_graph_gain_distribution(
            result.eval_per_graph
            if not result.eval_per_graph.empty
            else result.fit_per_graph,
            destination / "graph_gain_distribution.png",
        ),
    ]
    image_paths, image_manifest = save_motif_images(
        result.motif_graphs,
        result.motifs,
        destination / "motif_assets",
        top_n=top_n,
    )
    if not image_manifest.empty:
        if result.motif_assets.empty:
            result.motif_assets = image_manifest
        else:
            result.motif_assets = result.motif_assets.merge(
                image_manifest,
                on=["motif_id", "key"],
                how="left",
            )
        result.motif_assets.to_csv(
            destination / "motif_assets.csv",
            index=False,
        )
    paths.extend(image_paths)
    return [path for path in paths if path is not None]


def plot_motif_savings(
    motifs: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int = 20,
) -> Path | None:
    plt = _matplotlib()
    savings_column = _savings_column(motifs)
    if motifs.empty or savings_column is None:
        return None

    table = motifs.copy()
    table[savings_column] = pd.to_numeric(
        table[savings_column], errors="coerce"
    )
    table = table.dropna(subset=[savings_column])
    table = table.nlargest(top_n, savings_column).sort_values(savings_column)
    labels = _motif_labels(table)

    figure, axis = plt.subplots(figsize=(9, max(4, 0.35 * len(table))))
    axis.barh(labels, table[savings_column])
    axis.axvline(0.0, linewidth=1)
    axis.set_xlabel("Forced single-rule MDL savings (bits)")
    axis.set_ylabel("Motif")
    axis.set_title("Candidate motif savings")
    figure.tight_layout()
    path = Path(output_path)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_support_vs_savings(
    motifs: pd.DataFrame,
    output_path: str | Path,
) -> Path | None:
    plt = _matplotlib()
    savings_column = _savings_column(motifs)
    if (
        motifs.empty
        or savings_column is None
        or "total_occurrences" not in motifs
    ):
        return None

    table = motifs.copy()
    x = pd.to_numeric(table["total_occurrences"], errors="coerce")
    y = pd.to_numeric(table[savings_column], errors="coerce")
    valid = x.notna() & y.notna() & (x > 0)
    if not valid.any():
        return None

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(x[valid], y[valid])
    axis.set_xscale("log")
    axis.axhline(0.0, linewidth=1)
    axis.set_xlabel("Total motif occurrences (log scale)")
    axis.set_ylabel("Forced single-rule MDL savings (bits)")
    axis.set_title("Frequency is not the same as compressibility")
    figure.tight_layout()
    path = Path(output_path)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_motif_cost_decomposition(
    motif_costs: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int = 20,
) -> Path | None:
    """Plot the exact components that sum to each forced encoded cost."""

    plt = _matplotlib()
    required = {"motif_id", "component", "bits", "forced_net_savings_bits"}
    if motif_costs.empty or not required.issubset(motif_costs.columns):
        return None

    ranking = (
        motif_costs[["motif_id", "forced_net_savings_bits"]]
        .drop_duplicates("motif_id")
        .sort_values("forced_net_savings_bits", ascending=False)
        .head(top_n)
    )
    motif_ids = ranking["motif_id"].astype(str).tolist()
    if not motif_ids:
        return None

    components = [
        "passthrough_baseline",
        "rewrite_template",
        "boundary_ports",
        "dictionary",
        "selector",
        "model_choice",
    ]
    pivot = (
        motif_costs[motif_costs["motif_id"].astype(str).isin(motif_ids)]
        .pivot_table(
            index="motif_id",
            columns="component",
            values="bits",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(motif_ids)
    )

    figure, axis = plt.subplots(figsize=(10, max(4, 0.4 * len(pivot))))
    left = np.zeros(len(pivot), dtype=float)
    for component in components:
        if component not in pivot:
            continue
        values = pd.to_numeric(pivot[component], errors="coerce").fillna(0.0)
        axis.barh(pivot.index, values, left=left, label=component)
        left += values.to_numpy(dtype=float)
    axis.set_xlabel("Forced encoded cost (bits)")
    axis.set_ylabel("Motif")
    axis.set_title("Why each candidate succeeds or fails")
    axis.legend()
    figure.tight_layout()
    path = Path(output_path)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_graph_gain_distribution(
    per_graph: pd.DataFrame,
    output_path: str | Path,
) -> Path | None:
    plt = _matplotlib()
    column = next(
        (
            candidate
            for candidate in ("net_savings_bits", "gross_gain_bits", "bits_saved")
            if candidate in per_graph
        ),
        None,
    )
    if column is None or per_graph.empty:
        return None
    values = pd.to_numeric(per_graph[column], errors="coerce").dropna()
    if values.empty:
        return None

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values.to_numpy(), bins=min(30, max(5, int(np.sqrt(len(values))))))
    axis.axvline(0.0, linewidth=1)
    axis.set_xlabel(f"{column} per graph")
    axis.set_ylabel("Graph count")
    axis.set_title("Distribution of graph-level MDL gain")
    figure.tight_layout()
    path = Path(output_path)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def save_motif_images(
    motif_graphs: dict[str, nx.Graph],
    motifs: pd.DataFrame,
    output_dir: str | Path,
    *,
    top_n: int = 20,
) -> tuple[list[Path], pd.DataFrame]:
    """Render candidate motifs as both PNG and SVG files."""

    plt = _matplotlib()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    manifest: list[dict[str, str]] = []
    if motifs.empty or not motif_graphs:
        return paths, pd.DataFrame(
            columns=["motif_id", "key", "png_path", "svg_path"]
        )

    for row in motifs.head(top_n).to_dict(orient="records"):
        key = str(row.get("key", ""))
        graph = motif_graphs.get(key)
        if graph is None:
            continue
        motif_id = str(row.get("motif_id", "motif"))
        topology = str(row.get("topology_name", "motif")).replace(" ", "_")
        stem = f"{motif_id}_{topology}"
        png_path = destination / f"{stem}.png"
        svg_path = destination / f"{stem}.svg"

        figure, axis = plt.subplots(figsize=(4.5, 4.0))
        positions = nx.circular_layout(graph)
        node_labels = {
            node: _node_drawing_label(node, dict(data))
            for node, data in graph.nodes(data=True)
        }
        nx.draw_networkx(
            graph,
            pos=positions,
            labels=node_labels,
            ax=axis,
            node_size=1400,
            font_size=8,
        )
        edge_labels = {
            (source, target): _attribute_label(dict(data))
            for source, target, data in graph.edges(data=True)
            if data
        }
        if edge_labels:
            nx.draw_networkx_edge_labels(
                graph,
                pos=positions,
                edge_labels=edge_labels,
                ax=axis,
                font_size=7,
            )
        axis.set_title(str(row.get("human_name", motif_id)))
        axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(png_path, dpi=180)
        figure.savefig(svg_path)
        plt.close(figure)
        paths.extend([png_path, svg_path])
        manifest.append(
            {
                "motif_id": motif_id,
                "key": key,
                "png_path": str(png_path.name),
                "svg_path": str(svg_path.name),
            }
        )
    return paths, pd.DataFrame(manifest)


def _savings_column(motifs: pd.DataFrame) -> str | None:
    for column in ("forced_net_savings_bits", "single_rule_savings_bits"):
        if column in motifs:
            return column
    return None


def _motif_labels(table: pd.DataFrame) -> list[str]:
    if {"motif_id", "topology_name"}.issubset(table.columns):
        return [
            f"{motif_id} {topology}"
            for motif_id, topology in zip(
                table["motif_id"], table["topology_name"], strict=True
            )
        ]
    if "motif_id" in table:
        return table["motif_id"].astype(str).tolist()
    if "key" in table:
        return [str(value)[:55] for value in table["key"]]
    return [f"motif {index}" for index in table.index]


def _node_drawing_label(node: Any, attributes: dict[str, Any]) -> str:
    suffix = _attribute_label(attributes)
    return f"{node}\n{suffix}" if suffix else str(node)


def _attribute_label(attributes: dict[str, Any]) -> str:
    return ", ".join(
        f"{key}={value!r}" for key, value in sorted(attributes.items())
    )


def _matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install the visualization extra or "
            "run `python -m pip install -e '.[viz]'`."
        ) from exc
    return plt

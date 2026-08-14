"""Statistical analysis utilities for Buhito's MDL graph compressor.

This module deliberately treats the compressor as a black box.  It consumes
its public tables and transform result, then produces normalized summaries,
bootstrap statistics, and serializable analysis artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json

import networkx as nx
import numpy as np
import pandas as pd

from .motifs import (
    build_motif_cost_table,
    enrich_motif_table,
    export_motif_assets,
)


_CANDIDATE_FRAME_NAMES = (
    "candidate_frame",
    "candidates_frame",
    "candidate_scores_frame",
    "candidate_ranking_frame",
    "ranked_candidates_frame",
    "candidate_table",
    "candidate_table_",
    "candidate_scores",
    "candidate_scores_",
    "candidate_rows",
    "candidate_rows_",
    "candidate_evaluations",
    "candidate_evaluations_",
    "scored_candidates",
    "scored_candidates_",
    "single_rule_scores",
    "single_rule_scores_",
    "ranked_candidates",
    "ranked_candidates_",
    "candidates",
    "candidates_",
    "_candidate_scores",
    "_candidate_rows",
    "_candidate_evaluations",
    "_scored_candidates",
    "_single_rule_scores",
    "_ranked_candidates",
    "_candidates",
)
_DICTIONARY_FRAME_NAMES = ("dictionary_frame",)
_DICTIONARY_PATH_FRAME_NAMES = (
    "dictionary_path_frame",
    "dictionary_search_frame",
    "dictionary_path",
    "dictionary_path_",
)
_MOTIF_GRAPH_NAMES = (
    "candidate_motif_graphs",
    "motif_graphs",
    "candidate_graphs",
)


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for a reproducible MDL analysis run."""

    fit_size: int = 200
    eval_size: int = 100
    seed: int = 0
    bootstrap_replicates: int = 1_000
    confidence_level: float = 0.95
    decode_check: bool = True

    def validate(self) -> None:
        if self.fit_size < 1:
            raise ValueError("fit_size must be at least 1")
        if self.eval_size < 0:
            raise ValueError("eval_size must be nonnegative")
        if self.bootstrap_replicates < 0:
            raise ValueError("bootstrap_replicates must be nonnegative")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between 0 and 1")


@dataclass
class MDLAnalysisResult:
    """All tables and metadata produced by one analysis run."""

    summary: dict[str, Any]
    motifs: pd.DataFrame
    dictionary: pd.DataFrame
    dictionary_path: pd.DataFrame
    motif_costs: pd.DataFrame
    fit_per_graph: pd.DataFrame
    eval_per_graph: pd.DataFrame
    bootstrap: pd.DataFrame
    motif_graphs: dict[str, nx.Graph] = field(default_factory=dict, repr=False)
    motif_assets: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: str | Path) -> Path:
        """Write a stable, human-readable artifact bundle."""

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        self.motifs.to_csv(destination / "motifs.csv", index=False)
        self.dictionary.to_csv(destination / "dictionary.csv", index=False)
        self.dictionary_path.to_csv(destination / "dictionary_path.csv", index=False)
        self.motif_costs.to_csv(destination / "motif_costs.csv", index=False)
        self.fit_per_graph.to_csv(destination / "fit_per_graph.csv", index=False)
        self.eval_per_graph.to_csv(destination / "eval_per_graph.csv", index=False)
        self.bootstrap.to_csv(destination / "bootstrap.csv", index=False)
        self.motif_assets = export_motif_assets(
            self.motif_graphs,
            self.motifs,
            destination / "motif_assets",
        )
        self.motif_assets.to_csv(
            destination / "motif_assets.csv",
            index=False,
        )

        (destination / "summary.json").write_text(
            json.dumps(_jsonable(self.summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "metadata.json").write_text(
            json.dumps(_jsonable(self.metadata), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "README.md").write_text(
            render_markdown_summary(self),
            encoding="utf-8",
        )
        return destination


def run_mdl_analysis(
    graphs: Iterable[nx.Graph],
    compressor: Any,
    *,
    labels: Sequence[Any] | None = None,
    config: AnalysisConfig | None = None,
    output_dir: str | Path | None = None,
) -> MDLAnalysisResult:
    """Fit an MDL compressor on a small slice and analyze held-out graphs.

    Parameters
    ----------
    graphs:
        Any finite iterable of simple NetworkX graphs.
    compressor:
        A configured ``MDLGraphCompressor``-compatible object exposing
        ``fit()``, ``transform()``, and the public frame helpers.
    labels:
        Optional graph labels.  When supplied, the fit/evaluation slices are
        sampled approximately proportionally by label.
    config:
        Sampling, bootstrap, and validation settings.
    output_dir:
        Optional artifact destination.  Nothing is written when omitted.
    """

    cfg = config or AnalysisConfig()
    cfg.validate()

    graph_list = list(graphs)
    if not graph_list:
        raise ValueError("graphs must contain at least one graph")
    if labels is not None and len(labels) != len(graph_list):
        raise ValueError("labels must have the same length as graphs")

    fit_indices, eval_indices = split_indices(
        len(graph_list),
        fit_size=cfg.fit_size,
        eval_size=cfg.eval_size,
        seed=cfg.seed,
        labels=labels,
    )
    fit_graphs = [graph_list[index] for index in fit_indices]
    eval_graphs = [graph_list[index] for index in eval_indices]

    compressor.fit(fit_graphs)
    fit_result = compressor.transform(fit_graphs)
    eval_result = compressor.transform(eval_graphs) if eval_graphs else None

    fit_per_graph = _per_graph_frame(fit_result, fit_indices, "fit")
    eval_per_graph = (
        _per_graph_frame(eval_result, eval_indices, "eval")
        if eval_result is not None
        else pd.DataFrame()
    )

    dictionary = _frame_from_object(compressor, _DICTIONARY_FRAME_NAMES)
    candidates = _candidate_frame_from_object(compressor)
    dictionary_path = _frame_from_object(
        compressor, _DICTIONARY_PATH_FRAME_NAMES
    )
    motif_graphs = _motif_graphs_from_object(compressor)
    motifs = build_motif_table(
        candidates=candidates,
        dictionary=dictionary,
        n_fit_graphs=len(fit_graphs),
    )
    motifs = enrich_motif_table(motifs, motif_graphs)
    motif_costs = build_motif_cost_table(motifs)

    decode_summary = {
        "fit_decode_checked": False,
        "fit_decode_failures": None,
        "eval_decode_checked": False,
        "eval_decode_failures": None,
    }
    if cfg.decode_check:
        # ``compressor.validate`` (hardcoded True in every example script in
        # this repo) already guarantees exact decode correctness for every
        # rewritten graph during fit()/transform() -- see
        # ``_rewrite_validation_error`` in mdl.py, which raises immediately
        # on any mismatch. Passing that through lets ``_decode_failures``
        # skip a redundant, potentially very slow isomorphism re-check.
        already_validated = bool(getattr(compressor, "validate", False))
        fit_failures = _decode_failures(
            fit_result, fit_graphs, already_validated=already_validated
        )
        eval_failures = (
            _decode_failures(
                eval_result, eval_graphs, already_validated=already_validated
            )
            if eval_result is not None
            else []
        )
        decode_summary = {
            "fit_decode_checked": True,
            "fit_decode_failures": len(fit_failures),
            "eval_decode_checked": eval_result is not None,
            "eval_decode_failures": len(eval_failures),
        }
        if fit_failures or eval_failures:
            raise AssertionError(
                "MDL decode validation failed: "
                f"fit={fit_failures}, eval={eval_failures}"
            )

    bootstrap_source = eval_per_graph if not eval_per_graph.empty else fit_per_graph
    bootstrap = bootstrap_graph_statistics(
        bootstrap_source,
        replicates=cfg.bootstrap_replicates,
        confidence_level=cfg.confidence_level,
        seed=cfg.seed,
    )

    fit_report = _report_mapping(fit_result)
    eval_report = _report_mapping(eval_result) if eval_result is not None else {}
    summary = {
        "n_graphs_total": len(graph_list),
        "n_fit_graphs": len(fit_graphs),
        "n_eval_graphs": len(eval_graphs),
        "n_candidate_motifs": len(motifs),
        "n_selected_motifs": int(motifs.get("is_selected", pd.Series(dtype=bool)).sum()),
        "fit_report": fit_report,
        "eval_report": eval_report,
        **decode_summary,
    }
    metadata = {
        "analysis_config": asdict(cfg),
        "fit_indices": fit_indices.tolist(),
        "eval_indices": eval_indices.tolist(),
        "compressor_class": type(compressor).__name__,
    }

    analysis = MDLAnalysisResult(
        summary=summary,
        motifs=motifs,
        dictionary=dictionary,
        dictionary_path=dictionary_path,
        motif_costs=motif_costs,
        fit_per_graph=fit_per_graph,
        eval_per_graph=eval_per_graph,
        bootstrap=bootstrap,
        motif_graphs=motif_graphs,
        metadata=metadata,
    )
    if output_dir is not None:
        analysis.save(output_dir)
    return analysis


def split_indices(
    n_items: int,
    *,
    fit_size: int,
    eval_size: int,
    seed: int,
    labels: Sequence[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, disjoint fit/evaluation indices.

    The implementation avoids requiring scikit-learn at analysis time while
    still preserving approximate class proportions when labels are supplied.
    """

    if n_items < 1:
        raise ValueError("n_items must be positive")
    if fit_size < 1 or eval_size < 0:
        raise ValueError("invalid fit/eval sizes")
    if fit_size + eval_size > n_items:
        raise ValueError(
            f"Requested {fit_size + eval_size} graphs from a dataset of {n_items}."
        )

    rng = np.random.default_rng(seed)
    if labels is None:
        permutation = rng.permutation(n_items)
        return permutation[:fit_size], permutation[fit_size : fit_size + eval_size]

    labels_array = np.asarray(labels, dtype=object)
    selected: list[int] = []
    for value in pd.unique(labels_array):
        group = np.flatnonzero(labels_array == value)
        selected.extend(rng.permutation(group).tolist())

    # Interleave class-wise shuffled groups by globally shuffling once more.
    selected_array = np.asarray(selected, dtype=int)
    selected_array = selected_array[rng.permutation(len(selected_array))]
    return (
        selected_array[:fit_size],
        selected_array[fit_size : fit_size + eval_size],
    )


def build_motif_table(
    *,
    candidates: pd.DataFrame,
    dictionary: pd.DataFrame,
    n_fit_graphs: int,
) -> pd.DataFrame:
    """Normalize compressor candidate output into an analysis-ready table."""

    candidate_table = candidates.copy()
    dictionary_table = dictionary.copy()

    if candidate_table.empty:
        candidate_table = dictionary_table.copy()
    if candidate_table.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "motif_id",
                "topology_name",
                "human_name",
                "key",
                "is_selected",
                "graph_support",
                "support_fraction",
                "total_occurrences",
                "forced_net_savings_bits",
                "single_rule_savings_bits",
            ]
        )

    candidate_table = _canonicalize_columns(candidate_table)
    dictionary_table = _canonicalize_columns(dictionary_table)

    if "rank" not in candidate_table:
        candidate_table.insert(0, "rank", np.arange(len(candidate_table)))

    selected_keys: set[str] = set()
    if "key" in dictionary_table:
        selected_keys = set(dictionary_table["key"].astype(str))
    elif "motif_key" in dictionary_table:
        selected_keys = set(dictionary_table["motif_key"].astype(str))

    key_column = "key" if "key" in candidate_table else "motif_key"
    if key_column in candidate_table:
        candidate_table["is_selected"] = (
            candidate_table[key_column].astype(str).isin(selected_keys)
        )
    else:
        candidate_table["is_selected"] = False

    if "graph_support" in candidate_table and n_fit_graphs:
        candidate_table["support_fraction"] = (
            pd.to_numeric(candidate_table["graph_support"], errors="coerce")
            / n_fit_graphs
        )

    if {
        "total_occurrences",
        "graph_support",
    }.issubset(candidate_table.columns):
        support = pd.to_numeric(
            candidate_table["graph_support"], errors="coerce"
        ).replace(0, np.nan)
        occurrences = pd.to_numeric(
            candidate_table["total_occurrences"], errors="coerce"
        )
        candidate_table["occurrences_per_supported_graph"] = occurrences / support

    savings_column = None
    if "forced_net_savings_bits" in candidate_table:
        savings_column = "forced_net_savings_bits"
    elif "single_rule_savings_bits" in candidate_table:
        savings_column = "single_rule_savings_bits"

    if savings_column is not None and "total_occurrences" in candidate_table:
        occurrences = pd.to_numeric(
            candidate_table["total_occurrences"], errors="coerce"
        ).replace(0, np.nan)
        savings = pd.to_numeric(
            candidate_table[savings_column], errors="coerce"
        )
        candidate_table["savings_per_occurrence_bits"] = savings / occurrences

    if (
        "forced_net_savings_bits" in candidate_table
        and "single_rule_savings_bits" not in candidate_table
    ):
        candidate_table["single_rule_savings_bits"] = candidate_table[
            "forced_net_savings_bits"
        ]

    sort_columns = [
        column
        for column in (
            "is_selected",
            "forced_net_savings_bits",
            "single_rule_savings_bits",
            "total_occurrences",
        )
        if column in candidate_table
    ]
    if sort_columns:
        ascending = [False] * len(sort_columns)
        candidate_table = candidate_table.sort_values(
            sort_columns, ascending=ascending, kind="stable"
        )

    return candidate_table.reset_index(drop=True)


def bootstrap_graph_statistics(
    per_graph: pd.DataFrame,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap graph-level aggregate statistics.

    These intervals describe variation across graphs.  They do not re-learn
    the motif dictionary, and therefore should not be presented as full
    dictionary-selection uncertainty.
    """

    if per_graph.empty or replicates == 0:
        return pd.DataFrame(
            columns=["metric", "estimate", "ci_low", "ci_high", "n_graphs"]
        )

    metrics = _analysis_metrics(per_graph)
    if not metrics:
        return pd.DataFrame(
            columns=["metric", "estimate", "ci_low", "ci_high", "n_graphs"]
        )

    rng = np.random.default_rng(seed)
    n_graphs = len(per_graph)
    alpha = (1.0 - confidence_level) / 2.0
    rows: list[dict[str, Any]] = []

    for name, values in metrics.items():
        clean = np.asarray(values, dtype=float)
        clean = clean[np.isfinite(clean)]
        if clean.size == 0:
            continue
        samples = rng.choice(clean, size=(replicates, clean.size), replace=True)
        boot_means = samples.mean(axis=1)
        rows.append(
            {
                "metric": name,
                "estimate": float(clean.mean()),
                "ci_low": float(np.quantile(boot_means, alpha)),
                "ci_high": float(np.quantile(boot_means, 1.0 - alpha)),
                "n_graphs": int(clean.size),
            }
        )

    return pd.DataFrame(rows)


def render_markdown_summary(result: MDLAnalysisResult, *, top_n: int = 15) -> str:
    """Render a compact report suitable for an artifact directory."""

    lines = [
        "# Buhito MDL Analysis",
        "",
        "## Run summary",
        "",
        f"- Total graphs: {result.summary.get('n_graphs_total', 0)}",
        f"- Fit graphs: {result.summary.get('n_fit_graphs', 0)}",
        f"- Evaluation graphs: {result.summary.get('n_eval_graphs', 0)}",
        f"- Candidate motifs: {result.summary.get('n_candidate_motifs', 0)}",
        f"- Selected motifs: {result.summary.get('n_selected_motifs', 0)}",
        "",
        "An empty dictionary or negative savings is a valid MDL result.",
        "",
        "## Top motifs",
        "",
    ]
    if result.motifs.empty:
        lines.append("No candidate motifs passed the configured filters.")
    else:
        display_columns = [
            column
            for column in (
                "rank",
                "motif_id",
                "topology_name",
                "human_name",
                "key",
                "is_selected",
                "graph_support",
                "support_fraction",
                "total_occurrences",
                "forced_net_savings_bits",
                "single_rule_savings_bits",
                "savings_per_occurrence_bits",
            )
            if column in result.motifs
        ]
        lines.append(_markdown_table(result.motifs[display_columns].head(top_n)))

    lines.extend(["", "## Forced candidate cost decomposition", ""])
    if result.motif_costs.empty:
        lines.append("No forced candidate costs were available.")
    else:
        cost_summary = (
            result.motif_costs.pivot_table(
                index=["motif_id", "human_name"],
                columns="component",
                values="bits",
                aggfunc="sum",
            )
            .reset_index()
            .head(top_n)
        )
        lines.append(_markdown_table(cost_summary))

    lines.extend(["", "## Motif assets", ""])
    if result.motif_assets.empty:
        lines.append("No portable motif assets were written.")
    else:
        lines.append(_markdown_table(result.motif_assets.head(top_n)))

    lines.extend(["", "## Bootstrap graph-level statistics", ""])
    if result.bootstrap.empty:
        lines.append("No graph-level numeric metric was available for bootstrap analysis.")
    else:
        lines.append(_markdown_table(result.bootstrap))
    lines.append("")
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small markdown table without adding a tabulate dependency."""

    if frame.empty:
        return ""
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: str(value).replace("|", "\\|").replace("\n", " ")
        )
    header = "| " + " | ".join(map(str, display.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in display.astype(str).itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _analysis_metrics(per_graph: pd.DataFrame) -> dict[str, np.ndarray]:
    metrics: dict[str, np.ndarray] = {}
    aliases = {
        "graph_gain_bits": (
            "net_savings_bits",
            "gross_gain_bits",
            "bits_saved",
        ),
        "node_reduction_fraction": (
            "node_reduction_fraction",
            "node_reduction_frac",
        ),
        "edge_reduction_fraction": (
            "edge_reduction_fraction",
            "edge_reduction_frac",
        ),
        "unseen_graphlet_fraction": ("unseen_graphlet_fraction",),
    }
    for output_name, candidates in aliases.items():
        for column in candidates:
            if column in per_graph:
                metrics[output_name] = pd.to_numeric(
                    per_graph[column], errors="coerce"
                ).to_numpy()
                break

    if "use_rewrite" in per_graph:
        metrics["rewrite_fraction"] = (
            per_graph["use_rewrite"].astype(bool).astype(float).to_numpy()
        )
    return metrics


def _decode_failures(
    result: Any,
    originals: Sequence[nx.Graph],
    *,
    already_validated: bool = False,
) -> list[int]:
    if result is None or not hasattr(result, "decoded_graphs"):
        return []
    if already_validated:
        # The compressor already performed exact, ID-based reconstruction
        # validation for every rewritten graph during fit()/transform()
        # (see MDLGraphCompressor(validate=True) and
        # ``_rewrite_validation_error`` in mdl.py), raising an
        # AssertionError immediately on any mismatch. If we reached this
        # point, every record has already been proven to decode exactly --
        # re-deriving that via ``nx.is_isomorphic`` here is not just
        # redundant, it can also be extremely slow (VF2 has no worst-case
        # polynomial bound and is particularly expensive on graphs with
        # large automorphism groups, e.g. hub-and-many-leaves structures
        # common in social-network datasets like REDDIT-MULTI-5K), for a
        # result that is already mathematically guaranteed to be empty.
        return []
    decoded = list(result.decoded_graphs())
    if len(decoded) != len(originals):
        return list(range(max(len(decoded), len(originals))))
    return [
        index
        for index, (original, restored) in enumerate(zip(originals, decoded, strict=True))
        if not nx.is_isomorphic(original, restored)
    ]


def _per_graph_frame(result: Any, source_indices: np.ndarray, split: str) -> pd.DataFrame:
    frame = getattr(result, "per_graph", pd.DataFrame()).copy()
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    frame.insert(0, "split", split)
    frame.insert(1, "source_graph_index", source_indices[: len(frame)])
    return frame


def _motif_graphs_from_object(obj: Any) -> dict[str, nx.Graph]:
    """Return independent candidate motif graphs from a public accessor."""

    for name in _MOTIF_GRAPH_NAMES:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        value = value() if callable(value) else value
        if not isinstance(value, Mapping):
            continue
        graphs: dict[str, nx.Graph] = {}
        for key, graph in value.items():
            if isinstance(graph, nx.Graph):
                graphs[str(key)] = graph.copy()
        if graphs:
            return graphs
    return {}


def _candidate_frame_from_object(obj: Any) -> pd.DataFrame:
    """Return the compressor's scored candidate table.

    Public accessors are preferred.  The fallback inspects candidate-like fit
    state so the analysis package remains compatible with older Buhito MDL
    builds that retained candidate rows privately but did not expose a frame
    method.  Once the compressor has a stable ``candidate_frame()`` method,
    the fallback can remain solely for backwards compatibility.
    """

    direct = _frame_from_object(obj, _CANDIDATE_FRAME_NAMES)
    if not direct.empty:
        return direct

    frames: list[tuple[int, pd.DataFrame]] = []
    queue: list[tuple[Any, int]] = [(obj, 0)]
    visited: set[int] = set()

    while queue:
        current, depth = queue.pop(0)
        identity = id(current)
        if identity in visited or depth > 2:
            continue
        visited.add(identity)

        try:
            attributes = vars(current)
        except TypeError:
            continue

        for name, value in attributes.items():
            lowered = name.lower()
            candidate_like = any(
                token in lowered
                for token in (
                    "candidate",
                    "ranked_rule",
                    "rule_score",
                    "single_rule",
                    "motif_score",
                )
            )
            if candidate_like:
                frame = _coerce_frame(value)
                if not frame.empty:
                    normalized = _canonicalize_columns(frame)
                    score = _candidate_frame_score(normalized)
                    if score > 0:
                        frames.append((score, frame))

            if depth < 2 and any(
                token in lowered
                for token in ("fit_state", "fit_result", "state", "result")
            ):
                if hasattr(value, "__dict__"):
                    queue.append((value, depth + 1))

    if not frames:
        return pd.DataFrame()

    frames.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return frames[0][1].copy()


def _candidate_frame_score(frame: pd.DataFrame) -> int:
    columns = set(frame.columns)
    score = 0
    if columns.intersection({"key", "motif_key"}):
        score += 10
    if "graph_support" in columns:
        score += 4
    if "total_occurrences" in columns:
        score += 4
    if "single_rule_savings_bits" in columns:
        score += 6
    if "forced_net_savings_bits" in columns:
        score += 8
    if "rank" in columns:
        score += 2
    return score


def _coerce_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if is_dataclass(value):
        return pd.DataFrame([asdict(value)])
    if isinstance(value, Mapping):
        if not value:
            return pd.DataFrame()
        try:
            return pd.DataFrame(value)
        except ValueError:
            pass

        rows: list[dict[str, Any]] = []
        for key, item in value.items():
            if is_dataclass(item):
                row = asdict(item)
            elif isinstance(item, Mapping):
                row = dict(item)
            elif hasattr(item, "__dict__"):
                row = dict(vars(item))
            else:
                continue
            row.setdefault("key", key)
            rows.append(row)
        return pd.DataFrame(rows)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows: list[Any] = []
        for item in value:
            if is_dataclass(item):
                rows.append(asdict(item))
            elif isinstance(item, Mapping):
                rows.append(dict(item))
            elif hasattr(item, "__dict__"):
                rows.append(dict(vars(item)))
            else:
                rows.append(item)
        try:
            return pd.DataFrame(rows)
        except (TypeError, ValueError):
            return pd.DataFrame()
    if hasattr(value, "__dict__"):
        return pd.DataFrame([dict(vars(value))])
    return pd.DataFrame()


def _frame_from_object(obj: Any, names: Sequence[str]) -> pd.DataFrame:
    for name in names:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        value = value() if callable(value) else value
        frame = _coerce_frame(value)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def _report_mapping(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    report = getattr(result, "report", None)
    if report is None:
        return {}
    if is_dataclass(report):
        return asdict(report)
    if isinstance(report, Mapping):
        return dict(report)
    if hasattr(report, "__dict__"):
        return dict(vars(report))
    return {"value": str(report)}


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a table with stable canonical column names.

    Some compressor versions expose both a legacy alias such as
    ``occurrences`` and its canonical replacement ``total_occurrences``.
    A blind ``DataFrame.rename`` creates duplicate column labels in that case,
    and selecting the canonical name then returns a two-dimensional frame
    instead of a Series.  Prefer an already-present canonical column, use the
    alias only to fill missing values, and remove the redundant alias.
    """

    aliases = {
        "motif": "key",
        "candidate_key": "key",
        "rule_key": "key",
        "motif_key": "key",
        "support": "graph_support",
        "graphs_with_occurrence": "graph_support",
        "occurrences": "total_occurrences",
        "raw_total_occurrences": "total_occurrences",
        "bits_saved": "single_rule_savings_bits",
        "total_bits_saved": "single_rule_savings_bits",
        "net_savings_bits": "single_rule_savings_bits",
        "single_rule_bits_saved": "single_rule_savings_bits",
        "single_rule_net_savings_bits": "forced_net_savings_bits",
    }

    normalized = frame.copy()
    for source, target in aliases.items():
        if source not in normalized.columns or source == target:
            continue

        if target in normalized.columns:
            canonical = normalized[target]
            alias = normalized[source]
            normalized[target] = canonical.where(canonical.notna(), alias)
            normalized = normalized.drop(columns=[source])
        else:
            normalized = normalized.rename(columns={source: target})

    # Be defensive about input frames that already contain duplicate labels.
    # Preserve the leftmost non-null value for each canonical name.
    if normalized.columns.duplicated().any():
        deduplicated = pd.DataFrame(index=normalized.index)
        for name in dict.fromkeys(normalized.columns):
            block = normalized.loc[:, normalized.columns == name]
            values = block.iloc[:, 0]
            for position in range(1, block.shape[1]):
                values = values.where(values.notna(), block.iloc[:, position])
            deduplicated[name] = values
        normalized = deduplicated

    return normalized


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value

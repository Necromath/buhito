"""Fitted, dataset-independent graph compression pipeline."""

from __future__ import annotations

import gzip
import hashlib
import pickle
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .dictionary import (
    DictionarySelection,
    Selector,
    _apply_selector,
    _boundary_bits,
    _dictionary_bits,
    _records_frame,
    _selected_rewrite_components,
    _template_alphabets,
    _template_incidence_graph,
)
from .models import (
    _EDGE_LABEL,
    _NODE_LABEL,
    Alphabets,
    CompressionResult,
    EncodedGraph,
    GraphSchema,
    MotifRule,
    Occurrence,
    Rewrite,
    _base_bits,
    _build_alphabets,
    _stable_token,
)
from .recognition import (
    BuhitoGraphletEnumerator,
    GraphletEnumerator,
    _Candidate,
    _discover_candidates,
    _find_frozen_occurrences,
    _FoundRecord,
)
from .substitution import _rewrite_graph, _rewrite_validation_error


class MDLGraphCompressor:
    """Learn and apply a lossless graphlet MDL dictionary.

    The estimator follows a train/transform contract so dictionary discovery is
    performed only on training graphs.  Candidate graphlets are ranked by their
    single-rule MDL gain, and the best joint prefix of at most ``n_rules`` is
    retained.  The corpus selector is then applied independently to each
    transformed dataset.

    Parameters
    ----------
    graphlet_sizes:
        Connected induced graphlet sizes considered by the enumerator.
    n_rules:
        Maximum dictionary size.
    min_graph_support, min_occurrences:
        Candidate filters on the fit corpus.
    max_candidates:
        Maximum number of support-ranked candidates that receive the expensive
        exact MDL evaluation.
    node_label_keys, edge_label_keys:
        Attributes preserved by the code.  ``None`` means topology-only.
    selector:
        ``"sparse"`` uses the corpus-level enumerative selector from the final
        notebooks; ``"per_graph"`` uses one flag per graph; ``"all_eligible"``
        rewrites every graph containing a selected motif.
    model_choice_bits:
        Optional cost for choosing baseline-only versus rewrite-capable corpus
        code.
    min_rule_savings_bits:
        Minimum single-rule training gain before a candidate is allowed into the
        joint-prefix search. Use ``-math.inf`` for diagnostic experiments.
    dictionary_selection:
        ``"best"`` selects the best prefix including the empty dictionary.
        ``"best_nonempty"`` selects the best non-empty prefix when any candidate
        passes the threshold. ``"fixed"`` selects the longest evaluated prefix.
        The latter two modes are useful when a scientifically informative negative
        MDL result should still produce compressed graphs for downstream analysis.
    enumerator:
        Graphlet occurrence backend.  Defaults to Buhito BFS.
    cache_dir:
        Optional directory for corpus-specific occurrence caches.
    validate:
        Decode and check every candidate rewrite during fit/transform.
    progress:
        Print coarse progress messages.
    """

    def __init__(
        self,
        *,
        graphlet_sizes: Sequence[int] = (3,),
        n_rules: int = 5,
        min_graph_support: int = 2,
        min_occurrences: int = 2,
        max_candidates: int = 100,
        node_label_keys: str | Sequence[str] | None = None,
        edge_label_keys: str | Sequence[str] | None = None,
        selector: Selector = "sparse",
        model_choice_bits: float = 1.0,
        min_rule_savings_bits: float = 0.0,
        dictionary_selection: DictionarySelection = "best",
        enumerator: GraphletEnumerator | None = None,
        cache_dir: str | Path | None = None,
        validate: bool = True,
        progress: bool = False,
    ) -> None:
        sizes = tuple(sorted({int(size) for size in graphlet_sizes}))
        if not sizes or min(sizes) < 2:
            raise ValueError("graphlet_sizes must contain integers >= 2.")
        if n_rules < 0 or max_candidates < 1:
            raise ValueError("n_rules must be >= 0 and max_candidates must be >= 1.")
        if selector not in {"sparse", "per_graph", "all_eligible"}:
            raise ValueError(f"Unknown selector {selector!r}.")
        if dictionary_selection not in {"best", "best_nonempty", "fixed"}:
            raise ValueError(
                f"Unknown dictionary_selection {dictionary_selection!r}."
            )

        self.graphlet_sizes = sizes
        self.n_rules = int(n_rules)
        self.min_graph_support = int(min_graph_support)
        self.min_occurrences = int(min_occurrences)
        self.max_candidates = int(max_candidates)
        self.schema = GraphSchema.from_keys(
            node_label_keys=node_label_keys,
            edge_label_keys=edge_label_keys,
        )
        self.selector = selector
        self.model_choice_bits = float(model_choice_bits)
        self.min_rule_savings_bits = float(min_rule_savings_bits)
        self.dictionary_selection = dictionary_selection
        self.enumerator = enumerator or BuhitoGraphletEnumerator()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.validate = bool(validate)
        self.progress = bool(progress)

        self.rules_: tuple[MotifRule, ...] | None = None
        self.alphabets_: Alphabets | None = None
        self.seen_base_keys_: set[Hashable] | None = None
        self.candidate_table_: pd.DataFrame | None = None
        self.candidate_motifs_: dict[str, nx.Graph] | None = None
        self.dictionary_path_: pd.DataFrame | None = None
        self.training_result_: CompressionResult | None = None

    def _check_fitted(self) -> None:
        if self.rules_ is None or self.alphabets_ is None:
            raise RuntimeError("Call fit before transform.")

    def _normalize_many(self, graphs: Iterable[nx.Graph]) -> list[nx.Graph]:
        return [self.schema.normalize(graph) for graph in graphs]

    def _corpus_digest(self, graphs: list[nx.Graph]) -> str:
        digest = hashlib.sha256()
        digest.update(repr(self.graphlet_sizes).encode("utf-8"))
        enumerator_name = getattr(
            self.enumerator,
            "name",
            type(self.enumerator).__name__,
        )
        digest.update(enumerator_name.encode("utf-8"))
        for graph in graphs:
            node_rows = tuple(
                (node, _stable_token(data[_NODE_LABEL]))
                for node, data in sorted(graph.nodes(data=True))
            )
            edge_rows = tuple(
                sorted(
                    (
                        min(source, target),
                        max(source, target),
                        _stable_token(data[_EDGE_LABEL]),
                    )
                    for source, target, data in graph.edges(data=True)
                )
            )
            digest.update(repr((node_rows, edge_rows)).encode("utf-8"))
        return digest.hexdigest()

    def _enumerate_many(
        self, graphs: list[nx.Graph]
    ) -> list[dict[Hashable, list[frozenset[int]]]]:
        cache_path: Path | None = None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_name = f"mdl_occurrences_{self._corpus_digest(graphs)}.pkl.gz"
            cache_path = self.cache_dir / cache_name
            if cache_path.exists():
                if self.progress:
                    print(f"Loading MDL occurrence cache: {cache_path}")
                with gzip.open(cache_path, "rb") as handle:
                    return pickle.load(handle)

        raw: list[dict[Hashable, list[frozenset[int]]]] = []
        for index, graph in enumerate(graphs):
            if self.progress and (index == 0 or index % 100 == 0):
                print(f"Enumerating graphlets {index}/{len(graphs)}")
            raw.append(self.enumerator.enumerate(graph, self.graphlet_sizes))

        if cache_path is not None:
            with gzip.open(cache_path, "wb") as handle:
                pickle.dump(raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if self.progress:
                print(f"Saved MDL occurrence cache: {cache_path}")
        return raw

    def _support_table(
        self,
        candidates: dict[str, _Candidate],
        found_list: list[dict[str, _FoundRecord]],
    ) -> pd.DataFrame:
        graph_support: Counter[str] = Counter()
        occurrences: Counter[str] = Counter()
        for found in found_list:
            for key, record in found.items():
                graph_support[key] += 1
                occurrences[key] += len(record.instances)

        rows: list[dict[str, Any]] = []
        for key, candidate in candidates.items():
            temporary_rule = MotifRule(
                key=key,
                base_key=candidate.base_key,
                motif=candidate.motif,
                rank=0,
            )
            rows.append(
                {
                    "key": key,
                    "base_key": candidate.base_key,
                    "graph_support": graph_support[key],
                    "occurrences": occurrences[key],
                    "motif_nodes": candidate.motif.number_of_nodes(),
                    "motif_edges": candidate.motif.number_of_edges(),
                    "automorphism_order": temporary_rule.automorphism_order,
                    "orbit_sizes": temporary_rule.orbit_sizes,
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "key",
                    "base_key",
                    "graph_support",
                    "occurrences",
                    "motif_nodes",
                    "motif_edges",
                    "automorphism_order",
                    "orbit_sizes",
                ]
            )
        return (
            pd.DataFrame(rows)
            .sort_values(
                ["graph_support", "occurrences", "key"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )

    @staticmethod
    def _occurrences_from_found(
        found_list: list[dict[str, _FoundRecord]],
        rules: tuple[MotifRule, ...],
    ) -> list[dict[int, list[Occurrence]]]:
        result: list[dict[int, list[Occurrence]]] = []
        for found in found_list:
            per_rule: dict[int, list[Occurrence]] = {}
            for rule in rules:
                record = found.get(rule.key)
                per_rule[rule.rank] = list(record.instances) if record else []
            result.append(per_rule)
        return result

    def _build_records(
        self,
        graphs: list[nx.Graph],
        raw_occurrences: list[dict[Hashable, list[frozenset[int]]]],
        rules: tuple[MotifRule, ...],
        occurrences_by_graph: list[dict[int, list[Occurrence]]],
        alphabets: Alphabets,
        seen_base_keys: set[Hashable],
    ) -> list[EncodedGraph]:
        template_alphabets = _template_alphabets(alphabets, rules)
        records: list[EncodedGraph] = []

        for graph_index, (graph, raw, occurrences_by_rule) in enumerate(
            zip(graphs, raw_occurrences, occurrences_by_graph)
        ):
            baseline_bits = _base_bits(graph, alphabets, complete=True)
            total_graphlet_occurrences = sum(len(value) for value in raw.values())
            unseen_graphlet_occurrences = sum(
                len(value)
                for key, value in raw.items()
                if key not in seen_base_keys
            )
            candidate_occurrences = sum(
                len(occurrences) for occurrences in occurrences_by_rule.values()
            )

            rewrite: Rewrite | None = None
            template_bits: float | None = None
            boundary_bits: float | None = None
            rewrite_bits: float | None = None
            selected_occurrences = 0
            rules_used = 0

            if candidate_occurrences:
                rewrite = _rewrite_graph(graph, rules, occurrences_by_rule)
                if rewrite.selected:
                    if self.validate:
                        validation_error = _rewrite_validation_error(
                            graph, rewrite
                        )
                        if validation_error is not None:
                            raise AssertionError(
                                "MDL rewrite failed reconstruction on graph "
                                f"{graph_index}: {validation_error}."
                            )
                    incidence = _template_incidence_graph(rewrite.template)
                    template_bits = _base_bits(
                        incidence, template_alphabets, complete=True
                    )
                    boundary_bits = _boundary_bits(
                        rewrite, quotient_by_automorphisms=True
                    )
                    rewrite_bits = template_bits + boundary_bits
                    selected_occurrences = len(rewrite.selected)
                    rules_used = len({rank for rank, _ in rewrite.selected})
                else:
                    rewrite = None

            records.append(
                EncodedGraph(
                    graph_index=graph_index,
                    baseline_graph=graph,
                    rewrite=rewrite,
                    use_rewrite=False,
                    baseline_bits=baseline_bits,
                    template_bits=template_bits,
                    boundary_bits=boundary_bits,
                    rewrite_bits=rewrite_bits,
                    candidate_occurrences=candidate_occurrences,
                    selected_occurrences=selected_occurrences,
                    rules_used=rules_used,
                    total_graphlet_occurrences=total_graphlet_occurrences,
                    unseen_graphlet_occurrences=unseen_graphlet_occurrences,
                )
            )
        return records

    def _evaluate_normalized(
        self,
        graphs: list[nx.Graph],
        raw_occurrences: list[dict[Hashable, list[frozenset[int]]]],
        rules: tuple[MotifRule, ...],
        occurrences_by_graph: list[dict[int, list[Occurrence]]],
        alphabets: Alphabets,
        seen_base_keys: set[Hashable],
        *,
        require_nonempty_rewrite: bool = False,
    ) -> CompressionResult:
        records = self._build_records(
            graphs,
            raw_occurrences,
            rules,
            occurrences_by_graph,
            alphabets,
            seen_base_keys,
        )
        dictionary_bits = _dictionary_bits(rules, alphabets)
        report, selector_curve = _apply_selector(
            records,
            selector=self.selector,
            dictionary_bits=dictionary_bits,
            model_choice_bits=self.model_choice_bits,
            require_nonempty_rewrite=require_nonempty_rewrite,
        )
        return CompressionResult(
            schema=self.schema,
            rules=rules,
            records=records,
            report=report,
            per_graph=_records_frame(records),
            selector_curve=selector_curve,
        )

    def fit(self, graphs: Iterable[nx.Graph]) -> MDLGraphCompressor:
        normalized = self._normalize_many(graphs)
        if not normalized:
            raise ValueError("fit requires at least one graph.")
        alphabets = _build_alphabets(normalized)
        raw = self._enumerate_many(normalized)
        candidates, found_list = _discover_candidates(normalized, raw)
        support = self._support_table(candidates, found_list)
        filtered = support[
            (support["graph_support"] >= self.min_graph_support)
            & (support["occurrences"] >= self.min_occurrences)
        ].head(self.max_candidates)

        seen_base_keys = {key for graph_raw in raw for key in graph_raw}
        candidate_rows: list[dict[str, Any]] = []
        for position, row in enumerate(filtered.itertuples(index=False), start=1):
            if self.progress:
                print(f"Scoring candidate {position}/{len(filtered)}: {row.key}")
            candidate = candidates[row.key]
            rule = MotifRule(
                key=candidate.key,
                base_key=candidate.base_key,
                motif=candidate.motif.copy(),
                rank=0,
            )
            occurrences = self._occurrences_from_found(found_list, (rule,))
            forced_result = self._evaluate_normalized(
                normalized,
                raw,
                (rule,),
                occurrences,
                alphabets,
                seen_base_keys,
                require_nonempty_rewrite=True,
            )

            forced_components = _selected_rewrite_components(
                forced_result.records
            )
            best_model_report, _ = _apply_selector(
                forced_result.records,
                selector=self.selector,
                dictionary_bits=_dictionary_bits((rule,), alphabets),
                model_choice_bits=self.model_choice_bits,
            )
            candidate_rows.append(
                {
                    **row._asdict(),
                    "total_occurrences": row.occurrences,
                    **{
                        f"forced_{name}": value
                        for name, value in asdict(forced_result.report).items()
                    },
                    **{
                        f"forced_{name}": value
                        for name, value in forced_components.items()
                    },
                    **{
                        f"best_model_{name}": value
                        for name, value in asdict(best_model_report).items()
                    },
                    # Backwards-compatible alias. New analysis code should use
                    # ``forced_net_savings_bits`` explicitly.
                    "single_rule_net_savings_bits": (
                        forced_result.report.net_savings_bits
                    ),
                }
            )

        candidate_table = pd.DataFrame(candidate_rows)
        if not candidate_table.empty:
            candidate_table = candidate_table.sort_values(
                ["forced_net_savings_bits", "graph_support", "occurrences"],
                ascending=[False, False, False],
            ).reset_index(drop=True)
            candidate_table.insert(
                0,
                "rank",
                np.arange(1, len(candidate_table) + 1, dtype=int),
            )
            candidate_table.insert(
                1,
                "motif_id",
                [f"M{rank:03d}" for rank in candidate_table["rank"]],
            )
            candidate_table["passes_min_rule_savings"] = (
                candidate_table["forced_net_savings_bits"]
                >= self.min_rule_savings_bits
            )

        ranked_keys: list[str] = []
        allowed_keys: list[str] = []
        if not candidate_table.empty and self.n_rules:
            ranked_keys = candidate_table["key"].head(self.n_rules).tolist()
            allowed_keys = candidate_table.loc[
                candidate_table["passes_min_rule_savings"],
                "key",
            ].head(self.n_rules).tolist()

        path_rows: list[dict[str, Any]] = []
        prefix_results: list[tuple[tuple[MotifRule, ...], CompressionResult]] = []

        # Empty dictionary is a valid and important null result.
        empty_result = self._evaluate_normalized(
            normalized,
            raw,
            (),
            [dict() for _ in normalized],
            alphabets,
            seen_base_keys,
        )
        prefix_results.append(((), empty_result))
        path_rows.append(
            {
                "n_rules": 0,
                "keys": (),
                "rule_keys": (),
                "is_empty": True,
                "eligible_for_selection": True,
                "dictionary_selection": self.dictionary_selection,
                **_selected_rewrite_components(empty_result.records),
                **asdict(empty_result.report),
            }
        )

        for prefix_size in range(1, len(ranked_keys) + 1):
            keys = ranked_keys[:prefix_size]
            rules = tuple(
                MotifRule(
                    key=key,
                    base_key=candidates[key].base_key,
                    motif=candidates[key].motif.copy(),
                    rank=rank,
                )
                for rank, key in enumerate(keys)
            )
            occurrences = self._occurrences_from_found(found_list, rules)
            result = self._evaluate_normalized(
                normalized,
                raw,
                rules,
                occurrences,
                alphabets,
                seen_base_keys,
                require_nonempty_rewrite=True,
            )
            prefix_results.append((rules, result))
            path_rows.append(
                {
                    "n_rules": prefix_size,
                    "keys": tuple(keys),
                    "rule_keys": tuple(keys),
                    "is_empty": False,
                    "eligible_for_selection": prefix_size <= len(allowed_keys),
                    "dictionary_selection": self.dictionary_selection,
                    **_selected_rewrite_components(result.records),
                    **asdict(result.report),
                }
            )

        eligible_prefix_results = prefix_results[: len(allowed_keys) + 1]
        if self.dictionary_selection == "best":
            selection_pool = eligible_prefix_results
            best_rules, best_result = max(
                selection_pool,
                key=lambda item: item[1].report.net_savings_bits,
            )
        elif self.dictionary_selection == "best_nonempty":
            selection_pool = [
                item for item in eligible_prefix_results if item[0]
            ]
            if selection_pool:
                best_rules, best_result = max(
                    selection_pool,
                    key=lambda item: item[1].report.net_savings_bits,
                )
            else:
                best_rules, best_result = prefix_results[0]
        else:  # fixed
            best_rules, best_result = eligible_prefix_results[-1]

        selected_keys = tuple(rule.key for rule in best_rules)
        if not candidate_table.empty:
            candidate_table["is_selected"] = candidate_table["key"].isin(
                selected_keys
            )
        for path_row in path_rows:
            path_row["is_best"] = tuple(path_row["keys"]) == selected_keys

        self.rules_ = best_rules
        self.alphabets_ = alphabets
        self.seen_base_keys_ = seen_base_keys
        self.candidate_table_ = candidate_table
        self.candidate_motifs_ = {
            key: candidates[key].motif.copy()
            for key in candidate_table.get("key", pd.Series(dtype=str)).tolist()
        }
        self.dictionary_path_ = pd.DataFrame(path_rows)
        self.training_result_ = best_result
        return self

    def transform(self, graphs: Iterable[nx.Graph]) -> CompressionResult:
        self._check_fitted()
        assert self.rules_ is not None
        assert self.alphabets_ is not None
        assert self.seen_base_keys_ is not None

        normalized = self._normalize_many(graphs)
        raw = self._enumerate_many(normalized)
        occurrences_by_graph = [
            _find_frozen_occurrences(graph, graph_raw, self.rules_)
            for graph, graph_raw in zip(normalized, raw)
        ]
        return self._evaluate_normalized(
            normalized,
            raw,
            self.rules_,
            occurrences_by_graph,
            self.alphabets_,
            self.seen_base_keys_,
        )

    def rule_prefix(self, n_rules: int) -> tuple[MotifRule, ...]:
        """Return the first ``n_rules`` scored motifs as a fresh rule prefix.

        The prefix order is the deterministic forced-MDL candidate ranking used
        during :meth:`fit`.  Returned rules and motif graphs are independent
        copies, so callers may safely use them for diagnostic prefix sweeps
        without mutating the fitted estimator or its selected dictionary.
        """

        self._check_fitted()
        if n_rules < 0:
            raise ValueError("n_rules must be nonnegative.")
        assert self.candidate_table_ is not None
        assert self.candidate_motifs_ is not None
        available = len(self.candidate_table_)
        if n_rules > available:
            raise ValueError(
                f"Requested {n_rules} rules, but only {available} scored "
                "candidates are available."
            )
        if n_rules == 0:
            return ()

        ranked = (
            self.candidate_table_
            .sort_values("rank")
            .head(n_rules)
        )
        return tuple(
            MotifRule(
                key=str(row.key),
                base_key=row.base_key,
                motif=self.candidate_motifs_[str(row.key)].copy(),
                rank=rank,
            )
            for rank, row in enumerate(ranked.itertuples(index=False))
        )

    def transform_rule_prefix(
        self,
        graphs: Iterable[nx.Graph],
        n_rules: int,
        *,
        require_nonempty_rewrite: bool = False,
    ) -> CompressionResult:
        """Transform graphs with a ranked dictionary prefix.

        This method is intended for controlled compression--speed--quality
        sweeps.  It reuses the fitted alphabets, candidate ranking, and
        occurrence cache while leaving :attr:`rules_` unchanged.

        Parameters
        ----------
        graphs:
            Graphs to transform.
        n_rules:
            Number of top-ranked candidate rules in the diagnostic prefix.
            Zero is the identity/baseline representation.
        require_nonempty_rewrite:
            When true, the corpus selector must encode at least one available
            rewrite.  This reports the actual forced-representation MDL cost
            rather than silently falling back to the empty model.
        """

        self._check_fitted()
        assert self.alphabets_ is not None
        assert self.seen_base_keys_ is not None
        rules = self.rule_prefix(n_rules)
        normalized = self._normalize_many(graphs)
        raw = self._enumerate_many(normalized)
        occurrences_by_graph = [
            _find_frozen_occurrences(graph, graph_raw, rules)
            for graph, graph_raw in zip(normalized, raw)
        ]
        return self._evaluate_normalized(
            normalized,
            raw,
            rules,
            occurrences_by_graph,
            self.alphabets_,
            self.seen_base_keys_,
            require_nonempty_rewrite=require_nonempty_rewrite,
        )

    def fit_transform(self, graphs: Iterable[nx.Graph]) -> CompressionResult:
        graph_list = list(graphs)
        self.fit(graph_list)
        assert self.training_result_ is not None
        return self.training_result_

    def dictionary_frame(self) -> pd.DataFrame:
        self._check_fitted()
        assert self.rules_ is not None
        candidate_ids: dict[str, str] = {}
        if self.candidate_table_ is not None and not self.candidate_table_.empty:
            candidate_ids = dict(
                zip(
                    self.candidate_table_["key"].astype(str),
                    self.candidate_table_["motif_id"].astype(str),
                    strict=True,
                )
            )
        return pd.DataFrame(
            [
                {
                    "rank": rule.rank,
                    "motif_id": candidate_ids.get(rule.key),
                    "key": rule.key,
                    "base_key": rule.base_key,
                    "motif_nodes": rule.motif.number_of_nodes(),
                    "motif_edges": rule.motif.number_of_edges(),
                    "automorphism_order": rule.automorphism_order,
                    "orbit_sizes": rule.orbit_sizes,
                }
                for rule in self.rules_
            ]
        )

    def candidate_frame(self) -> pd.DataFrame:
        """Return exact forced and best-model accounting for scored motifs."""

        self._check_fitted()
        assert self.candidate_table_ is not None
        return self.candidate_table_.copy()

    def candidate_motif_graphs(
        self,
        *,
        restored: bool = True,
    ) -> dict[str, nx.Graph]:
        """Return copies of every scored candidate motif keyed by candidate key.

        Parameters
        ----------
        restored:
            When true, expose the user-facing node and edge attributes selected
            by :class:`GraphSchema`. When false, retain Buhito's normalized
            internal labels. Returned graphs are always independent copies.
        """

        self._check_fitted()
        assert self.candidate_motifs_ is not None
        if restored:
            return {
                key: self.schema.restore(motif)
                for key, motif in self.candidate_motifs_.items()
            }
        return {
            key: motif.copy()
            for key, motif in self.candidate_motifs_.items()
        }

    def candidate_motif(
        self,
        key: str,
        *,
        restored: bool = True,
    ) -> nx.Graph:
        """Return one scored candidate motif as an independent graph copy."""

        motifs = self.candidate_motif_graphs(restored=restored)
        try:
            return motifs[str(key)]
        except KeyError as exc:
            available = ", ".join(motifs) or "<none>"
            raise KeyError(
                f"Unknown motif key {key!r}. Available keys: {available}"
            ) from exc

    def dictionary_path_frame(self) -> pd.DataFrame:
        """Return every evaluated dictionary prefix, including the null model."""

        self._check_fitted()
        assert self.dictionary_path_ is not None
        return self.dictionary_path_.copy()

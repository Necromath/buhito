"""Pure occurrence-ranking heuristics for the compression-agent pilot.

Each function receives immutable numeric features and returns an ascending
priority tuple. The evaluator appends deterministic rule and occurrence tie
breaks, packs disjoint occurrences, and scores the result with the unchanged
MDL accounting. These priorities are search heuristics, not codelengths.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

Features = Mapping[str, int | float]
Priority = tuple[int | float, ...]


def baseline(features: Features) -> Priority:
    """Reproduce the core boundary-first heuristic; MDL evaluates its result."""
    k = features["motif_nodes"]
    e = features["motif_edges"]
    b = features["boundary_edges"]
    return (b, -e, -k)


def largest_first(features: Features) -> Priority:
    """Prioritize larger contractions heuristically; MDL evaluates the result."""
    k = features["motif_nodes"]
    e = features["motif_edges"]
    b = features["boundary_edges"]
    return (-k, -e, b)


def edge_per_boundary(features: Features) -> Priority:
    """Rank a structural-yield proxy per boundary cost; MDL remains the score."""
    k = features["motif_nodes"]
    e = features["motif_edges"]
    b = features["boundary_edges"]
    return (-(e + k - 1) / (1 + b), b, -e, -k)


def conflict_aware(features: Features) -> Priority:
    """Favor contraction yield per competing occurrence; MDL scores the result."""
    k = features["motif_nodes"]
    e = features["motif_edges"]
    b = features["boundary_edges"]
    c = features["conflicts"]
    return (-(k - 1) / (1 + c), b, -e, -k)


CANDIDATES: dict[str, Callable[[Features], Priority]] = {
    "baseline": baseline,
    "largest_first": largest_first,
    "edge_per_boundary": edge_per_boundary,
    "conflict_aware": conflict_aware,
}

"""Bit-length primitives for Buhito's enumerative-v1 accountant.

These integer-valued description lengths are used for cross-dataset reporting.

They are intentionally distinct from the analytical MDL objective implemented
in :mod:`buhito.mdl`.
"""

from __future__ import annotations

from math import comb


CODEC_VERSION = "enumerative-v1"


def ceil_log2_int(value: int) -> int:
    """Return ceil(log2(value)) exactly for positive integer ``value``."""
    value = int(value)

    if value <= 0:
        raise ValueError("value must be positive")

    if value == 1:
        return 0

    return (value - 1).bit_length()


def positive_integer_bits(value: int) -> int:
    """Elias-delta code length for a positive integer."""
    value = int(value)

    if value <= 0:
        raise ValueError("value must be positive")

    length = value.bit_length()

    return length + 2 * length.bit_length() - 2


def nonnegative_integer_bits(value: int) -> int:
    """Elias-delta code length for a nonnegative integer via ``value + 1``."""
    value = int(value)

    if value < 0:
        raise ValueError("value must be nonnegative")

    return positive_integer_bits(value + 1)


def symbol_width(cardinality: int) -> int:
    """Fixed-width code length for one symbol from an alphabet."""
    cardinality = int(cardinality)

    if cardinality < 0:
        raise ValueError("cardinality must be nonnegative")

    if cardinality <= 1:
        return 0

    return ceil_log2_int(cardinality)


def subset_bits(universe: int, chosen: int) -> int:
    """Enumerative fixed-cardinality subset code.

    Returns::

        ceil(log2(binomial(universe, chosen)))

    using integer arithmetic.
    """
    universe = int(universe)
    chosen = int(chosen)

    if universe < 0:
        raise ValueError("universe must be nonnegative")

    if chosen < 0 or chosen > universe:
        raise ValueError(
            "chosen must satisfy 0 <= chosen <= universe"
        )

    return ceil_log2_int(comb(universe, chosen))


def simple_edge_universe(n_nodes: int) -> int:
    """Number of possible edges in an undirected loop-free simple graph."""
    n_nodes = int(n_nodes)

    if n_nodes < 0:
        raise ValueError("n_nodes must be nonnegative")

    return comb(n_nodes, 2)


def simple_topology_bits(n_nodes: int, n_edges: int) -> int:
    """Enumerative topology length for a labeled simple graph."""
    return subset_bits(
        simple_edge_universe(n_nodes),
        n_edges,
    )

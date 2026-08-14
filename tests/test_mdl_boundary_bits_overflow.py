"""Regression tests for the boundary-bits overflow fix in ``buhito.mdl``.

Covers the crash originally seen when a high-degree boundary motif produced
an astronomically large orbit-count "alphabet size" for
``_dirichlet_multinomial_bits``, plus the related silent-correctness bug
where the old implementation collapsed to a constant value for large (but
not yet overflowing) alphabet sizes.
"""

from __future__ import annotations

import math
from collections import Counter

import pytest

from buhito.mdl import _dirichlet_multinomial_bits


def test_small_alphabet_matches_known_value():
    """Sanity check against a hand-computable case; guards against silent
    regressions in the normal-sized-alphabet code path."""
    counts = Counter({"a": 5, "b": 3})
    bits = _dirichlet_multinomial_bits(counts, alphabet_size=4)
    assert math.isfinite(bits)
    assert bits == pytest.approx(12.678072, abs=1e-5)


def test_huge_alphabet_size_does_not_raise():
    """Reproduces the original OverflowError scenario: an alphabet size far
    beyond what a float64 can represent."""
    huge_alphabet_size = 3 ** 3000
    counts = Counter({"a": 7, "b": 3, "c": 1})
    bits = _dirichlet_multinomial_bits(counts, huge_alphabet_size)
    assert math.isfinite(bits)
    assert bits > 0


def test_monotonic_in_alphabet_size():
    """Boundary-bit cost must strictly increase with alphabet size. The
    pre-fix implementation silently flatlined to a constant value once the
    alphabet size exceeded ~1e18, well before it would actually overflow."""
    counts = Counter({"a": 5, "b": 2})
    previous = None
    for exponent in (1, 5, 10, 15, 20, 50, 100, 300, 1000, 100_000):
        bits = _dirichlet_multinomial_bits(counts, 10 ** exponent)
        assert math.isfinite(bits)
        if previous is not None:
            assert bits > previous, (
                f"boundary bits did not increase at alphabet size 10^{exponent}"
            )
        previous = bits


def test_zero_or_negative_alphabet_size_returns_zero():
    counts = Counter({"a": 1})
    assert _dirichlet_multinomial_bits(counts, alphabet_size=0) == 0.0
    assert _dirichlet_multinomial_bits(counts, alphabet_size=-5) == 0.0

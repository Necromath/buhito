import pytest

from buhito.bitcodes import (
    CODEC_VERSION,
    ceil_log2_int,
    nonnegative_integer_bits,
    positive_integer_bits,
    simple_edge_universe,
    simple_topology_bits,
    subset_bits,
    symbol_width,
)


def test_codec_version_is_explicit():
    assert CODEC_VERSION == "enumerative-v1"


def test_ceil_log2_integer_boundaries():
    assert ceil_log2_int(1) == 0
    assert ceil_log2_int(2) == 1
    assert ceil_log2_int(3) == 2
    assert ceil_log2_int(4) == 2
    assert ceil_log2_int(5) == 3
    assert ceil_log2_int(2**100) == 100
    assert ceil_log2_int(2**100 + 1) == 101


def test_positive_integer_code_matches_legacy_zinc_definition():
    expected = {
        1: 1,
        2: 4,
        3: 4,
        4: 5,
        5: 5,
        6: 5,
        7: 5,
        8: 8,
        15: 8,
        16: 9,
        32: 10,
        100: 11,
        1000: 16,
        10000: 20,
    }

    for value, bits in expected.items():
        assert positive_integer_bits(value) == bits


def test_nonnegative_integer_code_matches_legacy():
    expected = {
        0: 1,
        1: 4,
        2: 4,
        3: 5,
        4: 5,
        5: 5,
        6: 5,
        7: 8,
        8: 8,
        9: 8,
    }

    for value, bits in expected.items():
        assert nonnegative_integer_bits(value) == bits


def test_integer_code_validation():
    with pytest.raises(ValueError):
        positive_integer_bits(0)

    with pytest.raises(ValueError):
        nonnegative_integer_bits(-1)


def test_symbol_width():
    assert symbol_width(0) == 0
    assert symbol_width(1) == 0
    assert symbol_width(2) == 1
    assert symbol_width(3) == 2
    assert symbol_width(4) == 2
    assert symbol_width(5) == 3
    assert symbol_width(8) == 3
    assert symbol_width(9) == 4


def test_subset_code_matches_legacy_examples():
    assert subset_bits(3, 1) == 2
    assert subset_bits(6, 3) == 5
    assert subset_bits(10, 2) == 6
    assert subset_bits(10, 5) == 8
    assert subset_bits(45, 10) == 32
    assert subset_bits(100, 5) == 27
    assert subset_bits(1000, 10) == 78


def test_subset_boundary_cases():
    assert subset_bits(10, 0) == 0
    assert subset_bits(10, 10) == 0

    with pytest.raises(ValueError):
        subset_bits(10, 11)


def test_simple_edge_universe():
    assert simple_edge_universe(0) == 0
    assert simple_edge_universe(1) == 0
    assert simple_edge_universe(2) == 1
    assert simple_edge_universe(3) == 3
    assert simple_edge_universe(5) == 10


def test_simple_topology_bits():
    assert simple_topology_bits(3, 1) == 2
    assert simple_topology_bits(4, 3) == 5

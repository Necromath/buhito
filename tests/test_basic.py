"""Minimal installation and public API smoke tests."""

import buhito


def test_package_imports() -> None:
    assert buhito.__name__ == "buhito"


def test_core_public_api_is_available() -> None:
    assert callable(buhito.generate_subgraphs_breadthwise)
    assert callable(buhito.generate_subgraphs_depthwise)
    assert callable(buhito.MDLGraphCompressor)

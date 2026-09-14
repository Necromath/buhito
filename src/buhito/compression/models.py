"""Schemas and result objects shared by the compression stages.

These compatibility exports give the refactor one stable dependency direction:
recognition, dictionary selection, and substitution may depend on models, while
models must not depend on those stages.
"""

from ..mdl import (
    CompressionResult,
    DatasetCode,
    EncodedGraph,
    GraphSchema,
    MotifRule,
    Occurrence,
    Rewrite,
)

__all__ = [
    "CompressionResult",
    "DatasetCode",
    "EncodedGraph",
    "GraphSchema",
    "MotifRule",
    "Occurrence",
    "Rewrite",
]

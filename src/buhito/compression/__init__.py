"""Public, dataset-independent graph compression API.

The implementation is being migrated from :mod:`buhito.mdl` into the
recognize -> count/select -> substitute stages exposed here.  Imports from
`buhito.mdl` remain supported during that migration.
"""

from .dictionary import DictionarySelection
from .models import (
    CompressionResult,
    DatasetCode,
    EncodedGraph,
    GraphSchema,
    MotifRule,
    Occurrence,
    Rewrite,
)
from .pipeline import MDLGraphCompressor
from .recognition import (
    BuhitoGraphletEnumerator,
    ExhaustiveGraphletEnumerator,
    GraphletEnumerator,
)
from .substitution import decode_rewrite, labeled_isomorphic

__all__ = [
    "BuhitoGraphletEnumerator",
    "CompressionResult",
    "DatasetCode",
    "DictionarySelection",
    "EncodedGraph",
    "ExhaustiveGraphletEnumerator",
    "GraphSchema",
    "GraphletEnumerator",
    "MDLGraphCompressor",
    "MotifRule",
    "Occurrence",
    "Rewrite",
    "decode_rewrite",
    "labeled_isomorphic",
]

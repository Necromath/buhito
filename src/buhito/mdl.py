"""Backward-compatible imports for the compression package.

New code should import from :mod:`buhito.compression`.  This module remains so
existing experiments and serialized research scripts continue to run.
"""

from .compression.dictionary import DictionarySelection, Selector
from .compression.models import (
    _EDGE_LABEL as _EDGE_LABEL,
)
from .compression.models import (
    _NODE_KIND as _NODE_KIND,
)
from .compression.models import (
    _NODE_LABEL as _NODE_LABEL,
)
from .compression.models import (
    _PORTS as _PORTS,
)
from .compression.models import (
    Alphabets,
    CompressionResult,
    DatasetCode,
    EncodedGraph,
    GraphSchema,
    MotifRule,
    Occurrence,
    Rewrite,
    labeled_isomorphic,
)
from .compression.models import (
    _dirichlet_multinomial_bits as _dirichlet_multinomial_bits,
)
from .compression.pipeline import MDLGraphCompressor
from .compression.recognition import (
    BuhitoGraphletEnumerator,
    ExhaustiveGraphletEnumerator,
    GraphletEnumerator,
)
from .compression.recognition import (
    _make_motif as _make_motif,
)
from .compression.substitution import (
    _decode_rewrite_original_ids as _decode_rewrite_original_ids,
)
from .compression.substitution import (
    _rewrite_graph as _rewrite_graph,
)
from .compression.substitution import (
    _rewrite_validation_error as _rewrite_validation_error,
)
from .compression.substitution import (
    _validate_rewrite_exact as _validate_rewrite_exact,
)
from .compression.substitution import (
    decode_rewrite,
)

__all__ = [
    "Alphabets",
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
    "Selector",
    "decode_rewrite",
    "labeled_isomorphic",
]

"""High-level fitted compression pipeline.

`MDLGraphCompressor` already implements the leakage-safe estimator contract:
fit the dictionary on training graphs, then transform validation or test graphs
with that frozen dictionary.  It is re-exported here while its implementation
is split among the stage modules.
"""

from ..mdl import MDLGraphCompressor

__all__ = ["MDLGraphCompressor"]

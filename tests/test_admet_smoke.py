import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_example():
    path = Path(__file__).parents[1] / "examples" / "admet" / "admet_smoke.py"
    spec = importlib.util.spec_from_file_location("admet_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_admet_smoke_helpers_do_not_require_optional_dependencies():
    example = _load_example()
    assert example._is_binary_target(np.array([0, 1, 1]))
    assert not example._is_binary_target(np.array([0.1, 0.2, 0.3]))

    frame = pd.DataFrame({"Drug": list("abcdef"), "Y": range(6)})
    first = example._limited(frame, 3, seed=7)
    second = example._limited(frame, 3, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert len(example._limited(frame, 0, seed=7)) == len(frame)

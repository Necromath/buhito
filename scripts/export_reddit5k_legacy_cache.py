"""
One-time exporter for the exploratory REDDIT-5K notebook.

Run this INSIDE the old notebook's live kernel before closing it:

    %run -i export_reddit5k_legacy_cache.py

The script saves expensive fingerprint dictionaries and selected result tables
to artifacts/reddit5k_clean/legacy_export/.  The clean notebook can then import
them instead of recomputing them.

The old chain cache is deliberately exported but is not automatically reused by
the clean notebook because the exploratory benchmark used min_length=2, whereas
the corrected experiment uses min_length=3.
"""

from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd

EXPORT_DIR = Path("artifacts") / "reddit5k_clean" / "legacy_export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

required = ["versions", "balanced_ids"]
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(
        "Run this with `%run -i export_reddit5k_legacy_cache.py` in the OLD "
        f"notebook after the expensive cells. Missing variables: {missing}"
    )

_ids = [int(i) for i in balanced_ids]
if "labels" in globals():
    _labels = np.asarray(labels, dtype=int)
elif "y" in globals():
    _labels = np.asarray([int(y[i]) for i in _ids], dtype=int)
else:
    raise RuntimeError("Could not find `labels` or `y` in the old kernel.")

metadata = {
    "created_unix": time.time(),
    "balanced_ids": _ids,
    "labels": _labels.tolist(),
    "methods": sorted(versions.keys()),
    "notes": {
        "uncompressed": "degree-labeled Buhito depth-3 pilot cache",
        "leaf_bag": "min_bag_size=2",
        "slashburn": "num_hubs=5, max_component_size=10",
        "chain": "legacy min_length=2; do not reuse for corrected min_length=3",
        "comp_label": "num_hubs=5, no degree label",
    },
}

for method, payload in versions.items():
    target = EXPORT_DIR / f"{method}.joblib"
    print(f"Saving {method!r} -> {target}")
    joblib.dump(payload, target, compress=3)

(EXPORT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

for name in [
    "tradeoff_df",
    "f1_table",
    "leaf_mdl_df",
    "slash_mdl_df",
    "two_stage_mdl_df",
    "leaf_mdl_summary",
    "slash_mdl_summary",
    "two_stage_summary",
    "structural_mdl_comparison",
    "chain_df_t3",
    "chain_mdl_t3",
]:
    obj = globals().get(name)
    if isinstance(obj, pd.DataFrame):
        target = EXPORT_DIR / f"{name}.csv"
        obj.to_csv(target, index=True)
        print(f"Saved table {name!r} -> {target}")

print("\nLegacy export complete.")
print(f"Directory: {EXPORT_DIR.resolve()}")

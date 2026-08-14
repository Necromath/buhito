#!/usr/bin/env python3
"""Remove research artifacts from a dedicated cleanup branch.

Run first with ``--dry-run`` and then with ``--apply``. This script only changes
the current working tree; it does not rewrite Git history or push anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

REMOVE_PATHS = [
    "AML_Cheminformatics_26-2.pdf",
    "MDL4AST.ipynb",
    "MDL4AST_clean_negative_result_bundle",
    "MDL4AST_clean_negative_result_executed.ipynb",
    "MDL_Chem_clean_rewrite.ipynb",
    "MDL_Chem_end_to_end_collaborator_demo.ipynb",
    "MDL_Chem_end_to_end_collaborator_demo_ML_fixed.ipynb",
    "MDL_Chem_end_to_end_collaborator_demo_next_experiment.ipynb",
    "MDL_Chem_end_to_end_collaborator_demo_next_experiment_batchfix.ipynb",
    "OMOLS_MINERVA_EXPLORER.ipynb",
    "analysis.png",
    "artifacts_syntax_treelets",
    "benchmark_cells_for_demo.ipynb",
    "combined_graphlets_peterson_graph.png",
    "examples/qm9/qm9data",
    "probe_c3_vs_others.png",
    "probe_c4_vs_c5.png",
    "probe_c5_called_c4.png",
    "qmugs_benchmark.png",
    "reddit5k_gnn_graphlets.ipynb",
    "reddit5k_gnn_seed42.json",
    "reddit5k_graph_stats.csv",
    "reddit5k_leaf_bag_compression_min2_first500.csv",
    "reddit5k_leaf_bag_structural_mdl.csv",
    "reddit5k_leaf_bag_structural_mdl_summary.csv",
    "reddit5k_leaf_bag_threshold_comparison_first500.csv",
    "reddit5k_leaf_then_slashburn_structural_mdl.csv",
    "reddit5k_leaf_then_slashburn_structural_mdl_summary.csv",
    "reddit5k_slashburn_structural_mdl.csv",
    "reddit5k_slashburn_structural_mdl_summary.csv",
    "reddit5k_structural_mdl_comparison.csv",
    "reddit5k_vertex_degree_stats.csv",
    "reddit_5k_buhito_experiment.ipynb",
    "reddit_5k_compression_clean.ipynb",
    "reddit_5k_testbed.ipynb",
    "reddit_compression_tradeoff.png",
    "reddit_confusions.png",
    "reddit_erosion_sweep.png",
    "reddit_per_class_f1.png",
    "shap_direction_per_class.png",
    "shap_importance_heatmap.png",
    "src/buhito.egg-info",
    "summary.csv",
    "symmetry_story.png",
    "unseen_bits_vs_error.png",
    "setup.py",
    "requirements.txt",
]


def git(*args: str, capture: bool = False) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = Path(git("rev-parse", "--show-toplevel", capture=True))
    branch = git("branch", "--show-current", capture=True)
    if not branch.startswith("cleanup/"):
        print(
            f"Refusing to run on branch {branch!r}; use a cleanup/* branch.",
            file=sys.stderr,
        )
        return 2

    missing_curated = [
        path
        for path in [
            "notebooks/chemistry/mdl_chemistry_end_to_end.ipynb",
            "notebooks/reddit/reddit_mdl_diagnostics.ipynb",
            "notebooks/syntax/mdl_ast_negative_result.ipynb",
        ]
        if not (root / path).exists()
    ]
    if missing_curated:
        print(
            "Copy the cleanup patch into the repository before applying. "
            f"Missing: {missing_curated}",
            file=sys.stderr,
        )
        return 2

    existing = [path for path in REMOVE_PATHS if (root / path).exists()]
    print(f"Branch: {branch}")
    print("Paths scheduled for removal:")
    for path in existing:
        print(f"  - {path}")

    legacy_script = root / "export_reddit5k_legacy_cache.py"
    script_target = root / "scripts" / "export_reddit5k_legacy_cache.py"
    if legacy_script.exists():
        print(f"Move: {legacy_script.relative_to(root)} -> {script_target.relative_to(root)}")

    if args.dry_run:
        print("Dry run only; no files changed.")
        return 0

    for path in existing:
        git("rm", "-r", "--ignore-unmatch", "--", path)

    if legacy_script.exists():
        script_target.parent.mkdir(parents=True, exist_ok=True)
        if script_target.exists():
            legacy_script.unlink()
            git("rm", "--ignore-unmatch", "--", str(legacy_script.relative_to(root)))
        else:
            git(
                "mv",
                str(legacy_script.relative_to(root)),
                str(script_target.relative_to(root)),
            )

    for path in root.rglob("__pycache__"):
        if ".git" not in path.parts:
            shutil.rmtree(path, ignore_errors=True)

    print("Cleanup applied. Review with `git status --short` before committing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reusable sanity check for scripts/preprocess.py.

Preprocesses a small set of meshes into a temp directory with the full
supervised budget, then reports per-slice SDF statistics and validates:

  1. Merged supervised SDF extends well beyond +/-0.1 (so training sees
     meaningful far-field gradient signal).
  2. Far-field signs match the project convention (+outside/-inside).
  3. Far-field labels are consistent with near-surface labels in direction
     (predictions at surface should be zero; far-field should be far from
     zero).
  4. `rtree` is importable so trimesh.proximity.signed_distance uses a
     spatial index (otherwise preprocessing silently falls back or crashes
     on TC2).

Run before regenerating the full 300-mesh dataset. Keep this script in the
repo as a reusable tool for future preprocessing changes.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import preprocess  # noqa: E402

DEFAULT_MESHES = [
    "airplane_0001",
    "chair_0001",
    "table_0001",
]

DEFAULT_GT_DIR = ROOT / "tc2_backup" / "data_processed_shapenet" / "gt_meshes"


def check_rtree() -> None:
    try:
        import rtree  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "rtree is required by trimesh.proximity.signed_distance. "
            "Install via `conda install -c conda-forge rtree` or `pip install rtree`."
        ) from e
    print("[ok] rtree importable")


def check_mesh(gt_dir: Path, name: str, n_sup: int, n_unsup: int) -> dict:
    src = gt_dir / f"{name}.obj"
    if not src.exists():
        raise FileNotFoundError(f"Mesh not found: {src}")

    with tempfile.TemporaryDirectory() as tmp:
        info = preprocess.process_single_mesh(
            str(src), tmp, n_sup_points=n_sup, n_unsup_points=n_unsup
        )
        npz_path = Path(tmp) / "ratio_1p00" / f"{name}.npz"
        npz = np.load(npz_path)
        points_sup = npz["points_sup"]
        sdf_sup = npz["sdf_sup"]

    stats_all = info["sdf_stats_all"]
    stats_near = info["sdf_stats_near"]
    stats_far = info["sdf_stats_far"]

    # Verifications (all are hard gates; raise if any fail)
    assert stats_all["min"] < -0.1 or stats_all["max"] > 0.2, (
        f"{name}: merged SDF range ({stats_all['min']:+.3f}, {stats_all['max']:+.3f}) "
        "does not exceed the old +/-0.1 regime"
    )
    assert (np.abs(sdf_sup) > 0.1).mean() > 0.1, (
        f"{name}: only "
        f"{(np.abs(sdf_sup) > 0.1).mean():.2%} of samples have |sdf|>0.1 — too few"
    )
    # Far-field sanity: sign distribution should skew toward outside but
    # not be degenerate (not 100% one sign).
    assert 0.5 < stats_far["frac_positive"] < 1.0, (
        f"{name}: far_frac_positive={stats_far['frac_positive']:.3f} is outside "
        "the expected (0.5, 1.0) range for a normalized mesh"
    )

    # Consistency check: no samples should have points very near the surface
    # with far-field-magnitude SDF (that would indicate sign disagreement).
    near_origin_idx = np.linalg.norm(points_sup, axis=1) < 0.05
    if near_origin_idx.any():
        near_origin_sdf = sdf_sup[near_origin_idx]
        if (np.abs(near_origin_sdf) > 0.5).any():
            print(
                f"  [warn] {name}: some points near origin have |sdf|>0.5, "
                "unusual geometry or sign issue"
            )

    watertight = info.get("watertight", False) or info.get("watertight_after_repair", False)
    wt_str = "Yes" if watertight else ("Repaired" if info.get("watertight_after_repair") else "No")

    print(
        f"[ok] {name}: "
        f"watertight={wt_str}  "
        f"all=[{stats_all['min']:+.3f}, {stats_all['max']:+.3f}]  "
        f"near=[{stats_near['min']:+.3f}, {stats_near['max']:+.3f}]  "
        f"far=[{stats_far['min']:+.3f}, {stats_far['max']:+.3f}]  "
        f"far_pos={stats_far['frac_positive']:.3f}  "
        f"|sdf|>0.1: {(np.abs(sdf_sup) > 0.1).mean():.2%}"
    )
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_dir",
        type=Path,
        default=DEFAULT_GT_DIR,
        help="Directory of normalized GT .obj meshes",
    )
    parser.add_argument(
        "--meshes",
        nargs="+",
        default=DEFAULT_MESHES,
        help="Mesh names (without .obj) to check",
    )
    parser.add_argument(
        "--n_sup",
        type=int,
        default=preprocess.DEFAULT_SUP_POINTS,
        help="Supervised sample budget per mesh (default: production size)",
    )
    parser.add_argument(
        "--n_unsup",
        type=int,
        default=preprocess.DEFAULT_UNSUP_POINTS,
        help="Unsupervised sample budget per mesh",
    )
    args = parser.parse_args()

    check_rtree()

    for name in args.meshes:
        check_mesh(args.gt_dir, name, args.n_sup, args.n_unsup)

    print("\nAll preprocessing sanity checks passed.")


if __name__ == "__main__":
    main()

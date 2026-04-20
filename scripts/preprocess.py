"""
SDF data preprocessing pipeline.

Converts raw mesh files into .npz files containing supervised SDF samples
(surface points + multi-scale offsets) and unsupervised points (uniform in unit sphere).

Designed for easy data source swap: only --mesh_dir changes when switching
from Thingi10K to ShapeNet.

Usage:
    python scripts/preprocess.py --mesh_dir data/raw --output_dir data/processed
    python scripts/preprocess.py --mesh_dir /path/to/shapenet/category --output_dir data/processed
"""

import argparse
import glob
import os
import json
from multiprocessing import Pool
from functools import partial

import numpy as np
import trimesh
from trimesh.proximity import signed_distance


# Multi-scale offset parameters
EPSILON_SCALES = [0.005, 0.01, 0.05]
OFFSET_MULTIPLIERS = [-2, -1, 1, 2]

# Supervision ratio splits
SUPERVISION_RATIOS = [1.0, 0.5, 0.1, 0.05]

# Default sample counts
DEFAULT_SUP_POINTS = 250_000
DEFAULT_UNSUP_POINTS = 250_000

# Fraction of the supervised budget spent on far-field (uniform-in-sphere)
# samples with exact point-to-mesh signed distance. The rest stays on the
# near-surface multi-scale offsets. Far-field supervision is what prevents
# the decoder from collapsing to SDF~=0 everywhere.
FAR_FIELD_FRACTION = 0.2

# Meshes with more than this many faces are decimated before signed_distance
# queries. trimesh.proximity.signed_distance is O(F*N) in the worst case; some
# ShapeNet meshes have >100K faces, which makes uniform-in-sphere queries
# take minutes per mesh. Decimating to ~10K faces brings a ~10x speedup with
# <1% sign-flip rate at |sdf|>0.1 (measured on airplane_0000, 101k faces).
SDF_DECIMATE_THRESHOLD = 15_000
SDF_DECIMATE_TARGET = 10_000


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalize mesh to fit inside unit sphere (centered at origin)."""
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid
    max_dist = np.max(np.linalg.norm(mesh.vertices, axis=1))
    if max_dist > 0:
        mesh.vertices /= max_dist
    return mesh


def sample_surface_points(mesh: trimesh.Trimesh, n_points: int):
    """Sample points on mesh surface with face normals.

    Returns:
        points: (n_points, 3) surface points
        normals: (n_points, 3) face normals at sampled points
    """
    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
    normals = mesh.face_normals[face_indices]
    return np.array(points, dtype=np.float32), np.array(normals, dtype=np.float32)


def generate_multiscale_sdf_samples(
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    n_total: int,
):
    """Generate multi-scale offset SDF samples from surface points.

    For each surface point, create offset points at multiple epsilon scales
    along the surface normal. SDF value approximated as j * epsilon.

    Args:
        surface_points: (N, 3) points on mesh surface
        surface_normals: (N, 3) normals at surface points
        n_total: total number of supervised samples to generate

    Returns:
        points: (n_total, 3) offset points
        sdf_values: (n_total,) approximate SDF values
    """
    n_scales = len(EPSILON_SCALES)
    n_offsets = len(OFFSET_MULTIPLIERS)
    n_combos = n_scales * n_offsets  # 12 combinations per surface point

    # How many surface points we need to base offsets on
    n_base = n_total // n_combos
    # Randomly select base surface points (with replacement if needed)
    indices = np.random.choice(len(surface_points), size=n_base, replace=True)
    base_points = surface_points[indices]  # (n_base, 3)
    base_normals = surface_normals[indices]  # (n_base, 3)

    all_points = []
    all_sdf = []

    for eps in EPSILON_SCALES:
        for j in OFFSET_MULTIPLIERS:
            offset = j * eps
            offset_points = base_points + offset * base_normals
            sdf_values = np.full(n_base, offset, dtype=np.float32)

            all_points.append(offset_points)
            all_sdf.append(sdf_values)

    all_points = np.concatenate(all_points, axis=0)  # (n_base * 12, 3)
    all_sdf = np.concatenate(all_sdf, axis=0)  # (n_base * 12,)

    # Trim or pad to exact n_total
    if len(all_points) > n_total:
        perm = np.random.permutation(len(all_points))[:n_total]
        all_points = all_points[perm]
        all_sdf = all_sdf[perm]
    elif len(all_points) < n_total:
        # Pad with additional random offset samples
        deficit = n_total - len(all_points)
        extra_idx = np.random.choice(len(surface_points), size=deficit, replace=True)
        extra_eps = np.random.choice(EPSILON_SCALES, size=deficit)
        extra_j = np.random.choice(OFFSET_MULTIPLIERS, size=deficit)
        extra_offset = extra_j * extra_eps
        extra_points = surface_points[extra_idx] + (extra_offset[:, None] * surface_normals[extra_idx])
        extra_sdf = extra_offset.astype(np.float32)
        all_points = np.concatenate([all_points, extra_points], axis=0)
        all_sdf = np.concatenate([all_sdf, extra_sdf], axis=0)

    # Shuffle
    perm = np.random.permutation(len(all_points))
    return all_points[perm].astype(np.float32), all_sdf[perm].astype(np.float32)


def sample_unsupervised_points(n_points: int):
    """Sample points uniformly in the unit sphere.

    These points have no GT SDF — used for Eikonal regularization only.

    Returns:
        points: (n_points, 3)
    """
    # Rejection sampling for uniform distribution in unit sphere
    points = []
    total = 0
    while total < n_points:
        batch = np.random.uniform(-1, 1, size=(n_points * 2, 3))
        norms = np.linalg.norm(batch, axis=1)
        inside = batch[norms <= 1.0]
        points.append(inside)
        total += len(inside)
    points = np.concatenate(points, axis=0)[:n_points]
    return points.astype(np.float32)


def process_single_mesh(
    mesh_path: str,
    output_dir: str,
    n_sup_points: int = DEFAULT_SUP_POINTS,
    n_unsup_points: int = DEFAULT_UNSUP_POINTS,
    seed: int | None = None,
) -> dict:
    """Process a single mesh file into .npz files at all supervision ratios.

    If ``seed`` is provided, set numpy's global RNG so that the outputs are
    reproducible under parallel workers. When not given, uses whatever global
    seeding the caller established. Deterministic seeding is derived from the
    mesh name so each mesh's samples are stable regardless of worker order.

    Supervised samples are a union of near-surface multi-scale offsets (fine
    detail) and far-field uniform-in-sphere samples with exact signed
    distances (prevents the decoder from collapsing to SDF~=0 everywhere).

    Sign convention in the saved npz: positive = outside, negative = inside.
    trimesh.proximity.signed_distance uses the opposite sign (+inside/-outside),
    so far-field distances are negated before concatenation.

    Args:
        mesh_path: path to input mesh file
        output_dir: base output directory
        n_sup_points: number of supervised SDF samples at 100%
        n_unsup_points: number of unsupervised points

    Returns:
        dict with processing info (name, watertight status, sample counts, sdf stats)
    """
    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
    info = {"name": mesh_name, "path": mesh_path}

    # Per-mesh deterministic seed so parallel workers produce reproducible output.
    if seed is not None:
        per_mesh_seed = (seed * 1_000_003 + (hash(mesh_name) & 0x7FFFFFFF)) & 0x7FFFFFFF
        np.random.seed(per_mesh_seed)

    # Load mesh
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
    except Exception as e:
        info["error"] = str(e)
        print(f"  ERROR loading {mesh_name}: {e}")
        return info

    # Normalize
    mesh = normalize_mesh(mesh)

    # Watertight check
    info["watertight"] = mesh.is_watertight
    if not mesh.is_watertight:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
        info["watertight_after_repair"] = mesh.is_watertight

    info["n_vertices"] = len(mesh.vertices)
    info["n_faces"] = len(mesh.faces)

    # Split budget: near-surface multi-scale + far-field signed distance
    n_far = int(round(n_sup_points * FAR_FIELD_FRACTION))
    n_near = n_sup_points - n_far

    # Sample surface points (for near-surface offsets)
    surface_points, surface_normals = sample_surface_points(mesh, n_near)

    # Near-surface multi-scale offsets (positive = outside by construction)
    near_points, near_sdf = generate_multiscale_sdf_samples(
        surface_points, surface_normals, n_near
    )

    # Far-field uniform-in-sphere + signed distance.
    # trimesh returns +inside/-outside; project convention is +outside/-inside,
    # so negate.
    #
    # Decimate heavy meshes before the signed_distance call; exact SDF on
    # 100k-face meshes takes minutes per mesh and would blow the TC2 budget.
    # At 10k-face decimation, sign-flip rate is <1% for points with |sdf|>0.1.
    far_points = sample_unsupervised_points(n_far)
    sdf_mesh = mesh
    info["sdf_mesh_decimated"] = False
    if len(mesh.faces) > SDF_DECIMATE_THRESHOLD:
        try:
            sdf_mesh = mesh.simplify_quadric_decimation(face_count=SDF_DECIMATE_TARGET)
            info["sdf_mesh_decimated"] = True
        except Exception as e:
            info["sdf_mesh_decimate_error"] = str(e)
    info["sdf_mesh_faces"] = int(len(sdf_mesh.faces))
    far_sdf = (-signed_distance(sdf_mesh, far_points)).astype(np.float32)

    # Merge and shuffle so batches mix near and far
    sup_points = np.concatenate([near_points, far_points], axis=0)
    sup_sdf = np.concatenate([near_sdf, far_sdf], axis=0)
    perm = np.random.permutation(len(sup_points))
    sup_points = sup_points[perm].astype(np.float32)
    sup_sdf = sup_sdf[perm].astype(np.float32)

    # Audit stats — per-slice so the bug that motivated this rewrite is
    # easy to spot in logs if it ever regresses
    info["n_near"] = int(n_near)
    info["n_far"] = int(n_far)
    info["far_backend"] = "trimesh.proximity.signed_distance"
    info["far_sign_inverted"] = True
    info["sdf_stats_all"] = {
        "min": float(sup_sdf.min()),
        "max": float(sup_sdf.max()),
        "mean_abs": float(np.abs(sup_sdf).mean()),
    }
    info["sdf_stats_near"] = {
        "min": float(near_sdf.min()),
        "max": float(near_sdf.max()),
        "mean_abs": float(np.abs(near_sdf).mean()),
    }
    info["sdf_stats_far"] = {
        "min": float(far_sdf.min()),
        "max": float(far_sdf.max()),
        "mean_abs": float(np.abs(far_sdf).mean()),
        "frac_positive": float((far_sdf > 0).mean()),
    }

    # Generate unsupervised points (Eikonal)
    unsup_points = sample_unsupervised_points(n_unsup_points)

    # Save GT mesh (normalized) for evaluation
    gt_dir = os.path.join(output_dir, "gt_meshes")
    os.makedirs(gt_dir, exist_ok=True)
    mesh.export(os.path.join(gt_dir, f"{mesh_name}.obj"))

    # Save at each supervision ratio
    for ratio in SUPERVISION_RATIOS:
        ratio_str = f"ratio_{ratio:.2f}".replace(".", "p")
        ratio_dir = os.path.join(output_dir, ratio_str)
        os.makedirs(ratio_dir, exist_ok=True)

        # Subsample supervised points
        n_sup = int(n_sup_points * ratio)
        indices = np.random.choice(len(sup_points), size=n_sup, replace=False)
        ratio_sup_points = sup_points[indices]
        ratio_sup_sdf = sup_sdf[indices]

        # Save .npz
        npz_path = os.path.join(ratio_dir, f"{mesh_name}.npz")
        np.savez_compressed(
            npz_path,
            points_sup=ratio_sup_points,
            sdf_sup=ratio_sup_sdf,
            points_unsup=unsup_points,
        )

        info[f"n_sup_{ratio_str}"] = n_sup

    info["n_unsup"] = n_unsup_points
    return info


def write_data_audit(audit_results: list, output_path: str):
    """Write data audit summary as markdown."""
    total = len(audit_results)
    watertight = sum(1 for r in audit_results if r.get("watertight", False)
                     or r.get("watertight_after_repair", False))
    non_watertight_after_repair = sum(
        1 for r in audit_results
        if r.get("watertight") is False
        and r.get("watertight_after_repair") is False
    )
    errors = sum(1 for r in audit_results if "error" in r)

    # Aggregate SDF stats (only over shapes with far-field samples)
    shapes_with_stats = [r for r in audit_results if "sdf_stats_all" in r]
    if shapes_with_stats:
        all_mins = [r["sdf_stats_all"]["min"] for r in shapes_with_stats]
        all_maxs = [r["sdf_stats_all"]["max"] for r in shapes_with_stats]
        all_abs_means = [r["sdf_stats_all"]["mean_abs"] for r in shapes_with_stats]
        far_frac_pos = [r["sdf_stats_far"]["frac_positive"] for r in shapes_with_stats]

    with open(output_path, "w") as f:
        f.write("# Data Preprocessing Audit\n\n")
        f.write(f"- **Total meshes processed:** {total}\n")
        f.write(f"- **Watertight (original or repaired):** {watertight}\n")
        f.write(f"- **Non-watertight after repair:** {non_watertight_after_repair}\n")
        f.write(f"- **Errors:** {errors}\n")
        f.write(f"- **Supervision ratios:** {SUPERVISION_RATIOS}\n")
        f.write(f"- **Epsilon scales:** {EPSILON_SCALES}\n")
        f.write(f"- **Offset multipliers:** {OFFSET_MULTIPLIERS}\n")
        f.write(f"- **Far-field fraction:** {FAR_FIELD_FRACTION}\n")
        f.write("- **Far-field backend:** trimesh.proximity.signed_distance "
                "(negated to match +outside/-inside convention)\n\n")

        if shapes_with_stats:
            f.write("## Supervised SDF Statistics (100% split, aggregate across shapes)\n\n")
            f.write(f"- min over shapes: {min(all_mins):+.4f}\n")
            f.write(f"- max over shapes: {max(all_maxs):+.4f}\n")
            f.write(f"- mean |sdf| range: [{min(all_abs_means):.4f}, {max(all_abs_means):.4f}]\n")
            f.write(f"- far-field positive fraction range: "
                    f"[{min(far_frac_pos):.3f}, {max(far_frac_pos):.3f}]\n\n")

        f.write("## Per-Mesh Details\n\n")
        f.write("| Mesh | Vertices | Faces | Watertight | Sup Points (100%) "
                "| Unsup Points | SDF min | SDF max | Far +% |\n")
        f.write("|------|----------|-------|------------|--------------------"
                "|--------------|---------|---------|--------|\n")

        for r in audit_results:
            if "error" in r:
                f.write(f"| {r['name']} | ERROR | - | - | - | - | - | - | - |\n")
                continue

            wt = r.get("watertight", False) or r.get("watertight_after_repair", False)
            wt_str = "Yes" if wt else "No"
            if not r.get("watertight") and r.get("watertight_after_repair"):
                wt_str = "Repaired"

            n_sup = r.get("n_sup_ratio_1p00", "?")
            n_unsup = r.get("n_unsup", "?")
            stats = r.get("sdf_stats_all", {})
            far_stats = r.get("sdf_stats_far", {})
            sdf_min = f"{stats.get('min', 0):+.3f}" if stats else "?"
            sdf_max = f"{stats.get('max', 0):+.3f}" if stats else "?"
            far_pos = f"{far_stats.get('frac_positive', 0):.2f}" if far_stats else "?"
            f.write(f"| {r['name']} | {r['n_vertices']} | {r['n_faces']} "
                    f"| {wt_str} | {n_sup} | {n_unsup} "
                    f"| {sdf_min} | {sdf_max} | {far_pos} |\n")

    print(f"\nAudit written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess meshes into SDF samples")
    parser.add_argument("--mesh_dir", type=str, required=True,
                        help="Directory containing input mesh files (OBJ/STL/OFF)")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Directory for output .npz files")
    parser.add_argument("--n_sup_points", type=int, default=DEFAULT_SUP_POINTS,
                        help="Number of supervised SDF samples at 100%%")
    parser.add_argument("--n_unsup_points", type=int, default=DEFAULT_UNSUP_POINTS,
                        help="Number of unsupervised points per shape")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Number of parallel worker processes (default 1 = serial)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip meshes that already have all ratio npz files")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all mesh files
    extensions = ["*.obj", "*.stl", "*.off", "*.ply"]
    mesh_files = []
    for ext in extensions:
        mesh_files.extend(glob.glob(os.path.join(args.mesh_dir, ext)))
    mesh_files.sort()

    # Optionally skip meshes that already have all SUPERVISION_RATIOS outputs
    if args.skip_existing:
        def all_ratios_exist(mp):
            name = os.path.splitext(os.path.basename(mp))[0]
            for ratio in SUPERVISION_RATIOS:
                ratio_str = f"ratio_{ratio:.2f}".replace(".", "p")
                if not os.path.exists(os.path.join(args.output_dir, ratio_str, f"{name}.npz")):
                    return False
            return True
        before = len(mesh_files)
        mesh_files = [mp for mp in mesh_files if not all_ratios_exist(mp)]
        print(f"skip_existing: {before - len(mesh_files)} meshes already processed, "
              f"{len(mesh_files)} remaining")

    if not mesh_files:
        print(f"No mesh files found in {args.mesh_dir}")
        print(f"Supported formats: {extensions}")
        return

    print(f"Found {len(mesh_files)} mesh files in {args.mesh_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Supervised points: {args.n_sup_points}, Unsupervised points: {args.n_unsup_points}")
    print(f"Supervision ratios: {SUPERVISION_RATIOS}")
    print(f"Epsilon scales: {EPSILON_SCALES}, Offsets: {OFFSET_MULTIPLIERS}\n")

    worker = partial(
        process_single_mesh,
        output_dir=args.output_dir,
        n_sup_points=args.n_sup_points,
        n_unsup_points=args.n_unsup_points,
        seed=args.seed,
    )

    audit_results = []
    if args.n_workers > 1:
        print(f"Using {args.n_workers} worker processes")
        with Pool(args.n_workers) as pool:
            for i, info in enumerate(pool.imap_unordered(worker, mesh_files)):
                audit_results.append(info)
                print(f"[{i+1}/{len(mesh_files)}] done: {info.get('name', '?')}", flush=True)
    else:
        for i, mesh_path in enumerate(mesh_files):
            print(f"[{i+1}/{len(mesh_files)}] Processing {os.path.basename(mesh_path)}...",
                  flush=True)
            audit_results.append(worker(mesh_path))

    # Write audit
    audit_path = os.path.join(args.output_dir, "data_audit.md")
    write_data_audit(audit_results, audit_path)

    # Also save audit as JSON for programmatic access
    audit_json_path = os.path.join(args.output_dir, "data_audit.json")
    with open(audit_json_path, "w") as f:
        json.dump(audit_results, f, indent=2)

    # Print summary of output structure
    print(f"\nOutput structure:")
    for ratio in SUPERVISION_RATIOS:
        ratio_str = f"ratio_{ratio:.2f}".replace(".", "p")
        ratio_dir = os.path.join(args.output_dir, ratio_str)
        n_files = len(glob.glob(os.path.join(ratio_dir, "*.npz")))
        print(f"  {ratio_str}/: {n_files} .npz files")
    print(f"  gt_meshes/: normalized GT meshes for evaluation")


if __name__ == "__main__":
    main()

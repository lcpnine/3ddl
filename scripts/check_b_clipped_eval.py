"""
Check B: Sphere-clipped evaluation on EXP-09/seed42.

Tests hypothesis #3: PE fails because eval grid corners (r=1.73) are OOD
relative to training distribution (all points at r <= 1.0).

Runs evaluation twice on the same checkpoint:
  - Standard: full [-1,1]^3 grid  (baseline, should match results.json)
  - Clipped:  sphere mask r<=1.0  (SDF set to +1.0 outside sphere before MC)

Usage:
    python scripts/check_b_clipped_eval.py \
        --exp_dir experiments/EXP-09/seed42 \
        --data_dir data/processed_shapenet \
        --n_shapes 30
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import trimesh
import yaml
from scipy.spatial import KDTree
from skimage import measure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluate import (
    load_model_and_config,
    chamfer_distance,
    normal_consistency,
)


def extract_mesh_clipped(model, latent_code, resolution=128, clip_sphere=False,
                          batch_size=65536, device=torch.device("cpu")):
    """Marching cubes with optional sphere clipping.

    clip_sphere=True: set SDF=+1.0 for all grid points with r > 1.0 before MC.
    This simulates what the model would see if it had never been queried OOD.
    """
    coords = np.linspace(-1.0, 1.0, resolution)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
    grid_points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    radii = np.linalg.norm(grid_points, axis=1)
    in_sphere = radii <= 1.0

    sdf_values = np.full(len(grid_points), 1.0, dtype=np.float32)

    # Only query model for in-sphere points (or all points if unclipped)
    query_mask = in_sphere if clip_sphere else np.ones(len(grid_points), dtype=bool)
    query_pts = grid_points[query_mask]

    grid_tensor = torch.from_numpy(query_pts).float().to(device)
    preds = np.zeros(len(query_pts), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(grid_tensor), batch_size):
            bp = grid_tensor[i:i + batch_size]
            zb = latent_code.expand(len(bp), -1)
            preds[i:i + len(bp)] = model(zb, bp).squeeze(-1).cpu().numpy()

    sdf_values[query_mask] = preds
    sdf_values = sdf_values.reshape(resolution, resolution, resolution)

    try:
        verts, faces, normals, _ = measure.marching_cubes(
            sdf_values, level=0.0, spacing=(2.0 / resolution,) * 3
        )
        verts = verts - 1.0
        return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    except Exception as e:
        return None


def run_check_b(exp_dir, data_dir, n_shapes=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, latent_codes, config = load_model_and_config(exp_dir, device)
    use_pe = config.get("use_pe", False)
    pe_levels = config.get("pe_levels", 6)
    print(f"Model: use_pe={use_pe}, pe_levels={pe_levels}")

    supervision_ratio = config.get("supervision_ratio", 1.0)
    ratio_str = f"ratio_{supervision_ratio:.2f}".replace(".", "p")
    ratio_dir = os.path.join(data_dir, ratio_str)
    npz_files = sorted([f for f in os.listdir(ratio_dir) if f.endswith(".npz")])
    shape_names = [os.path.splitext(f)[0] for f in npz_files][:n_shapes]

    gt_mesh_dir = config.get("gt_mesh_dir", os.path.join(data_dir, "gt_meshes"))
    resolution = 128  # use 128 for speed

    results_std = []
    results_clip = []

    for idx, shape_name in enumerate(shape_names):
        gt_path = os.path.join(gt_mesh_dir, f"{shape_name}.obj")
        if not os.path.exists(gt_path):
            continue
        gt_mesh = trimesh.load(gt_path, force="mesh")

        idx_t = torch.tensor([idx], device=device)
        with torch.no_grad():
            z = latent_codes(idx_t)

        print(f"[{idx+1}/{len(shape_names)}] {shape_name}", end="  ")

        # Standard (unclipped)
        mesh_std = extract_mesh_clipped(model, z, resolution=resolution,
                                        clip_sphere=False, device=device)
        cd_std = chamfer_distance(mesh_std, gt_mesh) if mesh_std else None
        nc_std = normal_consistency(mesh_std, gt_mesh) if mesh_std else None

        # Clipped (sphere mask)
        mesh_clip = extract_mesh_clipped(model, z, resolution=resolution,
                                         clip_sphere=True, device=device)
        cd_clip = chamfer_distance(mesh_clip, gt_mesh) if mesh_clip else None
        nc_clip = normal_consistency(mesh_clip, gt_mesh) if mesh_clip else None

        std_str  = f"CD={cd_std:.4f} NC={nc_std:.4f}" if cd_std else "FAILED"
        clip_str = f"CD={cd_clip:.4f} NC={nc_clip:.4f}" if cd_clip else "FAILED"
        print(f"std=[{std_str}]  clip=[{clip_str}]")

        if cd_std:
            results_std.append((cd_std, nc_std))
        if cd_clip:
            results_clip.append((cd_clip, nc_clip))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if results_std:
        cd_std_mean = np.mean([r[0] for r in results_std])
        nc_std_mean = np.mean([r[1] for r in results_std])
        print(f"Standard  (full cube):  CD={cd_std_mean:.4f}  NC={nc_std_mean:.4f}  "
              f"(n={len(results_std)})")

    if results_clip:
        cd_clip_mean = np.mean([r[0] for r in results_clip])
        nc_clip_mean = np.mean([r[1] for r in results_clip])
        print(f"Clipped   (r<=1 sphere): CD={cd_clip_mean:.4f}  NC={nc_clip_mean:.4f}  "
              f"(n={len(results_clip)})")

    if results_std and results_clip:
        cd_ratio = cd_clip_mean / cd_std_mean
        print(f"\nCD ratio (clipped/standard): {cd_ratio:.3f}")
        if cd_ratio < 0.7:
            print("=> CONFIRMED: Sphere clipping substantially improves PE reconstruction.")
            print("   Root cause: training/inference distribution mismatch (hypothesis #3).")
        elif cd_ratio < 0.9:
            print("=> PARTIAL: Some improvement from clipping, but not the sole cause.")
        else:
            print("=> INCONCLUSIVE: Clipping does not substantially improve reconstruction.")
            print("   PE failure may be due to frequency/formula issues (#2/#4).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="experiments/EXP-09/seed42")
    parser.add_argument("--data_dir", default="data/processed_shapenet")
    parser.add_argument("--n_shapes", type=int, default=30,
                        help="Number of shapes to evaluate (default 30 for speed)")
    args = parser.parse_args()
    run_check_b(args.exp_dir, args.data_dir, args.n_shapes)

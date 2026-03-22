"""
Evaluation script for DeepSDF: mesh extraction + metric computation.

Metrics:
  - Chamfer Distance (CD) @ 30k points
  - IoU @ 128^3 and 256^3
  - Normal Consistency @ 30k points

Usage:
    python src/evaluate.py --exp_dir experiments/EXP-01/seed42 \
        --data_dir data/processed --output results.json
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


def load_model_and_config(exp_dir: str, device: torch.device):
    """Load trained model from best checkpoint."""
    # Import model classes
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from model import DeepSDF, LatentCodes

    # Load config
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Reconstruct model
    model = DeepSDF(
        latent_dim=config.get("latent_dim", 256),
        hidden_dim=config.get("hidden_dim", 512),
        num_layers=config.get("num_layers", 8),
        skip_layer=config.get("skip_layer", 4),
        use_pe=config.get("use_pe", False),
        pe_levels=config.get("pe_levels", 6),
    ).to(device)

    # Load checkpoint
    ckpt_path = os.path.join(exp_dir, "checkpoints", "best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(exp_dir, "checkpoints", "latest.pt")
    print(f"Loading checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load latent codes
    latent_state = checkpoint["latent_codes_state_dict"]
    num_shapes = latent_state["embedding.weight"].shape[0]
    latent_codes = LatentCodes(num_shapes, config.get("latent_dim", 256)).to(device)
    latent_codes.load_state_dict(latent_state)
    latent_codes.eval()

    return model, latent_codes, config


def extract_mesh_marching_cubes(
    model: torch.nn.Module,
    latent_code: torch.Tensor,
    resolution: int = 256,
    threshold: float = 0.0,
    batch_size: int = 65536,
    device: torch.device = torch.device("cpu"),
) -> trimesh.Trimesh | None:
    """Extract mesh from learned SDF using marching cubes.

    Args:
        model: trained DeepSDF model
        latent_code: (1, latent_dim) latent vector for the shape
        resolution: grid resolution for marching cubes
        threshold: isosurface threshold (0 for SDF)
        batch_size: points per forward pass (memory control)
        device: torch device

    Returns:
        Extracted trimesh.Trimesh or None if extraction fails
    """
    # Create 3D grid in [-1, 1]^3
    coords = np.linspace(-1.0, 1.0, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(coords, coords, coords, indexing="ij")
    grid_points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

    # Evaluate SDF on grid in batches
    sdf_values = np.zeros(len(grid_points), dtype=np.float32)
    grid_tensor = torch.from_numpy(grid_points).float().to(device)

    with torch.no_grad():
        for i in range(0, len(grid_tensor), batch_size):
            batch_pts = grid_tensor[i:i + batch_size]
            z_batch = latent_code.expand(len(batch_pts), -1)
            pred = model(z_batch, batch_pts)
            sdf_values[i:i + len(batch_pts)] = pred.squeeze(-1).cpu().numpy()

    sdf_values = sdf_values.reshape(resolution, resolution, resolution)

    # Marching cubes
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            sdf_values, level=threshold, spacing=(2.0 / resolution,) * 3
        )
        # Shift vertices to [-1, 1] range
        verts = verts - 1.0

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        return mesh
    except Exception as e:
        print(f"  Marching cubes failed: {e}")
        return None


def chamfer_distance(
    mesh_pred: trimesh.Trimesh,
    mesh_gt: trimesh.Trimesh,
    n_points: int = 30000,
) -> float:
    """Bidirectional Chamfer Distance between two meshes.

    Sample n_points from each mesh, compute mean nearest-neighbor distance
    in both directions.

    Returns:
        Mean bidirectional CD (lower is better)
    """
    pts_pred = trimesh.sample.sample_surface(mesh_pred, n_points)[0]
    pts_gt = trimesh.sample.sample_surface(mesh_gt, n_points)[0]

    # pred -> gt
    tree_gt = KDTree(pts_gt)
    dist_pred_to_gt, _ = tree_gt.query(pts_pred)

    # gt -> pred
    tree_pred = KDTree(pts_pred)
    dist_gt_to_pred, _ = tree_pred.query(pts_gt)

    cd = (np.mean(dist_pred_to_gt ** 2) + np.mean(dist_gt_to_pred ** 2)) / 2.0
    return float(cd)


def compute_iou(
    mesh_pred: trimesh.Trimesh,
    mesh_gt: trimesh.Trimesh,
    resolution: int = 128,
) -> float:
    """Volumetric IoU between two meshes via voxelization.

    Args:
        mesh_pred: predicted mesh
        mesh_gt: ground truth mesh
        resolution: voxel grid resolution

    Returns:
        IoU value (higher is better)
    """
    # Create evaluation grid
    coords = np.linspace(-1.0, 1.0, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(coords, coords, coords, indexing="ij")
    grid_points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

    # Check occupancy (inside mesh = True)
    occ_pred = mesh_pred.contains(grid_points)
    occ_gt = mesh_gt.contains(grid_points)

    intersection = np.sum(occ_pred & occ_gt)
    union = np.sum(occ_pred | occ_gt)

    if union == 0:
        return 0.0
    return float(intersection / union)


def normal_consistency(
    mesh_pred: trimesh.Trimesh,
    mesh_gt: trimesh.Trimesh,
    n_points: int = 30000,
) -> float:
    """Normal Consistency between two meshes.

    Sample points from GT mesh with normals. For each GT point, find nearest
    point on predicted mesh surface via KD-tree. Compute mean |dot(n_gt, n_pred)|.

    Returns:
        Mean normal consistency (higher is better, max 1.0)
    """
    # Sample GT points with normals
    pts_gt, face_idx_gt = trimesh.sample.sample_surface(mesh_gt, n_points)
    normals_gt = mesh_gt.face_normals[face_idx_gt]

    # Sample pred points with normals
    pts_pred, face_idx_pred = trimesh.sample.sample_surface(mesh_pred, n_points)
    normals_pred = mesh_pred.face_normals[face_idx_pred]

    # For each GT point, find nearest pred point
    tree_pred = KDTree(pts_pred)
    _, nearest_idx = tree_pred.query(pts_gt)

    # Compute normal dot product
    matched_normals = normals_pred[nearest_idx]
    dots = np.abs(np.sum(normals_gt * matched_normals, axis=-1))

    return float(np.mean(dots))


def check_divergence(exp_dir: str, config: dict) -> bool:
    """Check if training diverged by reading the training log."""
    log_path = os.path.join(exp_dir, "train.log")
    if not os.path.exists(log_path):
        return False

    baseline_epoch = config.get("divergence_baseline_epoch", 10)
    check_epoch = config.get("divergence_check_epoch", 500)
    threshold = config.get("divergence_ratio_threshold", 0.5)

    baseline_loss = None
    check_loss = None

    import csv
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            if epoch == baseline_epoch:
                baseline_loss = float(row["L_sdf"])
            if epoch == check_epoch:
                check_loss = float(row["L_sdf"])

    if baseline_loss is None or check_loss is None:
        return False

    ratio = check_loss / (baseline_loss + 1e-10)
    return ratio > threshold


def evaluate_experiment(
    exp_dir: str,
    data_dir: str,
    output_path: str = None,
    voxel_resolutions: list = None,
    skip_iou: bool = False,
):
    """Full evaluation pipeline for one experiment run.

    Args:
        exp_dir: path to experiment directory (e.g., experiments/EXP-01/seed42)
        data_dir: path to preprocessed data directory
        output_path: where to save results JSON (default: exp_dir/results.json)
        voxel_resolutions: IoU resolutions (default: [128, 256])
    """
    if voxel_resolutions is None:
        voxel_resolutions = [128, 256]
    if output_path is None:
        output_path = os.path.join(exp_dir, "results.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, latent_codes, config = load_model_and_config(exp_dir, device)

    n_eval_points = config.get("n_eval_points", 30000)
    mc_resolution = config.get("mc_resolution", 256)

    # Check divergence
    diverged = check_divergence(exp_dir, config)
    if diverged:
        print("*** WARNING: This run appears to have DIVERGED ***")

    # Find GT meshes
    gt_mesh_dir = config.get("gt_mesh_dir", os.path.join(data_dir, "gt_meshes"))
    gt_files = sorted([f for f in os.listdir(gt_mesh_dir) if f.endswith(".obj")])

    # Find which shapes were used (based on ratio dir)
    supervision_ratio = config.get("supervision_ratio", 1.0)
    ratio_str = f"ratio_{supervision_ratio:.2f}".replace(".", "p")
    ratio_dir = os.path.join(data_dir, ratio_str)
    npz_files = sorted([f for f in os.listdir(ratio_dir) if f.endswith(".npz")])
    shape_names = [os.path.splitext(f)[0] for f in npz_files]

    num_shapes = config.get("num_shapes", -1)
    if num_shapes > 0:
        shape_names = shape_names[:num_shapes]

    print(f"\nEvaluating {len(shape_names)} shapes")
    print(f"MC resolution: {mc_resolution}")
    print(f"IoU resolutions: {voxel_resolutions}")
    print(f"Eval points: {n_eval_points}\n")

    # Evaluate each shape
    per_shape_results = []
    for shape_idx, shape_name in enumerate(shape_names):
        t0 = time.time()
        print(f"[{shape_idx+1}/{len(shape_names)}] {shape_name}...", end=" ")

        # Load GT mesh
        gt_path = os.path.join(gt_mesh_dir, f"{shape_name}.obj")
        if not os.path.exists(gt_path):
            print(f"GT mesh not found: {gt_path}, skipping")
            continue
        gt_mesh = trimesh.load(gt_path, force='mesh')

        # Get latent code
        idx_tensor = torch.tensor([shape_idx], device=device)
        with torch.no_grad():
            z = latent_codes(idx_tensor)

        # Extract predicted mesh
        pred_mesh = extract_mesh_marching_cubes(
            model, z, resolution=mc_resolution,
            threshold=config.get("eval_threshold", 0.0),
            device=device,
        )

        if pred_mesh is None or len(pred_mesh.vertices) < 10:
            print("FAILED (mesh extraction)")
            per_shape_results.append({
                "shape": shape_name, "status": "failed",
            })
            continue

        # Compute metrics
        result = {"shape": shape_name, "status": "ok"}

        # Chamfer Distance
        cd = chamfer_distance(pred_mesh, gt_mesh, n_points=n_eval_points)
        result["chamfer_distance"] = cd

        # IoU at each resolution
        if not skip_iou:
            for res in voxel_resolutions:
                iou = compute_iou(pred_mesh, gt_mesh, resolution=res)
                result[f"iou_{res}"] = iou

        # Normal Consistency
        nc = normal_consistency(pred_mesh, gt_mesh, n_points=n_eval_points)
        result["normal_consistency"] = nc

        elapsed = time.time() - t0
        iou_str = ""
        if not skip_iou:
            iou_str = "  " + "  ".join(f"IoU@{r}={result[f'iou_{r}']:.4f}" for r in voxel_resolutions)
        print(f"CD={cd:.6f}  NC={nc:.4f}{iou_str}  [{elapsed:.1f}s]")

        # Save reconstructed mesh
        recon_dir = os.path.join(exp_dir, "reconstructions")
        os.makedirs(recon_dir, exist_ok=True)
        pred_mesh.export(os.path.join(recon_dir, f"{shape_name}.obj"))

        per_shape_results.append(result)

    # Aggregate statistics (exclude failed shapes)
    ok_results = [r for r in per_shape_results if r.get("status") == "ok"]

    aggregate = {}
    if ok_results:
        for metric in ["chamfer_distance", "normal_consistency"] + \
                      [f"iou_{r}" for r in voxel_resolutions]:
            values = [r[metric] for r in ok_results if metric in r]
            if values:
                aggregate[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

    # Build final report
    report = {
        "experiment": os.path.basename(os.path.dirname(exp_dir)),
        "seed": config.get("seed", "?"),
        "supervision_ratio": config.get("supervision_ratio", 1.0),
        "use_eikonal": config.get("use_eikonal", True),
        "use_pe": config.get("use_pe", False),
        "pe_levels": config.get("pe_levels", 6) if config.get("use_pe") else None,
        "diverged": diverged,
        "n_shapes_evaluated": len(ok_results),
        "n_shapes_failed": len(per_shape_results) - len(ok_results),
        "mc_resolution": mc_resolution,
        "n_eval_points": n_eval_points,
        "aggregate": aggregate,
        "per_shape": per_shape_results,
    }

    # Save results
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Shapes: {len(ok_results)} ok, {len(per_shape_results) - len(ok_results)} failed")
    if diverged:
        print("*** DIVERGED RUN — results may be unreliable ***")
    if aggregate:
        for metric, stats in aggregate.items():
            print(f"  {metric}: {stats['mean']:.6f} +/- {stats['std']:.6f}")
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepSDF experiment")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment directory (e.g., experiments/EXP-01/seed42)")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Preprocessed data directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: exp_dir/results.json)")
    parser.add_argument("--voxel_res", type=int, nargs="+", default=[128, 256],
                        help="IoU voxel resolutions")
    parser.add_argument("--skip_iou", action="store_true",
                        help="Skip IoU computation (slow on CPU)")
    args = parser.parse_args()

    evaluate_experiment(
        exp_dir=args.exp_dir,
        data_dir=args.data_dir,
        output_path=args.output,
        voxel_resolutions=args.voxel_res,
        skip_iou=args.skip_iou,
    )


if __name__ == "__main__":
    main()

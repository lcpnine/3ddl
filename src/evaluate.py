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
import gc
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


def _resolve_checkpoint(exp_dir: str, mode: str = "auto") -> str:
    """Resolve checkpoint path according to the requested selection policy."""
    best = os.path.join(exp_dir, "checkpoints", "best.pt")
    latest = os.path.join(exp_dir, "checkpoints", "latest.pt")

    if mode == "auto":
        if os.path.exists(best):
            return best
        if os.path.exists(latest):
            return latest
        raise FileNotFoundError(
            f"No checkpoint found in {os.path.join(exp_dir, 'checkpoints')}"
        )
    if mode == "best":
        if not os.path.exists(best):
            raise FileNotFoundError(f"best.pt not found: {best}")
        return best
    if mode == "latest":
        if not os.path.exists(latest):
            raise FileNotFoundError(f"latest.pt not found: {latest}")
        return latest
    raise ValueError(f"Unknown checkpoint mode: {mode!r}")


def load_model_and_config(
    exp_dir: str,
    device: torch.device,
    checkpoint_mode: str = "auto",
):
    """Load trained model from the requested checkpoint."""
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
    ckpt_path = _resolve_checkpoint(exp_dir, checkpoint_mode)
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

    return model, latent_codes, config, ckpt_path


def test_time_optimize_latent(
    model: torch.nn.Module,
    points_sup: torch.Tensor,
    sdf_sup: torch.Tensor,
    latent_dim: int,
    n_iters: int = 800,
    lr: float = 5e-3,
    lambda_reg: float = 1e-4,
    clamp_dist: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Optimize a latent code for a single shape with frozen decoder.

    Follows DeepSDF (Park et al. CVPR 2019) Section 4.2 test-time optimization.
    Initializes z=0, runs Adam on L_sdf(z) + lambda_reg * ||z||^2 while
    keeping all decoder weights frozen.

    Args:
        model: DeepSDF decoder (weights frozen — must already be in .eval() mode)
        points_sup: (N, 3) xyz coordinates of supervised SDF samples
        sdf_sup: (N,) SDF values at those points
        latent_dim: dimensionality of the latent code
        n_iters: number of Adam optimization iterations (default 800)
        lr: Adam learning rate (default 5e-3)
        lambda_reg: L2 regularization weight on ||z||^2 (default 1e-4)
        clamp_dist: SDF clamping delta, matches training (default 0.1)
        device: torch device

    Returns:
        z: (1, latent_dim) optimized latent code (detached)
    """
    z = torch.zeros(1, latent_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)

    pts = points_sup.float().to(device)
    sdf_gt_clamped = torch.clamp(sdf_sup.float().to(device), -clamp_dist, clamp_dist)

    # Subsample to avoid OOM — 250k pts/shape is too large for CPU TTO
    max_pts = 16384
    if len(pts) > max_pts:
        perm = torch.randperm(len(pts))[:max_pts]
        pts = pts[perm]
        sdf_gt_clamped = sdf_gt_clamped[perm]

    model.eval()
    for _ in range(n_iters):
        optimizer.zero_grad()
        pred = model(z.expand(len(pts), -1), pts)
        pred_clamped = torch.clamp(pred.squeeze(-1), -clamp_dist, clamp_dist)
        loss = torch.mean(torch.abs(pred_clamped - sdf_gt_clamped))
        loss = loss + lambda_reg * torch.sum(z ** 2)
        loss.backward()
        optimizer.step()

    return z.detach()


def discover_val_shape_names(
    data_dir: str,
    train_shape_names: list,
    supervision_ratio: float,
) -> list:
    """Return shape names present in data_dir that are NOT in train_shape_names.

    Uses the ratio directory (same source as dataset.py) to enumerate all
    available shapes, then subtracts the train set.
    """
    ratio_str = f"ratio_{supervision_ratio:.2f}".replace(".", "p")
    ratio_dir = os.path.join(data_dir, ratio_str)
    all_names = sorted([
        os.path.splitext(f)[0] for f in os.listdir(ratio_dir) if f.endswith(".npz")
    ])
    train_set = set(train_shape_names)
    return [n for n in all_names if n not in train_set]


def extract_mesh_marching_cubes(
    model: torch.nn.Module,
    latent_code: torch.Tensor,
    resolution: int = 256,
    threshold: float = 0.0,
    batch_size: int = 65536,
    device: torch.device = torch.device("cpu"),
    sphere_clip: bool = True,
    clip_radius: float = 1.0,
) -> trimesh.Trimesh | None:
    """Extract mesh from learned SDF using marching cubes.

    Args:
        model: trained DeepSDF model
        latent_code: (1, latent_dim) latent vector for the shape
        resolution: grid resolution for marching cubes
        threshold: isosurface threshold (0 for SDF)
        batch_size: points per forward pass (memory control)
        device: torch device
        sphere_clip: if True, clamp SDF to +1.0 outside the unit sphere before
            marching cubes. GT meshes are normalized to the unit sphere, so cube
            corners (r~1.73) have no training coverage. PE models produce
            catastrophic oscillations there; clamping prevents spurious
            zero-crossings. Validated in check_b_clipped_eval.py (49% CD drop).
        clip_radius: radius of the sphere used for clipping (default 1.0)

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

    # Sphere clipping: set SDF to +1.0 outside the unit sphere.
    # GT meshes are normalized to unit sphere (preprocess.py:36-43), so cube
    # corners (r~1.73) were never seen during training. Clamping forces SDF
    # positive (outside surface) in unseen regions, preventing spurious
    # zero-crossings. Validated in check_b_clipped_eval.py: 49% CD improvement.
    if sphere_clip:
        r_sq = grid_x ** 2 + grid_y ** 2 + grid_z ** 2
        sdf_values[r_sq > clip_radius ** 2] = 1.0

    # Marching cubes
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            sdf_values, level=threshold, spacing=(2.0 / (resolution - 1),) * 3
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

    cd = (np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)) / 2.0
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
    mc_resolution_override: int = None,
    eval_split: str = "train",
    checkpoint_mode: str = "auto",
    sphere_clip: bool = True,
    tto_n_iters: int = 800,
    tto_lr: float = 5e-3,
    tto_lambda_reg: float = 1e-4,
    tto_clamp_dist: float = 0.1,
):
    """Full evaluation pipeline for one experiment run.

    Args:
        exp_dir: path to experiment directory (e.g., experiments/EXP-01/seed42)
        data_dir: path to preprocessed data directory
        output_path: where to save results JSON (default: exp_dir/results.json)
        voxel_resolutions: IoU resolutions (default: [128, 256])
        eval_split: which shapes to evaluate — "train" (default, backward-compat),
            "val" (test-time latent optimization on held-out shapes), or "all"
        sphere_clip: clamp SDF to +1.0 outside unit sphere before marching cubes
        tto_n_iters: test-time optimization iterations (val split only)
        tto_lr: TTO Adam learning rate
        tto_lambda_reg: TTO latent L2 regularization weight
        tto_clamp_dist: TTO SDF clamping delta
    """
    if voxel_resolutions is None:
        voxel_resolutions = [128, 256]
    if output_path is None:
        output_path = os.path.join(exp_dir, "results.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, latent_codes, config, ckpt_path = load_model_and_config(
        exp_dir, device, checkpoint_mode=checkpoint_mode
    )

    n_eval_points = config.get("n_eval_points", 30000)
    mc_resolution = mc_resolution_override or config.get("mc_resolution", 256)

    # Check divergence
    diverged = check_divergence(exp_dir, config)
    if diverged:
        print("*** WARNING: This run appears to have DIVERGED ***")

    # Find GT meshes
    gt_mesh_dir = config.get("gt_mesh_dir", os.path.join(data_dir, "gt_meshes"))
    gt_files = sorted([f for f in os.listdir(gt_mesh_dir) if f.endswith(".obj")])

    # Reconstruct exact shape→latent-index mapping used during training.
    # train_shapes.json is written by train.py and contains shape names in the
    # shuffled order that was used to assign latent code indices 0..n_train-1.
    train_shapes_path = os.path.join(exp_dir, "train_shapes.json")
    if os.path.exists(train_shapes_path):
        with open(train_shapes_path) as f:
            shape_names = json.load(f)
        shape_order = "train_shapes.json"
        print(f"  Loaded {len(shape_names)} train shapes from train_shapes.json")
    else:
        # Fallback for legacy checkpoints trained before train_shapes.json was
        # introduced. Uses sorted alphabetical order (the old pre-shuffle behaviour).
        print("  WARNING: train_shapes.json not found — using sorted fallback for legacy run")
        shape_order = "sorted_fallback"
        supervision_ratio = config.get("supervision_ratio", 1.0)
        ratio_str = f"ratio_{supervision_ratio:.2f}".replace(".", "p")
        ratio_dir = os.path.join(data_dir, ratio_str)
        npz_files = sorted([f for f in os.listdir(ratio_dir) if f.endswith(".npz")])
        shape_names = [os.path.splitext(f)[0] for f in npz_files]
        num_shapes = config.get("num_shapes", -1)
        if num_shapes > 0:
            shape_names = shape_names[:num_shapes]
        train_split = config.get("train_split", 0.75)
        n_train = int(len(shape_names) * train_split)
        shape_names = shape_names[:n_train]
        print(f"  Restricting to {n_train} train shapes (latent codes 0..{n_train-1} were optimized)")

    # Build unified shape list with split tags
    train_shape_list = list(shape_names)
    val_shape_list = []
    if eval_split in ("val", "all"):
        supervision_ratio = config.get("supervision_ratio", 1.0)
        val_shape_list = discover_val_shape_names(data_dir, train_shape_list, supervision_ratio)
        print(f"  Discovered {len(val_shape_list)} val shapes for TTO evaluation")

    shapes_to_eval = []
    if eval_split in ("train", "all"):
        shapes_to_eval += [{"name": n, "split": "train"} for n in train_shape_list]
    if eval_split in ("val", "all"):
        shapes_to_eval += [{"name": n, "split": "val"} for n in val_shape_list]

    print(f"\nEvaluating {len(shapes_to_eval)} shapes (split={eval_split}, sphere_clip={sphere_clip})")
    print(f"MC resolution: {mc_resolution}")
    print(f"IoU resolutions: {voxel_resolutions}")
    print(f"Eval points: {n_eval_points}\n")

    metric_seed = 0  # fixed, independent of training seed for cross-seed comparability

    latent_dim = config.get("latent_dim", 256)
    supervision_ratio = config.get("supervision_ratio", 1.0)
    ratio_str = f"ratio_{supervision_ratio:.2f}".replace(".", "p")
    ratio_dir = os.path.join(data_dir, ratio_str)

    per_shape_results = []
    for item_idx, item in enumerate(shapes_to_eval):
        shape_name = item["name"]
        split = item["split"]
        t0 = time.time()
        print(f"[{item_idx+1}/{len(shapes_to_eval)}] [{split}] {shape_name}...", end=" ", flush=True)

        # Load GT mesh
        gt_path = os.path.join(gt_mesh_dir, f"{shape_name}.obj")
        if not os.path.exists(gt_path):
            print(f"GT mesh not found: {gt_path}, skipping")
            continue
        gt_mesh = trimesh.load(gt_path, force='mesh')

        # Get latent code: train shapes use stored embedding; val shapes use TTO
        if split == "train":
            shape_idx = train_shape_list.index(shape_name)
            idx_tensor = torch.tensor([shape_idx], device=device)
            with torch.no_grad():
                z = latent_codes(idx_tensor)
        else:
            npz_path = os.path.join(ratio_dir, f"{shape_name}.npz")
            if not os.path.exists(npz_path):
                print(f"SDF samples not found: {npz_path}, skipping")
                continue
            npz = np.load(npz_path)
            print(f"(TTO {tto_n_iters}it)...", end=" ", flush=True)
            z = test_time_optimize_latent(
                model,
                torch.from_numpy(npz["points_sup"]),
                torch.from_numpy(npz["sdf_sup"]),
                latent_dim=latent_dim,
                n_iters=tto_n_iters,
                lr=tto_lr,
                lambda_reg=tto_lambda_reg,
                clamp_dist=tto_clamp_dist,
                device=device,
            )

        # Extract predicted mesh
        pred_mesh = extract_mesh_marching_cubes(
            model, z, resolution=mc_resolution,
            threshold=config.get("eval_threshold", 0.0),
            device=device,
            sphere_clip=sphere_clip,
        )

        if pred_mesh is None or len(pred_mesh.vertices) < 10:
            print("FAILED (mesh extraction)")
            per_shape_results.append({"shape": shape_name, "split": split, "status": "failed"})
            continue

        # Compute metrics (fixed seed per shape for reproducibility)
        np.random.seed(metric_seed + item_idx)
        result = {"shape": shape_name, "split": split, "status": "ok"}

        try:
            cd = chamfer_distance(pred_mesh, gt_mesh, n_points=n_eval_points)
            result["chamfer_distance"] = cd

            if not skip_iou:
                for res in voxel_resolutions:
                    result[f"iou_{res}"] = compute_iou(pred_mesh, gt_mesh, resolution=res)

            nc = normal_consistency(pred_mesh, gt_mesh, n_points=n_eval_points)
            result["normal_consistency"] = nc

            elapsed = time.time() - t0
            iou_str = ""
            if not skip_iou:
                iou_str = "  " + "  ".join(f"IoU@{r}={result[f'iou_{r}']:.4f}" for r in voxel_resolutions)
            print(f"CD={cd:.6f}  NC={nc:.4f}{iou_str}  [{elapsed:.1f}s]")
        except Exception as e:
            print(f"FAILED (metrics: {e})")
            result["status"] = "failed"

        # Save reconstructed mesh (export failure does not invalidate metrics)
        if result["status"] == "ok":
            try:
                recon_dir = os.path.join(exp_dir, f"reconstructions_{split}")
                os.makedirs(recon_dir, exist_ok=True)
                save_mesh = pred_mesh
                if len(pred_mesh.faces) > 200000:
                    save_mesh = pred_mesh.simplify_quadric_decimation(200000)
                save_mesh.export(os.path.join(recon_dir, f"{shape_name}.obj"))
            except Exception as e:
                print(f"  (mesh export failed: {e})")

        per_shape_results.append(result)
        del gt_mesh, pred_mesh
        gc.collect()

    # Aggregate statistics (exclude failed shapes)
    ok_results = [r for r in per_shape_results if r.get("status") == "ok"]

    def _agg(results_subset):
        agg = {"n_ok": len(results_subset)}
        for metric in ["chamfer_distance", "normal_consistency"] + \
                      ([] if skip_iou else [f"iou_{r}" for r in voxel_resolutions]):
            vals = [r[metric] for r in results_subset if metric in r]
            if vals:
                agg[metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
        return agg

    aggregate = _agg(ok_results)
    aggregate["n_total"] = len(per_shape_results)
    aggregate["success_rate"] = float(len(ok_results) / max(1, len(per_shape_results)))

    aggregate_by_split = {}
    for sp in ("train", "val"):
        subset = [r for r in ok_results if r.get("split") == sp]
        if subset:
            aggregate_by_split[sp] = _agg(subset)

    # Build final report
    report = {
        "experiment": os.path.basename(os.path.dirname(exp_dir)),
        "seed": config.get("seed", "?"),
        "supervision_ratio": config.get("supervision_ratio", 1.0),
        "use_eikonal": config.get("use_eikonal", True),
        "use_pe": config.get("use_pe", False),
        "pe_levels": config.get("pe_levels", 6) if config.get("use_pe") else None,
        "diverged": diverged,
        "checkpoint": os.path.basename(ckpt_path),
        "shape_order": shape_order,
        "eval_split": eval_split,
        "sphere_clip": sphere_clip,
        "cd_formula": "L1",
        "tto_n_iters": tto_n_iters if eval_split in ("val", "all") else None,
        "n_shapes_evaluated": len(ok_results),
        "n_shapes_failed": len(per_shape_results) - len(ok_results),
        "mc_resolution": mc_resolution,
        "n_eval_points": n_eval_points,
        "aggregate": aggregate,
        "aggregate_by_split": aggregate_by_split,
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
            if isinstance(stats, dict):
                print(f"  {metric}: {stats['mean']:.6f} +/- {stats['std']:.6f}")
            else:
                print(f"  {metric}: {stats}")
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepSDF experiment")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment directory (e.g., experiments/EXP-01/seed42)")
    parser.add_argument("--data_dir", type=str, default="data/processed_shapenet",
                        help="Preprocessed data directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: exp_dir/results.json)")
    parser.add_argument("--voxel_res", type=int, nargs="+", default=[128, 256],
                        help="IoU voxel resolutions")
    parser.add_argument("--skip_iou", action="store_true",
                        help="Skip IoU computation (slow on CPU)")
    parser.add_argument("--mc_resolution", type=int, default=None,
                        help="Override marching cubes resolution (default: from config)")
    parser.add_argument("--eval_split", choices=["train", "val", "all"], default="train",
                        help="Shapes to evaluate: train (default), val (TTO), or all")
    parser.add_argument("--checkpoint_mode", choices=["auto", "best", "latest"], default="auto",
                        help="Checkpoint selection policy")
    parser.add_argument("--sphere_clip", action="store_true", default=True,
                        help="Clamp SDF to +1.0 outside unit sphere before marching cubes (default: on)")
    parser.add_argument("--no_sphere_clip", action="store_false", dest="sphere_clip",
                        help="Disable sphere clipping")
    parser.add_argument("--tto_n_iters", type=int, default=800,
                        help="Test-time optimization iterations (val split only)")
    parser.add_argument("--tto_lr", type=float, default=5e-3,
                        help="Test-time optimization learning rate")
    parser.add_argument("--tto_lambda_reg", type=float, default=1e-4,
                        help="Test-time optimization latent L2 regularization weight")
    args = parser.parse_args()

    evaluate_experiment(
        exp_dir=args.exp_dir,
        data_dir=args.data_dir,
        output_path=args.output,
        voxel_resolutions=args.voxel_res,
        skip_iou=args.skip_iou,
        mc_resolution_override=args.mc_resolution,
        eval_split=args.eval_split,
        checkpoint_mode=args.checkpoint_mode,
        sphere_clip=args.sphere_clip,
        tto_n_iters=args.tto_n_iters,
        tto_lr=args.tto_lr,
        tto_lambda_reg=args.tto_lambda_reg,
    )


if __name__ == "__main__":
    main()

"""
Training loop for semi-supervised DeepSDF.

Supports:
  - Mixed supervised/unsupervised training with Eikonal regularization
  - Ratio-dependent Eikonal warmup schedule
  - Gradient clipping (max_norm=1.0)
  - Checkpoint save/resume (latest=full state, best=weights only)
  - Divergence detection
  - CSV training log for diagnosis

Usage:
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --config_override supervision_ratio=0.1 use_pe=true
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import yaml

from model import DeepSDF, LatentCodes
from losses import (
    sdf_loss,
    eikonal_loss,
    latent_reg_loss,
    second_order_loss,
    compute_spatial_gradients,
    compute_eikonal_weight,
    get_warmup_epochs,
)
from dataset import SDFDataset, sample_batch_points


def load_config(config_path: str, overrides: list = None) -> dict:
    """Load YAML config and apply command-line overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if overrides:
        for override in overrides:
            key, value = override.split("=", 1)
            # Auto-convert types
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif "." in value:
                try:
                    value = float(value)
                except ValueError:
                    pass
            else:
                try:
                    value = int(value)
                except ValueError:
                    pass
            config[key] = value

    return config


def setup_experiment_dir(config: dict) -> str:
    """Create experiment directory and save config."""
    exp_name = config.get("exp_name", "debug")
    seed = config.get("seed", 42)
    exp_dir = os.path.join(config["exp_dir"], exp_name, f"seed{seed}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    # Save config
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return exp_dir


def save_checkpoint(
    path: str,
    model: DeepSDF,
    latent_codes: LatentCodes,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    epoch: int = 0,
    best_loss: float = float("inf"),
    full_state: bool = True,
):
    """Save checkpoint.

    full_state=True: model + latent + optimizer + scheduler (for resume)
    full_state=False: model + latent weights only (for evaluation)
    """
    state = {
        "model_state_dict": model.state_dict(),
        "latent_codes_state_dict": latent_codes.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
    }
    if full_state and optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if full_state and scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: DeepSDF,
    latent_codes: LatentCodes,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
) -> tuple:
    """Load checkpoint. Returns (start_epoch, best_loss)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    latent_codes.load_state_dict(checkpoint["latent_codes_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("best_loss", float("inf"))


def train(config: dict):
    """Main training function."""
    # Seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Experiment directory
    exp_dir = setup_experiment_dir(config)
    print(f"Experiment dir: {exp_dir}")

    # Dataset
    supervision_ratio = config.get("supervision_ratio", 1.0)
    num_shapes = config.get("num_shapes", -1)

    train_dataset = SDFDataset(
        data_dir=config["data_dir"],
        supervision_ratio=supervision_ratio,
        split="train",
        train_frac=config.get("train_split", 0.75),
        num_shapes=num_shapes,
    )
    val_dataset = SDFDataset(
        data_dir=config["data_dir"],
        supervision_ratio=supervision_ratio,
        split="val",
        train_frac=config.get("train_split", 0.75),
        num_shapes=num_shapes,
    )
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    print(f"Train shapes: {n_train}, Val shapes: {n_val}")
    print(f"Supervision ratio: {supervision_ratio}")

    # Model
    model = DeepSDF(
        latent_dim=config.get("latent_dim", 256),
        hidden_dim=config.get("hidden_dim", 512),
        num_layers=config.get("num_layers", 8),
        skip_layer=config.get("skip_layer", 4),
        use_pe=config.get("use_pe", False),
        pe_levels=config.get("pe_levels", 6),
    ).to(device)

    total_shapes = n_train + n_val
    latent_codes = LatentCodes(
        num_shapes=total_shapes,
        latent_dim=config.get("latent_dim", 256),
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"PE: {'L=' + str(config.get('pe_levels', 6)) if config.get('use_pe') else 'off'}")

    # Optimizer — jointly optimize model weights AND latent codes
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": config.get("lr", 5e-4)},
            {"params": latent_codes.parameters(), "lr": config.get("lr", 5e-4)},
        ],
        weight_decay=config.get("weight_decay", 0.0),
    )

    # LR scheduler
    scheduler_type = config.get("lr_scheduler", "step")
    if scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("lr_step_size", 500),
            gamma=config.get("lr_gamma", 0.5),
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.get("epochs", 1000)
        )
    else:
        scheduler = None

    # Loss config
    use_eikonal = config.get("use_eikonal", True)
    lambda_eik = config.get("lambda_eik", 0.1)
    lambda_z = config.get("lambda_z", 1e-4)
    lambda_2nd = config.get("lambda_2nd", 0.0)
    use_2nd = lambda_2nd > 0

    warmup_epochs = config.get("warmup_epochs", -1)
    if warmup_epochs < 0:
        warmup_epochs = get_warmup_epochs(supervision_ratio)
    print(f"Eikonal: {'on' if use_eikonal else 'off'}, "
          f"lambda={lambda_eik}, warmup={warmup_epochs}ep")

    # Training params
    epochs = config.get("epochs", 1000)
    batch_size = config.get("batch_size", 16384)
    grad_clip = config.get("grad_clip_max_norm", 1.0)
    checkpoint_every = config.get("checkpoint_every", 100)
    log_every = config.get("log_every", 10)

    # Divergence detection
    div_check_epoch = config.get("divergence_check_epoch", 500)
    div_baseline_epoch = config.get("divergence_baseline_epoch", 10)
    div_threshold = config.get("divergence_ratio_threshold", 0.5)
    sdf_loss_at_baseline = None

    # Resume from checkpoint
    latest_ckpt = os.path.join(exp_dir, "checkpoints", "latest.pt")
    start_epoch = 0
    best_val_loss = float("inf")
    if os.path.exists(latest_ckpt):
        print(f"Resuming from {latest_ckpt}")
        start_epoch, best_val_loss = load_checkpoint(
            latest_ckpt, model, latent_codes, optimizer, scheduler
        )
        print(f"Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    # CSV log
    log_path = os.path.join(exp_dir, "train.log")
    log_fields = [
        "epoch", "L_total", "L_sdf", "L_eik", "L_z", "L_2nd",
        "eik_weight", "grad_norm_mean", "lr", "time_s",
    ]
    log_file = open(log_path, "a", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    if start_epoch == 0:
        log_writer.writeheader()

    # ==================== TRAINING LOOP ====================
    print(f"\nStarting training: epochs {start_epoch+1}-{epochs}, "
          f"batch_size={batch_size}, grad_clip={grad_clip}")

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        epoch_start = time.time()

        epoch_losses = {
            "L_total": 0.0, "L_sdf": 0.0, "L_eik": 0.0,
            "L_z": 0.0, "L_2nd": 0.0,
        }
        epoch_grad_norms = []

        # Current Eikonal weight with warmup
        eik_weight = compute_eikonal_weight(epoch, warmup_epochs, lambda_eik) if use_eikonal else 0.0

        # Accumulate gradients across all shapes, step once per epoch
        optimizer.zero_grad()

        # Iterate over all training shapes
        for shape_idx in range(n_train):
            shape_data = train_dataset[shape_idx]
            batch = sample_batch_points(shape_data, batch_size, device)

            sup_points = batch["sup_points"]       # (N/2, 3), requires_grad
            sup_sdf_gt = batch["sup_sdf"]           # (N/2,)
            unsup_points = batch["unsup_points"]    # (N/2, 3), requires_grad

            # Get latent code for this shape
            shape_idx_tensor = torch.tensor([shape_idx], device=device)
            z = latent_codes(shape_idx_tensor)  # (1, latent_dim)

            # Expand latent code to match batch sizes
            z_sup = z.expand(len(sup_points), -1)
            z_unsup = z.expand(len(unsup_points), -1)

            # Forward pass — supervised points
            sup_pred = model(z_sup, sup_points)  # (N/2, 1)

            # L_sdf
            loss_sdf = sdf_loss(sup_pred, sup_sdf_gt)

            # Compute gradients for Eikonal on ALL points
            loss_eik = torch.tensor(0.0, device=device)
            loss_2nd = torch.tensor(0.0, device=device)
            mean_grad_norm = 0.0

            if use_eikonal:
                # Supervised point gradients
                sup_grads = compute_spatial_gradients(sup_pred, sup_points, create_graph=True)

                # Unsupervised point gradients
                unsup_pred = model(z_unsup, unsup_points)
                unsup_grads = compute_spatial_gradients(unsup_pred, unsup_points, create_graph=True)

                # Combined Eikonal loss
                all_grads = torch.cat([sup_grads, unsup_grads], dim=0)
                loss_eik = eikonal_loss(all_grads)

                mean_grad_norm = torch.norm(all_grads, dim=-1).mean().item()

                # Optional second-order loss
                if use_2nd:
                    all_points = torch.cat([sup_points, unsup_points], dim=0)
                    loss_2nd = second_order_loss(all_grads, all_points)

            # L_z (regularize the latent code used in this batch)
            loss_z = latent_reg_loss(z)

            # Total loss (divided by n_train for gradient averaging)
            loss_total = loss_sdf + eik_weight * loss_eik + lambda_z * loss_z
            if use_2nd:
                loss_total = loss_total + lambda_2nd * loss_2nd

            # Backward — accumulate gradients (divided by n_train for averaging)
            (loss_total / n_train).backward()

            # Accumulate losses (unscaled for logging)
            epoch_losses["L_total"] += loss_total.item()
            epoch_losses["L_sdf"] += loss_sdf.item()
            epoch_losses["L_eik"] += loss_eik.item()
            epoch_losses["L_z"] += loss_z.item()
            epoch_losses["L_2nd"] += loss_2nd.item()
            epoch_grad_norms.append(mean_grad_norm)

        # Gradient clipping and optimizer step (once per epoch)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(latent_codes.parameters()),
            max_norm=grad_clip,
        )
        optimizer.step()

        # Average over shapes
        for key in epoch_losses:
            epoch_losses[key] /= n_train

        avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
        elapsed = time.time() - epoch_start

        # Step LR scheduler
        if scheduler is not None:
            scheduler.step()

        # Divergence detection
        if epoch == div_baseline_epoch:
            sdf_loss_at_baseline = epoch_losses["L_sdf"]
        if epoch == div_check_epoch and sdf_loss_at_baseline is not None:
            ratio = epoch_losses["L_sdf"] / (sdf_loss_at_baseline + 1e-10)
            if ratio > div_threshold:
                print(f"\n*** DIVERGENCE DETECTED at epoch {epoch} ***")
                print(f"    L_sdf ratio: {ratio:.4f} (threshold: {div_threshold})")
                print(f"    L_sdf@{div_baseline_epoch}: {sdf_loss_at_baseline:.6f}")
                print(f"    L_sdf@{div_check_epoch}: {epoch_losses['L_sdf']:.6f}")

        # Log
        current_lr = optimizer.param_groups[0]["lr"]
        log_row = {
            "epoch": epoch,
            "L_total": f"{epoch_losses['L_total']:.6f}",
            "L_sdf": f"{epoch_losses['L_sdf']:.6f}",
            "L_eik": f"{epoch_losses['L_eik']:.6f}",
            "L_z": f"{epoch_losses['L_z']:.8f}",
            "L_2nd": f"{epoch_losses['L_2nd']:.6f}",
            "eik_weight": f"{eik_weight:.4f}",
            "grad_norm_mean": f"{avg_grad_norm:.4f}",
            "lr": f"{current_lr:.6f}",
            "time_s": f"{elapsed:.1f}",
        }
        log_writer.writerow(log_row)
        log_file.flush()

        if epoch % log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"L_total={epoch_losses['L_total']:.5f} "
                f"L_sdf={epoch_losses['L_sdf']:.5f} "
                f"L_eik={epoch_losses['L_eik']:.5f} "
                f"|grad|={avg_grad_norm:.3f} "
                f"eik_w={eik_weight:.3f} "
                f"lr={current_lr:.1e} "
                f"[{elapsed:.1f}s]"
            )

        # Validation
        val_loss = evaluate_val(model, latent_codes, val_dataset, batch_size,
                                device, n_train)

        # Checkpointing
        if epoch % checkpoint_every == 0:
            save_checkpoint(
                os.path.join(exp_dir, "checkpoints", "latest.pt"),
                model, latent_codes, optimizer, scheduler,
                epoch, best_val_loss, full_state=True,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                os.path.join(exp_dir, "checkpoints", "best.pt"),
                model, latent_codes,
                epoch=epoch, best_loss=best_val_loss, full_state=False,
            )

    # Save final checkpoint
    save_checkpoint(
        os.path.join(exp_dir, "checkpoints", "latest.pt"),
        model, latent_codes, optimizer, scheduler,
        epochs, best_val_loss, full_state=True,
    )

    log_file.close()
    print(f"\nTraining complete. Best val L_sdf: {best_val_loss:.6f}")
    print(f"Checkpoints: {os.path.join(exp_dir, 'checkpoints')}")


def evaluate_val(
    model: DeepSDF,
    latent_codes: LatentCodes,
    val_dataset: SDFDataset,
    batch_size: int,
    device: torch.device,
    train_offset: int,
) -> float:
    """Quick validation: compute mean L_sdf on validation shapes."""
    model.eval()
    total_loss = 0.0
    n_val = len(val_dataset)

    if n_val == 0:
        return float("inf")

    with torch.no_grad():
        for i in range(n_val):
            shape_data = val_dataset[i]
            # Val shapes start after train shapes in the latent code table
            shape_idx = train_offset + i

            sup_points = shape_data["points_sup"]
            sup_sdf_gt = shape_data["sdf_sup"]

            # Sample subset for fast validation
            n_sample = min(batch_size // 2, len(sup_points))
            indices = torch.randint(0, len(sup_points), (n_sample,))

            pts = sup_points[indices].to(device)
            sdf_gt = sup_sdf_gt[indices].to(device)

            idx_tensor = torch.tensor([shape_idx], device=device)
            z = latent_codes(idx_tensor).expand(n_sample, -1)

            pred = model(z, pts)
            loss = sdf_loss(pred, sdf_gt)
            total_loss += loss.item()

    model.train()
    return total_loss / n_val


def main():
    parser = argparse.ArgumentParser(description="Train semi-supervised DeepSDF")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument("--config_override", nargs="*", default=[],
                        help="Override config values: key=value pairs")
    args = parser.parse_args()

    config = load_config(args.config, args.config_override)

    # Print key config
    print("=" * 60)
    print("Semi-Supervised DeepSDF Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    if args.config_override:
        print(f"Overrides: {args.config_override}")
    print()

    train(config)


if __name__ == "__main__":
    main()

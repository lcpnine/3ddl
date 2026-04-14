"""
Check C: SDF cross-section visualization — PE vs non-PE model.

Queries SDF along the x-axis (y=0, z=0) from -1.5 to 1.5 for two models
and saves a comparison plot. Expected output:
  - Non-PE model: smooth, monotonically increasing SDF outside the shape
  - PE model: oscillating SDF values especially outside r > 1.0

Usage:
    python scripts/check_c_sdf_crosssection.py \
        --pe_exp_dir    experiments/EXP-09/seed42 \
        --nope_exp_dir  experiments/EXP-02/seed42 \
        --shape_idx     0 \
        --output        experiments/figures/check_c_sdf_crosssection.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluate import load_model_and_config


def query_sdf_along_axis(model, latent_code, axis="x", n_points=500, extent=1.5,
                          device=torch.device("cpu")):
    """Query SDF along a single axis through the origin."""
    t = np.linspace(-extent, extent, n_points)
    if axis == "x":
        pts = np.stack([t, np.zeros(n_points), np.zeros(n_points)], axis=1)
    elif axis == "y":
        pts = np.stack([np.zeros(n_points), t, np.zeros(n_points)], axis=1)
    else:
        pts = np.stack([np.zeros(n_points), np.zeros(n_points), t], axis=1)

    pts_t = torch.from_numpy(pts).float().to(device)
    with torch.no_grad():
        z = latent_code.expand(len(pts_t), -1)
        sdf = model(z, pts_t).squeeze(-1).cpu().numpy()

    return t, sdf


def run_check_c(pe_exp_dir, nope_exp_dir, shape_idx=0, output="check_c_crosssection.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load PE model
    model_pe, lc_pe, cfg_pe = load_model_and_config(pe_exp_dir, device)
    print(f"PE model:    use_pe={cfg_pe.get('use_pe')}, L={cfg_pe.get('pe_levels')}, "
          f"ratio={cfg_pe.get('supervision_ratio')}")

    # Load non-PE model
    model_nope, lc_nope, cfg_nope = load_model_and_config(nope_exp_dir, device)
    print(f"No-PE model: use_pe={cfg_nope.get('use_pe')}, "
          f"ratio={cfg_nope.get('supervision_ratio')}")

    idx_t = torch.tensor([shape_idx], device=device)
    with torch.no_grad():
        z_pe   = lc_pe(idx_t)
        z_nope = lc_nope(idx_t)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors_pe   = "#e74c3c"
    colors_nope = "#2ecc71"

    for ax_i, axis in enumerate(["x", "y", "z"]):
        ax = axes[ax_i]

        t_pe,   sdf_pe   = query_sdf_along_axis(model_pe,   z_pe,   axis=axis, device=device)
        t_nope, sdf_nope = query_sdf_along_axis(model_nope, z_nope, axis=axis, device=device)

        ax.plot(t_nope, sdf_nope, color=colors_nope, lw=1.5, label="No PE (EXP-02)", zorder=3)
        ax.plot(t_pe,   sdf_pe,   color=colors_pe,   lw=1.5, label=f"PE L={cfg_pe.get('pe_levels')} (EXP-09)", zorder=2)

        # Mark training distribution boundary
        ax.axvline(-1.0, color="gray", ls="--", lw=0.8, alpha=0.7, label="r=1.0 boundary")
        ax.axvline( 1.0, color="gray", ls="--", lw=0.8, alpha=0.7)
        ax.axhline( 0.0, color="black", ls="-", lw=0.5, alpha=0.5)

        # Shade OOD region
        ax.axvspan(-1.5, -1.0, alpha=0.08, color="orange", label="OOD region")
        ax.axvspan( 1.0,  1.5, alpha=0.08, color="orange")

        ax.set_xlabel(f"{axis} coordinate")
        ax.set_ylabel("SDF value")
        ax.set_title(f"SDF along {axis}-axis (shape {shape_idx})")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"SDF Cross-Section: PE vs No-PE\n"
        f"Orange = OOD region (r > 1.0, never seen during training)\n"
        f"Oscillations in PE curve outside r=1.0 confirm hypothesis #3",
        fontsize=10
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {output}")

    # Print numeric summary
    for label, t, sdf in [("No-PE", t_nope, sdf_nope), ("PE", t_pe, sdf_pe)]:
        ood_mask = np.abs(t) > 1.0
        in_mask  = np.abs(t) <= 1.0
        print(f"\n{label}:")
        print(f"  SDF std inside  r<=1.0: {sdf[in_mask].std():.4f}")
        print(f"  SDF std outside r>1.0:  {sdf[ood_mask].std():.4f}")
        print(f"  SDF min outside r>1.0:  {sdf[ood_mask].min():.4f}  "
              f"(expected: large positive for SDF)")
        print(f"  Oscillating outside:    "
              f"{'YES — PE failure confirmed' if sdf[ood_mask].std() > 0.1 else 'No'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pe_exp_dir",   default="experiments/EXP-09/seed42")
    parser.add_argument("--nope_exp_dir", default="experiments/EXP-02/seed42")
    parser.add_argument("--shape_idx",    type=int, default=0)
    parser.add_argument("--output",       default="experiments/figures/check_c_sdf_crosssection.png")
    args = parser.parse_args()
    run_check_c(args.pe_exp_dir, args.nope_exp_dir, args.shape_idx, args.output)

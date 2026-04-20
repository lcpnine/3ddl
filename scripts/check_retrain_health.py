"""Post-training sanity check for a retrained experiment.

Given an experiment directory (typically experiments/EXP-01/seed42 after the
first retrain finishes), inspect:

  1. Train log — L_sdf trajectory, final loss, divergence flag.
  2. Latent codes — norm distribution (tight ~0.18 was the degenerate
     regime; post-fix should spread).
  3. Grid SDF — predicted SDF over [-1,1]^3 should span both signs with
     meaningful magnitude, not collapse to near-zero like pre-fix.
  4. Eval results — CD/NC aggregate if results.json exists.

Usage:
    python scripts/check_retrain_health.py experiments/EXP-01/seed42
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import load_model_and_config  # noqa: E402


def inspect_train_log(exp_dir: Path) -> dict:
    log_path = exp_dir / "train.log"
    if not log_path.exists():
        return {"error": f"train.log missing at {log_path}"}
    rows = []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {"error": "train.log is empty"}
    l_sdf_final = float(rows[-1]["L_sdf"])
    l_sdf_at_10 = float(rows[min(9, len(rows) - 1)]["L_sdf"])
    l_sdf_min = min(float(r["L_sdf"]) for r in rows)
    return {
        "n_epochs": int(rows[-1]["epoch"]),
        "L_sdf_final": l_sdf_final,
        "L_sdf_at_epoch10": l_sdf_at_10,
        "L_sdf_min": l_sdf_min,
        "converged_below_0p01": l_sdf_final < 0.01,
        "degenerate_plateau": abs(l_sdf_final - 0.032) < 0.005,
    }


def inspect_latents_and_grid(exp_dir: Path) -> dict:
    device = torch.device("cpu")
    model, latent_codes, cfg, ckpt_path = load_model_and_config(
        str(exp_dir), device, checkpoint_mode="auto"
    )
    z = latent_codes.embedding.weight.detach()
    norms = z.norm(dim=1)

    # Sample grid SDF
    coords = np.linspace(-1.0, 1.0, 48)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
    grid = torch.from_numpy(
        np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)
    )
    # use first latent as a representative shape
    z0 = latent_codes(torch.tensor([0]))
    with torch.no_grad():
        vals = model(z0.expand(len(grid), -1), grid).squeeze(-1).numpy()

    return {
        "checkpoint": Path(ckpt_path).name,
        "n_latents": int(z.shape[0]),
        "latent_dim": int(z.shape[1]),
        "latent_norm_mean": float(norms.mean()),
        "latent_norm_std": float(norms.std()),
        "latent_norm_min": float(norms.min()),
        "latent_norm_max": float(norms.max()),
        "grid_sdf_min": float(vals.min()),
        "grid_sdf_max": float(vals.max()),
        "grid_sdf_mean": float(vals.mean()),
        "grid_sdf_frac_pos_over_0p05": float((vals > 0.05).mean()),
        "grid_sdf_frac_neg_below_neg_0p05": float((vals < -0.05).mean()),
    }


def inspect_results(exp_dir: Path) -> dict:
    for fname in ("results.json", "results_rerun_fixed_eval.json"):
        p = exp_dir / fname
        if not p.exists():
            continue
        d = json.load(open(p))
        agg = d.get("aggregate", {})
        cd = agg.get("chamfer_distance", {})
        nc = agg.get("normal_consistency", {})
        return {
            "source": fname,
            "n_ok": agg.get("n_ok"),
            "n_total": agg.get("n_total"),
            "CD_mean": cd.get("mean"),
            "NC_mean": nc.get("mean"),
            "checkpoint": d.get("checkpoint"),
        }
    return {"error": "no results.json found"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    args = parser.parse_args()

    exp_dir = args.exp_dir
    if not exp_dir.exists():
        raise SystemExit(f"exp_dir not found: {exp_dir}")

    print(f"=== {exp_dir} ===\n")

    train = inspect_train_log(exp_dir)
    print("[train.log]")
    for k, v in train.items():
        print(f"  {k}: {v}")

    print("\n[model/latents/grid]")
    try:
        info = inspect_latents_and_grid(exp_dir)
        for k, v in info.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  error: {e}")

    print("\n[results.json]")
    res = inspect_results(exp_dir)
    for k, v in res.items():
        print(f"  {k}: {v}")

    # Verdict
    print("\n=== Verdict ===")
    if train.get("degenerate_plateau"):
        print("  FAIL: L_sdf ~0.032 still — degenerate solution NOT fixed")
    elif train.get("L_sdf_final", 1.0) > 0.02:
        print("  WARN: L_sdf > 0.02 — training may be underfit, inspect further")
    else:
        print("  PASS: L_sdf converged below 0.02")


if __name__ == "__main__":
    main()

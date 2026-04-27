"""Export a handful of reconstructed meshes per experiment for the qualitative figure.

Runs marching cubes (sphere-clipped, MC res 128) on the target shapes for each
(experiment, seed) using locally-backed-up checkpoints. Avoids SLURM and avoids
rerunning full evaluation. Writes meshes into tc2_backup/experiments/<EXP>/seed<S>/reconstructions/.

Latent indices are looked up from results_rerun_fixed_eval.json's per_shape order,
which matches train_shapes.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import extract_mesh_marching_cubes, load_model_and_config  # noqa: E402

TARGETS = ["airplane_0001", "chair_0001", "table_0001"]
EXPERIMENTS = ["EXP-01", "EXP-02", "EXP-04", "EXP-06", "EXP-11"]
SEED = "seed42"
MC_RES = 128


def export_for_experiment(exp_id: str) -> None:
    exp_dir_local = ROOT / "tc2_backup" / "experiments" / exp_id / SEED
    rerun_json = ROOT / "experiments" / exp_id / SEED / "results_rerun_fixed_eval.json"
    recon_out = exp_dir_local / "reconstructions"
    recon_out.mkdir(parents=True, exist_ok=True)

    with open(rerun_json) as f:
        ps = json.load(f)["per_shape"]
    idx_map = {p["shape"]: i for i, p in enumerate(ps)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, latent_codes, config, ckpt_path = load_model_and_config(
        str(exp_dir_local), device, checkpoint_mode="auto"
    )
    print(f"[{exp_id}] loaded {Path(ckpt_path).name} on {device}")

    for name in TARGETS:
        if name not in idx_map:
            print(f"  [{name}] missing in per_shape, skip")
            continue
        out_path = recon_out / f"{name}.obj"
        if out_path.exists():
            print(f"  [{name}] already exists, skip")
            continue

        idx = idx_map[name]
        with torch.no_grad():
            z = latent_codes(torch.tensor([idx], device=device))
        mesh = extract_mesh_marching_cubes(
            model, z, resolution=MC_RES, device=device, sphere_clip=True
        )
        if mesh is None or len(mesh.vertices) < 10:
            print(f"  [{name}] mesh extraction failed")
            continue
        mesh.export(str(out_path))
        print(f"  [{name}] -> {out_path.relative_to(ROOT)}  (V={len(mesh.vertices)} F={len(mesh.faces)})")


def main() -> None:
    for exp in EXPERIMENTS:
        export_for_experiment(exp)


if __name__ == "__main__":
    main()

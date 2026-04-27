#!/usr/bin/env python3
"""Extract reconstruction meshes for a small set of shapes from trained checkpoints.

For each (EXP_ID, seed) pair, load the final checkpoint and write a .obj mesh
for every shape name in --shapes (one per category, typically) into
experiments/EXP-XX/seed{seed}/sample_reconstructions/.

This is lightweight compared to a full evaluate.py run because it skips
every shape not listed, skips metric computation, and skips cleanup.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import extract_mesh_marching_cubes, load_model_and_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True,
                   help="e.g. experiments/EXP-04/seed42")
    p.add_argument("--shapes", nargs="*", default=None,
                   help="Shape names, e.g. airplane_0001 chair_0001 table_0001. "
                        "If omitted with --all_shapes, every shape in train_shapes.json is processed.")
    p.add_argument("--all_shapes", action="store_true",
                   help="Process every shape in train_shapes.json (overrides --shapes)")
    p.add_argument("--mc_resolution", type=int, default=128)
    p.add_argument("--out_subdir", default="sample_reconstructions")
    p.add_argument("--sphere_clip", action="store_true", default=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, latent_codes, config, ckpt_path = load_model_and_config(
        args.exp_dir, device, checkpoint_mode="auto",
    )

    train_shapes_path = Path(args.exp_dir) / "train_shapes.json"
    with train_shapes_path.open() as f:
        train_shapes = json.load(f)

    name_to_idx: dict[str, int] = {}
    if isinstance(train_shapes, list):
        for idx, name in enumerate(train_shapes):
            name_to_idx[name] = idx
    elif isinstance(train_shapes, dict):
        for name, idx in train_shapes.items():
            name_to_idx[name] = int(idx)
    else:
        raise ValueError(f"Unexpected train_shapes.json type: {type(train_shapes)}")

    out_dir = Path(args.exp_dir) / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_shapes:
        target_shapes = sorted(name_to_idx.keys())
    elif args.shapes:
        target_shapes = list(args.shapes)
    else:
        raise SystemExit("Pass either --shapes or --all_shapes")

    n_ok = n_fail = n_skip = 0
    for shape in target_shapes:
        if shape not in name_to_idx:
            print(f"[skip] {shape} not in train_shapes.json")
            n_skip += 1
            continue
        out_path = out_dir / f"{shape}.obj"
        if out_path.exists() and out_path.stat().st_size > 0:
            n_ok += 1
            continue  # already extracted, allow resume
        idx = name_to_idx[shape]
        z = latent_codes.embedding.weight[idx].detach().unsqueeze(0)
        try:
            mesh = extract_mesh_marching_cubes(
                model, z, resolution=args.mc_resolution,
                device=device, sphere_clip=args.sphere_clip,
            )
        except Exception as e:
            print(f"[fail] {shape}: {e}")
            n_fail += 1
            continue
        if mesh is None or len(mesh.faces) == 0:
            print(f"[fail] {shape}: empty mesh")
            n_fail += 1
            continue
        mesh.export(out_path)
        n_ok += 1
        if n_ok % 25 == 0:
            print(f"[progress] {n_ok}/{len(target_shapes)} ok")
    print(f"[summary] ok={n_ok} fail={n_fail} skip={n_skip}")


if __name__ == "__main__":
    main()

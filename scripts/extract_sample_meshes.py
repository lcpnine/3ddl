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
    p.add_argument("--shapes", nargs="+", required=True,
                   help="Shape names, e.g. airplane_0001 chair_0001 table_0001")
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

    for shape in args.shapes:
        if shape not in name_to_idx:
            print(f"[skip] {shape} not in train_shapes.json")
            continue
        idx = name_to_idx[shape]
        z = latent_codes.embedding.weight[idx].detach().unsqueeze(0)
        try:
            mesh = extract_mesh_marching_cubes(
                model, z, resolution=args.mc_resolution,
                device=device, sphere_clip=args.sphere_clip,
            )
        except Exception as e:
            print(f"[fail] {shape}: {e}")
            continue
        out_path = out_dir / f"{shape}.obj"
        mesh.export(out_path)
        print(f"[ok] {shape}: {out_path}  ({len(mesh.faces)} faces)")


if __name__ == "__main__":
    main()

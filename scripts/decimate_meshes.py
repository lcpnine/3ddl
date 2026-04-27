#!/usr/bin/env python3
"""Quadric-decimate every .obj in --src to ~target_faces and write to --dst.

Used to make the per-experiment all_reconstructions/ directories small
enough to tar+pull over a slow link. 3000 faces is more than enough for
matplotlib mplot3d visualization of post-fix meshes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--target_faces", type=int, default=3000)
    args = p.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    objs = sorted(args.src.glob("*.obj"))
    print(f"Decimating {len(objs)} meshes from {args.src} -> {args.dst}")

    for i, src_path in enumerate(objs, start=1):
        dst_path = args.dst / src_path.name
        if dst_path.exists() and dst_path.stat().st_size > 0:
            continue
        try:
            mesh = trimesh.load(src_path, process=False, force="mesh")
            n = len(mesh.faces)
            if n > args.target_faces:
                # trimesh 4.x uses face_count keyword.
                try:
                    mesh = mesh.simplify_quadric_decimation(face_count=args.target_faces)
                except TypeError:
                    pct = max(0.01, min(0.99, args.target_faces / n))
                    mesh = mesh.simplify_quadric_decimation(percent=pct)
            mesh.export(dst_path)
        except Exception as e:
            print(f"[fail] {src_path.name}: {e}")
        if i % 25 == 0:
            print(f"[progress] {i}/{len(objs)}")
    print("DONE")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render a grid comparing ground-truth meshes to per-experiment reconstructions.

Rows: shapes (airplane / chair / table)
Cols: GT, EXP-01, EXP-02, EXP-03, EXP-04, EXP-05, EXP-06, EXP-11

Loads .obj files from experiments/EXP-XX/seed42/sample_reconstructions/ and
ground-truth meshes from data/processed_shapenet/gt_meshes/ (or the local
data/processed/gt_meshes directory). Renders each with matplotlib's
mplot3d using a shaded Poly3DCollection at a fixed camera angle.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-codex"))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
OUTDIR = EXP_DIR / "figures"

EXPERIMENTS = [
    ("EXP-01", "01\n100%\nno Eik"),
    ("EXP-02", "02\n100%\n+Eik"),
    ("EXP-03", "03\n50%\n+Eik"),
    ("EXP-04", "04\n10%\n+Eik"),
    ("EXP-05", "05\n5%\n+Eik"),
    ("EXP-06", "06\n10%\n+Eik+PE6"),
    ("EXP-11", "11\n10%\n+Eik+PE4"),
]

SHAPES = ["airplane_0003", "chair_0003", "table_0003"]
SEED = "seed42"


def load_mesh(path: Path) -> trimesh.Trimesh | None:
    if not path.exists():
        return None
    try:
        m = trimesh.load(path, process=False, force="mesh")
        if hasattr(m, "vertices") and len(m.vertices) > 0:
            return m
    except Exception as exc:
        print(f"[warn] failed to load {path}: {exc}")
    return None


def render_mesh(ax, mesh: trimesh.Trimesh | None, title: str, azim: float = 35.0) -> None:
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)
    if mesh is None or len(mesh.faces) == 0:
        ax.text(0.5, 0.5, 0.5, "(no mesh)", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#aa0000")
        return

    verts = mesh.vertices.copy()
    faces = mesh.faces

    # Normalize to unit sphere for visual consistency across experiments.
    centre = 0.5 * (verts.max(axis=0) + verts.min(axis=0))
    verts = verts - centre
    scale = np.max(np.linalg.norm(verts, axis=1))
    if scale > 0:
        verts = verts / scale

    # Preprocess stores meshes with y-axis as vertical (ShapeNet convention).
    # mplot3d defaults to z-up, so swap axes: plot (x, z, y) so the vertical
    # dimension of the shape matches the vertical axis of the plot.
    verts = verts[:, [0, 2, 1]]

    tri = verts[faces]  # (F, 3, 3)
    ls = LightSource(azdeg=315, altdeg=45)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    normals = normals / lengths
    intensity = ls.shade_normals(normals, fraction=1.0)
    colors = plt.cm.bone(0.35 + 0.55 * intensity)

    pc = Poly3DCollection(tri, facecolors=colors, edgecolors=(0, 0, 0, 0.04),
                          linewidths=0.15)
    ax.add_collection3d(pc)

    limit = 1.05
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=azim)


def find_gt_mesh(shape: str) -> Path | None:
    candidates = [
        ROOT / "data" / "processed_shapenet" / "gt_meshes" / f"{shape}.obj",
        ROOT / "data" / "processed" / "gt_meshes" / f"{shape}.obj",
        ROOT / "experiments" / "gt_meshes" / f"{shape}.obj",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_name", default="mesh_comparison.png")
    p.add_argument("--azim", type=float, default=35.0)
    args = p.parse_args()

    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 220,
    })

    n_cols = 1 + len(EXPERIMENTS)  # GT + each experiment
    n_rows = len(SHAPES)

    fig = plt.figure(figsize=(2.1 * n_cols, 2.3 * n_rows + 0.3))
    axes = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1, projection="3d")
            row.append(ax)
        axes.append(row)

    col_titles = ["GT"] + [label for _, label in EXPERIMENTS]

    for r, shape in enumerate(SHAPES):
        gt_path = find_gt_mesh(shape)
        gt_title = col_titles[0] if r == 0 else ""
        render_mesh(axes[r][0], load_mesh(gt_path) if gt_path else None,
                    gt_title, azim=args.azim)
        # Left-side row label via axis text so the shape name stays visible.
        axes[r][0].text2D(-0.08, 0.5, shape,
                          transform=axes[r][0].transAxes,
                          rotation=90, ha="center", va="center", fontsize=10,
                          fontweight="bold")
        for c, (exp, label) in enumerate(EXPERIMENTS, start=1):
            recon_path = EXP_DIR / exp / SEED / "sample_reconstructions" / f"{shape}.obj"
            mesh = load_mesh(recon_path)
            title = col_titles[c] if r == 0 else ""
            render_mesh(axes[r][c], mesh, title, azim=args.azim)

    fig.suptitle(
        "Reconstructed meshes per experiment vs. ground truth (seed 42)",
        fontsize=13, y=0.995,
    )
    fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.02,
                        wspace=0.05, hspace=0.05)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / args.out_name
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the report's Fig. 2 qualitative mesh panel.

This figure is report-specific and intentionally does not depend on the
`experiments/figures/per_shape_comparisons` strips, which may carry older
labels or subset choices. Instead, it renders the current mesh assets
directly from the experiment directories used in the final report.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-codex"))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "report" / "figures" / "qualitative_meshes.png"
EXP_DIR = ROOT / "experiments"
GT_DIR_CANDIDATES = [
    ROOT / "data" / "processed_shapenet" / "gt_meshes",
    ROOT / "data" / "processed" / "gt_meshes",
    ROOT / "tc2_backup" / "data_processed_shapenet" / "gt_meshes",
]

SHAPES = ["airplane_0001", "chair_0001", "table_0001"]
EXPERIMENTS = [
    ("GT", None),
    ("Baseline\nEXP-01", "EXP-01"),
    ("100% + Eik\nEXP-02", "EXP-02"),
    ("50% + Eik\nEXP-03", "EXP-03"),
    ("10% + Eik\nEXP-04", "EXP-04"),
    ("10% + Eik + PE4\nEXP-11", "EXP-11"),
    ("10% + Eik + PE6\nEXP-06", "EXP-06"),
]
RECON_SUBDIR_CANDIDATES = [
    "sample_reconstructions",
    "all_reconstructions_decim",
    "all_reconstructions",
    "reconstructions",
]


def gt_path_for(shape: str) -> Path | None:
    for d in GT_DIR_CANDIDATES:
        p = d / f"{shape}.obj"
        if p.exists():
            return p
    return None


def recon_path_for(exp_id: str, shape: str) -> Path | None:
    for subdir in RECON_SUBDIR_CANDIDATES:
        p = EXP_DIR / exp_id / "seed42" / subdir / f"{shape}.obj"
        if p.exists():
            return p
    return None


def load_mesh(path: Path | None) -> trimesh.Trimesh | None:
    if path is None or not path.exists():
        return None
    try:
        mesh = trimesh.load(path, process=False, force="mesh")
        if hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
            return mesh
    except Exception:
        pass
    return None


def render_mesh_to_image(mesh: trimesh.Trimesh | None) -> np.ndarray:
    if mesh is None or len(mesh.faces) == 0:
        return np.full((360, 360, 3), 255, dtype=np.uint8)

    max_faces = 40000
    if len(mesh.faces) > max_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(max_faces)
        except Exception:
            pass

    verts = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.faces)

    center = 0.5 * (verts.max(axis=0) + verts.min(axis=0))
    verts = verts - center
    scale = np.max(np.linalg.norm(verts, axis=1))
    if scale > 0:
        verts = verts / scale
    verts = verts[:, [0, 2, 1]]

    fig = plt.figure(figsize=(3.2, 3.2), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=faces,
        color="#6f8fa6",
        edgecolor="none",
        alpha=0.98,
        shade=True,
    )
    ax.view_init(elev=20, azim=35)
    limit = 1.05
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(SHAPES)
    n_cols = len(EXPERIMENTS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, shape in enumerate(SHAPES):
        for col, (label, exp_id) in enumerate(EXPERIMENTS):
            ax = axes[row, col]
            if exp_id is None:
                mesh = load_mesh(gt_path_for(shape))
            else:
                mesh = load_mesh(recon_path_for(exp_id, shape))

            if mesh is None:
                ax.text(
                    0.5, 0.5, "N/A",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#888888",
                )
            else:
                ax.imshow(render_mesh_to_image(mesh))
            ax.axis("off")

            if row == 0:
                ax.set_title(label, fontsize=10, fontweight="bold", pad=8)
            if col == 0:
                ax.set_ylabel(
                    shape.split("_")[0].capitalize(),
                    fontsize=11,
                    fontweight="bold",
                    rotation=90,
                    labelpad=12,
                )

    plt.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render a per-shape comparison strip for every shape in the dataset.

For each shape with a ground-truth mesh and at least one reconstruction
under experiments/EXP-XX/seed42/all_reconstructions/, render a single PNG
showing GT next to the reconstruction from every post-fix experiment.
Output goes to experiments/figures/per_shape_comparisons/{shape}.png.

Then assemble the per-shape PNGs into a multi-page PDF per category
(experiments/figures/comparison_{category}.pdf).

Run after scripts/extract_sample_meshes.py --all_shapes has produced all
2,093 .obj files (7 experiments x 299 shapes).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-codex"))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
FIG_DIR = EXP_DIR / "figures"
PER_SHAPE_DIR = FIG_DIR / "per_shape_comparisons"
GT_DIR_CANDIDATES = [
    ROOT / "data" / "processed_shapenet" / "gt_meshes",
    ROOT / "data" / "processed" / "gt_meshes",
]

EXPERIMENTS = [
    ("EXP-01", "01\n100% no Eik"),
    ("EXP-02", "02\n100% +Eik"),
    ("EXP-03", "03\n50% +Eik"),
    ("EXP-04", "04\n10% +Eik"),
    ("EXP-05", "05\n5% +Eik"),
    ("EXP-06", "06\n10% +Eik+PE6"),
    ("EXP-11", "11\n10% +Eik+PE4"),
]
CATEGORIES = ["airplane", "chair", "table"]
SEED = "seed42"
RECON_SUBDIR = "all_reconstructions"


def gt_path_for(shape: str) -> Path | None:
    for d in GT_DIR_CANDIDATES:
        p = d / f"{shape}.obj"
        if p.exists():
            return p
    return None


def load_mesh(path: Path | None) -> trimesh.Trimesh | None:
    if path is None or not path.exists():
        return None
    try:
        m = trimesh.load(path, process=False, force="mesh")
        if hasattr(m, "vertices") and len(m.vertices) > 0:
            return m
    except Exception:
        pass
    return None


def render_mesh(ax, mesh: trimesh.Trimesh | None, title: str, azim: float = 35.0) -> None:
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)
    if mesh is None or len(mesh.faces) == 0:
        ax.text(0.5, 0.5, 0.5, "(no mesh)", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#aa0000")
        return

    verts = mesh.vertices.copy()
    faces = mesh.faces

    centre = 0.5 * (verts.max(axis=0) + verts.min(axis=0))
    verts = verts - centre
    scale = np.max(np.linalg.norm(verts, axis=1))
    if scale > 0:
        verts = verts / scale
    verts = verts[:, [0, 2, 1]]

    tri = verts[faces]
    ls = LightSource(azdeg=315, altdeg=45)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    normals = normals / lengths
    intensity = ls.shade_normals(normals, fraction=1.0)
    colors = plt.cm.bone(0.35 + 0.55 * intensity)

    pc = Poly3DCollection(tri, facecolors=colors, edgecolors=(0, 0, 0, 0.04),
                          linewidths=0.1)
    ax.add_collection3d(pc)

    limit = 1.05
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=azim)


def discover_shapes() -> list[str]:
    """Shapes that exist in train_shapes.json AND have a GT mesh."""
    train_shapes_json = EXP_DIR / "EXP-01" / SEED / "train_shapes.json"
    import json
    with train_shapes_json.open() as f:
        shapes = json.load(f)
    if isinstance(shapes, dict):
        shapes = list(shapes.keys())
    return sorted(set(shapes))


def render_one_strip(shape: str, out_path: Path, azim: float = 35.0) -> bool:
    n_cols = 1 + len(EXPERIMENTS)
    fig = plt.figure(figsize=(1.7 * n_cols, 2.3))
    axes = []
    for c in range(n_cols):
        ax = fig.add_subplot(1, n_cols, c + 1, projection="3d")
        axes.append(ax)

    gt_mesh = load_mesh(gt_path_for(shape))
    render_mesh(axes[0], gt_mesh, "GT", azim=azim)
    for c, (exp, label) in enumerate(EXPERIMENTS, start=1):
        recon = EXP_DIR / exp / SEED / RECON_SUBDIR / f"{shape}.obj"
        render_mesh(axes[c], load_mesh(recon), label, azim=azim)

    fig.suptitle(shape, fontsize=10, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.78, bottom=0.02,
                        wspace=0.05)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return True


def assemble_pdf(category: str, shapes: list[str], out_pdf: Path,
                 azim: float = 35.0, rows_per_page: int = 6) -> None:
    """Multi-page PDF: rows_per_page shapes per page, each row is one strip."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    n_cols = 1 + len(EXPERIMENTS)
    with PdfPages(out_pdf) as pdf:
        for page_start in range(0, len(shapes), rows_per_page):
            page_shapes = shapes[page_start:page_start + rows_per_page]
            n_rows = len(page_shapes)
            fig = plt.figure(figsize=(1.7 * n_cols, 2.0 * n_rows + 0.5))
            for r, shape in enumerate(page_shapes):
                gt_mesh = load_mesh(gt_path_for(shape))
                ax = fig.add_subplot(n_rows, n_cols, r * n_cols + 1,
                                     projection="3d")
                render_mesh(ax, gt_mesh, "GT" if r == 0 else "", azim=azim)
                ax.text2D(-0.08, 0.5, shape, transform=ax.transAxes,
                          rotation=90, ha="center", va="center", fontsize=8,
                          fontweight="bold")
                for c, (exp, label) in enumerate(EXPERIMENTS, start=1):
                    recon = EXP_DIR / exp / SEED / RECON_SUBDIR / f"{shape}.obj"
                    ax2 = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1,
                                          projection="3d")
                    render_mesh(ax2, load_mesh(recon),
                                label if r == 0 else "", azim=azim)
            page_num = page_start // rows_per_page + 1
            total_pages = (len(shapes) + rows_per_page - 1) // rows_per_page
            fig.suptitle(f"{category} reconstructions  (page {page_num}/{total_pages})",
                         fontsize=11, y=0.998)
            fig.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.02,
                                wspace=0.05, hspace=0.1)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"[pdf] wrote {out_pdf} ({len(shapes)} shapes)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--azim", type=float, default=35.0)
    p.add_argument("--rows_per_page", type=int, default=6)
    p.add_argument("--skip_strips", action="store_true",
                   help="Skip per-shape PNG strips; only build PDFs")
    p.add_argument("--skip_pdfs", action="store_true",
                   help="Skip multi-page PDFs; only render per-shape PNGs")
    args = p.parse_args()

    shapes = discover_shapes()
    by_cat: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    for s in shapes:
        cat = s.split("_")[0]
        if cat in by_cat:
            by_cat[cat].append(s)

    for cat, lst in by_cat.items():
        print(f"  {cat}: {len(lst)} shapes")

    if not args.skip_strips:
        PER_SHAPE_DIR.mkdir(parents=True, exist_ok=True)
        for i, shape in enumerate(shapes, start=1):
            out_png = PER_SHAPE_DIR / f"{shape}.png"
            render_one_strip(shape, out_png, azim=args.azim)
            if i % 25 == 0:
                print(f"[strip] {i}/{len(shapes)}")

    if not args.skip_pdfs:
        for cat, lst in by_cat.items():
            if not lst:
                continue
            out_pdf = FIG_DIR / f"comparison_{cat}.pdf"
            assemble_pdf(cat, lst, out_pdf, azim=args.azim,
                         rows_per_page=args.rows_per_page)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate per-shape qualitative comparison images for all categories.

For each category (airplane, chair, table), saves one image per shape showing
GT vs all experiments side-by-side in a single row.

Output: report/figures/qualitative/{category}/{shape_name}.png
"""

import io
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-codex"))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "tc2_backup"
GT_DIR = BACKUP / "data_processed_shapenet" / "gt_meshes"
QUALDIR = ROOT / "report" / "figures" / "qualitative"

CATEGORIES = ["airplane", "chair", "table"]

EXPERIMENTS = [
    ("GT",                   None),
    ("EXP-01\n(baseline)",   "EXP-01"),
    ("EXP-02\n(100%+Eik)",   "EXP-02"),
    ("EXP-03\n(50%+Eik)",    "EXP-03"),
    ("EXP-04\n(10%+Eik)",    "EXP-04"),
    ("EXP-05\n(5%+Eik)",     "EXP-05"),
]


def render_mesh_to_image(mesh: trimesh.Trimesh):
    """Render a mesh to a 2D image using matplotlib 3D projection."""
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    # Center and scale mesh
    vertices = np.array(mesh.vertices)
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    scale = (vertices.max(axis=0) - vertices.min(axis=0)).max()
    vertices = (vertices - center) / scale

    faces = np.array(mesh.faces)

    # Subsample for rendering speed if mesh is large
    if len(faces) > 50000:
        indices = np.random.choice(len(faces), 50000, replace=False)
        faces = faces[indices]

    # Plot with light shading
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color="#6ba3d6",
        edgecolor="none",
        alpha=0.9,
        shade=True,
    )

    # Set view angle
    ax.view_init(elev=25, azim=135)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.6, 0.6)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Render to array via PNG buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf))[:, :, :3]
    return img


def main():
    for category in CATEGORIES:
        cat_dir = QUALDIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        shapes = sorted(f.stem for f in GT_DIR.glob(f"{category}_*.obj"))
        print(f"\n[{category}] {len(shapes)} shapes")

        for shape_name in shapes:
            cols = []
            for label, exp_id in EXPERIMENTS:
                if exp_id is None:
                    path = GT_DIR / f"{shape_name}.obj"
                else:
                    path = (
                        BACKUP
                        / "experiments"
                        / exp_id
                        / "seed42"
                        / "reconstructions"
                        / f"{shape_name}.obj"
                    )
                cols.append((label, path))

            n_cols = len(cols)
            fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.5))

            for ax, (label, path) in zip(axes, cols):
                if path.exists():
                    try:
                        mesh = trimesh.load(str(path), force="mesh")
                        img = render_mesh_to_image(mesh)
                        ax.imshow(img)
                    except Exception as e:
                        ax.text(
                            0.5, 0.5, f"Error:\n{e}",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=7, color="red",
                        )
                else:
                    ax.text(
                        0.5, 0.5, "N/A",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=10, color="#aaa",
                    )
                ax.set_title(label, fontsize=9, fontweight="bold")
                ax.axis("off")

            fig.suptitle(shape_name, fontsize=12, fontweight="bold", y=1.01)
            plt.tight_layout()
            out = cat_dir / f"{shape_name}.png"
            fig.savefig(out, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"  {out.name}")


if __name__ == "__main__":
    main()

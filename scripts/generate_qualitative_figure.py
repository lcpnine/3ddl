#!/usr/bin/env python3
"""
Generate qualitative mesh comparison figure for the report.

Shows GT vs EXP-01 (baseline) vs EXP-02 (Eikonal) vs EXP-04 (10%+Eik)
for one shape per category (airplane, chair, table).
Renders each mesh from a fixed viewpoint using trimesh + matplotlib.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-codex"))

import matplotlib.pyplot as plt
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "tc2_backup"
GT_DIR = BACKUP / "data_processed_shapenet" / "gt_meshes"
OUTDIR = ROOT / "report" / "figures"

# Pick one shape per category. Meshes are produced by scripts/export_qualitative_meshes.py
# (sphere-clipped marching cubes with the fixed evaluator).
SHAPES = ["airplane_0001", "chair_0001", "table_0001"]
EXPERIMENTS = [
    ("GT", None),
    ("EXP-01\n(100% base)", "EXP-01"),
    ("EXP-04\n(10% +Eik)", "EXP-04"),
    ("EXP-11\n(10% +Eik+PE L=4)", "EXP-11"),
    ("EXP-06\n(10% +Eik+PE L=6)", "EXP-06"),
]


def render_mesh_to_image(mesh: trimesh.Trimesh):
    """Render a mesh to a 2D image using matplotlib 3D projection."""
    import io
    from PIL import Image

    # Decimate heavy meshes (PE extraction produces >1M faces) so structure is
    # preserved instead of destroyed by random face subsampling.
    max_faces = 40000
    if len(mesh.faces) > max_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(max_faces)
        except Exception:
            pass

    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    # Center and scale mesh
    vertices = np.array(mesh.vertices)
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    scale = (vertices.max(axis=0) - vertices.min(axis=0)).max()
    vertices = (vertices - center) / scale

    faces = np.array(mesh.faces)

    # Plot with light shading
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color="#6ba3d6",
        edgecolor="none",
        alpha=0.95,
        shade=True,
    )

    # Set view angle
    ax.view_init(elev=25, azim=135)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.6, 0.6)
    ax.set_box_aspect([1, 1, 1])
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Render to array via PNG buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf))[:, :, :3]
    return img


def find_valid_shapes():
    """Find shapes present in all experiment reconstructions."""
    valid = {}
    for category_prefix in ["airplane", "chair", "table"]:
        # Get shapes available in all experiments
        available = None
        for _, exp_id in EXPERIMENTS:
            if exp_id is None:
                # GT
                shapes = {
                    f.stem
                    for f in GT_DIR.glob(f"{category_prefix}_*.obj")
                }
            else:
                recon_dir = BACKUP / "experiments" / exp_id / "seed42" / "reconstructions"
                shapes = {
                    f.stem
                    for f in recon_dir.glob(f"{category_prefix}_*.obj")
                }
            if available is None:
                available = shapes
            else:
                available &= shapes

        if available:
            # Pick the first one alphabetically
            valid[category_prefix] = sorted(available)[0]
        else:
            print(f"WARNING: No shape available in all experiments for {category_prefix}")

    return valid


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    valid_shapes = find_valid_shapes()
    print(f"Selected shapes: {valid_shapes}")

    categories = sorted(valid_shapes.keys())
    n_rows = len(categories)
    n_cols = len(EXPERIMENTS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, cat in enumerate(categories):
        shape_name = valid_shapes[cat]
        for col, (label, exp_id) in enumerate(EXPERIMENTS):
            ax = axes[row, col]

            if exp_id is None:
                mesh_path = GT_DIR / f"{shape_name}.obj"
            else:
                mesh_path = (
                    BACKUP
                    / "experiments"
                    / exp_id
                    / "seed42"
                    / "reconstructions"
                    / f"{shape_name}.obj"
                )

            if mesh_path.exists():
                try:
                    mesh = trimesh.load(str(mesh_path), force="mesh")
                    img = render_mesh_to_image(mesh)
                    ax.imshow(img)
                except Exception as e:
                    ax.text(
                        0.5, 0.5, f"Error:\n{e}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8,
                    )
            else:
                ax.text(
                    0.5, 0.5, "Not\navailable",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="#999",
                )

            ax.axis("off")

            if row == 0:
                ax.set_title(label, fontsize=12, fontweight="bold", pad=8)
            if col == 0:
                ax.set_ylabel(
                    cat.capitalize(),
                    fontsize=12,
                    fontweight="bold",
                    rotation=90,
                    labelpad=12,
                )
                ax.yaxis.set_visible(True)
                ax.yaxis.label.set_visible(True)

    fig.suptitle(
        "Qualitative Reconstruction Comparison (fixed evaluator, sphere-clipped)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    out_path = OUTDIR / "qualitative_meshes.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

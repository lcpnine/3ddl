"""
Generate or download 3D meshes for development and pipeline validation.

Primary: attempts to download Thingi10K meshes from known sources.
Fallback: generates diverse parametric meshes using trimesh primitives,
providing watertight shapes suitable for SDF sampling experiments.

Usage:
    python scripts/download_thingi10k.py --output_dir data/raw --num_meshes 20
"""

import argparse
import os
import urllib.request
import trimesh
import trimesh.creation
import numpy as np


# Thingi10K download sources (try in order)
THINGI10K_SOURCES = [
    "https://ten-thousand-models.appspot.com/api/mesh/{model_id}",
]

THINGI10K_IDS = [
    49911, 50384, 51922, 53698, 54500, 55105, 56862, 57803, 59003, 60502,
    61340, 62784, 63901, 64500, 65200, 66100, 67500, 68200, 69100, 70500,
]


def generate_primitive_meshes(num_meshes: int) -> list[tuple[str, trimesh.Trimesh]]:
    """Generate diverse parametric meshes as fallback when downloads fail."""
    generators = [
        ("sphere", lambda: trimesh.creation.icosphere(subdivisions=3, radius=1.0)),
        ("box", lambda: trimesh.creation.box(extents=[1.5, 1.0, 0.8])),
        ("cylinder", lambda: trimesh.creation.cylinder(radius=0.5, height=1.5, sections=32)),
        ("capsule", lambda: trimesh.creation.capsule(radius=0.4, height=1.2)),
        ("torus_thick", lambda: _make_torus(0.6, 0.25)),
        ("torus_thin", lambda: _make_torus(0.7, 0.12)),
        ("cone", lambda: trimesh.creation.cone(radius=0.6, height=1.4, sections=32)),
        ("ellipsoid", lambda: _make_ellipsoid(1.0, 0.6, 0.4)),
        ("tall_box", lambda: trimesh.creation.box(extents=[0.5, 0.5, 2.0])),
        ("flat_disc", lambda: trimesh.creation.cylinder(radius=0.8, height=0.15, sections=48)),
        ("ico_lo", lambda: trimesh.creation.icosphere(subdivisions=1, radius=1.0)),
        ("ico_hi", lambda: trimesh.creation.icosphere(subdivisions=4, radius=0.9)),
        ("squat_cyl", lambda: trimesh.creation.cylinder(radius=0.9, height=0.4, sections=24)),
        ("thin_cyl", lambda: trimesh.creation.cylinder(radius=0.15, height=1.8, sections=16)),
        ("wide_box", lambda: trimesh.creation.box(extents=[2.0, 1.0, 0.3])),
        ("cube", lambda: trimesh.creation.box(extents=[1.0, 1.0, 1.0])),
        ("capsule_long", lambda: trimesh.creation.capsule(radius=0.25, height=2.0)),
        ("torus_fat", lambda: _make_torus(0.4, 0.35)),
        ("hex_prism", lambda: trimesh.creation.cylinder(radius=0.6, height=1.0, sections=6)),
        ("oct_prism", lambda: trimesh.creation.cylinder(radius=0.6, height=1.0, sections=8)),
    ]
    return generators[:num_meshes]


def _make_torus(major_r: float, minor_r: float) -> trimesh.Trimesh:
    """Create a torus mesh."""
    angles_major = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    angles_minor = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    vertices = []
    for a in angles_major:
        for b in angles_minor:
            x = (major_r + minor_r * np.cos(b)) * np.cos(a)
            y = (major_r + minor_r * np.cos(b)) * np.sin(a)
            z = minor_r * np.sin(b)
            vertices.append([x, y, z])
    vertices = np.array(vertices)
    n_maj, n_min = len(angles_major), len(angles_minor)
    faces = []
    for i in range(n_maj):
        for j in range(n_min):
            v0 = i * n_min + j
            v1 = i * n_min + (j + 1) % n_min
            v2 = ((i + 1) % n_maj) * n_min + (j + 1) % n_min
            v3 = ((i + 1) % n_maj) * n_min + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))


def _make_ellipsoid(a: float, b: float, c: float) -> trimesh.Trimesh:
    """Create an ellipsoid by scaling an icosphere."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mesh.vertices *= np.array([a, b, c])
    return mesh


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalize mesh to fit inside unit sphere (centered at origin)."""
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid
    max_dist = np.max(np.linalg.norm(mesh.vertices, axis=1))
    if max_dist > 0:
        mesh.vertices /= max_dist
    return mesh


def try_download_thingi10k(model_id: int, output_dir: str) -> str | None:
    """Try to download a Thingi10K mesh. Returns filepath or None."""
    filepath = os.path.join(output_dir, f"thingi10k_{model_id}.stl")
    if os.path.exists(filepath):
        print(f"  Already exists: {filepath}")
        return filepath

    for source_template in THINGI10K_SOURCES:
        url = source_template.format(model_id=model_id)
        try:
            urllib.request.urlretrieve(url, filepath)
            return filepath
        except Exception:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description="Download/generate meshes for development")
    parser.add_argument("--output_dir", type=str, default="data/raw",
                        help="Directory to save meshes")
    parser.add_argument("--num_meshes", type=int, default=20,
                        help="Number of meshes (max 20)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    downloaded = []
    watertight = []
    non_watertight = []
    source = "unknown"

    # Try downloading from Thingi10K first
    print(f"Attempting to download Thingi10K meshes to {args.output_dir}/\n")
    ids_to_try = THINGI10K_IDS[:args.num_meshes]

    for model_id in ids_to_try:
        filepath = try_download_thingi10k(model_id, args.output_dir)
        if filepath is not None:
            try:
                mesh = trimesh.load(filepath, force='mesh')
                mesh = normalize_mesh(mesh)
                mesh.export(filepath)
                downloaded.append(filepath)
                if mesh.is_watertight:
                    watertight.append(filepath)
                else:
                    non_watertight.append(filepath)
                print(f"  Downloaded: {os.path.basename(filepath)} "
                      f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")
            except Exception as e:
                print(f"  Failed to load {filepath}: {e}")
                os.remove(filepath)

    if downloaded:
        source = "thingi10k"
    else:
        # Fallback: generate parametric meshes
        print("Thingi10K API unavailable. Generating parametric meshes as fallback.\n")
        source = "parametric"
        generators = generate_primitive_meshes(args.num_meshes)

        for name, gen_fn in generators:
            filepath = os.path.join(args.output_dir, f"parametric_{name}.stl")
            if os.path.exists(filepath):
                print(f"  Already exists: {filepath}")
                mesh = trimesh.load(filepath, force='mesh')
            else:
                mesh = gen_fn()
                mesh = normalize_mesh(mesh)
                mesh.export(filepath)

            is_wt = mesh.is_watertight
            status = "watertight" if is_wt else "NOT watertight"
            print(f"  {os.path.basename(filepath)}: {len(mesh.vertices)} verts, "
                  f"{len(mesh.faces)} faces, {status}")

            if is_wt:
                watertight.append(filepath)
            else:
                trimesh.repair.fix_normals(mesh)
                trimesh.repair.fill_holes(mesh)
                if mesh.is_watertight:
                    print(f"    -> Repaired to watertight")
                    watertight.append(filepath)
                    mesh.export(filepath)
                else:
                    non_watertight.append(filepath)

            downloaded.append(filepath)

    print(f"\n{'='*60}")
    print(f"DOWNLOAD & AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"Source:            {source}")
    print(f"Total generated:   {len(downloaded)}")
    print(f"Watertight:        {len(watertight)}")
    print(f"Non-watertight:    {len(non_watertight)}")
    print(f"Output directory:  {os.path.abspath(args.output_dir)}")

    if non_watertight:
        print(f"\nNon-watertight meshes (will use surface sampling fallback):")
        for f in non_watertight:
            print(f"  - {os.path.basename(f)}")

    print(f"\nReady for preprocessing: python scripts/preprocess.py --mesh_dir {args.output_dir}")


if __name__ == "__main__":
    main()

"""
Scan ShapeNet categories and select usable meshes for preprocessing.

Selects shapes that load successfully and have sufficient geometry.
ShapeNet meshes are rarely watertight, but our pipeline (surface sampling
+ normal offsets) works with any mesh that has faces and normals.

Usage:
    python scripts/scan_watertight.py --data_dir data/shapenet_raw --output_dir data/shapenet_raw
"""

import argparse
import json
import os
import random

import trimesh


CATEGORIES = {
    "02691156": "airplane",
    "04256520": "chair",
    "04379243": "table",
}

MIN_FACES = 100  # Minimum faces to be considered usable


def find_obj_path(cat_dir: str, shape_id: str) -> str:
    """Find the OBJ file for a shape."""
    candidates = [
        os.path.join(cat_dir, shape_id, "models", "model_normalized.obj"),
        os.path.join(cat_dir, shape_id, "model_normalized.obj"),
        os.path.join(cat_dir, shape_id, "models", "model.obj"),
        os.path.join(cat_dir, shape_id, "model.obj"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def scan_category(cat_dir: str, cat_id: str, cat_name: str, target: int = 100) -> dict:
    """Scan shapes and select usable ones (load OK + enough faces)."""
    shapes = sorted([d for d in os.listdir(cat_dir)
                     if os.path.isdir(os.path.join(cat_dir, d))])

    # Shuffle for random selection (deterministic seed)
    random.seed(42)
    random.shuffle(shapes)

    results = {
        "category_id": cat_id,
        "category_name": cat_name,
        "total": len(shapes),
        "scanned": 0,
        "selected": [],
        "too_few_faces": 0,
        "errors": 0,
    }

    for i, shape_id in enumerate(shapes):
        if len(results["selected"]) >= target:
            print(f"  Reached target of {target} usable shapes after scanning {i} shapes")
            break

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(shapes)}] found {len(results['selected'])} usable so far...")

        obj_path = find_obj_path(cat_dir, shape_id)
        if not obj_path:
            results["errors"] += 1
            continue

        try:
            mesh = trimesh.load(obj_path, force='mesh', process=False)
            if len(mesh.faces) >= MIN_FACES:
                results["selected"].append({
                    "shape_id": shape_id,
                    "obj_path": obj_path,
                    "n_vertices": len(mesh.vertices),
                    "n_faces": len(mesh.faces),
                    "watertight": bool(mesh.is_watertight),
                })
            else:
                results["too_few_faces"] += 1
        except Exception:
            results["errors"] += 1

        results["scanned"] = i + 1

    return results


def main():
    parser = argparse.ArgumentParser(description="Scan ShapeNet for usable meshes")
    parser.add_argument("--data_dir", type=str, default="data/shapenet_raw",
                        help="Directory with ShapeNet category folders")
    parser.add_argument("--output_dir", type=str, default="data/shapenet_raw",
                        help="Directory to save scan results")
    parser.add_argument("--target", type=int, default=100,
                        help="Target number of usable shapes per category")
    args = parser.parse_args()

    all_results = {}

    for cat_id, cat_name in CATEGORIES.items():
        cat_dir = os.path.join(args.data_dir, cat_id)
        if not os.path.exists(cat_dir):
            print(f"Category {cat_name} ({cat_id}) not found at {cat_dir}, skipping")
            continue

        print(f"\nScanning {cat_name} ({cat_id})...")
        results = scan_category(cat_dir, cat_id, cat_name, target=args.target)
        all_results[cat_id] = results

        n_sel = len(results["selected"])
        print(f"  Scanned: {results['scanned']} / {results['total']}")
        print(f"  Selected: {n_sel}")
        print(f"  Too few faces: {results['too_few_faces']}")
        print(f"  Errors: {results['errors']}")
        if n_sel > 0:
            faces = [s["n_faces"] for s in results["selected"]]
            print(f"  Face count range: {min(faces)} - {max(faces)}")

    # Save results
    output_path = os.path.join(args.output_dir, "mesh_selection.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n=== Summary ===")
    total_selected = 0
    for cat_id, results in all_results.items():
        n_sel = len(results["selected"])
        total_selected += n_sel
        print(f"  {results['category_name']} ({cat_id}): {n_sel} selected / {results['total']} total")
    print(f"  Total selected: {total_selected}")


if __name__ == "__main__":
    main()

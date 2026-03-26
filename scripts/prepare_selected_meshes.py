"""
Copy selected ShapeNet meshes to a flat directory for preprocessing.

Reads mesh_selection.json and copies the selected OBJ files to data/raw_shapenet/
with names like airplane_0001.obj, chair_0042.obj, etc.

Usage:
    python scripts/prepare_selected_meshes.py
"""

import json
import os
import shutil


def main():
    selection_path = "data/shapenet_raw/mesh_selection.json"
    output_dir = "data/raw_shapenet"

    with open(selection_path) as f:
        selection = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    total = 0
    for cat_id, results in selection.items():
        cat_name = results["category_name"]
        for i, shape in enumerate(results["selected"]):
            obj_path = shape["obj_path"]
            out_name = f"{cat_name}_{i:04d}.obj"
            out_path = os.path.join(output_dir, out_name)
            shutil.copy2(obj_path, out_path)
            total += 1

        print(f"{cat_name}: copied {len(results['selected'])} meshes")

    print(f"\nTotal: {total} meshes in {output_dir}")


if __name__ == "__main__":
    main()

"""
Download ShapeNet Core v2 categories from HuggingFace and extract OBJ files.

Downloads target categories as zip files, extracts model/model_normalized.obj,
and deletes the zip to save disk space.

Usage:
    python scripts/download_shapenet.py --token <HF_TOKEN> --output_dir data/shapenet_raw
"""

import argparse
import os
import zipfile

from huggingface_hub import hf_hub_download


REPO_ID = "ShapeNet/ShapeNetCore"

# Target categories
CATEGORIES = {
    "02691156": "airplane",
    "04256520": "chair",
    "04379243": "table",
}


def download_and_extract_category(cat_id: str, cat_name: str, token: str, output_dir: str):
    """Download a category zip and extract OBJ files."""
    zip_name = f"{cat_id}.zip"
    cat_output = os.path.join(output_dir, cat_id)

    if os.path.exists(cat_output) and len(os.listdir(cat_output)) > 0:
        n_existing = len(os.listdir(cat_output))
        print(f"  {cat_name} ({cat_id}): already exists with {n_existing} items, skipping download")
        return

    print(f"\nDownloading {cat_name} ({cat_id})...")
    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=zip_name,
        repo_type="dataset",
        token=token,
        local_dir=output_dir,
    )
    print(f"  Downloaded to: {zip_path}")

    # Extract
    print(f"  Extracting...")
    os.makedirs(cat_output, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)

    # Count extracted shapes
    if os.path.exists(cat_output):
        shapes = [d for d in os.listdir(cat_output)
                  if os.path.isdir(os.path.join(cat_output, d))]
        print(f"  Extracted {len(shapes)} shapes")
    else:
        print(f"  WARNING: Expected directory {cat_output} not found")

    # Delete zip to save disk
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"  Deleted zip to save disk")


def main():
    parser = argparse.ArgumentParser(description="Download ShapeNet categories from HuggingFace")
    parser.add_argument("--token", type=str, required=True, help="HuggingFace access token")
    parser.add_argument("--output_dir", type=str, default="data/shapenet_raw",
                        help="Output directory for downloaded data")
    parser.add_argument("--categories", type=str, nargs="+", default=list(CATEGORIES.keys()),
                        help="Category IDs to download (default: all 3 targets)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for cat_id in args.categories:
        cat_name = CATEGORIES.get(cat_id, "unknown")
        download_and_extract_category(cat_id, cat_name, args.token, args.output_dir)

    # Summary
    print("\n=== Download Summary ===")
    for cat_id in args.categories:
        cat_name = CATEGORIES.get(cat_id, "unknown")
        cat_dir = os.path.join(args.output_dir, cat_id)
        if os.path.exists(cat_dir):
            shapes = [d for d in os.listdir(cat_dir)
                      if os.path.isdir(os.path.join(cat_dir, d))]
            print(f"  {cat_name} ({cat_id}): {len(shapes)} shapes")
        else:
            print(f"  {cat_name} ({cat_id}): NOT FOUND")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stitch existing per-shape PNG strips into a multi-page PDF per category.

Each page contains rows_per_page PNG strips stacked vertically. The strips
are loaded as images and embedded as raster -- no matplotlib re-render, so
the PDFs are O(strips_size) rather than O(triangles).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
FIG_DIR = EXP_DIR / "figures"
STRIP_DIR = FIG_DIR / "per_shape_comparisons"
CATEGORIES = ["airplane", "chair", "table"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows_per_page", type=int, default=6)
    args = p.parse_args()

    if not STRIP_DIR.exists():
        raise SystemExit(f"No strip dir at {STRIP_DIR}")

    by_cat: dict[str, list[Path]] = {c: [] for c in CATEGORIES}
    for p in sorted(STRIP_DIR.glob("*.png")):
        cat = p.stem.split("_")[0]
        if cat in by_cat:
            by_cat[cat].append(p)

    for cat, paths in by_cat.items():
        if not paths:
            continue
        pages: list[Image.Image] = []
        for start in range(0, len(paths), args.rows_per_page):
            chunk = paths[start:start + args.rows_per_page]
            imgs = [Image.open(p).convert("RGB") for p in chunk]
            w = max(im.width for im in imgs)
            h = sum(im.height for im in imgs)
            page = Image.new("RGB", (w, h), "white")
            y = 0
            for im in imgs:
                page.paste(im, ((w - im.width) // 2, y))
                y += im.height
            pages.append(page)

        out = FIG_DIR / f"comparison_{cat}.pdf"
        first, rest = pages[0], pages[1:]
        first.save(out, save_all=True, append_images=rest, format="PDF")
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"[pdf] {out.name}: {len(paths)} shapes, {len(pages)} pages, "
              f"{size_mb:.1f} MB")


if __name__ == "__main__":
    main()

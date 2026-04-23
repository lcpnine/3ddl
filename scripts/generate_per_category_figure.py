#!/usr/bin/env python3
"""Per-category, per-experiment comparison figure.

Builds box plots of per-shape CD and NC across the 10 post-fix priority
experiments, broken down by ShapeNet category (airplane / chair / table).
Multi-seed experiments pool all seeds so every box contains every per-shape
evaluation recorded under that configuration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-codex"))

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
OUTDIR = EXP_DIR / "figures"

POST_FIX_RUNS = [
    ("EXP-01", ["seed42"], "01\n100%\nno Eik"),
    ("EXP-02", ["seed42"], "02\n100%\n+Eik"),
    ("EXP-03", ["seed42"], "03\n50%\n+Eik"),
    ("EXP-04", ["seed42", "seed123", "seed456"], "04\n10%\n+Eik"),
    ("EXP-05", ["seed42"], "05\n5%\n+Eik"),
    ("EXP-06", ["seed42", "seed123", "seed456"], "06\n10%\n+Eik+PE6"),
    ("EXP-11", ["seed42"], "11\n10%\n+Eik+PE4"),
]

CATEGORIES = ["airplane", "chair", "table"]
CAT_COLORS = {"airplane": "#1f77b4", "chair": "#d98c10", "table": "#2ca02c"}


def load_per_shape(exp: str, seed: str) -> list[dict]:
    path = EXP_DIR / exp / seed / "results.json"
    with path.open() as f:
        data = json.load(f)
    return data["per_shape"]


def collect_by_category(exp: str, seeds: list[str]) -> dict[str, dict[str, list[float]]]:
    bucket = {cat: {"cd": [], "nc": []} for cat in CATEGORIES}
    for seed in seeds:
        for entry in load_per_shape(exp, seed):
            if entry.get("status") != "ok":
                continue
            cat = entry["shape"].split("_")[0]
            if cat not in bucket:
                continue
            bucket[cat]["cd"].append(entry["chamfer_distance"])
            bucket[cat]["nc"].append(entry["normal_consistency"])
    return bucket


def draw_metric(ax, metric_key: str, ylabel: str, higher_better: bool) -> None:
    n_exp = len(POST_FIX_RUNS)
    n_cat = len(CATEGORIES)
    width = 0.22
    offsets = np.linspace(-(n_cat - 1) / 2, (n_cat - 1) / 2, n_cat) * width * 1.2
    positions = np.arange(n_exp)

    for cat_idx, cat in enumerate(CATEGORIES):
        box_positions = positions + offsets[cat_idx]
        values_per_exp = []
        for exp_id, seeds, _label in POST_FIX_RUNS:
            bucket = collect_by_category(exp_id, seeds)
            values_per_exp.append(bucket[cat][metric_key])
        bp = ax.boxplot(
            values_per_exp,
            positions=box_positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color=CAT_COLORS[cat], linewidth=1.1),
            capprops=dict(color=CAT_COLORS[cat], linewidth=1.1),
        )
        for box in bp["boxes"]:
            box.set_facecolor(CAT_COLORS[cat])
            box.set_alpha(0.55)
            box.set_edgecolor(CAT_COLORS[cat])

    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, _, label in POST_FIX_RUNS], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    if higher_better:
        ax.set_title(f"{ylabel} per ShapeNet category (higher is better)")
    else:
        ax.set_title(f"{ylabel} per ShapeNet category (lower is better)")


def legend_patches():
    from matplotlib.patches import Patch
    return [Patch(facecolor=CAT_COLORS[c], edgecolor=CAT_COLORS[c], alpha=0.55, label=c.title())
            for c in CATEGORIES]


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.8), sharex=True)
    draw_metric(axes[0], "cd", "Chamfer Distance", higher_better=False)
    draw_metric(axes[1], "nc", "Normal Consistency", higher_better=True)
    axes[1].set_xlabel("Experiment")

    axes[0].legend(handles=legend_patches(), loc="upper right", frameon=True)

    total_shapes = 0
    for exp, seeds, _lbl in POST_FIX_RUNS:
        total_shapes += sum(
            len([e for e in load_per_shape(exp, s) if e.get("status") == "ok"])
            for s in seeds
        )
    fig.suptitle(
        f"Per-category CD and NC across post-fix runs (pooled per-shape evaluations; n={total_shapes})",
        fontsize=11, y=1.01,
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / "per_category_cd_nc.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate report-ready figures from experiments/experiment_log.md.

This script treats the markdown results table as the canonical local source,
which is important because many raw experiment artifacts only exist on TC2.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-codex"))

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "experiments" / "experiment_log.md"
DEFAULT_OUTDIR = ROOT / "experiments" / "figures"


@dataclass
class ResultRow:
    exp_id: str
    seed_label: str
    cd_mean: float | None
    cd_std: float | None
    nc_mean: float | None
    nc_std: float | None
    status: str


def parse_metric(cell: str) -> tuple[float | None, float | None]:
    cell = cell.strip().replace("**", "")
    if cell in {"pending eval", "training", "pending", "skipped", "n/a", "-", "—"}:
        return None, None

    match = re.match(r"([0-9.]+)\s*\+/-\s*([0-9.]+|n/a)", cell)
    if not match:
        return None, None

    mean = float(match.group(1))
    std_token = match.group(2)
    std = None if std_token == "n/a" else float(std_token)
    return mean, std


def parse_results_table(log_path: Path) -> list[ResultRow]:
    lines = log_path.read_text().splitlines()
    start = None
    rows: list[ResultRow] = []

    for idx, line in enumerate(lines):
        if line.strip() == "## Results":
            start = idx + 1
            break

    if start is None:
        raise ValueError(f"Could not find results table in {log_path}")

    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            break
        if not stripped.startswith("|"):
            continue
        if stripped.startswith("|----"):
            continue

        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) != 7 or parts[0] == "ID":
            continue

        cd_mean, cd_std = parse_metric(parts[2])
        nc_mean, nc_std = parse_metric(parts[3])

        rows.append(
            ResultRow(
                exp_id=parts[0],
                seed_label=parts[1].replace("**", ""),
                cd_mean=cd_mean,
                cd_std=cd_std,
                nc_mean=nc_mean,
                nc_std=nc_std,
                status=parts[6].replace("**", ""),
            )
        )

    return rows


def get_row(rows: list[ResultRow], exp_id: str, seed_label: str) -> ResultRow:
    for row in rows:
        if row.exp_id == exp_id and row.seed_label == seed_label:
            return row
    raise KeyError(f"Missing row for {exp_id} / {seed_label}")


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_label_efficiency(rows: list[ResultRow], outdir: Path) -> Path:
    ratios = [100, 50, 10, 5]
    positions = np.arange(len(ratios))
    no_pe = {
        100: get_row(rows, "EXP-02", "42"),
        50: get_row(rows, "EXP-03", "42"),
        10: get_row(rows, "EXP-04", "3-seed"),
        5: get_row(rows, "EXP-05", "42"),
    }
    pe_l4 = {
        100: get_row(rows, "EXP-10", "42"),
        10: get_row(rows, "EXP-11", "42"),
        5: get_row(rows, "EXP-12", "42"),
    }
    pe_l6 = {
        100: get_row(rows, "EXP-09", "42"),
        10: get_row(rows, "EXP-06", "3-seed"),
        5: get_row(rows, "EXP-07", "42"),
    }

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    def line(series: dict[int, ResultRow], label: str, color: str, marker: str) -> None:
        ys = []
        yerr = []
        for ratio in ratios:
            row = series.get(ratio)
            ys.append(np.nan if row is None else row.cd_mean)
            yerr.append(np.nan if row is None or row.cd_std is None else row.cd_std)

        ax.errorbar(
            positions,
            ys,
            yerr=yerr,
            label=label,
            color=color,
            marker=marker,
            linewidth=2.4,
            markersize=7,
            capsize=4,
        )

    line(no_pe, "Eikonal, no PE", "#146356", "o")
    line(pe_l4, "Eikonal + PE (L=4)", "#c84c09", "s")
    line(pe_l6, "Eikonal + PE (L=6)", "#8b1e3f", "^")

    ax.set_xticks(positions)
    ax.set_xticklabels(["100%", "50%", "10%", "5%"])
    ax.set_xlabel("Supervision Ratio")
    ax.set_ylabel("Chamfer Distance (lower is better)")
    ax.set_title("Label Efficiency With and Without Fourier PE")
    # Dynamic y-range so the chart works across broken/fixed preprocessing regimes
    all_cd = [r.cd_mean for r in [*no_pe.values(), *pe_l4.values(), *pe_l6.values()] if r.cd_mean is not None]
    if all_cd:
        lo, hi = min(all_cd), max(all_cd)
        pad = (hi - lo) * 0.2 or 0.01
        ax.set_ylim(max(0, lo - pad), hi + pad)
    ax.legend(frameon=True)

    path = outdir / "label_efficiency_cd.png"
    save(fig, path)
    return path


def generate_ablation(rows: list[ResultRow], outdir: Path) -> Path:
    selected = [
        ("EXP-01", "42", "100%\nbase"),
        ("EXP-02", "42", "100%\n+Eik"),
        ("EXP-04", "3-seed", "10%\n+Eik"),
        ("EXP-05", "42", "5%\n+Eik"),
        ("EXP-06", "3-seed", "10%\n+Eik+PE6"),
        ("EXP-08", "42", "10%\n+PE6+L2"),
        ("EXP-10", "42", "100%\n+Eik+PE4"),
        ("EXP-11", "42", "10%\n+Eik+PE4"),
        ("EXP-12", "42", "5%\n+Eik+PE4"),
    ]
    rows_sel = [get_row(rows, exp_id, seed) for exp_id, seed, _ in selected]
    labels = [label for _, _, label in selected]
    x = np.arange(len(labels))
    width = 0.38

    cd_vals = [row.cd_mean for row in rows_sel]
    nc_vals = [row.nc_mean for row in rows_sel]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True)

    cd_err = [0.0 if row.cd_std is None else row.cd_std for row in rows_sel]
    nc_err = [0.0 if row.nc_std is None else row.nc_std for row in rows_sel]

    axes[0].bar(x, cd_vals, yerr=cd_err, capsize=3, color="#1f6f8b", width=width)
    axes[0].set_ylabel("Chamfer Distance")
    axes[0].set_title("Core Experiment Ablations")
    axes[0].axhline(get_row(rows, "EXP-02", "42").cd_mean, color="#666", linestyle="--", linewidth=1)

    axes[1].bar(x, nc_vals, yerr=nc_err, capsize=3, color="#d98c10", width=width)
    axes[1].set_ylabel("Normal Consistency")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    path = outdir / "ablation_cd_nc.png"
    save(fig, path)
    return path


def generate_pe_frequency(rows: list[ResultRow], outdir: Path) -> Path:
    ratios = ["100%", "10%", "5%"]
    no_pe = [
        get_row(rows, "EXP-02", "42"),
        get_row(rows, "EXP-04", "3-seed"),
        get_row(rows, "EXP-05", "42"),
    ]
    pe_l4 = [
        get_row(rows, "EXP-10", "42"),
        get_row(rows, "EXP-11", "42"),
        get_row(rows, "EXP-12", "42"),
    ]
    pe_l6 = [
        get_row(rows, "EXP-09", "42"),
        get_row(rows, "EXP-06", "3-seed"),
        get_row(rows, "EXP-07", "42"),
    ]

    x = np.arange(len(ratios))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    ax.bar(x - width, [r.cd_mean for r in no_pe], width=width, label="No PE", color="#146356")
    ax.bar(x, [r.cd_mean for r in pe_l4], width=width, label="PE L=4", color="#c84c09")
    ax.bar(x + width, [r.cd_mean for r in pe_l6], width=width, label="PE L=6", color="#8b1e3f")

    ax.set_xticks(x)
    ax.set_xticklabels(ratios)
    ax.set_xlabel("Supervision Ratio")
    ax.set_ylabel("Chamfer Distance")
    ax.set_title("PE Frequency Ablation")
    ax.legend(frameon=True)

    path = outdir / "pe_frequency_ablation_cd.png"
    save(fig, path)
    return path


def generate_summary(rows: list[ResultRow], outdir: Path) -> Path:
    path = outdir / "figure_manifest.txt"
    lines = [
        "Generated from experiments/experiment_log.md",
        "",
        "Key values:",
        f"- Baseline (EXP-01, no Eik, no PE, 100%): CD {get_row(rows, 'EXP-01', '42').cd_mean:.4f}  NC {get_row(rows, 'EXP-01', '42').nc_mean:.4f}",
        f"- 10% + Eikonal (no PE, 3-seed): CD {get_row(rows, 'EXP-04', '3-seed').cd_mean:.4f}  NC {get_row(rows, 'EXP-04', '3-seed').nc_mean:.4f}",
        f"- PE L=6 at 10% (3-seed): CD {get_row(rows, 'EXP-06', '3-seed').cd_mean:.4f}  NC {get_row(rows, 'EXP-06', '3-seed').nc_mean:.4f}",
        f"- PE L=4 at 10%: CD {get_row(rows, 'EXP-11', '42').cd_mean:.4f}  NC {get_row(rows, 'EXP-11', '42').nc_mean:.4f}",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate project figures from the experiment log")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Path to experiment_log.md")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory for figures")
    args = parser.parse_args()

    setup_style()
    rows = parse_results_table(args.log)
    args.outdir.mkdir(parents=True, exist_ok=True)

    outputs = [
        generate_label_efficiency(rows, args.outdir),
        generate_ablation(rows, args.outdir),
        generate_pe_frequency(rows, args.outdir),
        generate_summary(rows, args.outdir),
    ]

    print("Generated:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

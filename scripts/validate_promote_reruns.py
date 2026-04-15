#!/usr/bin/env python3
"""
Validate and promote rerun results after harvest.

Usage:
    python scripts/validate_promote_reruns.py [--promote]

Without --promote: validates only, prints a report.
With --promote: also copies rerun -> results.json (backing up any existing results.json).
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

MANIFEST = "slurm/rerun_manifest.txt"
EXPERIMENTS_DIR = "experiments"


def load_manifest():
    rows = []
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Bad manifest line: {line!r}")
            checkpoint_mode, expected_n_total, result_path = parts
            rows.append((checkpoint_mode, int(expected_n_total), result_path))
    return rows


def validate_result(result_path, checkpoint_mode, expected_n_total):
    """Returns (passed, errors, data) for one result file."""
    errors = []

    if not os.path.exists(result_path):
        return False, [f"File missing: {result_path}"], None

    try:
        data = json.load(open(result_path))
    except Exception as e:
        return False, [f"JSON parse error: {e}"], None

    agg = data.get("aggregate", {})
    ckpt = data.get("checkpoint")
    order = data.get("shape_order")
    n_ok = agg.get("n_ok")
    n_total = agg.get("n_total")
    cd = agg.get("chamfer_distance", {}).get("mean") if isinstance(agg.get("chamfer_distance"), dict) else None
    nc = agg.get("normal_consistency", {}).get("mean") if isinstance(agg.get("normal_consistency"), dict) else None

    want_ckpt = f"{checkpoint_mode}.pt"

    if n_total != expected_n_total:
        errors.append(f"n_total={n_total} != expected {expected_n_total}")
    if n_ok is None or n_ok <= 0:
        errors.append(f"n_ok={n_ok} (must be > 0)")
    if ckpt != want_ckpt:
        errors.append(f"checkpoint={ckpt!r} != expected {want_ckpt!r}")
    if order != "train_shapes.json":
        errors.append(f"shape_order={order!r} != 'train_shapes.json'")
    if cd is None:
        errors.append("chamfer_distance.mean missing")
    if nc is None:
        errors.append("normal_consistency.mean missing")

    passed = len(errors) == 0
    return passed, errors, data


def promote(result_path):
    """Copy result_path -> results.json, backing up existing file."""
    exp_dir = os.path.dirname(result_path)
    dest = os.path.join(exp_dir, "results.json")
    if os.path.exists(dest):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(exp_dir, f"results_legacy_{ts}.json")
        shutil.copy2(dest, backup)
        print(f"  Backed up {dest} -> {os.path.basename(backup)}")
    shutil.copy2(result_path, dest)
    print(f"  Promoted {os.path.basename(result_path)} -> results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true",
                        help="Copy rerun results to results.json after validation")
    args = parser.parse_args()

    rows = load_manifest()
    print(f"Manifest: {len(rows)} entries\n")

    all_passed = True
    results_data = []

    for checkpoint_mode, expected_n_total, result_path in rows:
        passed, errors, data = validate_result(result_path, checkpoint_mode, expected_n_total)
        status = "PASS" if passed else "FAIL"
        exp_dir = os.path.dirname(result_path)

        if passed:
            agg = data["aggregate"]
            cd = agg["chamfer_distance"]["mean"]
            cd_std = agg["chamfer_distance"]["std"]
            nc = agg["normal_consistency"]["mean"]
            nc_std = agg["normal_consistency"]["std"]
            n_ok = agg["n_ok"]
            n_total = agg["n_total"]
            ckpt = data["checkpoint"]
            order = data["shape_order"]
            print(f"{status}  {exp_dir:40s}  ckpt={ckpt}  n={n_ok}/{n_total}"
                  f"  CD={cd:.4f}±{cd_std:.4f}  NC={nc:.4f}±{nc_std:.4f}")
            results_data.append((exp_dir, checkpoint_mode, data))
        else:
            all_passed = False
            print(f"{status}  {exp_dir}")
            for e in errors:
                print(f"       {e}")

    print()
    if not all_passed:
        print("VALIDATION FAILED — fix errors before promoting")
        sys.exit(1)

    print("All 16 results validated.")

    if args.promote:
        print("\nPromoting results...")
        for checkpoint_mode, expected_n_total, result_path in rows:
            promote(result_path)
        print("\nAll results promoted to results.json.")
        print("Next: update experiments/experiment_log.md with corrected CD/NC values.")
    else:
        print("Run with --promote to copy rerun results to results.json.")


if __name__ == "__main__":
    main()

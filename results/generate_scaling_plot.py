#!/usr/bin/env python3
"""
Generate scaling plots from experiment results.

Reads SNES Newton JSON result files across different MPI process counts,
aggregates repeated runs (median time), and produces:
  1. Strong scaling plot: wall time vs number of processes (log-log)
  2. Speedup plot: speedup vs number of processes (log-log)

Usage:
  python3 results/generate_scaling_plot.py results/experiment_001/
  python3 results/generate_scaling_plot.py results/experiment_001/ --output results/experiment_001/scaling.png
  python3 results/generate_scaling_plot.py results/experiment_001/ --solver custom_newton
"""
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import argparse
import json
import os
import sys
from collections import defaultdict
import statistics

import matplotlib
matplotlib.use("Agg")

# Mapping from mesh file level -> table level
MESH_TO_TABLE_LEVEL = {5: 4, 6: 5, 7: 6, 8: 7, 9: 8}
EXPECTED_DOFS = {4: 2945, 5: 12033, 6: 48641, 7: 195585, 8: 784385}


def load_and_aggregate(exp_dir, solver="snes_newton"):
    """Load all runs for a given solver and aggregate by (nprocs, mesh_level).

    Returns:
        dict of {nprocs: {table_level: median_time}}
    """
    # Collect: (nprocs, mesh_level) -> [times]
    raw = defaultdict(lambda: defaultdict(list))

    for fname in sorted(os.listdir(exp_dir)):
        if not fname.startswith(solver) or not fname.endswith(".json"):
            continue
        fpath = os.path.join(exp_dir, fname)
        with open(fpath) as f:
            data = json.load(f)
        nprocs = data["metadata"]["nprocs"]
        for r in data["results"]:
            ml = r["mesh_level"]
            tl = MESH_TO_TABLE_LEVEL.get(ml, ml)
            raw[nprocs][tl].append(r["time"])

    # Aggregate to median
    result = {}
    for nprocs in sorted(raw):
        result[nprocs] = {}
        for tl in sorted(raw[nprocs]):
            result[nprocs][tl] = statistics.median(raw[nprocs][tl])

    return result


def generate_scaling_plot(exp_dir, solver="snes_newton", output=None, dpi=150):
    data = load_and_aggregate(exp_dir, solver)

    if not data:
        print(f"No data found for solver '{solver}' in {exp_dir}")
        sys.exit(1)

    nprocs_list = sorted(data.keys())
    levels = sorted(next(iter(data.values())).keys())

    # Only plot levels with enough data to show scaling (skip very small problems)
    levels_to_plot = [l for l in levels if EXPECTED_DOFS.get(l, 0) >= 10000]
    if not levels_to_plot:
        levels_to_plot = levels

    # Color map
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(levels_to_plot)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: Wall time vs nprocs ---
    for i, lvl in enumerate(levels_to_plot):
        dofs = EXPECTED_DOFS.get(lvl, 0)
        times = [data[np_][lvl] for np_ in nprocs_list if lvl in data[np_]]
        nps = [np_ for np_ in nprocs_list if lvl in data[np_]]
        label = f"lvl {lvl} ({dofs:,} dofs)"
        ax1.plot(nps, times, "o-", color=colors[i], label=label, markersize=6, linewidth=1.5)

    # Ideal scaling reference line (from serial time of largest problem)
    largest_lvl = levels_to_plot[-1]
    t1 = data[1][largest_lvl] if 1 in data else data[nprocs_list[0]][largest_lvl]
    ref_nps = np.array(nprocs_list)
    ax1.plot(ref_nps, t1 / ref_nps, "k--", alpha=0.4, linewidth=1, label="ideal scaling")

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of MPI processes")
    ax1.set_ylabel("Wall time [s]")
    ax1.set_title(f"Strong Scaling — {solver.replace('_', ' ').title()}")
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.xaxis.set_major_locator(ticker.FixedLocator(nprocs_list))
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, which="both", alpha=0.3)

    # --- Right: Speedup vs nprocs ---
    for i, lvl in enumerate(levels_to_plot):
        dofs = EXPECTED_DOFS.get(lvl, 0)
        t_serial = data[1][lvl] if 1 in data else None
        if t_serial is None:
            continue
        speedups = [t_serial / data[np_][lvl] for np_ in nprocs_list if lvl in data[np_]]
        nps = [np_ for np_ in nprocs_list if lvl in data[np_]]
        label = f"lvl {lvl} ({dofs:,} dofs)"
        ax2.plot(nps, speedups, "o-", color=colors[i], label=label, markersize=6, linewidth=1.5)

    # Ideal speedup line
    ax2.plot(ref_nps, ref_nps, "k--", alpha=0.4, linewidth=1, label="ideal speedup")

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=2)
    ax2.set_xlabel("Number of MPI processes")
    ax2.set_ylabel("Speedup (T₁ / Tₙ)")
    ax2.set_title("Parallel Speedup")
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.xaxis.set_major_locator(ticker.FixedLocator(nprocs_list))
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.yaxis.set_major_locator(ticker.FixedLocator(nprocs_list))
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    if output is None:
        output = os.path.join(exp_dir, "scaling.png")
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"Scaling plot saved to {output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate scaling plots from experiment results")
    parser.add_argument("exp_dir", help="Path to experiment directory (e.g. results/experiment_001/)")
    parser.add_argument("--output", "-o", help="Output image path (default: <exp_dir>/scaling.png)")
    parser.add_argument("--solver", default="snes_newton", help="Solver to plot (default: snes_newton)")
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI (default: 150)")
    args = parser.parse_args()

    generate_scaling_plot(args.exp_dir, solver=args.solver, output=args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()

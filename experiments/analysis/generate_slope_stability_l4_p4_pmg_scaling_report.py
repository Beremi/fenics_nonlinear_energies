#!/usr/bin/env python3
"""Generate a scaling report for the L4 P4 PMG solver at 4/8/16 ranks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("artifacts/raw_results/slope_stability_l4_p4_pmg_scaling_lambda1/summary.json")
DEFAULT_OUTPUT = Path("artifacts/reports/slope_stability_l4_p4_pmg_scaling_lambda1")


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| ranks | solve [s] | setup [s] | total [s] | speedup vs 4 | efficiency vs 4 | Newton | linear | omega | u_max | worst true rel |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    t4 = float(rows[0]["solve_time_sec"])
    r4 = int(rows[0]["ranks"])
    for row in rows:
        speedup = t4 / float(row["solve_time_sec"])
        efficiency = speedup / (int(row["ranks"]) / r4)
        lines.append(
            "| "
            f"{row['ranks']} | {_fmt(float(row['solve_time_sec']))} | {_fmt(float(row['setup_time_sec']))} | "
            f"{_fmt(float(row['total_time_sec']))} | {_fmt(speedup)} | {_fmt(efficiency)} | "
            f"{row['newton_iterations']} | {row['linear_iterations']} | {_fmt(float(row['omega']), 6)} | "
            f"{_fmt(float(row['u_max']), 6)} | {_fmt(float(row['worst_true_relative_residual']), 6)} |"
        )
    return "\n".join(lines)


def _plot(rows: list[dict[str, object]], out_dir: Path) -> None:
    ranks = np.array([int(row["ranks"]) for row in rows], dtype=np.int32)
    solve = np.array([float(row["solve_time_sec"]) for row in rows], dtype=np.float64)
    setup = np.array([float(row["setup_time_sec"]) for row in rows], dtype=np.float64)
    total = np.array([float(row["total_time_sec"]) for row in rows], dtype=np.float64)
    speedup = solve[0] / solve
    ideal = ranks / ranks[0]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=180)

    ax = axes[0]
    ax.plot(ranks, solve, marker="o", linewidth=2.0, label="solve")
    ax.plot(ranks, setup, marker="s", linewidth=2.0, label="setup")
    ax.plot(ranks, total, marker="^", linewidth=2.0, label="total")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("time [s]")
    ax.set_title("L4 P4 PMG timing (log-log)")
    ax.set_xticks(ranks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(ranks, speedup, marker="o", linewidth=2.0, label="measured")
    ax.plot(ranks, ideal, linestyle="--", linewidth=1.8, label="ideal")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("speedup vs 4 ranks")
    ax.set_title("L4 P4 PMG strong scaling")
    ax.set_xticks(ranks)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "scaling.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = sorted(json.loads(args.input.read_text(encoding="utf-8")), key=lambda row: int(row["ranks"]))
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot(rows, out_dir)

    report = f"""# `L4` `P4` PMG Scaling

Solver configuration:

- level: `4`
- discretisation: same-mesh `P4`
- preconditioner: `PCMG`
- hierarchy: `P4 -> P2 -> P1` with an additional `L3 P1` tail
- nonlinear setting: `--no-use_trust_region`
- linear setting: `fgmres`, `rtol=1e-2`, `max_it=100`

## Scaling Table

{_table(rows)}

## Graph

![L4 P4 PMG scaling](scaling.png)
"""

    (out_dir / "README.md").write_text(report, encoding="utf-8")
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

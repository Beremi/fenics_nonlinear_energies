#!/usr/bin/env python3
"""Generate a scaling report for the featured L5 P4 PMG solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("artifacts/raw_results/slope_stability_l5_p4_pmg_scaling_lambda1/summary.json")
DEFAULT_OUTPUT = Path("artifacts/reports/slope_stability_l5_p4_pmg_scaling_lambda1")


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| ranks | success | solve [s] | one-time setup [s] | total [s] | solve speedup | efficiency | Newton | linear | omega | u_max | worst true rel |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    success_rows = [row for row in rows if bool(row["solver_success"])]
    base_row = success_rows[0] if success_rows else rows[0]
    t0 = float(base_row["solve_time_sec"])
    r0 = int(base_row["ranks"])
    for row in rows:
        speedup = t0 / float(row["solve_time_sec"])
        efficiency = speedup / (int(row["ranks"]) / r0)
        lines.append(
            "| "
            f"{row['ranks']} | {row['solver_success']} | {_fmt(float(row['solve_time_sec']))} | {_fmt(float(row['setup_time_sec']))} | "
            f"{_fmt(float(row['total_time_sec']))} | {_fmt(speedup)} | {_fmt(efficiency)} | "
            f"{row['newton_iterations']} | {row['linear_iterations']} | {_fmt(float(row['omega']), 6)} | "
            f"{_fmt(float(row['u_max']), 6)} | {_fmt(float(row['worst_true_relative_residual']), 6)} |"
        )
    return "\n".join(lines)


def _breakdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| ranks | problem build [s] | assembler init [s] | PMG bootstrap [s] | solve [s] | finalize [s] | outside solve [s] | outside setup+solve [s] |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['ranks']} | {_fmt(float(row.get('problem_build_time_sec', 0.0)))} | "
            f"{_fmt(float(row.get('assembler_setup_time_sec', 0.0)))} | "
            f"{_fmt(float(row.get('solver_bootstrap_time_sec', 0.0)))} | "
            f"{_fmt(float(row['solve_time_sec']))} | {_fmt(float(row.get('finalize_time_sec', 0.0)))} | "
            f"{_fmt(float(row.get('outside_solve_time_sec', 0.0)))} | "
            f"{_fmt(float(row.get('outside_setup_solve_time_sec', 0.0)))} |"
        )
    return "\n".join(lines)


def _plot(rows: list[dict[str, object]], out_dir: Path) -> None:
    ranks = np.array([int(row["ranks"]) for row in rows], dtype=np.int32)
    solve = np.array([float(row["solve_time_sec"]) for row in rows], dtype=np.float64)
    setup = np.array([float(row["setup_time_sec"]) for row in rows], dtype=np.float64)
    total = np.array([float(row["total_time_sec"]) for row in rows], dtype=np.float64)
    outside_solve = np.array([float(row.get("outside_solve_time_sec", 0.0)) for row in rows], dtype=np.float64)
    success_mask = np.array([bool(row["solver_success"]) for row in rows], dtype=bool)
    success_rows = [row for row in rows if bool(row["solver_success"])]
    base_row = success_rows[0] if success_rows else rows[0]
    base_ranks = int(base_row["ranks"])
    base_solve = float(base_row["solve_time_sec"])
    speedup = np.array(
        [base_solve / float(row["solve_time_sec"]) if bool(row["solver_success"]) else np.nan for row in rows],
        dtype=np.float64,
    )
    ideal = ranks / base_ranks

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=180)

    ax = axes[0]
    ax.plot(ranks, solve, marker="o", linewidth=2.0, label="solve")
    ax.plot(ranks, setup, marker="s", linewidth=2.0, label="one-time setup")
    ax.plot(ranks, outside_solve, marker="d", linewidth=2.0, label="outside solve")
    ax.plot(ranks, total, marker="^", linewidth=2.0, label="total")
    if np.any(~success_mask):
        ax.scatter(ranks[~success_mask], solve[~success_mask], marker="x", s=80, color="crimson", label="nonconverged")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("time [s]")
    ax.set_title("L5 P4 PMG timing")
    ax.set_xticks(ranks)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(ranks[success_mask], speedup[success_mask], marker="o", linewidth=2.0, label="measured")
    ax.plot(ranks, ideal, linestyle="--", linewidth=1.8, label="ideal")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel(f"speedup vs {base_ranks} ranks")
    ax.set_title("L5 P4 PMG strong scaling")
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

    report = f"""# `L5` `P4` PMG Scaling

Featured solver configuration:

- level: `5`
- discretisation: same-mesh `P4`
- preconditioner: `PCMG`
- hierarchy: `P4 -> P2 -> P1` with an additional `L4 P1` tail
- nonlinear setting: `--no-use_trust_region`
- linear setting: `fgmres`, `rtol=1e-2`, `max_it=100`

The `8`-rank run reaches the same response metrics as the converged runs, but
under these exact settings it stops at Newton `maxit=100` rather than
satisfying the nonlinear stop test. The speedup panel therefore uses the
smallest successful rank count as its baseline.

`one-time setup` below is the corrected pre-solve cost:

- problem-data build before assembler creation
- assembler / PETSc object initialisation
- PMG hierarchy bootstrap before Newton starts

That is larger than the old narrow `setup_time` field, which only covered the
assembler-construction slice.

## Scaling Table

{_table(rows)}

## Timing Breakdown

{_breakdown_table(rows)}

## Graph

![L5 P4 PMG scaling](scaling.png)
"""

    (out_dir / "README.md").write_text(report, encoding="utf-8")
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

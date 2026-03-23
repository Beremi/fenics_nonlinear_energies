#!/usr/bin/env python3
"""Generate a report for the L5 P4 PMG maxit=20 comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "slope_stability_l5_p4_pmg_part_scaling_lambda1_maxit20"
    / "summary.json"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "slope_stability_l5_p4_pmg_part_scaling_lambda1_maxit20"
)
OUTPUT_SUMMARY = OUTPUT_DIR / "summary.json"


def _fmt(value: float | int | str | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _plot_overall(rows: list[dict[str, object]], out: Path) -> None:
    x = np.array([int(row["ranks"]) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for key, label in [
        ("solve_time_sec", "solve"),
        ("steady_state_total_time_sec", "steady-state total"),
        ("end_to_end_total_time_sec", "end-to-end total"),
    ]:
        y = np.array([float(row[key]) for row in rows], dtype=float)
        ax.plot(x, y, marker="o", label=label)
        ax.plot(x, y[0] / x, linestyle="--", alpha=0.45, label=f"{label} ideal")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("ranks")
    ax.set_ylabel("time [s]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_consistency(rows: list[dict[str, object]], out: Path) -> None:
    x = np.array([int(row["ranks"]) for row in rows], dtype=float)
    ref_energy = float(rows[-1]["energy"])
    energy_error = np.array(
        [abs(float(row["energy"]) - ref_energy) + 1e-16 for row in rows], dtype=float
    )
    linear_iters = np.array([int(row["linear_iterations"]) for row in rows], dtype=float)
    worst_rel = np.array(
        [float(row["worst_true_relative_residual"]) for row in rows], dtype=float
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    axes[0].plot(x, energy_error, marker="o", label="|E - E(rank 8)|")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("ranks")
    axes[0].set_ylabel("final energy error")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(x, linear_iters, marker="o", label="linear iterations")
    axes[1].plot(x, worst_rel, marker="s", label="worst true rel residual")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("ranks")
    axes[1].set_ylabel("count / residual")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    rows = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    rows.sort(key=lambda row: int(row["ranks"]))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    for row in rows:
        row["steady_state_overhead_sec"] = float(row["steady_state_total_time_sec"]) - float(
            row["solve_time_sec"]
        )
        row["warmup_overhead_sec"] = float(row["end_to_end_total_time_sec"]) - float(
            row["steady_state_total_time_sec"]
        )

    _plot_overall(rows, OUTPUT_DIR / "overall_timing_loglog.png")
    _plot_consistency(rows, OUTPUT_DIR / "final_state_consistency.png")

    ref_steady = float(rows[0]["steady_state_total_time_sec"])
    ref_end = float(rows[0]["end_to_end_total_time_sec"])
    ref_solve = float(rows[0]["solve_time_sec"])
    ref_energy = float(rows[-1]["energy"])

    lines: list[str] = []
    lines.append("# L5 P4 PMG Part Scaling With Newton Cap 20")
    lines.append("")
    lines.append(
        "This note repeats the instrumented `L5`, `P4` explicit-Galerkin PMG benchmark with the same PMG "
        "settings as the original part-scaling study, but forces `--maxit 20`."
    )
    lines.append("")
    lines.append("## Why The `2`- And `4`-Rank Runs Failed Before")
    lines.append("")
    lines.append("The earlier `2`- and `4`-rank failures were not linear-solver failures.")
    lines.append("")
    lines.append("- Every inner linear solve converged with `CONVERGED_RTOL`.")
    lines.append("- The nonlinear solve simply hit the Newton iteration cap before satisfying the final nonlinear stop test.")
    lines.append("- With the cap fixed to `20`, the `1`, `2`, and `4` rank runs all land at essentially the same final state as the converged `8`-rank run.")
    lines.append("")
    lines.append(
        "That points to a nonlinear-stop / convergence-speed issue in this instrumented PMG path, not to a divergent linear-preconditioner path."
    )
    lines.append("")
    lines.append("## PMG Setting")
    lines.append("")
    lines.append("- finest problem: `L5`, `P4`, `lambda=1.0`")
    lines.append("- hierarchy: `same_mesh_p4_p2_p1_lminus1_p1`")
    lines.append("- MG variant: `explicit_pmg`")
    lines.append("- lower-level operator policy: `galerkin_refresh`")
    lines.append("- smoothers:")
    lines.append("  - `P4`: `richardson + sor`, `3` steps")
    lines.append("  - `P2`: `richardson + sor`, `3` steps")
    lines.append("  - `P1`: `richardson + sor`, `3` steps")
    lines.append("- coarse solve:")
    lines.append("  - `cg + hypre`")
    lines.append("  - `nodal_coarsen = 6`")
    lines.append("  - `vec_interp_variant = 3`")
    lines.append("  - `strong_threshold = 0.5`")
    lines.append("  - `coarsen_type = HMIS`")
    lines.append("  - `max_iter = 4`")
    lines.append("  - `tol = 0.0`")
    lines.append("  - `relax_type_all = symmetric-SOR/Jacobi`")
    lines.append("")
    lines.append("## Timing Table")
    lines.append("")
    lines.append("| ranks | success | solve [s] | steady-state [s] | end-to-end [s] | steady overhead [s] | warmup overhead [s] | steady speedup | end-to-end speedup |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['ranks']} | {'yes' if row['solver_success'] else 'no'} | "
            f"{_fmt(row['solve_time_sec'])} | {_fmt(row['steady_state_total_time_sec'])} | "
            f"{_fmt(row['end_to_end_total_time_sec'])} | {_fmt(row['steady_state_overhead_sec'])} | "
            f"{_fmt(row['warmup_overhead_sec'])} | "
            f"{_fmt(ref_steady / float(row['steady_state_total_time_sec']))} | "
            f"{_fmt(ref_end / float(row['end_to_end_total_time_sec']))} |"
        )
    lines.append("")
    lines.append("## Final Energy And Linear Error At `maxit=20`")
    lines.append("")
    lines.append("| ranks | success | Newton | linear | energy | |E - E(rank 8)| | omega | u_max | worst true rel | last true rel | last KSP reason |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        energy = float(row["energy"])
        lines.append(
            f"| {row['ranks']} | {'yes' if row['solver_success'] else 'no'} | {row['newton_iterations']} | "
            f"{row['linear_iterations']} | {_fmt(energy, 9)} | {_fmt(abs(energy - ref_energy), 9)} | "
            f"{_fmt(row['omega'], 9)} | {_fmt(row['u_max'], 9)} | "
            f"{_fmt(row['worst_true_relative_residual'], 6)} | {_fmt(row['last_true_relative_residual'], 6)} | "
            f"`{row['last_reason_name']}` |"
        )
    lines.append("")
    lines.append("## Main Takeaway")
    lines.append("")
    lines.append(
        "At a fixed `20` Newton iterations, all rank counts end at essentially the same energy and linear residual quality."
    )
    lines.append("")
    lines.append(f"- `1` vs `8` energy difference: `{abs(float(rows[0]['energy']) - ref_energy):.3e}`")
    lines.append(f"- `2` vs `8` energy difference: `{abs(float(rows[1]['energy']) - ref_energy):.3e}`")
    lines.append(f"- `4` vs `8` energy difference: `{abs(float(rows[2]['energy']) - ref_energy):.3e}`")
    lines.append("")
    lines.append("So the previous `2`- and `4`-rank failures are better interpreted as:")
    lines.append("")
    lines.append("- slower nonlinear convergence under the stop test, not")
    lines.append("- a materially different final state or a broken linear-preconditioner solve")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append("### Overall Timing")
    lines.append("")
    lines.append("![Overall timing log-log scaling](overall_timing_loglog.png)")
    lines.append("")
    lines.append("### Final State Consistency")
    lines.append("")
    lines.append("![Final state consistency](final_state_consistency.png)")
    lines.append("")
    lines.append("## Raw Data")
    lines.append("")
    lines.append(
        "- matching raw summary: `/home/michal/repos/fenics_nonlinear_energies/artifacts/raw_results/slope_stability_l5_p4_pmg_part_scaling_lambda1_maxit20/summary.json`"
    )

    for name in ("README.md", "report.md"):
        (OUTPUT_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

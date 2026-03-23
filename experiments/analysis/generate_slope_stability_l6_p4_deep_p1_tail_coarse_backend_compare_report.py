#!/usr/bin/env python3
"""Generate the L6 P4 deep-tail coarse-backend comparison report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "slope_stability_l6_p4_deep_p1_tail_coarse_backend_compare_lambda1_np8_maxit20"
)
REPORT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "slope_stability_l6_p4_deep_p1_tail_coarse_backend_compare_lambda1_np8_maxit20"
)
SUMMARY_PATH = RAW_ROOT / "summary.json"
README_PATH = REPORT_ROOT / "README.md"
PLOT_PATH = REPORT_ROOT / "timing_compare.png"


def _load_rows() -> list[dict[str, object]]:
    return list(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))


def _fmt(x: object, digits: int = 3) -> str:
    if x is None:
        return ""
    try:
        val = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def _plot(rows: list[dict[str, object]]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(labels))
    steady = np.asarray([float(row.get("steady_state_total_time_sec", np.nan)) for row in rows])
    solve = np.asarray([float(row.get("solve_time_sec", np.nan)) for row in rows])
    coarse = np.asarray([float(row.get("coarse_observed_time_sec", np.nan)) for row in rows])

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    width = 0.26
    ax.bar(x - width, steady, width=width, label="steady-state total")
    ax.bar(x, solve, width=width, label="solve")
    ax.bar(x + width, coarse, width=width, label="cumulative coarse observed")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("time [s]")
    ax.set_title("L6 P4 deep-tail PMG coarse backend comparison on 8 ranks")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)


def main() -> None:
    rows = _load_rows()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    _plot(rows)

    current = next((row for row in rows if str(row.get("variant")) == "current"), None)
    lines: list[str] = []
    lines.append("# L6 P4 deep-P1-tail PMG: coarse-backend comparison on 8 ranks")
    lines.append("")
    lines.append("Kept solver setting:")
    lines.append("- hierarchy: `1:1,2:1,3:1,4:1,5:1,6:1,6:2,6:4`")
    lines.append("- top problem: `L6`, `P4`, `lambda=1.0`")
    lines.append("- nonlinear: `armijo`, `maxit=20`, `--no-use_trust_region`")
    lines.append("- linear: `fgmres`, `ksp_max_it=15`, `--accept_ksp_maxit_direction`")
    lines.append("- smoothers: `richardson + sor`, `3` steps on `P4/P2/P1`")
    lines.append("- coarse hypre recipe when used: `nodal_coarsen=6`, `vec_interp_variant=3`, `strong_threshold=0.5`, `coarsen_type=HMIS`, `max_iter=4`, `tol=0.0`, `relax_type_all=symmetric-SOR/Jacobi`")
    lines.append("- note: every row is intentionally capped at `20` Newton iterations, so `status=failed` below means `hit maxit`, not `linear solver broke`")
    lines.append("")
    lines.append("## Timing Comparison")
    lines.append("")
    lines.append("| variant | steady-state [s] | solve [s] | coarse observed [s] | linear its | ls evals | final energy | worst true rel | status |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        lines.append(
            f"| {row['label']} | {_fmt(row.get('steady_state_total_time_sec'))} | {_fmt(row.get('solve_time_sec'))} | "
            f"{_fmt(row.get('coarse_observed_time_sec'))} | {int(row.get('linear_iterations', 0))} | "
            f"{int(row.get('line_search_evals', 0))} | {_fmt(row.get('energy'), 9)} | "
            f"{_fmt(row.get('worst_true_relative_residual'), 3)} | {row.get('status', '')} |"
        )
    lines.append("")
    if current is not None:
        lines.append("## Delta vs current")
        lines.append("")
        lines.append("| variant | steady delta [s] | solve delta [s] | coarse delta [s] | linear delta | energy delta |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            if row is current:
                continue
            lines.append(
                f"| {row['label']} | "
                f"{_fmt(float(row.get('steady_state_total_time_sec', np.nan)) - float(current.get('steady_state_total_time_sec', np.nan)))} | "
                f"{_fmt(float(row.get('solve_time_sec', np.nan)) - float(current.get('solve_time_sec', np.nan)))} | "
                f"{_fmt(float(row.get('coarse_observed_time_sec', np.nan)) - float(current.get('coarse_observed_time_sec', np.nan)))} | "
                f"{int(row.get('linear_iterations', 0)) - int(current.get('linear_iterations', 0))} | "
                f"{_fmt(float(row.get('energy', np.nan)) - float(current.get('energy', np.nan)), 9)} |"
            )
        lines.append("")
    lines.append("## Coarse Diagnostics")
    lines.append("")
    lines.append("| variant | backend | coarse KSP | coarse PC | coarse solves | coarse outer its | last coarse reason | last coarse residual |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | --- | ---: |")
    for row in rows:
        lines.append(
            f"| {row['label']} | {row.get('mg_coarse_backend', '')} | {row.get('coarse_last_ksp_type', '')} | "
            f"{row.get('coarse_last_pc_type', '')} | {int(row.get('coarse_solve_invocations', 0))} | "
            f"{int(row.get('coarse_outer_iterations', 0))} | {row.get('coarse_last_reason_name', '')} | "
            f"{_fmt(row.get('coarse_last_residual_norm'), 3)} |"
        )
    lines.append("")
    lines.append("## Plot")
    lines.append("")
    lines.append(f"![timing comparison](./{PLOT_PATH.name})")
    lines.append("")
    lines.append("## Raw Data")
    lines.append("")
    lines.append(f"- summary: `{SUMMARY_PATH}`")
    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

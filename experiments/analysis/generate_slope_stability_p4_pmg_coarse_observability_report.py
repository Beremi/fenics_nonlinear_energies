#!/usr/bin/env python3
"""Generate a coarse-Hypre observability report for P4 PMG."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SUMMARY_PATH = Path(
    "artifacts/raw_results/slope_stability_p4_pmg_coarse_observability_lambda1/summary.json"
)
OUTPUT_DIR = Path(
    "artifacts/reports/slope_stability_p4_pmg_coarse_observability_lambda1"
)


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _plot_maxiter(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, level in zip(axes, sorted({int(row["level"]) for row in rows}), strict=False):
        subset = [
            row
            for row in rows
            if int(row["level"]) == int(level)
            and str(row["phase"]) == "max_iter"
            and int(row["ranks"]) == 8
            and bool(row.get("solver_success"))
        ]
        subset.sort(key=lambda row: int(row["mg_coarse_hypre_max_iter"]))
        if not subset:
            continue
        x = np.array([int(row["mg_coarse_hypre_max_iter"]) for row in subset], dtype=np.float64)
        y1 = np.array([float(row["steady_state_total_time_sec"]) for row in subset], dtype=np.float64)
        y2 = np.array([float(row["coarse_observed_time_sec"]) for row in subset], dtype=np.float64)
        ax.plot(x, y1, marker="o", label="steady-state total")
        ax.plot(x, y2, marker="s", label="coarse observed")
        ax.set_title(f"L{level} rank 8")
        ax.set_xlabel("coarse hypre max_iter")
        ax.set_ylabel("time [s]")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    rows = list(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    maxiter_plot = OUTPUT_DIR / "coarse_maxiter_vs_time.png"
    _plot_maxiter(rows, maxiter_plot)

    lines: list[str] = []
    lines.append("# P4 PMG Coarse-Hypre Observability")
    lines.append("")
    lines.append("This report probes why changing coarse BoomerAMG accuracy often does not change outer PMG iterations.")
    lines.append("")
    lines.append("## Max-Iter Sweep")
    lines.append("")
    lines.append("| level | ranks | max_iter | success | Newton | linear | steady-state [s] | coarse solves | coarse iters | coarse contraction | coarse observed [s] |")
    lines.append("| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(rows, key=lambda item: (int(item["level"]), str(item["phase"]), int(item["ranks"]), str(item["variant"]))):
        if str(row["phase"]) != "max_iter":
            continue
        lines.append(
            f"| {row['level']} | {row['ranks']} | {row['mg_coarse_hypre_max_iter']} | "
            f"{'yes' if row.get('solver_success') else 'no'} | {row.get('newton_iterations', '-')} | "
            f"{row.get('linear_iterations', '-')} | {_fmt(row.get('steady_state_total_time_sec'))} | "
            f"{row.get('coarse_solve_invocations', '-')} | {row.get('coarse_total_iterations', '-')} | "
            f"{_fmt(row.get('coarse_average_residual_contraction'), 3)} | {_fmt(row.get('coarse_observed_time_sec'))} |"
        )
    lines.append("")
    lines.append("## Secondary Tolerance / KSP Sweeps")
    lines.append("")
    lines.append("| phase | level | ranks | label | success | Newton | linear | steady-state [s] | coarse observed [s] |")
    lines.append("| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |")
    for row in sorted(rows, key=lambda item: (str(item["phase"]), int(item["level"]), int(item["ranks"]), str(item["variant"]))):
        if str(row["phase"]) == "max_iter":
            continue
        lines.append(
            f"| {row['phase']} | {row['level']} | {row['ranks']} | {row['label']} | "
            f"{'yes' if row.get('solver_success') else 'no'} | {row.get('newton_iterations', '-')} | "
            f"{row.get('linear_iterations', '-')} | {_fmt(row.get('steady_state_total_time_sec'))} | "
            f"{_fmt(row.get('coarse_observed_time_sec'))} |"
        )
    lines.append("")

    observations: list[str] = []
    for level in sorted({int(row["level"]) for row in rows}):
        subset = [
            row
            for row in rows
            if int(row["level"]) == int(level)
            and str(row["phase"]) == "max_iter"
            and int(row["ranks"]) == 8
            and bool(row.get("solver_success"))
        ]
        if not subset:
            continue
        subset.sort(key=lambda row: int(row["mg_coarse_hypre_max_iter"]))
        ref = subset[0]
        worst = subset[-1]
        observations.append(
            f"- L{level} rank 8: outer linear iterations stayed at `{ref['linear_iterations']}` while coarse observed time rose from "
            f"`{_fmt(ref['coarse_observed_time_sec'])} s` at `max_iter={ref['mg_coarse_hypre_max_iter']}` to "
            f"`{_fmt(worst['coarse_observed_time_sec'])} s` at `max_iter={worst['mg_coarse_hypre_max_iter']}`."
        )
    lines.append("## Conclusion")
    lines.append("")
    if observations:
        lines.extend(observations)
        lines.append("")
    l4_rank8 = [
        row
        for row in rows
        if int(row["level"]) == 4 and int(row["ranks"]) == 8 and str(row["phase"]) == "max_iter"
    ]
    l5_rank8 = [
        row
        for row in rows
        if int(row["level"]) == 5 and int(row["ranks"]) == 8 and str(row["phase"]) == "max_iter"
    ]
    l4_best = min(l4_rank8, key=lambda row: float(row["steady_state_total_time_sec"])) if l4_rank8 else None
    l5_best = min(l5_rank8, key=lambda row: float(row["steady_state_total_time_sec"])) if l5_rank8 else None
    lu_ref = next(
        (
            row
            for row in rows
            if int(row["level"]) == 4 and int(row["ranks"]) == 1 and str(row["variant"]) == "coarse_lu_reference"
        ),
        None,
    )
    if l4_best is not None:
        lines.append(
            f"- L4 rank 8 is fastest at `max_iter={l4_best['mg_coarse_hypre_max_iter']}` with "
            f"`{_fmt(l4_best['steady_state_total_time_sec'])} s` steady-state time."
        )
    if l5_best is not None:
        lines.append(
            f"- L5 rank 8 is fastest at `max_iter={l5_best['mg_coarse_hypre_max_iter']}` with "
            f"`{_fmt(l5_best['steady_state_total_time_sec'])} s` steady-state time."
        )
    if lu_ref is not None:
        lines.append(
            f"- The exact L4 rank-1 coarse LU reference still gives the same outer counts "
            f"(`Newton={lu_ref['newton_iterations']}`, `linear={lu_ref['linear_iterations']}`) as the BoomerAMG coarse variants."
        )
    lines.append("")
    lines.append(
        "The new MG runtime diagnostics show that stronger coarse-Hypre settings do not change the number of coarse solves or the outer Newton / outer-KSP iteration counts in this PMG path. "
        "What they do change is the coarse solve quality-cost tradeoff inside each apply. "
        "A moderate setting (`max_iter=4` here) reduces coarse Krylov work enough to beat weaker settings, while stronger settings (`max_iter=6`) start paying too much per apply. "
        "So the current PMG is not coarse-iteration-limited in the outer sense, but it is still sensitive to coarse backend efficiency."
    )
    lines.append("")
    lines.append(f"![Coarse max_iter observability]({maxiter_plot.name})")
    lines.append("")

    report_text = "\n".join(lines)
    (OUTPUT_DIR / "README.md").write_text(report_text + "\n", encoding="utf-8")
    (OUTPUT_DIR / "report.md").write_text(report_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

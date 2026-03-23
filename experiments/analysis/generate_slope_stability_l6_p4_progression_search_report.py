#!/usr/bin/env python3
"""Generate a markdown report for the L6 P4 progression search."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_SUMMARY = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "slope_stability_l6_p4_progression_search_lambda1_np8_staged"
    / "summary.json"
)
REPORT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "slope_stability_l6_p4_progression_search_lambda1_np8_staged"
)
README_PATH = REPORT_DIR / "README.md"
REPORT_PATH = REPORT_DIR / "report.md"
REPORT_SUMMARY = REPORT_DIR / "summary.json"
BAR_PLOT = REPORT_DIR / "end_to_end_totals.png"


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _load_rows() -> list[dict[str, object]]:
    return json.loads(RAW_SUMMARY.read_text(encoding="utf-8"))


def _build_plot(rows: list[dict[str, object]]) -> None:
    labels = [f"{row['name']}\n({row['maxit']})" for row in rows]
    values = [float(row["end_to_end_total_time_sec"]) for row in rows]
    colors = []
    for row in rows:
        if not bool(row["solver_success"]):
            colors.append("tab:red")
        elif str(row.get("stage")) == "final":
            colors.append("tab:green")
        else:
            colors.append("tab:blue")
    x = np.arange(len(rows), dtype=np.float64)
    plt.figure(figsize=(max(10.0, 0.9 * len(rows)), 5.5))
    plt.bar(x, values, color=colors)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("End-to-end total time [s]")
    plt.title("L6 P4 PMG Progression Search On 8 Ranks")
    plt.tight_layout()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(BAR_PLOT, dpi=180)
    plt.close()


def _table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| name | stage | maxit | hierarchy | end-to-end [s] | steady-state [s] | solve [s] | Newton | Linear | energy | true rel | status |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        hierarchy = (
            str(row["mg_strategy"])
            if not row.get("mg_custom_hierarchy")
            else f"`{row['mg_custom_hierarchy']}`"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    str(row["stage"]),
                    str(int(row["maxit"])),
                    hierarchy,
                    _fmt(row["end_to_end_total_time_sec"]),
                    _fmt(row["steady_state_total_time_sec"]),
                    _fmt(row["solve_time_sec"]),
                    str(int(row["newton_iterations"])),
                    str(int(row["linear_iterations"])),
                    _fmt(row["energy"], 6),
                    _fmt(row["worst_true_relative_residual"], 6),
                    str(row["status"]),
                ]
            )
            + " |"
        )
    return lines


def main() -> None:
    rows = _load_rows()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_SUMMARY.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _build_plot(rows)

    successful = [row for row in rows if bool(row["solver_success"]) and str(row.get("stage")) == "final"]
    best = min(successful, key=lambda row: float(row["end_to_end_total_time_sec"])) if successful else None
    baseline = next((row for row in rows if str(row["name"]) == "final_short_p1_tail"), None)

    lines: list[str] = [
        "# L6 P4 PMG Progression Search On 8 Ranks",
        "",
        "This report compares curated multilevel progressions for the assembled `L6`, `P4` PMG solver on `8` MPI ranks.",
        "",
        "The search is staged:",
        "- screen runs use `maxit=1` to identify promising bottoms",
        "- finalist runs use `maxit=20` for the actual timing comparison",
        "",
        "## Fixed Solver Setting",
        "",
        "- `pc_type=mg`, `mg_variant=legacy_pmg`",
        "- outer linear solver: `fgmres`, `ksp_rtol=1e-2`, `ksp_max_it=100`",
        "- nonlinear globalization: `--no-use_trust_region`",
        "- distribution: `overlap_p2p`",
        "- top/mid/bottom smoothers: `richardson + sor`, `3` steps",
        "- coarse solve: `cg + hypre(boomeramg)`",
        "- coarse Hypre settings: `nodal_coarsen=6`, `vec_interp_variant=3`, `strong_threshold=0.5`, `coarsen_type=HMIS`, `max_iter=4`, `tol=0.0`, `relax_type_all=symmetric-SOR/Jacobi`",
        "",
        "## Main Result",
        "",
    ]

    if best is None:
        lines.append("- No progression completed successfully in this search window.")
    else:
        lines.append(
            "- Best successful progression: "
            f"`{best['name']}` at `{_fmt(best['end_to_end_total_time_sec'])} s` end-to-end "
            f"and `{_fmt(best['solve_time_sec'])} s` solve time."
        )
        if baseline is not None:
            delta = (
                (float(best["end_to_end_total_time_sec"]) / float(baseline["end_to_end_total_time_sec"]) - 1.0)
                * 100.0
            )
            lines.append(
                "- Relative to the current short-tail baseline, that is "
                f"`{_fmt(delta, 1)}%` on end-to-end time."
            )
        if best.get("mg_custom_hierarchy"):
            lines.append(f"- Winning custom hierarchy: `{best['mg_custom_hierarchy']}`")
        else:
            lines.append(f"- Winning built-in hierarchy: `{best['mg_strategy']}`")

    lines.extend(
        [
            "",
            "## Results",
            "",
            *_table(rows),
            "",
            "## Plot",
            "",
            "![End-to-end totals](end_to_end_totals.png)",
            "",
            "## Reading Guide",
            "",
            "- `short_p1_tail` means `L5 P1 -> L6 P1 -> L6 P2 -> L6 P4`.",
            "- `same_mesh_only` removes the `L5 P1` tail and keeps only `L6 P1 -> L6 P2 -> L6 P4`.",
            "- `full_p1_tail` and `full_p1_tail_no_p2` test whether a much deeper `P1` bottom helps.",
            "- `full_p2_chain` and `full_p4_chain` test whether the optimal bottom is actually higher order.",
            "- If a family looked competitive, the runner expands only that family with extra tail depths instead of sweeping every possibility.",
        ]
    )

    text = "\n".join(lines) + "\n"
    README_PATH.write_text(text, encoding="utf-8")
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"Wrote report to {README_PATH}")


if __name__ == "__main__":
    main()

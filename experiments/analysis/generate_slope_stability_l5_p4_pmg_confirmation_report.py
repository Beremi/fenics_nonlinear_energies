#!/usr/bin/env python3
"""Generate the L5 PMG confirmation report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(
    "artifacts/raw_results/slope_stability_l5_p4_pmg_confirmation_lambda1/summary.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/slope_stability_l5_p4_pmg_confirmation_lambda1"
)


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _groups(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    for name in grouped:
        grouped[name] = sorted(grouped[name], key=lambda row: int(row["ranks"]))
    return dict(grouped)


def _table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| variant | ranks | success | steady-state [s] | end-to-end [s] | solve [s] | Newton | linear | ls evals | worst true rel |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['variant_label']} | {row['ranks']} | {row['solver_success']} | "
            f"{_fmt(row['steady_state_total_time_sec'])} | {_fmt(row['end_to_end_total_time_sec'])} | "
            f"{_fmt(row['solve_time_sec'])} | {row['newton_iterations']} | {row['linear_iterations']} | "
            f"{row['line_search_evals']} | {_fmt(row['worst_true_relative_residual'], 5)} |"
        )
    return "\n".join(lines)


def _plot(groups: dict[str, list[dict[str, object]]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=180)
    colors = {"tuned_hypre_nonmg": "#8172b3"}
    for name, rows in groups.items():
        ranks = np.array([int(row["ranks"]) for row in rows], dtype=np.int32)
        steady = np.array([float(row["steady_state_total_time_sec"]) for row in rows], dtype=np.float64)
        end_to_end = np.array(
            [float(row["end_to_end_total_time_sec"]) for row in rows], dtype=np.float64
        )
        color = colors.get(name, "#4c72b0")
        axes[0].plot(ranks, end_to_end, marker="o", linewidth=2.0, color=color, label=f"{rows[0]['variant_label']} end-to-end")
        axes[1].plot(ranks, steady, marker="o", linewidth=2.0, color=color, label=f"{rows[0]['variant_label']} steady-state")
    for ax, title in zip(axes, ["End-to-end total", "Steady-state total"], strict=True):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 4, 8])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("time [s]")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "scaling.png", bbox_inches="tight")
    plt.close(fig)


def _breakdown_plot(rows: list[dict[str, object]], out_dir: Path, key_map: list[tuple[str, str]], filename: str, title: str) -> None:
    ranks = np.array([int(row["ranks"]) for row in rows], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=180)
    bottom = np.zeros_like(ranks, dtype=np.float64)
    palette = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974", "#64b5cd"]
    for (label, key), color in zip(key_map, palette, strict=False):
        vals = np.array([float(row.get(key, 0.0)) for row in rows], dtype=np.float64)
        ax.bar(ranks, vals, bottom=bottom, width=0.55, label=label, color=color)
        bottom += vals
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("time [s]")
    ax.set_title(title)
    ax.set_xticks(ranks)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = sorted(json.loads(args.input.read_text(encoding="utf-8")), key=lambda row: (str(row["variant"]), int(row["ranks"])))
    groups = _groups(rows)
    pmg_variant = next(name for name in groups if name != "tuned_hypre_nonmg")
    pmg_rows = groups[pmg_variant]

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot(groups, out_dir)
    _breakdown_plot(
        pmg_rows,
        out_dir,
        [
            ("problem build", "problem_build_time_sec"),
            ("JAX warmup", "assembler_warmup_time_sec"),
            ("MG hierarchy", "mg_hierarchy_build_time_sec"),
            ("MG configure", "mg_configure_time_sec"),
        ],
        "setup_breakdown.png",
        f"Setup breakdown: {pmg_rows[0]['variant_label']}",
    )
    _breakdown_plot(
        pmg_rows,
        out_dir,
        [
            ("energy", "energy_total_time_sec"),
            ("gradient", "gradient_total_time_sec"),
            ("hessian", "hessian_total_time_sec"),
            ("line search", "line_search_time_sec"),
            ("PC setup", "linear_pc_setup_time_sec"),
            ("KSP solve", "linear_ksp_solve_time_sec"),
        ],
        "solve_breakdown.png",
        f"Solve breakdown: {pmg_rows[0]['variant_label']}",
    )

    pmg_by_rank = {int(row["ranks"]): row for row in pmg_rows}
    hypre_by_rank = {int(row["ranks"]): row for row in groups.get("tuned_hypre_nonmg", [])}
    hypre_speed_note = (
        f"{_fmt(float(hypre_by_rank[8]['steady_state_total_time_sec']) / float(pmg_by_rank[8]['steady_state_total_time_sec']))}x slower for Hypre"
        if 8 in hypre_by_rank
        else "not run in this confirmation pass"
    )
    report = f"""# `L5` `P4` PMG Confirmation

This report confirms the accepted `L4` PMG configuration on `L5`, and compares it with the tuned non-MG `Hypre/BoomerAMG` baseline.

## Overview

{_table(rows)}

## Outcome

- PMG confirmation variant: {pmg_rows[0]['variant_label']}
- PMG `1 -> 8` steady-state speedup: {_fmt(float(pmg_by_rank[1]['steady_state_total_time_sec']) / float(pmg_by_rank[8]['steady_state_total_time_sec']))}x
- PMG `1 -> 8` end-to-end speedup: {_fmt(float(pmg_by_rank[1]['end_to_end_total_time_sec']) / float(pmg_by_rank[8]['end_to_end_total_time_sec']))}x
- `8`-rank PMG vs tuned Hypre steady-state: {hypre_speed_note}

## Graphs

![L5 confirmation scaling](scaling.png)

![L5 PMG setup breakdown](setup_breakdown.png)

![L5 PMG solve breakdown](solve_breakdown.png)
"""

    (out_dir / "README.md").write_text(report, encoding="utf-8")
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

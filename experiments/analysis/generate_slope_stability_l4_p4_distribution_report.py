#!/usr/bin/env python3
"""Generate a report for the L4 P4 PMG distribution study."""

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
    "artifacts/raw_results/slope_stability_l4_p4_distribution_study_lambda1/summary.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/slope_stability_l4_p4_distribution_study_lambda1"
)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _variant_groups(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    for variant in grouped:
        grouped[variant] = sorted(grouped[variant], key=lambda row: int(row["ranks"]))
    return dict(grouped)


def _overview_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| variant | ranks | success | setup [s] | solve [s] | total [s] | Newton | linear | energy eval [s] | grad [s] | hess [s] | ls [s] |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['variant_label']} | {row['ranks']} | {row['solver_success']} | "
            f"{_fmt(float(row['setup_time_sec']))} | {_fmt(float(row['solve_time_sec']))} | "
            f"{_fmt(float(row['total_time_sec']))} | {row['newton_iterations']} | "
            f"{row['linear_iterations']} | {_fmt(float(row['energy_eval_time_sec']))} | "
            f"{_fmt(float(row['newton_grad_phase_time_sec']))} | "
            f"{_fmt(float(row['newton_hessian_phase_time_sec']))} | "
            f"{_fmt(float(row['line_search_time_sec']))} |"
        )
    return "\n".join(lines)


def _rank8_table(rows: list[dict[str, object]]) -> str:
    rank8 = [row for row in rows if int(row["ranks"]) == 8]
    lines = [
        "| variant | problem build [s] | assembler warmup [s] | MG hierarchy [s] | MG configure [s] | linear assembly [s] | PC setup [s] | KSP solve [s] | operator apply [s] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rank8:
        lines.append(
            "| "
            f"{row['variant_label']} | {_fmt(float(row['problem_build_time_sec']))} | "
            f"{_fmt(float(row['assembler_warmup_time_sec']))} | "
            f"{_fmt(float(row['mg_hierarchy_build_time_sec']))} | "
            f"{_fmt(float(row['mg_configure_time_sec']))} | "
            f"{_fmt(float(row['linear_assemble_time_sec']))} | "
            f"{_fmt(float(row['linear_pc_setup_time_sec']))} | "
            f"{_fmt(float(row['linear_ksp_solve_time_sec']))} | "
            f"{_fmt(float(row['linear_operator_apply_time_sec']))} |"
        )
    return "\n".join(lines)


def _speedup_table(groups: dict[str, list[dict[str, object]]]) -> str:
    lines = [
        "| variant | total speedup 2x | total speedup 4x | total speedup 8x | solve speedup 2x | solve speedup 4x | solve speedup 8x |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, rows in groups.items():
        base = rows[0]
        total0 = float(base["total_time_sec"])
        solve0 = float(base["solve_time_sec"])
        by_rank = {int(row["ranks"]): row for row in rows}
        lines.append(
            "| "
            f"{rows[0]['variant_label']} | "
            f"{_fmt(total0 / float(by_rank[2]['total_time_sec']))} | "
            f"{_fmt(total0 / float(by_rank[4]['total_time_sec']))} | "
            f"{_fmt(total0 / float(by_rank[8]['total_time_sec']))} | "
            f"{_fmt(solve0 / float(by_rank[2]['solve_time_sec']))} | "
            f"{_fmt(solve0 / float(by_rank[4]['solve_time_sec']))} | "
            f"{_fmt(solve0 / float(by_rank[8]['solve_time_sec']))} |"
        )
    return "\n".join(lines)


def _plot(groups: dict[str, list[dict[str, object]]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), dpi=180)
    colors = {
        "baseline_replicated": "#404040",
        "distributed_compute_only": "#c44e52",
        "distributed_setup_only": "#4c72b0",
        "distributed_full": "#55a868",
        "distributed_best_effort": "#8172b3",
    }

    for variant, rows in groups.items():
        label = str(rows[0]["variant_label"])
        ranks = np.array([int(row["ranks"]) for row in rows], dtype=np.int32)
        total = np.array([float(row["total_time_sec"]) for row in rows], dtype=np.float64)
        solve = np.array([float(row["solve_time_sec"]) for row in rows], dtype=np.float64)
        setup = np.array([float(row["setup_time_sec"]) for row in rows], dtype=np.float64)
        color = colors.get(variant, None)

        axes[0].plot(ranks, total, marker="o", linewidth=2.0, label=label, color=color)
        axes[1].plot(ranks, solve, marker="o", linewidth=2.0, label=label, color=color)
        axes[2].plot(ranks, setup, marker="o", linewidth=2.0, label=label, color=color)

    for ax, title in zip(
        axes,
        ["Total time", "Solve time", "One-time setup"],
        strict=True,
    ):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 4, 8])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("time [s]")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "scaling.png", bbox_inches="tight")
    plt.close(fig)


def _conclusion(groups: dict[str, list[dict[str, object]]]) -> str:
    baseline = groups["baseline_replicated"]
    rank8_rows = [rows[-1] for rows in groups.values()]
    best_rank8 = min(rank8_rows, key=lambda row: float(row["total_time_sec"]))
    total_speedup = float(baseline[0]["total_time_sec"]) / float(best_rank8["total_time_sec"])
    solve_speedup = float(baseline[0]["solve_time_sec"]) / float(best_rank8["solve_time_sec"])
    return (
        "The scaling story separates cleanly into two parts: point-to-point overlap exchange "
        "reduces the compute-side callback cost, while root-built case data and transfer "
        "construction reduce the replicated setup cost only when they avoid rebuilding global "
        "objects on every rank. The most important comparison is baseline `1` rank versus the "
        f"best `8`-rank variant (`{best_rank8['variant_label']}`): total time improves by about "
        f"`{_fmt(total_speedup)}`x and solve time by about `{_fmt(solve_speedup)}`x."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda row: (str(row["variant"]), int(row["ranks"])))
    groups = _variant_groups(rows)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot(groups, out_dir)

    report = f"""# `L4` `P4` PMG Distribution Study

This report compares five variants of the same `L4` `P4` PMG solve:

- `Baseline`: replicated same-mesh level construction plus overlap `Allgatherv`
- `P2P only`: point-to-point overlap exchange, but replicated setup
- `Root-build only`: replicated overlap exchange, but root-built problem/hierarchy setup
- `Full distributed`: point-to-point overlap exchange plus root-built problem/hierarchy setup
- `P2P + owned-row MG`: point-to-point overlap exchange plus distributed owned-row PMG transfer construction

All runs use the same nonlinear and linear settings:

- geometry/discretisation: `L4`, same-mesh `P4`
- hierarchy: `P4 -> P2 -> P1` with an `L3 P1` tail
- nonlinear solver: Newton with line search, `--no-use_trust_region`
- linear solver: `fgmres + PCMG`
- tolerances: `ksp_rtol=1e-2`, `ksp_max_it=100`

## Overview

{_overview_table(rows)}

## Speedup

{_speedup_table(groups)}

## `8`-Rank Breakdown

{_rank8_table(rows)}

## Conclusion

{_conclusion(groups)}

## Graph

![L4 P4 PMG distribution scaling](scaling.png)
"""

    (out_dir / "README.md").write_text(report, encoding="utf-8")
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

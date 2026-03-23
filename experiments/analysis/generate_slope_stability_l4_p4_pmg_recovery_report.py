#!/usr/bin/env python3
"""Generate the L4 P4 PMG recovery report."""

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
    "artifacts/raw_results/slope_stability_l4_p4_pmg_recovery_lambda1/summary.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/slope_stability_l4_p4_pmg_recovery_lambda1"
)
PMG_ORDER = [
    "original_baseline",
    "coarse_hypre_only",
    "distributed_setup_hypre",
    "distributed_setup_hypre_armijo",
    "distributed_cached_hypre",
    "distributed_cached_hypre_armijo",
]


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _variant_groups(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    for variant in grouped:
        grouped[variant] = sorted(grouped[variant], key=lambda row: int(row["ranks"]))
    return dict(grouped)


def _accepted_variants(groups: dict[str, list[dict[str, object]]]) -> list[str]:
    accepted: list[str] = []
    current: str | None = None
    for variant in PMG_ORDER:
        rows = groups.get(variant)
        if not rows or any(not bool(row["solver_success"]) for row in rows):
            continue
        if current is None:
            accepted.append(variant)
            current = variant
            continue
        current_rows = {int(row["ranks"]): row for row in groups[current]}
        cand_rows = {int(row["ranks"]): row for row in rows}
        newton_ok = all(
            float(cand_rows[r]["newton_iterations"])
            <= 1.10 * float(current_rows[r]["newton_iterations"])
            for r in cand_rows
        )
        linear_ok = all(
            float(cand_rows[r]["linear_iterations"])
            <= 1.10 * float(current_rows[r]["linear_iterations"])
            for r in cand_rows
        )
        improves_steady = float(cand_rows[8]["steady_state_total_time_sec"]) < float(
            current_rows[8]["steady_state_total_time_sec"]
        )
        improves_end_to_end = float(cand_rows[8]["end_to_end_total_time_sec"]) < float(
            current_rows[8]["end_to_end_total_time_sec"]
        )
        if newton_ok and linear_ok and (improves_steady or improves_end_to_end):
            accepted.append(variant)
            current = variant
    return accepted


def _overview_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| variant | ranks | success | end-to-end [s] | steady-state [s] | solve [s] | Newton | linear | ls evals | worst true rel |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['variant_label']} | {row['ranks']} | {row['solver_success']} | "
            f"{_fmt(row['end_to_end_total_time_sec'])} | {_fmt(row['steady_state_total_time_sec'])} | "
            f"{_fmt(row['solve_time_sec'])} | {row['newton_iterations']} | "
            f"{row['linear_iterations']} | {row['line_search_evals']} | "
            f"{_fmt(row['worst_true_relative_residual'], 5)} |"
        )
    return "\n".join(lines)


def _accepted_rounds_table(groups: dict[str, list[dict[str, object]]], accepted: list[str]) -> str:
    lines = [
        "| round | variant | 8-rank steady-state [s] | 8-rank end-to-end [s] | 8-rank Newton | 8-rank linear | accepted |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    accepted_set = set(accepted)
    for variant in PMG_ORDER:
        rows = groups.get(variant)
        if not rows:
            continue
        rank8 = {int(row["ranks"]): row for row in rows}.get(8)
        if rank8 is None:
            continue
        lines.append(
            "| "
            f"{rank8['round']} | {rank8['variant_label']} | "
            f"{_fmt(rank8['steady_state_total_time_sec'])} | "
            f"{_fmt(rank8['end_to_end_total_time_sec'])} | "
            f"{rank8['newton_iterations']} | {rank8['linear_iterations']} | "
            f"{'yes' if variant in accepted_set else 'no'} |"
        )
    return "\n".join(lines)


def _comparison_table(
    groups: dict[str, list[dict[str, object]]],
    accepted: list[str],
) -> str:
    if not accepted:
        return "No fully successful PMG variants were accepted."
    final_variant = accepted[-1]
    compare_names = [accepted[0], final_variant]
    if "tuned_hypre_nonmg" in groups:
        compare_names.append("tuned_hypre_nonmg")

    lines = [
        "| variant | ranks | setup [s] | solve [s] | end-to-end [s] | MG transfer [s] | transfer cache I/O [s] | energy [s] | grad [s] | hess [s] | ls [s] | PC setup [s] | KSP [s] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in compare_names:
        for row in groups.get(variant, []):
            lines.append(
                "| "
                f"{row['variant_label']} | {row['ranks']} | {_fmt(row['one_time_setup_time_sec'])} | "
                f"{_fmt(row['solve_time_sec'])} | {_fmt(row['end_to_end_total_time_sec'])} | "
                f"{_fmt(row.get('mg_transfer_build_time_sec', 0.0))} | "
                f"{_fmt(row.get('mg_transfer_cache_io_time_sec', 0.0))} | "
                f"{_fmt(row['energy_total_time_sec'])} | {_fmt(row['gradient_total_time_sec'])} | "
                f"{_fmt(row['hessian_total_time_sec'])} | {_fmt(row['line_search_time_sec'])} | "
                f"{_fmt(row['linear_pc_setup_time_sec'])} | {_fmt(row['linear_ksp_solve_time_sec'])} |"
            )
    return "\n".join(lines)


def _rank8_diagnosis(groups: dict[str, list[dict[str, object]]], accepted: list[str]) -> str:
    if not accepted:
        return "No accepted PMG variant available for rank-8 diagnosis."
    final_row = {int(row["ranks"]): row for row in groups[accepted[-1]]}[8]
    pieces = [
        ("problem build", float(final_row["problem_build_time_sec"])),
        ("assembler warmup", float(final_row["assembler_warmup_time_sec"])),
        ("MG hierarchy", float(final_row["mg_hierarchy_build_time_sec"])),
        ("MG configure", float(final_row["mg_configure_time_sec"])),
        ("energy", float(final_row["energy_total_time_sec"])),
        ("gradient", float(final_row["gradient_total_time_sec"])),
        ("hessian", float(final_row["hessian_total_time_sec"])),
        ("line search", float(final_row["line_search_time_sec"])),
        ("PC setup", float(final_row["linear_pc_setup_time_sec"])),
        ("KSP solve", float(final_row["linear_ksp_solve_time_sec"])),
    ]
    pieces.sort(key=lambda item: item[1], reverse=True)
    return ", ".join(f"{name}={_fmt(value)} s" for name, value in pieces[:5])


def _plot_scaling(
    groups: dict[str, list[dict[str, object]]],
    accepted: list[str],
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=180)
    compare = list(accepted)
    if "tuned_hypre_nonmg" in groups:
        compare.append("tuned_hypre_nonmg")
    colors = {
        "original_baseline": "#404040",
        "coarse_hypre_only": "#4c72b0",
        "distributed_setup_hypre": "#55a868",
        "distributed_setup_hypre_armijo": "#c44e52",
        "distributed_cached_hypre": "#64b5cd",
        "distributed_cached_hypre_armijo": "#dd8452",
        "tuned_hypre_nonmg": "#8172b3",
    }
    for variant in compare:
        rows = groups.get(variant, [])
        if not rows:
            continue
        ranks = np.array([int(row["ranks"]) for row in rows], dtype=np.int32)
        total = np.array([float(row["end_to_end_total_time_sec"]) for row in rows], dtype=np.float64)
        steady = np.array(
            [float(row["steady_state_total_time_sec"]) for row in rows], dtype=np.float64
        )
        ax.plot(
            ranks,
            total,
            marker="o",
            linewidth=2.0,
            color=colors.get(variant, None),
            label=f"{rows[0]['variant_label']} end-to-end",
        )
        ax.plot(
            ranks,
            steady,
            marker="s",
            linewidth=2.0,
            linestyle="--",
            color=colors.get(variant, None),
            label=f"{rows[0]['variant_label']} steady-state",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("time [s]")
    ax.set_title("L4 P4 PMG recovery scaling")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "scaling.png", bbox_inches="tight")
    plt.close(fig)


def _plot_setup_breakdown(
    groups: dict[str, list[dict[str, object]]],
    accepted: list[str],
    out_dir: Path,
) -> None:
    if not accepted:
        return
    rows = groups[accepted[-1]]
    ranks = np.array([int(row["ranks"]) for row in rows], dtype=np.int32)
    pieces = {
        "problem build": np.array([float(row["problem_build_time_sec"]) for row in rows]),
        "assembler init": np.array([float(row["assembler_setup_time_sec"]) for row in rows]),
        "JAX warmup": np.array([float(row["assembler_warmup_time_sec"]) for row in rows]),
        "MG hierarchy": np.array([float(row["mg_hierarchy_build_time_sec"]) for row in rows]),
        "MG transfer": np.array([float(row.get("mg_transfer_build_time_sec", 0.0)) for row in rows]),
        "MG configure": np.array([float(row["mg_configure_time_sec"]) for row in rows]),
    }
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974", "#64b5cd"]
    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=180)
    bottom = np.zeros_like(ranks, dtype=np.float64)
    for (label, values), color in zip(pieces.items(), colors, strict=True):
        ax.bar(ranks, values, bottom=bottom, width=0.55, label=label, color=color)
        bottom += values
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("time [s]")
    ax.set_title(f"Setup breakdown: {rows[0]['variant_label']}")
    ax.set_xticks(ranks)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "setup_breakdown.png", bbox_inches="tight")
    plt.close(fig)


def _plot_solve_breakdown(
    groups: dict[str, list[dict[str, object]]],
    accepted: list[str],
    out_dir: Path,
) -> None:
    if not accepted:
        return
    rows = groups[accepted[-1]]
    ranks = np.array([int(row["ranks"]) for row in rows], dtype=np.int32)
    pieces = {
        "energy": np.array([float(row["energy_total_time_sec"]) for row in rows]),
        "gradient": np.array([float(row["gradient_total_time_sec"]) for row in rows]),
        "hessian": np.array([float(row["hessian_total_time_sec"]) for row in rows]),
        "line search": np.array([float(row["line_search_time_sec"]) for row in rows]),
        "PC setup": np.array([float(row["linear_pc_setup_time_sec"]) for row in rows]),
        "KSP solve": np.array([float(row["linear_ksp_solve_time_sec"]) for row in rows]),
    }
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#64b5cd", "#dd8452"]
    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=180)
    bottom = np.zeros_like(ranks, dtype=np.float64)
    for (label, values), color in zip(pieces.items(), colors, strict=True):
        ax.bar(ranks, values, bottom=bottom, width=0.55, label=label, color=color)
        bottom += values
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("time [s]")
    ax.set_title(f"Solve breakdown: {rows[0]['variant_label']}")
    ax.set_xticks(ranks)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "solve_breakdown.png", bbox_inches="tight")
    plt.close(fig)


def _plot_callbacks(
    groups: dict[str, list[dict[str, object]]],
    accepted: list[str],
    out_dir: Path,
) -> None:
    if not accepted:
        return
    rows = groups[accepted[-1]]
    ranks = np.array([int(row["ranks"]) for row in rows], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=180)
    ax.plot(
        ranks,
        [float(row["energy_total_time_sec"]) for row in rows],
        marker="o",
        linewidth=2.0,
        label="energy",
    )
    ax.plot(
        ranks,
        [float(row["gradient_total_time_sec"]) for row in rows],
        marker="o",
        linewidth=2.0,
        label="gradient",
    )
    ax.plot(
        ranks,
        [float(row["hessian_total_time_sec"]) for row in rows],
        marker="o",
        linewidth=2.0,
        label="hessian",
    )
    ax.plot(
        ranks,
        [float(row["line_search_time_sec"]) for row in rows],
        marker="o",
        linewidth=2.0,
        label="line search",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("time [s]")
    ax.set_title(f"Callback phases: {rows[0]['variant_label']}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "callback_phases.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda row: (int(row["round"]), str(row["variant"]), int(row["ranks"])))
    groups = _variant_groups(rows)
    accepted = _accepted_variants(groups)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_scaling(groups, accepted, out_dir)
    _plot_setup_breakdown(groups, accepted, out_dir)
    _plot_solve_breakdown(groups, accepted, out_dir)
    _plot_callbacks(groups, accepted, out_dir)

    final_variant = accepted[-1] if accepted else None
    final_rows = groups.get(final_variant, []) if final_variant else []
    end_to_end_speedup = None
    steady_speedup = None
    if final_rows:
        by_rank = {int(row["ranks"]): row for row in final_rows}
        end_to_end_speedup = float(by_rank[1]["end_to_end_total_time_sec"]) / float(
            by_rank[8]["end_to_end_total_time_sec"]
        )
        steady_speedup = float(by_rank[1]["steady_state_total_time_sec"]) / float(
            by_rank[8]["steady_state_total_time_sec"]
        )

    report = f"""# `L4` `P4` PMG Recovery

This report benchmarks a stepwise recovery of the featured `P4 -> P2 -> P1` PMG path on `L4` at `lambda = 1.0` across `1/2/4/8` MPI ranks.

- `Original baseline`: overlap `Allgatherv`, replicated same-mesh level construction, replicated transfers, Jacobi coarse solve
- `Coarse Hypre only`: original build/distribution path, but elasticity-style BoomerAMG on the PMG coarse solve with rigid-mode nullspaces
- `Distributed setup + coarse Hypre`: point-to-point overlap exchange, HDF5-backed root build, owned-row transfer construction, coarse BoomerAMG
- `Distributed + coarse Hypre + Armijo`: same distributed PMG path with Armijo line search
- `Tuned Hypre baseline`: non-MG tuned BoomerAMG comparison

## Overview

{_overview_table(rows)}

## Accepted Rounds

{_accepted_rounds_table(groups, accepted)}

## Final Comparison

{_comparison_table(groups, accepted)}

## Outcome

- Accepted PMG variants: {", ".join(groups[name][0]["variant_label"] for name in accepted) if accepted else "none"}
- Final PMG: {groups[final_variant][0]["variant_label"] if final_variant else "none"}
- `1 -> 8` end-to-end speedup of final PMG: {_fmt(end_to_end_speedup) if end_to_end_speedup is not None else "-"}x
- `1 -> 8` steady-state speedup of final PMG: {_fmt(steady_speedup) if steady_speedup is not None else "-"}x
- Largest rank-8 time buckets in the final PMG path: {_rank8_diagnosis(groups, accepted)}

## Graphs

![Recovery scaling](scaling.png)

![Setup breakdown](setup_breakdown.png)

![Solve breakdown](solve_breakdown.png)

![Callback phases](callback_phases.png)
"""

    (out_dir / "README.md").write_text(report, encoding="utf-8")
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

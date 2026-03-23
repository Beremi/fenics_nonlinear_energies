#!/usr/bin/env python3
"""Generate a report for the L4 P4 PMG coarse-Hypre max-iter sweep."""

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
    "artifacts/raw_results/slope_stability_l4_p4_mg_coarse_hypre_maxiter_sweep_lambda1/summary.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/slope_stability_l4_p4_mg_coarse_hypre_maxiter_sweep_lambda1"
)


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return sorted(rows, key=lambda row: (int(row["ranks"]), int(row["mg_coarse_hypre_max_iter"])))


def _by_rank(rows: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["ranks"])].append(row)
    for rank in grouped:
        grouped[rank] = sorted(grouped[rank], key=lambda row: int(row["mg_coarse_hypre_max_iter"]))
    return dict(grouped)


def _overview_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| ranks | coarse max_iter | success | steady-state [s] | end-to-end [s] | solve [s] | Newton | linear | KSP [s] | PC setup [s] | worst true rel |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['ranks']} | {row['mg_coarse_hypre_max_iter']} | {row['solver_success']} | "
            f"{_fmt(row.get('steady_state_total_time_sec'))} | {_fmt(row.get('end_to_end_total_time_sec'))} | "
            f"{_fmt(row.get('solve_time_sec'))} | {row.get('newton_iterations', '-')} | "
            f"{row.get('linear_iterations', '-')} | {_fmt(row.get('linear_ksp_solve_time_sec'))} | "
            f"{_fmt(row.get('linear_pc_setup_time_sec'))} | {_fmt(row.get('worst_true_relative_residual'), 5)} |"
        )
    return "\n".join(lines)


def _best_table(grouped: dict[int, list[dict[str, object]]]) -> str:
    lines = [
        "| ranks | best coarse max_iter | best steady-state [s] | baseline max_iter=2 [s] | ratio vs 2 | Newton | linear |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ranks, rows in grouped.items():
        success_rows = [row for row in rows if bool(row.get("solver_success", False))]
        if not success_rows:
            continue
        best = min(success_rows, key=lambda row: float(row["steady_state_total_time_sec"]))
        ref = next((row for row in success_rows if int(row["mg_coarse_hypre_max_iter"]) == 2), None)
        ratio = None
        if ref is not None and float(ref["steady_state_total_time_sec"]) > 0.0:
            ratio = float(best["steady_state_total_time_sec"]) / float(ref["steady_state_total_time_sec"])
        lines.append(
            "| "
            f"{ranks} | {best['mg_coarse_hypre_max_iter']} | {_fmt(best['steady_state_total_time_sec'])} | "
            f"{_fmt(None if ref is None else ref['steady_state_total_time_sec'])} | {_fmt(ratio, 3)} | "
            f"{best['newton_iterations']} | {best['linear_iterations']} |"
        )
    return "\n".join(lines)


def _plot_times(grouped: dict[int, list[dict[str, object]]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=180)
    for ranks, rows in grouped.items():
        xs = np.array([int(row["mg_coarse_hypre_max_iter"]) for row in rows], dtype=np.int32)
        ys = np.array(
            [float(row["steady_state_total_time_sec"]) if row.get("solver_success") else np.nan for row in rows],
            dtype=np.float64,
        )
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=f"{ranks} ranks")
    ax.set_xlabel("coarse Hypre max_iter")
    ax.set_ylabel("steady-state total [s]")
    ax.set_title("L4 P4 PMG: coarse HYPRE max_iter sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "steady_state_vs_maxiter.png", bbox_inches="tight")
    plt.close(fig)


def _plot_linear(grouped: dict[int, list[dict[str, object]]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=180)
    for ranks, rows in grouped.items():
        xs = np.array([int(row["mg_coarse_hypre_max_iter"]) for row in rows], dtype=np.int32)
        ys = np.array(
            [float(row["linear_iterations"]) if row.get("solver_success") else np.nan for row in rows],
            dtype=np.float64,
        )
        ax.plot(xs, ys, marker="s", linewidth=2.0, label=f"{ranks} ranks")
    ax.set_xlabel("coarse Hypre max_iter")
    ax.set_ylabel("outer linear iterations")
    ax.set_title("L4 P4 PMG: linear iterations vs coarse HYPRE max_iter")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "linear_iterations_vs_maxiter.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = _load_rows(args.input)
    grouped = _by_rank(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot_times(grouped, args.output_dir)
    _plot_linear(grouped, args.output_dir)

    notes = [
        "# L4 P4 PMG coarse-Hypre max_iter sweep",
        "",
        "This sweep keeps the accepted `distributed_cached_hypre` PMG path fixed and varies only `--mg_coarse_hypre_max_iter`.",
        "",
        "Fixed settings:",
        "- `level=4`, `elem_degree=4`, `lambda_target=1.0`",
        "- PMG hierarchy: `same_mesh_p4_p2_p1_lminus1_p1`",
        "- `mg_variant=legacy_pmg`, `pc_type=mg`, `ksp_type=fgmres`",
        "- coarse backend: `cg + hypre(boomeramg)`",
        "- elasticity-style BoomerAMG settings: `nodal_coarsen=6`, `vec_interp_variant=3`, `strong_threshold=0.5`, `coarsen_type=HMIS`, `tol=0.0`, `relax_type_all=symmetric-SOR/Jacobi`",
        "",
        "## Best by rank",
        _best_table(grouped),
        "",
        "## Full table",
        _overview_table(rows),
        "",
        "## Plots",
        "- `steady_state_vs_maxiter.png`",
        "- `linear_iterations_vs_maxiter.png`",
        "",
    ]
    report = "\n".join(notes)
    (args.output_dir / "README.md").write_text(report, encoding="utf-8")
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

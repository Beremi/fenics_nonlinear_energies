#!/usr/bin/env python3
"""Generate docs assets for the promoted Plasticity3D L1_2/lambda=1 local PMG results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.analysis import generate_plasticity3d_impl_scaling_assets as impl_assets


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_SUMMARY = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "source_compare"
    / "plasticity3d_l1_2_lambda1_grad1e2_local_pmg_scaling"
    / "comparison_summary.json"
)
DEFAULT_MIXED_SUMMARY = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "source_compare"
    / "plasticity3d_l1_2_lambda1_grad1e2_scaling"
    / "comparison_summary.json"
)
DEFAULT_SOURCEFIXED_SUMMARY = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "source_compare"
    / "plasticity3d_l1_2_lambda1_grad1e2_scaling_all_pmg"
    / "comparison_summary.json"
)
DEFAULT_ASSET_DIR = REPO_ROOT / "docs" / "assets" / "plasticity3d"

LOCAL_IMPL = "local_constitutiveAD_local_pmg_armijo"
SOURCE_IMPL = "source_local_pmg_armijo"
LOCAL_SOURCEFIXED_IMPL = "local_constitutiveAD_local_pmg_sourcefixed_armijo"
SOURCE_SOURCEFIXED_IMPL = "source_local_pmg_sourcefixed_armijo"

SUMMARY_JSON_NAME = "plasticity3d_l1_2_lambda1_local_pmg_assets_summary.json"
OVERALL_PLOT = "plasticity3d_l1_2_lambda1_local_pmg_scaling_overall.png"
COMMON_PLOT = "plasticity3d_l1_2_lambda1_local_pmg_common_components.png"
BREAKDOWN_PLOT = "plasticity3d_l1_2_lambda1_local_pmg_component_breakdown.png"
LOCAL_VS_SOURCE_PLOT = "plasticity3d_l1_2_lambda1_local_vs_source.png"
SOURCEFIXED_PLOT = "plasticity3d_l1_2_lambda1_sourcefixed_compare.png"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _load_enriched_rows(summary_path: Path) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    summary = _read_json(summary_path)
    implementation_order = impl_assets._implementation_order(summary)
    enriched_rows = []
    for row in list(summary.get("rows", [])):
        if not isinstance(row, dict):
            continue
        result_path = impl_assets._repo_path(str(row.get("result_json", "") or ""))
        if not result_path.exists():
            continue
        enriched_rows.append(impl_assets._enrich_row(dict(row)))
    rows_by_impl = impl_assets._rows_by_impl(enriched_rows, implementation_order)
    return summary, rows_by_impl


def _local_component_rows(rows: list[dict[str, object]]) -> list[dict[str, float]]:
    baseline_total = _safe_float(rows[0]["wall_time_s"]) if rows else float("nan")
    out: list[dict[str, float]] = []
    for row in rows:
        ranks = int(row["ranks"])
        comp = dict(row["components"])
        wall = _safe_float(row.get("wall_time_s"))
        solve = _safe_float(row.get("solve_time_s"))
        speedup = baseline_total / wall if np.isfinite(baseline_total) and np.isfinite(wall) and wall > 0.0 else float("nan")
        efficiency = speedup / float(ranks) if np.isfinite(speedup) and ranks > 0 else float("nan")
        out.append(
            {
                "ranks": ranks,
                "status": str(row.get("status", "")),
                "wall_time_s": wall,
                "solve_time_s": solve,
                "speedup": speedup,
                "efficiency": efficiency,
                "nit": int(row.get("nit", 0)),
                "linear_iterations_total": int(row.get("linear_iterations_total", 0)),
                "final_metric": _safe_float(row.get("final_metric")),
                "energy": _safe_float(row.get("energy")),
                "omega": _safe_float(row.get("omega")),
                "u_max": _safe_float(row.get("u_max")),
                "backend_build_s": _safe_float(comp.get("backend_build_s")),
                "initial_guess_s": _safe_float(comp.get("initial_guess_s")),
                "newton_gradient_top_s": _safe_float(comp.get("newton_gradient_top_s")),
                "newton_line_search_s": _safe_float(comp.get("newton_line_search_s")),
                "newton_linear_assemble_s": _safe_float(comp.get("newton_linear_assemble_s")),
                "newton_linear_setup_s": _safe_float(comp.get("newton_linear_setup_s")),
                "newton_linear_solve_s": _safe_float(comp.get("newton_linear_solve_s")),
                "callback_hessian_s": _safe_float(comp.get("callback_hessian_s")),
                "callback_gradient_s": _safe_float(comp.get("callback_gradient_s")),
            }
        )
    return out


def _comparison_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    rows = [dict(row) for row in summary.get("rows", []) if isinstance(row, dict)]
    rows.sort(key=lambda row: (int(row.get("ranks", 10**6)), str(row.get("implementation", ""))))
    out: list[dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "implementation": str(row.get("implementation", "")),
                "display_label": str(row.get("display_label", "")),
                "ranks": int(row.get("ranks", 0)),
                "status": str(row.get("status", "")),
                "wall_time_s": _safe_float(row.get("wall_time_s")),
                "solve_time_s": _safe_float(row.get("solve_time_s")),
                "nit": int(row.get("nit", 0)),
                "linear_iterations_total": int(row.get("linear_iterations_total", 0)),
                "final_metric": _safe_float(row.get("final_metric")),
                "message": str(row.get("message", "")),
                "result_json": str(row.get("result_json", "")),
            }
        )
    return out


def _find_rows(rows: list[dict[str, object]], implementation: str) -> list[dict[str, object]]:
    selected = [dict(row) for row in rows if str(row.get("implementation", "")) == implementation]
    selected.sort(key=lambda row: int(row.get("ranks", 10**6)))
    return selected


def _plot_local_vs_source(rows: list[dict[str, object]], out_path: Path) -> None:
    local_rows = _find_rows(rows, LOCAL_IMPL)
    source_rows = _find_rows(rows, SOURCE_IMPL)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for selected, label, color in (
        (local_rows, "local_constitutiveAD + local_pmg", "#1f77b4"),
        (source_rows, "source + local_pmg", "#d62728"),
    ):
        ranks = np.asarray([int(row["ranks"]) for row in selected], dtype=np.int64)
        wall = np.asarray([_safe_float(row.get("wall_time_s")) for row in selected], dtype=np.float64)
        solve = np.asarray([_safe_float(row.get("solve_time_s")) for row in selected], dtype=np.float64)
        axes[0].plot(ranks, wall, marker="o", linewidth=2.0, color=color, label=label)
        axes[1].plot(ranks, solve, marker="o", linewidth=2.0, color=color, label=label)
        impl_assets._plot_ideal_reference(axes[0], ranks, wall, color=color)
        impl_assets._plot_ideal_reference(axes[1], ranks, solve, color=color)
    for ax, title, ylabel in (
        (axes[0], "Matched Local-vs-Source Wall Time", "Wall time [s]"),
        (axes[1], "Matched Local-vs-Source Solve Time", "Solve time [s]"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel(ylabel)
    axes[1].plot([], [], linewidth=1.2, linestyle=(0, (4, 3)), color="#555555", alpha=0.7, label="Ideal 1/r")
    axes[1].legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_sourcefixed(rows: list[dict[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    series = (
        (LOCAL_IMPL, "local_constitutiveAD + local_pmg", "#1f77b4"),
        (SOURCE_IMPL, "source + local_pmg", "#d62728"),
        (LOCAL_SOURCEFIXED_IMPL, "local_constitutiveAD + local_pmg_sourcefixed", "#2ca02c"),
        (SOURCE_SOURCEFIXED_IMPL, "source + local_pmg_sourcefixed", "#ff7f0e"),
    )
    for implementation, label, color in series:
        selected = _find_rows(rows, implementation)
        if not selected:
            continue
        converged = [row for row in selected if str(row.get("status", "")) == "completed"]
        failed = [row for row in selected if str(row.get("status", "")) != "completed"]
        if converged:
            ranks = np.asarray([int(row["ranks"]) for row in converged], dtype=np.int64)
            wall = np.asarray([_safe_float(row.get("wall_time_s")) for row in converged], dtype=np.float64)
            solve = np.asarray([_safe_float(row.get("solve_time_s")) for row in converged], dtype=np.float64)
            axes[0].plot(ranks, wall, marker="o", linewidth=2.0, color=color, label=label)
            axes[1].plot(ranks, solve, marker="o", linewidth=2.0, color=color, label=label)
            impl_assets._plot_ideal_reference(axes[0], ranks, wall, color=color)
            impl_assets._plot_ideal_reference(axes[1], ranks, solve, color=color)
        for ax, key in ((axes[0], "wall_time_s"), (axes[1], "solve_time_s")):
            if not failed:
                continue
            x = np.asarray([int(row["ranks"]) for row in failed], dtype=np.int64)
            y = np.asarray([_safe_float(row.get(key)) for row in failed], dtype=np.float64)
            ax.scatter(
                x,
                y,
                marker="x",
                s=70,
                color=color,
                alpha=0.9,
            )
            for row in failed:
                ax.annotate(
                    "maxit",
                    (int(row["ranks"]), _safe_float(row.get(key))),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                    color=color,
                )
    for ax, title, ylabel in (
        (axes[0], "Alternative PMG Profile Wall Time", "Wall time [s]"),
        (axes[1], "Alternative PMG Profile Solve Time", "Solve time [s]"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel(ylabel)
    axes[1].plot([], [], linewidth=1.2, linestyle=(0, (4, 3)), color="#555555", alpha=0.7, label="Ideal 1/r")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate docs assets for the promoted Plasticity3D L1_2/lambda=1 local PMG results."
    )
    parser.add_argument("--local-summary", type=Path, default=DEFAULT_LOCAL_SUMMARY)
    parser.add_argument("--mixed-summary", type=Path, default=DEFAULT_MIXED_SUMMARY)
    parser.add_argument("--sourcefixed-summary", type=Path, default=DEFAULT_SOURCEFIXED_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ASSET_DIR)
    args = parser.parse_args()

    local_summary_path = Path(args.local_summary).resolve()
    mixed_summary_path = Path(args.mixed_summary).resolve()
    sourcefixed_summary_path = Path(args.sourcefixed_summary).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    local_summary, local_rows_by_impl = _load_enriched_rows(local_summary_path)
    local_rows = list(local_rows_by_impl.get(LOCAL_IMPL, []))
    if not local_rows:
        raise RuntimeError(f"No enriched local rows found in {local_summary_path}")

    local_labels = impl_assets._implementation_labels(local_summary, [LOCAL_IMPL])
    local_colors = impl_assets._implementation_colors([LOCAL_IMPL])

    impl_assets._plot_overall_scaling(
        local_rows_by_impl,
        implementation_order=[LOCAL_IMPL],
        labels=local_labels,
        colors=local_colors,
        out_path=out_dir / OVERALL_PLOT,
        show_ideal=True,
    )
    impl_assets._plot_common_components(
        local_rows_by_impl,
        implementation_order=[LOCAL_IMPL],
        labels=local_labels,
        colors=local_colors,
        out_path=out_dir / COMMON_PLOT,
        show_ideal=True,
    )
    impl_assets._plot_impl_breakdown(
        local_rows,
        title="local_constitutiveAD + local_pmg Component Scaling",
        component_keys=[
            ("newton_linear_assemble_s", "Linear assemble"),
            ("newton_linear_setup_s", "Linear setup"),
            ("newton_linear_solve_s", "Linear solve"),
            ("newton_line_search_s", "Line search"),
            ("callback_hessian_s", "Hessian callback"),
            ("callback_gradient_s", "Gradient callback"),
        ],
        color=local_colors[LOCAL_IMPL],
        out_path=out_dir / BREAKDOWN_PLOT,
        show_ideal=True,
    )

    mixed_summary = _read_json(mixed_summary_path)
    mixed_rows = _comparison_rows(mixed_summary)
    _plot_local_vs_source(mixed_rows, out_dir / LOCAL_VS_SOURCE_PLOT)

    sourcefixed_summary = _read_json(sourcefixed_summary_path)
    sourcefixed_rows = _comparison_rows(sourcefixed_summary)
    _plot_sourcefixed(sourcefixed_rows, out_dir / SOURCEFIXED_PLOT)

    payload = {
        "recommended_stack": {
            "mesh_name": "hetero_ssr_L1_2",
            "space": "P4",
            "lambda_target": 1.0,
            "assembly": "local_constitutiveAD",
            "solver": "local_pmg",
            "line_search": "armijo",
            "stop_metric_name": str(local_summary.get("stop_metric_name", "grad_norm")),
            "grad_stop_tol": _safe_float(local_summary.get("grad_stop_tol")),
            "maxit": int(local_summary.get("maxit", 0)),
            "ksp_rtol": _safe_float(local_summary.get("linear_solver_rtol")),
            "ksp_max_it": 100,
            "threads": "OMP/JAX/BLAS = 1",
            "hierarchy": "same_mesh_p4_p2_p1",
            "fine_smoothers": "chebyshev + jacobi, 5 steps",
            "coarse_solve": "cg + hypre",
        },
        "local_only_scaling": _local_component_rows(local_rows),
        "local_vs_source_comparison": mixed_rows,
        "sourcefixed_comparison": sourcefixed_rows,
        "assets": {
            "overall_scaling": OVERALL_PLOT,
            "common_components": COMMON_PLOT,
            "local_breakdown": BREAKDOWN_PLOT,
            "local_vs_source": LOCAL_VS_SOURCE_PLOT,
            "sourcefixed": SOURCEFIXED_PLOT,
        },
    }
    (out_dir / SUMMARY_JSON_NAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

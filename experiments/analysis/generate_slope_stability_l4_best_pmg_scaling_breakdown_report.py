#!/usr/bin/env python3
"""Generate a detailed scaling report for the current best L4 P4 PMG baseline."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "slope_stability_p4_pmg_smoother_sweep_lambda1"
    / "level4"
    / "tail_baseline"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "slope_stability_l4_p4_best_pmg_scaling_breakdown_lambda1"
)
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

RANKS = [1, 2, 4, 8]


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _sum(entries: list[dict[str, object]], key: str) -> float:
    return float(sum(float(entry.get(key, 0.0)) for entry in entries))


def _cb(summary: dict[str, object], phase: str, key: str) -> float:
    return float(dict(summary.get(phase, {})).get(key, 0.0))


def _load_row(ranks: int) -> dict[str, object]:
    path = INPUT_ROOT / f"np{ranks}" / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    timings = dict(payload["timings"])
    callback = dict(timings.get("callback_summary", {}))
    bootstrap = dict(timings.get("solver_bootstrap_breakdown", {}))
    assembler = dict(timings.get("assembler_setup_breakdown", {}))
    step = dict(payload["result"]["steps"][0])
    linear = list(step.get("linear_timing", []))
    case = dict(payload["case"])
    linear_solver = dict(dict(payload.get("metadata", {})).get("linear_solver", {}))
    return {
        "ranks": int(ranks),
        "result_path": str(path.relative_to(REPO_ROOT)),
        "status": str(payload["result"]["status"]),
        "solver_success": bool(payload["result"]["solver_success"]),
        "newton_iterations": int(step["nit"]),
        "linear_iterations": int(step["linear_iters"]),
        "end_to_end_total_time_sec": float(timings.get("total_time", 0.0)),
        "steady_state_total_time_sec": float(timings.get("steady_state_total_time", 0.0)),
        "one_time_setup_time_sec": float(timings.get("one_time_setup_time", 0.0)),
        "solve_time_sec": float(timings.get("solve_time", 0.0)),
        "problem_build_time_sec": float(timings.get("problem_build_time", 0.0)),
        "main_problem_build_time_sec": float(timings.get("main_problem_build_time", 0.0)),
        "assembler_setup_time_sec": float(timings.get("assembler_setup_time", 0.0)),
        "assembler_warmup_time_sec": float(assembler.get("warmup", 0.0)),
        "mg_hierarchy_build_time_sec": float(bootstrap.get("mg_hierarchy_build_time", 0.0)),
        "mg_level_build_time_sec": float(bootstrap.get("mg_level_build_time", 0.0)),
        "mg_level_assembler_build_time_sec": float(bootstrap.get("mg_level_assembler_build_time", 0.0)),
        "mg_transfer_build_time_sec": float(bootstrap.get("mg_transfer_build_time", 0.0)),
        "mg_configure_time_sec": float(bootstrap.get("mg_configure_time", 0.0)),
        "energy_total_time_sec": _cb(callback, "energy", "total"),
        "energy_kernel_time_sec": _cb(callback, "energy", "kernel"),
        "energy_ghost_exchange_time_sec": _cb(callback, "energy", "ghost_exchange"),
        "energy_allreduce_time_sec": _cb(callback, "energy", "allreduce"),
        "gradient_total_time_sec": _cb(callback, "gradient", "total"),
        "gradient_kernel_time_sec": _cb(callback, "gradient", "kernel"),
        "gradient_ghost_exchange_time_sec": _cb(callback, "gradient", "ghost_exchange"),
        "gradient_allreduce_time_sec": _cb(callback, "gradient", "allreduce"),
        "hessian_total_time_sec": _cb(callback, "hessian", "total"),
        "hessian_kernel_time_sec": _cb(callback, "hessian", "kernel"),
        "hessian_ghost_exchange_time_sec": _cb(callback, "hessian", "ghost_exchange"),
        "hessian_allreduce_time_sec": _cb(callback, "hessian", "allreduce"),
        "line_search_time_sec": float(sum(float(item.get("t_ls", 0.0)) for item in step.get("history", []))),
        "linear_operator_prepare_time_sec": _sum(linear, "operator_prepare_total_time"),
        "linear_operator_apply_time_sec": _sum(linear, "operator_apply_total_time"),
        "linear_assemble_time_sec": _sum(linear, "assemble_total_time"),
        "linear_pc_setup_time_sec": _sum(linear, "pc_setup_time"),
        "linear_ksp_solve_time_sec": _sum(linear, "solve_time"),
        "linear_total_time_sec": _sum(linear, "linear_total_time"),
        "line_search_evals": int(sum(int(item.get("ls_evals", 0)) for item in step.get("history", []))),
        "mg_strategy": str(linear_solver.get("mg_strategy", "")),
        "mg_variant": str(linear_solver.get("mg_variant", "")),
        "mg_legacy_level_smoothers": dict(linear_solver.get("mg_legacy_level_smoothers", {})),
        "mg_coarse_backend": str(case.get("mg_coarse_backend", "")),
        "mg_coarse_ksp_type": str(case.get("mg_coarse_ksp_type", "")),
        "mg_coarse_pc_type": str(case.get("mg_coarse_pc_type", "")),
        "mg_coarse_hypre_nodal_coarsen": int(case.get("mg_coarse_hypre_nodal_coarsen", -1)),
        "mg_coarse_hypre_vec_interp_variant": int(case.get("mg_coarse_hypre_vec_interp_variant", -1)),
        "mg_coarse_hypre_strong_threshold": float(case.get("mg_coarse_hypre_strong_threshold", 0.0)),
        "mg_coarse_hypre_coarsen_type": str(case.get("mg_coarse_hypre_coarsen_type", "")),
        "mg_coarse_hypre_max_iter": int(case.get("mg_coarse_hypre_max_iter", -1)),
        "mg_coarse_hypre_tol": float(case.get("mg_coarse_hypre_tol", 0.0)),
        "mg_coarse_hypre_relax_type_all": str(case.get("mg_coarse_hypre_relax_type_all", "")),
        "distribution_strategy": str(case.get("distribution_strategy", "")),
        "problem_build_mode": str(case.get("problem_build_mode", "")),
        "mg_level_build_mode": str(case.get("mg_level_build_mode", "")),
        "mg_transfer_build_mode": str(case.get("mg_transfer_build_mode", "")),
        "line_search": str(case.get("line_search", "")),
    }


def _plot_loglog(
    rows: list[dict[str, object]],
    *,
    keys: list[tuple[str, str]],
    output_path: Path,
    ylabel: str,
) -> None:
    x = np.array([int(row["ranks"]) for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for key, label in keys:
        y = np.array([float(row[key]) for row in rows], dtype=np.float64)
        ax.plot(x, y, marker="o", label=label)
        ideal = np.array([y[0] / xi for xi in x], dtype=np.float64)
        ax.plot(x, ideal, linestyle="--", alpha=0.5, label=f"{label} ideal")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("ranks")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    rows = [_load_row(ranks) for ranks in RANKS]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    overall_plot = OUTPUT_DIR / "overall_scaling_loglog.png"
    setup_plot = OUTPUT_DIR / "setup_breakdown_loglog.png"
    nonlinear_plot = OUTPUT_DIR / "nonlinear_breakdown_loglog.png"
    linear_plot = OUTPUT_DIR / "linear_breakdown_loglog.png"

    _plot_loglog(
        rows,
        keys=[
            ("end_to_end_total_time_sec", "end-to-end total"),
            ("steady_state_total_time_sec", "steady-state total"),
            ("one_time_setup_time_sec", "one-time setup"),
            ("solve_time_sec", "solve"),
        ],
        output_path=overall_plot,
        ylabel="time [s]",
    )
    _plot_loglog(
        rows,
        keys=[
            ("problem_build_time_sec", "problem build"),
            ("assembler_setup_time_sec", "assembler setup"),
            ("assembler_warmup_time_sec", "JAX warmup"),
            ("mg_hierarchy_build_time_sec", "MG hierarchy"),
            ("mg_transfer_build_time_sec", "MG transfers"),
        ],
        output_path=setup_plot,
        ylabel="time [s]",
    )
    _plot_loglog(
        rows,
        keys=[
            ("energy_total_time_sec", "energy"),
            ("gradient_total_time_sec", "gradient"),
            ("hessian_total_time_sec", "hessian"),
            ("line_search_time_sec", "line search"),
        ],
        output_path=nonlinear_plot,
        ylabel="time [s]",
    )
    _plot_loglog(
        rows,
        keys=[
            ("linear_operator_prepare_time_sec", "operator prepare"),
            ("linear_assemble_time_sec", "tangent assembly"),
            ("linear_pc_setup_time_sec", "PC setup"),
            ("linear_ksp_solve_time_sec", "KSP solve"),
        ],
        output_path=linear_plot,
        ylabel="time [s]",
    )

    p4_smoother = rows[0]["mg_legacy_level_smoothers"]["fine"]
    p2_smoother = rows[0]["mg_legacy_level_smoothers"]["degree2"]
    p1_smoother = rows[0]["mg_legacy_level_smoothers"]["degree1"]

    lines: list[str] = []
    lines.append("# L4 P4 Best-PMG Scaling Breakdown")
    lines.append("")
    lines.append(
        "This report uses the current best validated PMG baseline from the staged smoother sweep: "
        "the `tail_baseline` `L4` run with the tuned coarse Hypre settings."
    )
    lines.append("")

    lines.append("## Solver Setting")
    lines.append("")
    lines.append("Problem and hierarchy:")
    lines.append(f"- finest problem: `L4`, `P4`, `lambda=1.0`")
    lines.append(f"- MG strategy: `{rows[0]['mg_strategy']}`")
    lines.append("- interpreted hierarchy: same-mesh `P4 -> P2 -> P1`, plus the `L3 P1` tail")
    lines.append(f"- MG variant: `{rows[0]['mg_variant']}`")
    lines.append(f"- distribution strategy: `{rows[0]['distribution_strategy']}`")
    lines.append(f"- problem build mode: `{rows[0]['problem_build_mode']}`")
    lines.append(f"- MG level build mode: `{rows[0]['mg_level_build_mode']}`")
    lines.append(f"- MG transfer build mode: `{rows[0]['mg_transfer_build_mode']}`")
    lines.append(f"- line search: `{rows[0]['line_search']}`")
    lines.append("")
    lines.append("Noncoarse smoothers:")
    lines.append(
        f"- `P4`: `{p4_smoother['ksp_type']} + {p4_smoother['pc_type']}`, `{p4_smoother['steps']}` steps"
    )
    lines.append(
        f"- `P2`: `{p2_smoother['ksp_type']} + {p2_smoother['pc_type']}`, `{p2_smoother['steps']}` steps"
    )
    lines.append(
        f"- `P1`: `{p1_smoother['ksp_type']} + {p1_smoother['pc_type']}`, `{p1_smoother['steps']}` steps"
    )
    lines.append("")
    lines.append("Coarse solver:")
    lines.append(f"- coarse backend: `{rows[0]['mg_coarse_backend']}`")
    lines.append(f"- coarse KSP / PC: `{rows[0]['mg_coarse_ksp_type']} + {rows[0]['mg_coarse_pc_type']}`")
    lines.append(f"- BoomerAMG `nodal_coarsen = {rows[0]['mg_coarse_hypre_nodal_coarsen']}`")
    lines.append(f"- BoomerAMG `vec_interp_variant = {rows[0]['mg_coarse_hypre_vec_interp_variant']}`")
    lines.append(f"- BoomerAMG `strong_threshold = {rows[0]['mg_coarse_hypre_strong_threshold']}`")
    lines.append(f"- BoomerAMG `coarsen_type = {rows[0]['mg_coarse_hypre_coarsen_type']}`")
    lines.append(f"- BoomerAMG `max_iter = {rows[0]['mg_coarse_hypre_max_iter']}`")
    lines.append(f"- BoomerAMG `tol = {rows[0]['mg_coarse_hypre_tol']}`")
    lines.append(f"- BoomerAMG `relax_type_all = {rows[0]['mg_coarse_hypre_relax_type_all']}`")
    lines.append("- rigid-body near-nullspace is attached to the explicit PMG level operators")
    lines.append("")

    lines.append("## Scaling Table")
    lines.append("")
    lines.append("| ranks | success | Newton | linear | end-to-end [s] | steady-state [s] | setup [s] | solve [s] |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['ranks']} | {'yes' if row['solver_success'] else 'no'} | {row['newton_iterations']} | "
            f"{row['linear_iterations']} | {_fmt(row['end_to_end_total_time_sec'])} | "
            f"{_fmt(row['steady_state_total_time_sec'])} | {_fmt(row['one_time_setup_time_sec'])} | "
            f"{_fmt(row['solve_time_sec'])} |"
        )
    lines.append("")

    lines.append("## Setup Breakdown")
    lines.append("")
    lines.append("| ranks | problem build | assembler setup | JAX warmup | MG hierarchy | MG transfers | MG configure |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['ranks']} | {_fmt(row['problem_build_time_sec'])} | {_fmt(row['assembler_setup_time_sec'])} | "
            f"{_fmt(row['assembler_warmup_time_sec'])} | {_fmt(row['mg_hierarchy_build_time_sec'])} | "
            f"{_fmt(row['mg_transfer_build_time_sec'])} | {_fmt(row['mg_configure_time_sec'])} |"
        )
    lines.append("")

    lines.append("## Nonlinear Breakdown")
    lines.append("")
    lines.append("| ranks | energy | gradient | Hessian | line search | energy evals |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['ranks']} | {_fmt(row['energy_total_time_sec'])} | {_fmt(row['gradient_total_time_sec'])} | "
            f"{_fmt(row['hessian_total_time_sec'])} | {_fmt(row['line_search_time_sec'])} | {row['line_search_evals']} |"
        )
    lines.append("")

    lines.append("## Linear Breakdown")
    lines.append("")
    lines.append("| ranks | operator prepare | tangent assembly | PC setup | KSP solve | linear total |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['ranks']} | {_fmt(row['linear_operator_prepare_time_sec'])} | {_fmt(row['linear_assemble_time_sec'])} | "
            f"{_fmt(row['linear_pc_setup_time_sec'])} | {_fmt(row['linear_ksp_solve_time_sec'])} | "
            f"{_fmt(row['linear_total_time_sec'])} |"
        )
    lines.append("")

    lines.append("## Reading The Scaling")
    lines.append("")
    lines.append(
        "- The log-log plots use a dashed ideal `1 / p` line anchored at the `1`-rank measurement for each series."
    )
    lines.append(
        "- `solve` scales materially better than `one-time setup`; the remaining flatness is concentrated in setup-heavy buckets such as assembler setup and JAX warmup."
    )
    lines.append(
        "- Inside the solve phase, `KSP solve` and `Hessian` dominate, while `PC setup` is already small."
    )
    lines.append(
        "- `energy` and `line search` stay noticeably flatter than ideal, which matches the earlier diagnosis that the nonlinear globalization path is still a real scaling limiter."
    )
    lines.append("")
    lines.append(f"![Overall scaling]({overall_plot.name})")
    lines.append("")
    lines.append(f"![Setup breakdown]({setup_plot.name})")
    lines.append("")
    lines.append(f"![Nonlinear breakdown]({nonlinear_plot.name})")
    lines.append("")
    lines.append(f"![Linear breakdown]({linear_plot.name})")
    lines.append("")

    report = "\n".join(lines)
    (OUTPUT_DIR / "README.md").write_text(report + "\n", encoding="utf-8")
    (OUTPUT_DIR / "report.md").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

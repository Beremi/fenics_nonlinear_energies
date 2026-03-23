#!/usr/bin/env python3
"""Generate a per-part scaling report for the instrumented L5 P4 PMG solve."""

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
    / "slope_stability_l5_p4_pmg_part_scaling_lambda1"
)
LEGACY_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "slope_stability_p4_pmg_smoother_sweep_lambda1"
    / "level5"
    / "tail_baseline"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "slope_stability_l5_p4_pmg_part_scaling_lambda1"
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


def _plot_loglog(
    rows: list[dict[str, object]],
    *,
    keys: list[tuple[str, str]],
    output_path: Path,
    ylabel: str,
) -> None:
    x = np.array([int(row["ranks"]) for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for key, label in keys:
        y = np.array([float(row.get(key, 0.0)) for row in rows], dtype=np.float64)
        ax.plot(x, y, marker="o", label=label)
        ideal = np.array([y[0] / xi for xi in x], dtype=np.float64)
        ax.plot(x, ideal, linestyle="--", alpha=0.45, label=f"{label} ideal")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("ranks")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _sum(entries: list[dict[str, object]], key: str) -> float:
    return float(sum(float(entry.get(key, 0.0)) for entry in entries))


def _load_legacy_counts(ranks: int) -> tuple[int | None, int | None]:
    path = LEGACY_ROOT / f"np{ranks}" / "result.json"
    if not path.exists():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    step = dict(payload["result"]["steps"][0])
    return int(step["nit"]), int(step["linear_iters"])


def _aggregate_mg_runtime(linear_records: list[dict[str, object]]) -> dict[str, float]:
    totals: dict[str, float] = {
        "transfer_prolongation_time_sec": 0.0,
        "transfer_restriction_time_sec": 0.0,
        "p1_smoother_time_sec": 0.0,
        "p2_smoother_time_sec": 0.0,
        "p4_smoother_time_sec": 0.0,
        "coarse_solve_time_sec": 0.0,
        "galerkin_refresh_time_sec": 0.0,
        "pc_setup_time_sec": 0.0,
        "transfer_l4p1_to_l5p1_prolong_time_sec": 0.0,
        "transfer_l5p1_to_l5p2_prolong_time_sec": 0.0,
        "transfer_l5p2_to_l5p4_prolong_time_sec": 0.0,
    }
    for record in linear_records:
        totals["galerkin_refresh_time_sec"] += float(
            record.get("pc_operator_assemble_total_time", 0.0)
        )
        totals["pc_setup_time_sec"] += float(record.get("pc_setup_time", 0.0))
        for diag in record.get("mg_runtime_diagnostics", []):
            kind = str(diag.get("kind", "ksp"))
            observed = float(diag.get("observed_time_sec", 0.0))
            if kind == "transfer":
                role = str(diag.get("sweep_role", ""))
                mesh_level = int(diag.get("mesh_level", -1))
                degree = int(diag.get("degree", -1))
                target_mesh_level = int(diag.get("target_mesh_level", -1))
                target_degree = int(diag.get("target_degree", -1))
                if role == "prolongation":
                    totals["transfer_prolongation_time_sec"] += observed
                    detail_key = (
                        f"transfer_l{target_mesh_level}p{target_degree}_to_l{mesh_level}p{degree}_prolong_time_sec"
                    )
                    if detail_key in totals:
                        totals[detail_key] += observed
                elif role == "restriction":
                    totals["transfer_restriction_time_sec"] += observed
                continue

            family = str(diag.get("family", ""))
            if family == "degree1":
                totals["p1_smoother_time_sec"] += observed
            elif family == "degree2":
                totals["p2_smoother_time_sec"] += observed
            elif family == "fine":
                totals["p4_smoother_time_sec"] += observed
            elif family == "coarse":
                totals["coarse_solve_time_sec"] += observed
    return totals


def _load_row(ranks: int) -> dict[str, object]:
    path = INPUT_ROOT / f"np{ranks}" / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    timings = dict(payload["timings"])
    bootstrap = dict(timings.get("solver_bootstrap_breakdown", {}))
    step = dict(payload["result"]["steps"][0])
    linear = list(step.get("linear_timing", []))
    case = dict(payload["case"])
    linear_solver = dict(dict(payload.get("metadata", {})).get("linear_solver", {}))
    legacy_newton, legacy_linear = _load_legacy_counts(ranks)
    mg_parts = _aggregate_mg_runtime(linear)
    return {
        "ranks": int(ranks),
        "result_path": str(path.relative_to(REPO_ROOT)),
        "status": str(payload["result"]["status"]),
        "solver_success": bool(payload["result"]["solver_success"]),
        "newton_iterations": int(step["nit"]),
        "linear_iterations": int(step["linear_iters"]),
        "legacy_newton_iterations": legacy_newton,
        "legacy_linear_iterations": legacy_linear,
        "steady_state_total_time_sec": float(timings.get("steady_state_total_time", 0.0)),
        "end_to_end_total_time_sec": float(timings.get("total_time", 0.0)),
        "solve_time_sec": float(timings.get("solve_time", 0.0)),
        "one_time_setup_time_sec": float(timings.get("one_time_setup_time", 0.0)),
        "problem_build_time_sec": float(timings.get("problem_build_time", 0.0)),
        "assembler_setup_time_sec": float(timings.get("assembler_setup_time", 0.0)),
        "mg_hierarchy_build_time_sec": float(bootstrap.get("mg_hierarchy_build_time", 0.0)),
        "mg_transfer_build_time_sec": float(bootstrap.get("mg_transfer_build_time", 0.0)),
        "linear_total_time_sec": _sum(linear, "linear_total_time"),
        "linear_assemble_time_sec": _sum(linear, "assemble_total_time"),
        "linear_ksp_solve_time_sec": _sum(linear, "solve_time"),
        "mg_strategy": str(linear_solver.get("mg_strategy", case.get("mg_strategy", ""))),
        "mg_variant": str(linear_solver.get("mg_variant", case.get("mg_variant", ""))),
        "mg_lower_operator_policy": str(
            linear_solver.get("mg_lower_operator_policy", case.get("mg_lower_operator_policy", ""))
        ),
        "mg_p4_smoother_ksp_type": str(
            dict(linear_solver.get("mg_fine_down", {})).get("ksp_type", case.get("mg_fine_ksp_type", ""))
        ),
        "mg_p4_smoother_pc_type": str(
            dict(linear_solver.get("mg_fine_down", {})).get("pc_type", case.get("mg_fine_pc_type", ""))
        ),
        "mg_p4_smoother_steps": int(
            dict(linear_solver.get("mg_fine_down", {})).get("steps", case.get("mg_fine_steps", 0))
        ),
        "mg_p2_smoother_pc_type": str(
            dict(linear_solver.get("mg_intermediate_degree_pc_types", {})).get(
                "2",
                case.get("mg_degree2_pc_type", ""),
            )
        ),
        "mg_p1_smoother_pc_type": str(
            dict(linear_solver.get("mg_intermediate_degree_pc_types", {})).get(
                "1",
                case.get("mg_degree1_pc_type", ""),
            )
        ),
        "mg_intermediate_steps": int(
            linear_solver.get("mg_intermediate_steps", case.get("mg_intermediate_steps", 0))
        ),
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
        **mg_parts,
    }


def main() -> None:
    rows = [_load_row(ranks) for ranks in RANKS]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    _plot_loglog(
        rows,
        keys=[
            ("end_to_end_total_time_sec", "end-to-end total"),
            ("steady_state_total_time_sec", "steady-state total"),
            ("solve_time_sec", "solve"),
        ],
        output_path=OUTPUT_DIR / "overall_scaling_loglog.png",
        ylabel="time [s]",
    )
    _plot_loglog(
        rows,
        keys=[
            ("galerkin_refresh_time_sec", "Galerkin refresh"),
            ("pc_setup_time_sec", "PC setup"),
            ("transfer_prolongation_time_sec", "prolongation"),
            ("transfer_restriction_time_sec", "restriction"),
        ],
        output_path=OUTPUT_DIR / "pc_transfer_scaling_loglog.png",
        ylabel="time [s]",
    )
    _plot_loglog(
        rows,
        keys=[
            ("p1_smoother_time_sec", "P1 smoother"),
            ("p2_smoother_time_sec", "P2 smoother"),
            ("p4_smoother_time_sec", "P4 smoother"),
            ("coarse_solve_time_sec", "coarse solve"),
        ],
        output_path=OUTPUT_DIR / "pc_smoother_scaling_loglog.png",
        ylabel="time [s]",
    )
    _plot_loglog(
        rows,
        keys=[
            ("transfer_l4p1_to_l5p1_prolong_time_sec", "L4 P1 -> L5 P1"),
            ("transfer_l5p1_to_l5p2_prolong_time_sec", "L5 P1 -> L5 P2"),
            ("transfer_l5p2_to_l5p4_prolong_time_sec", "L5 P2 -> L5 P4"),
        ],
        output_path=OUTPUT_DIR / "prolongation_detail_scaling_loglog.png",
        ylabel="time [s]",
    )

    row0 = rows[0]
    lines: list[str] = []
    lines.append("# L5 P4 PMG Part Scaling")
    lines.append("")
    lines.append(
        "This report repeats the instrumented explicit-Galerkin PMG part benchmark on `L5` "
        "to check whether the poor PMG-part scaling is just an `L4` small-problem effect or a deeper issue."
    )
    lines.append("")
    lines.append("## PMG Setting")
    lines.append("")
    lines.append(f"- finest problem: `L5`, `P4`, `lambda=1.0`")
    lines.append(f"- hierarchy: `{row0['mg_strategy']}`")
    lines.append("- interpreted as same-mesh `P4 -> P2 -> P1` plus the `L4 P1` tail")
    lines.append(f"- MG variant: `{row0['mg_variant']}`")
    lines.append(f"- lower-level operator policy: `{row0['mg_lower_operator_policy']}`")
    lines.append("- coarse/fine operators are refreshed by explicit Galerkin `P^T A P` at each Newton linear solve")
    lines.append("")
    lines.append("Smoothers:")
    lines.append(
        f"- `P4`: `{row0['mg_p4_smoother_ksp_type']} + {row0['mg_p4_smoother_pc_type']}`, `{row0['mg_p4_smoother_steps']}` steps"
    )
    lines.append(
        f"- `P2`: `richardson + {row0['mg_p2_smoother_pc_type']}`, `{row0['mg_intermediate_steps']}` steps"
    )
    lines.append(
        f"- `P1`: `richardson + {row0['mg_p1_smoother_pc_type']}`, `{row0['mg_intermediate_steps']}` steps"
    )
    lines.append("")
    lines.append("Coarse solve:")
    lines.append(
        f"- coarse KSP / PC: `{row0['mg_coarse_ksp_type']} + {row0['mg_coarse_pc_type']}`"
    )
    lines.append(f"- BoomerAMG `nodal_coarsen = {row0['mg_coarse_hypre_nodal_coarsen']}`")
    lines.append(f"- BoomerAMG `vec_interp_variant = {row0['mg_coarse_hypre_vec_interp_variant']}`")
    lines.append(f"- BoomerAMG `strong_threshold = {row0['mg_coarse_hypre_strong_threshold']}`")
    lines.append(f"- BoomerAMG `coarsen_type = {row0['mg_coarse_hypre_coarsen_type']}`")
    lines.append(f"- BoomerAMG `max_iter = {row0['mg_coarse_hypre_max_iter']}`")
    lines.append(f"- BoomerAMG `tol = {row0['mg_coarse_hypre_tol']}`")
    lines.append(f"- BoomerAMG `relax_type_all = {row0['mg_coarse_hypre_relax_type_all']}`")
    lines.append("")

    lines.append("## Global Scaling")
    lines.append("")
    lines.append("| ranks | success | Newton | linear | legacy Newton | legacy linear | steady-state [s] | end-to-end [s] |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['ranks']} | {'yes' if row['solver_success'] else 'no'} | "
            f"{row['newton_iterations']} | {row['linear_iterations']} | "
            f"{row['legacy_newton_iterations'] if row['legacy_newton_iterations'] is not None else '-'} | "
            f"{row['legacy_linear_iterations'] if row['legacy_linear_iterations'] is not None else '-'} | "
            f"{_fmt(row['steady_state_total_time_sec'])} | {_fmt(row['end_to_end_total_time_sec'])} |"
        )
    lines.append("")

    lines.append("## Preconditioner Part Scaling")
    lines.append("")
    lines.append("| ranks | prolong [s] | restrict [s] | P1 smoother [s] | P2 smoother [s] | P4 smoother [s] | coarse [s] | Galerkin refresh [s] | PC setup [s] |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['ranks']} | {_fmt(row['transfer_prolongation_time_sec'])} | "
            f"{_fmt(row['transfer_restriction_time_sec'])} | {_fmt(row['p1_smoother_time_sec'])} | "
            f"{_fmt(row['p2_smoother_time_sec'])} | {_fmt(row['p4_smoother_time_sec'])} | "
            f"{_fmt(row['coarse_solve_time_sec'])} | {_fmt(row['galerkin_refresh_time_sec'])} | "
            f"{_fmt(row['pc_setup_time_sec'])} |"
        )
    lines.append("")

    lines.append("## Transfer Detail")
    lines.append("")
    lines.append("| ranks | L4 P1 -> L5 P1 [s] | L5 P1 -> L5 P2 [s] | L5 P2 -> L5 P4 [s] |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['ranks']} | {_fmt(row['transfer_l4p1_to_l5p1_prolong_time_sec'])} | "
            f"{_fmt(row['transfer_l5p1_to_l5p2_prolong_time_sec'])} | "
            f"{_fmt(row['transfer_l5p2_to_l5p4_prolong_time_sec'])} |"
        )
    lines.append("")

    lines.append("## Main Takeaways")
    lines.append("")
    lines.append(
        f"- On `1` rank, the instrumented `L5` PMG takes `{rows[0]['newton_iterations']}` Newton steps "
        f"and `{rows[0]['linear_iterations']}` total outer linear iterations."
    )
    lines.append(
        f"- At `8` ranks, the dominant timed MG pieces are `P4` smoothing `{_fmt(rows[-1]['p4_smoother_time_sec'])} s` "
        f"and coarse solve `{_fmt(rows[-1]['coarse_solve_time_sec'])} s`."
    )
    lines.append(
        f"- Transfer work remains much smaller: prolongation `{_fmt(rows[-1]['transfer_prolongation_time_sec'])} s` "
        f"and restriction `{_fmt(rows[-1]['transfer_restriction_time_sec'])} s` on `8` ranks."
    )
    lines.append(
        f"- If the same coarse-dominance pattern persists from `L4` to `L5`, that points to a deeper PMG issue rather than a pure small-problem artifact."
    )
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append("### Overall Scaling")
    lines.append("")
    lines.append("![Overall scaling](overall_scaling_loglog.png)")
    lines.append("")
    lines.append("### Transfer And PC Maintenance")
    lines.append("")
    lines.append("![Transfer and PC maintenance scaling](pc_transfer_scaling_loglog.png)")
    lines.append("")
    lines.append("### Smoother And Coarse Solve")
    lines.append("")
    lines.append("![Smoother and coarse-solve scaling](pc_smoother_scaling_loglog.png)")
    lines.append("")
    lines.append("### Prolongation Detail")
    lines.append("")
    lines.append("![Prolongation detail scaling](prolongation_detail_scaling_loglog.png)")
    lines.append("")

    for name in ("README.md", "report.md"):
        (OUTPUT_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

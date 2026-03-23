#!/usr/bin/env python3
"""Generate a per-part scaling report for the instrumented L6 P4 PMG solve with maxit=20."""

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
    / "slope_stability_l6_p4_pmg_part_scaling_lambda1_maxit20"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "slope_stability_l6_p4_pmg_part_scaling_lambda1_maxit20"
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
        "transfer_l5p1_to_l6p1_prolong_time_sec": 0.0,
        "transfer_l6p1_to_l6p2_prolong_time_sec": 0.0,
        "transfer_l6p2_to_l6p4_prolong_time_sec": 0.0,
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
    mg_parts = _aggregate_mg_runtime(linear)
    return {
        "ranks": int(ranks),
        "result_path": str(path.relative_to(REPO_ROOT)),
        "status": str(payload["result"]["status"]),
        "solver_success": bool(payload["result"]["solver_success"]),
        "newton_iterations": int(step["nit"]),
        "linear_iterations": int(step["linear_iters"]),
        "energy": float(step.get("energy", float("nan"))),
        "omega": float(step.get("omega", float("nan"))),
        "u_max": float(step.get("u_max", float("nan"))),
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
        "worst_true_relative_residual": float(
            dict(step.get("linear_summary", {})).get("worst_true_relative_residual", float("nan"))
        ),
        "last_true_relative_residual": float(
            dict(step.get("linear_summary", {})).get("last_true_relative_residual", float("nan"))
        ),
        "last_reason_name": str(
            dict(step.get("linear_summary", {})).get("last_reason_name", "")
        ),
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

    for row in rows:
        row["steady_state_overhead_sec"] = float(row["steady_state_total_time_sec"]) - float(
            row["solve_time_sec"]
        )
        row["warmup_overhead_sec"] = float(row["end_to_end_total_time_sec"]) - float(
            row["steady_state_total_time_sec"]
        )

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
            ("problem_build_time_sec", "problem build"),
            ("assembler_setup_time_sec", "assembler setup"),
            ("mg_hierarchy_build_time_sec", "MG hierarchy build"),
            ("mg_transfer_build_time_sec", "MG transfer build"),
        ],
        output_path=OUTPUT_DIR / "setup_breakdown_loglog.png",
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
            ("transfer_l5p1_to_l6p1_prolong_time_sec", "L5 P1 -> L6 P1"),
            ("transfer_l6p1_to_l6p2_prolong_time_sec", "L6 P1 -> L6 P2"),
            ("transfer_l6p2_to_l6p4_prolong_time_sec", "L6 P2 -> L6 P4"),
        ],
        output_path=OUTPUT_DIR / "prolongation_detail_scaling_loglog.png",
        ylabel="time [s]",
    )
    _plot_loglog(
        rows,
        keys=[
            ("linear_assemble_time_sec", "linear assemble"),
            ("linear_ksp_solve_time_sec", "linear KSP solve"),
            ("solve_time_sec", "solve total"),
        ],
        output_path=OUTPUT_DIR / "linear_phase_scaling_loglog.png",
        ylabel="time [s]",
    )

    row0 = rows[0]
    ref_steady = float(rows[0]["steady_state_total_time_sec"])
    ref_end = float(rows[0]["end_to_end_total_time_sec"])
    ref_energy = float(rows[-1]["energy"])

    lines: list[str] = []
    lines.append("# L6 P4 PMG Part Scaling With Newton Cap 20")
    lines.append("")
    lines.append(
        "This report repeats the instrumented explicit-Galerkin PMG part benchmark on `L6` "
        "with `--maxit 20` so we can measure every exposed PMG piece while keeping the run feasible."
    )
    lines.append("")
    lines.append("## PMG Setting")
    lines.append("")
    lines.append(f"- finest problem: `L6`, `P4`, `lambda=1.0`")
    lines.append(f"- hierarchy: `{row0['mg_strategy']}`")
    lines.append("- interpreted as same-mesh `P4 -> P2 -> P1` plus the `L5 P1` tail")
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
    lines.append("## Global Timing Scaling")
    lines.append("")
    lines.append("| ranks | success | Newton | linear | solve [s] | steady-state [s] | end-to-end [s] | steady overhead [s] | warmup overhead [s] | steady speedup | end-to-end speedup |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['ranks']} | {'yes' if row['solver_success'] else 'no'} | {row['newton_iterations']} | "
            f"{row['linear_iterations']} | {_fmt(row['solve_time_sec'])} | {_fmt(row['steady_state_total_time_sec'])} | "
            f"{_fmt(row['end_to_end_total_time_sec'])} | {_fmt(row['steady_state_overhead_sec'])} | "
            f"{_fmt(row['warmup_overhead_sec'])} | {_fmt(ref_steady / float(row['steady_state_total_time_sec']))} | "
            f"{_fmt(ref_end / float(row['end_to_end_total_time_sec']))} |"
        )
    lines.append("")
    lines.append("## Setup Breakdown")
    lines.append("")
    lines.append("| ranks | problem build [s] | assembler setup [s] | MG hierarchy [s] | MG transfer build [s] |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['ranks']} | {_fmt(row['problem_build_time_sec'])} | {_fmt(row['assembler_setup_time_sec'])} | "
            f"{_fmt(row['mg_hierarchy_build_time_sec'])} | {_fmt(row['mg_transfer_build_time_sec'])} |"
        )
    lines.append("")
    lines.append("## Preconditioner Part Scaling")
    lines.append("")
    lines.append("| ranks | prolong [s] | restrict [s] | P1 smoother [s] | P2 smoother [s] | P4 smoother [s] | coarse [s] | Galerkin refresh [s] | PC setup [s] |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['ranks']} | {_fmt(row['transfer_prolongation_time_sec'])} | {_fmt(row['transfer_restriction_time_sec'])} | "
            f"{_fmt(row['p1_smoother_time_sec'])} | {_fmt(row['p2_smoother_time_sec'])} | {_fmt(row['p4_smoother_time_sec'])} | "
            f"{_fmt(row['coarse_solve_time_sec'])} | {_fmt(row['galerkin_refresh_time_sec'])} | {_fmt(row['pc_setup_time_sec'])} |"
        )
    lines.append("")
    lines.append("## Transfer Detail")
    lines.append("")
    lines.append("| ranks | L5 P1 -> L6 P1 [s] | L6 P1 -> L6 P2 [s] | L6 P2 -> L6 P4 [s] |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['ranks']} | {_fmt(row['transfer_l5p1_to_l6p1_prolong_time_sec'])} | "
            f"{_fmt(row['transfer_l6p1_to_l6p2_prolong_time_sec'])} | {_fmt(row['transfer_l6p2_to_l6p4_prolong_time_sec'])} |"
        )
    lines.append("")
    lines.append("## Final Energy And Linear Error At `maxit=20`")
    lines.append("")
    lines.append("| ranks | success | energy | |E - E(rank 8)| | omega | u_max | worst true rel | last true rel | last KSP reason |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        energy = float(row["energy"])
        lines.append(
            f"| {row['ranks']} | {'yes' if row['solver_success'] else 'no'} | {_fmt(energy, 9)} | {_fmt(abs(energy - ref_energy), 9)} | "
            f"{_fmt(row['omega'], 9)} | {_fmt(row['u_max'], 9)} | {_fmt(row['worst_true_relative_residual'], 6)} | "
            f"{_fmt(row['last_true_relative_residual'], 6)} | `{row['last_reason_name']}` |"
        )
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append("### Overall Scaling")
    lines.append("")
    lines.append("![Overall scaling](overall_scaling_loglog.png)")
    lines.append("")
    lines.append("### Setup Breakdown")
    lines.append("")
    lines.append("![Setup breakdown scaling](setup_breakdown_loglog.png)")
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
    lines.append("### Linear Phase Breakdown")
    lines.append("")
    lines.append("![Linear phase scaling](linear_phase_scaling_loglog.png)")
    lines.append("")
    lines.append("## Raw Data")
    lines.append("")
    lines.append(
        "- matching raw summary: `/home/michal/repos/fenics_nonlinear_energies/artifacts/raw_results/slope_stability_l6_p4_pmg_part_scaling_lambda1_maxit20/summary.json`"
    )

    for name in ("README.md", "report.md"):
        (OUTPUT_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

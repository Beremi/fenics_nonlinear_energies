#!/usr/bin/env python3
"""Compare golden_fixed vs armijo on the L6 P4 deep-P1-tail PMG run at 8 MPI ranks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.problems.slope_stability.support import ensure_same_mesh_case_hdf5


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SOLVER = (
    REPO_ROOT
    / "src"
    / "problems"
    / "slope_stability"
    / "jax_petsc"
    / "solve_slope_stability_dof.py"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "slope_stability_l6_p4_deep_p1_tail_line_search_compare_lambda1_np8_maxit20"
)
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

RANKS = 8
CUSTOM_HIERARCHY = "1:1,2:1,3:1,4:1,5:1,6:1,6:2,6:4"
COMMON_ARGS = [
    "--level",
    "6",
    "--elem_degree",
    "4",
    "--lambda-target",
    "1.0",
    "--profile",
    "performance",
    "--pc_type",
    "mg",
    "--mg_strategy",
    "custom_mixed",
    "--mg_custom_hierarchy",
    CUSTOM_HIERARCHY,
    "--mg_variant",
    "legacy_pmg",
    "--ksp_type",
    "fgmres",
    "--ksp_rtol",
    "1e-2",
    "--ksp_max_it",
    "100",
    "--maxit",
    "20",
    "--benchmark_mode",
    "warmup_once_then_solve",
    "--save-history",
    "--save-linear-timing",
    "--quiet",
    "--no-use_trust_region",
    "--distribution_strategy",
    "overlap_p2p",
    "--problem_build_mode",
    "root_bcast",
    "--mg_level_build_mode",
    "root_bcast",
    "--mg_transfer_build_mode",
    "owned_rows",
    "--mg_p4_smoother_ksp_type",
    "richardson",
    "--mg_p4_smoother_pc_type",
    "sor",
    "--mg_p4_smoother_steps",
    "3",
    "--mg_p2_smoother_ksp_type",
    "richardson",
    "--mg_p2_smoother_pc_type",
    "sor",
    "--mg_p2_smoother_steps",
    "3",
    "--mg_p1_smoother_ksp_type",
    "richardson",
    "--mg_p1_smoother_pc_type",
    "sor",
    "--mg_p1_smoother_steps",
    "3",
    "--mg_coarse_backend",
    "hypre",
    "--mg_coarse_ksp_type",
    "cg",
    "--mg_coarse_pc_type",
    "hypre",
    "--mg_coarse_hypre_nodal_coarsen",
    "6",
    "--mg_coarse_hypre_vec_interp_variant",
    "3",
    "--mg_coarse_hypre_strong_threshold",
    "0.5",
    "--mg_coarse_hypre_coarsen_type",
    "HMIS",
    "--mg_coarse_hypre_max_iter",
    "4",
    "--mg_coarse_hypre_tol",
    "0.0",
    "--mg_coarse_hypre_relax_type_all",
    "symmetric-SOR/Jacobi",
]

VARIANTS = [
    {
        "name": "golden_fixed",
        "label": "Golden fixed",
        "args": ["--line_search", "golden_fixed"],
    },
    {
        "name": "armijo",
        "label": "Armijo",
        "args": ["--line_search", "armijo"],
    },
    {
        "name": "armijo_ksp15",
        "label": "Armijo + ksp_max_it=15 + accept cap",
        "args": [
            "--line_search",
            "armijo",
            "--ksp_max_it",
            "15",
            "--accept_ksp_maxit_direction",
        ],
    },
]


def _ensure_assets() -> None:
    for level in range(1, 7):
        for degree in (1, 2, 4):
            ensure_same_mesh_case_hdf5(level, degree)


def _load_rows() -> list[dict[str, object]]:
    if SUMMARY_PATH.exists():
        return list(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    return []


def _write_rows(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _sum(items: list[dict[str, object]], key: str) -> float:
    return float(sum(float(item.get(key, 0.0)) for item in items))


def _callback_value(callback: dict[str, object], phase: str, key: str) -> float:
    return float(dict(callback.get(str(phase), {})).get(str(key), 0.0))


def _command(variant: dict[str, object], out: Path) -> list[str]:
    return [
        "mpiexec",
        "-n",
        str(RANKS),
        str(PYTHON),
        "-u",
        str(SOLVER),
        *COMMON_ARGS,
        *list(variant["args"]),
        "--out",
        str(out),
    ]


def _aggregate(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload.get("result", {}))
    steps = list(result.get("steps", []))
    last_step = steps[-1] if steps else {}
    history = list(last_step.get("history", []))
    linear = list(last_step.get("linear_timing", []))
    timings = dict(payload.get("timings", {}))
    callback = dict(timings.get("callback_summary", {}))
    linear_summary = dict(last_step.get("linear_summary", {}))

    return {
        "solver_success": bool(result.get("solver_success", False)),
        "status": str(result.get("status", "")),
        "message": str(last_step.get("message", "")),
        "newton_iterations": int(last_step.get("nit", len(steps))),
        "linear_iterations": int(last_step.get("linear_iters", 0)),
        "line_search_evals": int(sum(int(item.get("ls_evals", 0)) for item in history)),
        "line_search_time_sec": float(sum(float(item.get("t_ls", 0.0)) for item in history)),
        "gradient_stage_time_sec": float(sum(float(item.get("t_grad", 0.0)) for item in history)),
        "hessian_stage_time_sec": float(sum(float(item.get("t_hess", 0.0)) for item in history)),
        "update_time_sec": float(sum(float(item.get("t_update", 0.0)) for item in history)),
        "iteration_time_sec": float(sum(float(item.get("t_iter", 0.0)) for item in history)),
        "worst_true_relative_residual": float(
            linear_summary.get("worst_true_relative_residual", float("nan"))
        ),
        "one_time_setup_time_sec": float(timings.get("one_time_setup_time", timings.get("setup_time", 0.0))),
        "steady_state_setup_time_sec": float(timings.get("steady_state_setup_time", 0.0)),
        "solve_time_sec": float(timings.get("solve_time", 0.0)),
        "steady_state_total_time_sec": float(timings.get("steady_state_total_time", 0.0)),
        "end_to_end_total_time_sec": float(timings.get("total_time", 0.0)),
        "problem_build_time_sec": float(timings.get("problem_build_time", 0.0)),
        "assembler_setup_time_sec": float(timings.get("assembler_setup_time", 0.0)),
        "solver_bootstrap_time_sec": float(timings.get("solver_bootstrap_time", 0.0)),
        "energy_total_time_sec": _callback_value(callback, "energy", "total"),
        "gradient_total_time_sec": _callback_value(callback, "gradient", "total"),
        "hessian_total_time_sec": _callback_value(callback, "hessian", "total"),
        "linear_pc_setup_time_sec": _sum(linear, "pc_setup_time"),
        "linear_ksp_solve_time_sec": _sum(linear, "solve_time"),
        "linear_assemble_time_sec": _sum(linear, "assemble_total_time"),
        "energy": float(last_step.get("energy", float("nan"))),
        "omega": float(last_step.get("omega", float("nan"))),
        "u_max": float(last_step.get("u_max", float("nan"))),
    }


def _run_case(variant: dict[str, object]) -> dict[str, object]:
    case_dir = OUTPUT_ROOT / str(variant["name"])
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"

    if not result_path.exists():
        proc = subprocess.run(
            _command(variant, result_path),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            return {
                "variant": str(variant["name"]),
                "label": str(variant["label"]),
                "line_search": str(variant["name"]),
                "solver_success": False,
                "status": "subprocess_failed",
                "message": proc.stderr.strip().splitlines()[-1]
                if proc.stderr.strip()
                else "subprocess failed",
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "result_json": str(result_path),
            }

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    linear_solver = dict(payload.get("metadata", {}).get("linear_solver", {}))
    aggregated = _aggregate(payload)
    aggregated.update(
        {
            "variant": str(variant["name"]),
            "label": str(variant["label"]),
            "line_search": str(payload.get("case", {}).get("line_search", variant["name"])),
            "ksp_max_it": int(payload.get("case", {}).get("ksp_max_it", 0)),
            "accept_ksp_maxit_direction": bool(
                payload.get("case", {}).get("accept_ksp_maxit_direction", False)
            ),
            "ranks": int(RANKS),
            "mg_custom_hierarchy": str(linear_solver.get("mg_custom_hierarchy", CUSTOM_HIERARCHY)),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "result_json": str(result_path),
        }
    )
    return aggregated


def main() -> None:
    _ensure_assets()
    rows = _load_rows()
    completed = {str(row.get("variant", "")) for row in rows}
    for variant in VARIANTS:
        if str(variant["name"]) in completed:
            continue
        rows.append(_run_case(variant))
        _write_rows(rows)


if __name__ == "__main__":
    main()

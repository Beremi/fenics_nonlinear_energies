#!/usr/bin/env python3
"""Run L6 P2 deep-P1-tail PMG scaling cases at 1/2/4/8 MPI ranks with Newton maxit=20."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.problems.slope_stability.support import ensure_same_mesh_case_hdf5


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SOLVER = REPO_ROOT / "src" / "problems" / "slope_stability" / "jax_petsc" / "solve_slope_stability_dof.py"
OUTPUT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "slope_stability_l6_p2_deep_p1_tail_scaling_lambda1_maxit20"
)
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

RANKS = [1, 2, 4, 8]
CUSTOM_HIERARCHY = "1:1,2:1,3:1,4:1,5:1,6:1,6:2"
COMMON_ARGS = [
    "--level",
    "6",
    "--elem_degree",
    "2",
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


def _load_rows() -> list[dict[str, object]]:
    if SUMMARY_PATH.exists():
        return list(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    return []


def _write_rows(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _command(ranks: int, out: Path) -> list[str]:
    return [
        "mpiexec",
        "-n",
        str(ranks),
        str(PYTHON),
        "-u",
        str(SOLVER),
        *COMMON_ARGS,
        "--out",
        str(out),
    ]


def _run_case(ranks: int) -> dict[str, object]:
    case_dir = OUTPUT_ROOT / f"np{ranks}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"

    if not result_path.exists():
        proc = subprocess.run(
            _command(ranks, result_path),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            return {
                "ranks": int(ranks),
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
    result = dict(payload.get("result", {}))
    timings = dict(payload.get("timings", {}))
    steps = list(result.get("steps", []))
    last_step = steps[-1] if steps else {}
    linear_summary = dict(last_step.get("linear_summary", {}))
    setup_time = float(timings.get("one_time_setup_time", timings.get("setup_time", 0.0)))
    solve_time = float(timings.get("solve_time", 0.0))
    total_time = float(timings.get("total_time", 0.0))
    return {
        "ranks": int(ranks),
        "solver_success": bool(result.get("solver_success", False)),
        "status": str(result.get("status", "")),
        "message": str(last_step.get("message", "")),
        "level": int(payload["mesh"]["level"]),
        "h": float(payload["mesh"]["h"]),
        "nodes": int(payload["mesh"]["nodes"]),
        "elements": int(payload["mesh"]["elements"]),
        "free_dofs": int(payload["mesh"]["free_dofs"]),
        "mg_custom_hierarchy": str(payload["metadata"]["linear_solver"].get("mg_custom_hierarchy", CUSTOM_HIERARCHY)),
        "setup_time_sec": setup_time,
        "assembler_setup_time_sec": float(timings.get("assembler_setup_time", timings.get("setup_time", 0.0))),
        "problem_build_time_sec": float(timings.get("problem_build_time", 0.0)),
        "solver_bootstrap_time_sec": float(timings.get("solver_bootstrap_time", 0.0)),
        "finalize_time_sec": float(
            timings.get("finalize_time", max(0.0, total_time - setup_time - solve_time))
        ),
        "solve_time_sec": solve_time,
        "steady_state_total_time_sec": float(timings.get("steady_state_total_time", 0.0)),
        "total_time_sec": total_time,
        "outside_solve_time_sec": max(0.0, total_time - solve_time),
        "outside_setup_solve_time_sec": max(0.0, total_time - setup_time - solve_time),
        "newton_iterations": int(last_step.get("nit", len(steps))),
        "linear_iterations": int(last_step.get("linear_iters", 0)),
        "energy": float(last_step.get("energy", float("nan"))),
        "omega": float(last_step.get("omega", float("nan"))),
        "u_max": float(last_step.get("u_max", float("nan"))),
        "worst_true_relative_residual": float(
            linear_summary.get("worst_true_relative_residual", float("nan"))
        ),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "result_json": str(result_path),
    }


def main() -> None:
    for level in range(1, 7):
        for degree in (1, 2, 4):
            ensure_same_mesh_case_hdf5(level, degree)

    rows = _load_rows()
    completed = {int(row["ranks"]) for row in rows}
    for ranks in RANKS:
        if ranks in completed:
            continue
        rows.append(_run_case(ranks))
        _write_rows(rows)


if __name__ == "__main__":
    main()

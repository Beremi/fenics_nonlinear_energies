#!/usr/bin/env python3
"""Run L4 P4 instrumented PMG scaling cases at 1/2/4/8 MPI ranks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


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
    / "slope_stability_l4_p4_pmg_part_scaling_lambda1"
)
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

RANKS = [1, 2, 4, 8]
COMMON_ARGS = [
    "--level",
    "4",
    "--elem_degree",
    "4",
    "--lambda-target",
    "1.0",
    "--profile",
    "performance",
    "--pc_type",
    "mg",
    "--mg_variant",
    "explicit_pmg",
    "--mg_strategy",
    "same_mesh_p4_p2_p1_lminus1_p1",
    "--mg_lower_operator_policy",
    "galerkin_refresh",
    "--ksp_type",
    "fgmres",
    "--ksp_rtol",
    "1e-2",
    "--ksp_max_it",
    "100",
    "--mg_fine_ksp_type",
    "richardson",
    "--mg_fine_pc_type",
    "sor",
    "--mg_fine_steps",
    "3",
    "--mg_intermediate_steps",
    "3",
    "--mg_degree2_pc_type",
    "sor",
    "--mg_degree1_pc_type",
    "sor",
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
    "--distribution_strategy",
    "overlap_p2p",
    "--benchmark_mode",
    "warmup_once_then_solve",
    "--save-linear-timing",
    "--no-use_trust_region",
    "--quiet",
    "--out",
]


def _load_rows() -> list[dict[str, object]]:
    if SUMMARY_PATH.exists():
        return list(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    return []


def _write_rows(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _run_case(ranks: int) -> dict[str, object]:
    case_dir = OUTPUT_ROOT / f"np{ranks}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"

    if not result_path.exists():
        proc = subprocess.run(
            [
                "mpiexec",
                "-n",
                str(ranks),
                str(PYTHON),
                "-u",
                str(SOLVER),
                *COMMON_ARGS,
                str(result_path),
            ],
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
    step = payload["result"]["steps"][0]
    linear_summary = dict(step.get("linear_summary", {}))
    return {
        "ranks": int(ranks),
        "solver_success": bool(payload["result"]["solver_success"]),
        "status": str(payload["result"]["status"]),
        "message": str(step.get("message", "")),
        "newton_iterations": int(step["nit"]),
        "linear_iterations": int(step["linear_iters"]),
        "solve_time_sec": float(payload["timings"]["solve_time"]),
        "steady_state_total_time_sec": float(payload["timings"]["steady_state_total_time"]),
        "end_to_end_total_time_sec": float(payload["timings"]["total_time"]),
        "worst_true_relative_residual": float(
            linear_summary.get("worst_true_relative_residual", float("nan"))
        ),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "result_json": str(result_path),
    }


def main() -> None:
    rows = _load_rows()
    completed = {int(row["ranks"]) for row in rows}
    for ranks in RANKS:
        if ranks in completed:
            continue
        rows.append(_run_case(ranks))
        _write_rows(rows)


if __name__ == "__main__":
    main()

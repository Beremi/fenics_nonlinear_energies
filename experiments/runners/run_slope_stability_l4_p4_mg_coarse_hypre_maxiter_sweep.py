#!/usr/bin/env python3
"""Sweep MG coarse-HYPRE BoomerAMG max-iter on the accepted L4 P4 PMG path."""

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
    / "slope_stability_l4_p4_mg_coarse_hypre_maxiter_sweep_lambda1"
)
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

LEVEL = 4
DEGREE = 4
RANKS = [1, 2, 4, 8]
MAX_ITERS = [1, 2, 3, 4, 6]

COMMON_ARGS = [
    "--level",
    str(LEVEL),
    "--elem_degree",
    str(DEGREE),
    "--lambda-target",
    "1.0",
    "--profile",
    "performance",
    "--pc_type",
    "mg",
    "--mg_strategy",
    "same_mesh_p4_p2_p1_lminus1_p1",
    "--mg_variant",
    "legacy_pmg",
    "--ksp_type",
    "fgmres",
    "--ksp_rtol",
    "1e-2",
    "--ksp_max_it",
    "100",
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
    "--mg_coarse_hypre_tol",
    "0.0",
    "--mg_coarse_hypre_relax_type_all",
    "symmetric-SOR/Jacobi",
    "--line_search",
    "golden_fixed",
]


def _run_case(ranks: int, coarse_max_iter: int, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.json"
    stdout_path = out_dir / "stdout.txt"
    stderr_path = out_dir / "stderr.txt"
    if out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        return _row_from_payload(payload, ranks=ranks, coarse_max_iter=coarse_max_iter, out_dir=out_dir, returncode=0)
    command = [
        "mpiexec",
        "-n",
        str(ranks),
        str(PYTHON),
        "-u",
        str(SOLVER),
        *COMMON_ARGS,
        "--mg_coarse_hypre_max_iter",
        str(coarse_max_iter),
        "--out",
        str(out_path),
    ]
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    payload = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else None
    row: dict[str, object] = {
        "ranks": int(ranks),
        "mg_coarse_hypre_max_iter": int(coarse_max_iter),
        "command": command,
        "returncode": int(proc.returncode),
        "stdout_path": str(stdout_path.relative_to(REPO_ROOT)),
        "stderr_path": str(stderr_path.relative_to(REPO_ROOT)),
    }
    if payload is None:
        row["solver_success"] = False
        row["error"] = "missing_result_json"
        return row
    return _row_from_payload(payload, ranks=ranks, coarse_max_iter=coarse_max_iter, out_dir=out_dir, returncode=proc.returncode)


def _row_from_payload(
    payload: dict[str, object],
    *,
    ranks: int,
    coarse_max_iter: int,
    out_dir: Path,
    returncode: int,
) -> dict[str, object]:
    result = payload["result"]
    step = result["steps"][0]
    linear = payload["metadata"]["linear_solver"]
    timings = payload["timings"]
    bootstrap = timings.get("solver_bootstrap_breakdown", {})
    callback = timings.get("callback_summary", {})
    history = step.get("history", [])
    linear_timing = step.get("linear_timing", [])
    return {
        "ranks": int(ranks),
        "mg_coarse_hypre_max_iter": int(coarse_max_iter),
        "returncode": int(returncode),
        "solver_success": bool(result["solver_success"]),
        "result_path": str((out_dir / "result.json").relative_to(REPO_ROOT)),
        "stdout_path": str((out_dir / "stdout.txt").relative_to(REPO_ROOT)),
        "stderr_path": str((out_dir / "stderr.txt").relative_to(REPO_ROOT)),
        "variant": f"coarse_maxit_{coarse_max_iter}",
        "steady_state_total_time_sec": float(result["steady_state_total_time"]),
        "end_to_end_total_time_sec": float(result["total_time"]),
        "one_time_setup_time_sec": float(result["one_time_setup_time"]),
        "solve_time_sec": float(result["solve_time_total"]),
        "problem_build_time_sec": float(timings.get("problem_build_time", 0.0)),
        "assembler_setup_time_sec": float(timings.get("assembler_setup_time", 0.0)),
        "assembler_warmup_time_sec": float(timings.get("assembler_setup_breakdown", {}).get("warmup", 0.0)),
        "mg_hierarchy_build_time_sec": float(bootstrap.get("mg_hierarchy_build_time", 0.0)),
        "mg_transfer_build_time_sec": float(bootstrap.get("mg_transfer_build_time", 0.0)),
        "mg_configure_time_sec": float(bootstrap.get("mg_configure_time", 0.0)),
        "energy_total_time_sec": float(callback.get("energy", {}).get("total", 0.0)),
        "gradient_total_time_sec": float(callback.get("gradient", {}).get("total", 0.0)),
        "hessian_total_time_sec": float(callback.get("hessian", {}).get("total", 0.0)),
        "line_search_time_sec": float(sum(float(entry.get("t_ls", 0.0)) for entry in history)),
        "linear_pc_setup_time_sec": float(sum(float(entry.get("pc_setup_time", 0.0)) for entry in linear_timing)),
        "linear_ksp_solve_time_sec": float(sum(float(entry.get("solve_time", 0.0)) for entry in linear_timing)),
        "linear_assemble_time_sec": float(sum(float(entry.get("assemble_total_time", 0.0)) for entry in linear_timing)),
        "newton_iterations": int(step["nit"]),
        "linear_iterations": int(step["linear_iters"]),
        "line_search_evals": int(sum(int(entry.get("ls_evals", 0)) for entry in history)),
        "worst_true_relative_residual": float(
            step.get("linear_summary", {}).get("worst_true_relative_residual", 0.0)
        ),
        "mg_coarse_backend": str(linear["mg_coarse_backend"]),
        "mg_coarse_pc_type": str(linear["mg_coarse_pc_type"]),
        "mg_coarse_hypre_nodal_coarsen": int(linear["mg_coarse_hypre_nodal_coarsen"]),
        "mg_coarse_hypre_vec_interp_variant": int(linear["mg_coarse_hypre_vec_interp_variant"]),
        "mg_coarse_hypre_strong_threshold": linear["mg_coarse_hypre_strong_threshold"],
        "mg_coarse_hypre_coarsen_type": str(linear["mg_coarse_hypre_coarsen_type"]),
        "mg_coarse_hypre_tol": float(linear["mg_coarse_hypre_tol"]),
        "mg_coarse_hypre_relax_type_all": str(linear["mg_coarse_hypre_relax_type_all"]),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for degree in (1, 2, 4):
        ensure_same_mesh_case_hdf5(LEVEL, degree)

    rows: list[dict[str, object]] = []
    for coarse_max_iter in MAX_ITERS:
        for ranks in RANKS:
            out_dir = OUTPUT_ROOT / f"maxit_{coarse_max_iter}" / f"np{ranks}"
            rows.append(_run_case(ranks, coarse_max_iter, out_dir))
            SUMMARY_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    SUMMARY_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(SUMMARY_PATH), "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

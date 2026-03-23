#!/usr/bin/env python3
"""Run L5 P4 PMG scaling cases at 8/16/32 MPI ranks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SOLVER = REPO_ROOT / "src" / "problems" / "slope_stability" / "jax_petsc" / "solve_slope_stability_dof.py"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "raw_results" / "slope_stability_l5_p4_pmg_scaling_lambda1"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

RANKS = [8, 16, 32]
COMMON_ARGS = [
    "--level",
    "5",
    "--elem_degree",
    "4",
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
    "--save-linear-timing",
    "--quiet",
    "--no-use_trust_region",
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
                "message": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "subprocess failed",
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "result_json": str(result_path),
            }

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    step = payload["result"]["steps"][0]
    linear_summary = dict(step.get("linear_summary", {}))
    timings = dict(payload.get("timings", {}))
    setup_time = float(timings.get("one_time_setup_time", timings["setup_time"]))
    solve_time = float(timings["solve_time"])
    total_time = float(timings["total_time"])
    return {
        "ranks": int(ranks),
        "solver_success": bool(payload["result"]["solver_success"]),
        "status": str(payload["result"]["status"]),
        "message": str(step.get("message", "")),
        "level": int(payload["mesh"]["level"]),
        "h": float(payload["mesh"]["h"]),
        "nodes": int(payload["mesh"]["nodes"]),
        "elements": int(payload["mesh"]["elements"]),
        "free_dofs": int(payload["mesh"]["free_dofs"]),
        "setup_time_sec": setup_time,
        "assembler_setup_time_sec": float(timings.get("assembler_setup_time", timings["setup_time"])),
        "problem_build_time_sec": float(timings.get("problem_build_time", 0.0)),
        "solver_bootstrap_time_sec": float(timings.get("solver_bootstrap_time", 0.0)),
        "finalize_time_sec": float(timings.get("finalize_time", max(0.0, total_time - setup_time - solve_time))),
        "solve_time_sec": solve_time,
        "total_time_sec": total_time,
        "outside_solve_time_sec": max(0.0, total_time - solve_time),
        "outside_setup_solve_time_sec": max(0.0, total_time - setup_time - solve_time),
        "newton_iterations": int(step["nit"]),
        "linear_iterations": int(step["linear_iters"]),
        "energy": float(step["energy"]),
        "omega": float(step["omega"]),
        "u_max": float(step["u_max"]),
        "worst_true_relative_residual": float(linear_summary.get("worst_true_relative_residual", float("nan"))),
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

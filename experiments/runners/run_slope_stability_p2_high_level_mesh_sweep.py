#!/usr/bin/env python3
"""Run a higher-level mesh sweep for P2 slope stability at fixed MPI size."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SOLVER = REPO_ROOT / "src" / "problems" / "slope_stability" / "jax_petsc" / "solve_slope_stability_dof.py"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "raw_results" / "slope_stability_p2_high_level_mesh_sweep_lambda1_np16"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

LEVELS = [4, 5, 6]
RANKS = 16
LAMBDA_TARGET = 1.0
METHODS = {
    "p2_hypre_boomeramg": {
        "pc_type": "hypre",
        "description": "P2 Hypre BoomerAMG",
    },
    "p2_same_mesh_p1": {
        "pc_type": "mg",
        "mg_strategy": "same_mesh_p2_p1",
        "description": "P2 -> same-mesh P1",
    },
    "p2_same_mesh_p1_lminus1": {
        "pc_type": "mg",
        "mg_strategy": "same_mesh_p2_p1_lminus1_p1",
        "description": "P2 -> same-mesh P1 -> level-1 P1",
    },
}


def _load_rows() -> list[dict[str, object]]:
    if SUMMARY_PATH.exists():
        return list(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    return []


def _write_rows(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _command(*, level: int, method: dict[str, object], out: Path) -> list[str]:
    command = [
        "mpiexec",
        "-n",
        str(RANKS),
        str(PYTHON),
        "-u",
        str(SOLVER),
        "--level",
        str(level),
        "--elem_degree",
        "2",
        "--lambda-target",
        str(LAMBDA_TARGET),
        "--profile",
        "performance",
        "--pc_type",
        str(method["pc_type"]),
        "--ksp_type",
        "fgmres",
        "--ksp_rtol",
        "1e-2",
        "--ksp_max_it",
        "100",
        "--no-use_trust_region",
        "--quiet",
        "--out",
        str(out),
    ]
    mg_strategy = method.get("mg_strategy")
    if mg_strategy is not None:
        command.extend(["--mg_strategy", str(mg_strategy)])
    return command


def _run_case(level: int, method_name: str, method: dict[str, object]) -> dict[str, object]:
    case_dir = OUTPUT_ROOT / method_name / f"level{level}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"

    if not result_path.exists():
        proc = subprocess.run(
            _command(level=level, method=method, out=result_path),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            return {
                "level": int(level),
                "ranks": int(RANKS),
                "method": method_name,
                "description": str(method["description"]),
                "pc_type": str(method["pc_type"]),
                "mg_strategy": None if method.get("mg_strategy") is None else str(method["mg_strategy"]),
                "solver_success": False,
                "status": "subprocess_failed",
                "message": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "subprocess failed",
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "result_json": str(result_path),
            }

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    step = payload["result"]["steps"][0]
    return {
        "level": int(level),
        "h": float(payload["mesh"]["h"]),
        "nodes": int(payload["mesh"]["nodes"]),
        "elements": int(payload["mesh"]["elements"]),
        "free_dofs": int(payload["mesh"]["free_dofs"]),
        "ranks": int(RANKS),
        "method": method_name,
        "description": str(method["description"]),
        "pc_type": str(method["pc_type"]),
        "mg_strategy": None if method.get("mg_strategy") is None else str(method["mg_strategy"]),
        "solver_success": bool(payload["result"]["solver_success"]),
        "status": str(payload["result"]["status"]),
        "message": str(step["message"]),
        "setup_time_sec": float(payload["timings"]["setup_time"]),
        "solve_time_sec": float(payload["timings"]["solve_time"]),
        "total_time_sec": float(payload["timings"]["total_time"]),
        "newton_iterations": int(step["nit"]),
        "linear_iterations": int(step["linear_iters"]),
        "energy": float(step["energy"]),
        "omega": float(step["omega"]),
        "u_max": float(step["u_max"]),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "result_json": str(result_path),
    }


def main() -> None:
    rows = _load_rows()
    completed = {(int(row["level"]), str(row["method"])) for row in rows}
    for level in LEVELS:
        for method_name, method in METHODS.items():
            if (level, method_name) in completed:
                continue
            rows.append(_run_case(level, method_name, method))
            _write_rows(rows)


if __name__ == "__main__":
    main()

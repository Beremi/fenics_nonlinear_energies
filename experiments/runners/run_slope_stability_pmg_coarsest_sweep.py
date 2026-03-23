#!/usr/bin/env python3
"""Run an L5 PETSc multigrid coarsest-level/process sweep."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SOLVER = REPO_ROOT / "src" / "problems" / "slope_stability" / "jax_petsc" / "solve_slope_stability_dof.py"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "raw_results" / "slope_stability_pmg_coarsest_sweep_lambda1"

RANKS = [1, 2, 4, 8, 16, 32]
COARSEST_LEVELS = [1, 2, 3, 4]


def _run_case(*, ranks: int, coarsest_level: int) -> dict[str, object]:
    case_dir = OUTPUT_ROOT / f"c{coarsest_level}_np{ranks}"
    case_dir.mkdir(parents=True, exist_ok=True)
    output_json = case_dir / "result.json"
    command = [
        "mpiexec",
        "-n",
        str(ranks),
        str(PYTHON),
        "-u",
        str(SOLVER),
        "--level",
        "5",
        "--lambda-target",
        "1.0",
        "--profile",
        "performance",
        "--pc_type",
        "mg",
        "--ksp_type",
        "fgmres",
        "--ksp_rtol",
        "1e-2",
        "--ksp_max_it",
        "200",
        "--mg_coarsest_level",
        str(coarsest_level),
        "--no-use_trust_region",
        "--save-linear-timing",
        "--quiet",
        "--out",
        str(output_json),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    step = payload["result"]["steps"][0]
    return {
        "ranks": int(ranks),
        "coarsest_level": int(coarsest_level),
        "solver_success": bool(payload["result"]["solver_success"]),
        "status": str(payload["result"]["status"]),
        "message": str(step["message"]),
        "total_time_sec": float(payload["timings"]["total_time"]),
        "setup_time_sec": float(payload["timings"]["setup_time"]),
        "solve_time_sec": float(payload["timings"]["solve_time"]),
        "newton_iterations": int(step["nit"]),
        "linear_iterations": int(step["linear_iters"]),
        "energy": float(step["energy"]),
        "omega": float(step["omega"]),
        "u_max": float(step["u_max"]),
        "result_json": str(output_json),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "summary.json"
    rows: list[dict[str, object]] = []
    for coarsest_level in COARSEST_LEVELS:
        for ranks in RANKS:
            rows.append(_run_case(ranks=ranks, coarsest_level=coarsest_level))
            summary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

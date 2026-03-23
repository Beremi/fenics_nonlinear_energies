#!/usr/bin/env python3
"""Benchmark mixed-order PCMG strategies for slope stability."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SOLVER = REPO_ROOT / "src" / "problems" / "slope_stability" / "jax_petsc" / "solve_slope_stability_dof.py"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "raw_results" / "slope_stability_hp_strategy_bench_level4_lambda1_v2"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

LEVEL = 4
LAMBDA_TARGET = 1.0
RANKS = [1, 8, 16, 32]
STRATEGIES = {
    "p2_hypre_boomeramg": {
        "elem_degree": 2,
        "pc_type": "hypre",
        "description": "L4 P2 Hypre BoomerAMG",
    },
    "p2_same_mesh_p1": {
        "elem_degree": 2,
        "pc_type": "mg",
        "mg_strategy": "same_mesh_p2_p1",
        "description": "L4 P2 -> L4 P1",
    },
    "p2_same_mesh_p1_lminus1": {
        "elem_degree": 2,
        "pc_type": "mg",
        "mg_strategy": "same_mesh_p2_p1_lminus1_p1",
        "description": "L4 P2 -> L4 P1 -> L3 P1",
    },
    "p4_hypre_boomeramg": {
        "elem_degree": 4,
        "pc_type": "hypre",
        "description": "L4 P4 Hypre BoomerAMG",
    },
    "p4_same_mesh_p1": {
        "elem_degree": 4,
        "pc_type": "mg",
        "mg_strategy": "same_mesh_p4_p1",
        "description": "L4 P4 -> L4 P1",
    },
    "p4_same_mesh_p2_p1": {
        "elem_degree": 4,
        "pc_type": "mg",
        "mg_strategy": "same_mesh_p4_p2_p1",
        "description": "L4 P4 -> L4 P2 -> L4 P1",
    },
    "p4_same_mesh_p1_lminus1": {
        "elem_degree": 4,
        "pc_type": "mg",
        "mg_strategy": "same_mesh_p4_p1_lminus1_p1",
        "description": "L4 P4 -> L4 P1 -> L3 P1",
    },
    "p4_same_mesh_p2_p1_lminus1": {
        "elem_degree": 4,
        "pc_type": "mg",
        "mg_strategy": "same_mesh_p4_p2_p1_lminus1_p1",
        "description": "L4 P4 -> L4 P2 -> L4 P1 -> L3 P1",
    },
}


def _load_rows() -> list[dict[str, object]]:
    if SUMMARY_PATH.exists():
        return list(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    return []


def _write_rows(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _command(*, ranks: int, elem_degree: int, pc_type: str, mg_strategy: str | None, out: Path) -> list[str]:
    command = [
        "mpiexec",
        "-n",
        str(ranks),
        str(PYTHON),
        "-u",
        str(SOLVER),
        "--level",
        str(LEVEL),
        "--elem_degree",
        str(elem_degree),
        "--lambda-target",
        str(LAMBDA_TARGET),
        "--profile",
        "performance",
        "--pc_type",
        str(pc_type),
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
    if mg_strategy:
        command.extend(["--mg_strategy", str(mg_strategy)])
    return command


def _run_case(name: str, config: dict[str, object], ranks: int) -> dict[str, object]:
    case_dir = OUTPUT_ROOT / name / f"np{ranks}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        proc = subprocess.run(
            _command(
                ranks=ranks,
                elem_degree=int(config["elem_degree"]),
                pc_type=str(config.get("pc_type", "mg")),
                mg_strategy=None if "mg_strategy" not in config else str(config["mg_strategy"]),
                out=result_path,
            ),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            return {
                "strategy": name,
                "description": str(config["description"]),
                "elem_degree": int(config["elem_degree"]),
                "mg_strategy": str(config["mg_strategy"]),
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
    return {
        "strategy": name,
        "description": str(config["description"]),
        "elem_degree": int(config["elem_degree"]),
        "pc_type": str(config.get("pc_type", "mg")),
        "mg_strategy": None if "mg_strategy" not in config else str(config["mg_strategy"]),
        "ranks": int(ranks),
        "solver_success": bool(payload["result"]["solver_success"]),
        "status": str(payload["result"]["status"]),
        "message": str(step["message"]),
        "nodes": int(payload["mesh"]["nodes"]),
        "elements": int(payload["mesh"]["elements"]),
        "free_dofs": int(payload["mesh"]["free_dofs"]),
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
    completed = {(row["strategy"], int(row["ranks"])) for row in rows}
    for name, config in STRATEGIES.items():
        for ranks in RANKS:
            if (name, ranks) in completed:
                continue
            rows.append(_run_case(name, config, ranks))
            _write_rows(rows)


if __name__ == "__main__":
    main()

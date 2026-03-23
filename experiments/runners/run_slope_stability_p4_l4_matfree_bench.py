#!/usr/bin/env python3
"""Benchmark matrix-free P4 L4 slope-stability operator variants."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SOLVER = REPO_ROOT / "src" / "problems" / "slope_stability" / "jax_petsc" / "solve_slope_stability_dof.py"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "raw_results" / "slope_stability_p4_l4_matfree_bench"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

LEVEL = 4
LAMBDA_TARGET = 1.0
RANKS = [1, 8, 16, 32]
BASE_ARGS = [
    "--level",
    str(LEVEL),
    "--elem_degree",
    "4",
    "--lambda-target",
    str(LAMBDA_TARGET),
    "--profile",
    "performance",
    "--ksp_type",
    "fgmres",
    "--ksp_rtol",
    "1e-2",
    "--ksp_max_it",
    "100",
    "--no-use_trust_region",
    "--save-linear-timing",
    "--quiet",
]

STRATEGIES = {
    "assembled_hypre": {
        "description": "Assembled P4 fine operator + Hypre BoomerAMG",
        "pc_type": "hypre",
        "operator_mode": "assembled",
        "ranks": list(RANKS),
    },
    "assembled_mg_best": {
        "description": "Assembled P4 fine operator + best same-mesh PCMG",
        "pc_type": "mg",
        "operator_mode": "assembled",
        "mg_strategy": "same_mesh_p4_p2_p1_lminus1_p1",
        "ranks": list(RANKS),
    },
    "matfree_element_hypre": {
        "description": "Matrix-free element-HVP fine operator + Hypre BoomerAMG",
        "pc_type": "hypre",
        "operator_mode": "matfree_element",
        "ranks": list(RANKS),
    },
    "matfree_overlap_hypre": {
        "description": "Matrix-free overlap-functional fine operator + Hypre BoomerAMG",
        "pc_type": "hypre",
        "operator_mode": "matfree_overlap",
        "ranks": list(RANKS),
    },
    "matfree_element_mg_direct": {
        "description": "Direct shell fine operator + PCMG attempt",
        "pc_type": "mg",
        "operator_mode": "matfree_element",
        "mg_strategy": "same_mesh_p4_p2_p1_lminus1_p1",
        "ranks": [1],
    },
    "matfree_overlap_mg_direct": {
        "description": "Direct overlap shell fine operator + PCMG attempt",
        "pc_type": "mg",
        "operator_mode": "matfree_overlap",
        "mg_strategy": "same_mesh_p4_p2_p1_lminus1_p1",
        "ranks": [1],
    },
}


def _load_rows() -> list[dict[str, object]]:
    if SUMMARY_PATH.exists():
        return list(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    return []


def _write_rows(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _sum_linear_timing(step: dict[str, object], key: str) -> float:
    records = list(step.get("linear_timing", []))
    return float(sum(float(record.get(key, 0.0)) for record in records))


def _command(*, ranks: int, config: dict[str, object], out: Path) -> list[str]:
    command = [
        "mpiexec",
        "-n",
        str(ranks),
        str(PYTHON),
        "-u",
        str(SOLVER),
        *BASE_ARGS,
        "--pc_type",
        str(config["pc_type"]),
        "--operator_mode",
        str(config["operator_mode"]),
        "--out",
        str(out),
    ]
    mg_strategy = config.get("mg_strategy")
    if mg_strategy:
        command.extend(["--mg_strategy", str(mg_strategy)])
    return command


def _row_from_payload(
    *,
    name: str,
    config: dict[str, object],
    ranks: int,
    payload: dict[str, object],
    stdout_path: Path,
    stderr_path: Path,
    result_path: Path,
) -> dict[str, object]:
    step = payload["result"]["steps"][0]
    linear_records = list(step.get("linear_timing", []))
    operator_apply_calls = int(
        sum(int(record.get("operator_apply_calls", 0)) for record in linear_records)
    )
    operator_apply_total = _sum_linear_timing(step, "operator_apply_total_time")
    return {
        "strategy": name,
        "description": str(config["description"]),
        "ranks": int(ranks),
        "pc_type": str(config["pc_type"]),
        "operator_mode": str(config["operator_mode"]),
        "mg_strategy": None if "mg_strategy" not in config else str(config["mg_strategy"]),
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
        "operator_prepare_total_sec": _sum_linear_timing(step, "operator_prepare_total_time"),
        "operator_prepare_allgatherv_sec": _sum_linear_timing(
            step, "operator_prepare_allgatherv"
        ),
        "operator_prepare_build_v_local_sec": _sum_linear_timing(
            step, "operator_prepare_build_v_local"
        ),
        "operator_prepare_linearize_sec": _sum_linear_timing(
            step, "operator_prepare_linearize"
        ),
        "operator_apply_calls": int(operator_apply_calls),
        "operator_apply_total_sec": float(operator_apply_total),
        "operator_apply_allgatherv_sec": _sum_linear_timing(
            step, "operator_apply_allgatherv"
        ),
        "operator_apply_build_v_local_sec": _sum_linear_timing(
            step, "operator_apply_build_v_local"
        ),
        "operator_apply_kernel_sec": _sum_linear_timing(step, "operator_apply_kernel"),
        "operator_apply_scatter_sec": _sum_linear_timing(step, "operator_apply_scatter"),
        "operator_apply_avg_ms": (
            1.0e3 * operator_apply_total / operator_apply_calls
            if operator_apply_calls > 0
            else 0.0
        ),
        "pc_operator_assemble_total_sec": _sum_linear_timing(
            step, "pc_operator_assemble_total_time"
        ),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "result_json": str(result_path),
    }


def _run_case(name: str, config: dict[str, object], ranks: int) -> dict[str, object]:
    case_dir = OUTPUT_ROOT / name / f"np{ranks}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"

    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return _row_from_payload(
            name=name,
            config=config,
            ranks=ranks,
            payload=payload,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            result_path=result_path,
        )

    proc = subprocess.run(
        _command(ranks=ranks, config=config, out=result_path),
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
            "ranks": int(ranks),
            "pc_type": str(config["pc_type"]),
            "operator_mode": str(config["operator_mode"]),
            "mg_strategy": None if "mg_strategy" not in config else str(config["mg_strategy"]),
            "solver_success": False,
            "status": "subprocess_failed",
            "message": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "subprocess failed",
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "result_json": str(result_path),
        }
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return _row_from_payload(
        name=name,
        config=config,
        ranks=ranks,
        payload=payload,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        result_path=result_path,
    )


def main() -> None:
    rows = _load_rows()
    completed = {(str(row["strategy"]), int(row["ranks"])) for row in rows}
    for name, config in STRATEGIES.items():
        for ranks in list(config["ranks"]):
            if (name, int(ranks)) in completed:
                continue
            rows.append(_run_case(name, config, int(ranks)))
            _write_rows(rows)


if __name__ == "__main__":
    main()

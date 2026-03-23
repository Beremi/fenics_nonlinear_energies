#!/usr/bin/env python3
"""Benchmark PETSc preconditioner reuse for L5 same-mesh P4 multigrid."""

from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_OUT_DIR = (
    REPO_ROOT / "artifacts" / "raw_results" / "slope_stability_l5_p4_pmg_reuse_bench_lambda1"
)


COMMON_ARGS = [
    "--level",
    "5",
    "--elem_degree",
    "4",
    "--lambda-target",
    "1.0",
    "--profile",
    "performance",
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


def _run_case(out_root: Path, name: str, args: list[str]) -> dict[str, object]:
    case_dir = out_root / name
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"

    command = [
        str(PYTHON),
        "-u",
        str(SOLVER),
        *args,
        "--out",
        str(result_path),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    payload = None
    if completed.returncode == 0 and result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))

    return {
        "name": name,
        "args": list(args),
        "returncode": int(completed.returncode),
        "result_path": str(result_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "payload": payload,
    }


def _summarize_run(run: dict[str, object], *, description: str, hierarchy: str) -> dict[str, object]:
    payload = run.get("payload")
    row = {
        "name": str(run["name"]),
        "description": str(description),
        "hierarchy": str(hierarchy),
        "returncode": int(run["returncode"]),
        "result_json": str(run["result_path"]),
        "stdout_path": str(run["stdout_path"]),
        "stderr_path": str(run["stderr_path"]),
    }
    if payload is None:
        row.update(
            {
                "solver_success": False,
                "status": "error",
                "message": "solver process failed",
            }
        )
        return row

    step = payload["result"]["steps"][0]
    linear_summary = dict(step.get("linear_summary", {}))
    linear_records = list(step.get("linear_timing", []))
    row.update(
        {
            "solver_success": bool(payload["result"]["solver_success"]),
            "status": str(payload["result"]["status"]),
            "message": str(step.get("message", "")),
            "energy": float(step.get("energy", math.nan)),
            "omega": float(step.get("omega", math.nan)),
            "u_max": float(step.get("u_max", math.nan)),
            "newton_iterations": int(step.get("nit", 0)),
            "linear_iterations": int(step.get("linear_iters", 0)),
            "all_ksp_converged": bool(linear_summary.get("all_converged", False)),
            "n_linear_solves": int(linear_summary.get("n_solves", 0)),
            "worst_true_relative_residual": float(
                linear_summary.get("worst_true_relative_residual", math.nan)
            ),
            "last_ksp_reason_name": str(linear_summary.get("last_reason_name", "")),
            "operator_mode": str(payload["metadata"]["linear_solver"]["operator_mode"]),
            "pc_type": str(payload["metadata"]["linear_solver"]["pc_type"]),
            "pc_reuse_preconditioner": bool(
                payload["metadata"]["linear_solver"].get("pc_reuse_preconditioner", False)
            ),
            "mg_strategy": str(payload["metadata"]["linear_solver"]["mg_strategy"]),
            "mg_variant": str(payload["metadata"]["linear_solver"]["mg_variant"]),
            "solve_time_sec": float(payload["timings"]["solve_time"]),
            "setup_time_sec": float(payload["timings"]["setup_time"]),
            "total_time_sec": float(payload["timings"]["total_time"]),
            "operator_prepare_time_sec": float(
                sum(float(record.get("operator_prepare_total_time", 0.0)) for record in linear_records)
            ),
            "fine_operator_assembly_time_sec": float(
                sum(float(record.get("assemble_total_time", 0.0)) for record in linear_records)
            ),
            "fine_pmat_step_assembly_time_sec": float(
                sum(float(record.get("fine_pmat_step_assembly_time", 0.0)) for record in linear_records)
            ),
            "pc_setup_time_sec": float(
                sum(float(record.get("pc_setup_time", 0.0)) for record in linear_records)
            ),
            "ksp_solve_time_sec": float(
                sum(float(record.get("solve_time", 0.0)) for record in linear_records)
            ),
            "linear_total_time_sec": float(
                sum(float(record.get("linear_total_time", 0.0)) for record in linear_records)
            ),
            "fine_p4_operator_assembly_zero": bool(
                all(float(record.get("assemble_total_time", 0.0)) == 0.0 for record in linear_records)
            ),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cases = [
        (
            "pmg_same_mesh_current",
            [
                *COMMON_ARGS,
                "--pc_type",
                "mg",
                "--mg_strategy",
                "same_mesh_p4_p2_p1",
                "--mg_variant",
                "legacy_pmg",
            ],
            "Current legacy PCMG on L5 same-mesh P4->P2->P1",
            "same_mesh_p4_p2_p1",
        ),
        (
            "pmg_same_mesh_reused_pc",
            [
                *COMMON_ARGS,
                "--pc_type",
                "mg",
                "--mg_strategy",
                "same_mesh_p4_p2_p1",
                "--mg_variant",
                "legacy_pmg",
                "--pc_reuse_preconditioner",
            ],
            "Legacy PCMG with PETSc preconditioner reuse on L5 same-mesh P4->P2->P1",
            "same_mesh_p4_p2_p1",
        ),
        (
            "pmg_l4tail_current",
            [
                *COMMON_ARGS,
                "--pc_type",
                "mg",
                "--mg_strategy",
                "same_mesh_p4_p2_p1_lminus1_p1",
                "--mg_variant",
                "legacy_pmg",
            ],
            "Current legacy PCMG on L5 P4->P2->P1 with an extra L4 P1 tail level",
            "same_mesh_p4_p2_p1_lminus1_p1",
        ),
        (
            "pmg_l4tail_reused_pc",
            [
                *COMMON_ARGS,
                "--pc_type",
                "mg",
                "--mg_strategy",
                "same_mesh_p4_p2_p1_lminus1_p1",
                "--mg_variant",
                "legacy_pmg",
                "--pc_reuse_preconditioner",
            ],
            "Legacy PCMG with PETSc preconditioner reuse on L5 P4->P2->P1 plus L4 P1 tail",
            "same_mesh_p4_p2_p1_lminus1_p1",
        ),
        (
            "hypre_boomeramg",
            [
                *COMMON_ARGS,
                "--pc_type",
                "hypre",
            ],
            "Hypre BoomerAMG baseline on L5 P4",
            "hypre_boomeramg",
        ),
    ]

    rows = []
    for name, case_args, description, hierarchy in cases:
        run = _run_case(out_root, name, case_args)
        rows.append(_summarize_run(run, description=description, hierarchy=hierarchy))

    summary = {
        "runner": "slope_stability_l5_p4_pmg_reuse_bench",
        "lambda_target": 1.0,
        "rows": rows,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

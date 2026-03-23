#!/usr/bin/env python3
"""Benchmark staggered fine-Pmat variants on L2 same-mesh P4->P2->P1."""

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
    REPO_ROOT / "artifacts" / "raw_results" / "slope_stability_l2_p4_staggered_pmat_bench_lambda1"
)


COMMON_ARGS = [
    "--level",
    "2",
    "--elem_degree",
    "4",
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
    "100",
    "--mg_strategy",
    "same_mesh_p4_p2_p1",
    "--fine_pmat_stagger_period",
    "2",
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
        "mpiexec",
        "-n",
        "1",
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


def _summarize_run(run: dict[str, object], *, description: str) -> dict[str, object]:
    payload = run.get("payload")
    row = {
        "name": str(run["name"]),
        "description": str(description),
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
    first_record = linear_records[0] if linear_records else {}
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
            "first_ksp_reason_name": str(first_record.get("ksp_reason_name", "")),
            "operator_mode": str(payload["metadata"]["linear_solver"]["operator_mode"]),
            "pc_type": str(payload["metadata"]["linear_solver"]["pc_type"]),
            "mg_variant": str(payload["metadata"]["linear_solver"]["mg_variant"]),
            "fine_pmat_policy": str(
                payload["metadata"]["linear_solver"].get("fine_pmat_policy", "same_operator")
            ),
            "fine_pmat_stagger_period": int(
                payload["metadata"]["linear_solver"].get("fine_pmat_stagger_period", 0)
            ),
            "solve_time_sec": float(payload["timings"]["solve_time"]),
            "setup_time_sec": float(payload["timings"]["setup_time"]),
            "total_time_sec": float(payload["timings"]["total_time"]),
            "operator_prepare_time_sec": float(
                sum(float(record.get("operator_prepare_total_time", 0.0)) for record in linear_records)
            ),
            "operator_apply_time_sec": float(
                sum(float(record.get("operator_apply_total_time", 0.0)) for record in linear_records)
            ),
            "ksp_solve_time_sec": float(
                sum(float(record.get("solve_time", 0.0)) for record in linear_records)
            ),
            "fine_pmat_setup_assembly_time_sec": float(
                payload["timings"].get("fine_pmat_setup_assembly_time", 0.0)
            ),
            "fine_pmat_step_assembly_time_sec": float(
                sum(float(record.get("fine_pmat_step_assembly_time", 0.0)) for record in linear_records)
            ),
            "fine_pmat_update_steps": int(
                sum(1 for record in linear_records if bool(record.get("fine_pmat_updated_this_step", False)))
            ),
            "fine_pmat_reuse_steps": int(
                sum(1 for record in linear_records if not bool(record.get("fine_pmat_updated_this_step", False)))
            ),
            "fine_p4_operator_assembly_zero": bool(
                all(float(record.get("assemble_total_time", 0.0)) == 0.0 for record in linear_records)
            ),
            "lower_level_assembly_time_sec": float(
                sum(
                    float(level.get("assembly_total_time", 0.0))
                    for record in linear_records
                    for level in record.get("pc_operator_mg_level_records", [])
                )
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
            "baseline_assembled_legacy_full",
            [
                *COMMON_ARGS,
                "--operator_mode",
                "assembled",
                "--mg_variant",
                "legacy_pmg",
            ],
            "Assembled legacy PCMG baseline",
        ),
        (
            "matfree_legacy_pmg_staggered_whole",
            [
                *COMMON_ARGS,
                "--operator_mode",
                "matfree_overlap",
                "--mg_variant",
                "legacy_pmg",
                "--fine_pmat_policy",
                "staggered_whole",
            ],
            "Matrix-free P4 operator with whole fine Pmat preconditioner updated every other Newton step",
        ),
        (
            "matfree_explicit_pmg_staggered_smoother_only_fixed",
            [
                *COMMON_ARGS,
                "--operator_mode",
                "matfree_overlap",
                "--mg_variant",
                "explicit_pmg",
                "--fine_pmat_policy",
                "staggered_smoother_only",
                "--mg_lower_operator_policy",
                "fixed_setup",
                "--mg_fine_ksp_type",
                "chebyshev",
                "--mg_fine_pc_type",
                "jacobi",
                "--mg_fine_steps",
                "4",
                "--mg_degree2_pc_type",
                "sor",
                "--mg_degree1_pc_type",
                "jacobi",
            ],
            "Matrix-free P4 operator with staggered fine smoother matrix and fixed lower P2/P1 levels",
        ),
        (
            "matfree_explicit_pmg_staggered_smoother_only_refresh_attempt",
            [
                *COMMON_ARGS,
                "--operator_mode",
                "matfree_overlap",
                "--mg_variant",
                "explicit_pmg",
                "--fine_pmat_policy",
                "staggered_smoother_only",
                "--mg_lower_operator_policy",
                "refresh_each_newton",
                "--mg_fine_ksp_type",
                "chebyshev",
                "--mg_fine_pc_type",
                "jacobi",
                "--mg_fine_steps",
                "4",
                "--mg_degree2_pc_type",
                "sor",
                "--mg_degree1_pc_type",
                "jacobi",
            ],
            "Attempted fine-smoother-only stagger with refreshed lower P2/P1 levels; currently fails during PETSc PC setup on the first reuse step",
        ),
    ]

    rows = []
    for name, case_args, description in cases:
        run = _run_case(out_root, name, case_args)
        rows.append(_summarize_run(run, description=description))

    summary = {
        "runner": "slope_stability_l2_p4_staggered_pmat_bench",
        "lambda_target": 1.0,
        "rows": rows,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

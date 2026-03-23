#!/usr/bin/env python3
"""Stage the matrix-free P4 MG recovery sweep and final L4 benchmark."""

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
    REPO_ROOT / "artifacts" / "raw_results" / "slope_stability_p4_l4_matfree_mg_recovery_lambda1"
)
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"


BASE_ARGS = [
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
    "--save-linear-timing",
    "--quiet",
    "--no-use_trust_region",
    "--operator_mode",
    "matfree_overlap",
    "--mg_strategy",
    "same_mesh_p4_p2_p1",
]

CASES = [
    {
        "stage": "stage0",
        "name": "assembled_default_control",
        "description": "Current assembled control MG on the first Newton step",
        "level": 2,
        "ranks": 1,
        "args": [
            "--operator_mode",
            "assembled",
            "--mg_variant",
            "legacy_pmg",
            "--maxit",
            "1",
            "--ksp_max_it",
            "50",
        ],
    },
    {
        "stage": "stage0",
        "name": "assembled_explicit_refresh_control",
        "description": "Assembled fine operator with the explicit same-mesh hierarchy",
        "level": 2,
        "ranks": 1,
        "args": [
            "--operator_mode",
            "assembled",
            "--mg_variant",
            "explicit_pmg",
            "--mg_lower_operator_policy",
            "refresh_each_newton",
            "--maxit",
            "1",
            "--ksp_max_it",
            "50",
        ],
    },
    {
        "stage": "stage0",
        "name": "matfree_explicit_refresh_control",
        "description": "Matrix-free fine operator with the explicit same-mesh hierarchy",
        "level": 2,
        "ranks": 1,
        "args": [
            "--mg_variant",
            "explicit_pmg",
            "--mg_lower_operator_policy",
            "refresh_each_newton",
            "--maxit",
            "1",
            "--ksp_max_it",
            "50",
        ],
    },
    {
        "stage": "stage1",
        "name": "explicit_pmg_richardson_refresh",
        "description": "Explicit PCMG, refresh lower levels, fine Richardson",
        "level": 2,
        "ranks": 1,
        "args": [
            "--mg_variant",
            "explicit_pmg",
            "--mg_lower_operator_policy",
            "refresh_each_newton",
            "--mg_fine_ksp_type",
            "richardson",
            "--ksp_max_it",
            "50",
        ],
    },
    {
        "stage": "stage1",
        "name": "explicit_pmg_gmres_refresh",
        "description": "Explicit PCMG, refresh lower levels, fine GMRES",
        "level": 2,
        "ranks": 1,
        "args": [
            "--mg_variant",
            "explicit_pmg",
            "--mg_lower_operator_policy",
            "refresh_each_newton",
            "--mg_fine_ksp_type",
            "gmres",
            "--ksp_max_it",
            "50",
        ],
    },
    {
        "stage": "stage1",
        "name": "explicit_pmg_gcr_refresh",
        "description": "Explicit PCMG, refresh lower levels, fine GCR",
        "level": 2,
        "ranks": 1,
        "args": [
            "--mg_variant",
            "explicit_pmg",
            "--mg_lower_operator_policy",
            "refresh_each_newton",
            "--mg_fine_ksp_type",
            "gcr",
            "--ksp_max_it",
            "50",
        ],
    },
    {
        "stage": "stage1",
        "name": "explicit_pmg_fgmres_refresh",
        "description": "Explicit PCMG, refresh lower levels, fine FGMRES",
        "level": 2,
        "ranks": 1,
        "args": [
            "--mg_variant",
            "explicit_pmg",
            "--mg_lower_operator_policy",
            "refresh_each_newton",
            "--mg_fine_ksp_type",
            "fgmres",
            "--ksp_max_it",
            "50",
        ],
    },
    {
        "stage": "stage1",
        "name": "outer_pcksp_gmres_refresh",
        "description": "Outer FGMRES with a KSP preconditioner and inner GMRES+PCMG",
        "level": 2,
        "ranks": 1,
        "args": [
            "--mg_variant",
            "outer_pcksp",
            "--mg_lower_operator_policy",
            "refresh_each_newton",
            "--outer_pcksp_inner_ksp_type",
            "gmres",
            "--ksp_max_it",
            "50",
        ],
    },
    {
        "stage": "stage1",
        "name": "outer_pcksp_gcr_refresh",
        "description": "Outer FGMRES with a KSP preconditioner and inner GCR+PCMG",
        "level": 2,
        "ranks": 1,
        "args": [
            "--mg_variant",
            "outer_pcksp",
            "--mg_lower_operator_policy",
            "refresh_each_newton",
            "--outer_pcksp_inner_ksp_type",
            "gcr",
            "--ksp_max_it",
            "50",
        ],
    },
    {
        "stage": "stage1",
        "name": "outer_pcksp_fgmres_refresh",
        "description": "Outer FGMRES with a KSP preconditioner and inner FGMRES+PCMG",
        "level": 2,
        "ranks": 1,
        "args": [
            "--mg_variant",
            "outer_pcksp",
            "--mg_lower_operator_policy",
            "refresh_each_newton",
            "--outer_pcksp_inner_ksp_type",
            "fgmres",
            "--ksp_max_it",
            "50",
        ],
    },
    {
        "stage": "stage1",
        "name": "explicit_pmg_fgmres_fixed_p2sor",
        "description": "Winning PETSc-only path: matrix-free P4, fixed lower P2/P1, fine FGMRES, P2 SOR",
        "level": 2,
        "ranks": 1,
        "args": [
            "--mg_variant",
            "explicit_pmg",
            "--mg_lower_operator_policy",
            "fixed_setup",
            "--mg_fine_ksp_type",
            "fgmres",
            "--mg_fine_steps",
            "4",
            "--mg_intermediate_steps",
            "4",
            "--mg_degree2_pc_type",
            "sor",
            "--mg_degree1_pc_type",
            "jacobi",
            "--ksp_max_it",
            "100",
        ],
    },
]

for ranks in (1, 8, 16, 32):
    CASES.append(
        {
            "stage": "stage3",
            "name": f"assembled_baseline_np{ranks}",
            "description": "Assembled same-hierarchy P4 baseline",
            "level": 4,
            "ranks": ranks,
            "args": [
                "--operator_mode",
                "assembled",
                "--mg_variant",
                "legacy_pmg",
                "--ksp_max_it",
                "100",
                "--tolg",
                "2e-3",
                "--tolg_rel",
                "2e-3",
                "--tolx_rel",
                "2e-3",
            ],
        }
    )
    CASES.append(
        {
            "stage": "stage3",
            "name": f"matfree_candidate_np{ranks}",
            "description": "Chosen no-fine-assembly candidate",
            "level": 4,
            "ranks": ranks,
            "args": [
                "--mg_variant",
                "explicit_pmg",
                "--mg_lower_operator_policy",
                "fixed_setup",
                "--mg_fine_ksp_type",
                "fgmres",
                "--mg_fine_steps",
                "4",
                "--mg_intermediate_steps",
                "4",
                "--mg_degree2_pc_type",
                "sor",
                "--mg_degree1_pc_type",
                "jacobi",
                "--ksp_max_it",
                "100",
                "--tolg",
                "2e-3",
                "--tolg_rel",
                "2e-3",
                "--tolx_rel",
                "2e-3",
            ],
        }
    )


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


def _command(case: dict[str, object], out: Path) -> list[str]:
    return [
        "mpiexec",
        "-n",
        str(case["ranks"]),
        str(PYTHON),
        "-u",
        str(SOLVER),
        "--level",
        str(case["level"]),
        *BASE_ARGS,
        *list(case["args"]),
        "--out",
        str(out),
    ]


def _row_from_payload(
    *,
    case: dict[str, object],
    payload: dict[str, object],
    stdout_path: Path,
    stderr_path: Path,
    result_path: Path,
) -> dict[str, object]:
    step = payload["result"]["steps"][0]
    linear_timing = list(step.get("linear_timing", []))
    linear_summary = dict(step.get("linear_summary", {}))
    return {
        "stage": str(case["stage"]),
        "name": str(case["name"]),
        "description": str(case["description"]),
        "level": int(case["level"]),
        "ranks": int(case["ranks"]),
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
        "all_ksp_converged": bool(linear_summary.get("all_converged", False)),
        "ksp_failures": int(linear_summary.get("n_failed", 0)),
        "worst_true_relative_residual": float(
            linear_summary.get("worst_true_relative_residual", 0.0)
        ),
        "reason_names": list(linear_summary.get("reason_names", [])),
        "operator_mode": str(payload["metadata"]["linear_solver"]["operator_mode"]),
        "mg_variant": str(payload["metadata"]["linear_solver"].get("mg_variant", "none")),
        "mg_lower_operator_policy": str(
            payload["metadata"]["linear_solver"].get("mg_lower_operator_policy", "default")
        ),
        "mg_operator_policy": str(
            payload["metadata"]["linear_solver"].get("mg_operator_policy", "default")
        ),
        "mg_fine_down": dict(payload["metadata"]["linear_solver"].get("mg_fine_down", {})),
        "mg_fine_up": dict(payload["metadata"]["linear_solver"].get("mg_fine_up", {})),
        "assembly_time_sec": _sum_linear_timing(step, "assemble_total_time"),
        "operator_prepare_time_sec": _sum_linear_timing(
            step, "operator_prepare_total_time"
        ),
        "operator_apply_time_sec": _sum_linear_timing(step, "operator_apply_total_time"),
        "lower_level_assembly_time_sec": _sum_linear_timing(
            step, "pc_operator_assemble_total_time"
        ),
        "lower_level_state_transfer_time_sec": _sum_linear_timing(
            step, "pc_operator_state_transfer_time"
        ),
        "pc_setup_time_sec": _sum_linear_timing(step, "pc_setup_time"),
        "ksp_solve_time_sec": _sum_linear_timing(step, "solve_time"),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "result_json": str(result_path),
    }


def _run_case(case: dict[str, object]) -> dict[str, object]:
    case_dir = OUTPUT_ROOT / str(case["stage"]) / str(case["name"])
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"

    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return _row_from_payload(
            case=case,
            payload=payload,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            result_path=result_path,
        )

    proc = subprocess.run(
        _command(case, result_path),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        return {
            "stage": str(case["stage"]),
            "name": str(case["name"]),
            "description": str(case["description"]),
            "level": int(case["level"]),
            "ranks": int(case["ranks"]),
            "solver_success": False,
            "status": "subprocess_failed",
            "message": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "subprocess failed",
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "result_json": str(result_path),
        }

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return _row_from_payload(
        case=case,
        payload=payload,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        result_path=result_path,
    )


def main() -> None:
    rows = _load_rows()
    completed = {(str(row["name"]), int(row["ranks"])) for row in rows}
    for case in CASES:
        key = (str(case["name"]), int(case["ranks"]))
        if key in completed:
            continue
        rows.append(_run_case(case))
        _write_rows(rows)


if __name__ == "__main__":
    main()

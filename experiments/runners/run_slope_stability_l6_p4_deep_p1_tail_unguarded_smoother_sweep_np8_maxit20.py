#!/usr/bin/env python3
"""Run the L6 P4 deep-tail top-smoother sweep with unguarded capped KSP directions."""

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
    / "slope_stability_l6_p4_deep_p1_tail_unguarded_smoother_sweep_lambda1_np8_maxit20"
)
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

RANKS = 8
CUSTOM_HIERARCHY = "1:1,2:1,3:1,4:1,5:1,6:1,6:2,6:4"
COMMON_ARGS = [
    "--level",
    "6",
    "--elem_degree",
    "4",
    "--lambda-target",
    "1.0",
    "--profile",
    "performance",
    "--pc_type",
    "mg",
    "--mg_strategy",
    "custom_mixed",
    "--mg_custom_hierarchy",
    CUSTOM_HIERARCHY,
    "--mg_variant",
    "legacy_pmg",
    "--ksp_type",
    "fgmres",
    "--ksp_rtol",
    "1e-2",
    "--ksp_max_it",
    "15",
    "--accept_ksp_maxit_direction",
    "--no-guard_ksp_maxit_direction",
    "--maxit",
    "20",
    "--benchmark_mode",
    "warmup_once_then_solve",
    "--save-history",
    "--save-linear-timing",
    "--quiet",
    "--no-use_trust_region",
    "--line_search",
    "armijo",
    "--distribution_strategy",
    "overlap_p2p",
    "--problem_build_mode",
    "root_bcast",
    "--mg_level_build_mode",
    "root_bcast",
    "--mg_transfer_build_mode",
    "owned_rows",
    "--mg_p2_smoother_ksp_type",
    "richardson",
    "--mg_p2_smoother_pc_type",
    "sor",
    "--mg_p2_smoother_steps",
    "3",
    "--mg_p1_smoother_ksp_type",
    "richardson",
    "--mg_p1_smoother_pc_type",
    "sor",
    "--mg_p1_smoother_steps",
    "3",
    "--mg_coarse_backend",
    "rank0_lu_broadcast",
]

VARIANTS = [
    {
        "name": "baseline_richardson_sor_3",
        "label": "Baseline: richardson+sor (3), unguarded",
        "args": [
            "--mg_p4_smoother_ksp_type",
            "richardson",
            "--mg_p4_smoother_pc_type",
            "sor",
            "--mg_p4_smoother_steps",
            "3",
        ],
    },
    {
        "name": "richardson_sor_2",
        "label": "richardson+sor (2)",
        "args": [
            "--mg_p4_smoother_ksp_type",
            "richardson",
            "--mg_p4_smoother_pc_type",
            "sor",
            "--mg_p4_smoother_steps",
            "2",
        ],
    },
    {
        "name": "richardson_jacobi_2",
        "label": "richardson+jacobi (2)",
        "args": [
            "--mg_p4_smoother_ksp_type",
            "richardson",
            "--mg_p4_smoother_pc_type",
            "jacobi",
            "--mg_p4_smoother_steps",
            "2",
        ],
    },
    {
        "name": "richardson_jacobi_3",
        "label": "richardson+jacobi (3)",
        "args": [
            "--mg_p4_smoother_ksp_type",
            "richardson",
            "--mg_p4_smoother_pc_type",
            "jacobi",
            "--mg_p4_smoother_steps",
            "3",
        ],
    },
    {
        "name": "richardson_jacobi_4",
        "label": "richardson+jacobi (4)",
        "args": [
            "--mg_p4_smoother_ksp_type",
            "richardson",
            "--mg_p4_smoother_pc_type",
            "jacobi",
            "--mg_p4_smoother_steps",
            "4",
        ],
    },
    {
        "name": "chebyshev_jacobi_2",
        "label": "chebyshev+jacobi (2)",
        "args": [
            "--mg_p4_smoother_ksp_type",
            "chebyshev",
            "--mg_p4_smoother_pc_type",
            "jacobi",
            "--mg_p4_smoother_steps",
            "2",
        ],
    },
    {
        "name": "chebyshev_jacobi_3",
        "label": "chebyshev+jacobi (3)",
        "args": [
            "--mg_p4_smoother_ksp_type",
            "chebyshev",
            "--mg_p4_smoother_pc_type",
            "jacobi",
            "--mg_p4_smoother_steps",
            "3",
        ],
    },
    {
        "name": "chebyshev_jacobi_4",
        "label": "chebyshev+jacobi (4)",
        "args": [
            "--mg_p4_smoother_ksp_type",
            "chebyshev",
            "--mg_p4_smoother_pc_type",
            "jacobi",
            "--mg_p4_smoother_steps",
            "4",
        ],
    },
    {
        "name": "richardson_asm_1",
        "label": "richardson+asm (1)",
        "args": [
            "--mg_p4_smoother_ksp_type",
            "richardson",
            "--mg_p4_smoother_pc_type",
            "asm",
            "--mg_p4_smoother_steps",
            "1",
        ],
    },
    {
        "name": "richardson_asm_2",
        "label": "richardson+asm (2)",
        "args": [
            "--mg_p4_smoother_ksp_type",
            "richardson",
            "--mg_p4_smoother_pc_type",
            "asm",
            "--mg_p4_smoother_steps",
            "2",
        ],
    },
    {
        "name": "fgmres_none_2",
        "label": "fgmres+none (2)",
        "args": [
            "--mg_p4_smoother_ksp_type",
            "fgmres",
            "--mg_p4_smoother_pc_type",
            "none",
            "--mg_p4_smoother_steps",
            "2",
        ],
    },
    {
        "name": "fgmres_none_3",
        "label": "fgmres+none (3)",
        "args": [
            "--mg_p4_smoother_ksp_type",
            "fgmres",
            "--mg_p4_smoother_pc_type",
            "none",
            "--mg_p4_smoother_steps",
            "3",
        ],
    },
]


def _ensure_assets() -> None:
    for level in range(1, 7):
        for degree in (1, 2, 4):
            ensure_same_mesh_case_hdf5(level, degree)


def _load_rows() -> list[dict[str, object]]:
    if SUMMARY_PATH.exists():
        return list(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    return []


def _write_rows(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _sum(items: list[dict[str, object]], key: str) -> float:
    return float(sum(float(item.get(key, 0.0)) for item in items))


def _command(variant: dict[str, object], out: Path) -> list[str]:
    return [
        "mpiexec",
        "-n",
        str(RANKS),
        str(PYTHON),
        "-u",
        str(SOLVER),
        *COMMON_ARGS,
        *list(variant["args"]),
        "--out",
        str(out),
    ]


def _aggregate(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload.get("result", {}))
    steps = list(result.get("steps", []))
    last_step = dict(steps[-1]) if steps else {}
    history = list(last_step.get("history", []))
    linear = list(last_step.get("linear_timing", []))
    timings = dict(payload.get("timings", {}))
    linear_summary = dict(last_step.get("linear_summary", {}))

    return {
        "solver_success": bool(result.get("solver_success", False)),
        "status": str(result.get("status", "")),
        "message": str(last_step.get("message", "")),
        "newton_iterations": int(last_step.get("nit", len(steps))),
        "linear_iterations": int(last_step.get("linear_iters", 0)),
        "line_search_evals": int(sum(int(item.get("ls_evals", 0)) for item in history)),
        "line_search_time_sec": float(sum(float(item.get("t_ls", 0.0)) for item in history)),
        "accepted_capped_step_count": int(
            last_step.get("accepted_capped_step_count", 0)
        ),
        "worst_true_relative_residual": float(
            linear_summary.get("worst_true_relative_residual", float("nan"))
        ),
        "steady_state_total_time_sec": float(timings.get("steady_state_total_time", 0.0)),
        "solve_time_sec": float(timings.get("solve_time", 0.0)),
        "one_time_setup_time_sec": float(
            timings.get("one_time_setup_time", timings.get("setup_time", 0.0))
        ),
        "linear_ksp_solve_time_sec": _sum(linear, "solve_time"),
        "linear_assemble_time_sec": _sum(linear, "assemble_total_time"),
        "energy": float(last_step.get("energy", float("nan"))),
        "final_grad_norm": float(
            last_step.get("final_grad_norm", result.get("final_grad_norm", float("nan")))
        ),
        "omega": float(last_step.get("omega", float("nan"))),
        "u_max": float(last_step.get("u_max", float("nan"))),
    }


def _run_case(variant: dict[str, object]) -> dict[str, object]:
    case_dir = OUTPUT_ROOT / str(variant["name"])
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"

    if not result_path.exists():
        proc = subprocess.run(
            _command(variant, result_path),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            return {
                "variant": str(variant["name"]),
                "label": str(variant["label"]),
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
    aggregated = _aggregate(payload)
    aggregated.update(
        {
            "variant": str(variant["name"]),
            "label": str(variant["label"]),
            "result_json": str(result_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
    )
    return aggregated


def main() -> None:
    _ensure_assets()
    rows = _load_rows()
    existing = {str(row.get("variant")): row for row in rows}
    for variant in VARIANTS:
        if str(variant["name"]) in existing:
            continue
        row = _run_case(variant)
        rows.append(row)
        existing[str(variant["name"])] = row
        _write_rows(rows)


if __name__ == "__main__":
    main()

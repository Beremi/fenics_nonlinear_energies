#!/usr/bin/env python3
"""Compare the optimized L6 P4 deep-tail baseline against a skipped-tail hierarchy."""

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
    / "slope_stability_l6_p4_skipped_tail_compare_lambda1_np8_maxit20"
)
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

RANKS = 8
SMOKE_RANKS = 2
BASELINE_HIERARCHY = "1:1,2:1,3:1,4:1,5:1,6:1,6:2,6:4"
SKIPPED_HIERARCHY = "1:1,6:1,6:2,6:4"
SMOKE_BASELINE_HIERARCHY = "1:1,2:1,3:1,3:2,3:4"
SMOKE_SKIPPED_HIERARCHY = "1:1,3:1,3:2,3:4"

COMMON_ARGS = [
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
    "--mg_variant",
    "legacy_pmg",
    "--ksp_type",
    "fgmres",
    "--ksp_rtol",
    "1e-2",
    "--ksp_max_it",
    "15",
    "--accept_ksp_maxit_direction",
    "--guard_ksp_maxit_direction",
    "--ksp_maxit_direction_true_rel_cap",
    "6e-2",
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
    "--mg_p4_smoother_ksp_type",
    "richardson",
    "--mg_p4_smoother_pc_type",
    "sor",
    "--mg_p4_smoother_steps",
    "3",
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
    "--no-reuse_hessian_value_buffers",
]

BENCHMARK_VARIANTS = [
    {
        "name": "baseline_full_p1_tail",
        "label": "Baseline full P1 tail",
        "hierarchy": BASELINE_HIERARCHY,
    },
    {
        "name": "candidate_skip_intermediate_p1",
        "label": "Candidate skip intermediate P1",
        "hierarchy": SKIPPED_HIERARCHY,
    },
]

SMOKE_VARIANTS = [
    {
        "name": "smoke_baseline_full_p1_tail",
        "label": "Smoke baseline full P1 tail",
        "hierarchy": SMOKE_BASELINE_HIERARCHY,
        "level": 3,
    },
    {
        "name": "smoke_skip_intermediate_p1",
        "label": "Smoke skip intermediate P1",
        "hierarchy": SMOKE_SKIPPED_HIERARCHY,
        "level": 3,
    },
]


def _ensure_assets() -> None:
    for level in range(1, 7):
        for degree in (1, 2, 4):
            ensure_same_mesh_case_hdf5(level, degree)


def _load_summary() -> dict[str, object]:
    if SUMMARY_PATH.exists():
        return dict(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    return {"smoke": [], "rows": []}


def _save_summary(smoke: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"smoke": smoke, "rows": rows}
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sum(items: list[dict[str, object]], key: str) -> float:
    return float(sum(float(item.get(key, 0.0)) for item in items))


def _command(
    *,
    ranks: int,
    level: int,
    hierarchy: str,
    maxit: int,
    out: Path,
) -> list[str]:
    return [
        "mpiexec",
        "-n",
        str(ranks),
        str(PYTHON),
        "-u",
        str(SOLVER),
        "--level",
        str(level),
        "--mg_custom_hierarchy",
        str(hierarchy),
        "--maxit",
        str(maxit),
        *COMMON_ARGS,
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
    linear_solver = dict(payload.get("metadata", {}).get("linear_solver", {}))
    return {
        "solver_success": bool(result.get("solver_success", False)),
        "status": str(result.get("status", "")),
        "message": str(last_step.get("message", "")),
        "newton_iterations": int(last_step.get("nit", len(steps))),
        "linear_iterations": int(last_step.get("linear_iters", 0)),
        "line_search_evals": int(sum(int(item.get("ls_evals", 0)) for item in history)),
        "line_search_time_sec": float(sum(float(item.get("t_ls", 0.0)) for item in history)),
        "accepted_capped_step_count": int(last_step.get("accepted_capped_step_count", 0)),
        "steady_state_total_time_sec": float(timings.get("steady_state_total_time", 0.0)),
        "solve_time_sec": float(timings.get("solve_time", 0.0)),
        "one_time_setup_time_sec": float(
            timings.get("one_time_setup_time", timings.get("setup_time", 0.0))
        ),
        "linear_pc_setup_time_sec": _sum(linear, "pc_setup_time"),
        "linear_ksp_solve_time_sec": _sum(linear, "solve_time"),
        "linear_assemble_time_sec": _sum(linear, "assemble_total_time"),
        "energy": float(last_step.get("energy", float("nan"))),
        "final_grad_norm": float(
            last_step.get("final_grad_norm", result.get("final_grad_norm", float("nan")))
        ),
        "omega": float(last_step.get("omega", float("nan"))),
        "u_max": float(last_step.get("u_max", float("nan"))),
        "worst_true_relative_residual": float(
            linear_summary.get("worst_true_relative_residual", float("nan"))
        ),
        "mg_custom_hierarchy": str(
            linear_solver.get("mg_custom_hierarchy", payload.get("case", {}).get("mg_custom_hierarchy", ""))
        ),
    }


def _run_case(
    *,
    name: str,
    label: str,
    level: int,
    hierarchy: str,
    ranks: int,
    maxit: int,
) -> dict[str, object]:
    case_dir = OUTPUT_ROOT / str(name)
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"

    if not result_path.exists():
        proc = subprocess.run(
            _command(ranks=ranks, level=level, hierarchy=hierarchy, maxit=maxit, out=result_path),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            return {
                "name": str(name),
                "label": str(label),
                "level": int(level),
                "ranks": int(ranks),
                "maxit": int(maxit),
                "mg_custom_hierarchy": str(hierarchy),
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
            "name": str(name),
            "label": str(label),
            "level": int(level),
            "ranks": int(ranks),
            "maxit": int(maxit),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "result_json": str(result_path),
        }
    )
    return aggregated


def main() -> None:
    _ensure_assets()
    summary = _load_summary()
    smoke_rows = list(summary.get("smoke", []))
    rows = list(summary.get("rows", []))
    smoke_done = {str(row.get("name")) for row in smoke_rows}
    done = {str(row.get("name")) for row in rows}

    for variant in SMOKE_VARIANTS:
        if variant["name"] in smoke_done:
            continue
        smoke_rows.append(
            _run_case(
                name=str(variant["name"]),
                label=str(variant["label"]),
                level=int(variant["level"]),
                hierarchy=str(variant["hierarchy"]),
                ranks=SMOKE_RANKS,
                maxit=2,
            )
        )
        _save_summary(smoke_rows, rows)

    for variant in BENCHMARK_VARIANTS:
        if variant["name"] in done:
            continue
        rows.append(
            _run_case(
                name=str(variant["name"]),
                label=str(variant["label"]),
                level=6,
                hierarchy=str(variant["hierarchy"]),
                ranks=RANKS,
                maxit=20,
            )
        )
        _save_summary(smoke_rows, rows)


if __name__ == "__main__":
    main()

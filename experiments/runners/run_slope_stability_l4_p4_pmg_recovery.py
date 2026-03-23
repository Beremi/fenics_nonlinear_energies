#!/usr/bin/env python3
"""Run the L4 P4 PMG recovery matrix with detailed timing extraction."""

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
    / "slope_stability_l4_p4_pmg_recovery_lambda1"
)
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

LEVEL = 4
DEGREE = 4
RANKS = [1, 2, 4, 8]
COMMON_ARGS = [
    "--level",
    str(LEVEL),
    "--elem_degree",
    str(DEGREE),
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
    "--benchmark_mode",
    "warmup_once_then_solve",
    "--save-history",
    "--save-linear-timing",
    "--quiet",
    "--no-use_trust_region",
]

VARIANTS = [
    {
        "name": "original_baseline",
        "label": "Original baseline",
        "kind": "pmg",
        "round": 0,
        "description": "Allgatherv overlap, replicated level/transfer build, Jacobi coarse solve",
        "args": [
            "--distribution_strategy",
            "overlap_allgather",
            "--problem_build_mode",
            "replicated",
            "--mg_level_build_mode",
            "replicated",
            "--mg_transfer_build_mode",
            "replicated",
            "--mg_coarse_backend",
            "jacobi",
            "--mg_coarse_ksp_type",
            "cg",
            "--mg_coarse_pc_type",
            "jacobi",
            "--line_search",
            "golden_fixed",
        ],
    },
    {
        "name": "coarse_hypre_only",
        "label": "Coarse Hypre only",
        "kind": "pmg",
        "round": 1,
        "description": "Original distribution/build path with coarse BoomerAMG and level nullspaces",
        "args": [
            "--distribution_strategy",
            "overlap_allgather",
            "--problem_build_mode",
            "replicated",
            "--mg_level_build_mode",
            "replicated",
            "--mg_transfer_build_mode",
            "replicated",
            "--mg_coarse_backend",
            "hypre",
            "--mg_coarse_ksp_type",
            "cg",
            "--mg_coarse_pc_type",
            "hypre",
            "--mg_coarse_hypre_nodal_coarsen",
            "6",
            "--mg_coarse_hypre_vec_interp_variant",
            "3",
            "--line_search",
            "golden_fixed",
        ],
    },
    {
        "name": "distributed_setup_hypre",
        "label": "Distributed setup + coarse Hypre",
        "kind": "pmg",
        "round": 2,
        "description": "P2P overlap, HDF5-backed root build, owned-row transfer build, coarse BoomerAMG",
        "args": [
            "--distribution_strategy",
            "overlap_p2p",
            "--problem_build_mode",
            "root_bcast",
            "--mg_level_build_mode",
            "root_bcast",
            "--mg_transfer_build_mode",
            "owned_rows",
            "--mg_coarse_backend",
            "hypre",
            "--mg_coarse_ksp_type",
            "cg",
            "--mg_coarse_pc_type",
            "hypre",
            "--mg_coarse_hypre_nodal_coarsen",
            "6",
            "--mg_coarse_hypre_vec_interp_variant",
            "3",
            "--line_search",
            "golden_fixed",
        ],
    },
    {
        "name": "distributed_setup_hypre_armijo",
        "label": "Distributed + coarse Hypre + Armijo",
        "kind": "pmg",
        "round": 3,
        "description": "Same distributed/coarse-Hypre path with Armijo line search",
        "args": [
            "--distribution_strategy",
            "overlap_p2p",
            "--problem_build_mode",
            "root_bcast",
            "--mg_level_build_mode",
            "root_bcast",
            "--mg_transfer_build_mode",
            "owned_rows",
            "--mg_coarse_backend",
            "hypre",
            "--mg_coarse_ksp_type",
            "cg",
            "--mg_coarse_pc_type",
            "hypre",
            "--mg_coarse_hypre_nodal_coarsen",
            "6",
            "--mg_coarse_hypre_vec_interp_variant",
            "3",
            "--line_search",
            "armijo",
        ],
    },
    {
        "name": "distributed_cached_hypre",
        "label": "Distributed + cached transfers",
        "kind": "pmg",
        "round": 4,
        "description": "Same distributed/coarse-Hypre path rerun with cached node-transfer data",
        "args": [
            "--distribution_strategy",
            "overlap_p2p",
            "--problem_build_mode",
            "root_bcast",
            "--mg_level_build_mode",
            "root_bcast",
            "--mg_transfer_build_mode",
            "owned_rows",
            "--mg_coarse_backend",
            "hypre",
            "--mg_coarse_ksp_type",
            "cg",
            "--mg_coarse_pc_type",
            "hypre",
            "--mg_coarse_hypre_nodal_coarsen",
            "6",
            "--mg_coarse_hypre_vec_interp_variant",
            "3",
            "--line_search",
            "golden_fixed",
        ],
    },
    {
        "name": "distributed_cached_hypre_armijo",
        "label": "Distributed + cached + Armijo",
        "kind": "pmg",
        "round": 5,
        "description": "Cached distributed PMG path with Armijo line search",
        "args": [
            "--distribution_strategy",
            "overlap_p2p",
            "--problem_build_mode",
            "root_bcast",
            "--mg_level_build_mode",
            "root_bcast",
            "--mg_transfer_build_mode",
            "owned_rows",
            "--mg_coarse_backend",
            "hypre",
            "--mg_coarse_ksp_type",
            "cg",
            "--mg_coarse_pc_type",
            "hypre",
            "--mg_coarse_hypre_nodal_coarsen",
            "6",
            "--mg_coarse_hypre_vec_interp_variant",
            "3",
            "--line_search",
            "armijo",
        ],
    },
    {
        "name": "tuned_hypre_nonmg",
        "label": "Tuned Hypre baseline",
        "kind": "hypre",
        "round": 99,
        "description": "Top-level tuned BoomerAMG without PMG",
        "args": [
            "--pc_type",
            "hypre",
            "--ksp_type",
            "cg",
            "--ksp_max_it",
            "300",
            "--distribution_strategy",
            "overlap_p2p",
            "--problem_build_mode",
            "root_bcast",
            "--line_search",
            "golden_fixed",
            "--hypre_nodal_coarsen",
            "6",
            "--hypre_vec_interp_variant",
            "3",
        ],
    },
]


def _ensure_assets() -> None:
    for level in (3, 4):
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


def _callback_value(callback: dict[str, object], phase: str, key: str) -> float:
    return float(dict(callback.get(str(phase), {})).get(str(key), 0.0))


def _aggregate(payload: dict[str, object]) -> dict[str, float | int | str | bool | None]:
    result = dict(payload["result"])
    step = dict(result["steps"][0])
    history = list(step.get("history", []))
    linear = list(step.get("linear_timing", []))
    case = dict(payload["case"])
    linear_solver = dict(dict(payload["metadata"]).get("linear_solver", {}))
    timings = dict(payload["timings"])
    callback = dict(timings.get("callback_summary", {}))
    setup_breakdown = dict(timings.get("assembler_setup_breakdown", {}))
    bootstrap_breakdown = dict(timings.get("solver_bootstrap_breakdown", {}))
    linear_summary = dict(step.get("linear_summary", {}))

    return {
        "solver_success": bool(result["solver_success"]),
        "status": str(result["status"]),
        "message": str(step.get("message", "")),
        "line_search": str(case.get("line_search", "")),
        "benchmark_mode": str(case.get("benchmark_mode", "")),
        "distribution_strategy": str(case.get("distribution_strategy", "")),
        "problem_build_mode": str(case.get("problem_build_mode", "")),
        "mg_level_build_mode": str(case.get("mg_level_build_mode", "")),
        "mg_transfer_build_mode": str(case.get("mg_transfer_build_mode", "")),
        "mg_coarse_backend": str(case.get("mg_coarse_backend", "")),
        "mg_coarse_ksp_type": str(case.get("mg_coarse_ksp_type", "")),
        "mg_coarse_pc_type": str(case.get("mg_coarse_pc_type", "")),
        "mg_coarse_hypre_nodal_coarsen": int(case.get("mg_coarse_hypre_nodal_coarsen", -1)),
        "mg_coarse_hypre_vec_interp_variant": int(
            case.get("mg_coarse_hypre_vec_interp_variant", -1)
        ),
        "mg_coarse_hypre_strong_threshold": (
            None
            if case.get("mg_coarse_hypre_strong_threshold") is None
            else float(case["mg_coarse_hypre_strong_threshold"])
        ),
        "mg_coarse_hypre_coarsen_type": str(case.get("mg_coarse_hypre_coarsen_type", "")),
        "use_near_nullspace": bool(case.get("use_near_nullspace", False)),
        "newton_iterations": int(step["nit"]),
        "linear_iterations": int(step["linear_iters"]),
        "line_search_evals": int(sum(int(item.get("ls_evals", 0)) for item in history)),
        "worst_true_relative_residual": float(
            linear_summary.get("worst_true_relative_residual", float("nan"))
        ),
        "energy": float(step["energy"]),
        "omega": float(step["omega"]),
        "u_max": float(step["u_max"]),
        "one_time_setup_time_sec": float(timings.get("one_time_setup_time", 0.0)),
        "steady_state_setup_time_sec": float(timings.get("steady_state_setup_time", 0.0)),
        "solve_time_sec": float(timings.get("solve_time", 0.0)),
        "end_to_end_total_time_sec": float(timings.get("total_time", 0.0)),
        "steady_state_total_time_sec": float(timings.get("steady_state_total_time", 0.0)),
        "benchmark_total_time_sec": float(timings.get("benchmark_total_time", 0.0)),
        "problem_build_time_sec": float(timings.get("problem_build_time", 0.0)),
        "main_problem_build_time_sec": float(timings.get("main_problem_build_time", 0.0)),
        "preconditioner_problem_build_time_sec": float(
            timings.get("preconditioner_problem_build_time", 0.0)
        ),
        "assembler_setup_time_sec": float(timings.get("assembler_setup_time", 0.0)),
        "assembler_distribution_setup_time_sec": float(
            setup_breakdown.get("distribution_setup", 0.0)
        ),
        "assembler_matrix_setup_time_sec": float(setup_breakdown.get("matrix_setup", 0.0)),
        "assembler_ksp_create_time_sec": float(setup_breakdown.get("ksp_create", 0.0)),
        "assembler_warmup_time_sec": float(setup_breakdown.get("warmup", 0.0)),
        "mg_bootstrap_time_sec": float(timings.get("solver_bootstrap_time", 0.0)),
        "mg_hierarchy_build_time_sec": float(
            bootstrap_breakdown.get("mg_hierarchy_build_time", 0.0)
        ),
        "mg_level_build_time_sec": float(bootstrap_breakdown.get("mg_level_build_time", 0.0)),
        "mg_transfer_build_time_sec": float(
            bootstrap_breakdown.get("mg_transfer_build_time", 0.0)
        ),
        "mg_transfer_cache_hits": int(bootstrap_breakdown.get("mg_transfer_cache_hits", 0)),
        "mg_transfer_cache_io_time_sec": float(
            bootstrap_breakdown.get("mg_transfer_cache_io_time", 0.0)
        ),
        "mg_transfer_cache_build_time_sec": float(
            bootstrap_breakdown.get("mg_transfer_cache_build_time", 0.0)
        ),
        "mg_transfer_mapping_time_sec": float(
            bootstrap_breakdown.get("mg_transfer_mapping_time", 0.0)
        ),
        "mg_transfer_matrix_build_time_sec": float(
            bootstrap_breakdown.get("mg_transfer_matrix_build_time", 0.0)
        ),
        "mg_level_assembler_build_time_sec": float(
            bootstrap_breakdown.get("mg_level_assembler_build_time", 0.0)
        ),
        "mg_configure_time_sec": float(bootstrap_breakdown.get("mg_configure_time", 0.0)),
        "energy_eval_calls": int(callback.get("energy", {}).get("calls", 0)),
        "energy_total_time_sec": _callback_value(callback, "energy", "total"),
        "energy_kernel_time_sec": _callback_value(callback, "energy", "kernel"),
        "energy_ghost_exchange_time_sec": _callback_value(callback, "energy", "ghost_exchange"),
        "energy_allgather_time_sec": _callback_value(callback, "energy", "allgatherv"),
        "energy_allreduce_time_sec": _callback_value(callback, "energy", "allreduce"),
        "gradient_eval_calls": int(callback.get("gradient", {}).get("calls", 0)),
        "gradient_total_time_sec": _callback_value(callback, "gradient", "total"),
        "gradient_kernel_time_sec": _callback_value(callback, "gradient", "kernel"),
        "gradient_ghost_exchange_time_sec": _callback_value(
            callback, "gradient", "ghost_exchange"
        ),
        "gradient_allgather_time_sec": _callback_value(callback, "gradient", "allgatherv"),
        "hessian_eval_calls": int(callback.get("hessian", {}).get("calls", 0)),
        "hessian_total_time_sec": _callback_value(callback, "hessian", "total"),
        "hessian_kernel_time_sec": _callback_value(callback, "hessian", "hvp_compute"),
        "hessian_ghost_exchange_time_sec": _callback_value(
            callback, "hessian", "ghost_exchange"
        ),
        "hessian_allgather_time_sec": _callback_value(callback, "hessian", "allgatherv"),
        "hessian_assembly_time_sec": _callback_value(callback, "hessian", "coo_assembly"),
        "line_search_time_sec": _sum(history, "t_ls"),
        "newton_grad_phase_time_sec": _sum(history, "t_grad"),
        "newton_hessian_phase_time_sec": _sum(history, "t_hess"),
        "newton_update_time_sec": _sum(history, "t_update"),
        "linear_operator_prepare_time_sec": _sum(linear, "operator_prepare_total_time"),
        "linear_assemble_time_sec": _sum(linear, "assemble_total_time"),
        "linear_pc_operator_time_sec": _sum(linear, "pc_operator_assemble_total_time"),
        "linear_pc_setup_time_sec": _sum(linear, "pc_setup_time"),
        "linear_ksp_solve_time_sec": _sum(linear, "solve_time"),
        "linear_total_time_sec": _sum(linear, "linear_total_time"),
        "linear_operator_apply_time_sec": _sum(linear, "operator_apply_total_time"),
        "linear_operator_exchange_time_sec": _sum(linear, "operator_apply_allgatherv")
        + _sum(linear, "operator_apply_ghost_exchange"),
        "linear_assembly_exchange_time_sec": _sum(linear, "assemble_p2p_exchange"),
        "pc_type_effective": str(linear_solver.get("pc_type_effective", "")),
        "reuse_mode": (
            "stale_preconditioner"
            if bool(linear_solver.get("pc_reuse_preconditioner", False))
            else "object_only"
        ),
    }


def _run_case(variant: dict[str, object], ranks: int) -> dict[str, object]:
    case_dir = OUTPUT_ROOT / str(variant["name"]) / f"np{ranks}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"

    if not result_path.exists():
        proc = subprocess.run(
            [
                "mpiexec",
                "-n",
                str(ranks),
                str(PYTHON),
                "-u",
                str(SOLVER),
                *COMMON_ARGS,
                *list(variant["args"]),
                "--out",
                str(result_path),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            return {
                "variant": str(variant["name"]),
                "variant_label": str(variant["label"]),
                "variant_kind": str(variant["kind"]),
                "round": int(variant["round"]),
                "description": str(variant["description"]),
                "ranks": int(ranks),
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
    row = {
        "variant": str(variant["name"]),
        "variant_label": str(variant["label"]),
        "variant_kind": str(variant["kind"]),
        "round": int(variant["round"]),
        "description": str(variant["description"]),
        "ranks": int(ranks),
        "level": int(payload["mesh"]["level"]),
        "elem_degree": int(payload["case"]["element_degree"]),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "result_json": str(result_path),
    }
    row.update(_aggregate(payload))
    return row


def main() -> None:
    _ensure_assets()
    rows = _load_rows()
    existing = {(str(row["variant"]), int(row["ranks"])) for row in rows}

    for variant in VARIANTS:
        for ranks in RANKS:
            key = (str(variant["name"]), int(ranks))
            if key in existing:
                continue
            rows.append(_run_case(variant, ranks))
            rows.sort(key=lambda item: (int(item["round"]), str(item["variant"]), int(item["ranks"])))
            _write_rows(rows)

    rows.sort(key=lambda item: (int(item["round"]), str(item["variant"]), int(item["ranks"])))
    _write_rows(rows)


if __name__ == "__main__":
    main()

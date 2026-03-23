#!/usr/bin/env python3
"""Run a piece-by-piece distribution study for the L4 P4 PMG solver."""

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
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "slope_stability_l4_p4_distribution_study_lambda1"
)
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

RANKS = [1, 2, 4, 8]
COMMON_ARGS = [
    "--level",
    "4",
    "--elem_degree",
    "4",
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
    "--no-use_trust_region",
    "--save-history",
    "--save-linear-timing",
    "--quiet",
]

VARIANTS = [
    {
        "name": "baseline_replicated",
        "label": "Baseline",
        "args": [
            "--distribution_strategy",
            "overlap_allgather",
            "--problem_build_mode",
            "replicated",
            "--mg_level_build_mode",
            "replicated",
            "--mg_transfer_build_mode",
            "replicated",
        ],
    },
    {
        "name": "distributed_compute_only",
        "label": "P2P only",
        "args": [
            "--distribution_strategy",
            "overlap_p2p",
            "--problem_build_mode",
            "replicated",
            "--mg_level_build_mode",
            "replicated",
            "--mg_transfer_build_mode",
            "replicated",
        ],
    },
    {
        "name": "distributed_setup_only",
        "label": "Root-build only",
        "args": [
            "--distribution_strategy",
            "overlap_allgather",
            "--problem_build_mode",
            "root_bcast",
            "--mg_level_build_mode",
            "root_bcast",
            "--mg_transfer_build_mode",
            "root_bcast",
        ],
    },
    {
        "name": "distributed_full",
        "label": "Full distributed",
        "args": [
            "--distribution_strategy",
            "overlap_p2p",
            "--problem_build_mode",
            "root_bcast",
            "--mg_level_build_mode",
            "root_bcast",
            "--mg_transfer_build_mode",
            "root_bcast",
        ],
    },
    {
        "name": "distributed_best_effort",
        "label": "P2P + owned-row MG",
        "args": [
            "--distribution_strategy",
            "overlap_p2p",
            "--problem_build_mode",
            "replicated",
            "--mg_level_build_mode",
            "replicated",
            "--mg_transfer_build_mode",
            "owned_rows",
        ],
    },
]


def _load_rows() -> list[dict[str, object]]:
    if SUMMARY_PATH.exists():
        return list(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    return []


def _write_rows(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _aggregate(payload: dict[str, object]) -> dict[str, float | int]:
    step = dict(payload["result"]["steps"][0])
    history = list(step.get("history", []))
    linear = list(step.get("linear_timing", []))
    callback = dict(payload["timings"].get("callback_summary", {}))
    linear_summary = dict(step.get("linear_summary", {}))
    setup_breakdown = dict(payload["timings"].get("assembler_setup_breakdown", {}))
    bootstrap_breakdown = dict(payload["timings"].get("solver_bootstrap_breakdown", {}))

    def _sum(items, key):
        return float(sum(float(item.get(key, 0.0)) for item in items))

    return {
        "newton_iterations": int(step["nit"]),
        "linear_iterations": int(step["linear_iters"]),
        "energy": float(step["energy"]),
        "omega": float(step["omega"]),
        "u_max": float(step["u_max"]),
        "worst_true_relative_residual": float(
            linear_summary.get("worst_true_relative_residual", float("nan"))
        ),
        "energy_eval_calls": int(callback.get("energy", {}).get("calls", 0)),
        "energy_eval_time_sec": float(callback.get("energy", {}).get("total", 0.0)),
        "gradient_eval_calls": int(callback.get("gradient", {}).get("calls", 0)),
        "gradient_eval_time_sec": float(callback.get("gradient", {}).get("total", 0.0)),
        "hessian_eval_calls": int(callback.get("hessian", {}).get("calls", 0)),
        "hessian_eval_time_sec": float(callback.get("hessian", {}).get("total", 0.0)),
        "line_search_time_sec": _sum(history, "t_ls"),
        "line_search_evals": int(sum(int(item.get("ls_evals", 0)) for item in history)),
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
        "problem_build_time_sec": float(payload["timings"].get("problem_build_time", 0.0)),
        "main_problem_build_time_sec": float(
            payload["timings"].get("main_problem_build_time", 0.0)
        ),
        "assembler_setup_time_sec": float(
            payload["timings"].get("assembler_setup_time", 0.0)
        ),
        "assembler_warmup_time_sec": float(setup_breakdown.get("warmup", 0.0)),
        "assembler_distribution_setup_time_sec": float(
            setup_breakdown.get("distribution_setup", 0.0)
        ),
        "mg_bootstrap_time_sec": float(
            payload["timings"].get("solver_bootstrap_time", 0.0)
        ),
        "mg_hierarchy_build_time_sec": float(
            bootstrap_breakdown.get("mg_hierarchy_build_time", 0.0)
        ),
        "mg_level_assembler_build_time_sec": float(
            bootstrap_breakdown.get("mg_level_assembler_build_time", 0.0)
        ),
        "mg_configure_time_sec": float(
            bootstrap_breakdown.get("mg_configure_time", 0.0)
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
        "ranks": int(ranks),
        "solver_success": bool(payload["result"]["solver_success"]),
        "status": str(payload["result"]["status"]),
        "message": str(payload["result"]["steps"][0].get("message", "")),
        "level": int(payload["mesh"]["level"]),
        "h": float(payload["mesh"]["h"]),
        "nodes": int(payload["mesh"]["nodes"]),
        "elements": int(payload["mesh"]["elements"]),
        "free_dofs": int(payload["mesh"]["free_dofs"]),
        "setup_time_sec": float(payload["timings"]["one_time_setup_time"]),
        "solve_time_sec": float(payload["timings"]["solve_time"]),
        "total_time_sec": float(payload["timings"]["total_time"]),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "result_json": str(result_path),
        "distribution_strategy": str(payload["case"]["distribution_strategy"]),
        "problem_build_mode": str(payload["case"]["problem_build_mode"]),
        "mg_level_build_mode": str(payload["case"]["mg_level_build_mode"]),
        "mg_transfer_build_mode": str(payload["case"]["mg_transfer_build_mode"]),
    }
    row.update(_aggregate(payload))
    return row


def main() -> None:
    rows = _load_rows()
    completed = {(str(row["variant"]), int(row["ranks"])) for row in rows}
    for variant in VARIANTS:
        for ranks in RANKS:
            key = (str(variant["name"]), int(ranks))
            if key in completed:
                continue
            rows.append(_run_case(variant, ranks))
            _write_rows(rows)


if __name__ == "__main__":
    main()

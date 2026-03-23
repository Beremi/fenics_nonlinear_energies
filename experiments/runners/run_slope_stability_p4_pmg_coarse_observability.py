#!/usr/bin/env python3
"""Probe how coarse-Hypre settings affect PMG behavior on L4/L5 P4 runs."""

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
    / "slope_stability_p4_pmg_coarse_observability_lambda1"
)
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

PHASE_TARGETS = {
    "max_iter": [(4, 1), (4, 8), (5, 8)],
    "reference": [(4, 1)],
}

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
    "--mg_coarse_hypre_strong_threshold",
    "0.5",
    "--mg_coarse_hypre_coarsen_type",
    "HMIS",
    "--mg_coarse_hypre_relax_type_all",
    "symmetric-SOR/Jacobi",
    "--line_search",
    "golden_fixed",
]

VARIANTS = [
    {
        "name": "maxiter_1",
        "phase": "max_iter",
        "label": "BoomerAMG max_iter=1",
        "args": ["--mg_coarse_hypre_max_iter", "1", "--mg_coarse_hypre_tol", "0.0"],
    },
    {
        "name": "maxiter_2",
        "phase": "max_iter",
        "label": "BoomerAMG max_iter=2",
        "args": ["--mg_coarse_hypre_max_iter", "2", "--mg_coarse_hypre_tol", "0.0"],
    },
    {
        "name": "maxiter_4",
        "phase": "max_iter",
        "label": "BoomerAMG max_iter=4",
        "args": ["--mg_coarse_hypre_max_iter", "4", "--mg_coarse_hypre_tol", "0.0"],
    },
    {
        "name": "maxiter_6",
        "phase": "max_iter",
        "label": "BoomerAMG max_iter=6",
        "args": ["--mg_coarse_hypre_max_iter", "6", "--mg_coarse_hypre_tol", "0.0"],
    },
]

EXACT_VARIANTS = [
    {
        "name": "coarse_lu_reference",
        "phase": "reference",
        "label": "Exact coarse LU reference",
        "level": 4,
        "ranks": 1,
        "args": ["--mg_coarse_backend", "lu", "--mg_coarse_ksp_type", "preonly", "--mg_coarse_pc_type", "lu"],
    }
]


def _ensure_assets() -> None:
    for level in (4, 5):
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


def _collect_mg_family(linear_timing: list[dict[str, object]], family: str) -> dict[str, float]:
    observed_time = 0.0
    solve_invocations = 0
    total_iterations = 0
    contractions: list[float] = []
    for entry in linear_timing:
        for diag in list(entry.get("mg_runtime_diagnostics", [])):
            if str(diag.get("family")) != str(family):
                continue
            observed_time += float(diag.get("observed_time_sec", 0.0))
            solve_invocations += int(diag.get("solve_invocations", 0))
            total_iterations += int(diag.get("total_iterations", 0))
            contraction = diag.get("average_residual_contraction")
            if contraction is not None:
                contractions.append(float(contraction))
    return {
        "observed_time_sec": float(observed_time),
        "solve_invocations": int(solve_invocations),
        "total_iterations": int(total_iterations),
        "average_residual_contraction": (
            None if not contractions else float(sum(contractions) / len(contractions))
        ),
    }


def _aggregate(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload["result"])
    step = dict(result["steps"][0])
    history = list(step.get("history", []))
    linear = list(step.get("linear_timing", []))
    case = dict(payload["case"])
    timings = dict(payload["timings"])
    callback = dict(timings.get("callback_summary", {}))
    bootstrap = dict(timings.get("solver_bootstrap_breakdown", {}))
    linear_summary = dict(step.get("linear_summary", {}))
    coarse = _collect_mg_family(linear, "coarse")
    fine = _collect_mg_family(linear, "fine")
    degree2 = _collect_mg_family(linear, "degree2")
    degree1 = _collect_mg_family(linear, "degree1")
    return {
        "solver_success": bool(result["solver_success"]),
        "status": str(result["status"]),
        "message": str(step.get("message", "")),
        "newton_iterations": int(step["nit"]),
        "linear_iterations": int(step["linear_iters"]),
        "line_search_evals": int(sum(int(item.get("ls_evals", 0)) for item in history)),
        "worst_true_relative_residual": float(
            linear_summary.get("worst_true_relative_residual", float("nan"))
        ),
        "steady_state_total_time_sec": float(timings.get("steady_state_total_time", 0.0)),
        "end_to_end_total_time_sec": float(timings.get("total_time", 0.0)),
        "solve_time_sec": float(timings.get("solve_time", 0.0)),
        "problem_build_time_sec": float(timings.get("problem_build_time", 0.0)),
        "assembler_warmup_time_sec": float(
            dict(timings.get("assembler_setup_breakdown", {})).get("warmup", 0.0)
        ),
        "mg_hierarchy_build_time_sec": float(bootstrap.get("mg_hierarchy_build_time", 0.0)),
        "mg_transfer_build_time_sec": float(bootstrap.get("mg_transfer_build_time", 0.0)),
        "mg_configure_time_sec": float(bootstrap.get("mg_configure_time", 0.0)),
        "energy_total_time_sec": _callback_value(callback, "energy", "total"),
        "gradient_total_time_sec": _callback_value(callback, "gradient", "total"),
        "hessian_total_time_sec": _callback_value(callback, "hessian", "total"),
        "line_search_time_sec": float(sum(float(item.get("t_ls", 0.0)) for item in history)),
        "linear_pc_setup_time_sec": _sum(linear, "pc_setup_time"),
        "linear_ksp_solve_time_sec": _sum(linear, "solve_time"),
        "linear_assemble_time_sec": _sum(linear, "assemble_total_time"),
        "coarse_observed_time_sec": float(coarse["observed_time_sec"]),
        "coarse_solve_invocations": int(coarse["solve_invocations"]),
        "coarse_total_iterations": int(coarse["total_iterations"]),
        "coarse_average_residual_contraction": coarse["average_residual_contraction"],
        "fine_observed_time_sec": float(fine["observed_time_sec"]),
        "p2_observed_time_sec": float(degree2["observed_time_sec"]),
        "p1_observed_time_sec": float(degree1["observed_time_sec"]),
        "mg_coarse_backend": str(case.get("mg_coarse_backend", "")),
        "mg_coarse_ksp_type": str(case.get("mg_coarse_ksp_type", "")),
        "mg_coarse_pc_type": str(case.get("mg_coarse_pc_type", "")),
        "mg_coarse_hypre_max_iter": int(case.get("mg_coarse_hypre_max_iter", -1)),
        "mg_coarse_hypre_tol": float(case.get("mg_coarse_hypre_tol", 0.0)),
    }


def _run_case(*, level: int, ranks: int, variant: dict[str, object]) -> dict[str, object]:
    case_dir = OUTPUT_ROOT / f"level{level}" / str(variant["name"]) / f"np{ranks}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"
    if not result_path.exists():
        command = [
            "mpiexec",
            "-n",
            str(ranks),
            str(PYTHON),
            "-u",
            str(SOLVER),
            "--level",
            str(level),
            *COMMON_ARGS,
            *list(variant["args"]),
            "--out",
            str(result_path),
        ]
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0 and not result_path.exists():
            return {
                "level": int(level),
                "ranks": int(ranks),
                "variant": str(variant["name"]),
                "phase": str(variant["phase"]),
                "label": str(variant["label"]),
                "solver_success": False,
                "status": "subprocess_failed",
                "message": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "subprocess failed",
                "stdout_path": str(stdout_path.relative_to(REPO_ROOT)),
                "stderr_path": str(stderr_path.relative_to(REPO_ROOT)),
            }
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    row = {
        "level": int(level),
        "ranks": int(ranks),
        "variant": str(variant["name"]),
        "phase": str(variant["phase"]),
        "label": str(variant["label"]),
        "result_path": str(result_path.relative_to(REPO_ROOT)),
        "stdout_path": str(stdout_path.relative_to(REPO_ROOT)),
        "stderr_path": str(stderr_path.relative_to(REPO_ROOT)),
    }
    row.update(_aggregate(payload))
    return row


def main() -> None:
    _ensure_assets()
    rows = _load_rows()
    existing = {
        (int(row["level"]), int(row["ranks"]), str(row["variant"]))
        for row in rows
    }

    for variant in VARIANTS:
        for level, ranks in PHASE_TARGETS[str(variant["phase"])]:
            key = (int(level), int(ranks), str(variant["name"]))
            if key in existing:
                continue
            rows.append(_run_case(level=int(level), ranks=int(ranks), variant=variant))
            _write_rows(rows)
    for variant in EXACT_VARIANTS:
        for level, ranks in PHASE_TARGETS["reference"]:
            if int(variant["level"]) != int(level) or int(variant["ranks"]) != int(ranks):
                continue
            key = (int(level), int(ranks), str(variant["name"]))
            if key in existing:
                continue
            rows.append(_run_case(level=int(level), ranks=int(ranks), variant=variant))
            _write_rows(rows)

    rows.sort(key=lambda row: (int(row["level"]), str(row["phase"]), str(row["variant"]), int(row["ranks"])))
    _write_rows(rows)


if __name__ == "__main__":
    main()

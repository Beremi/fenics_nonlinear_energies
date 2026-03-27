#!/usr/bin/env python3
"""Run the JAX + PETSc arctan-resonance timing and scaling campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SOLVER = REPO_ROOT / "src" / "problems" / "plaplace_up_arctan" / "jax_petsc" / "solve_case.py"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "raw_results" / "plaplace_up_arctan_petsc"
DEFAULT_SUMMARY = DEFAULT_OUT_DIR / "summary.json"
DEFAULT_SERIAL_SUMMARY = REPO_ROOT / "artifacts" / "raw_results" / "plaplace_up_arctan_full" / "summary.json"
DEFAULT_TIME_LIMIT_S = 200.0
DEFAULT_MAX_LEVEL = 13
P2_START_LEVEL = 8
P3_START_LEVEL = 8
SCALING_RANKS = (1, 2, 4, 8, 16)
PMG_CONFIG = {
    "pc_type": "mg",
    "ksp_type": "fgmres",
    "ksp_rtol": 1.0e-8,
    "ksp_max_it": 1200,
    "merit_ksp_type": "cg",
    "merit_ksp_rtol": 1.0e-10,
    "merit_ksp_max_it": 400,
    "mg_coarsest_level": 2,
    "mg_smoother_ksp_type": "chebyshev",
    "mg_smoother_pc_type": "jacobi",
    "mg_smoother_steps": 2,
    "distribution_strategy": "overlap_p2p",
    "element_reorder_mode": "block_xyz",
    "local_hessian_mode": "element",
}
THREAD_ENV = {
    "JAX_PLATFORMS": "cpu",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}
SUITE_REVISION = "jax_petsc_pmg_timing_scaling_v2"


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _normalize_command(argv: list[str]) -> str:
    parts: list[str] = []
    for part in argv:
        text = str(part)
        if text == str(PYTHON):
            text = "./.venv/bin/python"
        elif text.startswith(str(REPO_ROOT) + "/"):
            text = text[len(str(REPO_ROOT)) + 1 :]
        parts.append(text)
    return shlex.join(parts)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run(argv: list[str], *, env: dict[str, str] | None = None, timeout_s: float = 21600.0) -> dict[str, Any]:
    env_map = os.environ.copy()
    env_map.update(env or {})
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            env=env_map,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed = time.perf_counter() - t0
        return {
            "argv": [str(item) for item in argv],
            "command": _normalize_command(argv),
            "exit_code": int(proc.returncode),
            "elapsed_s": float(elapsed),
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - t0
        return {
            "argv": [str(item) for item in argv],
            "command": _normalize_command(argv),
            "exit_code": 124,
            "elapsed_s": float(elapsed),
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
        }


def _save_run_logs(run_info: dict[str, Any], log_prefix: Path) -> dict[str, Any]:
    stdout_path = log_prefix.with_suffix(".stdout.txt")
    stderr_path = log_prefix.with_suffix(".stderr.txt")
    stdout_path.write_text(str(run_info.pop("stdout")), encoding="utf-8")
    stderr_path.write_text(str(run_info.pop("stderr")), encoding="utf-8")
    run_info["stdout_path"] = _repo_rel(stdout_path)
    run_info["stderr_path"] = _repo_rel(stderr_path)
    return run_info


def _reference_payload(serial_summary_dir: Path, name: str) -> dict[str, Any]:
    return _read_json(serial_summary_dir / "references" / name / "output.json")


def _solver_args(
    *,
    p: float,
    level: int,
    lambda1: float,
    lambda_level: int,
    init_state: Path,
    state_out: Path,
    result_out: Path,
    nproc: int,
    epsilon: float = 1.0e-8,
    maxit: int = 40,
    init_scale: float = 0.9,
) -> list[str]:
    argv = [
        str(PYTHON),
        "-u",
        str(SOLVER),
        "--p",
        f"{float(p):g}",
        "--level",
        str(int(level)),
        "--lambda1",
        f"{float(lambda1):.16g}",
        "--lambda-level",
        str(int(lambda_level)),
        "--epsilon",
        f"{float(epsilon):.1e}",
        "--maxit",
        str(int(maxit)),
        "--init-state",
        str(init_state),
        "--init-scale",
        f"{float(init_scale):.16g}",
        "--pc-type",
        str(PMG_CONFIG["pc_type"]),
        "--ksp-type",
        str(PMG_CONFIG["ksp_type"]),
        "--ksp-rtol",
        f"{float(PMG_CONFIG['ksp_rtol']):.1e}",
        "--ksp-max-it",
        str(int(PMG_CONFIG["ksp_max_it"])),
        "--merit-ksp-type",
        str(PMG_CONFIG["merit_ksp_type"]),
        "--merit-ksp-rtol",
        f"{float(PMG_CONFIG['merit_ksp_rtol']):.1e}",
        "--merit-ksp-max-it",
        str(int(PMG_CONFIG["merit_ksp_max_it"])),
        "--mg-coarsest-level",
        str(int(PMG_CONFIG["mg_coarsest_level"])),
        "--mg-smoother-ksp-type",
        str(PMG_CONFIG["mg_smoother_ksp_type"]),
        "--mg-smoother-pc-type",
        str(PMG_CONFIG["mg_smoother_pc_type"]),
        "--mg-smoother-steps",
        str(int(PMG_CONFIG["mg_smoother_steps"])),
        "--distribution-strategy",
        str(PMG_CONFIG["distribution_strategy"]),
        "--element-reorder-mode",
        str(PMG_CONFIG["element_reorder_mode"]),
        "--local-hessian-mode",
        str(PMG_CONFIG["local_hessian_mode"]),
        "--nproc",
        str(int(nproc)),
        "--state-out",
        str(state_out),
        "--out",
        str(result_out),
        "--quiet",
    ]
    if int(nproc) <= 1:
        return argv
    return [
        "mpiexec",
        "--bind-to",
        "none",
        "-n",
        str(int(nproc)),
        *argv,
    ]


def _status_reason(run_info: dict[str, Any]) -> str:
    stderr = str(run_info.get("stderr", "")).strip()
    stdout = str(run_info.get("stdout", "")).strip()
    if stderr:
        return stderr.splitlines()[-1]
    if stdout:
        return stdout.splitlines()[-1]
    return "subprocess failed"


def _flatten_case_row(
    *,
    study: str,
    p: float,
    level: int,
    nprocs: int,
    lambda1: float,
    lambda_level: int,
    lambda_source: str,
    init_state: Path,
    case_dir: Path,
    result_path: Path,
    run_info: dict[str, Any] | None,
) -> dict[str, Any]:
    if not result_path.exists():
        return {
            "study": str(study),
            "p": float(p),
            "level": int(level),
            "nprocs": int(nprocs),
            "status": "subprocess_failed",
            "message": "result file missing",
            "lambda1": float(lambda1),
            "lambda_level": int(lambda_level),
            "lambda_source": str(lambda_source),
            "init_state_path": _repo_rel(init_state),
            "case_dir": _repo_rel(case_dir),
            "result_json": _repo_rel(result_path),
            "command": "" if run_info is None else str(run_info.get("command", "")),
        }
    payload = _read_json(result_path)
    result = dict(payload.get("result", {}))
    problem = dict(payload.get("problem", {}))
    timings = dict(payload.get("timings", result.get("timings", {})))
    linear_solver = dict(payload.get("metadata", {}).get("linear_solver", result.get("linear_solver", {})))
    row = {
        "study": str(study),
        "p": float(p),
        "level": int(level),
        "nprocs": int(nprocs),
        "status": str(result.get("status", "failed")),
        "message": str(result.get("message", "")),
        "lambda1": float(lambda1),
        "lambda_level": int(lambda_level),
        "lambda_source": str(lambda_source),
        "free_dofs": int(problem.get("free_dofs", 0)),
        "total_dofs": int(problem.get("total_dofs", 0)),
        "h": float(problem.get("h", math.nan)),
        "J": float(result.get("J", math.nan)),
        "residual_norm": float(result.get("residual_norm", math.nan)),
        "gradient_residual_norm": float(result.get("gradient_residual_norm", math.nan)),
        "outer_iterations": int(result.get("outer_iterations", 0)),
        "accepted_step_count": int(result.get("accepted_step_count", 0)),
        "linear_iterations_total": int(timings.get("linear_iterations_total", 0)),
        "setup_time_s": float(timings.get("setup_time", 0.0)),
        "solve_time_s": float(timings.get("solve_time", 0.0)),
        "total_time_s": float(timings.get("total_time", 0.0)),
        "pc_type": str(linear_solver.get("pc_type", "")),
        "ksp_type": str(linear_solver.get("ksp_type", "")),
        "step_preconditioner_operator": str(linear_solver.get("step_preconditioner_operator", "")),
        "mg_smoother_ksp_type": str(PMG_CONFIG["mg_smoother_ksp_type"]),
        "mg_smoother_pc_type": str(PMG_CONFIG["mg_smoother_pc_type"]),
        "init_state_path": _repo_rel(init_state),
        "state_path": _repo_rel(Path(str(result.get("state_out", "")))) if result.get("state_out") else "",
        "case_dir": _repo_rel(case_dir),
        "result_json": _repo_rel(result_path),
    }
    if run_info is not None:
        row["command"] = str(run_info.get("command", ""))
        row["run_elapsed_s"] = float(run_info.get("elapsed_s", 0.0))
        if "stdout_path" in run_info:
            row["stdout_path"] = str(run_info["stdout_path"])
        if "stderr_path" in run_info:
            row["stderr_path"] = str(run_info["stderr_path"])
        row["subprocess_exit_code"] = int(run_info.get("exit_code", 0))
    return row


def _run_solver_case(
    *,
    root: Path,
    study: str,
    p: float,
    level: int,
    nprocs: int,
    lambda1: float,
    lambda_level: int,
    lambda_source: str,
    init_state: Path,
    force: bool,
    timeout_s: float = 21600.0,
) -> dict[str, Any]:
    case_dir = root / study / f"p{int(round(float(p)))}" / f"L{int(level)}" / f"np{int(nprocs)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    state_out = case_dir / "state.npz"
    row_path = case_dir / "row.json"
    if row_path.exists() and not force:
        return dict(_read_json(row_path))
    if result_path.exists() and not force:
        row = _flatten_case_row(
            study=study,
            p=p,
            level=level,
            nprocs=nprocs,
            lambda1=lambda1,
            lambda_level=lambda_level,
            lambda_source=lambda_source,
            init_state=init_state,
            case_dir=case_dir,
            result_path=result_path,
            run_info=None,
        )
        _write_json(row_path, row)
        return row

    argv = _solver_args(
        p=p,
        level=level,
        lambda1=lambda1,
        lambda_level=lambda_level,
        init_state=init_state,
        state_out=state_out,
        result_out=result_path,
        nproc=nprocs,
    )
    run_info = _run(argv, env=THREAD_ENV, timeout_s=float(timeout_s))
    run_info = _save_run_logs(run_info, case_dir / "run")
    if int(run_info["exit_code"]) != 0:
        timed_out = bool(run_info.get("timed_out", False))
        row = {
            "study": str(study),
            "p": float(p),
            "level": int(level),
            "nprocs": int(nprocs),
            "status": "timed_out" if timed_out else "subprocess_failed",
            "message": (
                f"subprocess timed out after {float(timeout_s):.1f}s"
                if timed_out
                else _status_reason(run_info)
            ),
            "lambda1": float(lambda1),
            "lambda_level": int(lambda_level),
            "lambda_source": str(lambda_source),
            "init_state_path": _repo_rel(init_state),
            "case_dir": _repo_rel(case_dir),
            "result_json": _repo_rel(result_path),
            "command": str(run_info["command"]),
            "run_elapsed_s": float(run_info["elapsed_s"]),
            "total_time_s": float(run_info["elapsed_s"]),
            "stdout_path": str(run_info["stdout_path"]),
            "stderr_path": str(run_info["stderr_path"]),
            "subprocess_exit_code": int(run_info["exit_code"]),
        }
        _write_json(row_path, row)
        return row

    row = _flatten_case_row(
        study=study,
        p=p,
        level=level,
        nprocs=nprocs,
        lambda1=lambda1,
        lambda_level=lambda_level,
        lambda_source=lambda_source,
        init_state=init_state,
        case_dir=case_dir,
        result_path=result_path,
        run_info=run_info,
    )
    _write_json(row_path, row)
    return row


def _augment_scaling_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    baseline = next((row for row in rows if int(row["nprocs"]) == 1 and str(row["status"]) == "completed"), None)
    if baseline is None:
        return rows
    base_total = float(baseline.get("total_time_s", 0.0))
    base_solve = float(baseline.get("solve_time_s", 0.0))
    for row in rows:
        total = float(row.get("total_time_s", 0.0))
        solve = float(row.get("solve_time_s", 0.0))
        ranks = int(row["nprocs"])
        if str(row.get("status")) == "completed" and total > 0.0:
            row["speedup_total"] = base_total / total if base_total > 0.0 else math.nan
            row["efficiency_total"] = row["speedup_total"] / float(ranks)
        else:
            row["speedup_total"] = math.nan
            row["efficiency_total"] = math.nan
        if str(row.get("status")) == "completed" and solve > 0.0:
            row["speedup_solve"] = base_solve / solve if base_solve > 0.0 else math.nan
            row["efficiency_solve"] = row["speedup_solve"] / float(ranks)
        else:
            row["speedup_solve"] = math.nan
            row["efficiency_solve"] = math.nan
    return rows


def _run_mesh_ladder(
    *,
    out_dir: Path,
    p: float,
    start_level: int,
    init_state: Path,
    lambda1: float,
    lambda_source: str,
    time_limit_s: float,
    max_level: int,
    force: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    current_state = init_state
    stop_reason = "max_level_reached"
    for level in range(int(start_level), int(max_level) + 1):
        row = _run_solver_case(
            root=out_dir,
            study="mesh_ladder",
            p=p,
            level=level,
            nprocs=1,
            lambda1=lambda1,
            lambda_level=level,
            lambda_source=lambda_source,
            init_state=current_state,
            force=force,
            timeout_s=float(time_limit_s),
        )
        rows.append(row)
        if str(row.get("status")) == "timed_out":
            stop_reason = "timed_out"
            break
        if str(row.get("status")) != "completed":
            stop_reason = "solver_failed"
            break
        state_path = REPO_ROOT / str(row["state_path"])
        current_state = state_path
        if float(row.get("total_time_s", 0.0)) > float(time_limit_s):
            stop_reason = "wall_time_threshold_reached"
            break
    return {
        "rows": rows,
        "stop_reason": stop_reason,
        "last_success_level": max((int(row["level"]) for row in rows if str(row["status"]) == "completed"), default=None),
        "last_success_state_path": (
            str((REPO_ROOT / str(next(row["state_path"] for row in reversed(rows) if str(row["status"]) == "completed"))).resolve())
            if any(str(row["status"]) == "completed" for row in rows)
            else None
        ),
    }


def _run_strong_scaling(
    *,
    out_dir: Path,
    p: float,
    level: int,
    init_state: Path,
    lambda1: float,
    lambda_source: str,
    force: bool,
    baseline_row: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if baseline_row is not None:
        reused = dict(baseline_row)
        reused["study"] = "strong_scaling"
        reused["command"] = str(reused.get("command", "")) + "  # reused from mesh_ladder"
        rows.append(reused)
    for ranks in SCALING_RANKS:
        if int(ranks) == 1 and baseline_row is not None:
            continue
        rows.append(
            _run_solver_case(
                root=out_dir,
                study="strong_scaling",
                p=p,
                level=level,
                nprocs=ranks,
                lambda1=lambda1,
                lambda_level=level,
                lambda_source=lambda_source,
                init_state=init_state,
                force=force,
            )
        )
    return _augment_scaling_rows(rows)


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    statuses = sorted({str(row.get("status", "")) for row in rows})
    return {status: sum(1 for row in rows if str(row.get("status", "")) == status) for status in statuses}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--serial-summary", type=Path, default=DEFAULT_SERIAL_SUMMARY)
    parser.add_argument("--time-limit-s", type=float, default=DEFAULT_TIME_LIMIT_S)
    parser.add_argument("--max-level", type=int, default=DEFAULT_MAX_LEVEL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    serial_summary_dir = args.serial_summary.parent
    p2_reference = _reference_payload(serial_summary_dir, "p2_newton_l7")
    p3_reference = _reference_payload(serial_summary_dir, "p3_certified_l7")
    p2_init_state = serial_summary_dir / "references" / "p2_newton_l7" / "state.npz"
    p3_init_state = serial_summary_dir / "references" / "p3_certified_l7" / "cont_p3_direct_state.npz"
    p2_lambda1 = float(p2_reference["lambda1"])
    p3_lambda1 = float(p3_reference["certified"]["lambda1"])

    p2_mesh = _run_mesh_ladder(
        out_dir=args.out_dir,
        p=2.0,
        start_level=P2_START_LEVEL,
        init_state=p2_init_state,
        lambda1=p2_lambda1,
        lambda_source="exact",
        time_limit_s=float(args.time_limit_s),
        max_level=int(args.max_level),
        force=bool(args.force),
    )
    p3_mesh = _run_mesh_ladder(
        out_dir=args.out_dir,
        p=3.0,
        start_level=P3_START_LEVEL,
        init_state=p3_init_state,
        lambda1=p3_lambda1,
        lambda_source="frozen_l7_reference",
        time_limit_s=float(args.time_limit_s),
        max_level=int(args.max_level),
        force=bool(args.force),
    )

    scaling_rows: list[dict[str, Any]] = []
    finest_scaling_level = p2_mesh["last_success_level"]
    scaling_init_state = None
    if finest_scaling_level is not None:
        finest_mesh_row = next(
            row
            for row in p2_mesh["rows"]
            if int(row["level"]) == int(finest_scaling_level) and str(row["status"]) == "completed"
        )
        if int(finest_scaling_level) == P2_START_LEVEL:
            scaling_init_state = p2_init_state
        else:
            prev_level = int(finest_scaling_level) - 1
            prev_row = next(
                row
                for row in p2_mesh["rows"]
                if int(row["level"]) == prev_level and str(row["status"]) == "completed"
            )
            scaling_init_state = REPO_ROOT / str(prev_row["state_path"])
        scaling_rows = _run_strong_scaling(
            out_dir=args.out_dir,
            p=2.0,
            level=int(finest_scaling_level),
            init_state=Path(scaling_init_state),
            lambda1=p2_lambda1,
            lambda_source="exact",
            force=bool(args.force),
            baseline_row=finest_mesh_row,
        )

    rows = [*p2_mesh["rows"], *p3_mesh["rows"], *scaling_rows]
    summary = {
        "study": "plaplace_up_arctan_petsc",
        "suite_revision": SUITE_REVISION,
        "solver": "jax_petsc",
        "serial_summary": _repo_rel(args.serial_summary),
        "pmg_config": dict(PMG_CONFIG),
        "time_limit_s": float(args.time_limit_s),
        "max_level": int(args.max_level),
        "p2_mesh_stop_reason": str(p2_mesh["stop_reason"]),
        "p3_mesh_stop_reason": str(p3_mesh["stop_reason"]),
        "finest_scaling_level": None if finest_scaling_level is None else int(finest_scaling_level),
        "rows": rows,
        "status_counts": _status_counts(rows),
        "mesh_row_count": len(p2_mesh["rows"]) + len(p3_mesh["rows"]),
        "scaling_row_count": len(scaling_rows),
        "generated_case_count": len(rows),
    }
    _write_json(args.summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

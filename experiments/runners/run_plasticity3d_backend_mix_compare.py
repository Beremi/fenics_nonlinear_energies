#!/usr/bin/env python3
"""Run backend-mix Plasticity3D comparisons for `P4(L1), lambda=1.5`."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from time import perf_counter

from experiments.runners import (
    run_plasticity3d_p4_l1_lambda1p5_source_compare as source_env_tools,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
CASE_RUNNER = REPO_ROOT / "experiments" / "runners" / "run_plasticity3d_backend_mix_case.py"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "tmp" / "source_compare" / "slope_stability_petsc4py"
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "source_compare"
    / "plasticity3d_backend_mix"
)
SUMMARY_NAME = "comparison_summary.json"
RUNNER_NAME = "plasticity3d_backend_mix_compare"

ASSEMBLY_ORDER = ("local", "local_constitutiveAD", "source")
SOLVER_ORDER = ("local", "source")

NORMALIZED_ROW_KEYS = (
    "case_id",
    "assembly_backend",
    "solver_backend",
    "combo_label",
    "ranks",
    "status",
    "message",
    "solver_success",
    "exit_code",
    "wall_time_s",
    "solve_time_s",
    "nit",
    "linear_iterations_total",
    "final_metric",
    "final_metric_name",
    "energy",
    "omega",
    "u_max",
    "stdout_path",
    "stderr_path",
    "result_json",
    "case_dir",
    "command",
    "history_metric_name",
    "history_iterations",
    "history_metric",
    "initial_guess_enabled",
    "initial_guess_success",
    "initial_guess_ksp_iterations",
)


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tail_text(path: Path) -> str:
    if not path.exists():
        return ""
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    return lines[-1] if lines else ""


def _combo_label(assembly_backend: str, solver_backend: str) -> str:
    return f"{str(assembly_backend)} assembly + {str(solver_backend)} solver"


def _local_env() -> dict[str, str]:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["BLIS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"
    return env


def _mixed_env(source_root: Path) -> dict[str, str]:
    env = _local_env()
    env["FNE_REPO_ROOT"] = str(REPO_ROOT)
    repo_path = str(REPO_ROOT.resolve())
    source_path = str((source_root / "src").resolve())
    current = env.get("PYTHONPATH", "")
    parts = [repo_path, source_path]
    if current:
        parts.append(current)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _run_command(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    result_path: Path,
    output_grace_s: float = 20.0,
) -> tuple[int, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=stdout_fh,
            stderr=stderr_fh,
            text=True,
        )
        seen_output_at: float | None = None
        coerced_exit_code: int | None = None
        while True:
            return_code = proc.poll()
            if return_code is not None:
                proc.wait(timeout=1.0)
                exit_code = int(return_code)
                if coerced_exit_code is not None:
                    exit_code = int(coerced_exit_code)
                break
            if result_path.exists():
                if seen_output_at is None:
                    seen_output_at = perf_counter()
                elif perf_counter() - seen_output_at >= float(output_grace_s):
                    proc.terminate()
                    try:
                        proc.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=10.0)
                    coerced_exit_code = 0
                    exit_code = int(coerced_exit_code)
                    break
            time.sleep(1.0)
    return int(exit_code), float(perf_counter() - t0)


def _ensure_shared_source_env(source_root: Path) -> str:
    source_python, mode = source_env_tools.resolve_source_python(source_root)
    if source_python != PYTHON:
        raise RuntimeError(
            "Backend-mix runs require the shared repo `.venv` so both local and source "
            "packages are importable in one process. The source repo fell back to "
            f"`{source_python}`, mode={mode!r}."
        )
    return str(mode)


def _build_case_command(
    *,
    source_root: Path,
    case_dir: Path,
    result_path: Path,
    ranks: int,
    assembly_backend: str,
    solver_backend: str,
    stop_tol: float,
    maxit: int,
) -> list[str]:
    return [
        "mpiexec",
        "-n",
        str(ranks),
        str(PYTHON),
        "-u",
        str(CASE_RUNNER),
        "--assembly-backend",
        str(assembly_backend),
        "--solver-backend",
        str(solver_backend),
        "--source-root",
        str(source_root),
        "--out-dir",
        str(case_dir),
        "--output-json",
        str(result_path),
        "--stop-tol",
        str(stop_tol),
        "--maxit",
        str(maxit),
    ]


def _failed_row(
    *,
    case_id: str,
    assembly_backend: str,
    solver_backend: str,
    ranks: int,
    exit_code: int,
    message: str,
    case_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    result_path: Path,
    command: list[str],
) -> dict[str, object]:
    return {
        "case_id": str(case_id),
        "assembly_backend": str(assembly_backend),
        "solver_backend": str(solver_backend),
        "combo_label": _combo_label(assembly_backend, solver_backend),
        "ranks": int(ranks),
        "status": "failed",
        "message": str(message),
        "solver_success": False,
        "exit_code": int(exit_code),
        "wall_time_s": float("nan"),
        "solve_time_s": float("nan"),
        "nit": 0,
        "linear_iterations_total": 0,
        "final_metric": float("nan"),
        "final_metric_name": "",
        "energy": float("nan"),
        "omega": float("nan"),
        "u_max": float("nan"),
        "stdout_path": _repo_rel(stdout_path),
        "stderr_path": _repo_rel(stderr_path),
        "result_json": _repo_rel(result_path),
        "case_dir": _repo_rel(case_dir),
        "command": shlex.join(command),
        "history_metric_name": "",
        "history_iterations": [],
        "history_metric": [],
        "initial_guess_enabled": False,
        "initial_guess_success": False,
        "initial_guess_ksp_iterations": 0,
    }


def _normalize_payload(
    *,
    case_id: str,
    assembly_backend: str,
    solver_backend: str,
    ranks: int,
    exit_code: int,
    case_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    result_path: Path,
    command: list[str],
) -> dict[str, object]:
    payload = _read_json(result_path)
    history = [dict(item) for item in payload.get("history", [])]
    history_metric_name = str(payload.get("final_metric_name", "relative_correction"))
    history_iterations: list[int] = []
    history_metric: list[float] = []
    for item in history:
        if "it" in item:
            history_iterations.append(int(item.get("it", 0)))
        else:
            history_iterations.append(int(item.get("iteration", 0)))
        value = item.get("step_rel", item.get("metric", float("nan")))
        history_metric.append(float(value))

    initial_guess = dict(payload.get("initial_guess", {}))
    row = {
        "case_id": str(case_id),
        "assembly_backend": str(assembly_backend),
        "solver_backend": str(solver_backend),
        "combo_label": _combo_label(assembly_backend, solver_backend),
        "ranks": int(ranks),
        "status": str(payload.get("status", "failed")),
        "message": str(payload.get("message", "")),
        "solver_success": bool(payload.get("solver_success", False)),
        "exit_code": int(exit_code),
        "wall_time_s": float(payload.get("total_time", float("nan"))),
        "solve_time_s": float(payload.get("solve_time", float("nan"))),
        "nit": int(payload.get("nit", 0)),
        "linear_iterations_total": int(payload.get("linear_iterations_total", 0)),
        "final_metric": float(payload.get("final_metric", float("nan"))),
        "final_metric_name": str(history_metric_name),
        "energy": float(payload.get("energy", float("nan"))),
        "omega": float(payload.get("omega", float("nan"))),
        "u_max": float(payload.get("u_max", float("nan"))),
        "stdout_path": _repo_rel(stdout_path),
        "stderr_path": _repo_rel(stderr_path),
        "result_json": _repo_rel(result_path),
        "case_dir": _repo_rel(case_dir),
        "command": shlex.join(command),
        "history_metric_name": str(history_metric_name),
        "history_iterations": history_iterations,
        "history_metric": history_metric,
        "initial_guess_enabled": bool(initial_guess.get("enabled", False)),
        "initial_guess_success": bool(initial_guess.get("success", False)),
        "initial_guess_ksp_iterations": int(initial_guess.get("ksp_iterations", 0)),
    }
    return row


def _write_summary(
    *,
    summary_path: Path,
    rows_by_case: dict[str, dict[str, object]],
    source_root: Path,
    source_env_mode: str,
    ranks: int,
    stop_tol: float,
    maxit: int,
) -> None:
    order = {
        (solver_backend, assembly_backend): idx
        for idx, (solver_backend, assembly_backend) in enumerate(
            (
                ("local", "local"),
                ("local", "local_constitutiveAD"),
                ("local", "source"),
                ("source", "local"),
                ("source", "local_constitutiveAD"),
                ("source", "source"),
            )
        )
    }
    rows = sorted(
        rows_by_case.values(),
        key=lambda row: order.get(
            (str(row.get("solver_backend")), str(row.get("assembly_backend"))),
            10**6,
        ),
    )
    payload = {
        "runner": RUNNER_NAME,
        "source_root": _repo_rel(source_root),
        "source_env_mode": str(source_env_mode),
        "out_dir": _repo_rel(summary_path.parent),
        "ranks": int(ranks),
        "stop_metric_name": "relative_correction",
        "stop_tol": float(stop_tol),
        "maxit": int(maxit),
        "row_keys": list(NORMALIZED_ROW_KEYS),
        "rows": rows,
    }
    _write_json(summary_path, payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare local/source assembly and local/source solver backends."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--stop-tol", type=float, default=2.0e-3)
    parser.add_argument("--maxit", type=int, default=80)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / SUMMARY_NAME

    source_env_mode = _ensure_shared_source_env(source_root)
    env = _mixed_env(source_root)

    existing_rows: dict[str, dict[str, object]] = {}
    if bool(args.resume) and summary_path.exists():
        payload = _read_json(summary_path)
        for row in payload.get("rows", []):
            existing_rows[str(row.get("case_id", ""))] = dict(row)

    for solver_backend in SOLVER_ORDER:
        for assembly_backend in ASSEMBLY_ORDER:
            case_id = f"np{int(args.ranks)}:{assembly_backend}_assembly:{solver_backend}_solver"
            case_dir = (
                out_dir
                / "runs"
                / f"np{int(args.ranks)}"
                / f"solver_{solver_backend}"
                / f"assembly_{assembly_backend}"
            )
            stdout_path = case_dir / "stdout.txt"
            stderr_path = case_dir / "stderr.txt"
            result_path = case_dir / "output.json"

            existing = existing_rows.get(case_id)
            if (
                bool(args.resume)
                and existing is not None
                and str(existing.get("status", "")).startswith("completed")
                and result_path.exists()
            ):
                continue

            command = _build_case_command(
                source_root=source_root,
                case_dir=case_dir,
                result_path=result_path,
                ranks=int(args.ranks),
                assembly_backend=assembly_backend,
                solver_backend=solver_backend,
                stop_tol=float(args.stop_tol),
                maxit=int(args.maxit),
            )
            exit_code, _wall_time = _run_command(
                cmd=command,
                cwd=REPO_ROOT,
                env=env,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                result_path=result_path,
            )

            if exit_code != 0 or not result_path.exists():
                message = _tail_text(stderr_path) or _tail_text(stdout_path) or "subprocess failed"
                row = _failed_row(
                    case_id=case_id,
                    assembly_backend=assembly_backend,
                    solver_backend=solver_backend,
                    ranks=int(args.ranks),
                    exit_code=exit_code,
                    message=message,
                    case_dir=case_dir,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    result_path=result_path,
                    command=command,
                )
            else:
                row = _normalize_payload(
                    case_id=case_id,
                    assembly_backend=assembly_backend,
                    solver_backend=solver_backend,
                    ranks=int(args.ranks),
                    exit_code=exit_code,
                    case_dir=case_dir,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    result_path=result_path,
                    command=command,
                )

            existing_rows[case_id] = row
            _write_summary(
                summary_path=summary_path,
                rows_by_case=existing_rows,
                source_root=source_root,
                source_env_mode=source_env_mode,
                ranks=int(args.ranks),
                stop_tol=float(args.stop_tol),
                maxit=int(args.maxit),
            )

    print(summary_path)


if __name__ == "__main__":
    main()

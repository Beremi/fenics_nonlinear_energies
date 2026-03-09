#!/usr/bin/env python3
"""Run the final HE PETSc-backed benchmark suite with skip and summary logic."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path


SOLVERS = (
    {
        "name": "fenics_custom",
        "backend": "fenics",
        "nproc_threads": 1,
    },
    {
        "name": "jax_petsc_element",
        "backend": "element",
        "nproc_threads": 1,
    },
)


def _child_preexec() -> None:
    """Start the launched MPI case in its own session and tie it to this parent."""
    os.setsid()
    libc_path = ctypes.util.find_library("c")
    if not libc_path:
        return
    libc = ctypes.CDLL(libc_path, use_errno=True)
    pr_set_pdeathsig = 1
    if libc.prctl(pr_set_pdeathsig, signal.SIGTERM, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def _terminate_process_group(proc: subprocess.Popen[str], grace_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=[s["name"] for s in SOLVERS],
        choices=[s["name"] for s in SOLVERS],
    )
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--nprocs", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--total-steps", nargs="+", type=int, default=[96, 24])
    parser.add_argument("--step-time-limit-s", type=float, default=100.0)
    parser.add_argument("--trust-radius-init", type=float, default=2.0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiment_results_cache/he_final_suite_r2_0",
    )
    parser.add_argument(
        "--max-case-wall-s",
        type=float,
        default=3600.0,
        help="Hard wall timeout for a whole case, independent of the per-step limit.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from an existing summary.json in --out-dir when present.",
    )
    return parser


def _solver_config(name: str) -> dict:
    for solver in SOLVERS:
        if solver["name"] == name:
            return dict(solver)
    raise KeyError(name)


def _case_key(row: dict) -> tuple[str, int, int, int]:
    return (row["solver"], int(row["total_steps"]), int(row["level"]), int(row["nprocs"]))


def _is_harder_or_equal(failure: dict, solver: str, total_steps: int, level: int, nprocs: int) -> bool:
    if failure["solver"] != solver:
        return False
    if int(failure["nprocs"]) != int(nprocs):
        return False
    if int(failure["level"]) > int(level):
        return False
    # Fewer steps is harder, so a failure at 96 should block 24 at the same
    # rank/level, but not the other way around.
    if int(failure["total_steps"]) > int(total_steps):
        return False
    return True


def _case_blocked(failures: list[dict], solver: str, total_steps: int, level: int, nprocs: int) -> bool:
    return any(_is_harder_or_equal(f, solver, total_steps, level, nprocs) for f in failures)


def _sum_step_linear(step: dict) -> int:
    return int(sum(int(rec["ksp_its"]) for rec in step.get("linear_timing", [])))


def _sum_step_linear_time(step: dict, field: str) -> float:
    return float(sum(float(rec.get(field, 0.0)) for rec in step.get("linear_timing", [])))


def _sum_step_history(step: dict, field: str) -> float:
    return float(sum(float(rec.get(field, 0.0)) for rec in step.get("history", [])))


def _summarize_case(solver_name: str, total_steps: int, level: int, nprocs: int, payload: dict) -> dict:
    result = payload["result"]
    steps = list(result.get("steps", []))
    completed_steps = len(steps)
    total_newton = int(sum(int(step.get("nit", step.get("iters", 0))) for step in steps))
    total_linear = int(sum(_sum_step_linear(step) for step in steps))
    total_time = float(result.get("solve_time_total", 0.0))
    max_step_time = max((float(step.get("time", 0.0)) for step in steps), default=0.0)
    mean_step_time = total_time / completed_steps if completed_steps else None
    total_assembly = sum(_sum_step_linear_time(step, "assemble_total_time") for step in steps)
    if total_assembly == 0.0:
        total_assembly = sum(_sum_step_linear_time(step, "assemble_time") for step in steps)
    total_pc_setup = sum(_sum_step_linear_time(step, "pc_setup_time") for step in steps)
    total_ksp_solve = sum(_sum_step_linear_time(step, "solve_time") for step in steps)
    total_line_search = sum(_sum_step_history(step, "t_ls") for step in steps)
    total_tr_rejects = int(
        sum(int(rec.get("trust_rejects", 0)) for step in steps for rec in step.get("history", []))
    )
    final_energy = float(steps[-1]["energy"]) if steps else None
    failure_mode = None
    failure_time_s = None
    first_failed_step = None
    case_result = "completed"
    for step in steps:
        if step.get("kill_switch_exceeded"):
            first_failed_step = int(step["step"])
            failure_mode = "kill-switch"
            failure_time_s = float(step.get("time", 0.0))
            case_result = "kill-switch"
            completed_steps = max(first_failed_step - 1, 0)
            break
    if case_result == "completed" and steps:
        last_msg = str(steps[-1].get("message", ""))
        if "converged" not in last_msg.lower() or completed_steps < int(total_steps):
            first_failed_step = int(steps[-1]["step"]) + (0 if completed_steps < int(total_steps) else 1)
            failure_mode = last_msg or "stopped"
            case_result = "failed"
    if not steps:
        case_result = "failed"
        failure_mode = "no-steps"
        first_failed_step = 1

    return {
        "solver": solver_name,
        "backend": payload["case"]["backend"],
        "total_steps": int(total_steps),
        "level": int(level),
        "nprocs": int(nprocs),
        "completed_steps": int(completed_steps),
        "first_failed_step": first_failed_step,
        "failure_mode": failure_mode,
        "failure_time_s": failure_time_s,
        "total_newton_iters": total_newton,
        "total_linear_iters": total_linear,
        "total_time_s": total_time,
        "mean_step_time_s": mean_step_time,
        "max_step_time_s": max_step_time,
        "assembly_time_s": total_assembly,
        "pc_init_time_s": total_pc_setup,
        "ksp_solve_time_s": total_ksp_solve,
        "line_search_time_s": total_line_search,
        "trust_rejects": total_tr_rejects,
        "final_energy": final_energy,
        "result": case_result,
    }


def _write_summary_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# HE Final Suite Summary",
        "",
        "Data note:",
        "- This file is only the quick-reference index.",
        "- Full aggregated data are in `summary.json` in the same directory.",
        "- Full per-case data are in `*_steps*_l*_np*.json` and matching `*.log` files.",
        "- Each per-case JSON stores per-step data in `result.steps`, with per-Newton details in `history` and per-Newton linear timing in `linear_timing`.",
        "",
        "| Solver | Total steps | Level | MPI | Completed steps | Total Newton | Total linear | Total time [s] | Mean step [s] | Max step [s] | Result |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        def fmt(x):
            if x is None:
                return "-"
            if isinstance(x, float):
                return f"{x:.3f}"
            return str(x)
        lines.append(
            "| {solver} | {steps} | {level} | {nprocs} | {completed} | {nit} | {lit} | {tt} | {mean} | {maxs} | {result} |".format(
                solver=row["solver"],
                steps=row["total_steps"],
                level=row["level"],
                nprocs=row["nprocs"],
                completed=row["completed_steps"],
                nit=fmt(row["total_newton_iters"]),
                lit=fmt(row["total_linear_iters"]),
                tt=fmt(row["total_time_s"]),
                mean=fmt(row["mean_step_time_s"]),
                maxs=fmt(row["max_step_time_s"]),
                result=row["result"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_case(repo_root: Path, out_dir: Path, solver: dict, total_steps: int, level: int, nprocs: int, args) -> dict:
    case_name = f"{solver['name']}_steps{total_steps}_l{level}_np{nprocs}"
    case_json = out_dir / f"{case_name}.json"
    case_log = out_dir / f"{case_name}.log"
    case_cmd = [
        "mpirun", "-n", str(nprocs),
        "python", "-u", "experiment_scripts/run_trust_region_case.py",
        "--problem", "he",
        "--backend", solver["backend"],
        "--level", str(level),
        "--steps", str(total_steps),
        "--start-step", "1",
        "--total-steps", str(total_steps),
        "--profile", "performance",
        "--ksp-type", "gmres",
        "--pc-type", "gamg",
        "--ksp-rtol", "1e-1",
        "--ksp-max-it", "30",
        "--gamg-threshold", "0.05",
        "--gamg-agg-nsmooths", "1",
        "--pc-setup-on-ksp-cap",
        "--gamg-set-coordinates",
        "--use-near-nullspace",
        "--tolf", "1e-4",
        "--tolg", "1e-3",
        "--tolg-rel", "1e-3",
        "--tolx-rel", "1e-3",
        "--tolx-abs", "1e-10",
        "--maxit", "100",
        "--linesearch-a", "-0.5",
        "--linesearch-b", "2.0",
        "--use-trust-region",
        "--trust-radius-init", str(args.trust_radius_init),
        "--trust-radius-min", "1e-8",
        "--trust-radius-max", "1e6",
        "--trust-shrink", "0.5",
        "--trust-expand", "1.5",
        "--trust-eta-shrink", "0.05",
        "--trust-eta-expand", "0.75",
        "--trust-max-reject", "6",
        "--save-history",
        "--save-linear-timing",
        "--quiet",
        "--out", str(case_json),
    ]
    if getattr(args, "step_time_limit_s", None) is not None:
        case_cmd += ["--step-time-limit-s", str(args.step_time_limit_s)]
    if solver["backend"] == "element":
        case_cmd += [
            "--local-coloring",
            "--nproc-threads",
            str(solver["nproc_threads"]),
            "--element-reorder-mode",
            "block_xyz",
            "--local-hessian-mode",
            "element",
        ]
    shell_cmd = "source local_env/activate.sh >/dev/null && exec " + shlex.join(case_cmd)
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        ["bash", "-lc", shell_cmd],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=_child_preexec,
    )
    timed_out = False
    try:
        stdout, _ = proc.communicate(timeout=float(args.max_case_wall_s))
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or ""
        _terminate_process_group(proc)
    finally:
        if proc.poll() is None:
            _terminate_process_group(proc)
    wall = time.perf_counter() - t0
    case_log.write_text(stdout, encoding="utf-8")
    if timed_out and not case_json.exists():
        return {
            "solver": solver["name"],
            "backend": solver["backend"],
            "total_steps": int(total_steps),
            "level": int(level),
            "nprocs": int(nprocs),
            "completed_steps": 0,
            "first_failed_step": 1,
            "failure_mode": "case-timeout",
            "failure_time_s": wall,
            "total_newton_iters": None,
            "total_linear_iters": None,
            "total_time_s": None,
            "mean_step_time_s": None,
            "max_step_time_s": None,
            "assembly_time_s": None,
            "pc_init_time_s": None,
            "ksp_solve_time_s": None,
            "line_search_time_s": None,
            "trust_rejects": None,
            "final_energy": None,
            "result": "failed",
            "json_path": str(case_json),
            "log_path": str(case_log),
        }
    if not case_json.exists():
        return {
            "solver": solver["name"],
            "backend": solver["backend"],
            "total_steps": int(total_steps),
            "level": int(level),
            "nprocs": int(nprocs),
            "completed_steps": 0,
            "first_failed_step": 1,
            "failure_mode": f"exit-{proc.returncode}",
            "failure_time_s": wall,
            "total_newton_iters": None,
            "total_linear_iters": None,
            "total_time_s": None,
            "mean_step_time_s": None,
            "max_step_time_s": None,
            "assembly_time_s": None,
            "pc_init_time_s": None,
            "ksp_solve_time_s": None,
            "line_search_time_s": None,
            "trust_rejects": None,
            "final_energy": None,
            "result": "failed",
            "json_path": str(case_json),
            "log_path": str(case_log),
        }
    payload = json.load(case_json.open(encoding="utf-8"))
    row = _summarize_case(solver["name"], total_steps, level, nprocs, payload)
    row["json_path"] = str(case_json)
    row["log_path"] = str(case_log)
    return row


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    solvers = [_solver_config(name) for name in args.solvers]
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    rows: list[dict] = []
    failures: list[dict] = []
    done_keys: set[tuple[str, int, int, int]] = set()

    if args.resume and summary_json.exists():
        with summary_json.open(encoding="utf-8") as f:
            payload = json.load(f)
        rows = list(payload.get("rows", []))
        failures = [row for row in rows if row.get("result") != "completed"]
        done_keys = {_case_key(row) for row in rows}

    for solver in solvers:
        for total_steps in args.total_steps:
            for level in args.levels:
                for nprocs in args.nprocs:
                    case_key = (solver["name"], int(total_steps), int(level), int(nprocs))
                    if case_key in done_keys:
                        continue
                    if _case_blocked(failures, solver["name"], total_steps, level, nprocs):
                        rows.append(
                            {
                                "solver": solver["name"],
                                "backend": solver["backend"],
                                "total_steps": int(total_steps),
                                "level": int(level),
                                "nprocs": int(nprocs),
                                "completed_steps": 0,
                                "first_failed_step": None,
                                "failure_mode": "skipped-harder-case",
                                "failure_time_s": None,
                                "total_newton_iters": None,
                                "total_linear_iters": None,
                                "total_time_s": None,
                                "mean_step_time_s": None,
                                "max_step_time_s": None,
                                "assembly_time_s": None,
                                "pc_init_time_s": None,
                                "ksp_solve_time_s": None,
                                "line_search_time_s": None,
                                "trust_rejects": None,
                                "final_energy": None,
                                "result": "skipped",
                            }
                        )
                        done_keys.add(case_key)
                        continue
                    row = _run_case(repo_root, out_dir, solver, total_steps, level, nprocs, args)
                    rows.append(row)
                    done_keys.add(case_key)
                    if row["result"] != "completed":
                        failures.append(row)
                    with summary_json.open("w", encoding="utf-8") as f:
                        json.dump({"rows": rows}, f, indent=2)
                    _write_summary_markdown(summary_md, rows)
                    print(json.dumps(row, indent=2), flush=True)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)
    _write_summary_markdown(summary_md, rows)
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()

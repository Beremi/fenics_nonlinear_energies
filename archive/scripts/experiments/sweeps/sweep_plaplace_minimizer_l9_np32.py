#!/usr/bin/env python3
"""Sweep pLaplace minimizer settings on the finest mesh benchmark."""

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
    {"name": "fenics_custom", "backend": "fenics"},
    {
        "name": "jax_petsc_element",
        "backend": "element",
        "local_hessian_mode": "element",
    },
)


CASES = (
    {
        "name": "ls_ref",
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-3,
        "ksp_max_it": 200,
        "linesearch_tol": 1e-3,
        "use_trust_region": False,
    },
    {
        "name": "ls_loose",
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "linesearch_tol": 1e-1,
        "use_trust_region": False,
    },
    {
        "name": "tr_stcg_r0_5",
        "ksp_type": "stcg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "linesearch_tol": 1e-1,
        "use_trust_region": True,
        "trust_radius_init": 0.5,
        "trust_subproblem_line_search": True,
    },
    {
        "name": "tr_stcg_r1_0",
        "ksp_type": "stcg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "linesearch_tol": 1e-1,
        "use_trust_region": True,
        "trust_radius_init": 1.0,
        "trust_subproblem_line_search": True,
    },
    {
        "name": "tr_stcg_r2_0",
        "ksp_type": "stcg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "linesearch_tol": 1e-1,
        "use_trust_region": True,
        "trust_radius_init": 2.0,
        "trust_subproblem_line_search": True,
    },
)


def _child_preexec() -> None:
    os.setsid()
    libc_path = ctypes.util.find_library("c")
    if not libc_path:
        return
    libc = ctypes.CDLL(libc_path, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
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
    parser.add_argument("--level", type=int, default=9)
    parser.add_argument("--nprocs", type=int, default=32)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiment_results_cache/plaplace_minimizer_sweep_l9_np32",
    )
    parser.add_argument("--max-case-wall-s", type=float, default=7200.0)
    parser.add_argument("--profile", type=str, default="performance")
    parser.add_argument("--nproc-threads", type=int, default=1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _case_key(row: dict) -> tuple[str, str]:
    return row["solver"], row["config"]


def _case_name(solver_name: str, config_name: str) -> str:
    return f"{solver_name}_{config_name}"


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _summarize(payload: dict, solver_name: str, config_name: str) -> dict:
    result = payload["result"]
    steps = list(result.get("steps", []))
    step = steps[0] if steps else {}
    linear = int(sum(int(rec.get("ksp_its", 0)) for rec in step.get("linear_timing", [])))
    message = str(step.get("message", ""))
    return {
        "solver": solver_name,
        "config": config_name,
        "result": "completed" if "converged" in message.lower() else "failed",
        "total_time_s": float(result.get("solve_time_total", result.get("total_time", 0.0))),
        "newton_iters": int(step.get("nit", 0)),
        "linear_iters": linear,
        "final_energy": float(step.get("energy", float("nan"))),
        "message": message,
    }


def _write_summary(path: Path, rows: list[dict]) -> None:
    lines = [
        "# pLaplace Minimizer Sweep Summary",
        "",
        "| Solver | Config | Total time [s] | Newton | Linear | Final energy | Result |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {solver} | {config} | {tt} | {nit} | {lit} | {energy} | {result} |".format(
                solver=row["solver"],
                config=row["config"],
                tt=_fmt(row["total_time_s"]),
                nit=_fmt(row["newton_iters"]),
                lit=_fmt(row["linear_iters"]),
                energy=_fmt(row["final_energy"], 6),
                result=row["result"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"

    rows: list[dict] = []
    done: set[tuple[str, str]] = set()
    if args.resume and summary_json.exists():
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        rows = list(payload.get("rows", []))
        done = {_case_key(row) for row in rows}

    for solver in SOLVERS:
        for config in CASES:
            key = (solver["name"], config["name"])
            if key in done:
                continue
            case_name = _case_name(solver["name"], config["name"])
            case_json = out_dir / f"{case_name}.json"
            case_log = out_dir / f"{case_name}.log"

            case_cmd = [
                "mpirun",
                "-n",
                str(args.nprocs),
                "python",
                "-u",
                "experiment_scripts/run_trust_region_case.py",
                "--problem",
                "plaplace",
                "--backend",
                solver["backend"],
                "--level",
                str(args.level),
                "--profile",
                str(args.profile),
                "--ksp-type",
                str(config["ksp_type"]),
                "--pc-type",
                str(config["pc_type"]),
                "--ksp-rtol",
                str(config["ksp_rtol"]),
                "--ksp-max-it",
                str(config["ksp_max_it"]),
                "--gamg-threshold",
                "0.05",
                "--gamg-agg-nsmooths",
                "1",
                "--gamg-set-coordinates",
                "--tolf",
                "1e-4",
                "--tolg",
                "1e-3",
                "--tolg-rel",
                "1e-3",
                "--tolx-rel",
                "1e-3",
                "--tolx-abs",
                "1e-10",
                "--maxit",
                "100",
                "--linesearch-a",
                "-0.5",
                "--linesearch-b",
                "2.0",
                "--linesearch-tol",
                str(config["linesearch_tol"]),
                "--use-trust-region" if config.get("use_trust_region", False) else "--no-use-trust-region",
                "--trust-radius-init",
                str(config.get("trust_radius_init", 1.0)),
                "--trust-radius-min",
                "1e-8",
                "--trust-radius-max",
                "1e6",
                "--trust-shrink",
                "0.5",
                "--trust-expand",
                "1.5",
                "--trust-eta-shrink",
                "0.05",
                "--trust-eta-expand",
                "0.75",
                "--trust-max-reject",
                "6",
                "--trust-subproblem-line-search"
                if config.get("trust_subproblem_line_search", False)
                else "--no-trust-subproblem-line-search",
                "--save-history",
                "--save-linear-timing",
                "--quiet",
                "--out",
                str(case_json),
            ]
            if solver["backend"] == "element":
                case_cmd += [
                    "--local-coloring",
                    "--nproc-threads",
                    str(args.nproc_threads),
                    "--element-reorder-mode",
                    "block_xyz",
                    "--local-hessian-mode",
                    str(solver["local_hessian_mode"]),
                ]

            shell_cmd = "source local_env/activate.sh >/dev/null && exec " + shlex.join(case_cmd)
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
            case_log.write_text(stdout, encoding="utf-8")

            if timed_out or not case_json.exists():
                row = {
                    "solver": solver["name"],
                    "config": config["name"],
                    "result": "failed",
                    "total_time_s": None,
                    "newton_iters": None,
                    "linear_iters": None,
                    "final_energy": None,
                    "message": "timeout" if timed_out else f"missing-json exit={proc.returncode}",
                }
            else:
                payload = json.loads(case_json.read_text(encoding="utf-8"))
                row = _summarize(payload, solver["name"], config["name"])

            rows.append(row)
            rows.sort(key=_case_key)
            done.add(key)
            summary_json.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
            _write_summary(summary_md, rows)
            print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()

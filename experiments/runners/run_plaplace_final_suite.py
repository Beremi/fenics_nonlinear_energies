#!/usr/bin/env python3
"""Run the final pLaplace benchmark suite with frozen settings."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import math
import os
import shlex
import signal
import subprocess
import time
from itertools import zip_longest
from pathlib import Path

from src.core.benchmark.results import (
    summarize_single_step_case as _summarize_case,
    sum_step_history as _sum_step_history,
    sum_step_linear as _sum_step_linear,
    sum_step_linear_time as _sum_step_linear_time,
)


SOLVERS = (
    {
        "name": "fenics_custom",
        "backend": "fenics",
    },
    {
        "name": "jax_petsc_element",
        "backend": "element",
        "local_hessian_mode": "element",
    },
    {
        "name": "jax_petsc_local_sfd",
        "backend": "element",
        "local_hessian_mode": "sfd_local",
    },
)


def _child_preexec() -> None:
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


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())
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
    parser.add_argument("--levels", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument("--nprocs", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--max-case-wall-s", type=float, default=7200.0)
    parser.add_argument("--step-time-limit-s", type=float, default=None)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/raw_results/plaplace_final_suite",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--write-case-markdown",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--profile", type=str, default="reference")
    parser.add_argument("--ksp-type", type=str, default="cg")
    parser.add_argument("--pc-type", type=str, default="hypre")
    parser.add_argument("--ksp-rtol", type=float, default=1e-1)
    parser.add_argument("--ksp-max-it", type=int, default=30)
    parser.add_argument("--gamg-threshold", type=float, default=0.05)
    parser.add_argument("--gamg-agg-nsmooths", type=int, default=1)
    parser.add_argument(
        "--gamg-set-coordinates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pc-setup-on-ksp-cap",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--tolf", type=float, default=1e-4)
    parser.add_argument("--tolg", type=float, default=1e-3)
    parser.add_argument("--tolg-rel", type=float, default=1e-3)
    parser.add_argument("--tolx-rel", type=float, default=1e-3)
    parser.add_argument("--tolx-abs", type=float, default=1e-10)
    parser.add_argument("--maxit", type=int, default=100)

    parser.add_argument("--linesearch-a", type=float, default=-0.5)
    parser.add_argument("--linesearch-b", type=float, default=2.0)
    parser.add_argument("--linesearch-tol", type=float, default=1e-1)
    parser.add_argument(
        "--use-trust-region",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--trust-radius-init", type=float, default=1.0)
    parser.add_argument("--trust-radius-min", type=float, default=1e-8)
    parser.add_argument("--trust-radius-max", type=float, default=1e6)
    parser.add_argument("--trust-shrink", type=float, default=0.5)
    parser.add_argument("--trust-expand", type=float, default=1.5)
    parser.add_argument("--trust-eta-shrink", type=float, default=0.05)
    parser.add_argument("--trust-eta-expand", type=float, default=0.75)
    parser.add_argument("--trust-max-reject", type=int, default=6)
    parser.add_argument(
        "--trust-subproblem-line-search",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--retry-on-failure",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--nproc-threads", type=int, default=1)
    parser.add_argument("--element-reorder-mode", type=str, default="block_xyz")
    parser.add_argument(
        "--local-coloring",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _solver_config(name: str) -> dict:
    for solver in SOLVERS:
        if solver["name"] == name:
            return dict(solver)
    raise KeyError(name)


def _case_key(row: dict) -> tuple[str, int, int]:
    return (row["solver"], int(row["level"]), int(row["nprocs"]))


def _case_name(solver_name: str, level: int, nprocs: int) -> str:
    return f"{solver_name}_l{level}_np{nprocs}"


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.{digits}f}"
    return str(value)


def _assemble_time(rec: dict) -> float:
    return float(rec.get("assemble_total_time", rec.get("assemble_time", 0.0)))


def _write_case_markdown(case_name: str, md_path: Path, payload: dict, row: dict, repo_root: Path) -> None:
    case = payload["case"]
    result = payload["result"]
    steps = list(result.get("steps", []))

    lines = [
        f"# {case_name}",
        "",
        "## Run Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Solver | `{row['solver']}` |",
        f"| Backend | `{row['backend']}` |",
        f"| Mesh level | `{row['level']}` |",
        f"| MPI ranks | `{row['nprocs']}` |",
        f"| Result | `{row['result']}` |",
        f"| Failure mode | `{row.get('failure_mode') or '-'}` |",
        f"| Total DOFs | `{result.get('total_dofs', '-')}` |",
        f"| Free DOFs | `{result.get('free_dofs', '-')}` |",
        f"| Setup time [s] | `{_fmt(result.get('setup_time'))}` |",
        f"| Total solve time [s] | `{_fmt(result.get('solve_time_total', result.get('total_time')))}` |",
        f"| Total Newton iterations | `{row['total_newton_iters']}` |",
        f"| Total linear iterations | `{row['total_linear_iters']}` |",
        f"| Total assembly time [s] | `{_fmt(row['assembly_time_s'])}` |",
        f"| Total PC init time [s] | `{_fmt(row['pc_init_time_s'])}` |",
        f"| Total KSP solve time [s] | `{_fmt(row['ksp_solve_time_s'])}` |",
        f"| Total line-search time [s] | `{_fmt(row['line_search_time_s'])}` |",
        f"| Final energy | `{_fmt(row['final_energy'], 6)}` |",
        f"| Raw JSON | `{_display_path(md_path.with_suffix('.json'), repo_root)}` |",
        f"| Raw log | `{_display_path(md_path.with_suffix('.log'), repo_root)}` |",
        "",
        "## Frozen Settings",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| `ksp_type` | `{case.get('ksp_type')}` |",
        f"| `pc_type` | `{case.get('pc_type')}` |",
        f"| `ksp_rtol` | `{case.get('ksp_rtol')}` |",
        f"| `ksp_max_it` | `{case.get('ksp_max_it')}` |",
        f"| `use_trust_region` | `{case.get('use_trust_region')}` |",
        f"| `trust_subproblem_line_search` | `{case.get('trust_subproblem_line_search')}` |",
        f"| `linesearch_interval` | `[{case.get('linesearch_a')}, {case.get('linesearch_b')}]` |",
        f"| `linesearch_tol` | `{case.get('linesearch_tol')}` |",
        f"| `trust_radius_init` | `{case.get('trust_radius_init')}` |",
        f"| `local_hessian_mode` | `{case.get('local_hessian_mode') or '-'}` |",
        "",
        "## Step Summary",
        "",
        "| Step | Time [s] | Newton | Linear | Energy | Message |",
        "|---:|---:|---:|---:|---:|---|",
    ]

    for step in steps:
        lines.append(
            "| {step} | {time} | {nit} | {linear} | {energy} | {msg} |".format(
                step=step.get("step"),
                time=_fmt(step.get("time")),
                nit=step.get("nit", step.get("iters", "-")),
                linear=_sum_step_linear(step),
                energy=_fmt(step.get("energy"), 6),
                msg=str(step.get("message", "")).replace("|", "\\|"),
            )
        )

    for step in steps:
        lines.extend(
            [
                "",
                f"## Step {step.get('step')}",
                "",
                "| Field | Value |",
                "|---|---|",
                f"| Step time [s] | `{_fmt(step.get('time'))}` |",
                f"| Newton iterations | `{step.get('nit', step.get('iters', '-'))}` |",
                f"| Linear iterations | `{_sum_step_linear(step)}` |",
                f"| Energy | `{_fmt(step.get('energy'), 6)}` |",
                f"| Message | `{step.get('message', '')}` |",
                "",
                "| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |",
                "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        history = step.get("history", [])
        linear_timing = step.get("linear_timing", [])
        for hist, lin in zip_longest(history, linear_timing, fillvalue=None):
            hist = hist or {}
            lin = lin or {}
            lines.append(
                "| {it} | {energy} | {dE} | {grad} | {grad_post} | {alpha} | {ksp} | {accepted} | {ls_evals} | {assemble} | {pc} | {solve} | {t_ls} | {radius} | {rho} |".format(
                    it=hist.get("it", "-"),
                    energy=_fmt(hist.get("energy"), 6),
                    dE=_fmt(hist.get("dE"), 6),
                    grad=_fmt(hist.get("grad_norm"), 6),
                    grad_post=_fmt(hist.get("grad_norm_post"), 6),
                    alpha=_fmt(hist.get("alpha"), 6),
                    ksp=hist.get("ksp_its", lin.get("ksp_its", "-")),
                    accepted="yes" if hist.get("accepted_step") else "no",
                    ls_evals=hist.get("ls_evals", "-"),
                    assemble=_fmt(_assemble_time(lin)),
                    pc=_fmt(lin.get("pc_setup_time")),
                    solve=_fmt(lin.get("solve_time")),
                    t_ls=_fmt(hist.get("t_ls")),
                    radius=_fmt(hist.get("trust_radius"), 6),
                    rho=_fmt(hist.get("trust_ratio"), 6),
                )
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_markdown(path: Path, rows: list[dict], repo_root: Path) -> None:
    lines = [
        "# pLaplace Final Suite Summary",
        "",
        "Data note:",
        "- This file is the quick-reference whole-run index.",
        "- Full aggregated data are in `summary.json` in the same directory.",
        "- Full per-case raw data are in `*_l*_np*.json` and matching `*.log` files.",
        "- Human-readable per-case reports are in matching `*.md` files.",
        "",
        "| Solver | Level | MPI | Newton | Linear | Total time [s] | Assembly [s] | PC init [s] | KSP solve [s] | Result | JSON | MD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        json_rel = _display_path(Path(row["json_path"]), repo_root)
        md_rel = _display_path(Path(row["md_path"]), repo_root)
        lines.append(
            "| {solver} | {level} | {nprocs} | {nit} | {lit} | {tt} | {asm} | {pc} | {ksp} | {result} | `{json_rel}` | `{md_rel}` |".format(
                solver=row["solver"],
                level=row["level"],
                nprocs=row["nprocs"],
                nit=_fmt(row["total_newton_iters"]),
                lit=_fmt(row["total_linear_iters"]),
                tt=_fmt(row["total_time_s"]),
                asm=_fmt(row["assembly_time_s"]),
                pc=_fmt(row["pc_init_time_s"]),
                ksp=_fmt(row["ksp_solve_time_s"]),
                result=row["result"],
                json_rel=json_rel,
                md_rel=md_rel,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_json(path: Path, rows: list[dict], args: argparse.Namespace) -> None:
    payload = {
        "suite": {
            "name": "plaplace_final_suite",
            "settings": {
                "profile": args.profile,
                "ksp_type": args.ksp_type,
                "pc_type": args.pc_type,
                "ksp_rtol": args.ksp_rtol,
                "ksp_max_it": args.ksp_max_it,
                "gamg_threshold": args.gamg_threshold,
                "gamg_agg_nsmooths": args.gamg_agg_nsmooths,
                "gamg_set_coordinates": args.gamg_set_coordinates,
                "pc_setup_on_ksp_cap": args.pc_setup_on_ksp_cap,
                "tolf": args.tolf,
                "tolg": args.tolg,
                "tolg_rel": args.tolg_rel,
                "tolx_rel": args.tolx_rel,
                "tolx_abs": args.tolx_abs,
                "maxit": args.maxit,
                "linesearch_a": args.linesearch_a,
                "linesearch_b": args.linesearch_b,
                "linesearch_tol": args.linesearch_tol,
                "use_trust_region": args.use_trust_region,
                "trust_radius_init": args.trust_radius_init,
                "trust_radius_min": args.trust_radius_min,
                "trust_radius_max": args.trust_radius_max,
                "trust_shrink": args.trust_shrink,
                "trust_expand": args.trust_expand,
                "trust_eta_shrink": args.trust_eta_shrink,
                "trust_eta_expand": args.trust_eta_expand,
                "trust_max_reject": args.trust_max_reject,
                "trust_subproblem_line_search": args.trust_subproblem_line_search,
                "retry_on_failure": args.retry_on_failure,
                "step_time_limit_s": args.step_time_limit_s,
                "element_reorder_mode": args.element_reorder_mode,
                "local_coloring": args.local_coloring,
                "nproc_threads": args.nproc_threads,
            },
        },
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_case(repo_root: Path, out_dir: Path, solver: dict, level: int, nprocs: int, args) -> dict:
    case_name = _case_name(solver["name"], level, nprocs)
    case_json = out_dir / f"{case_name}.json"
    case_log = out_dir / f"{case_name}.log"
    case_md = out_dir / f"{case_name}.md"

    python_exe = str(repo_root / ".venv" / "bin" / "python")
    case_cmd = [
        "mpirun",
        "-n",
        str(nprocs),
        python_exe,
        "-u",
        "experiments/runners/run_trust_region_case.py",
        "--problem",
        "plaplace",
        "--backend",
        solver["backend"],
        "--level",
        str(level),
        "--profile",
        str(args.profile),
        "--ksp-type",
        str(args.ksp_type),
        "--pc-type",
        str(args.pc_type),
        "--ksp-rtol",
        str(args.ksp_rtol),
        "--ksp-max-it",
        str(args.ksp_max_it),
        "--gamg-threshold",
        str(args.gamg_threshold),
        "--gamg-agg-nsmooths",
        str(args.gamg_agg_nsmooths),
        "--gamg-set-coordinates" if args.gamg_set_coordinates else "--no-gamg-set-coordinates",
        "--pc-setup-on-ksp-cap" if args.pc_setup_on_ksp_cap else "--no-pc-setup-on-ksp-cap",
        "--tolf",
        str(args.tolf),
        "--tolg",
        str(args.tolg),
        "--tolg-rel",
        str(args.tolg_rel),
        "--tolx-rel",
        str(args.tolx_rel),
        "--tolx-abs",
        str(args.tolx_abs),
        "--maxit",
        str(args.maxit),
        "--linesearch-a",
        str(args.linesearch_a),
        "--linesearch-b",
        str(args.linesearch_b),
        "--linesearch-tol",
        str(args.linesearch_tol),
        "--use-trust-region" if args.use_trust_region else "--no-use-trust-region",
        "--trust-radius-init",
        str(args.trust_radius_init),
        "--trust-radius-min",
        str(args.trust_radius_min),
        "--trust-radius-max",
        str(args.trust_radius_max),
        "--trust-shrink",
        str(args.trust_shrink),
        "--trust-expand",
        str(args.trust_expand),
        "--trust-eta-shrink",
        str(args.trust_eta_shrink),
        "--trust-eta-expand",
        str(args.trust_eta_expand),
        "--trust-max-reject",
        str(args.trust_max_reject),
        "--trust-subproblem-line-search"
        if args.trust_subproblem_line_search
        else "--no-trust-subproblem-line-search",
        "--retry-on-failure" if args.retry_on_failure else "--no-retry-on-failure",
        "--save-history",
        "--save-linear-timing",
        "--quiet",
        "--out",
        str(case_json),
    ]
    if args.step_time_limit_s is not None:
        case_cmd += ["--step-time-limit-s", str(args.step_time_limit_s)]
    if solver["backend"] == "element":
        case_cmd += [
            "--local-coloring" if args.local_coloring else "--no-local-coloring",
            "--nproc-threads",
            str(args.nproc_threads),
            "--element-reorder-mode",
            str(args.element_reorder_mode),
            "--local-hessian-mode",
            str(solver["local_hessian_mode"]),
        ]

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        case_cmd,
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
            "final_energy": None,
            "result": "failed",
            "json_path": str(case_json),
            "log_path": str(case_log),
            "md_path": str(case_md),
        }

    if not case_json.exists():
        return {
            "solver": solver["name"],
            "backend": solver["backend"],
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
            "final_energy": None,
            "result": "failed",
            "json_path": str(case_json),
            "log_path": str(case_log),
            "md_path": str(case_md),
        }

    payload = json.loads(case_json.read_text(encoding="utf-8"))
    row = _summarize_case(solver["name"], level, nprocs, payload)
    row["json_path"] = str(case_json)
    row["log_path"] = str(case_log)
    row["md_path"] = str(case_md)
    if args.write_case_markdown:
        _write_case_markdown(case_name, case_md, payload, row, repo_root)
    return row


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    solvers = [_solver_config(name) for name in args.solvers]
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    rows: list[dict] = []
    done_keys: set[tuple[str, int, int]] = set()

    if args.resume and summary_json.exists():
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        rows = list(payload.get("rows", []))
        done_keys = {_case_key(row) for row in rows}

    for solver in solvers:
        for level in args.levels:
            for nprocs in args.nprocs:
                case_key = (solver["name"], int(level), int(nprocs))
                if case_key in done_keys:
                    continue
                row = _run_case(repo_root, out_dir, solver, level, nprocs, args)
                rows.append(row)
                rows.sort(key=lambda existing: _case_key(existing))
                done_keys.add(case_key)
                _write_summary_json(summary_json, rows, args)
                _write_summary_markdown(summary_md, rows, repo_root)
                print(json.dumps(row, indent=2), flush=True)

    _write_summary_json(summary_json, rows, args)
    _write_summary_markdown(summary_md, rows, repo_root)
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()

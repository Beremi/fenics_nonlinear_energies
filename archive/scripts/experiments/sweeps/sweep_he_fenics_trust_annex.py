#!/usr/bin/env python3
"""Sweep FEniCS custom HE trust-region settings with a per-step kill-switch."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import math
import re
from pathlib import Path


STEP_FINISH_RE = re.compile(r"Step (\d+) finished:")


CASES = (
    {"name": "ls_only", "use_trust_region": False, "trust_radius_init": None},
    {"name": "tr_r0_2", "use_trust_region": True, "trust_radius_init": 0.2},
    {"name": "tr_r0_5", "use_trust_region": True, "trust_radius_init": 0.5},
    {"name": "tr_r1_0", "use_trust_region": True, "trust_radius_init": 1.0},
    {"name": "tr_r2_0", "use_trust_region": True, "trust_radius_init": 2.0},
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--nprocs", type=int, default=32)
    parser.add_argument("--total-steps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--step-timeout-s", type=float, default=100.0)
    parser.add_argument("--case-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiment_results_cache/he_fenics_trust_annex_l4_np32",
    )
    return parser


def _sum_step_linear_iters(step: dict) -> int:
    return int(sum(int(rec["ksp_its"]) for rec in step.get("linear_timing", [])))


def _sum_step_linear_time(step: dict, field: str) -> float:
    return float(sum(float(rec.get(field, 0.0)) for rec in step.get("linear_timing", [])))


def _sum_step_history(step: dict, field: str) -> float:
    return float(sum(float(rec.get(field, 0.0)) for rec in step.get("history", [])))


def _summarize_result(case: dict, payload: dict) -> dict:
    steps = list(payload.get("steps", []))
    completed_steps = len(steps)
    total_newton = int(sum(int(step.get("iters", 0)) for step in steps))
    total_linear = int(sum(_sum_step_linear_iters(step) for step in steps))
    total_time = float(payload.get("solve_time_total", 0.0))
    max_step_time = max((float(step.get("time", 0.0)) for step in steps), default=0.0)
    mean_step_time = total_time / completed_steps if completed_steps > 0 else math.nan
    total_assembly = sum(_sum_step_linear_time(step, "assemble_time") for step in steps)
    total_pc_setup = sum(_sum_step_linear_time(step, "pc_setup_time") for step in steps)
    total_ksp_solve = sum(_sum_step_linear_time(step, "solve_time") for step in steps)
    total_line_search = sum(_sum_step_history(step, "t_ls") for step in steps)
    total_tr_rejects = int(sum(int(rec.get("trust_rejects", 0)) for step in steps for rec in step.get("history", [])))
    final_energy = float(steps[-1]["energy"]) if steps else math.nan
    result = "completed"
    if completed_steps < int(case["steps"]):
        result = str(steps[-1].get("message", "stopped")) if steps else "stopped"
    return {
        "case": case["name"],
        "use_trust_region": bool(case["use_trust_region"]),
        "trust_radius_init": case["trust_radius_init"],
        "completed_steps": completed_steps,
        "first_failed_step": None,
        "failure_mode": None,
        "failure_time_s": None,
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
        "result": result,
    }


def _run_case(repo_root: Path, out_dir: Path, case: dict, args) -> dict:
    case_out = out_dir / f"{case['name']}.json"
    case_log = out_dir / f"{case['name']}.log"
    cmd_parts = [
        "source local_env/activate.sh",
        "&&",
        "stdbuf", "-oL", "-eL",
        "mpirun", "-n", str(args.nprocs),
        "python", "-u", "experiment_scripts/run_he_fenics_custom_case.py",
        "--level", str(args.level),
        "--steps", str(args.steps),
        "--start-step", "1",
        "--total-steps", str(args.total_steps),
        "--maxit", "100",
        "--linesearch-a", "-0.5",
        "--linesearch-b", "2.0",
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
        "--require-all-convergence",
        "--retry-on-nonfinite",
        "--retry-on-maxit",
        "--fail-fast",
        "--save-history",
        "--save-linear-timing",
        "--step-time-limit-s", str(args.step_timeout_s),
        "--out", shlex.quote(str(case_out)),
    ]
    if case["use_trust_region"]:
        cmd_parts.extend([
            "--use-trust-region",
            "--trust-radius-init", str(case["trust_radius_init"]),
            "--trust-radius-min", "1e-8",
            "--trust-radius-max", "1e6",
            "--trust-shrink", "0.5",
            "--trust-expand", "1.5",
            "--trust-eta-shrink", "0.05",
            "--trust-eta-expand", "0.75",
            "--trust-max-reject", "6",
        ])
    else:
        cmd_parts.append("--no-use-trust-region")

    shell_cmd = " ".join(cmd_parts)
    timeout_s = float(args.case_timeout_s)
    try:
        proc = subprocess.run(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=10s",
                f"{timeout_s}s",
                "bash",
                "-lc",
                shell_cmd,
            ],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s + 20.0,
            check=False,
        )
        case_log.write_text(proc.stdout, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        case_log.write_text(stdout, encoding="utf-8")
        return {
            "case": case["name"],
            "use_trust_region": bool(case["use_trust_region"]),
            "trust_radius_init": case["trust_radius_init"],
            "completed_steps": 0,
            "first_failed_step": 1,
            "failure_mode": "process-timeout",
            "failure_time_s": round(float(timeout_s), 3),
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
            "result": "process-timeout",
            "log_path": str(case_log),
            "json_path": str(case_out),
        }

    completed_steps = 0
    for match in STEP_FINISH_RE.finditer(proc.stdout):
        completed_steps = int(match.group(1))

    if proc.returncode == 124 and not case_out.exists():
        return {
            "case": case["name"],
            "use_trust_region": bool(case["use_trust_region"]),
            "trust_radius_init": case["trust_radius_init"],
            "completed_steps": completed_steps,
            "first_failed_step": int(completed_steps + 1),
            "failure_mode": "hard-timeout",
            "failure_time_s": round(float(timeout_s), 3),
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
            "result": "hard-timeout",
            "log_path": str(case_log),
            "json_path": str(case_out),
        }

    if not case_out.exists():
        return {
            "case": case["name"],
            "use_trust_region": bool(case["use_trust_region"]),
            "trust_radius_init": case["trust_radius_init"],
            "completed_steps": completed_steps,
            "first_failed_step": int(completed_steps + 1),
            "failure_mode": f"exit-{proc.returncode}",
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
            "result": "no-json",
            "log_path": str(case_log),
            "json_path": str(case_out),
        }

    with case_out.open(encoding="utf-8") as f:
        payload = json.load(f)
    summary = _summarize_result(case, payload)
    steps = payload.get("steps", [])
    for step in steps:
        if step.get("kill_switch_exceeded"):
            summary["completed_steps"] = max(int(step.get("step", 1)) - 1, 0)
            summary["first_failed_step"] = int(step.get("step", 1))
            summary["failure_mode"] = "kill-switch"
            summary["failure_time_s"] = float(step.get("time", 0.0))
            summary["result"] = "kill-switch"
            break
    summary["log_path"] = str(case_log)
    summary["json_path"] = str(case_out)
    return summary


def _write_markdown(out_path: Path, rows: list[dict]) -> None:
    lines = [
        "# FEniCS custom trust-radius sweep",
        "",
        "| Case | Trust region | Radius init | Completed steps | First failed step | Failure mode | Total Newton | Total linear | Total time [s] | Assembly [s] | PC init [s] | KSP solve [s] | Line search [s] | Final energy | Result |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        def fmt(x):
            if x is None:
                return "-"
            if isinstance(x, float):
                return f"{x:.3f}"
            return str(x)
        lines.append(
            "| {case} | {tr} | {rad} | {completed} | {failed} | {mode} | {nit} | {lit} | {tt} | {asm} | {pc} | {solve} | {ls} | {energy} | {result} |".format(
                case=row["case"],
                tr="yes" if row["use_trust_region"] else "no",
                rad=fmt(row["trust_radius_init"]),
                completed=fmt(row["completed_steps"]),
                failed=fmt(row["first_failed_step"]),
                mode=row["failure_mode"] or "-",
                nit=fmt(row["total_newton_iters"]),
                lit=fmt(row["total_linear_iters"]),
                tt=fmt(row["total_time_s"]),
                asm=fmt(row["assembly_time_s"]),
                pc=fmt(row["pc_init_time_s"]),
                solve=fmt(row["ksp_solve_time_s"]),
                ls=fmt(row["line_search_time_s"]),
                energy=fmt(row["final_energy"]),
                result=row["result"],
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in CASES:
        case = dict(case)
        case["steps"] = int(args.steps)
        rows.append(_run_case(repo_root, out_dir, case, args))

    summary = {
        "level": int(args.level),
        "nprocs": int(args.nprocs),
        "steps": int(args.steps),
        "total_steps": int(args.total_steps),
        "step_timeout_s": float(args.step_timeout_s),
        "rows": rows,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_markdown(out_dir / "summary.md", rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

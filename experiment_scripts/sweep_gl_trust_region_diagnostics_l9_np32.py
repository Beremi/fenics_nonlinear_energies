#!/usr/bin/env python3
"""Detailed GL trust-region diagnostics on the fine benchmark case."""

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
from collections import Counter
from pathlib import Path


SOLVERS = (
    {"name": "fenics_custom", "backend": "fenics", "extra_args": []},
    {
        "name": "jax_petsc_element",
        "backend": "element",
        "extra_args": [
            "--local-coloring",
            "--nproc-threads",
            "1",
            "--element-reorder-mode",
            "block_xyz",
            "--local-hessian-mode",
            "element",
        ],
    },
)


CASES = (
    {
        "name": "ls_ref",
        "kind": "line_search",
        "profile": "reference",
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-3,
        "ksp_max_it": 200,
        "linesearch_a": -0.5,
        "linesearch_b": 2.0,
        "linesearch_tol": 1e-3,
        "use_trust_region": False,
        "maxit": 100,
    },
    {
        "name": "ls_loose",
        "kind": "line_search",
        "profile": "performance",
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "linesearch_a": -0.5,
        "linesearch_b": 2.0,
        "linesearch_tol": 1e-1,
        "use_trust_region": False,
        "maxit": 100,
    },
    {
        "name": "tr_stcg_postls_r0_05",
        "kind": "trust_stcg",
        "profile": "performance",
        "ksp_type": "stcg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "linesearch_a": -0.5,
        "linesearch_b": 2.0,
        "linesearch_tol": 1e-1,
        "use_trust_region": True,
        "trust_radius_init": 0.05,
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
        "trust_max_reject": 6,
        "trust_subproblem_line_search": True,
        "maxit": 100,
    },
    {
        "name": "tr_stcg_postls_r0_2",
        "kind": "trust_stcg",
        "profile": "performance",
        "ksp_type": "stcg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "linesearch_a": -0.5,
        "linesearch_b": 2.0,
        "linesearch_tol": 1e-1,
        "use_trust_region": True,
        "trust_radius_init": 0.2,
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
        "trust_max_reject": 6,
        "trust_subproblem_line_search": True,
        "maxit": 100,
    },
    {
        "name": "tr_stcg_postls_r0_5",
        "kind": "trust_stcg",
        "profile": "performance",
        "ksp_type": "stcg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "linesearch_a": -0.5,
        "linesearch_b": 2.0,
        "linesearch_tol": 1e-1,
        "use_trust_region": True,
        "trust_radius_init": 0.5,
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
        "trust_max_reject": 6,
        "trust_subproblem_line_search": True,
        "maxit": 100,
    },
    {
        "name": "tr_stcg_postls_r1_0",
        "kind": "trust_stcg",
        "profile": "performance",
        "ksp_type": "stcg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "linesearch_a": -0.5,
        "linesearch_b": 2.0,
        "linesearch_tol": 1e-1,
        "use_trust_region": True,
        "trust_radius_init": 1.0,
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
        "trust_max_reject": 6,
        "trust_subproblem_line_search": True,
        "maxit": 100,
    },
    {
        "name": "tr_stcg_postls_r2_0",
        "kind": "trust_stcg",
        "profile": "performance",
        "ksp_type": "stcg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "linesearch_a": -0.5,
        "linesearch_b": 2.0,
        "linesearch_tol": 1e-1,
        "use_trust_region": True,
        "trust_radius_init": 2.0,
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
        "trust_max_reject": 6,
        "trust_subproblem_line_search": True,
        "maxit": 100,
    },
    {
        "name": "tr_stcg_postls_r4_0",
        "kind": "trust_stcg",
        "profile": "performance",
        "ksp_type": "stcg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "linesearch_a": -0.5,
        "linesearch_b": 2.0,
        "linesearch_tol": 1e-1,
        "use_trust_region": True,
        "trust_radius_init": 4.0,
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
        "trust_max_reject": 6,
        "trust_subproblem_line_search": True,
        "maxit": 100,
    },
    {
        "name": "tr_stcg_only_r1_0",
        "kind": "trust_stcg",
        "profile": "performance",
        "ksp_type": "stcg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "linesearch_a": -0.5,
        "linesearch_b": 2.0,
        "linesearch_tol": 1e-1,
        "use_trust_region": True,
        "trust_radius_init": 1.0,
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
        "trust_max_reject": 6,
        "trust_subproblem_line_search": False,
        "maxit": 100,
    },
    {
        "name": "tr_stcg_postls_r1_0_ls1e_3",
        "kind": "trust_stcg",
        "profile": "reference",
        "ksp_type": "stcg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "linesearch_a": -0.5,
        "linesearch_b": 2.0,
        "linesearch_tol": 1e-3,
        "use_trust_region": True,
        "trust_radius_init": 1.0,
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
        "trust_max_reject": 6,
        "trust_subproblem_line_search": True,
        "maxit": 100,
    },
    {
        "name": "tr_stcg_postls_r1_0_ksp1e_8_it200",
        "kind": "trust_stcg",
        "profile": "reference",
        "ksp_type": "stcg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-8,
        "ksp_max_it": 200,
        "linesearch_a": -0.5,
        "linesearch_b": 2.0,
        "linesearch_tol": 1e-1,
        "use_trust_region": True,
        "trust_radius_init": 1.0,
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
        "trust_max_reject": 6,
        "trust_subproblem_line_search": True,
        "maxit": 100,
    },
    {
        "name": "tr_stcg_postls_r1_0_int0_1",
        "kind": "trust_stcg",
        "profile": "performance",
        "ksp_type": "stcg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "linesearch_a": 0.0,
        "linesearch_b": 1.0,
        "linesearch_tol": 1e-1,
        "use_trust_region": True,
        "trust_radius_init": 1.0,
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
        "trust_max_reject": 6,
        "trust_subproblem_line_search": True,
        "maxit": 100,
    },
    {
        "name": "tr_stcg_postls_r1_0_maxit300",
        "kind": "trust_stcg",
        "profile": "performance",
        "ksp_type": "stcg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "linesearch_a": -0.5,
        "linesearch_b": 2.0,
        "linesearch_tol": 1e-1,
        "use_trust_region": True,
        "trust_radius_init": 1.0,
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
        "trust_max_reject": 6,
        "trust_subproblem_line_search": True,
        "maxit": 300,
    },
    {
        "name": "tr_2d_gmres_hypre_r0_2",
        "kind": "trust_2d",
        "profile": "reference",
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-3,
        "ksp_max_it": 200,
        "linesearch_a": -0.5,
        "linesearch_b": 2.0,
        "linesearch_tol": 1e-1,
        "use_trust_region": True,
        "trust_radius_init": 0.2,
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
        "trust_max_reject": 6,
        "trust_subproblem_line_search": False,
        "maxit": 100,
    },
    {
        "name": "tr_2d_gmres_hypre_r1_0",
        "kind": "trust_2d",
        "profile": "reference",
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-3,
        "ksp_max_it": 200,
        "linesearch_a": -0.5,
        "linesearch_b": 2.0,
        "linesearch_tol": 1e-1,
        "use_trust_region": True,
        "trust_radius_init": 1.0,
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
        "trust_max_reject": 6,
        "trust_subproblem_line_search": False,
        "maxit": 100,
    },
    {
        "name": "tr_2d_gmres_hypre_r2_0",
        "kind": "trust_2d",
        "profile": "reference",
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-3,
        "ksp_max_it": 200,
        "linesearch_a": -0.5,
        "linesearch_b": 2.0,
        "linesearch_tol": 1e-1,
        "use_trust_region": True,
        "trust_radius_init": 2.0,
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
        "trust_max_reject": 6,
        "trust_subproblem_line_search": False,
        "maxit": 100,
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
        default="experiment_results_cache/gl_trust_region_diagnostics_l9_np32",
    )
    parser.add_argument("--max-case-wall-s", type=float, default=7200.0)
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
        if not math.isfinite(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def _last_finite(values) -> float | None:
    for value in reversed(list(values)):
        finite_value = _safe_float(value)
        if finite_value is not None:
            return finite_value
    return None


def _reason_summary(linear_records: list[dict]) -> tuple[str, str | None]:
    counts = Counter(str(rec.get("ksp_reason_name", "UNKNOWN")) for rec in linear_records)
    if not counts:
        return "-", None
    dominant = counts.most_common(1)[0][0]
    summary = ", ".join(f"{name}:{count}" for name, count in counts.most_common(3))
    return summary, dominant


def _tail_direction_norm(history: list[dict], tail_len: int = 10) -> float | None:
    values: list[float] = []
    for rec in history[-tail_len:]:
        alpha = _safe_float(rec.get("alpha"))
        step_norm = _safe_float(rec.get("step_norm"))
        if alpha is None or step_norm is None or abs(alpha) <= 1e-14:
            continue
        values.append(step_norm / abs(alpha))
    if not values:
        return None
    values.sort()
    return values[len(values) // 2]


def _summarize(payload: dict, solver: dict, config: dict) -> dict:
    result = payload["result"]
    steps = list(result.get("steps", []))
    step = steps[0] if steps else {}
    history = list(step.get("history", []))
    linear_records = list(step.get("linear_timing", []))
    total_linear = int(
        step.get(
            "linear_iters",
            sum(int(rec.get("ksp_its", 0)) for rec in linear_records),
        )
    )
    message = str(step.get("message", result.get("message", "")))
    completed = "converged" in message.lower()
    final_grad = _last_finite(
        [rec.get("grad_norm_post") for rec in history] + [rec.get("grad_norm") for rec in history]
    )
    initial_grad = _safe_float(history[0].get("grad_norm")) if history else None
    grad_ratio = (
        final_grad / initial_grad
        if final_grad is not None and initial_grad not in (None, 0.0)
        else None
    )
    accepted_steps = sum(bool(rec.get("accepted_step")) for rec in history)
    trust_rejects = sum(int(rec.get("trust_rejects", 0)) for rec in history)
    reason_summary, dominant_reason = _reason_summary(linear_records)
    max_ksp_its = max((int(rec.get("ksp_its", 0)) for rec in linear_records), default=0)
    step_length_hits = sum(
        1 for rec in linear_records if str(rec.get("ksp_reason_name")) == "CONVERGED_STEP_LENGTH"
    )
    neg_curve_hits = sum(
        1 for rec in linear_records if str(rec.get("ksp_reason_name")) == "CONVERGED_NEG_CURVE"
    )
    tail_dir_norm = _tail_direction_norm(history)
    return {
        "solver": solver["name"],
        "backend": solver["backend"],
        "config": config["name"],
        "kind": config["kind"],
        "profile": config["profile"],
        "result": "completed" if completed else "failed",
        "total_time_s": _safe_float(result.get("solve_time_total", result.get("total_time"))),
        "newton_iters": int(step.get("nit", 0)),
        "linear_iters": int(total_linear),
        "final_energy": _safe_float(step.get("energy")),
        "final_grad_norm": final_grad,
        "initial_grad_norm": initial_grad,
        "grad_ratio": grad_ratio,
        "accepted_steps": int(accepted_steps),
        "trust_rejects": int(trust_rejects),
        "max_ksp_its": int(max_ksp_its),
        "ksp_reason_summary": reason_summary,
        "dominant_ksp_reason": dominant_reason,
        "step_length_hits": int(step_length_hits),
        "neg_curve_hits": int(neg_curve_hits),
        "tail_direction_norm": tail_dir_norm,
        "message": message,
    }


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, rows: list[dict]) -> None:
    baseline_energy: dict[str, float] = {}
    for solver_name in {row["solver"] for row in rows}:
        for target in ("ls_ref", "ls_loose"):
            match = next(
                (
                    row
                    for row in rows
                    if row["solver"] == solver_name
                    and row["config"] == target
                    and row["final_energy"] is not None
                ),
                None,
            )
            if match is not None:
                baseline_energy[solver_name] = float(match["final_energy"])
                break

    lines = [
        "# GL Trust-Region Diagnostic Sweep",
        "",
        "Benchmark: `level 9`, `np=32`.",
        "",
        "| Solver | Config | Kind | Result | Time [s] | Newton | Linear | Final energy | Energy gap vs LS ref | Final ||g|| | Grad ratio | Accepted | TR rejects | Max KSP it | Dominant KSP reason | Tail dir norm | Message |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        base_energy = baseline_energy.get(row["solver"])
        energy_gap = (
            row["final_energy"] - base_energy
            if row["final_energy"] is not None and base_energy is not None
            else None
        )
        lines.append(
            "| {solver} | {config} | {kind} | {result} | {time} | {nit} | {lit} | {energy} | {gap} | {grad} | {grad_ratio} | {accepted} | {rejects} | {max_ksp} | {reason} | {tail_dir} | {message} |".format(
                solver=row["solver"],
                config=row["config"],
                kind=row["kind"],
                result=row["result"],
                time=_fmt(row["total_time_s"]),
                nit=_fmt(row["newton_iters"]),
                lit=_fmt(row["linear_iters"]),
                energy=_fmt(row["final_energy"], 6),
                gap=_fmt(energy_gap, 6),
                grad=_fmt(row["final_grad_norm"], 6),
                grad_ratio=_fmt(row["grad_ratio"], 3),
                accepted=_fmt(row["accepted_steps"]),
                rejects=_fmt(row["trust_rejects"]),
                max_ksp=_fmt(row["max_ksp_its"]),
                reason=(row["dominant_ksp_reason"] or "-"),
                tail_dir=_fmt(row["tail_direction_norm"], 6),
                message=str(row["message"]).replace("|", "/"),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Kind = trust_stcg` means PETSc trust-subproblem KSP (`ksp_type=stcg`).",
            "- `Kind = trust_2d` means the older reduced 2D trust hybrid: trust region stays on, but the inner linear solve uses the standard Newton direction (`ksp_type=gmres`) and the trust step is built in the reduced subspace.",
            "- `Tail dir norm` is the median of `||p||` over the last 10 iterations, reconstructed from `step_norm / |alpha|`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_case_command(
    repo_root: Path,
    solver: dict,
    config: dict,
    case_json: Path,
    level: int,
    nprocs: int,
) -> list[str]:
    cmd = [
        "mpirun",
        "-n",
        str(nprocs),
        "python",
        "-u",
        "experiment_scripts/run_trust_region_case.py",
        "--problem",
        "gl",
        "--backend",
        solver["backend"],
        "--level",
        str(level),
        "--out",
        str(case_json),
        "--profile",
        str(config["profile"]),
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
        "1e-6",
        "--tolg",
        "1e-5",
        "--tolg-rel",
        "1e-3",
        "--tolx-rel",
        "1e-3",
        "--tolx-abs",
        "1e-10",
        "--maxit",
        str(config["maxit"]),
        "--linesearch-a",
        str(config["linesearch_a"]),
        "--linesearch-b",
        str(config["linesearch_b"]),
        "--linesearch-tol",
        str(config["linesearch_tol"]),
        "--use-trust-region" if config["use_trust_region"] else "--no-use-trust-region",
        "--trust-radius-init",
        str(config.get("trust_radius_init", 1.0)),
        "--trust-radius-min",
        "1e-8",
        "--trust-radius-max",
        "1e6",
        "--trust-shrink",
        str(config.get("trust_shrink", 0.5)),
        "--trust-expand",
        str(config.get("trust_expand", 1.5)),
        "--trust-eta-shrink",
        str(config.get("trust_eta_shrink", 0.05)),
        "--trust-eta-expand",
        str(config.get("trust_eta_expand", 0.75)),
        "--trust-max-reject",
        str(config.get("trust_max_reject", 6)),
        "--trust-subproblem-line-search"
        if config.get("trust_subproblem_line_search", False)
        else "--no-trust-subproblem-line-search",
        "--save-history",
        "--save-linear-timing",
        "--quiet",
    ]
    cmd.extend(solver["extra_args"])
    return cmd


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
            case_cmd = _build_case_command(
                repo_root=repo_root,
                solver=solver,
                config=config,
                case_json=case_json,
                level=int(args.level),
                nprocs=int(args.nprocs),
            )

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
                rows.append(
                    {
                        "solver": solver["name"],
                        "backend": solver["backend"],
                        "config": config["name"],
                        "kind": config["kind"],
                        "profile": config["profile"],
                        "result": "failed",
                        "total_time_s": None,
                        "newton_iters": None,
                        "linear_iters": None,
                        "final_energy": None,
                        "final_grad_norm": None,
                        "initial_grad_norm": None,
                        "grad_ratio": None,
                        "accepted_steps": None,
                        "trust_rejects": None,
                        "max_ksp_its": None,
                        "ksp_reason_summary": None,
                        "dominant_ksp_reason": None,
                        "step_length_hits": None,
                        "neg_curve_hits": None,
                        "tail_direction_norm": None,
                        "message": "case-timeout" if timed_out else f"missing-json (exit={proc.returncode})",
                    }
                )
            else:
                payload = json.loads(case_json.read_text(encoding="utf-8"))
                rows.append(_summarize(payload, solver, config))

            _write_summary(summary_json, rows)
            _write_markdown(summary_md, rows)

    _write_summary(summary_json, rows)
    _write_markdown(summary_md, rows)


if __name__ == "__main__":
    main()

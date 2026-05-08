#!/usr/bin/env python3
"""Compare element AD, local colored SFD, and constitutive AD routes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_RUNNER = REPO_ROOT / "experiments/runners/run_trust_region_case.py"
PLASTICITY3D_RUNNER = REPO_ROOT / "experiments/runners/run_plasticity3d_backend_mix_case.py"
PLAPLACE_SCALING = REPO_ROOT / "experiments/analysis/docs_assets/data/plaplace/strong_scaling.csv"
RAW_ROOT = REPO_ROOT / "artifacts/raw_results/derivative_route_compare"
REPORT_ROOT = REPO_ROOT / "artifacts/reports/derivative_route_compare"


@dataclass(frozen=True)
class CaseSpec:
    key: str
    benchmark: str
    benchmark_label: str
    route: str
    route_label: str
    problem: str
    level: int
    nprocs: int
    steps: int
    wall_cap_s: float
    runner: str
    local_hessian_mode: str = ""
    assembly_backend: str = ""


CSV_FIELDS = (
    "mode",
    "benchmark",
    "benchmark_label",
    "route",
    "route_label",
    "problem",
    "level",
    "nprocs",
    "steps_requested",
    "completed_steps",
    "result",
    "failure_mode",
    "returncode",
    "wall_time_s",
    "solve_time_s",
    "total_time_s",
    "setup_time_s",
    "newton_iters",
    "krylov_iters",
    "line_search_evals",
    "trust_rejects",
    "hessian_hvp_time_s",
    "hessian_time_s",
    "sfd_colors_min",
    "sfd_colors_max",
    "sfd_colors_unique",
    "final_energy",
    "omega",
    "u_max",
    "json_path",
    "log_path",
    "command",
)


def _he_common(level: int, nprocs: int, *, local_hessian_mode: str) -> CaseSpec:
    return CaseSpec(
        key=f"he_l{level}_step1_np{nprocs}_{local_hessian_mode}",
        benchmark=f"he_l{level}_step1_np{nprocs}",
        benchmark_label=f"HyperElasticity L{level} first step",
        route="element_ad" if local_hessian_mode == "element" else "colored_sfd",
        route_label="Element AD" if local_hessian_mode == "element" else "Colored SFD",
        problem="he",
        level=level,
        nprocs=nprocs,
        steps=1,
        wall_cap_s=120.0 if nprocs <= 2 else 420.0,
        runner="he",
        local_hessian_mode=local_hessian_mode,
    )


def _p3d_common(nprocs: int, *, assembly_backend: str) -> CaseSpec:
    route_labels = {
        "local": ("element_ad", "Element AD"),
        "local_sfd": ("colored_sfd", "Colored SFD"),
        "local_constitutiveAD": ("constitutive_ad", "Constitutive AD"),
    }
    route, label = route_labels[assembly_backend]
    return CaseSpec(
        key=f"plasticity3d_p2_l1_lambda155_np{nprocs}_{route}",
        benchmark=f"plasticity3d_p2_l1_lambda155_np{nprocs}",
        benchmark_label=r"Plasticity3D P2(L1), lambda=1.55",
        route=route,
        route_label=label,
        problem="plasticity3d",
        level=1,
        nprocs=nprocs,
        steps=1,
        wall_cap_s=120.0 if nprocs <= 2 else 300.0,
        runner="plasticity3d",
        assembly_backend=assembly_backend,
    )


def _plaplace_cases() -> list[CaseSpec]:
    return [
        CaseSpec(
            key="plaplace_l9_np32_element_ad",
            benchmark="plaplace_l9_np32",
            benchmark_label="p-Laplace L9",
            route="element_ad",
            route_label="Element AD",
            problem="plaplace",
            level=9,
            nprocs=32,
            steps=1,
            wall_cap_s=0.0,
            runner="plaplace_docs",
        ),
        CaseSpec(
            key="plaplace_l9_np32_colored_sfd",
            benchmark="plaplace_l9_np32",
            benchmark_label="p-Laplace L9",
            route="colored_sfd",
            route_label="Colored SFD",
            problem="plaplace",
            level=9,
            nprocs=32,
            steps=1,
            wall_cap_s=0.0,
            runner="plaplace_docs",
        ),
    ]


def build_case_matrix(mode: str) -> list[CaseSpec]:
    he_level, nprocs = (1, 2) if mode == "smoke" else (4, 32)
    cases = _plaplace_cases()
    cases.extend(
        _he_common(he_level, nprocs, local_hessian_mode=hessian_mode)
        for hessian_mode in ("element", "sfd_local")
    )
    cases.extend(
        _p3d_common(nprocs, assembly_backend=backend)
        for backend in ("local", "local_sfd", "local_constitutiveAD")
    )
    return cases


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    env["MPLBACKEND"] = "Agg"
    return env


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sum_history(step: dict[str, Any], key: str) -> float:
    return float(sum(float(rec.get(key, 0.0)) for rec in step.get("history", [])))


def _sum_flat_history(history: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(rec.get(key, 0.0)) for rec in history))


def _sum_step_linear(step: dict[str, Any]) -> int:
    if "linear_iters" in step:
        return int(step.get("linear_iters") or 0)
    return int(sum(int(rec.get("ksp_its", 0)) for rec in step.get("linear_timing", [])))


def _plaplace_solver_name(case: CaseSpec) -> str:
    return "jax_petsc_element" if case.route == "element_ad" else "jax_petsc_local_sfd"


def _plasticity3d_assembly_details(payload: dict[str, Any]) -> dict[str, Any]:
    callbacks = dict(payload.get("assembly_callbacks") or {})
    hessian = dict(callbacks.get("hessian") or {})
    diagnostics = dict(payload.get("assembler_rank_diagnostics") or {})
    coloring = dict(diagnostics.get("sfd_coloring") or {})
    unique = coloring.get("colors_unique", "")
    if isinstance(unique, list):
        unique_text = " ".join(str(int(v)) for v in unique)
    else:
        unique_text = str(unique or "")
    return {
        "hessian_hvp_time_s": (
            ""
            if _safe_float(hessian.get("hvp_compute")) is None
            else float(hessian["hvp_compute"])
        ),
        "hessian_time_s": (
            ""
            if _safe_float(hessian.get("total")) is None
            else float(hessian["total"])
        ),
        "sfd_colors_min": (
            ""
            if _safe_float(coloring.get("colors_min")) is None
            else int(coloring["colors_min"])
        ),
        "sfd_colors_max": (
            ""
            if _safe_float(coloring.get("colors_max")) is None
            else int(coloring["colors_max"])
        ),
        "sfd_colors_unique": unique_text,
    }


def summarize_plaplace_case(case: CaseSpec, *, mode: str) -> dict[str, Any]:
    rows = list(csv.DictReader(PLAPLACE_SCALING.open(newline="", encoding="utf-8")))
    solver = _plaplace_solver_name(case)
    selected = [
        row
        for row in rows
        if row.get("solver") == solver and int(row.get("nprocs", 0)) == int(case.nprocs)
    ]
    if not selected:
        raise RuntimeError(f"Missing p-Laplace docs row for {solver} at np={case.nprocs}.")
    row = selected[0]
    total_time = float(row["total_time_s"])
    return {
        "mode": mode,
        "benchmark": case.benchmark,
        "benchmark_label": case.benchmark_label,
        "route": case.route,
        "route_label": case.route_label,
        "problem": case.problem,
        "level": case.level,
        "nprocs": case.nprocs,
        "steps_requested": case.steps,
        "completed_steps": 1 if row.get("result") == "completed" else 0,
        "result": str(row.get("result", "completed")),
        "failure_mode": "",
        "returncode": "",
        "wall_time_s": total_time,
        "solve_time_s": "",
        "total_time_s": total_time,
        "setup_time_s": "",
        "newton_iters": int(row["total_newton_iters"]),
        "krylov_iters": int(row["total_linear_iters"]),
        "line_search_evals": "",
        "trust_rejects": "",
        "hessian_hvp_time_s": "",
        "hessian_time_s": "",
        "sfd_colors_min": "",
        "sfd_colors_max": "",
        "sfd_colors_unique": "",
        "final_energy": float(row["final_energy"]),
        "omega": "",
        "u_max": "",
        "json_path": "",
        "log_path": _display_path(PLAPLACE_SCALING),
        "command": "mined from tracked docs strong_scaling.csv",
    }


def build_command(case: CaseSpec, out_path: Path) -> list[str]:
    if case.runner == "he":
        return [
            "mpiexec",
            "-n",
            str(case.nprocs),
            sys.executable,
            "-u",
            str(CASE_RUNNER),
            "--problem",
            "he",
            "--backend",
            "element",
            "--level",
            str(case.level),
            "--out",
            str(out_path),
            "--steps",
            "1",
            "--start-step",
            "1",
            "--total-steps",
            "24",
            "--profile",
            "performance",
            "--ksp-type",
            "stcg",
            "--pc-type",
            "gamg",
            "--ksp-rtol",
            "1e-1",
            "--ksp-max-it",
            "30",
            "--gamg-threshold",
            "0.05",
            "--gamg-agg-nsmooths",
            "1",
            "--gamg-set-coordinates",
            "--use-near-nullspace",
            "--no-pc-setup-on-ksp-cap",
            "--tolf",
            "1e-4",
            "--tolg",
            "1e-3",
            "--tolg-rel",
            "1e-3",
            "--tolx-rel",
            "1e-4",
            "--tolx-abs",
            "1e-10",
            "--maxit",
            "100",
            "--linesearch-a",
            "-0.5",
            "--linesearch-b",
            "2.0",
            "--linesearch-tol",
            "1e-1",
            "--line-search",
            "armijo",
            "--use-trust-region",
            "--trust-subproblem-line-search",
            "--element-reorder-mode",
            "block_xyz",
            "--local-hessian-mode",
            case.local_hessian_mode,
            "--problem-build-mode",
            "replicated",
            "--distribution-strategy",
            "overlap_allgather",
            "--assembly-backend",
            "coo",
            "--local-coloring",
            "--save-history",
            "--save-linear-timing",
            "--quiet",
            "--nproc-threads",
            "1",
        ]
    if case.runner == "plasticity3d":
        return [
            "mpiexec",
            "-n",
            str(case.nprocs),
            sys.executable,
            "-u",
            str(PLASTICITY3D_RUNNER),
            "--out-dir",
            str(out_path.parent),
            "--output-json",
            str(out_path),
            "--assembly-backend",
            case.assembly_backend,
            "--solver-backend",
            "local_pmg_mumps",
            "--mesh-name",
            "hetero_ssr_L1",
            "--elem-degree",
            "2",
            "--constraint-variant",
            "glued_bottom",
            "--lambda-target",
            "1.55",
            "--pmg-strategy",
            "same_mesh_p2_p1",
            "--ksp-rtol",
            "1e-2",
            "--ksp-max-it",
            "100",
            "--convergence-mode",
            "all",
            "--stop-tol",
            "2e-3",
            "--grad-stop-tol",
            "1e-4",
            "--maxit",
            "80",
            "--line-search",
            "armijo",
            "--linesearch-tol",
            "1e-3",
            "--use-trust-region",
            "--trust-subproblem-line-search",
        ]
    raise ValueError(f"No command for runner={case.runner!r}")


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
    proc.wait(timeout=5.0)


def _child_preexec() -> None:
    os.setsid()


def _write_fallback_payload(
    *,
    case: CaseSpec,
    out_path: Path,
    command: list[str],
    result: str,
    failure_mode: str,
    returncode: int | None,
    wall_time_s: float,
) -> dict[str, Any]:
    payload = {
        "case": {
            "problem": case.problem,
            "level": case.level,
            "nprocs": case.nprocs,
            "route": case.route,
            "command": command,
        },
        "result": {
            "status": result,
            "failure_mode": failure_mode,
            "returncode": returncode,
            "wall_time_s": wall_time_s,
            "steps": [],
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def summarize_he_payload(
    *,
    mode: str,
    case: CaseSpec,
    payload: dict[str, Any],
    json_path: Path,
    log_path: Path,
    command: list[str],
    returncode: int | None,
    wall_time_s: float,
    launcher_failure: str | None = None,
) -> dict[str, Any]:
    result = dict(payload.get("result", {}))
    steps = list(result.get("steps", []))
    completed_steps = len(steps)
    if launcher_failure == "timeout":
        row_result = "timeout"
        failure_mode = "timeout"
    elif returncode not in (None, 0):
        row_result = "launcher_failed"
        failure_mode = f"returncode {returncode}"
    elif not steps:
        row_result = "failed"
        failure_mode = str(result.get("failure_mode") or "no-steps")
    elif completed_steps < case.steps:
        row_result = "failed"
        failure_mode = f"incomplete: {completed_steps}/{case.steps} steps"
    elif any(step.get("kill_switch_exceeded") for step in steps):
        row_result = "timeout"
        failure_mode = "step-time-limit"
    elif "converged" in str(steps[-1].get("message", "")).lower():
        row_result = "completed"
        failure_mode = ""
    else:
        row_result = "failed"
        failure_mode = str(steps[-1].get("message") or "not converged")

    final_energy = _safe_float(steps[-1].get("energy")) if steps else None
    return {
        "mode": mode,
        "benchmark": case.benchmark,
        "benchmark_label": case.benchmark_label,
        "route": case.route,
        "route_label": case.route_label,
        "problem": case.problem,
        "level": case.level,
        "nprocs": case.nprocs,
        "steps_requested": case.steps,
        "completed_steps": completed_steps,
        "result": row_result,
        "failure_mode": failure_mode,
        "returncode": "" if returncode is None else int(returncode),
        "wall_time_s": float(wall_time_s),
        "solve_time_s": "" if _safe_float(result.get("solve_time_total")) is None else float(result["solve_time_total"]),
        "total_time_s": "" if _safe_float(result.get("total_time")) is None else float(result["total_time"]),
        "setup_time_s": "" if _safe_float(result.get("setup_time")) is None else float(result["setup_time"]),
        "newton_iters": int(sum(int(step.get("nit", 0)) for step in steps)),
        "krylov_iters": int(sum(_sum_step_linear(step) for step in steps)),
        "line_search_evals": int(sum(int(_sum_history(step, "ls_evals")) for step in steps)),
        "trust_rejects": int(sum(int(_sum_history(step, "trust_rejects")) for step in steps)),
        "hessian_hvp_time_s": "",
        "hessian_time_s": "",
        "sfd_colors_min": "",
        "sfd_colors_max": "",
        "sfd_colors_unique": "",
        "final_energy": "" if final_energy is None else float(final_energy),
        "omega": "",
        "u_max": "",
        "json_path": _display_path(json_path),
        "log_path": _display_path(log_path),
        "command": shlex.join(command),
    }


def summarize_plasticity3d_payload(
    *,
    mode: str,
    case: CaseSpec,
    payload: dict[str, Any],
    json_path: Path,
    log_path: Path,
    command: list[str],
    returncode: int | None,
    wall_time_s: float,
    launcher_failure: str | None = None,
) -> dict[str, Any]:
    status = str(payload.get("status", ""))
    message = str(payload.get("message", ""))
    if launcher_failure == "timeout":
        row_result = "timeout"
        failure_mode = "timeout"
    elif returncode not in (None, 0):
        row_result = "launcher_failed"
        failure_mode = f"returncode {returncode}"
    elif status == "completed":
        row_result = "completed"
        failure_mode = ""
    else:
        row_result = "failed"
        failure_mode = message or status or "not converged"

    history = list(payload.get("history", []))
    details = _plasticity3d_assembly_details(payload)
    return {
        "mode": mode,
        "benchmark": case.benchmark,
        "benchmark_label": case.benchmark_label,
        "route": case.route,
        "route_label": case.route_label,
        "problem": case.problem,
        "level": case.level,
        "nprocs": case.nprocs,
        "steps_requested": case.steps,
        "completed_steps": 1 if status == "completed" else 0,
        "result": row_result,
        "failure_mode": failure_mode,
        "returncode": "" if returncode is None else int(returncode),
        "wall_time_s": float(wall_time_s),
        "solve_time_s": "" if _safe_float(payload.get("solve_time")) is None else float(payload["solve_time"]),
        "total_time_s": "" if _safe_float(payload.get("total_time")) is None else float(payload["total_time"]),
        "setup_time_s": "",
        "newton_iters": int(payload.get("nit", 0)),
        "krylov_iters": int(payload.get("linear_iterations_total", 0)),
        "line_search_evals": int(_sum_flat_history(history, "ls_evals")),
        "trust_rejects": int(_sum_flat_history(history, "trust_rejects")),
        **details,
        "final_energy": "" if _safe_float(payload.get("energy")) is None else float(payload["energy"]),
        "omega": "" if _safe_float(payload.get("omega")) is None else float(payload["omega"]),
        "u_max": "" if _safe_float(payload.get("u_max")) is None else float(payload["u_max"]),
        "json_path": _display_path(json_path),
        "log_path": _display_path(log_path),
        "command": shlex.join(command),
    }


def summarize_payload(
    *,
    mode: str,
    case: CaseSpec,
    payload: dict[str, Any],
    json_path: Path,
    log_path: Path,
    command: list[str],
    returncode: int | None,
    wall_time_s: float,
    launcher_failure: str | None = None,
) -> dict[str, Any]:
    if case.runner == "he":
        return summarize_he_payload(
            mode=mode,
            case=case,
            payload=payload,
            json_path=json_path,
            log_path=log_path,
            command=command,
            returncode=returncode,
            wall_time_s=wall_time_s,
            launcher_failure=launcher_failure,
        )
    if case.runner == "plasticity3d":
        if "status" not in payload and isinstance(payload.get("result"), dict):
            payload = dict(payload["result"])
        return summarize_plasticity3d_payload(
            mode=mode,
            case=case,
            payload=payload,
            json_path=json_path,
            log_path=log_path,
            command=command,
            returncode=returncode,
            wall_time_s=wall_time_s,
            launcher_failure=launcher_failure,
        )
    raise ValueError(f"Cannot summarize runner={case.runner!r}")


def run_case(case: CaseSpec, *, mode: str, raw_dir: Path, resume: bool = True) -> dict[str, Any]:
    if case.runner == "plaplace_docs":
        return summarize_plaplace_case(case, mode=mode)

    case_dir = raw_dir / case.key
    case_dir.mkdir(parents=True, exist_ok=True)
    out_path = case_dir / "output.json"
    log_path = case_dir / "run.log"
    command = build_command(case, out_path)
    returncode: int | None = None
    launcher_failure: str | None = None
    wall_time_s = 0.0

    if resume and out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        if out_path.exists():
            out_path.unlink()
        start = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + shlex.join(command) + "\n\n")
            log.flush()
            proc = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=_base_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=_child_preexec,
            )
            try:
                returncode = proc.wait(timeout=max(1.0, float(case.wall_cap_s)))
            except subprocess.TimeoutExpired:
                launcher_failure = "timeout"
                _terminate_process_group(proc)
                returncode = proc.returncode
                log.write(f"\n[runner] timeout after {case.wall_cap_s:.3f} s\n")
        wall_time_s = time.perf_counter() - start

        if out_path.exists():
            try:
                payload = json.loads(out_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                launcher_failure = launcher_failure or "invalid-json"
                payload = _write_fallback_payload(
                    case=case,
                    out_path=out_path,
                    command=command,
                    result="failed",
                    failure_mode="invalid-json",
                    returncode=returncode,
                    wall_time_s=wall_time_s,
                )
        else:
            payload = _write_fallback_payload(
                case=case,
                out_path=out_path,
                command=command,
                result="timeout" if launcher_failure == "timeout" else "failed",
                failure_mode=launcher_failure or "missing-output",
                returncode=returncode,
                wall_time_s=wall_time_s,
            )

    return summarize_payload(
        mode=mode,
        case=case,
        payload=payload,
        json_path=out_path,
        log_path=log_path,
        command=command,
        returncode=returncode,
        wall_time_s=wall_time_s,
        launcher_failure=launcher_failure,
    )


def write_reports(rows: list[dict[str, Any]], *, mode: str, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"{mode}_summary.csv"
    json_path = report_dir / f"{mode}_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "mode": mode,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cases = build_case_matrix(args.mode)
    raw_dir = args.raw_root / args.mode

    if args.dry_run:
        for case in cases:
            if case.runner == "plaplace_docs":
                print(f"# {case.key}: mined from {_display_path(PLAPLACE_SCALING)}")
                continue
            out_path = raw_dir / case.key / "output.json"
            print(shlex.join(build_command(case, out_path)))
        return

    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] {case.key}", flush=True)
        row = run_case(case, mode=args.mode, raw_dir=raw_dir, resume=bool(args.resume))
        print(
            f"  -> {row['result']} newton={row['newton_iters']} "
            f"krylov={row['krylov_iters']} time={row['solve_time_s'] or row['total_time_s'] or row['wall_time_s']}",
            flush=True,
        )
        rows.append(row)
    write_reports(rows, mode=args.mode, report_dir=args.report_root)
    print(f"Wrote {args.report_root / (args.mode + '_summary.csv')}")


if __name__ == "__main__":
    main()

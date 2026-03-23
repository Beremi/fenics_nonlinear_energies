#!/usr/bin/env python3
"""Run L7 P4 deep-P1-tail PMG scaling cases at 1/2/4/8/16 MPI ranks with Newton maxit=20."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import psutil

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
    / "slope_stability_l7_p4_deep_p1_tail_scaling_lambda1_maxit20"
)
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

RANKS = [1, 2, 4, 8, 16]
CUSTOM_HIERARCHY = "1:1,2:1,3:1,4:1,5:1,6:1,7:1,7:2,7:4"
MEMORY_GUARD_GIB = 210.0
WATCH_PERIOD_S = 15.0

COMMON_ARGS = [
    "--level",
    "7",
    "--elem_degree",
    "4",
    "--lambda-target",
    "1.0",
    "--profile",
    "performance",
    "--pc_type",
    "mg",
    "--mg_strategy",
    "custom_mixed",
    "--mg_custom_hierarchy",
    CUSTOM_HIERARCHY,
    "--mg_variant",
    "legacy_pmg",
    "--ksp_type",
    "fgmres",
    "--ksp_rtol",
    "1e-2",
    "--ksp_max_it",
    "15",
    "--accept_ksp_maxit_direction",
    "--no-guard_ksp_maxit_direction",
    "--maxit",
    "20",
    "--benchmark_mode",
    "warmup_once_then_solve",
    "--save-history",
    "--save-linear-timing",
    "--quiet",
    "--no-use_trust_region",
    "--line_search",
    "armijo",
    "--distribution_strategy",
    "overlap_p2p",
    "--problem_build_mode",
    "rank_local",
    "--mg_level_build_mode",
    "rank_local",
    "--mg_transfer_build_mode",
    "owned_rows",
    "--mg_p4_smoother_ksp_type",
    "richardson",
    "--mg_p4_smoother_pc_type",
    "sor",
    "--mg_p4_smoother_steps",
    "3",
    "--mg_p2_smoother_ksp_type",
    "richardson",
    "--mg_p2_smoother_pc_type",
    "sor",
    "--mg_p2_smoother_steps",
    "3",
    "--mg_p1_smoother_ksp_type",
    "richardson",
    "--mg_p1_smoother_pc_type",
    "sor",
    "--mg_p1_smoother_steps",
    "3",
    "--mg_coarse_backend",
    "rank0_lu_broadcast",
]


def _ensure_assets() -> None:
    for level in range(1, 8):
        for degree in (1, 2, 4):
            ensure_same_mesh_case_hdf5(level, degree)


def _load_rows() -> list[dict[str, object]]:
    if SUMMARY_PATH.exists():
        return list(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    return []


def _write_rows(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _command(ranks: int, out: Path) -> list[str]:
    return [
        "mpiexec",
        "-n",
        str(ranks),
        str(PYTHON),
        "-u",
        str(SOLVER),
        *COMMON_ARGS,
        "--out",
        str(out),
    ]


def _sum(items: list[dict[str, object]], key: str) -> float:
    return float(sum(float(item.get(key, 0.0)) for item in items))


def _watch_process_tree(proc: subprocess.Popen[str], log_path: Path) -> bool:
    parent = psutil.Process(proc.pid)
    exceeded = False
    with log_path.open("w", encoding="utf-8") as fh:
        while proc.poll() is None:
            procs = [parent]
            try:
                procs.extend(parent.children(recursive=True))
            except psutil.Error:
                pass
            records: list[dict[str, object]] = []
            rss_total = 0.0
            for child in procs:
                try:
                    info = child.memory_info()
                    rss_gb = float(info.rss) / (1024.0**3)
                    cmd = " ".join(child.cmdline())[:240]
                    records.append(
                        {
                            "pid": int(child.pid),
                            "rss_gb": rss_gb,
                            "cmd": cmd,
                        }
                    )
                    rss_total += rss_gb
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            fh.write(
                json.dumps(
                    {
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "rss_gb_total": rss_total,
                        "proc_count": len(records),
                        "procs": records,
                    }
                )
                + "\n"
            )
            fh.flush()
            if rss_total >= MEMORY_GUARD_GIB:
                exceeded = True
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                time.sleep(5.0)
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break
            time.sleep(WATCH_PERIOD_S)
    return exceeded


def _aggregate(payload: dict[str, object], ranks: int) -> dict[str, object]:
    result = dict(payload.get("result", {}))
    steps = list(result.get("steps", []))
    last_step = dict(steps[-1]) if steps else {}
    history = list(last_step.get("history", []))
    linear = list(last_step.get("linear_timing", []))
    timings = dict(payload.get("timings", {}))
    linear_summary = dict(last_step.get("linear_summary", {}))
    linear_solver = dict(payload.get("metadata", {}).get("linear_solver", {}))
    mesh = dict(payload.get("mesh", {}))
    return {
        "ranks": int(ranks),
        "solver_success": bool(result.get("solver_success", False)),
        "status": str(result.get("status", "")),
        "message": str(last_step.get("message", "")),
        "level": int(mesh.get("level", 7)),
        "h": float(mesh.get("h", 0.0)),
        "nodes": int(mesh.get("nodes", 0)),
        "elements": int(mesh.get("elements", 0)),
        "free_dofs": int(mesh.get("free_dofs", 0)),
        "mg_custom_hierarchy": str(linear_solver.get("mg_custom_hierarchy", CUSTOM_HIERARCHY)),
        "newton_iterations": int(last_step.get("nit", len(linear))),
        "linear_iterations": int(last_step.get("linear_iters", 0)),
        "line_search_evals": int(sum(int(item.get("ls_evals", 0)) for item in history)),
        "line_search_time_sec": float(sum(float(item.get("t_ls", 0.0)) for item in history)),
        "accepted_capped_step_count": int(last_step.get("accepted_capped_step_count", 0)),
        "setup_time_sec": float(timings.get("setup_time", 0.0)),
        "one_time_setup_time_sec": float(
            timings.get("one_time_setup_time", timings.get("setup_time", 0.0))
        ),
        "steady_state_setup_time_sec": float(timings.get("steady_state_setup_time", 0.0)),
        "solve_time_sec": float(timings.get("solve_time", 0.0)),
        "steady_state_total_time_sec": float(timings.get("steady_state_total_time", 0.0)),
        "benchmark_total_time_sec": float(timings.get("benchmark_total_time", 0.0)),
        "total_time_sec": float(timings.get("total_time", 0.0)),
        "problem_build_time_sec": float(timings.get("problem_build_time", 0.0)),
        "assembler_setup_time_sec": float(timings.get("assembler_setup_time", 0.0)),
        "solver_bootstrap_time_sec": float(timings.get("solver_bootstrap_time", 0.0)),
        "finalize_time_sec": float(timings.get("finalize_time", 0.0)),
        "linear_pc_setup_time_sec": _sum(linear, "pc_setup_time"),
        "linear_ksp_solve_time_sec": _sum(linear, "solve_time"),
        "linear_assemble_time_sec": _sum(linear, "assemble_total_time"),
        "linear_total_time_sec": _sum(linear, "linear_total_time"),
        "energy": float(last_step.get("energy", float("nan"))),
        "final_grad_norm": float(
            last_step.get("final_grad_norm", result.get("final_grad_norm", float("nan")))
        ),
        "omega": float(last_step.get("omega", float("nan"))),
        "u_max": float(last_step.get("u_max", float("nan"))),
        "worst_true_relative_residual": float(
            linear_summary.get("worst_true_relative_residual", float("nan"))
        ),
    }


def _run_case(ranks: int) -> dict[str, object]:
    case_dir = OUTPUT_ROOT / f"np{ranks}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"
    memory_watch_path = case_dir / "memory_watch.jsonl"

    if not result_path.exists():
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("BLIS_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_fh:
            proc = subprocess.Popen(
                _command(ranks, result_path),
                cwd=REPO_ROOT,
                stdout=stdout_fh,
                stderr=stderr_fh,
                text=True,
                env=env,
                start_new_session=True,
            )
            memory_exceeded = _watch_process_tree(proc, memory_watch_path)
            returncode = proc.wait()
        if returncode != 0 and not result_path.exists():
            message = "memory guard killed process" if memory_exceeded else "subprocess failed"
            status = "memory_guard_killed" if memory_exceeded else "subprocess_failed"
            return {
                "ranks": int(ranks),
                "solver_success": False,
                "status": status,
                "message": message,
                "mg_custom_hierarchy": CUSTOM_HIERARCHY,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "result_json": str(result_path),
                "memory_watch_path": str(memory_watch_path),
            }

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    row = _aggregate(payload, ranks)
    row.update(
        {
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "result_json": str(result_path),
            "memory_watch_path": str(memory_watch_path),
        }
    )
    return row


def main() -> None:
    _ensure_assets()
    rows = _load_rows()
    completed = {int(row["ranks"]) for row in rows}
    for ranks in RANKS:
        if ranks in completed:
            continue
        rows.append(_run_case(ranks))
        rows = sorted(rows, key=lambda row: int(row["ranks"]))
        _write_rows(rows)


if __name__ == "__main__":
    main()

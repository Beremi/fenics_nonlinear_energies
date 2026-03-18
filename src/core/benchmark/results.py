"""Shared result-shaping helpers for benchmark solvers."""

from __future__ import annotations

from typing import Any, Mapping


def sum_step_linear(step: Mapping[str, Any]) -> int:
    """Return the total KSP iterations recorded for a solver step."""
    if "linear_iters" in step:
        return int(step["linear_iters"])
    return int(sum(int(rec.get("ksp_its", 0)) for rec in step.get("linear_timing", [])))


def assemble_time(record: Mapping[str, Any]) -> float:
    """Return the assemble time from either old or new timing record keys."""
    return float(record.get("assemble_total_time", record.get("assemble_time", 0.0)))


def sum_step_linear_time(step: Mapping[str, Any], field: str) -> float:
    """Sum one timing field across a step's linear timing records."""
    return float(sum(float(rec.get(field, 0.0)) for rec in step.get("linear_timing", [])))


def sum_step_history(step: Mapping[str, Any], field: str) -> float:
    """Sum one field across a step's nonlinear history records."""
    return float(sum(float(rec.get(field, 0.0)) for rec in step.get("history", [])))


def cumulative_linear_timing(linear_timing: list[dict]) -> dict[str, float]:
    """Aggregate timing totals from a step's per-linear-solve records."""
    asm_cumulative = sum(assemble_time(rec) for rec in linear_timing)
    pc_setup_cumulative = sum(float(rec.get("pc_setup_time", 0.0)) for rec in linear_timing)
    linear_solve_cumulative = sum(float(rec.get("solve_time", 0.0)) for rec in linear_timing)
    ksp_cumulative = sum(
        float(rec.get("pc_setup_time", 0.0)) + float(rec.get("solve_time", 0.0))
        for rec in linear_timing
    )
    return {
        "asm_time_cumulative": float(asm_cumulative),
        "pc_setup_time_cumulative": float(pc_setup_cumulative),
        "linear_solve_time_cumulative": float(linear_solve_cumulative),
        "ksp_time_cumulative": float(ksp_cumulative),
    }


def summarize_single_step_case(
    solver_name: str,
    level: int,
    nprocs: int,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize one scalar single-step benchmark payload into a summary row."""
    result = payload["result"]
    steps = list(result.get("steps", []))
    completed_steps = len(steps)
    total_newton = int(sum(int(step.get("nit", step.get("iters", 0))) for step in steps))
    total_linear = int(sum(sum_step_linear(step) for step in steps))
    total_time = float(result.get("solve_time_total", result.get("total_time", 0.0)))
    max_step_time = max((float(step.get("time", 0.0)) for step in steps), default=0.0)
    mean_step_time = total_time / completed_steps if completed_steps else None
    total_assembly = sum(sum_step_linear_time(step, "assemble_total_time") for step in steps)
    total_pc_setup = sum(sum_step_linear_time(step, "pc_setup_time") for step in steps)
    total_ksp_solve = sum(sum_step_linear_time(step, "solve_time") for step in steps)
    total_line_search = sum(sum_step_history(step, "t_ls") for step in steps)
    final_energy = float(steps[-1]["energy"]) if steps else None
    failure_mode = None
    failure_time_s = None
    first_failed_step = None
    case_result = "completed"

    if steps:
        for step in steps:
            if step.get("kill_switch_exceeded"):
                first_failed_step = int(step.get("step", 1))
                failure_mode = "kill-switch"
                failure_time_s = float(step.get("time", 0.0))
                case_result = "failed"
                break
        if case_result == "completed":
            last_msg = str(steps[-1].get("message", ""))
            if "converged" not in last_msg.lower():
                first_failed_step = int(steps[-1].get("step", 1))
                failure_mode = last_msg or "stopped"
                case_result = "failed"
    else:
        case_result = "failed"
        first_failed_step = 1
        failure_mode = "no-steps"

    return {
        "solver": solver_name,
        "backend": payload["case"]["backend"],
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
        "final_energy": final_energy,
        "result": case_result,
    }


def summarize_load_step_case(
    solver_name: str,
    total_steps: int,
    level: int,
    nprocs: int,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize one multi-step benchmark payload into a summary row."""
    result = payload["result"]
    steps = list(result.get("steps", []))
    completed_steps = len(steps)
    total_newton = int(sum(int(step.get("nit", step.get("iters", 0))) for step in steps))
    total_linear = int(sum(sum_step_linear(step) for step in steps))
    total_time = float(result.get("solve_time_total", result.get("total_time", 0.0)))
    max_step_time = max((float(step.get("time", 0.0)) for step in steps), default=0.0)
    mean_step_time = total_time / completed_steps if completed_steps else None
    total_assembly = sum(sum_step_linear_time(step, "assemble_total_time") for step in steps)
    if total_assembly == 0.0:
        total_assembly = sum(sum_step_linear_time(step, "assemble_time") for step in steps)
    total_pc_setup = sum(sum_step_linear_time(step, "pc_setup_time") for step in steps)
    total_ksp_solve = sum(sum_step_linear_time(step, "solve_time") for step in steps)
    total_line_search = sum(sum_step_history(step, "t_ls") for step in steps)
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


def summarize_pure_jax_load_step_case(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build the summary row used by the pure-JAX HE benchmark suite."""
    steps = list(payload.get("steps", []))
    return {
        "solver": payload["solver"],
        "level": int(payload["level"]),
        "total_steps": int(payload["total_steps"]),
        "total_dofs": int(payload["total_dofs"]),
        "free_dofs": int(payload["free_dofs"]),
        "time": float(payload["time"]),
        "total_newton_iters": int(payload["total_newton_iters"]),
        "total_linear_iters": int(payload["total_linear_iters"]),
        "max_step_time": max((float(step["time"]) for step in steps), default=0.0),
        "result": str(payload["result"]),
    }

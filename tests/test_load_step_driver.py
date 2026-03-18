from __future__ import annotations

import time

from src.core.benchmark.repair import build_retry_attempts, needs_solver_repair
from src.core.petsc.load_step_driver import (
    LoadStepAttempt,
    attempts_from_tuples,
    build_load_step_result,
    run_load_steps,
)


def test_run_load_steps_retries_and_records_attempt_metadata():
    attempts = [
        LoadStepAttempt("primary", (-0.5, 2.0), 1e-1, 30),
        LoadStepAttempt("repair", (-0.5, 1.0), 1e-2, 60),
    ]
    retry_events = []
    solve_calls = []

    def prepare_step(step_ctx):
        step_ctx.state["prepared"] = True

    def build_attempts(_step_ctx):
        return attempts

    def solve_attempt(step_ctx, attempt):
        solve_calls.append((step_ctx.step, attempt.name))
        if step_ctx.step == 1 and attempt.name == "primary":
            return {
                "message": "Maximum number of iterations reached",
                "fun": 1.0,
                "nit": 4,
            }, 0.25
        return {
            "message": "Converged (energy, step, gradient)",
            "fun": 0.5 * step_ctx.step,
            "nit": 2,
        }, 0.125 * step_ctx.step

    def should_retry(result, _step_ctx):
        return needs_solver_repair(result)

    def build_step_record(step_ctx, result, step_time, attempt):
        return {
            "step": step_ctx.step,
            "angle": step_ctx.angle,
            "time": step_time,
            "message": result["message"],
            "attempt": attempt.name,
            "ksp_rtol_used": attempt.linear_rtol,
            "ksp_max_it_used": attempt.linear_max_it,
        }

    def on_retry(step_ctx, attempt, attempt_idx, total_attempts):
        retry_events.append((step_ctx.step, attempt.name, attempt_idx, total_attempts))

    step_records = run_load_steps(
        start_step=1,
        num_steps=2,
        rotation_per_step=0.5,
        prepare_step=prepare_step,
        build_attempts=build_attempts,
        solve_attempt=solve_attempt,
        should_retry=should_retry,
        build_step_record=build_step_record,
        on_retry=on_retry,
    )

    assert solve_calls == [
        (1, "primary"),
        (1, "repair"),
        (2, "primary"),
    ]
    assert retry_events == [(1, "primary", 1, 2)]
    assert step_records == [
        {
            "step": 1,
            "angle": 0.5,
            "time": 0.125,
            "message": "Converged (energy, step, gradient)",
            "attempt": "repair",
            "ksp_rtol_used": 1e-2,
            "ksp_max_it_used": 60,
        },
        {
            "step": 2,
            "angle": 1.0,
            "time": 0.25,
            "message": "Converged (energy, step, gradient)",
            "attempt": "primary",
            "ksp_rtol_used": 1e-1,
            "ksp_max_it_used": 30,
        },
    ]


def test_run_load_steps_honors_stop_callback():
    seen_steps = []

    def prepare_step(_step_ctx):
        return None

    def build_attempts(_step_ctx):
        return [LoadStepAttempt("primary", (-0.5, 2.0), 1e-1, 30)]

    def solve_attempt(step_ctx, attempt):
        seen_steps.append((step_ctx.step, attempt.name))
        return {"message": "Converged", "fun": 0.0, "nit": 1}, 0.1

    def should_retry(_result, _step_ctx):
        return False

    def build_step_record(step_ctx, result, step_time, attempt):
        return {
            "step": step_ctx.step,
            "time": step_time,
            "message": result["message"],
            "attempt": attempt.name,
        }

    def should_stop(step_record, _result, _step_ctx):
        return step_record["step"] == 1

    step_records = run_load_steps(
        start_step=1,
        num_steps=3,
        rotation_per_step=0.5,
        prepare_step=prepare_step,
        build_attempts=build_attempts,
        solve_attempt=solve_attempt,
        should_retry=should_retry,
        build_step_record=build_step_record,
        should_stop=should_stop,
    )

    assert seen_steps == [(1, "primary")]
    assert len(step_records) == 1


def test_build_load_step_result_aggregates_time_and_merges_extra():
    total_runtime_start = time.perf_counter() - 1.5
    payload = build_load_step_result(
        mesh_level=2,
        total_dofs=123,
        setup_time=0.1234567,
        total_runtime_start=total_runtime_start,
        steps=[{"time": 0.25}, {"time": 0.5}],
        extra={"free_dofs": 100},
    )

    assert payload["mesh_level"] == 2
    assert payload["total_dofs"] == 123
    assert payload["free_dofs"] == 100
    assert payload["setup_time"] == 0.123457
    assert payload["solve_time_total"] == 0.75
    assert payload["total_time"] >= 1.5
    assert payload["steps"] == [{"time": 0.25}, {"time": 0.5}]


def test_build_retry_attempts_supports_generalized_repair_policy():
    attempts = build_retry_attempts(
        retry_on_nonfinite=True,
        retry_on_maxit=False,
        linesearch_interval=(-0.5, 2.0),
        ksp_rtol=1e-2,
        ksp_max_it=30,
        retry_rtol_factor=0.2,
        retry_linesearch_b=0.8,
        retry_ksp_max_it_factor=1.5,
        min_rtol=1e-8,
    )

    assert attempts == [
        ("primary", (-0.5, 2.0), 1e-2, 30),
        ("repair", (-0.5, 0.8), 2e-3, 45),
    ]
    assert attempts_from_tuples(attempts)[1].name == "repair"

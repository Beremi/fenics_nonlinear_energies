"""Shared load-step orchestration for multi-step nonlinear solvers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class LoadStepAttempt:
    """One nonlinear solve attempt for a load step."""

    name: str
    linesearch_interval: tuple[float, float]
    linear_rtol: float | None = None
    linear_max_it: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass
class LoadStepContext:
    """Mutable context carried through one load-step solve."""

    step: int
    angle: float
    state: dict[str, object] = field(default_factory=dict)


PrepareStepFn = Callable[[LoadStepContext], None]
BuildAttemptsFn = Callable[[LoadStepContext], Sequence[LoadStepAttempt]]
SolveAttemptFn = Callable[[LoadStepContext, LoadStepAttempt], tuple[dict[str, object], float]]
ShouldRetryFn = Callable[[dict[str, object], LoadStepContext], bool]
BuildRecordFn = Callable[
    [LoadStepContext, dict[str, object], float, LoadStepAttempt],
    dict[str, object],
]
ShouldStopFn = Callable[[dict[str, object], dict[str, object], LoadStepContext], bool]
OnRetryFn = Callable[[LoadStepContext, LoadStepAttempt, int, int], None]
OnStepCompleteFn = Callable[[dict[str, object], LoadStepContext], None]


def attempts_from_tuples(
    attempts: Sequence[tuple[str, tuple[float, float], float | None, int | None]]
) -> list[LoadStepAttempt]:
    """Convert legacy tuple-based retry schedules into dataclass attempts."""
    return [
        LoadStepAttempt(
            name=str(name),
            linesearch_interval=(float(interval[0]), float(interval[1])),
            linear_rtol=(None if linear_rtol is None else float(linear_rtol)),
            linear_max_it=(None if linear_max_it is None else int(linear_max_it)),
        )
        for name, interval, linear_rtol, linear_max_it in attempts
    ]


def run_load_steps(
    *,
    start_step: int,
    num_steps: int,
    rotation_per_step: float,
    prepare_step: PrepareStepFn,
    build_attempts: BuildAttemptsFn,
    solve_attempt: SolveAttemptFn,
    should_retry: ShouldRetryFn,
    build_step_record: BuildRecordFn,
    should_stop: ShouldStopFn | None = None,
    on_retry: OnRetryFn | None = None,
    on_step_complete: OnStepCompleteFn | None = None,
) -> list[dict[str, object]]:
    """Run a multi-step nonlinear solve with shared retry/stop handling."""
    step_records: list[dict[str, object]] = []

    for step in range(int(start_step), int(start_step) + int(num_steps)):
        step_ctx = LoadStepContext(
            step=int(step),
            angle=float(step * float(rotation_per_step)),
        )
        prepare_step(step_ctx)

        attempt_specs = list(build_attempts(step_ctx))
        if not attempt_specs:
            raise ValueError("build_attempts() must return at least one attempt")

        result: dict[str, object] | None = None
        step_time = 0.0
        used_attempt = attempt_specs[0]

        for idx, attempt in enumerate(attempt_specs):
            result, step_time = solve_attempt(step_ctx, attempt)
            used_attempt = attempt
            if should_retry(result, step_ctx) and idx + 1 < len(attempt_specs):
                if on_retry is not None:
                    on_retry(step_ctx, attempt, idx + 1, len(attempt_specs))
                continue
            break

        if result is None:
            raise RuntimeError("Load-step driver did not receive a nonlinear result")

        step_record = build_step_record(step_ctx, result, step_time, used_attempt)
        step_records.append(step_record)

        if on_step_complete is not None:
            on_step_complete(step_record, step_ctx)
        if should_stop is not None and should_stop(step_record, result, step_ctx):
            break

    return step_records


def build_load_step_result(
    *,
    mesh_level: int,
    total_dofs: int,
    setup_time: float,
    total_runtime_start: float,
    steps: Sequence[Mapping[str, Any]],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common outer JSON envelope for load-step solvers."""
    payload: dict[str, Any] = {
        "mesh_level": int(mesh_level),
        "total_dofs": int(total_dofs),
        "setup_time": float(round(setup_time, 6)),
        "solve_time_total": float(
            round(sum(float(step.get("time", 0.0)) for step in steps), 6)
        ),
        "total_time": float(round(time.perf_counter() - total_runtime_start, 6)),
        "steps": [dict(step) for step in steps],
    }
    if extra:
        payload.update(extra)
    return payload

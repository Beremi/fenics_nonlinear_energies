"""Thesis-faithful ray mountain pass algorithm."""

from __future__ import annotations

import numpy as np

from src.problems.plaplace_u3.thesis.directions import build_direction_context
from src.problems.plaplace_u3.thesis.functionals import compute_state_stats_free
from src.problems.plaplace_u3.thesis.presets import THESIS_MAXIT_RMPA_OA, THESIS_RMPA_DELTA0
from src.problems.plaplace_u3.thesis.solver_common import (
    build_objective_bundle,
    build_result_payload,
    ThesisProblem,
)


def run_rmpa(
    problem: ThesisProblem,
    *,
    direction: str,
    epsilon: float,
    maxit: int = THESIS_MAXIT_RMPA_OA,
    delta0: float = THESIS_RMPA_DELTA0,
    reference_error_w1p: float | None = None,
    state_out: str = "",
) -> dict[str, object]:
    """Run the thesis ray mountain pass algorithm on one problem."""
    objective = build_objective_bundle(problem, "J")
    directions = build_direction_context(problem, objective)

    current = np.asarray(problem.u_init, dtype=np.float64).copy()
    current *= float(problem.stats(current).scale_to_solution)

    history: list[dict[str, object]] = []
    direction_solves = 0
    message = "Maximum number of iterations reached"
    status = "maxit"

    for outer_it in range(1, int(maxit) + 1):
        current_stats = compute_state_stats_free(problem.params, current)
        dir_result = directions.compute(current, direction)
        direction_solves += int(dir_result.direction_solves)

        if dir_result.stop_measure <= float(epsilon):
            message = f"Stopping criterion {dir_result.stop_name} satisfied"
            status = "completed"
            break

        step_dir = np.asarray(dir_result.direction, dtype=np.float64).copy()
        accepted = False
        projected = current
        projected_stats = current_stats
        halves = 0

        while halves <= 60:
            trial = np.asarray(current + float(delta0) * step_dir, dtype=np.float64)
            trial_stats = compute_state_stats_free(problem.params, trial)
            if np.isfinite(trial_stats.scale_to_solution):
                projected_candidate = trial * float(trial_stats.scale_to_solution)
                projected_candidate_stats = compute_state_stats_free(problem.params, projected_candidate)
                if projected_candidate_stats.J < current_stats.J:
                    projected = projected_candidate
                    projected_stats = projected_candidate_stats
                    accepted = True
                    break
            step_dir *= 0.5
            halves += 1

        history.append(
            {
                "outer_it": int(outer_it),
                "J": float(current_stats.J),
                "I": float(current_stats.I),
                "c": float(current_stats.c),
                "stop_measure": float(dir_result.stop_measure),
                "stop_name": str(dir_result.stop_name),
                "descent_value": float(dir_result.descent_value),
                "delta0": float(delta0),
                "halves": int(halves),
                "accepted": bool(accepted),
                "trial_J": float(projected_stats.J),
                "trial_I": float(projected_stats.I),
            }
        )

        if not accepted:
            message = "RMPA halving failed to reduce the ray maximum"
            status = "failed"
            break

        current = np.asarray(projected, dtype=np.float64)
    else:
        current_stats = compute_state_stats_free(problem.params, current)

    return build_result_payload(
        method="rmpa",
        direction=direction,
        problem=problem,
        epsilon=float(epsilon),
        iterate_free=current,
        history=history,
        message=message,
        status=status,
        direction_solves=direction_solves,
        reference_error_w1p=reference_error_w1p,
        state_out=state_out,
        extra={
            "delta0": float(delta0),
            "objective_name": "J",
        },
    )

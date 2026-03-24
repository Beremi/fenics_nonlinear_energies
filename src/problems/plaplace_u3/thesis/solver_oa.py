"""Thesis-faithful optimization algorithms OA1 and OA2."""

from __future__ import annotations

import numpy as np

from src.core.serial.minimizers import golden_section_search
from src.problems.plaplace_u3.thesis.directions import build_direction_context
from src.problems.plaplace_u3.thesis.presets import (
    THESIS_MAXIT_RMPA_OA,
    THESIS_OA_DELTA_HAT,
    THESIS_OA_GOLDEN_TOL,
)
from src.problems.plaplace_u3.thesis.solver_common import (
    ThesisProblem,
    build_objective_bundle,
    build_result_payload,
)


def run_oa(
    problem: ThesisProblem,
    *,
    variant: str,
    direction: str,
    epsilon: float,
    maxit: int = THESIS_MAXIT_RMPA_OA,
    delta_hat: float = THESIS_OA_DELTA_HAT,
    golden_tol: float = THESIS_OA_GOLDEN_TOL,
    reference_error_w1p: float | None = None,
    state_out: str = "",
) -> dict[str, object]:
    """Run OA1 or OA2 on one thesis problem."""
    variant = str(variant).lower()
    if variant not in {"oa1", "oa2"}:
        raise ValueError(f"Unsupported OA variant {variant!r}")

    objective = build_objective_bundle(problem, "I")
    directions = build_direction_context(problem, objective)
    current = np.asarray(problem.u_init, dtype=np.float64).copy()

    history: list[dict[str, object]] = []
    direction_solves = 0
    message = "Maximum number of iterations reached"
    status = "maxit"

    for outer_it in range(1, int(maxit) + 1):
        raw_stats = problem.stats(current)
        dir_result = directions.compute(current, direction)
        direction_solves += int(dir_result.direction_solves)

        if dir_result.stop_measure <= float(epsilon):
            message = f"Stopping criterion {dir_result.stop_name} satisfied"
            status = "completed"
            break

        delta = min(float(delta_hat), 0.5 * float(raw_stats.seminorm_p))
        accepted = False
        alpha = 0.0
        halves = 0
        line_search_evals = 0
        trial = np.asarray(current, dtype=np.float64)
        trial_I = float(raw_stats.I)

        if variant == "oa1":
            step_dir = np.asarray(dir_result.direction, dtype=np.float64).copy()
            while halves <= 60:
                candidate = np.asarray(current + float(delta) * step_dir, dtype=np.float64)
                candidate_I = float(objective.value(candidate))
                if np.isfinite(candidate_I) and candidate_I < raw_stats.I:
                    trial = candidate
                    trial_I = candidate_I
                    alpha = float(delta / (2**halves))
                    accepted = True
                    break
                step_dir *= 0.5
                halves += 1
                line_search_evals += 1
        else:
            def _phi(step: float) -> float:
                nonlocal line_search_evals
                line_search_evals += 1
                return float(objective.value(current + float(step) * dir_result.direction))

            alpha_star, _ = golden_section_search(_phi, 0.0, float(delta), float(golden_tol))
            candidate = np.asarray(current + float(alpha_star) * dir_result.direction, dtype=np.float64)
            candidate_I = float(objective.value(candidate))
            if np.isfinite(candidate_I) and candidate_I < raw_stats.I:
                trial = candidate
                trial_I = candidate_I
                alpha = float(alpha_star)
                accepted = True

        trial_physical_stats = problem.stats(trial)
        trial_physical_scale = float(trial_physical_stats.scale_to_solution)

        history.append(
            {
                "outer_it": int(outer_it),
                "I": float(raw_stats.I),
                "c": float(raw_stats.c),
                "raw_J": float(raw_stats.J),
                "stop_measure": float(dir_result.stop_measure),
                "stop_name": str(dir_result.stop_name),
                "descent_value": float(dir_result.descent_value),
                "delta": float(delta),
                "alpha": float(alpha),
                "halves": int(halves),
                "accepted": bool(accepted),
                "trial_I": float(trial_I),
                "trial_scale": float(trial_physical_scale),
                "line_search_evals": int(line_search_evals),
            }
        )

        if not accepted:
            message = f"{variant.upper()} line search failed to decrease I"
            status = "failed"
            break

        current = np.asarray(trial, dtype=np.float64)
    else:
        raw_stats = problem.stats(current)

    return build_result_payload(
        method=variant,
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
            "delta_hat": float(delta_hat),
            "golden_tol": float(golden_tol),
            "objective_name": "I",
        },
    )

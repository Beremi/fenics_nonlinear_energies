"""Thesis-faithful ray mountain pass algorithm.

RMPA evolves directly on the energy ``J``. Each trial point is projected back to
the analytic ray maximizer before acceptance, which is why the line-search logic
is written in terms of the projected energy rather than the raw iterate.
"""

from __future__ import annotations

import numpy as np

from src.core.serial.minimizers import golden_section_search
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
    step_search: str = "halving",
    reference_error_w1p: float | None = None,
    state_out: str = "",
) -> dict[str, object]:
    """Run the thesis ray mountain pass algorithm on one problem."""
    step_search = str(step_search).lower()
    if step_search not in {"halving", "golden"}:
        raise ValueError(f"Unsupported RMPA step_search={step_search!r}")
    objective = build_objective_bundle(problem, "J")
    directions = build_direction_context(problem, objective)

    current = np.asarray(problem.u_init, dtype=np.float64).copy()
    current *= float(problem.stats(current).scale_to_solution)

    history: list[dict[str, object]] = []
    direction_solves = 0
    message = "Maximum number of iterations reached"
    status = "maxit"
    best_stop_measure: float | None = None
    best_stop_outer_it: int | None = None
    accepted_step_count = 0
    max_halves = 0
    final_halves = 0

    for outer_it in range(1, int(maxit) + 1):
        current_stats = compute_state_stats_free(problem.params, current)
        dir_result = directions.compute(current, direction)
        direction_solves += int(dir_result.direction_solves)
        stop_measure = float(dir_result.stop_measure)
        if best_stop_measure is None or stop_measure < best_stop_measure:
            best_stop_measure = stop_measure
            best_stop_outer_it = int(outer_it)

        if stop_measure <= float(epsilon):
            message = f"Stopping criterion {dir_result.stop_name} satisfied"
            status = "completed"
            break

        step_dir = np.asarray(dir_result.direction, dtype=np.float64).copy()
        accepted = False
        projected = current
        projected_stats = current_stats
        halves = 0
        alpha = 0.0

        def _try_projected_candidate(step: float) -> tuple[bool, np.ndarray, object]:
            trial = np.asarray(current + float(step) * step_dir, dtype=np.float64)
            trial_stats = compute_state_stats_free(problem.params, trial)
            if not np.isfinite(trial_stats.scale_to_solution):
                return False, projected, projected_stats
            projected_candidate = trial * float(trial_stats.scale_to_solution)
            projected_candidate_stats = compute_state_stats_free(problem.params, projected_candidate)
            if projected_candidate_stats.J < current_stats.J:
                return True, projected_candidate, projected_candidate_stats
            return False, projected_candidate, projected_candidate_stats

        if step_search == "golden":
            # The 1D thesis study uses a golden-section Step 6 variant. The
            # objective here is the energy of the ray-projected candidate.
            def _phi(step: float) -> float:
                trial = np.asarray(current + float(step) * step_dir, dtype=np.float64)
                trial_stats = compute_state_stats_free(problem.params, trial)
                if not np.isfinite(trial_stats.scale_to_solution):
                    return np.inf
                projected_candidate = trial * float(trial_stats.scale_to_solution)
                return float(compute_state_stats_free(problem.params, projected_candidate).J)

            alpha_star, _ = golden_section_search(_phi, 0.0, float(delta0), 1.0e-5)
            accepted, projected, projected_stats = _try_projected_candidate(alpha_star)
            if accepted:
                alpha = float(alpha_star)
            else:
                # When the golden minimizer sits effectively on the boundary,
                # tiny positive steps can still reduce the projected energy.
                for halves in range(0, 61):
                    alpha_try = float(delta0 / (2**halves))
                    accepted, projected, projected_stats = _try_projected_candidate(alpha_try)
                    if accepted:
                        alpha = alpha_try
                        break
        else:
            # The default square runs follow the thesis halving search until the
            # projected energy drops below the current ray maximum.
            while halves <= 60:
                alpha_try = float(delta0 / (2**halves))
                accepted, projected, projected_stats = _try_projected_candidate(alpha_try)
                if accepted:
                    alpha = alpha_try
                    break
                halves += 1
        max_halves = max(max_halves, int(halves))
        final_halves = int(halves)
        if accepted:
            accepted_step_count += 1

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
                "alpha": float(alpha),
                "halves": int(halves),
                "accepted": bool(accepted),
                "trial_J": float(projected_stats.J),
                "trial_I": float(projected_stats.I),
                "step_search": step_search,
            }
        )

        if not accepted:
            message = f"RMPA {step_search} step search failed to reduce the ray maximum"
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
            "configured_maxit": int(maxit),
            "best_stop_measure": None if best_stop_measure is None else float(best_stop_measure),
            "best_stop_outer_it": None if best_stop_outer_it is None else int(best_stop_outer_it),
            "accepted_step_count": int(accepted_step_count),
            "max_halves": int(max_halves),
            "final_halves": int(final_halves),
            "delta0": float(delta0),
            "step_search": step_search,
            "objective_name": "J",
        },
    )

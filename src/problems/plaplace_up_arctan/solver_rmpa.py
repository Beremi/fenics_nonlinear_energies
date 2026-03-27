"""Projected mountain-pass algorithms for the arctan-resonance energy."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.core.serial.minimizers import golden_section_search
from src.problems.plaplace_up_arctan.directions import (
    DIRECTION_MODEL_DVH,
    build_direction_context,
)
from src.problems.plaplace_up_arctan.solver_common import (
    ARCTAN_SOLVER_REVISION,
    ObjectiveBundle,
    ProblemInstance,
    build_objective_bundle,
    build_result_payload,
)


ProjectionResult = tuple[np.ndarray, float, float] | None


def _phi(objective: ObjectiveBundle, base_free: np.ndarray, scale: float) -> float:
    return float(objective.value(float(scale) * np.asarray(base_free, dtype=np.float64)))


def _bracket_peak_1d(
    phi: Callable[[float], float],
    *,
    initial_scale: float = 1.0,
    min_scale: float = 1.0e-8,
    max_scale: float = 1.0e6,
) -> tuple[float, float] | None:
    t_prev = 0.0
    f_prev = float(phi(t_prev))
    t_curr = float(initial_scale)
    f_curr = float(phi(t_curr))

    while (not np.isfinite(f_curr) or f_curr <= f_prev) and abs(t_curr) > float(min_scale):
        t_curr *= 0.5
        f_curr = float(phi(t_curr))
    if (not np.isfinite(f_curr)) or f_curr <= f_prev:
        return None

    while abs(t_curr) < float(max_scale):
        t_next = 2.0 * t_curr
        f_next = float(phi(t_next))
        if (not np.isfinite(f_next)) or f_next < f_curr:
            return (min(t_prev, t_next), max(t_prev, t_next))
        t_prev, f_prev = t_curr, f_curr
        t_curr, f_curr = t_next, f_next
    return None


def project_to_ray_max(
    objective: ObjectiveBundle,
    base_free: np.ndarray,
) -> ProjectionResult:
    base_free = np.asarray(base_free, dtype=np.float64)
    if np.linalg.norm(base_free) <= 0.0 or not np.all(np.isfinite(base_free)):
        return None

    bracket = _bracket_peak_1d(lambda scale: _phi(objective, base_free, scale))
    if bracket is None:
        return None

    def _neg_phi(scale: float) -> float:
        return -_phi(objective, base_free, scale)

    best_scale, _ = golden_section_search(_neg_phi, bracket[0], bracket[1], 1.0e-5)
    best_value = _phi(objective, base_free, best_scale)
    projected = best_scale * base_free
    return np.asarray(projected, dtype=np.float64), float(best_scale), float(best_value)


def _choose_negative_anchor_scale(
    objective: ObjectiveBundle,
    seed_free: np.ndarray,
    *,
    max_scale: float = 1.0e6,
) -> float | None:
    scale = 1.0
    seed_free = np.asarray(seed_free, dtype=np.float64)
    while float(objective.value(-scale * seed_free)) >= 0.0:
        scale *= 2.0
        if scale > float(max_scale):
            return None
    return float(scale)


def _phi_shifted(
    objective: ObjectiveBundle,
    anchor_free: np.ndarray,
    base_free: np.ndarray,
    scale: float,
) -> float:
    anchor_free = np.asarray(anchor_free, dtype=np.float64)
    base_free = np.asarray(base_free, dtype=np.float64)
    direction = base_free - anchor_free
    return float(objective.value(anchor_free + float(scale) * direction))


def project_to_shifted_line_max(
    objective: ObjectiveBundle,
    anchor_free: np.ndarray,
    base_free: np.ndarray,
) -> ProjectionResult:
    anchor_free = np.asarray(anchor_free, dtype=np.float64)
    base_free = np.asarray(base_free, dtype=np.float64)
    direction = base_free - anchor_free
    if np.linalg.norm(direction) <= 0.0 or not np.all(np.isfinite(direction)) or not np.all(np.isfinite(anchor_free)):
        return None

    bracket = _bracket_peak_1d(lambda scale: _phi_shifted(objective, anchor_free, base_free, scale))
    if bracket is None:
        return None

    def _neg_phi(scale: float) -> float:
        return -_phi_shifted(objective, anchor_free, base_free, scale)

    best_scale, _ = golden_section_search(_neg_phi, bracket[0], bracket[1], 1.0e-5)
    projected = anchor_free + float(best_scale) * direction
    best_value = float(objective.value(projected))
    return np.asarray(projected, dtype=np.float64), float(best_scale), float(best_value)


def _failed_payload(
    *,
    method_name: str,
    problem: ProblemInstance,
    epsilon: float,
    maxit: int,
    delta0: float,
    iterate_free: np.ndarray,
    message: str,
    objective: ObjectiveBundle,
    reference_error_w1p: float | None,
    state_out: str,
    projection_mode: str,
    extra_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    extra = {
        "configured_maxit": int(maxit),
        "best_stop_measure": None,
        "best_stop_outer_it": None,
        "accepted_step_count": 0,
        "max_halves": 0,
        "final_halves": 0,
        "delta0": float(delta0),
        "projection_mode": str(projection_mode),
        "projection_scale": float("nan"),
        "objective_name": "J",
        "direction_model": DIRECTION_MODEL_DVH,
        "solver_revision": ARCTAN_SOLVER_REVISION,
        "reported_iterate_source": "initial_seed",
        "best_residual_norm": float("inf"),
        "best_gradient_residual_norm": float("inf"),
        "best_residual_outer_it": 0,
    }
    if extra_fields:
        extra.update(dict(extra_fields))
    return build_result_payload(
        method=str(method_name),
        problem=problem,
        epsilon=float(epsilon),
        iterate_free=np.asarray(iterate_free, dtype=np.float64),
        history=[],
        message=str(message),
        status="failed",
        direction_solves=0,
        objective=objective,
        reference_error_w1p=reference_error_w1p,
        state_out=state_out,
        extra=extra,
    )


def _run_projected_rmpa(
    *,
    method_name: str,
    problem: ProblemInstance,
    epsilon: float,
    maxit: int,
    delta0: float,
    init_free: np.ndarray | None,
    objective: ObjectiveBundle,
    projector: Callable[[np.ndarray], ProjectionResult],
    init_failure_message: str,
    halving_failure_message: str,
    reference_error_w1p: float | None,
    state_out: str,
    projection_mode: str,
    extra_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    directions = build_direction_context(problem, objective)

    seed_free = np.asarray(problem.u_init if init_free is None else init_free, dtype=np.float64)
    projected_init = projector(seed_free)
    if projected_init is None and init_free is not None:
        seed_free = np.asarray(problem.u_init, dtype=np.float64)
        projected_init = projector(seed_free)
    if projected_init is None:
        return _failed_payload(
            method_name=str(method_name),
            problem=problem,
            epsilon=float(epsilon),
            maxit=int(maxit),
            delta0=float(delta0),
            iterate_free=seed_free,
            message=str(init_failure_message),
            objective=objective,
            reference_error_w1p=reference_error_w1p,
            state_out=state_out,
            projection_mode=str(projection_mode),
            extra_fields=extra_fields,
        )

    current, projection_scale, _ = projected_init
    history: list[dict[str, object]] = []
    direction_solves = 0
    message = "Maximum number of iterations reached"
    status = "maxit"
    best_stop_measure: float | None = None
    best_stop_outer_it: int | None = None
    accepted_step_count = 0
    max_halves = 0
    final_halves = 0
    best_iterate = np.asarray(current, dtype=np.float64).copy()
    best_dual_residual = float("inf")
    best_gradient_residual = float("inf")
    best_residual_outer_it = 0

    for outer_it in range(1, int(maxit) + 1):
        current_stats = problem.stats(current)
        dir_result = directions.compute(current)
        direction_solves += int(dir_result.direction_solves)
        dual_residual = float(dir_result.dual_residual_norm)
        gradient_residual = float(dir_result.gradient_residual_norm)
        if (dual_residual, gradient_residual) < (best_dual_residual, best_gradient_residual):
            best_dual_residual = dual_residual
            best_gradient_residual = gradient_residual
            best_iterate = np.asarray(current, dtype=np.float64).copy()
            best_residual_outer_it = int(outer_it)
        stop_measure = float(dir_result.stop_measure)
        if best_stop_measure is None or stop_measure < best_stop_measure:
            best_stop_measure = stop_measure
            best_stop_outer_it = int(outer_it)
        if stop_measure <= float(epsilon):
            message = f"Stopping criterion {dir_result.stop_name} satisfied"
            status = "completed"
            break

        accepted = False
        alpha = 0.0
        halves = 0
        projected = np.asarray(current, dtype=np.float64)
        projected_scale = float(projection_scale)
        projected_value = float(current_stats.J)
        while halves <= 60:
            alpha_try = float(delta0 / (2**halves))
            trial_base = np.asarray(current + alpha_try * np.asarray(dir_result.direction, dtype=np.float64), dtype=np.float64)
            projected_candidate = projector(trial_base)
            if projected_candidate is None:
                halves += 1
                continue
            candidate_free, candidate_scale, candidate_value = projected_candidate
            if candidate_value < float(objective.value(current)):
                accepted = True
                alpha = alpha_try
                projected = candidate_free
                projected_scale = candidate_scale
                projected_value = candidate_value
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
                "stop_measure": float(dir_result.stop_measure),
                "stop_name": str(dir_result.stop_name),
                "gradient_residual_norm": float(dir_result.gradient_residual_norm),
                "dual_residual_norm": float(dir_result.dual_residual_norm),
                "descent_value": float(dir_result.descent_value),
                "direction_model": str(dir_result.direction_model),
                "hessian_shift": float(dir_result.hessian_shift),
                "hessian_attempts": int(dir_result.hessian_attempts),
                "delta0": float(delta0),
                "alpha": float(alpha),
                "halves": int(halves),
                "accepted": bool(accepted),
                "trial_algorithm_value": float(projected_value),
                "projection_mode": str(projection_mode),
                "projection_scale": float(projected_scale),
                "used_gradient_fallback": bool(dir_result.used_gradient_fallback),
            }
        )
        if not accepted:
            message = str(halving_failure_message)
            status = "failed"
            break

        current = np.asarray(projected, dtype=np.float64)
        projection_scale = float(projected_scale)

    reported_iterate = np.asarray(current if status == "completed" else best_iterate, dtype=np.float64)
    extra = {
        "configured_maxit": int(maxit),
        "best_stop_measure": None if best_stop_measure is None else float(best_stop_measure),
        "best_stop_outer_it": None if best_stop_outer_it is None else int(best_stop_outer_it),
        "accepted_step_count": int(accepted_step_count),
        "max_halves": int(max_halves),
        "final_halves": int(final_halves),
        "delta0": float(delta0),
        "projection_mode": str(projection_mode),
        "projection_scale": float(projection_scale),
        "objective_name": "J",
        "direction_model": str(history[-1]["direction_model"]) if history else DIRECTION_MODEL_DVH,
        "solver_revision": ARCTAN_SOLVER_REVISION,
        "reported_iterate_source": "final" if status == "completed" else "best_dual_residual",
        "best_residual_norm": float(best_dual_residual),
        "best_gradient_residual_norm": float(best_gradient_residual),
        "best_residual_outer_it": int(best_residual_outer_it),
    }
    if extra_fields:
        extra.update(dict(extra_fields))
    if str(projection_mode) == "ray":
        extra["ray_scale"] = float(projection_scale)
    if str(projection_mode) == "shifted_line":
        extra["line_scale"] = float(projection_scale)

    return build_result_payload(
        method=str(method_name),
        problem=problem,
        epsilon=float(epsilon),
        iterate_free=reported_iterate,
        history=history,
        message=message,
        status=status,
        direction_solves=direction_solves,
        objective=objective,
        reference_error_w1p=reference_error_w1p,
        state_out=state_out,
        extra=extra,
    )


def run_rmpa(
    problem: ProblemInstance,
    *,
    epsilon: float,
    maxit: int = 250,
    delta0: float = 1.0,
    init_free: np.ndarray | None = None,
    reference_error_w1p: float | None = None,
    state_out: str = "",
) -> dict[str, object]:
    objective = build_objective_bundle(problem, "J")
    return _run_projected_rmpa(
        method_name="rmpa",
        problem=problem,
        epsilon=float(epsilon),
        maxit=int(maxit),
        delta0=float(delta0),
        init_free=None if init_free is None else np.asarray(init_free, dtype=np.float64),
        objective=objective,
        projector=lambda trial_base: project_to_ray_max(objective, trial_base),
        init_failure_message="Initial seed does not admit a finite ray projection",
        halving_failure_message="RMPA halving failed to reduce the projected ray maximum",
        reference_error_w1p=reference_error_w1p,
        state_out=state_out,
        projection_mode="ray",
    )


def run_rmpa_shifted(
    problem: ProblemInstance,
    *,
    epsilon: float,
    maxit: int = 250,
    delta0: float = 1.0,
    init_free: np.ndarray | None = None,
    reference_error_w1p: float | None = None,
    state_out: str = "",
) -> dict[str, object]:
    objective = build_objective_bundle(problem, "J")
    seed_free = np.asarray(problem.u_init if init_free is None else init_free, dtype=np.float64)
    anchor_scale = _choose_negative_anchor_scale(objective, seed_free)
    if anchor_scale is None and init_free is not None:
        seed_free = np.asarray(problem.u_init, dtype=np.float64)
        anchor_scale = _choose_negative_anchor_scale(objective, seed_free)
    if anchor_scale is None:
        return _failed_payload(
            method_name="rmpa_shifted",
            problem=problem,
            epsilon=float(epsilon),
            maxit=int(maxit),
            delta0=float(delta0),
            iterate_free=np.asarray(problem.u_init if init_free is None else init_free, dtype=np.float64),
            message="Could not find a negative anchor with J(anchor) < 0 for shifted-line projection",
            objective=objective,
            reference_error_w1p=reference_error_w1p,
            state_out=state_out,
            projection_mode="shifted_line",
            extra_fields={
                "anchor_scale": float("nan"),
                "anchor_energy": float("nan"),
            },
        )

    anchor_free = -float(anchor_scale) * np.asarray(seed_free, dtype=np.float64)
    anchor_energy = float(objective.value(anchor_free))
    return _run_projected_rmpa(
        method_name="rmpa_shifted",
        problem=problem,
        epsilon=float(epsilon),
        maxit=int(maxit),
        delta0=float(delta0),
        init_free=None if init_free is None else np.asarray(init_free, dtype=np.float64),
        objective=objective,
        projector=lambda trial_base: project_to_shifted_line_max(objective, anchor_free, trial_base),
        init_failure_message="Initial seed does not admit a finite shifted-line projection",
        halving_failure_message="Shifted-line RMPA halving failed to reduce the projected line maximum",
        reference_error_w1p=reference_error_w1p,
        state_out=state_out,
        projection_mode="shifted_line",
        extra_fields={
            "anchor_scale": float(anchor_scale),
            "anchor_energy": float(anchor_energy),
        },
    )

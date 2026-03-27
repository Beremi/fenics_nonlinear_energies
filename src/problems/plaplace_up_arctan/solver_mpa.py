"""Polygonal-chain mountain-pass algorithm for the arctan-resonance energy."""

from __future__ import annotations

import numpy as np

from src.problems.plaplace_up_arctan.directions import (
    DIRECTION_MODEL_DVH,
    build_direction_context,
)
from src.problems.plaplace_up_arctan.functionals import seminorm_full
from src.problems.plaplace_up_arctan.solver_common import (
    ARCTAN_SOLVER_REVISION,
    build_objective_bundle,
    build_result_payload,
    ObjectiveBundle,
    ProblemInstance,
)
from src.problems.plaplace_up_arctan.solver_rmpa import project_to_ray_max


def _quadratic_extremum(xs: list[float], ys: list[float], *, maximize: bool) -> float:
    coeff = np.polyfit(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), deg=2)
    a, b, _ = coeff
    if not np.isfinite(a) or abs(a) < 1.0e-14:
        idx = int(np.argmax(ys) if maximize else np.argmin(ys))
        return float(xs[idx])
    x_star = -b / (2.0 * a)
    lo = float(min(xs))
    hi = float(max(xs))
    x_star = float(np.clip(x_star, lo, hi))
    if maximize and a >= 0.0:
        return float(xs[int(np.argmax(ys))])
    if (not maximize) and a <= 0.0:
        return float(xs[int(np.argmin(ys))])
    return x_star


def _seminorm_difference(problem: ProblemInstance, a: np.ndarray, b: np.ndarray) -> float:
    diff_full = problem.expand_free(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return float(seminorm_full(problem.params, diff_full, exponent=problem.p))


def _search_neighbor_peak(
    problem: ProblemInstance,
    objective: ObjectiveBundle,
    z: np.ndarray,
    neighbor: np.ndarray,
    delta_min: float,
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    neighbor = np.asarray(neighbor, dtype=np.float64)
    fz = float(objective.value(z))
    xi = 0.5 * (z + neighbor)

    while _seminorm_difference(problem, z, xi) >= float(delta_min):
        f_xi = float(objective.value(xi))
        if f_xi > fz:
            xi_p = 0.5 * (z + xi)
            f_xi_p = float(objective.value(xi_p))
            t_star = _quadratic_extremum([0.0, 0.5, 1.0], [fz, f_xi_p, f_xi], maximize=True)
            candidate = z + t_star * (xi - z)
            f_candidate = float(objective.value(candidate))
            return np.asarray(candidate if f_candidate > fz else xi, dtype=np.float64)
        xi = 0.5 * (z + xi)
    return z


def _search_neighbor_lower(
    problem: ProblemInstance,
    objective: ObjectiveBundle,
    z: np.ndarray,
    neighbor: np.ndarray,
    delta_min: float,
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    neighbor = np.asarray(neighbor, dtype=np.float64)
    fz = float(objective.value(z))
    xi = 0.5 * (z + neighbor)

    while _seminorm_difference(problem, z, xi) >= float(delta_min):
        f_xi = float(objective.value(xi))
        if f_xi < fz:
            xi_p = 0.5 * (z + xi)
            f_xi_p = float(objective.value(xi_p))
            t_star = _quadratic_extremum([0.0, 0.5, 1.0], [fz, f_xi_p, f_xi], maximize=False)
            candidate = z + t_star * (xi - z)
            f_candidate = float(objective.value(candidate))
            return np.asarray(candidate if f_candidate < fz else xi, dtype=np.float64)
        xi = 0.5 * (z + xi)
    return z


def _repair_local_path_after_move(
    problem: ProblemInstance,
    objective: ObjectiveBundle,
    candidate: np.ndarray,
    left_neighbor: np.ndarray,
    right_neighbor: np.ndarray,
    delta_min: float,
) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=np.float64)
    candidate = _search_neighbor_lower(problem, objective, candidate, left_neighbor, delta_min)
    candidate = _search_neighbor_lower(problem, objective, candidate, right_neighbor, delta_min)
    return np.asarray(candidate, dtype=np.float64)


def _descent_step(
    problem: ProblemInstance,
    objective: ObjectiveBundle,
    z: np.ndarray,
    direction: np.ndarray,
    rho: float,
    left_neighbor: np.ndarray,
    right_neighbor: np.ndarray,
    delta_min: float,
) -> tuple[np.ndarray, bool, int]:
    z = np.asarray(z, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    fz = float(objective.value(z))
    step = float(rho) * direction
    halves = 0

    while halves <= 60:
        trial = z + step
        repaired_trial = _repair_local_path_after_move(
            problem,
            objective,
            trial,
            left_neighbor,
            right_neighbor,
            delta_min,
        )
        f_trial = float(objective.value(repaired_trial))
        if f_trial < fz:
            mid = z + 0.5 * step
            repaired_mid = _repair_local_path_after_move(
                problem,
                objective,
                mid,
                left_neighbor,
                right_neighbor,
                delta_min,
            )
            f_mid = float(objective.value(repaired_mid))
            values = [fz, f_mid, f_trial]
            t_star = _quadratic_extremum([0.0, 0.5, 1.0], values, maximize=False)
            candidate = _repair_local_path_after_move(
                problem,
                objective,
                z + t_star * step,
                left_neighbor,
                right_neighbor,
                delta_min,
            )
            f_candidate = float(objective.value(candidate))
            choices = [(f_trial, repaired_trial), (f_mid, repaired_mid)]
            if f_candidate < fz:
                choices.append((f_candidate, candidate))
            best_value, best_point = min(choices, key=lambda item: item[0])
            if best_value < fz:
                return np.asarray(best_point, dtype=np.float64), True, halves
        step *= 0.5
        halves += 1
    return z, False, halves


def _choose_positive_endpoint_scale(problem: ProblemInstance, objective: ObjectiveBundle, seed_free: np.ndarray) -> float:
    scale = 1.0
    seed_free = np.asarray(seed_free, dtype=np.float64)
    while float(objective.value(scale * seed_free)) >= 0.0:
        scale *= 2.0
        if scale > 1.0e6:
            break
    return float(scale)


def _choose_negative_endpoint_scale(problem: ProblemInstance, objective: ObjectiveBundle, seed_free: np.ndarray) -> float | None:
    scale = 1.0
    seed_free = np.asarray(seed_free, dtype=np.float64)
    while float(objective.value(-scale * seed_free)) >= 0.0:
        scale *= 2.0
        if scale > 1.0e6:
            return None
    return float(scale)


def _run_mpa(
    problem: ProblemInstance,
    *,
    method_name: str,
    endpoint_mode: str,
    epsilon: float,
    maxit: int = 500,
    num_nodes: int = 50,
    rho: float = 1.0,
    segment_tol_factor: float = 0.125,
    init_free: np.ndarray | None = None,
    reference_error_w1p: float | None = None,
    state_out: str = "",
) -> dict[str, object]:
    objective = build_objective_bundle(problem, "J")
    directions = build_direction_context(problem, objective)

    seed_free = np.asarray(problem.u_init if init_free is None else init_free, dtype=np.float64)
    projected_init = project_to_ray_max(objective, seed_free)
    if projected_init is None and init_free is not None:
        seed_free = np.asarray(problem.u_init, dtype=np.float64)
        projected_init = project_to_ray_max(objective, seed_free)
    positive_endpoint_scale = _choose_positive_endpoint_scale(problem, objective, seed_free)
    endpoint = positive_endpoint_scale * seed_free
    negative_endpoint_scale = None
    if str(endpoint_mode) == "symmetric":
        negative_endpoint_scale = _choose_negative_endpoint_scale(problem, objective, seed_free)
        if negative_endpoint_scale is None:
            negative_endpoint_scale = float(positive_endpoint_scale)
        left_endpoint = -float(negative_endpoint_scale) * seed_free
        right_endpoint = np.asarray(endpoint, dtype=np.float64)
        nodes = [
            np.asarray((1.0 - alpha) * left_endpoint + alpha * right_endpoint, dtype=np.float64)
            for alpha in np.linspace(0.0, 1.0, int(num_nodes))
        ]
    else:
        nodes = [alpha * endpoint for alpha in np.linspace(0.0, 1.0, int(num_nodes))]
    initial_spacing = _seminorm_difference(problem, nodes[1], nodes[0])
    delta_min = float(segment_tol_factor) * float(initial_spacing)

    history: list[dict[str, object]] = []
    direction_solves = 0
    message = "Maximum number of iterations reached"
    status = "maxit"
    current = np.asarray(seed_free if projected_init is None else projected_init[0], dtype=np.float64)
    reported_peak = np.asarray(current, dtype=np.float64)
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
        energies = np.asarray([float(objective.value(node)) for node in nodes], dtype=np.float64)
        inner_idx = int(np.argmax(energies[1:-1])) + 1
        z = np.asarray(nodes[inner_idx], dtype=np.float64)
        left_neighbor = np.asarray(nodes[inner_idx - 1], dtype=np.float64)
        right_neighbor = np.asarray(nodes[inner_idx + 1], dtype=np.float64)

        z = _search_neighbor_peak(problem, objective, z, left_neighbor, delta_min)
        z = _search_neighbor_peak(problem, objective, z, right_neighbor, delta_min)
        nodes[inner_idx] = np.asarray(z, dtype=np.float64)
        current = np.asarray(z, dtype=np.float64)
        reported_peak = np.asarray(current, dtype=np.float64).copy()
        current_stats = problem.stats(current)

        dir_result = directions.compute(current)
        stop_measure = float(dir_result.stop_measure)
        direction_solves += int(dir_result.direction_solves)
        dual_residual = float(dir_result.dual_residual_norm)
        gradient_residual = float(dir_result.gradient_residual_norm)
        if (dual_residual, gradient_residual) < (best_dual_residual, best_gradient_residual):
            best_dual_residual = dual_residual
            best_gradient_residual = gradient_residual
            best_iterate = np.asarray(current, dtype=np.float64).copy()
            best_residual_outer_it = int(outer_it)
        if best_stop_measure is None or stop_measure < best_stop_measure:
            best_stop_measure = stop_measure
            best_stop_outer_it = int(outer_it)
        if stop_measure <= float(epsilon):
            message = f"Stopping criterion {dir_result.stop_name} satisfied"
            status = "completed"
            break

        moved, accepted, halves = _descent_step(
            problem,
            objective,
            current,
            dir_result.direction,
            float(rho),
            left_neighbor,
            right_neighbor,
            delta_min,
        )
        max_halves = max(max_halves, int(halves))
        final_halves = int(halves)
        if accepted:
            accepted_step_count += 1

        history.append(
            {
                "outer_it": int(outer_it),
                "J": float(current_stats.J),
                "stop_measure": stop_measure,
                "stop_name": str(dir_result.stop_name),
                "gradient_residual_norm": float(dir_result.gradient_residual_norm),
                "dual_residual_norm": float(dir_result.dual_residual_norm),
                "descent_value": float(dir_result.descent_value),
                "direction_model": str(dir_result.direction_model),
                "hessian_shift": float(dir_result.hessian_shift),
                "hessian_attempts": int(dir_result.hessian_attempts),
                "accepted": bool(accepted),
                "halves": int(halves),
                "max_path_energy": float(np.max(energies)),
                "node_index": int(inner_idx),
                "used_gradient_fallback": bool(dir_result.used_gradient_fallback),
            }
        )
        if not accepted:
            message = "MPA descent step failed to lower J on the polygonal chain"
            status = "failed"
            break

        nodes[inner_idx] = np.asarray(moved, dtype=np.float64)
        current = np.asarray(nodes[inner_idx], dtype=np.float64)
    reported_iterate = np.asarray(reported_peak if status == "completed" else best_iterate, dtype=np.float64)
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
        extra={
            "configured_maxit": int(maxit),
            "best_stop_measure": None if best_stop_measure is None else float(best_stop_measure),
            "best_stop_outer_it": None if best_stop_outer_it is None else int(best_stop_outer_it),
            "accepted_step_count": int(accepted_step_count),
            "max_halves": int(max_halves),
            "final_halves": int(final_halves),
            "rho": float(rho),
            "num_nodes": int(num_nodes),
            "segment_tol_factor": float(segment_tol_factor),
            "endpoint_mode": str(endpoint_mode),
            "positive_endpoint_scale": float(positive_endpoint_scale),
            "negative_endpoint_scale": None if negative_endpoint_scale is None else float(negative_endpoint_scale),
            "objective_name": "J",
            "direction_model": str(history[-1]["direction_model"]) if history else DIRECTION_MODEL_DVH,
            "solver_revision": ARCTAN_SOLVER_REVISION,
            "reported_iterate_source": "peak" if status == "completed" else "best_dual_residual",
            "best_residual_norm": float(best_dual_residual),
            "best_gradient_residual_norm": float(best_gradient_residual),
            "best_residual_outer_it": int(best_residual_outer_it),
        },
    )


def run_mpa(
    problem: ProblemInstance,
    *,
    epsilon: float,
    maxit: int = 500,
    num_nodes: int = 50,
    rho: float = 1.0,
    segment_tol_factor: float = 0.125,
    init_free: np.ndarray | None = None,
    reference_error_w1p: float | None = None,
    state_out: str = "",
) -> dict[str, object]:
    return _run_mpa(
        problem,
        method_name="mpa",
        endpoint_mode="one_sided",
        epsilon=float(epsilon),
        maxit=int(maxit),
        num_nodes=int(num_nodes),
        rho=float(rho),
        segment_tol_factor=float(segment_tol_factor),
        init_free=None if init_free is None else np.asarray(init_free, dtype=np.float64),
        reference_error_w1p=reference_error_w1p,
        state_out=str(state_out),
    )


def run_mpa_symmetric(
    problem: ProblemInstance,
    *,
    epsilon: float,
    maxit: int = 500,
    num_nodes: int = 50,
    rho: float = 1.0,
    segment_tol_factor: float = 0.125,
    init_free: np.ndarray | None = None,
    reference_error_w1p: float | None = None,
    state_out: str = "",
) -> dict[str, object]:
    return _run_mpa(
        problem,
        method_name="mpa_symmetric",
        endpoint_mode="symmetric",
        epsilon=float(epsilon),
        maxit=int(maxit),
        num_nodes=int(num_nodes),
        rho=float(rho),
        segment_tol_factor=float(segment_tol_factor),
        init_free=None if init_free is None else np.asarray(init_free, dtype=np.float64),
        reference_error_w1p=reference_error_w1p,
        state_out=str(state_out),
    )

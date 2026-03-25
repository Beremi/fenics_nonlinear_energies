"""Classical polygonal-chain mountain pass algorithm used in the thesis.

The thesis MPA works on a discrete chain between the zero state and a large
positive endpoint. The implementation keeps that structure explicit: find the
current path peak, optionally refine the chain near that peak, then take one
descent step and repair the local path shape around the moved node.
"""

from __future__ import annotations

import numpy as np

from src.problems.plaplace_u3.thesis.directions import build_direction_context
from src.problems.plaplace_u3.thesis.functionals import seminorm_full
from src.problems.plaplace_u3.thesis.presets import (
    THESIS_MAXIT_MPA,
    THESIS_MPA_NUM_NODES,
    THESIS_MPA_RHO,
    THESIS_MPA_SEGMENT_TOL_FACTOR,
)
from src.problems.plaplace_u3.thesis.solver_common import (
    ThesisProblem,
    build_objective_bundle,
    build_result_payload,
)


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
        idx = int(np.argmax(ys))
        return float(xs[idx])
    if (not maximize) and a <= 0.0:
        idx = int(np.argmin(ys))
        return float(xs[idx])
    return x_star


def _seminorm_difference(problem: ThesisProblem, a: np.ndarray, b: np.ndarray) -> float:
    diff_full = problem.expand_free(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return float(seminorm_full(problem.params, diff_full, exponent=problem.p))


def _search_neighbor_peak(
    problem: ThesisProblem,
    objective,
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
    problem: ThesisProblem,
    objective,
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
    problem: ThesisProblem,
    objective,
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
    problem: ThesisProblem,
    objective,
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
            choices = [
                (f_trial, repaired_trial),
                (f_mid, repaired_mid),
            ]
            if f_candidate < fz:
                choices.append((f_candidate, candidate))
            best_value, best_point = min(choices, key=lambda item: item[0])
            if best_value < fz:
                return np.asarray(best_point, dtype=np.float64), True, halves
        step *= 0.5
        halves += 1
    return z, False, halves


def _descent_step_plain_halving(
    problem: ThesisProblem,
    objective,
    z: np.ndarray,
    direction: np.ndarray,
    rho: float,
    left_neighbor: np.ndarray,
    right_neighbor: np.ndarray,
    delta_min: float,
) -> tuple[np.ndarray, bool, int]:
    """Fallback thesis-style halving without interpolation."""
    z = np.asarray(z, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    fz = float(objective.value(z))
    step = float(rho) * direction
    halves = 0

    while halves <= 80:
        trial = z + step
        repaired_trial = _repair_local_path_after_move(
            problem,
            objective,
            trial,
            left_neighbor,
            right_neighbor,
            delta_min,
        )
        if float(objective.value(repaired_trial)) < fz:
            return np.asarray(repaired_trial, dtype=np.float64), True, halves
        step *= 0.5
        halves += 1
    return z, False, halves


def _refine_chain_around_peak(nodes: list[np.ndarray], peak_idx: int) -> list[np.ndarray]:
    """Insert two midpoints around a repeatedly selected peak node."""
    if len(nodes) < 7 or peak_idx <= 1 or peak_idx >= len(nodes) - 2:
        return [np.asarray(node, dtype=np.float64) for node in nodes]

    refined = [np.asarray(node, dtype=np.float64).copy() for node in nodes]
    left_mid = 0.5 * (refined[peak_idx - 1] + refined[peak_idx])
    right_mid = 0.5 * (refined[peak_idx] + refined[peak_idx + 1])

    trimmed = [refined[0], *refined[2:-2], refined[-1]]
    new_peak_idx = peak_idx - 1
    trimmed.insert(new_peak_idx, left_mid)
    trimmed.insert(new_peak_idx + 2, right_mid)
    return [np.asarray(node, dtype=np.float64) for node in trimmed]


def _detect_peak_cycle(node_history: list[int]) -> bool:
    if len(node_history) < 8:
        return False
    for cycle_len in range(1, 5):
        if len(node_history) < 2 * cycle_len:
            continue
        for end in range(2 * cycle_len, len(node_history) + 1):
            first = node_history[end - 2 * cycle_len : end - cycle_len]
            second = node_history[end - cycle_len : end]
            if first == second:
                return True
    return False


def _choose_endpoint_scale(problem: ThesisProblem, objective) -> float:
    """Pick a large endpoint whose energy is already below zero."""
    scale = max(1.0, float(problem.stats(problem.u_init).scale_to_solution))
    while float(objective.value(scale * problem.u_init)) >= 0.0:
        scale *= 2.0
        if scale > 1.0e6:  # pragma: no cover - defensive
            break
    return float(scale)


def run_mpa(
    problem: ThesisProblem,
    *,
    direction: str,
    epsilon: float,
    maxit: int = THESIS_MAXIT_MPA,
    num_nodes: int = THESIS_MPA_NUM_NODES,
    rho: float = THESIS_MPA_RHO,
    segment_tol_factor: float = THESIS_MPA_SEGMENT_TOL_FACTOR,
    reference_error_w1p: float | None = None,
    state_out: str = "",
) -> dict[str, object]:
    """Run the classical polygonal-chain MPA."""
    objective = build_objective_bundle(problem, "J")
    directions = build_direction_context(problem, objective)

    endpoint_scale = _choose_endpoint_scale(problem, objective)
    endpoint = endpoint_scale * np.asarray(problem.u_init, dtype=np.float64)
    nodes = [alpha * endpoint for alpha in np.linspace(0.0, 1.0, int(num_nodes))]
    initial_spacing = _seminorm_difference(problem, nodes[1], nodes[0])
    delta_min = float(segment_tol_factor) * float(initial_spacing)

    history: list[dict[str, object]] = []
    direction_solves = 0
    message = "Maximum number of iterations reached"
    status = "maxit"
    current = np.asarray(nodes[len(nodes) // 2], dtype=np.float64)
    reported_peak = np.asarray(current, dtype=np.float64)
    last_peak_idx: int | None = None
    same_peak_streak = 0
    best_stop_measure: float | None = None
    best_stop_outer_it: int | None = None
    accepted_step_count = 0
    max_halves = 0
    final_halves = 0
    refinement_count = 0
    node_history: list[int] = []

    for outer_it in range(1, int(maxit) + 1):
        energies = np.asarray([float(objective.value(node)) for node in nodes], dtype=np.float64)
        inner_idx = int(np.argmax(energies[1:-1])) + 1
        if inner_idx == last_peak_idx:
            same_peak_streak += 1
        else:
            same_peak_streak = 1
        last_peak_idx = inner_idx

        chain_refined = False
        if same_peak_streak >= 4:
            # If the same node remains the path maximum for several iterations,
            # add local resolution instead of repeatedly moving the coarse peak.
            nodes = _refine_chain_around_peak(nodes, inner_idx)
            energies = np.asarray([float(objective.value(node)) for node in nodes], dtype=np.float64)
            inner_idx = int(np.argmax(energies[1:-1])) + 1
            last_peak_idx = inner_idx
            same_peak_streak = 1
            chain_refined = True
            refinement_count += 1
        node_history.append(int(inner_idx))

        z = np.asarray(nodes[inner_idx], dtype=np.float64)
        left_neighbor = np.asarray(nodes[inner_idx - 1], dtype=np.float64)
        right_neighbor = np.asarray(nodes[inner_idx + 1], dtype=np.float64)

        # Thesis Step 5: locally maximize J toward each neighboring chain node.
        z = _search_neighbor_peak(problem, objective, z, left_neighbor, delta_min)
        z = _search_neighbor_peak(problem, objective, z, right_neighbor, delta_min)
        nodes[inner_idx] = np.asarray(z, dtype=np.float64)
        current = np.asarray(z, dtype=np.float64)
        reported_peak = np.asarray(current, dtype=np.float64).copy()
        current_stats = problem.stats(current)

        dir_result = directions.compute(current, direction)
        stop_measure = float(dir_result.stop_measure)
        direction_solves += int(dir_result.direction_solves)
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
        fallback_used = False
        if not accepted:
            moved, accepted, fallback_halves = _descent_step_plain_halving(
                problem,
                objective,
                current,
                dir_result.direction,
                float(rho),
                left_neighbor,
                right_neighbor,
                delta_min,
            )
            halves += int(fallback_halves)
            fallback_used = bool(accepted)
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
                "stop_measure": stop_measure,
                "stop_name": str(dir_result.stop_name),
                "descent_value": float(dir_result.descent_value),
                "accepted": bool(accepted),
                "halves": int(halves),
                "fallback_used": bool(fallback_used),
                "max_path_energy": float(np.max(energies)),
                "node_index": int(inner_idx),
                "chain_refined": bool(chain_refined),
            }
        )

        if not accepted:
            message = "MPA descent step failed to lower J on the polygonal chain"
            status = "failed"
            break

        nodes[inner_idx] = np.asarray(moved, dtype=np.float64)
        current = np.asarray(nodes[inner_idx], dtype=np.float64)

    return build_result_payload(
        method="mpa",
        direction=direction,
        problem=problem,
        epsilon=float(epsilon),
        iterate_free=reported_peak,
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
            "refinement_count": int(refinement_count),
            "distinct_peak_nodes": int(len(set(node_history))),
            "peak_cycle_detected": bool(_detect_peak_cycle(node_history)),
            "rho": float(rho),
            "num_nodes": int(num_nodes),
            "segment_tol_factor": float(segment_tol_factor),
            "objective_name": "J",
        },
    )

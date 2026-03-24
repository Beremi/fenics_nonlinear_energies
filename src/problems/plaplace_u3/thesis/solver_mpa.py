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


def _descent_step(
    objective,
    z: np.ndarray,
    direction: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, bool, int]:
    z = np.asarray(z, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    fz = float(objective.value(z))
    step = float(rho) * direction
    halves = 0

    while halves <= 60:
        trial = z + step
        f_trial = float(objective.value(trial))
        if f_trial < fz:
            mid = z + 0.5 * step
            f_mid = float(objective.value(mid))
            values = [fz, f_mid, f_trial]
            t_star = _quadratic_extremum([0.0, 0.5, 1.0], values, maximize=False)
            candidate = z + t_star * step
            f_candidate = float(objective.value(candidate))
            choices = [
                (f_trial, trial),
                (f_mid, mid),
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
    objective,
    z: np.ndarray,
    direction: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, bool, int]:
    """Fallback thesis-style halving without interpolation."""
    z = np.asarray(z, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    fz = float(objective.value(z))
    step = float(rho) * direction
    halves = 0

    while halves <= 80:
        trial = z + step
        if float(objective.value(trial)) < fz:
            return np.asarray(trial, dtype=np.float64), True, halves
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

        z = np.asarray(nodes[inner_idx], dtype=np.float64)

        # Thesis Step 5: locally maximize J toward each neighboring chain node.
        z = _search_neighbor_peak(problem, objective, z, nodes[inner_idx - 1], delta_min)
        z = _search_neighbor_peak(problem, objective, z, nodes[inner_idx + 1], delta_min)
        nodes[inner_idx] = np.asarray(z, dtype=np.float64)
        current = np.asarray(z, dtype=np.float64)
        reported_peak = np.asarray(current, dtype=np.float64).copy()
        current_stats = problem.stats(current)

        dir_result = directions.compute(current, direction)
        stop_measure = float(dir_result.stop_measure)
        direction_solves += int(dir_result.direction_solves)
        if stop_measure <= float(epsilon):
            message = f"Stopping criterion {dir_result.stop_name} satisfied"
            status = "completed"
            break

        moved, accepted, halves = _descent_step(objective, current, dir_result.direction, float(rho))
        fallback_used = False
        if not accepted:
            moved, accepted, fallback_halves = _descent_step_plain_halving(
                objective,
                current,
                dir_result.direction,
                float(rho),
            )
            halves += int(fallback_halves)
            fallback_used = bool(accepted)
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
        # Thesis Step 6: after moving the peak node downhill, repair the local
        # chain shape by searching for lower points toward each neighbor.
        nodes[inner_idx] = _search_neighbor_lower(problem, objective, nodes[inner_idx], nodes[inner_idx - 1], delta_min)
        nodes[inner_idx] = _search_neighbor_lower(problem, objective, nodes[inner_idx], nodes[inner_idx + 1], delta_min)
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
            "rho": float(rho),
            "num_nodes": int(num_nodes),
            "segment_tol_factor": float(segment_tol_factor),
            "objective_name": "J",
        },
    )

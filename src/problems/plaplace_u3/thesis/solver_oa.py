"""Thesis-faithful optimization algorithms OA1 and OA2.

Both variants minimize the scale-invariant quotient ``I``. OA1 uses a simple
halving acceptance step, while OA2 performs the thesis one-dimensional search on
the short interval ``[0, delta]``.
"""

from __future__ import annotations

import numpy as np

from src.core.serial.minimizers import golden_section_search
from src.problems.plaplace_u3.thesis.directions import build_direction_context
from src.problems.plaplace_u3.thesis.functionals import seminorm_full
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

_SKEW_SWAP_CACHE: dict[tuple[object, ...], np.ndarray] = {}


def _skew_swap_indices(problem: ThesisProblem) -> np.ndarray | None:
    if str(problem.geometry) != "square_pi" or str(problem.init_mode) != "skew":
        return None
    key = (
        str(problem.geometry),
        str(problem.init_mode),
        int(problem.dimension),
        int(problem.mesh_level),
        int(problem.free_dofs),
    )
    cached = _SKEW_SWAP_CACHE.get(key)
    if cached is not None:
        return cached

    nodes = np.asarray(problem.params["nodes"], dtype=np.float64)
    freedofs = np.asarray(problem.params["freedofs"], dtype=np.int64)
    full_to_free = {int(fd): idx for idx, fd in enumerate(freedofs)}
    coord_to_full = {
        tuple(np.round(nodes[node_idx], 12)): int(node_idx)
        for node_idx in range(nodes.shape[0])
    }
    swap = np.empty(freedofs.size, dtype=np.int64)
    for free_idx, full_idx in enumerate(freedofs):
        x, y = nodes[int(full_idx)]
        partner = coord_to_full[(round(float(y), 12), round(float(x), 12))]
        swap[free_idx] = full_to_free[int(partner)]
    _SKEW_SWAP_CACHE[key] = swap
    return swap


def _project_oa2_symmetry(problem: ThesisProblem, u_free: np.ndarray, *, variant: str) -> np.ndarray:
    # The skew square seed is antisymmetric under x <-> y, but the fixed triangle diagonal
    # orientation breaks that symmetry discretely. Projecting back to the seed's symmetry
    # class keeps OA2 on the thesis branch family instead of drifting to the principal branch.
    if str(variant).lower() != "oa2":
        return np.asarray(u_free, dtype=np.float64)
    swap = _skew_swap_indices(problem)
    if swap is None:
        return np.asarray(u_free, dtype=np.float64)
    values = np.asarray(u_free, dtype=np.float64)
    return 0.5 * (values - values[swap])


def _normalize_direction(problem: ThesisProblem, direction: np.ndarray) -> np.ndarray:
    """Normalize a direction in the discrete ``|.|_{1,p,0}`` seminorm."""
    direction = np.asarray(direction, dtype=np.float64)
    seminorm = float(
        seminorm_full(
            problem.params,
            problem.expand_free(direction),
            exponent=problem.p,
        )
    )
    if seminorm <= 0.0 or not np.isfinite(seminorm):
        return np.zeros_like(direction)
    return direction / seminorm


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
    current = _project_oa2_symmetry(problem, np.asarray(problem.u_init, dtype=np.float64).copy(), variant=variant)

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
            # OA1 keeps the simpler repeated-halving acceptance step.
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
            # OA2 minimizes I along the projected search segment [0, delta].
            line_direction = _project_oa2_symmetry(problem, dir_result.direction, variant=variant)
            if not np.allclose(line_direction, dir_result.direction):
                line_direction = _normalize_direction(problem, line_direction)

            def _phi(step: float) -> float:
                nonlocal line_search_evals
                line_search_evals += 1
                candidate = _project_oa2_symmetry(
                    problem,
                    current + float(step) * line_direction,
                    variant=variant,
                )
                return float(objective.value(candidate))

            alpha_star, _ = golden_section_search(_phi, 0.0, float(delta), float(golden_tol))
            candidate = _project_oa2_symmetry(
                problem,
                current + float(alpha_star) * line_direction,
                variant=variant,
            )
            candidate_I = float(objective.value(candidate))
            if np.isfinite(candidate_I) and candidate_I < raw_stats.I:
                trial = np.asarray(candidate, dtype=np.float64)
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

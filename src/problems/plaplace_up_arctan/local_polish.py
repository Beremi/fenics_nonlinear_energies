"""Certified local stationary solve for the arctan-resonance family."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:  # pragma: no cover - optional acceleration
    import pyamg
except ImportError:  # pragma: no cover - optional dependency
    pyamg = None

from src.problems.plaplace_up_arctan.solver_common import (
    ObjectiveBundle,
    ProblemInstance,
    build_result_payload,
)


CERTIFIED_DIRECTION_MODEL = "jax_autodiff_stationary_newton"


@dataclass
class MeritState:
    gradient: np.ndarray
    gradient_residual_norm: float
    dual_residual_norm: float
    merit: float


class _LinearSystemSolver:
    def __init__(self, matrix: sp.csr_matrix) -> None:
        self.matrix = matrix.tocsr()
        self._factorized = spla.factorized(self.matrix.tocsc()) if self.matrix.shape[0] <= 512 else None
        self._amg = None
        if self._factorized is None and pyamg is not None:
            self._amg = pyamg.smoothed_aggregation_solver(self.matrix)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=np.float64)
        if self._factorized is not None:
            return np.asarray(self._factorized(rhs), dtype=np.float64)
        if self._amg is not None:
            return np.asarray(self._amg.solve(rhs, tol=1.0e-8), dtype=np.float64)
        sol, info = spla.cg(self.matrix, rhs, rtol=1.0e-8, atol=0.0, maxiter=max(500, self.matrix.shape[0]))
        if info != 0:
            sol = spla.spsolve(self.matrix, rhs)
        return np.asarray(sol, dtype=np.float64)


def _symmetrize(matrix: sp.csr_matrix) -> sp.csr_matrix:
    matrix = matrix.tocsr()
    matrix = (0.5 * (matrix + matrix.T)).tocsr()
    matrix.eliminate_zeros()
    return matrix


def _solve_shifted(
    hessian: sp.csr_matrix,
    stiffness: sp.csr_matrix,
    rhs: np.ndarray,
    mu: float,
) -> np.ndarray | None:
    matrix = hessian if float(mu) <= 0.0 else (hessian + float(mu) * stiffness).tocsr()
    try:
        step = spla.spsolve(matrix.tocsc(), np.asarray(rhs, dtype=np.float64))
    except Exception:
        return None
    step = np.asarray(step, dtype=np.float64)
    if not np.all(np.isfinite(step)):
        return None
    return step


def merit_state(
    problem: ProblemInstance,
    objective: ObjectiveBundle,
    iterate_free: np.ndarray,
    *,
    stiffness_solver: _LinearSystemSolver | None = None,
) -> MeritState:
    iterate_free = np.asarray(iterate_free, dtype=np.float64)
    gradient = np.asarray(objective.grad(iterate_free), dtype=np.float64)
    grad_norm = float(np.linalg.norm(gradient))
    if gradient.size == 0 or grad_norm <= 0.0:
        return MeritState(
            gradient=gradient,
            gradient_residual_norm=grad_norm,
            dual_residual_norm=grad_norm,
            merit=0.5 * grad_norm * grad_norm,
        )
    solver = stiffness_solver if stiffness_solver is not None else _LinearSystemSolver(problem.stiffness)
    aux = solver.solve(gradient)
    dual_sq = float(np.dot(gradient, aux)) if np.all(np.isfinite(aux)) else float("nan")
    if not np.isfinite(dual_sq):
        dual_sq = grad_norm * grad_norm
    dual_residual = float(math.sqrt(max(dual_sq, 0.0)))
    return MeritState(
        gradient=gradient,
        gradient_residual_norm=grad_norm,
        dual_residual_norm=dual_residual,
        merit=0.5 * max(dual_sq, 0.0),
    )


def polish_stationary(
    problem: ProblemInstance,
    objective: ObjectiveBundle,
    *,
    init_free: np.ndarray,
    epsilon: float,
    maxit: int = 40,
    initial_mu: float = 1.0e-8,
    state_out: str = "",
    handoff_source: str = "raw_best_iterate",
) -> dict[str, object]:
    current = np.asarray(init_free, dtype=np.float64).copy()
    stiffness = problem.stiffness.tocsr()
    stiffness_solver = _LinearSystemSolver(stiffness)
    history: list[dict[str, object]] = []
    direction_solves = 0
    status = "maxit"
    message = "Maximum number of certification iterations reached"
    mu = float(initial_mu)
    best_iterate = current.copy()
    best_merit_state = merit_state(problem, objective, current, stiffness_solver=stiffness_solver)

    for outer_it in range(1, int(maxit) + 1):
        current_stats = problem.stats(current)
        current_merit = merit_state(problem, objective, current, stiffness_solver=stiffness_solver)
        if current_merit.dual_residual_norm < best_merit_state.dual_residual_norm:
            best_iterate = current.copy()
            best_merit_state = current_merit
        if current_merit.dual_residual_norm <= float(epsilon):
            status = "completed"
            message = "Stationarity merit converged"
            break

        hessian = _symmetrize(objective.ddf(current))
        accepted = False
        chosen_step = np.zeros_like(current)
        chosen_alpha = 0.0
        chosen_mu = mu
        chosen_merit = current_merit
        chosen_halves = 0
        regularization_attempts = 0

        attempt_mu = max(float(mu), 1.0e-12)
        for regularization_attempts in range(1, 9):
            step = _solve_shifted(hessian, stiffness, -current_merit.gradient, attempt_mu)
            direction_solves += 1
            if step is None:
                attempt_mu *= 10.0
                continue
            alpha = 1.0
            for halves in range(0, 21):
                candidate = np.asarray(current + alpha * step, dtype=np.float64)
                candidate_stats = problem.stats(candidate)
                if not np.isfinite(candidate_stats.J):
                    alpha *= 0.5
                    continue
                candidate_merit = merit_state(problem, objective, candidate, stiffness_solver=stiffness_solver)
                if np.isfinite(candidate_merit.merit) and candidate_merit.merit < current_merit.merit:
                    accepted = True
                    chosen_step = step
                    chosen_alpha = float(alpha)
                    chosen_mu = float(attempt_mu)
                    chosen_merit = candidate_merit
                    chosen_halves = int(halves)
                    current = candidate
                    break
                alpha *= 0.5
            if accepted:
                break
            attempt_mu *= 10.0

        history.append(
            {
                "outer_it": int(outer_it),
                "J": float(current_stats.J),
                "stop_measure": float(current_merit.dual_residual_norm),
                "stop_name": "stationary_merit",
                "gradient_residual_norm": float(current_merit.gradient_residual_norm),
                "dual_residual_norm": float(current_merit.dual_residual_norm),
                "merit": float(current_merit.merit),
                "alpha": float(chosen_alpha),
                "mu": float(chosen_mu),
                "accepted": bool(accepted),
                "halves": int(chosen_halves),
                "regularization_attempts": int(regularization_attempts),
                "step_norm": float(np.linalg.norm(chosen_step)),
                "trial_dual_residual_norm": float(chosen_merit.dual_residual_norm),
                "direction_model": CERTIFIED_DIRECTION_MODEL,
            }
        )
        if not accepted:
            status = "failed"
            message = "Certification Newton step failed to reduce the stationarity merit"
            break
        mu = max(1.0e-12, chosen_mu * (0.25 if chosen_alpha >= 0.5 else 1.0))
    else:
        current_merit = merit_state(problem, objective, current, stiffness_solver=stiffness_solver)
        if current_merit.dual_residual_norm < best_merit_state.dual_residual_norm:
            best_iterate = current.copy()
            best_merit_state = current_merit

    reported_iterate = np.asarray(current if status == "completed" else best_iterate, dtype=np.float64)
    return build_result_payload(
        method="certified_newton",
        problem=problem,
        epsilon=float(epsilon),
        iterate_free=reported_iterate,
        history=history,
        message=message,
        status=status,
        direction_solves=direction_solves,
        objective=objective,
        state_out=state_out,
        extra={
            "objective_name": "J",
            "direction_model": CERTIFIED_DIRECTION_MODEL,
            "handoff_source": str(handoff_source),
            "certified_newton_iters": int(len(history)),
            "reported_iterate_source": "final" if status == "completed" else "best_dual_residual",
            "best_residual_norm": float(best_merit_state.dual_residual_norm),
            "best_gradient_residual_norm": float(best_merit_state.gradient_residual_norm),
            "best_residual_outer_it": int(
                next(
                    (
                        item["outer_it"]
                        for item in history
                        if abs(float(item["dual_residual_norm"]) - best_merit_state.dual_residual_norm) <= 1.0e-14
                    ),
                    0,
                )
            ),
        },
    )

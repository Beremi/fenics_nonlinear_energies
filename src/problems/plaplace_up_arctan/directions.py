"""Thesis-style raw descent directions for the arctan-resonance family."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

try:  # pragma: no cover - optional acceleration
    import pyamg
except ImportError:  # pragma: no cover - optional dependency
    pyamg = None

from src.problems.plaplace_up_arctan.functionals import conjugate_exponent, seminorm_full
from src.problems.plaplace_up_arctan.solver_common import ObjectiveBundle, ProblemInstance


DIRECTION_MODEL_DVH = "thesis_dvh_aux_laplace"
DIRECTION_MODEL_AUTODIFF_HESSIAN = DIRECTION_MODEL_DVH


@dataclass
class DirectionResult:
    """One raw descent direction together with FE residual diagnostics."""

    direction: np.ndarray
    aux_state: np.ndarray | None
    stop_measure: float
    stop_name: str
    gradient: np.ndarray
    gradient_residual_norm: float
    dual_residual_norm: float
    descent_value: float
    direction_solves: int
    used_gradient_fallback: bool
    direction_model: str
    hessian_shift: float
    hessian_attempts: int


class _LinearDirectionSolver:
    """Solve the auxiliary Laplace problem used for ``d^{V_h}``."""

    def __init__(self, problem: ProblemInstance) -> None:
        self.problem = problem
        self.K = problem.stiffness.tocsr()
        self._factorized = spla.factorized(self.K.tocsc()) if self.K.shape[0] <= 512 else None
        self._amg = None
        if self._factorized is None and pyamg is not None:
            self._amg = pyamg.smoothed_aggregation_solver(self.K)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=np.float64)
        if self._factorized is not None:
            return np.asarray(self._factorized(rhs), dtype=np.float64)
        if self._amg is not None:
            return np.asarray(self._amg.solve(rhs, tol=1.0e-8), dtype=np.float64)
        sol, info = spla.cg(self.K, rhs, rtol=1.0e-8, atol=0.0, maxiter=max(500, self.K.shape[0]))
        if info != 0:
            sol = spla.spsolve(self.K, rhs)
        return np.asarray(sol, dtype=np.float64)


class DirectionContext:
    """Reusable raw direction builder for one problem/objective pair."""

    def __init__(self, problem: ProblemInstance, objective: ObjectiveBundle) -> None:
        self.problem = problem
        self.objective = objective
        self.linear_solver = _LinearDirectionSolver(problem)

    def compute(self, u_free: np.ndarray) -> DirectionResult:
        u_free = np.asarray(u_free, dtype=np.float64)
        gradient = np.asarray(self.objective.grad(u_free), dtype=np.float64)
        grad_norm = float(np.linalg.norm(gradient))
        if (not np.all(np.isfinite(gradient))) or grad_norm <= 0.0:
            return DirectionResult(
                direction=np.zeros_like(gradient),
                aux_state=None,
                stop_measure=grad_norm,
                stop_name="(5.7)",
                gradient=gradient,
                gradient_residual_norm=grad_norm,
                dual_residual_norm=grad_norm,
                descent_value=0.0,
                direction_solves=0,
                used_gradient_fallback=False,
                direction_model=DIRECTION_MODEL_DVH,
                hessian_shift=0.0,
                hessian_attempts=0,
            )

        aux = self.linear_solver.solve(gradient)
        aux_full = self.problem.expand_free(aux)
        aux_norm_p = float(seminorm_full(self.problem.params, aux_full, exponent=self.problem.p))
        aux_norm_q = float(seminorm_full(self.problem.params, aux_full, exponent=conjugate_exponent(self.problem.p)))
        if aux_norm_p <= 0.0 or not np.isfinite(aux_norm_p):
            direction = -gradient / grad_norm
            descent_value = float(np.dot(gradient, direction))
            return DirectionResult(
                direction=np.asarray(direction, dtype=np.float64),
                aux_state=np.asarray(aux, dtype=np.float64),
                stop_measure=grad_norm,
                stop_name="grad_fallback",
                gradient=gradient,
                gradient_residual_norm=grad_norm,
                dual_residual_norm=grad_norm,
                descent_value=descent_value,
                direction_solves=1,
                used_gradient_fallback=True,
                direction_model=DIRECTION_MODEL_DVH,
                hessian_shift=0.0,
                hessian_attempts=0,
            )

        dual_sq = float(np.dot(gradient, aux)) if np.all(np.isfinite(aux)) else float("nan")
        dual_norm = grad_norm if not np.isfinite(dual_sq) else float(np.sqrt(max(dual_sq, 0.0)))
        direction = -aux / aux_norm_p
        return DirectionResult(
            direction=np.asarray(direction, dtype=np.float64),
            aux_state=np.asarray(aux, dtype=np.float64),
            stop_measure=float(aux_norm_q),
            stop_name="(5.7)",
            gradient=gradient,
            gradient_residual_norm=grad_norm,
            dual_residual_norm=dual_norm,
            descent_value=float(np.dot(gradient, direction)),
            direction_solves=1,
            used_gradient_fallback=False,
            direction_model=DIRECTION_MODEL_DVH,
            hessian_shift=0.0,
            hessian_attempts=0,
        )


_DIRECTION_CONTEXT_CACHE: dict[tuple[object, ...], DirectionContext] = {}


def build_direction_context(problem: ProblemInstance, objective: ObjectiveBundle) -> DirectionContext:
    cache_key = (
        str(problem.geometry),
        int(problem.mesh_level),
        float(problem.p),
        float(problem.lambda1),
        int(problem.total_dofs),
        int(problem.free_dofs),
        str(objective.name),
    )
    context = _DIRECTION_CONTEXT_CACHE.get(cache_key)
    if context is None:
        context = DirectionContext(problem, objective)
        _DIRECTION_CONTEXT_CACHE[cache_key] = context
    else:
        context.objective = objective
    return context

"""Thesis descent directions and stopping criteria."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

try:  # pragma: no cover - optional acceleration
    import pyamg
except ImportError:  # pragma: no cover - optional dependency
    pyamg = None

from src.core.serial.minimizers import newton
from src.core.serial.sparse_solvers import HessSolverGenerator
from src.problems.plaplace_u3.thesis.functionals import (
    conjugate_exponent,
    seminorm_full,
)
from src.problems.plaplace_u3.thesis.presets import (
    THESIS_DIRECTION_EXACT,
    THESIS_DIRECTION_RN,
    THESIS_DIRECTION_VH,
)
from src.problems.plaplace_u3.thesis.solver_common import (
    ObjectiveBundle,
    ThesisProblem,
    build_direction_energy_bundle,
)

_DIRECTION_CONTEXT_CACHE: dict[tuple[object, ...], "DirectionContext"] = {}


@dataclass
class DirectionResult:
    """One descent direction together with its stopping metric."""

    direction: np.ndarray
    aux_state: np.ndarray | None
    stop_measure: float
    stop_name: str
    gradient: np.ndarray
    descent_value: float
    direction_solves: int


class _LinearDirectionSolver:
    """Solve the linear auxiliary problem used for ``d^V_h``."""

    def __init__(self, problem: ThesisProblem) -> None:
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
        if info != 0:  # pragma: no cover - solver fallback edge
            sol = spla.spsolve(self.K, rhs)
        return np.asarray(sol, dtype=np.float64)


class _ExactDirectionSolver:
    """Solve the nonlinear auxiliary problem used for the exact descent direction."""

    def __init__(self, problem: ThesisProblem) -> None:
        self.problem = problem
        self.derivator = build_direction_energy_bundle(problem)
        self.value, self.grad, self.ddf = self.derivator.get_derivatives()
        self.last_aux = np.zeros(problem.free_dofs, dtype=np.float64)
        self.linear_fallback = _LinearDirectionSolver(problem)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=np.float64)
        self.derivator.params["rhs"] = rhs
        initial = self.last_aux
        if not np.all(np.isfinite(initial)) or np.linalg.norm(initial) == 0.0:
            initial = self.linear_fallback.solve(rhs)
        solver_kind = "direct" if self.problem.free_dofs <= 256 else "amg"
        res = newton(
            lambda x: float(self.value(np.asarray(x, dtype=np.float64))),
            lambda x: np.asarray(self.grad(np.asarray(x, dtype=np.float64)), dtype=np.float64),
            HessSolverGenerator(
                lambda x: self.ddf(np.asarray(x, dtype=np.float64)),
                solver_type=solver_kind,
                tol=1.0e-2,
                maxiter=100,
            ),
            np.asarray(initial, dtype=np.float64),
            tolf=1.0e-10,
            tolg=1.0e-8,
            tolg_rel=0.0,
            maxit=40,
            linesearch_interval=(0.0, 2.0),
            linesearch_tol=1.0e-3,
            verbose=False,
        )
        self.last_aux = np.asarray(res["x"], dtype=np.float64)
        return np.asarray(self.last_aux, dtype=np.float64)


class DirectionContext:
    """Reusable direction builder for one objective on one FE problem."""

    def __init__(self, problem: ThesisProblem, objective: ObjectiveBundle) -> None:
        self.problem = problem
        self.objective = objective
        self.linear_solver = _LinearDirectionSolver(problem)
        self.exact_solver = _ExactDirectionSolver(problem)

    def compute(self, u_free: np.ndarray, direction_kind: str) -> DirectionResult:
        direction_kind = str(direction_kind)
        u_free = np.asarray(u_free, dtype=np.float64)
        gradient = np.asarray(self.objective.grad(u_free), dtype=np.float64)
        q = conjugate_exponent(self.problem.p)

        if direction_kind == THESIS_DIRECTION_RN:
            measure = float(np.linalg.norm(gradient, ord=q))
            seminorm_grad = float(np.linalg.norm(gradient))
            if seminorm_grad <= 0.0 or not np.isfinite(seminorm_grad):
                return DirectionResult(
                    direction=np.zeros_like(gradient),
                    aux_state=None,
                    stop_measure=measure,
                    stop_name="(5.8)",
                    gradient=gradient,
                    descent_value=0.0,
                    direction_solves=0,
                )
            direction = -gradient / seminorm_grad
            return DirectionResult(
                direction=np.asarray(direction, dtype=np.float64),
                aux_state=None,
                stop_measure=measure,
                stop_name="(5.8)",
                gradient=gradient,
                descent_value=float(np.dot(gradient, direction)),
                direction_solves=0,
            )

        if direction_kind == THESIS_DIRECTION_VH:
            aux = self.linear_solver.solve(gradient)
            aux_full = self.problem.expand_free(aux)
            aux_norm = float(seminorm_full(self.problem.params, aux_full, exponent=self.problem.p))
            q_norm = float(seminorm_full(self.problem.params, aux_full, exponent=q))
            if aux_norm <= 0.0 or not np.isfinite(aux_norm):
                direction = np.zeros_like(aux)
            else:
                direction = -aux / aux_norm
            return DirectionResult(
                direction=np.asarray(direction, dtype=np.float64),
                aux_state=np.asarray(aux, dtype=np.float64),
                stop_measure=q_norm,
                stop_name="(5.7)",
                gradient=gradient,
                descent_value=float(np.dot(gradient, direction)),
                direction_solves=1,
            )

        if direction_kind == THESIS_DIRECTION_EXACT:
            aux = self.exact_solver.solve(gradient)
            aux_full = self.problem.expand_free(aux)
            aux_norm = float(seminorm_full(self.problem.params, aux_full, exponent=self.problem.p))
            if aux_norm <= 0.0 or not np.isfinite(aux_norm):
                direction = np.zeros_like(aux)
            else:
                direction = -aux / aux_norm
            return DirectionResult(
                direction=np.asarray(direction, dtype=np.float64),
                aux_state=np.asarray(aux, dtype=np.float64),
                stop_measure=float(aux_norm ** (self.problem.p - 1.0)),
                stop_name="(5.6)",
                gradient=gradient,
                descent_value=float(np.dot(gradient, direction)),
                direction_solves=1,
            )

        raise ValueError(f"Unsupported direction_kind={direction_kind!r}")


def build_direction_context(problem: ThesisProblem, objective: ObjectiveBundle) -> DirectionContext:
    """Reuse the expensive direction solvers for repeated cases on the same mesh."""
    cache_key = (
        int(problem.dimension),
        str(problem.topology),
        str(problem.geometry),
        int(problem.mesh_level),
        float(problem.p),
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

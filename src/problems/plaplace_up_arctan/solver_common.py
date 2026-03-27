"""Shared wrappers and payload shaping for the arctan-resonance solvers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.core.benchmark.state_export import export_scalar_mesh_state_npz
from src.core.serial.jax_diff import EnergyDerivator
from src.problems.plaplace_up_arctan.functionals import (
    arctan_energy_free,
    assemble_stiffness_matrix,
    compute_state_stats_free,
    eigen_quotient_free,
    expand_free_vector,
    neg_arctan_energy_free,
    StateStats,
)
from src.problems.plaplace_up_arctan.support.mesh import MeshPLaplaceUPArctan2D


_OBJECTIVE_DERIVATOR_CACHE: dict[tuple[object, ...], EnergyDerivator] = {}
_PROBLEM_CACHE: dict[tuple[object, ...], "ProblemInstance"] = {}
ARCTAN_SOLVER_REVISION = "physical_J_dvh_certified_stationary_v1"


@dataclass
class ProblemInstance:
    """One cached structured FE problem instance."""

    topology: str
    geometry: str
    mesh_level: int
    p: float
    h: float
    init_mode: str
    seed: int
    lambda1: float
    lambda_level: int
    params: dict[str, object]
    adjacency: sp.coo_matrix
    stiffness: sp.csr_matrix
    u_init: np.ndarray
    plot_coords: np.ndarray
    plot_cells: np.ndarray

    @property
    def total_dofs(self) -> int:
        return int(np.asarray(self.params["u_0"], dtype=np.float64).size)

    @property
    def free_dofs(self) -> int:
        return int(np.asarray(self.params["freedofs"], dtype=np.int64).size)

    def expand_free(self, u_free: np.ndarray) -> np.ndarray:
        return expand_free_vector(
            np.asarray(u_free, dtype=np.float64),
            np.asarray(self.params["u_0"], dtype=np.float64),
            np.asarray(self.params["freedofs"], dtype=np.int64),
        )

    def stats(self, u_free: np.ndarray) -> StateStats:
        return compute_state_stats_free(self.params, np.asarray(u_free, dtype=np.float64))


@dataclass
class ObjectiveBundle:
    """Compiled objective callbacks for one FE problem."""

    name: str
    value: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    ddf: Callable[[np.ndarray], sp.csr_matrix]
    derivator: EnergyDerivator


def _problem_cache_key(
    *,
    level: int,
    p: float,
    geometry: str,
    init_mode: str,
    seed: int,
    lambda1: float,
    lambda_level: int,
) -> tuple[object, ...]:
    return (
        int(level),
        float(p),
        str(geometry),
        str(init_mode),
        int(seed),
        float(lambda1),
        int(lambda_level),
    )


def build_problem(
    *,
    level: int,
    p: float,
    geometry: str,
    init_mode: str,
    lambda1: float,
    lambda_level: int,
    seed: int = 0,
) -> ProblemInstance:
    cache_key = _problem_cache_key(
        level=int(level),
        p=float(p),
        geometry=str(geometry),
        init_mode=str(init_mode),
        seed=int(seed),
        lambda1=float(lambda1),
        lambda_level=int(lambda_level),
    )
    cached = _PROBLEM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mesh = MeshPLaplaceUPArctan2D(
        int(level),
        p=float(p),
        geometry=str(geometry),
        init_mode=str(init_mode),
        seed=int(seed),
    )
    params = dict(mesh.params)
    params["lambda1"] = float(lambda1)
    params["lambda_level"] = int(lambda_level)
    problem = ProblemInstance(
        topology="triangle",
        geometry=str(geometry),
        mesh_level=int(level),
        p=float(p),
        h=float(params["h"]),
        init_mode=str(init_mode),
        seed=int(seed),
        lambda1=float(lambda1),
        lambda_level=int(lambda_level),
        params=params,
        adjacency=mesh.adjacency.tocoo(),
        stiffness=assemble_stiffness_matrix(params),
        u_init=np.asarray(mesh.u_init, dtype=np.float64),
        plot_coords=np.asarray(params["nodes"], dtype=np.float64),
        plot_cells=np.asarray(params["elems"], dtype=np.int32),
    )
    _PROBLEM_CACHE[cache_key] = problem
    return problem


def _build_derivator(problem: ProblemInstance, objective: str) -> EnergyDerivator:
    objective = str(objective).lower()
    cache_key = (
        str(problem.geometry),
        int(problem.mesh_level),
        float(problem.p),
        float(problem.lambda1),
        int(problem.lambda_level),
        str(problem.init_mode),
        int(problem.seed),
        str(objective),
    )
    if cache_key not in _OBJECTIVE_DERIVATOR_CACHE:
        if objective == "j":
            fn = arctan_energy_free
            u_init = np.asarray(problem.u_init, dtype=np.float64)
        elif objective == "negj":
            fn = neg_arctan_energy_free
            u_init = np.asarray(problem.u_init, dtype=np.float64)
        elif objective == "eigen_q":
            fn = eigen_quotient_free
            u_init = np.asarray(problem.u_init, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported objective {objective!r}")
        _OBJECTIVE_DERIVATOR_CACHE[cache_key] = EnergyDerivator(
            fn,
            problem.params,
            problem.adjacency,
            u_init,
        )
    return _OBJECTIVE_DERIVATOR_CACHE[cache_key]


def build_objective_bundle(problem: ProblemInstance, objective: str) -> ObjectiveBundle:
    derivator = _build_derivator(problem, objective)
    value, grad, ddf = derivator.get_derivatives()
    return ObjectiveBundle(
        name=str(objective),
        value=lambda u: float(value(np.asarray(u, dtype=np.float64))),
        grad=lambda u: np.asarray(grad(np.asarray(u, dtype=np.float64)), dtype=np.float64),
        ddf=lambda u: ddf(np.asarray(u, dtype=np.float64)).tocsr(),
        derivator=derivator,
    )


def export_state_if_requested(
    path: str,
    *,
    problem: ProblemInstance,
    energy: float,
    u_full: np.ndarray,
    metadata: dict[str, object],
) -> str | None:
    if not path:
        return None
    export_scalar_mesh_state_npz(
        path,
        coords=np.asarray(problem.plot_coords, dtype=np.float64),
        triangles=np.asarray(problem.plot_cells, dtype=np.int32),
        u=np.asarray(u_full, dtype=np.float64),
        mesh_level=int(problem.mesh_level),
        problem_name="pLaplaceUPArctan",
        energy=float(energy),
        metadata={
            "geometry": str(problem.geometry),
            "p": float(problem.p),
            "lambda1": float(problem.lambda1),
            "lambda_level": int(problem.lambda_level),
            **metadata,
        },
    )
    return str(Path(path))


def residual_metrics(problem: ProblemInstance, objective: ObjectiveBundle, iterate_free: np.ndarray) -> tuple[float, float]:
    gradient = np.asarray(objective.grad(np.asarray(iterate_free, dtype=np.float64)), dtype=np.float64)
    grad_norm = float(np.linalg.norm(gradient))
    if grad_norm <= 0.0 or problem.stiffness.shape[0] == 0:
        return grad_norm, grad_norm
    aux = spla.spsolve(problem.stiffness, gradient)
    aux = np.asarray(aux, dtype=np.float64)
    dual_sq = float(np.dot(gradient, aux)) if np.all(np.isfinite(aux)) else float("nan")
    dual_norm = grad_norm if not np.isfinite(dual_sq) else math.sqrt(max(dual_sq, 0.0))
    return grad_norm, float(dual_norm)


def build_result_payload(
    *,
    method: str,
    problem: ProblemInstance,
    epsilon: float,
    iterate_free: np.ndarray,
    history: list[dict[str, object]],
    message: str,
    status: str,
    direction_solves: int,
    objective: ObjectiveBundle,
    reference_error_w1p: float | None = None,
    state_out: str = "",
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    iterate_free = np.asarray(iterate_free, dtype=np.float64)
    state_stats = problem.stats(iterate_free)
    full_state = problem.expand_free(iterate_free)
    state_path = export_state_if_requested(
        state_out,
        problem=problem,
        energy=float(state_stats.J),
        u_full=full_state,
        metadata={
            "method": str(method),
            "epsilon": float(epsilon),
            "seed_name": str(problem.init_mode),
        },
    )
    payload = {
        "gradient_residual_norm": None,
        "method": str(method),
        "geometry": str(problem.geometry),
        "level": int(problem.mesh_level),
        "h": float(problem.h),
        "p": float(problem.p),
        "lambda1": float(problem.lambda1),
        "lambda_level": int(problem.lambda_level),
        "epsilon": float(epsilon),
        "seed_name": str(problem.init_mode),
        "J": float(state_stats.J),
        "residual_norm": 0.0,
        "reference_error_w1p": None if reference_error_w1p is None else float(reference_error_w1p),
        "outer_iterations": int(len(history)),
        "direction_solves": int(direction_solves),
        "status": str(status),
        "message": str(message),
        "accepted_step_count": int(sum(1 for item in history if bool(item.get("accepted")))),
        "history": list(history),
        "iterate_free": iterate_free.tolist(),
        "state_out": state_path,
    }
    grad_norm, dual_norm = residual_metrics(problem, objective, iterate_free)
    payload["gradient_residual_norm"] = float(grad_norm)
    payload["residual_norm"] = float(dual_norm)
    if extra:
        payload.update(dict(extra))
    return payload

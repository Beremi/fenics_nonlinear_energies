"""Shared problem wrappers and result shaping for the thesis solvers.

This module is the bridge between mesh/functionals and the individual
algorithms: it builds a cached FE problem object, compiles objective bundles,
and converts a raw solver iterate into the JSON payload used by reports/tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse as sp

from src.core.benchmark.state_export import export_scalar_mesh_state_npz
from src.core.serial.jax_diff import EnergyDerivator
from src.problems.plaplace_u3.support.mesh import MeshPLaplaceU32D
from src.problems.plaplace_u3.thesis.functionals import (
    I_interval_free,
    I_triangle_free,
    J_interval_free,
    J_triangle_free,
    StateStats,
    assemble_stiffness_matrix,
    compute_state_stats_free,
    direction_interval_free,
    direction_triangle_free,
    expand_free_vector,
    rescale_free_to_solution,
)
from src.problems.plaplace_u3.thesis.mesh1d import MeshPLaplaceU31D


_OBJECTIVE_DERIVATOR_CACHE: dict[tuple[object, ...], EnergyDerivator] = {}
_DIRECTION_DERIVATOR_CACHE: dict[tuple[object, ...], EnergyDerivator] = {}
_PROBLEM_CACHE: dict[tuple[object, ...], "ThesisProblem"] = {}


@dataclass
class ThesisProblem:
    """One cached structured FE problem instance used by the thesis layer."""

    dimension: int
    topology: str
    geometry: str
    mesh_level: int
    p: float
    h: float
    init_mode: str
    seed: int
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


def _problem_cache_key(problem: ThesisProblem, *suffix: object) -> tuple[object, ...]:
    return (
        int(problem.dimension),
        str(problem.topology),
        str(problem.geometry),
        int(problem.mesh_level),
        float(problem.p),
        int(problem.total_dofs),
        int(problem.free_dofs),
        *suffix,
    )


@dataclass
class ObjectiveBundle:
    """Compiled objective callbacks for one FE problem."""

    name: str
    value: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    ddf: Callable[[np.ndarray], sp.csr_matrix]
    derivator: EnergyDerivator


def build_problem(
    *,
    dimension: int,
    level: int,
    p: float,
    geometry: str,
    init_mode: str,
    seed: int = 0,
) -> ThesisProblem:
    """Build one cached 1D or 2D structured thesis problem."""
    cache_key = (
        int(dimension),
        int(level),
        float(p),
        str(geometry),
        str(init_mode),
        int(seed),
    )
    cached = _PROBLEM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if int(dimension) == 2:
        mesh = MeshPLaplaceU32D(
            int(level),
            p=float(p),
            geometry=str(geometry),
            init_mode=str(init_mode),
            seed=int(seed),
        )
        params = dict(mesh.params)
        params["topology"] = "triangle"
        problem = ThesisProblem(
            dimension=2,
            topology="triangle",
            geometry=str(geometry),
            mesh_level=int(level),
            p=float(p),
            h=float(params["h"]),
            init_mode=str(init_mode),
            seed=int(seed),
            params=params,
            adjacency=mesh.adjacency.tocoo(),
            stiffness=assemble_stiffness_matrix(params),
            u_init=np.asarray(mesh.u_init, dtype=np.float64),
            plot_coords=np.asarray(params["nodes"], dtype=np.float64),
            plot_cells=np.asarray(params["elems"], dtype=np.int32),
        )
        _PROBLEM_CACHE[cache_key] = problem
        return problem

    if int(dimension) == 1:
        mesh = MeshPLaplaceU31D(
            int(level),
            p=float(p),
            geometry=str(geometry),
            init_mode=str(init_mode),
            seed=int(seed),
        )
        params = dict(mesh.params)
        coords = np.column_stack(
            (
                np.asarray(params["nodes_1d"], dtype=np.float64),
                np.zeros_like(np.asarray(params["nodes_1d"], dtype=np.float64)),
            )
        )
        problem = ThesisProblem(
            dimension=1,
            topology="interval",
            geometry=str(geometry),
            mesh_level=int(level),
            p=float(p),
            h=float(params["h"]),
            init_mode=str(init_mode),
            seed=int(seed),
            params=params,
            adjacency=mesh.adjacency.tocoo(),
            stiffness=assemble_stiffness_matrix(params),
            u_init=np.asarray(mesh.u_init, dtype=np.float64),
            plot_coords=coords,
            plot_cells=np.asarray(params["elems"], dtype=np.int32),
        )
        _PROBLEM_CACHE[cache_key] = problem
        return problem

    raise ValueError(f"Unsupported dimension={dimension!r}")


def _build_derivator(problem: ThesisProblem, objective: str) -> EnergyDerivator:
    if problem.topology == "triangle":
        fn = J_triangle_free if objective == "J" else I_triangle_free
    elif problem.topology == "interval":
        fn = J_interval_free if objective == "J" else I_interval_free
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported topology {problem.topology!r}")
    cache_key = _problem_cache_key(problem, "objective", str(objective).upper())
    if cache_key not in _OBJECTIVE_DERIVATOR_CACHE:
        _OBJECTIVE_DERIVATOR_CACHE[cache_key] = EnergyDerivator(fn, problem.params, problem.adjacency, problem.u_init)
    return _OBJECTIVE_DERIVATOR_CACHE[cache_key]


def build_objective_bundle(problem: ThesisProblem, objective: str) -> ObjectiveBundle:
    """Compile value/gradient/Hessian callbacks for ``J`` or ``I``."""
    objective = str(objective).upper()
    if objective not in {"J", "I"}:
        raise ValueError(f"Unsupported objective {objective!r}")
    derivator = _build_derivator(problem, objective)
    value, grad, ddf = derivator.get_derivatives()
    return ObjectiveBundle(
        name=objective,
        value=lambda u: float(value(np.asarray(u, dtype=np.float64))),
        grad=lambda u: np.asarray(grad(np.asarray(u, dtype=np.float64)), dtype=np.float64),
        ddf=lambda u: ddf(np.asarray(u, dtype=np.float64)).tocsr(),
        derivator=derivator,
    )


def build_direction_energy_bundle(problem: ThesisProblem) -> EnergyDerivator:
    """Compile the auxiliary convex energy for the exact descent direction."""
    params = dict(problem.params)
    params["rhs"] = np.zeros(problem.free_dofs, dtype=np.float64)
    if problem.topology == "triangle":
        fn = direction_triangle_free
    elif problem.topology == "interval":
        fn = direction_interval_free
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported topology {problem.topology!r}")
    cache_key = _problem_cache_key(problem, "direction_energy")
    if cache_key not in _DIRECTION_DERIVATOR_CACHE:
        _DIRECTION_DERIVATOR_CACHE[cache_key] = EnergyDerivator(
            fn,
            params,
            problem.adjacency,
            np.zeros(problem.free_dofs, dtype=np.float64),
        )
    return _DIRECTION_DERIVATOR_CACHE[cache_key]


def physical_solution_from_iterate(
    problem: ThesisProblem,
    method: str,
    iterate_free: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StateStats]:
    """Return the physical weak-solution state that should be reported.

    OA1/OA2 evolve on the scale-invariant quotient ``I`` and therefore need the
    analytic rescaling step before reporting ``J``. RMPA/MPA already evolve on
    the correctly scaled energy landscape, so their raw iterate is the reported
    physical state.
    """
    method = str(method).lower()
    if method in {"oa1", "oa2"}:
        return rescale_free_to_solution(problem.params, np.asarray(iterate_free, dtype=np.float64))
    scaled_full = problem.expand_free(np.asarray(iterate_free, dtype=np.float64))
    return (
        np.asarray(iterate_free, dtype=np.float64),
        scaled_full,
        compute_state_stats_free(problem.params, np.asarray(iterate_free, dtype=np.float64)),
    )


def export_state_if_requested(
    path: str,
    *,
    problem: ThesisProblem,
    method: str,
    energy: float,
    u_full: np.ndarray,
    metadata: dict[str, object],
) -> str | None:
    """Write the compact NPZ state file when requested."""
    if not path:
        return None
    export_scalar_mesh_state_npz(
        path,
        coords=np.asarray(problem.plot_coords, dtype=np.float64),
        triangles=np.asarray(problem.plot_cells, dtype=np.int32),
        u=np.asarray(u_full, dtype=np.float64),
        mesh_level=int(problem.mesh_level),
        problem_name=f"pLaplaceU3Thesis{problem.dimension}D",
        energy=float(energy),
        metadata={
            "method": str(method),
            "geometry": str(problem.geometry),
            "p": float(problem.p),
            "dimension": int(problem.dimension),
            **metadata,
        },
    )
    return str(Path(path))


def _best_stop_from_history(history: list[dict[str, object]]) -> tuple[float | None, int | None]:
    best_measure: float | None = None
    best_outer_it: int | None = None
    for item in history:
        stop_measure = item.get("stop_measure")
        outer_it = item.get("outer_it")
        if stop_measure is None or outer_it is None:
            continue
        measure = float(stop_measure)
        if (best_measure is None) or (measure < best_measure):
            best_measure = measure
            best_outer_it = int(outer_it)
    return best_measure, best_outer_it


def build_result_payload(
    *,
    method: str,
    direction: str,
    problem: ThesisProblem,
    epsilon: float,
    iterate_free: np.ndarray,
    history: list[dict[str, object]],
    message: str,
    status: str,
    direction_solves: int,
    reference_error_w1p: float | None = None,
    state_out: str = "",
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return the common JSON payload used by all thesis CLIs and reports."""
    physical_free, physical_full, stats = physical_solution_from_iterate(problem, method, iterate_free)
    state_path = export_state_if_requested(
        state_out,
        problem=problem,
        method=method,
        energy=float(stats.J),
        u_full=physical_full,
        metadata={
            "direction": str(direction),
            "epsilon": float(epsilon),
            "seed_name": str(problem.init_mode),
        },
    )
    payload = {
        "method": str(method),
        "direction": str(direction),
        "dimension": int(problem.dimension),
        "geometry": str(problem.geometry),
        "level": int(problem.mesh_level),
        "h": float(problem.h),
        "p": float(problem.p),
        "epsilon": float(epsilon),
        "seed_name": str(problem.init_mode),
        "J": float(stats.J),
        "I": float(stats.I),
        "c": float(stats.c),
        "reference_error_w1p": (
            None if reference_error_w1p is None else float(reference_error_w1p)
        ),
        "outer_iterations": int(len(history)),
        "direction_solves": int(direction_solves),
        "status": str(status),
        "message": str(message),
        "configured_maxit": None,
        "best_stop_measure": None,
        "best_stop_outer_it": None,
        "accepted_step_count": int(
            sum(1 for item in history if bool(item.get("accepted")))
        ),
        "max_halves": int(
            max((int(item.get("halves", 0)) for item in history if item.get("halves") is not None), default=0)
        ),
        "final_halves": int(history[-1].get("halves", 0)) if history else 0,
        "history": list(history),
        "raw_iterate_free": np.asarray(iterate_free, dtype=np.float64).tolist(),
        "physical_solution_free": np.asarray(physical_free, dtype=np.float64).tolist(),
        "state_out": state_path,
    }
    best_stop_measure, best_stop_outer_it = _best_stop_from_history(history)
    payload["best_stop_measure"] = (
        None if best_stop_measure is None else float(best_stop_measure)
    )
    payload["best_stop_outer_it"] = (
        None if best_stop_outer_it is None else int(best_stop_outer_it)
    )
    if extra:
        payload.update(dict(extra))
    return payload

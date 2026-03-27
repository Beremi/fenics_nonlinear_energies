"""Shared raw and certified solve workflows for the arctan-resonance family."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.problems.plaplace_up_arctan.local_polish import polish_stationary
from src.problems.plaplace_up_arctan.solver_common import ObjectiveBundle, build_objective_bundle
from src.problems.plaplace_up_arctan.solver_mpa import run_mpa, run_mpa_symmetric
from src.problems.plaplace_up_arctan.solver_rmpa import run_rmpa, run_rmpa_shifted


def run_raw_method(
    problem,
    *,
    method: str,
    epsilon: float,
    maxit: int,
    init_free: np.ndarray | None = None,
    state_out: str = "",
    delta0: float = 1.0,
    num_nodes: int = 30,
    rho: float = 1.0,
    segment_tol_factor: float = 0.125,
) -> dict[str, object]:
    method = str(method).lower()
    if method == "rmpa":
        return run_rmpa(
            problem,
            epsilon=float(epsilon),
            maxit=int(maxit),
            delta0=float(delta0),
            init_free=None if init_free is None else np.asarray(init_free, dtype=np.float64),
            state_out=str(state_out),
        )
    if method == "rmpa_shifted":
        return run_rmpa_shifted(
            problem,
            epsilon=float(epsilon),
            maxit=int(maxit),
            delta0=float(delta0),
            init_free=None if init_free is None else np.asarray(init_free, dtype=np.float64),
            state_out=str(state_out),
        )
    if method == "mpa_symmetric":
        return run_mpa_symmetric(
            problem,
            epsilon=float(epsilon),
            maxit=int(maxit),
            num_nodes=int(num_nodes),
            rho=float(rho),
            segment_tol_factor=float(segment_tol_factor),
            init_free=None if init_free is None else np.asarray(init_free, dtype=np.float64),
            state_out=str(state_out),
        )
    if method == "mpa":
        return run_mpa(
            problem,
            epsilon=float(epsilon),
            maxit=int(maxit),
            num_nodes=int(num_nodes),
            rho=float(rho),
            segment_tol_factor=float(segment_tol_factor),
            init_free=None if init_free is None else np.asarray(init_free, dtype=np.float64),
            state_out=str(state_out),
        )
    raise ValueError(f"Unsupported raw method {method!r}")


def certify_from_iterate(
    problem,
    *,
    iterate_free: np.ndarray,
    epsilon: float,
    maxit: int,
    state_out: str = "",
    handoff_source: str = "raw_best_iterate",
    objective: ObjectiveBundle | None = None,
) -> dict[str, object]:
    objective_eff = build_objective_bundle(problem, "J") if objective is None else objective
    return polish_stationary(
        problem,
        objective_eff,
        init_free=np.asarray(iterate_free, dtype=np.float64),
        epsilon=float(epsilon),
        maxit=int(maxit),
        state_out=str(state_out),
        handoff_source=str(handoff_source),
    )


def hybrid_solve(
    problem,
    *,
    method: str,
    raw_epsilon: float,
    raw_maxit: int,
    certified_tol: float,
    polish_maxit: int,
    init_free: np.ndarray | None = None,
    raw_state_out: str = "",
    certified_state_out: str = "",
    allow_certification: bool = True,
) -> dict[str, object]:
    raw_result = run_raw_method(
        problem,
        method=str(method),
        epsilon=float(raw_epsilon),
        maxit=int(raw_maxit),
        init_free=None if init_free is None else np.asarray(init_free, dtype=np.float64),
        state_out=str(raw_state_out),
    )
    certified_result = None
    if allow_certification:
        certified_result = certify_from_iterate(
            problem,
            iterate_free=np.asarray(raw_result["iterate_free"], dtype=np.float64),
            epsilon=float(certified_tol),
            maxit=int(polish_maxit),
            state_out=str(certified_state_out),
            handoff_source=f"{str(method).lower()}:{raw_result.get('reported_iterate_source', 'reported')}",
        )
    return {
        "method": str(method).lower(),
        "raw": raw_result,
        "certified": certified_result,
        "solution_track": "certified" if certified_result is not None else "raw",
    }


def ensure_state_path(base_dir: Path, name: str) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / name)

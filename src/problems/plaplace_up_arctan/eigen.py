"""Discrete first-eigenpair stage for the arctan-resonance workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.core.benchmark.state_export import export_scalar_mesh_state_npz
from src.problems.plaplace_up_arctan.directions import build_direction_context
from src.problems.plaplace_up_arctan.solver_common import build_objective_bundle, build_problem


def _normalize_lp(problem, u_free: np.ndarray) -> np.ndarray:
    stats = problem.stats(u_free)
    norm = float(stats.lp_norm)
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("Cannot normalize a zero/invalid eigenfunction candidate")
    return np.asarray(u_free, dtype=np.float64) / norm


def _positivity_ok(values: np.ndarray, tol: float = 1.0e-10) -> bool:
    return bool(np.all(np.asarray(values, dtype=np.float64) > tol))


def compute_lambda1_cached(
    *,
    cache_path: str | Path,
    state_out: str | Path | None = None,
    p: float = 3.0,
    level: int,
    geometry: str,
    init_mode: str = "sine",
    seed: int = 0,
    maxit: int = 120,
    epsilon: float = 2.0e-6,
    force: bool = False,
    init_free: np.ndarray | None = None,
) -> dict[str, object]:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    p = float(p)
    placeholder_lambda = 1.0
    problem = build_problem(
        level=int(level),
        p=p,
        geometry=str(geometry),
        init_mode=str(init_mode),
        lambda1=placeholder_lambda,
        lambda_level=int(level),
        seed=int(seed),
    )
    objective = build_objective_bundle(problem, "eigen_q")
    directions = build_direction_context(problem, objective)

    current = np.asarray(problem.u_init if init_free is None else init_free, dtype=np.float64).copy()
    current = np.maximum(current, 1.0e-8)
    current = _normalize_lp(problem, current)

    history: list[dict[str, object]] = []
    message = "Maximum number of iterations reached"
    status = "maxit"
    direction_solves = 0

    for outer_it in range(1, int(maxit) + 1):
        value = float(objective.value(current))
        dir_result = directions.compute(current)
        direction_solves += int(dir_result.direction_solves)
        if dir_result.stop_measure <= float(epsilon):
            message = f"Stopping criterion {dir_result.stop_name} satisfied"
            status = "completed"
            break

        accepted = False
        alpha = 0.0
        halves = 0
        trial = np.asarray(current, dtype=np.float64)
        while halves <= 60:
            alpha_try = 1.0 / (2**halves)
            candidate = np.asarray(current + alpha_try * np.asarray(dir_result.direction, dtype=np.float64), dtype=np.float64)
            if not _positivity_ok(candidate):
                halves += 1
                continue
            candidate = _normalize_lp(problem, candidate)
            candidate_value = float(objective.value(candidate))
            if np.isfinite(candidate_value) and candidate_value < value:
                accepted = True
                alpha = float(alpha_try)
                trial = candidate
                value = candidate_value
                break
            halves += 1

        history.append(
            {
                "outer_it": int(outer_it),
                "quotient": float(value),
                "stop_measure": float(dir_result.stop_measure),
                "accepted": bool(accepted),
                "alpha": float(alpha),
                "halves": int(halves),
            }
        )
        if not accepted:
            message = "Eigen line search failed to lower the quotient"
            status = "failed"
            break
        current = np.asarray(trial, dtype=np.float64)
    else:
        value = float(objective.value(current))

    current = _normalize_lp(problem, current)
    stats = problem.stats(current)
    full_state = problem.expand_free(current)
    payload = {
        "status": str(status),
        "message": str(message),
        "level": int(level),
        "p": float(p),
        "geometry": str(geometry),
        "lambda1": float(stats.a),
        "lambda_level": int(level),
        "quotient": float(value),
        "residual_norm": float(np.linalg.norm(objective.grad(current))),
        "normalization_error": abs(float(stats.lp_norm) - 1.0),
        "outer_iterations": int(len(history)),
        "direction_solves": int(direction_solves),
        "history": history,
        "eigenfunction_free": current.tolist(),
        "eigenfunction_min": float(np.min(full_state)),
        "eigenfunction_max": float(np.max(full_state)),
        "state_out": None,
    }
    if state_out:
        export_scalar_mesh_state_npz(
            state_out,
            coords=np.asarray(problem.plot_coords, dtype=np.float64),
            triangles=np.asarray(problem.plot_cells, dtype=np.int32),
            u=np.asarray(full_state, dtype=np.float64),
            mesh_level=int(level),
            problem_name="pLaplaceUPArctanEigen",
            energy=float(value),
            metadata={
                "lambda1": float(stats.a),
                "lambda_level": int(level),
            },
        )
        payload["state_out"] = str(Path(state_out))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

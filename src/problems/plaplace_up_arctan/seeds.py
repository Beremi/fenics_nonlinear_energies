"""Deterministic start-seed generators for the arctan-resonance family."""

from __future__ import annotations

import math

import numpy as np

from src.problems.plaplace_up_arctan.solver_common import ProblemInstance


SEED_SINE = "sine"
SEED_BUBBLE = "bubble"
SEED_TILTED = "tilted"
SEED_EIGENFUNCTION = "eigenfunction"


def _free_xy(problem: ProblemInstance) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(problem.params["nodes"], dtype=np.float64)
    freedofs = np.asarray(problem.params["freedofs"], dtype=np.int64)
    free_nodes = nodes[freedofs]
    return np.asarray(free_nodes[:, 0], dtype=np.float64), np.asarray(free_nodes[:, 1], dtype=np.float64)


def sine_seed(problem: ProblemInstance) -> np.ndarray:
    return np.asarray(problem.u_init, dtype=np.float64).copy()


def bubble_seed(problem: ProblemInstance) -> np.ndarray:
    x, y = _free_xy(problem)
    s = np.sin(math.pi * x) * np.sin(math.pi * y)
    return np.asarray(s * s, dtype=np.float64)


def tilted_seed(problem: ProblemInstance) -> np.ndarray:
    x, y = _free_xy(problem)
    s = np.sin(math.pi * x) * np.sin(math.pi * y)
    return np.asarray(s * (1.0 + 0.5 * s), dtype=np.float64)


def named_start_seed(
    problem: ProblemInstance,
    name: str,
    *,
    eigenfunction_free: np.ndarray | None = None,
) -> np.ndarray:
    key = str(name).lower()
    if key == SEED_SINE:
        return sine_seed(problem)
    if key == SEED_BUBBLE:
        return bubble_seed(problem)
    if key == SEED_TILTED:
        return tilted_seed(problem)
    if key == SEED_EIGENFUNCTION:
        if eigenfunction_free is None:
            raise ValueError("eigenfunction seed requested without eigenfunction data")
        return np.asarray(eigenfunction_free, dtype=np.float64).copy()
    raise ValueError(f"Unsupported start seed {name!r}")


def candidate_start_seed_names(*, p: float, method: str, has_eigenfunction: bool) -> list[str]:
    if float(p) == 2.0:
        if str(method) in {"mpa", "mpa_symmetric"}:
            return [SEED_SINE, SEED_BUBBLE, SEED_TILTED]
        return [SEED_SINE]
    names = [SEED_EIGENFUNCTION] if has_eigenfunction else [SEED_SINE]
    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped

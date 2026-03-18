"""Shared continuation and convergence helpers for topology optimisation."""

from __future__ import annotations

import numpy as np


def constitutive_plane_stress(young: float, poisson: float) -> np.ndarray:
    if young <= 0.0:
        raise ValueError("young must be positive.")
    if not (-1.0 < poisson < 0.5):
        raise ValueError("poisson must lie in (-1, 0.5).")
    prefactor = young / (1.0 - poisson**2)
    return prefactor * np.array(
        [
            [1.0, poisson, 0.0],
            [poisson, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - poisson)],
        ],
        dtype=np.float64,
    )


def message_is_converged(message: str) -> bool:
    msg = str(message).lower()
    return "converged" in msg or "satisfied" in msg


def staircase_p_step(
    p_penal: float,
    *,
    p_max: float,
    p_increment: float,
    continuation_interval: int,
    outer_it: int,
) -> float:
    if p_increment <= 0.0 or continuation_interval <= 0:
        return 0.0
    if p_penal >= p_max - 1e-12:
        return 0.0
    if outer_it % continuation_interval != 0:
        return 0.0
    return float(min(p_increment, p_max - p_penal))


def relative_state_change(current, previous, freedofs) -> float:
    if previous is None:
        return np.inf
    current_arr = np.asarray(current, dtype=np.float64)
    previous_arr = np.asarray(previous, dtype=np.float64)
    freedofs_arr = np.asarray(freedofs, dtype=np.int64)
    return float(
        np.linalg.norm(current_arr[freedofs_arr] - previous_arr[freedofs_arr])
        / max(1.0, np.linalg.norm(previous_arr[freedofs_arr]))
    )

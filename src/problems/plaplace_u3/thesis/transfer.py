"""Nested-grid and same-mesh error helpers for the structured thesis meshes."""

from __future__ import annotations

import math

import numpy as np

from src.problems.plaplace_u3.thesis.functionals import expand_free_vector, seminorm_full


def prolong_free_to_problem(
    source_params: dict[str, object],
    source_free: np.ndarray,
    target_params: dict[str, object],
) -> np.ndarray:
    """Evaluate the coarse structured FE state at the fine free nodes."""
    source_full = expand_free_vector(
        np.asarray(source_free, dtype=np.float64),
        np.asarray(source_params["u_0"], dtype=np.float64),
        np.asarray(source_params["freedofs"], dtype=np.int64),
    )
    topology = str(source_params["topology"])
    if topology != str(target_params["topology"]):
        raise ValueError("Nested transfer requires matching topology")

    if topology == "interval":
        target_nodes = np.asarray(target_params["nodes_1d"], dtype=np.float64)
        free = np.asarray(target_params["freedofs"], dtype=np.int64)
        h = float(source_params["h"])
        n_subdiv = int(round(math.pi / h))
        x = np.asarray(target_nodes[free], dtype=np.float64)
        cell = np.clip(np.floor(x / h).astype(np.int64), 0, n_subdiv - 1)
        x0 = cell.astype(np.float64) * h
        alpha = np.clip((x - x0) / h, 0.0, 1.0)
        values = (1.0 - alpha) * source_full[cell] + alpha * source_full[cell + 1]
        return values

    if topology == "triangle":
        target_nodes = np.asarray(target_params["nodes"], dtype=np.float64)
        target_free_nodes = target_nodes[np.asarray(target_params["freedofs"], dtype=np.int64)]
        h = float(source_params["h"])
        n_subdiv = int(round(math.pi / h))

        source_nodes = np.asarray(source_params["nodes"], dtype=np.float64)
        source_ij = np.rint(source_nodes / h).astype(np.int64)
        grid_values = np.full((n_subdiv + 1, n_subdiv + 1), np.nan, dtype=np.float64)
        grid_values[source_ij[:, 1], source_ij[:, 0]] = source_full

        x = np.asarray(target_free_nodes[:, 0], dtype=np.float64)
        y = np.asarray(target_free_nodes[:, 1], dtype=np.float64)
        cell_i = np.clip(np.floor(x / h).astype(np.int64), 0, n_subdiv - 1)
        cell_j = np.clip(np.floor(y / h).astype(np.int64), 0, n_subdiv - 1)
        x0 = cell_i.astype(np.float64) * h
        y0 = cell_j.astype(np.float64) * h
        rx = np.clip((x - x0) / h, 0.0, 1.0)
        ry = np.clip((y - y0) / h, 0.0, 1.0)

        ll = grid_values[cell_j, cell_i]
        lr = grid_values[cell_j, cell_i + 1]
        ul = grid_values[cell_j + 1, cell_i]
        ur = grid_values[cell_j + 1, cell_i + 1]

        lower = (rx + ry) <= 1.0
        values = np.empty_like(x)
        values[lower] = (1.0 - rx[lower] - ry[lower]) * ll[lower] + rx[lower] * lr[lower] + ry[lower] * ul[lower]
        values[~lower] = (rx[~lower] + ry[~lower] - 1.0) * ur[~lower] + (1.0 - rx[~lower]) * ul[~lower] + (1.0 - ry[~lower]) * lr[~lower]
        if np.any(~np.isfinite(values)):  # pragma: no cover - defensive
            raise ValueError("Structured prolongation encountered non-finite values")
        return values

    raise ValueError(f"Unsupported topology {topology!r}")  # pragma: no cover - defensive


def nested_w1p_error(
    source_params: dict[str, object],
    source_free: np.ndarray,
    target_params: dict[str, object],
    target_free: np.ndarray,
) -> float:
    """Compare one coarse solution against one finer reference on the finer grid."""
    source_on_target = prolong_free_to_problem(source_params, source_free, target_params)
    source_full = expand_free_vector(
        source_on_target,
        np.asarray(target_params["u_0"], dtype=np.float64),
        np.asarray(target_params["freedofs"], dtype=np.int64),
    )
    target_full = expand_free_vector(
        np.asarray(target_free, dtype=np.float64),
        np.asarray(target_params["u_0"], dtype=np.float64),
        np.asarray(target_params["freedofs"], dtype=np.int64),
    )
    diff = np.asarray(source_full - target_full, dtype=np.float64)
    return float(
        seminorm_full(target_params, diff, exponent=float(target_params["p"]))
    )


def same_mesh_w1p_error(
    params: dict[str, object],
    source_free: np.ndarray,
    target_free: np.ndarray,
) -> float:
    """Compare two solutions on the same FE space in ``|.|_{1,p,0}``."""
    source_full = expand_free_vector(
        np.asarray(source_free, dtype=np.float64),
        np.asarray(params["u_0"], dtype=np.float64),
        np.asarray(params["freedofs"], dtype=np.int64),
    )
    target_full = expand_free_vector(
        np.asarray(target_free, dtype=np.float64),
        np.asarray(params["u_0"], dtype=np.float64),
        np.asarray(params["freedofs"], dtype=np.int64),
    )
    diff = np.asarray(source_full - target_full, dtype=np.float64)
    return float(seminorm_full(params, diff, exponent=float(params["p"])))

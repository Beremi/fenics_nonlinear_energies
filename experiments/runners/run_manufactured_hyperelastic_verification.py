#!/usr/bin/env python3
"""Independent nonaffine P1 manufactured verification for hyperelasticity.

The verifier assembles the weak residual and consistent tangent directly from
the analytic first Piola stress.  It does not call the production JAX energy,
PETSc assembler, or mesh loader.  A smooth orientation-preserving deformation
is imposed on the whole boundary and an analytic body force makes it an exact
solution of the continuous problem.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from src.core.benchmark.run_record import atomic_write_json


C1 = 0.5
D1 = 5.0
AMPLITUDE = 0.05


def _exact_deformation(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    x, y, z = np.moveaxis(points, -1, 0)
    perturbation = (
        AMPLITUDE * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
    )
    result = points.copy()
    result[..., 0] += perturbation
    return result


def _exact_gradient(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    x, y, z = np.moveaxis(points, -1, 0)
    sx, sy, sz = np.sin(np.pi * x), np.sin(np.pi * y), np.sin(np.pi * z)
    cx, cy, cz = np.cos(np.pi * x), np.cos(np.pi * y), np.cos(np.pi * z)
    gradient = np.broadcast_to(np.eye(3), points.shape[:-1] + (3, 3)).copy()
    gradient[..., 0, 0] += AMPLITUDE * np.pi * cx * sy * sz
    gradient[..., 0, 1] += AMPLITUDE * np.pi * sx * cy * sz
    gradient[..., 0, 2] += AMPLITUDE * np.pi * sx * sy * cz
    return gradient


def _density_piola_tangent(
    deformation_gradient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return W, P, and dP/dF for one or more positive-determinant states."""

    deformation_gradient = np.asarray(deformation_gradient, dtype=np.float64)
    determinant = np.linalg.det(deformation_gradient)
    if np.any(~np.isfinite(determinant)) or np.any(determinant <= 0.0):
        raise ValueError("hyperelastic state must have finite positive determinant")
    inverse_transpose = np.swapaxes(np.linalg.inv(deformation_gradient), -1, -2)
    scalar = -2.0 * C1 + 2.0 * D1 * (determinant - 1.0) * determinant
    density = C1 * (
        np.sum(deformation_gradient**2, axis=(-2, -1))
        - 3.0
        - 2.0 * np.log(determinant)
    ) + D1 * (determinant - 1.0) ** 2
    piola = 2.0 * C1 * deformation_gradient + scalar[..., None, None] * inverse_transpose

    derivative_scalar = (
        2.0
        * D1
        * (2.0 * determinant - 1.0)[..., None, None]
        * determinant[..., None, None]
        * inverse_transpose
    )
    identity_term = np.einsum("ik,ab->iakb", np.eye(3), np.eye(3), optimize=True)
    tangent = (
        2.0 * C1 * identity_term
        + np.einsum(
            "...kb,...ia->...iakb",
            derivative_scalar,
            inverse_transpose,
            optimize=True,
        )
        - scalar[..., None, None, None, None]
        * np.einsum(
            "...ib,...ka->...iakb",
            inverse_transpose,
            inverse_transpose,
            optimize=True,
        )
    )
    return density, piola, tangent


def _body_force(points: np.ndarray) -> np.ndarray:
    """Return ``-Div(P(F_exact))`` from the closed-form rank-one deformation."""

    points = np.asarray(points, dtype=np.float64)
    x, y, z = np.moveaxis(points, -1, 0)
    sx, sy, sz = np.sin(np.pi * x), np.sin(np.pi * y), np.sin(np.pi * z)
    cx, cy, cz = np.cos(np.pi * x), np.cos(np.pi * y), np.cos(np.pi * z)
    phi = AMPLITUDE * sx * sy * sz
    phi_x = AMPLITUDE * np.pi * cx * sy * sz
    phi_y = AMPLITUDE * np.pi * sx * cy * sz
    phi_z = AMPLITUDE * np.pi * sx * sy * cz
    phi_xx = -(np.pi**2) * phi
    phi_yy = phi_xx
    phi_zz = phi_xx
    phi_xy = AMPLITUDE * np.pi**2 * cx * cy * sz
    phi_xz = AMPLITUDE * np.pi**2 * cx * sy * cz

    determinant = 1.0 + phi_x
    scalar = -2.0 * C1 + 2.0 * D1 * (determinant - 1.0) * determinant
    derivative_scalar = 2.0 * D1 * (2.0 * determinant - 1.0)
    ratio = scalar / determinant
    derivative_ratio = derivative_scalar / determinant - scalar / determinant**2

    derivative_p11 = 2.0 * C1 + derivative_ratio
    force = np.empty(points.shape, dtype=np.float64)
    force[..., 0] = -(
        derivative_p11 * phi_xx + 2.0 * C1 * (phi_yy + phi_zz)
    )
    force[..., 1] = (
        derivative_ratio * phi_xx * phi_y
        - (derivative_scalar - ratio) * phi_xy
    )
    force[..., 2] = (
        derivative_ratio * phi_xx * phi_z
        - (derivative_scalar - ratio) * phi_xz
    )
    return force


def _tetra_quadrature(order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Duffy-product quadrature; barycentric weights sum to one."""

    points, weights = np.polynomial.legendre.leggauss(int(order))
    unit = 0.5 * (points + 1.0)
    unit_weights = 0.5 * weights
    barycentric: list[list[float]] = []
    raw_weights: list[float] = []
    for u, wu in zip(unit, unit_weights, strict=True):
        for v, wv in zip(unit, unit_weights, strict=True):
            for w, ww in zip(unit, unit_weights, strict=True):
                r = u
                s = (1.0 - u) * v
                t = (1.0 - u) * (1.0 - v) * w
                barycentric.append([1.0 - r - s - t, r, s, t])
                raw_weights.append(wu * wv * ww * (1.0 - u) ** 2 * (1.0 - v))
    weights_array = np.asarray(raw_weights, dtype=np.float64)
    weights_array /= np.sum(weights_array)
    return np.asarray(barycentric, dtype=np.float64), weights_array


def _mesh(subdivisions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(subdivisions)
    if n < 2:
        raise ValueError("subdivisions must be at least two")
    axis = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    nodes = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    def node(i: int, j: int, k: int) -> int:
        return (i * (n + 1) + j) * (n + 1) + k

    template = np.array(
        [
            [0, 1, 2, 5],
            [0, 2, 4, 5],
            [2, 4, 5, 6],
            [1, 3, 2, 5],
            [3, 5, 7, 2],
            [2, 5, 7, 6],
        ],
        dtype=np.int64,
    )
    elements: list[np.ndarray] = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                cube = np.array(
                    [
                        node(i, j, k),
                        node(i, j, k + 1),
                        node(i, j + 1, k),
                        node(i, j + 1, k + 1),
                        node(i + 1, j, k),
                        node(i + 1, j, k + 1),
                        node(i + 1, j + 1, k),
                        node(i + 1, j + 1, k + 1),
                    ],
                    dtype=np.int64,
                )
                for local in template:
                    elem = cube[local].copy()
                    matrix = np.column_stack(
                        (
                            nodes[elem[1]] - nodes[elem[0]],
                            nodes[elem[2]] - nodes[elem[0]],
                            nodes[elem[3]] - nodes[elem[0]],
                        )
                    )
                    if np.linalg.det(matrix) < 0.0:
                        elem[[2, 3]] = elem[[3, 2]]
                    elements.append(elem)
    elems = np.asarray(elements, dtype=np.int64)
    boundary = np.any(np.isclose(nodes, 0.0) | np.isclose(nodes, 1.0), axis=1)
    return nodes, elems, boundary


def _geometry(nodes: np.ndarray, elems: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates = nodes[elems]
    matrices = np.stack(
        (
            coordinates[:, 1] - coordinates[:, 0],
            coordinates[:, 2] - coordinates[:, 0],
            coordinates[:, 3] - coordinates[:, 0],
        ),
        axis=-1,
    )
    determinants = np.linalg.det(matrices)
    if np.any(determinants <= 0.0):
        raise ValueError("manufactured mesh contains a non-positive tetrahedron")
    gradients = np.empty((elems.shape[0], 4, 3), dtype=np.float64)
    # Row ``a`` of B^{-1} is grad(N_{a+1}) when B stores the three
    # vertex-offset vectors as columns.
    gradients[:, 1:, :] = np.linalg.inv(matrices)
    gradients[:, 0, :] = -np.sum(gradients[:, 1:, :], axis=1)
    return determinants / 6.0, gradients


def _element_b_matrices(gradients: np.ndarray) -> np.ndarray:
    bmat = np.zeros((gradients.shape[0], 9, 12), dtype=np.float64)
    for local_node in range(4):
        for component in range(3):
            for axis in range(3):
                bmat[:, 3 * component + axis, 3 * local_node + component] = gradients[
                    :, local_node, axis
                ]
    return bmat


def _element_dofs(elems: np.ndarray) -> np.ndarray:
    return (
        3 * elems[:, :, None] + np.arange(3, dtype=np.int64)[None, None, :]
    ).reshape((elems.shape[0], 12))


def _load_vector(
    nodes: np.ndarray,
    elems: np.ndarray,
    volumes: np.ndarray,
    barycentric: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Assemble the body load in bounded-memory element chunks."""

    if int(chunk_size) < 1:
        raise ValueError("load quadrature chunk size must be positive")
    load = np.zeros((nodes.shape[0], 3), dtype=np.float64)
    for start in range(0, elems.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), elems.shape[0])
        elem_chunk = elems[start:stop]
        points = np.einsum(
            "qi,eid->eqd", barycentric, nodes[elem_chunk], optimize=True
        )
        force = _body_force(points)
        local = volumes[start:stop, None, None] * np.einsum(
            "q,eqd,qi->eid", weights, force, barycentric, optimize=True
        )
        np.add.at(load, elem_chunk.ravel(), local.reshape((-1, 3)))
    return load.ravel()


def _state_data(
    values: np.ndarray,
    elems: np.ndarray,
    volumes: np.ndarray,
    gradients: np.ndarray,
    bmat: np.ndarray,
    load: np.ndarray,
) -> tuple[float, np.ndarray, sparse.csr_matrix, np.ndarray, np.ndarray]:
    element_values = values.reshape((-1, 3))[elems]
    deformation_gradient = np.einsum(
        "eic,eia->eca", element_values, gradients, optimize=True
    )
    density, piola, tangent = _density_piola_tangent(deformation_gradient)
    # B^T P in the element/node/component ordering used by ``_element_dofs``.
    local_residual = volumes[:, None] * np.einsum(
        "eji,ej->ei", bmat, piola.reshape((-1, 9)), optimize=True
    )
    tangent_matrix = tangent.reshape((-1, 9, 9))
    local_tangent = volumes[:, None, None] * np.einsum(
        "eai,eab,ebj->eij", bmat, tangent_matrix, bmat, optimize=True
    )
    dofs = _element_dofs(elems)
    residual = -np.asarray(load, dtype=np.float64).copy()
    np.add.at(residual, dofs.ravel(), local_residual.ravel())
    rows = np.repeat(dofs, 12, axis=1).ravel()
    columns = np.tile(dofs, (1, 12)).ravel()
    matrix = sparse.coo_matrix(
        (local_tangent.ravel(), (rows, columns)),
        shape=(values.size, values.size),
    ).tocsr()
    energy = float(np.dot(volumes, density) - np.dot(load, values))
    return energy, residual, matrix, deformation_gradient, piola


def _residual_only(
    values: np.ndarray,
    elems: np.ndarray,
    volumes: np.ndarray,
    gradients: np.ndarray,
    bmat: np.ndarray,
    load: np.ndarray,
) -> np.ndarray:
    """Assemble a residual without constructing the sparse tangent."""

    element_values = values.reshape((-1, 3))[elems]
    deformation_gradient = np.einsum(
        "eic,eia->eca", element_values, gradients, optimize=True
    )
    _, piola, _ = _density_piola_tangent(deformation_gradient)
    local_residual = volumes[:, None] * np.einsum(
        "eji,ej->ei", bmat, piola.reshape((-1, 9)), optimize=True
    )
    dofs = _element_dofs(elems)
    residual = -np.asarray(load, dtype=np.float64).copy()
    np.add.at(residual, dofs.ravel(), local_residual.ravel())
    return residual


def _newton_solve(
    initial_values: np.ndarray,
    free: np.ndarray,
    elems: np.ndarray,
    volumes: np.ndarray,
    gradients: np.ndarray,
    bmat: np.ndarray,
    load: np.ndarray,
    *,
    relative_tolerance: float,
    max_iterations: int,
    absolute_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Solve one discrete problem and retain the terminal algebraic state."""

    values = np.asarray(initial_values, dtype=np.float64).copy()
    initial_residual: float | None = None
    history: list[dict[str, Any]] = []
    status = "iteration_cap"
    for iteration in range(int(max_iterations) + 1):
        energy, residual, tangent, deformation_gradient, _ = _state_data(
            values, elems, volumes, gradients, bmat, load
        )
        residual_free = residual[free]
        residual_norm = float(np.linalg.norm(residual_free))
        if initial_residual is None:
            initial_residual = residual_norm
        relative_residual = residual_norm / max(
            float(initial_residual), np.finfo(np.float64).tiny
        )
        minimum_determinant = float(np.min(np.linalg.det(deformation_gradient)))
        relative_converged = relative_residual <= float(relative_tolerance)
        absolute_converged = residual_norm <= float(absolute_tolerance)
        if relative_converged or absolute_converged:
            status = "converged"
            history.append(
                {
                    "iteration": int(iteration),
                    "residual_norm": float(residual_norm),
                    "relative_residual": float(relative_residual),
                    "energy": float(energy),
                    "alpha": None,
                    "minimum_element_determinant": minimum_determinant,
                    "convergence_reason": (
                        "relative" if relative_converged else "absolute"
                    ),
                }
            )
            break
        if iteration == int(max_iterations):
            break
        step = sparse_linalg.spsolve(tangent[free][:, free], -residual_free)
        directional = float(np.dot(residual_free, step))
        if (
            not np.all(np.isfinite(step))
            or not np.isfinite(directional)
            or directional >= 0.0
        ):
            status = "invalid_newton_direction"
            break
        alpha = 1.0
        accepted = False
        used_roundoff_acceptance = False
        accepted_trial_relative_residual: float | None = None
        accepted_relative_correction: float | None = None
        accepted_energy_roundoff_tolerance: float | None = None
        for _ in range(40):
            trial = values.copy()
            trial[free] += alpha * step
            try:
                trial_state = _state_data(
                    trial, elems, volumes, gradients, bmat, load
                )
                trial_energy = float(trial_state[0])
                trial_residual_norm = float(
                    np.linalg.norm(np.asarray(trial_state[1])[free])
                )
            except ValueError:
                trial_energy = np.inf
                trial_residual_norm = np.inf
            trial_relative_residual = trial_residual_norm / max(
                float(initial_residual), np.finfo(np.float64).tiny
            )
            relative_correction = float(
                np.linalg.norm(alpha * step)
                / max(np.linalg.norm(values[free]), 1.0)
            )
            armijo_passed = bool(
                np.isfinite(trial_energy)
                and trial_energy <= energy + 1.0e-4 * alpha * directional
            )
            energy_roundoff_tolerance = float(
                64.0
                * np.finfo(np.float64).eps
                * max(1.0, abs(float(energy)), abs(float(trial_energy)))
            )
            roundoff_converged = bool(
                np.isfinite(trial_energy)
                and trial_energy <= energy + energy_roundoff_tolerance
                and relative_correction <= math.sqrt(np.finfo(np.float64).eps)
                and (
                    trial_relative_residual <= float(relative_tolerance)
                    or trial_residual_norm <= float(absolute_tolerance)
                )
            )
            if armijo_passed or roundoff_converged:
                values = trial
                accepted = True
                used_roundoff_acceptance = bool(
                    roundoff_converged and not armijo_passed
                )
                accepted_trial_relative_residual = float(
                    trial_relative_residual
                )
                accepted_relative_correction = float(relative_correction)
                accepted_energy_roundoff_tolerance = float(
                    energy_roundoff_tolerance
                )
                break
            alpha *= 0.5
        history.append(
            {
                "iteration": int(iteration),
                "residual_norm": float(residual_norm),
                "relative_residual": float(relative_residual),
                "energy": float(energy),
                "alpha": float(alpha) if accepted else None,
                "minimum_element_determinant": minimum_determinant,
                "convergence_reason": None,
                "used_roundoff_acceptance": bool(used_roundoff_acceptance),
                "accepted_trial_relative_residual": (
                    accepted_trial_relative_residual
                ),
                "accepted_relative_correction": accepted_relative_correction,
                "energy_roundoff_tolerance": (
                    accepted_energy_roundoff_tolerance
                ),
            }
        )
        if not accepted:
            status = "line_search_failure"
            break

    energy, residual, tangent, deformation_gradient, piola = _state_data(
        values, elems, volumes, gradients, bmat, load
    )
    return {
        "values": values,
        "energy": float(energy),
        "residual": residual,
        "tangent": tangent,
        "deformation_gradient": deformation_gradient,
        "piola": piola,
        "status": str(status),
        "initial_residual_norm": float(initial_residual or 0.0),
        "final_residual_norm": float(np.linalg.norm(residual[free])),
        "final_relative_residual": float(
            np.linalg.norm(residual[free])
            / max(float(initial_residual or 0.0), np.finfo(np.float64).tiny)
        ),
        "history": history,
    }


def _discrete_solution_difference(
    primary_values: np.ndarray,
    refined_values: np.ndarray,
    elems: np.ndarray,
    volumes: np.ndarray,
    gradients: np.ndarray,
    barycentric: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Measure a load-induced discrete-solution change in the FE-error norms."""

    primary_element_values = primary_values.reshape((-1, 3))[elems]
    refined_element_values = refined_values.reshape((-1, 3))[elems]
    nodal_difference = primary_element_values - refined_element_values
    pointwise_difference = np.einsum(
        "qi,eid->eqd", barycentric, nodal_difference, optimize=True
    )
    gradient_difference = np.einsum(
        "eic,eia->eca", nodal_difference, gradients, optimize=True
    )
    primary_gradient = np.einsum(
        "eic,eia->eca", primary_element_values, gradients, optimize=True
    )
    refined_gradient = np.einsum(
        "eic,eia->eca", refined_element_values, gradients, optimize=True
    )
    _, primary_piola, _ = _density_piola_tangent(primary_gradient)
    _, refined_piola, _ = _density_piola_tangent(refined_gradient)
    return {
        "l2_displacement": math.sqrt(
            float(
                np.sum(
                    volumes[:, None]
                    * weights[None, :]
                    * np.sum(pointwise_difference**2, axis=-1)
                )
            )
        ),
        "h1_deformation": math.sqrt(
            float(np.dot(volumes, np.sum(gradient_difference**2, axis=(-2, -1))))
        ),
        "first_piola_l2": math.sqrt(
            float(
                np.dot(
                    volumes,
                    np.sum((primary_piola - refined_piola) ** 2, axis=(-2, -1)),
                )
            )
        ),
    }


def _solve_level(
    subdivisions: int,
    *,
    relative_tolerance: float,
    max_iterations: int,
    load_quadrature_order: int,
    load_quadrature_refinement_order: int,
    load_quadrature_confirmation_order: int,
    maximum_load_quadrature_error_fraction: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    nodes, elems, boundary_nodes = _mesh(int(subdivisions))
    volumes, gradients = _geometry(nodes, elems)
    bmat = _element_b_matrices(gradients)
    barycentric, weights = _tetra_quadrature(int(load_quadrature_order))
    refined_barycentric, refined_weights = _tetra_quadrature(
        int(load_quadrature_refinement_order)
    )
    confirmation_barycentric, confirmation_weights = _tetra_quadrature(
        int(load_quadrature_confirmation_order)
    )
    load = _load_vector(nodes, elems, volumes, barycentric, weights)
    refined_load = _load_vector(
        nodes, elems, volumes, refined_barycentric, refined_weights
    )
    confirmation_load = _load_vector(
        nodes, elems, volumes, confirmation_barycentric, confirmation_weights
    )
    boundary_dofs = np.repeat(boundary_nodes, 3)
    free = np.flatnonzero(~boundary_dofs)

    exact_nodes = _exact_deformation(nodes).ravel()
    initial_values = nodes.ravel().copy()
    initial_values[boundary_dofs] = exact_nodes[boundary_dofs]
    primary = _newton_solve(
        initial_values,
        free,
        elems,
        volumes,
        gradients,
        bmat,
        load,
        relative_tolerance=float(relative_tolerance),
        max_iterations=int(max_iterations),
    )
    primary_refined_load_difference = float(
        np.linalg.norm((load - refined_load)[free])
    )
    refined_interpolant_residual = _residual_only(
        exact_nodes, elems, volumes, gradients, bmat, refined_load
    )
    refined_interpolant_residual_norm = float(
        np.linalg.norm(refined_interpolant_residual[free])
    )
    refined_load_norm = float(np.linalg.norm(refined_load[free]))
    refined_absolute_tolerance = float(
        max(
            2.0e-1 * primary_refined_load_difference,
            100.0 * np.finfo(np.float64).eps * max(refined_load_norm, 1.0),
        )
    )
    refined = _newton_solve(
        primary["values"],
        free,
        elems,
        volumes,
        gradients,
        bmat,
        refined_load,
        relative_tolerance=float(relative_tolerance),
        max_iterations=int(max_iterations),
        absolute_tolerance=refined_absolute_tolerance,
    )
    values = primary["values"]
    energy = primary["energy"]
    residual = primary["residual"]
    tangent = primary["tangent"]
    deformation_gradient = primary["deformation_gradient"]
    points_q = np.einsum("qi,eid->eqd", barycentric, nodes[elems], optimize=True)
    numerical_q = np.einsum(
        "qi,eid->eqd", barycentric, values.reshape((-1, 3))[elems], optimize=True
    )
    exact_q = _exact_deformation(points_q)
    exact_gradient_q = _exact_gradient(points_q)
    _, exact_piola_q, _ = _density_piola_tangent(exact_gradient_q)
    numerical_gradient_q = deformation_gradient[:, None, :, :]
    _, numerical_piola_q, _ = _density_piola_tangent(numerical_gradient_q)

    l2_error = math.sqrt(
        float(
            np.sum(
                volumes[:, None]
                * weights[None, :]
                * np.sum((numerical_q - exact_q) ** 2, axis=-1)
            )
        )
    )
    h1_error = math.sqrt(
        float(
            np.sum(
                volumes[:, None]
                * weights[None, :]
                * np.sum((numerical_gradient_q - exact_gradient_q) ** 2, axis=(-2, -1))
            )
        )
    )
    stress_error = math.sqrt(
        float(
            np.sum(
                volumes[:, None]
                * weights[None, :]
                * np.sum((numerical_piola_q - exact_piola_q) ** 2, axis=(-2, -1))
            )
        )
    )
    solution_difference = _discrete_solution_difference(
        primary["values"],
        refined["values"],
        elems,
        volumes,
        gradients,
        refined_barycentric,
        refined_weights,
    )
    fe_errors = {
        "l2_displacement": float(l2_error),
        "h1_deformation": float(h1_error),
        "first_piola_l2": float(stress_error),
    }
    error_fractions = {
        key: float(solution_difference[key] / max(fe_errors[key], np.finfo(float).tiny))
        for key in fe_errors
    }
    refined_confirmation_load_difference = float(
        np.linalg.norm((refined_load - confirmation_load)[free])
    )
    reference_load_norm = float(np.linalg.norm(confirmation_load[free]))
    roundoff_floor = float(
        100.0 * np.finfo(np.float64).eps * max(reference_load_norm, 1.0)
    )
    load_reference_stable = bool(
        refined_confirmation_load_difference
        <= max(0.25 * primary_refined_load_difference, roundoff_floor)
    )
    load_quadrature_check = {
        "primary_order": int(load_quadrature_order),
        "refinement_order": int(load_quadrature_refinement_order),
        "confirmation_order": int(load_quadrature_confirmation_order),
        "free_load_primary_refinement_absolute_difference": primary_refined_load_difference,
        "free_load_primary_refinement_relative_difference": float(
            primary_refined_load_difference
            / max(reference_load_norm, np.finfo(np.float64).tiny)
        ),
        "free_load_primary_refinement_fraction_of_interpolant_consistency_residual": float(
            primary_refined_load_difference
            / max(refined_interpolant_residual_norm, np.finfo(np.float64).tiny)
        ),
        "refined_interpolant_consistency_residual_norm": refined_interpolant_residual_norm,
        "free_load_refinement_confirmation_absolute_difference": refined_confirmation_load_difference,
        "free_load_refinement_confirmation_relative_difference": float(
            refined_confirmation_load_difference
            / max(reference_load_norm, np.finfo(np.float64).tiny)
        ),
        "reference_load_stable": load_reference_stable,
        "refined_solution_status": str(refined["status"]),
        "refined_solution_newton_iterations": int(
            sum(item["alpha"] is not None for item in refined["history"])
        ),
        "refined_solution_final_relative_residual": float(
            refined["final_relative_residual"]
        ),
        "refined_solution_final_residual_norm": float(
            refined["final_residual_norm"]
        ),
        "refined_solution_absolute_tolerance": refined_absolute_tolerance,
        "refined_solution_terminal_residual_fraction_of_load_change": float(
            refined["final_residual_norm"]
            / max(primary_refined_load_difference, np.finfo(np.float64).tiny)
        ),
        "refined_solution_resolves_load_change": bool(
            refined["final_residual_norm"]
            <= max(
                2.5e-1 * primary_refined_load_difference,
                refined_absolute_tolerance,
            )
        ),
        "solution_difference": solution_difference,
        "fraction_of_fe_error": error_fractions,
        "maximum_fraction_of_fe_error": float(max(error_fractions.values())),
        "maximum_accepted_fraction_of_fe_error": float(
            maximum_load_quadrature_error_fraction
        ),
        "below_fe_error": bool(
            max(error_fractions.values())
            <= float(maximum_load_quadrature_error_fraction)
        ),
    }
    symmetry = tangent - tangent.T
    return {
        "subdivisions": int(subdivisions),
        "h": float(1.0 / int(subdivisions)),
        "free_dofs": int(free.size),
        "status": str(primary["status"]),
        "newton_iterations": int(
            sum(item["alpha"] is not None for item in primary["history"])
        ),
        "final_relative_residual": float(primary["final_relative_residual"]),
        "l2_displacement_error": float(l2_error),
        "h1_deformation_error": float(h1_error),
        "first_piola_l2_error": float(stress_error),
        "minimum_discrete_determinant": float(np.min(np.linalg.det(deformation_gradient))),
        "minimum_exact_determinant": float(np.min(np.linalg.det(exact_gradient_q))),
        "tangent_symmetry_defect": float(
            sparse_linalg.norm(symmetry)
            / max(float(sparse_linalg.norm(tangent)), np.finfo(np.float64).tiny)
        ),
        "energy": float(energy),
        "load_quadrature_check": load_quadrature_check,
        "wall_seconds": float(time.perf_counter() - started),
        "history": primary["history"],
    }


def _rate(coarse: dict[str, Any], fine: dict[str, Any], field: str) -> float:
    return float(
        math.log(float(coarse[field]) / float(fine[field]))
        / math.log(float(coarse["h"]) / float(fine["h"]))
    )


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def run(args: argparse.Namespace) -> dict[str, Any]:
    subdivisions = [int(value) for value in args.subdivisions]
    if len(subdivisions) < 2:
        raise ValueError("at least two mesh levels are required")
    if subdivisions != sorted(set(subdivisions)):
        raise ValueError("subdivisions must be strictly increasing and unique")
    quadrature_orders = (
        int(args.load_quadrature_order),
        int(args.load_quadrature_refinement_order),
        int(args.load_quadrature_confirmation_order),
    )
    if quadrature_orders[0] < 1 or not (
        quadrature_orders[0] < quadrature_orders[1] < quadrature_orders[2]
    ):
        raise ValueError(
            "load quadrature orders must be positive and strictly increasing"
        )
    if not 0.0 < float(args.maximum_load_quadrature_error_fraction) < 1.0:
        raise ValueError(
            "maximum load quadrature error fraction must lie strictly between zero and one"
        )

    levels = [
        _solve_level(
            int(mesh_subdivisions),
            relative_tolerance=float(args.relative_tolerance),
            max_iterations=int(args.max_iterations),
            load_quadrature_order=quadrature_orders[0],
            load_quadrature_refinement_order=quadrature_orders[1],
            load_quadrature_confirmation_order=quadrature_orders[2],
            maximum_load_quadrature_error_fraction=float(
                args.maximum_load_quadrature_error_fraction
            ),
        )
        for mesh_subdivisions in subdivisions
    ]
    fields = (
        "l2_displacement_error",
        "h1_deformation_error",
        "first_piola_l2_error",
    )
    rates = [
        {
            "coarse_subdivisions": int(coarse["subdivisions"]),
            "fine_subdivisions": int(fine["subdivisions"]),
            **{field: _rate(coarse, fine, field) for field in fields},
        }
        for coarse, fine in zip(levels[:-1], levels[1:], strict=True)
    ]
    final_rate = rates[-1]
    gates = {
        "all_levels_converged": all(level["status"] == "converged" for level in levels),
        "algebraic_residual": max(level["final_relative_residual"] for level in levels)
        <= float(args.relative_tolerance),
        "orientation": min(level["minimum_discrete_determinant"] for level in levels) > 0.5,
        "tangent_symmetry": max(level["tangent_symmetry_defect"] for level in levels) <= 1.0e-11,
        "l2_rate": float(final_rate["l2_displacement_error"]) >= 1.75,
        "h1_rate": float(final_rate["h1_deformation_error"]) >= 0.75,
        "stress_rate": float(final_rate["first_piola_l2_error"]) >= 0.75,
        "load_quadrature_reference_stable": all(
            level["load_quadrature_check"]["reference_load_stable"]
            for level in levels
        ),
        "load_quadrature_refined_solves_converged": all(
            level["load_quadrature_check"]["refined_solution_status"] == "converged"
            for level in levels
        ),
        "load_quadrature_refined_solves_resolve_load_change": all(
            level["load_quadrature_check"]["refined_solution_resolves_load_change"]
            for level in levels
        ),
        "load_quadrature_below_fe_error": all(
            level["load_quadrature_check"]["below_fe_error"] for level in levels
        ),
        "load_quadrature_below_consistency_error": max(
            level["load_quadrature_check"][
                "free_load_primary_refinement_fraction_of_interpolant_consistency_residual"
            ]
            for level in levels
        )
        <= float(args.maximum_load_quadrature_error_fraction),
    }
    repo_root = Path(__file__).resolve().parents[2]
    runner = Path(__file__).resolve()
    return {
        "schema_version": 2,
        "experiment_id": "EXP-VAL-001-HYPERELASTIC-NONAFFINE-MANUFACTURED",
        "status": "passed" if all(gates.values()) else "failed",
        "publication_evidence": False,
        "case": {
            "domain": "unit cube",
            "element": "P1 tetrahedron",
            "exact_deformation": (
                "y=(X+0.05*sin(pi*X)*sin(pi*Y)*sin(pi*Z),Y,Z)"
            ),
            "boundary_condition": "exact deformation on the whole boundary",
            "body_force": "analytic -Div(P(F_exact))",
            "C1": C1,
            "D1": D1,
            "load_quadrature": (
                f"{quadrature_orders[0]}x{quadrature_orders[0]}x"
                f"{quadrature_orders[0]} Gauss-Legendre Duffy product"
            ),
            "load_quadrature_refinement": (
                f"orders {quadrature_orders[1]} and {quadrature_orders[2]}"
            ),
            "error_quadrature": (
                f"{quadrature_orders[0]}x{quadrature_orders[0]}x"
                f"{quadrature_orders[0]} Gauss-Legendre Duffy product"
            ),
            "initialization": "identity deformation with exact boundary values",
        },
        "contract": {
            "subdivisions": subdivisions,
            "relative_algebraic_residual": float(args.relative_tolerance),
            "last_pair_minimum_rates": {
                "l2_displacement": 1.75,
                "h1_deformation": 0.75,
                "first_piola_l2": 0.75,
            },
            "minimum_determinant": 0.5,
            "tangent_symmetry_tolerance": 1.0e-11,
            "load_quadrature_orders": list(quadrature_orders),
            "maximum_load_quadrature_error_fraction": float(
                args.maximum_load_quadrature_error_fraction
            ),
        },
        "levels": levels,
        "rates": rates,
        "gates": gates,
        "limitations": [
            "This is an independent manufactured-formulation check, not a matched production-backend comparison.",
            "All boundaries are Dirichlet; reaction-force validation remains covered by the affine patch test.",
            "The dirty-worktree result is diagnostic until rerun from a clean frozen commit.",
        ],
        "provenance": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
            "git": _git_metadata(repo_root),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--subdivisions",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="strictly increasing mesh subdivisions; append 24 for a finer local check",
    )
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--load-quadrature-order", type=int, default=4)
    parser.add_argument("--load-quadrature-refinement-order", type=int, default=6)
    parser.add_argument("--load-quadrature-confirmation-order", type=int, default=8)
    parser.add_argument(
        "--maximum-load-quadrature-error-fraction",
        type=float,
        default=1.0e-2,
        help="maximum refined-load solution change divided by the FE error",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = run(args)
    atomic_write_json(Path(args.output).resolve(), payload)
    print(Path(args.output).resolve())
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

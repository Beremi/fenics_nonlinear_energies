#!/usr/bin/env python3
"""Manufactured P1 verification for the smooth p=3 Laplace problem.

The verifier is intentionally independent of the production JAX/PETSc energy
and assembly paths.  It assembles the scalar residual and consistent tangent
from the weak form with NumPy/SciPy, solves by damped Newton, and measures the
error against a manufactured solution whose gradient is bounded away from
zero.  This isolates spatial consistency from AD placement and MPI assembly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


AMPLITUDE = 0.1
P_EXPONENT = 3.0


def _quadrature_rule() -> tuple[np.ndarray, np.ndarray]:
    """Return degree-five Dunavant barycentric points and weights summing to one."""
    xi = np.array(
        [
            [
                0.1012865073235,
                0.7974269853531,
                0.1012865073235,
                0.4701420641051,
                0.4701420641051,
                0.0597158717898,
                1.0 / 3.0,
            ],
            [
                0.1012865073235,
                0.1012865073235,
                0.7974269853531,
                0.0597158717898,
                0.4701420641051,
                0.4701420641051,
                1.0 / 3.0,
            ],
        ],
        dtype=np.float64,
    )
    weights = np.array(
        [
            0.1259391805448,
            0.1259391805448,
            0.1259391805448,
            0.1323941527885,
            0.1323941527885,
            0.1323941527885,
            0.225,
        ],
        dtype=np.float64,
    )
    barycentric = np.column_stack((1.0 - xi[0] - xi[1], xi[0], xi[1]))
    weights /= np.sum(weights)
    return barycentric, weights


def _exact_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + y + AMPLITUDE * np.sin(np.pi * x) * np.sin(np.pi * y)


def _exact_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            1.0 + AMPLITUDE * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
            1.0 + AMPLITUDE * np.pi * np.sin(np.pi * x) * np.cos(np.pi * y),
        ),
        axis=-1,
    )


def _source(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return -div(|grad u| grad u) for the manufactured p=3 solution."""
    gradient = _exact_gradient(x, y)
    common = AMPLITUDE * np.pi**2
    h_xx = -common * np.sin(np.pi * x) * np.sin(np.pi * y)
    h_yy = h_xx
    h_xy = common * np.cos(np.pi * x) * np.cos(np.pi * y)
    norm = np.linalg.norm(gradient, axis=-1)
    hessian_gradient_x = h_xx * gradient[..., 0] + h_xy * gradient[..., 1]
    hessian_gradient_y = h_xy * gradient[..., 0] + h_yy * gradient[..., 1]
    quadratic = (
        gradient[..., 0] * hessian_gradient_x
        + gradient[..., 1] * hessian_gradient_y
    )
    return -(norm * (h_xx + h_yy) + quadratic / norm)


def _mesh(subdivisions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(subdivisions)
    axis = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    nodes = np.column_stack((xx.ravel(), yy.ravel()))
    elements: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(n):
            lower_left = i * (n + 1) + j
            upper_left = lower_left + 1
            lower_right = (i + 1) * (n + 1) + j
            upper_right = lower_right + 1
            elements.append((lower_left, lower_right, upper_right))
            elements.append((lower_left, upper_right, upper_left))
    elems = np.asarray(elements, dtype=np.int64)
    boundary = (
        np.isclose(nodes[:, 0], 0.0)
        | np.isclose(nodes[:, 0], 1.0)
        | np.isclose(nodes[:, 1], 0.0)
        | np.isclose(nodes[:, 1], 1.0)
    )
    return nodes, elems, boundary


def _geometry(nodes: np.ndarray, elems: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = nodes[elems]
    x0, y0 = coords[:, 0, 0], coords[:, 0, 1]
    x1, y1 = coords[:, 1, 0], coords[:, 1, 1]
    x2, y2 = coords[:, 2, 0], coords[:, 2, 1]
    twice_area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if np.any(twice_area <= 0.0):
        raise ValueError("manufactured mesh contains a non-positive triangle")
    gradients = np.empty((elems.shape[0], 3, 2), dtype=np.float64)
    gradients[:, 0, 0] = (y1 - y2) / twice_area
    gradients[:, 0, 1] = (x2 - x1) / twice_area
    gradients[:, 1, 0] = (y2 - y0) / twice_area
    gradients[:, 1, 1] = (x0 - x2) / twice_area
    gradients[:, 2, 0] = (y0 - y1) / twice_area
    gradients[:, 2, 1] = (x1 - x0) / twice_area
    return 0.5 * twice_area, gradients


def _load_vector(
    nodes: np.ndarray,
    elems: np.ndarray,
    area: np.ndarray,
    barycentric: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    coords_q = np.einsum("qi,eid->eqd", barycentric, nodes[elems], optimize=True)
    source_q = _source(coords_q[..., 0], coords_q[..., 1])
    local = area[:, None] * np.einsum(
        "q,eq,qi->ei", weights, source_q, barycentric, optimize=True
    )
    load = np.zeros(nodes.shape[0], dtype=np.float64)
    np.add.at(load, elems.ravel(), local.ravel())
    return load


def _internal_state(
    values: np.ndarray,
    elems: np.ndarray,
    gradients: np.ndarray,
    area: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    element_gradient = np.einsum(
        "ei,eid->ed", values[elems], gradients, optimize=True
    )
    gradient_norm = np.linalg.norm(element_gradient, axis=1)
    if np.any(gradient_norm <= 0.0):
        raise FloatingPointError("manufactured Newton iterate reached zero element gradient")
    flux = gradient_norm[:, None] * element_gradient
    local_residual = area[:, None] * np.einsum(
        "ed,eid->ei", flux, gradients, optimize=True
    )
    identity = np.broadcast_to(np.eye(2), (elems.shape[0], 2, 2))
    constitutive = gradient_norm[:, None, None] * identity + (
        np.einsum("ei,ej->eij", element_gradient, element_gradient, optimize=True)
        / gradient_norm[:, None, None]
    )
    local_tangent = area[:, None, None] * np.einsum(
        "eia,eab,ejb->eij", gradients, constitutive, gradients, optimize=True
    )
    return element_gradient, local_residual, local_tangent


def _assemble_residual_and_tangent(
    values: np.ndarray,
    elems: np.ndarray,
    gradients: np.ndarray,
    area: np.ndarray,
    load: np.ndarray,
) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray]:
    element_gradient, local_residual, local_tangent = _internal_state(
        values, elems, gradients, area
    )
    residual = -np.asarray(load, dtype=np.float64).copy()
    np.add.at(residual, elems.ravel(), local_residual.ravel())
    rows = np.repeat(elems, 3, axis=1).ravel()
    cols = np.tile(elems, (1, 3)).ravel()
    tangent = sparse.coo_matrix(
        (local_tangent.ravel(), (rows, cols)),
        shape=(values.size, values.size),
    ).tocsr()
    return residual, tangent, element_gradient


def _energy(
    values: np.ndarray,
    elems: np.ndarray,
    gradients: np.ndarray,
    area: np.ndarray,
    load: np.ndarray,
) -> float:
    element_gradient = np.einsum(
        "ei,eid->ed", values[elems], gradients, optimize=True
    )
    internal = np.sum(area * np.linalg.norm(element_gradient, axis=1) ** P_EXPONENT)
    return float(internal / P_EXPONENT - np.dot(load, values))


def _solve_level(subdivisions: int, *, relative_tolerance: float, max_iterations: int) -> dict[str, Any]:
    nodes, elems, boundary = _mesh(subdivisions)
    area, gradients = _geometry(nodes, elems)
    barycentric, weights = _quadrature_rule()
    load = _load_vector(nodes, elems, area, barycentric, weights)
    free = np.flatnonzero(~boundary)
    values = nodes[:, 0] + nodes[:, 1]
    values[boundary] = _exact_solution(nodes[boundary, 0], nodes[boundary, 1])

    initial_norm: float | None = None
    history: list[dict[str, Any]] = []
    status = "iteration_cap"
    for iteration in range(int(max_iterations) + 1):
        residual, tangent, element_gradient = _assemble_residual_and_tangent(
            values, elems, gradients, area, load
        )
        residual_free = residual[free]
        residual_norm = float(np.linalg.norm(residual_free))
        if initial_norm is None:
            initial_norm = residual_norm
        relative_residual = residual_norm / max(initial_norm, np.finfo(float).tiny)
        min_gradient_norm = float(np.min(np.linalg.norm(element_gradient, axis=1)))
        if relative_residual <= float(relative_tolerance):
            status = "converged"
            history.append(
                {
                    "iteration": iteration,
                    "residual_norm": residual_norm,
                    "relative_residual": relative_residual,
                    "energy": _energy(values, elems, gradients, area, load),
                    "alpha": None,
                    "minimum_element_gradient_norm": min_gradient_norm,
                }
            )
            break
        if iteration == int(max_iterations):
            break

        tangent_free = tangent[free][:, free]
        step = sparse_linalg.spsolve(tangent_free, -residual_free)
        if not np.all(np.isfinite(step)):
            raise FloatingPointError("manufactured Newton solve produced a nonfinite step")
        directional = float(np.dot(residual_free, step))
        if not np.isfinite(directional) or directional >= 0.0:
            raise RuntimeError("manufactured Newton direction is not a descent direction")
        old_energy = _energy(values, elems, gradients, area, load)
        alpha = 1.0
        accepted = False
        for _ in range(40):
            trial = values.copy()
            trial[free] += alpha * step
            trial_energy = _energy(trial, elems, gradients, area, load)
            if np.isfinite(trial_energy) and trial_energy <= old_energy + 1.0e-4 * alpha * directional:
                values = trial
                accepted = True
                break
            alpha *= 0.5
        history.append(
            {
                "iteration": iteration,
                "residual_norm": residual_norm,
                "relative_residual": relative_residual,
                "energy": old_energy,
                "alpha": alpha if accepted else None,
                "minimum_element_gradient_norm": min_gradient_norm,
            }
        )
        if not accepted:
            status = "line_search_failure"
            break

    coords_q = np.einsum("qi,eid->eqd", barycentric, nodes[elems], optimize=True)
    numerical_q = np.einsum("qi,ei->eq", barycentric, values[elems], optimize=True)
    exact_q = _exact_solution(coords_q[..., 0], coords_q[..., 1])
    exact_gradient_q = _exact_gradient(coords_q[..., 0], coords_q[..., 1])
    numerical_gradient = np.einsum(
        "ei,eid->ed", values[elems], gradients, optimize=True
    )
    l2_squared = float(
        np.sum(area[:, None] * weights[None, :] * (numerical_q - exact_q) ** 2)
    )
    h1_squared = float(
        np.sum(
            area[:, None]
            * weights[None, :]
            * np.sum((numerical_gradient[:, None, :] - exact_gradient_q) ** 2, axis=-1)
        )
    )
    final_residual, final_tangent, final_element_gradient = _assemble_residual_and_tangent(
        values, elems, gradients, area, load
    )
    symmetry = final_tangent - final_tangent.T
    return {
        "subdivisions": int(subdivisions),
        "h": 1.0 / float(subdivisions),
        "nodes": int(nodes.shape[0]),
        "elements": int(elems.shape[0]),
        "free_dofs": int(free.size),
        "status": status,
        "newton_iterations": int(sum(entry["alpha"] is not None for entry in history)),
        "initial_residual_norm": float(initial_norm if initial_norm is not None else 0.0),
        "final_residual_norm": float(np.linalg.norm(final_residual[free])),
        "final_relative_residual": float(
            np.linalg.norm(final_residual[free])
            / max(float(initial_norm or 0.0), np.finfo(float).tiny)
        ),
        "l2_error": math.sqrt(max(l2_squared, 0.0)),
        "h1_seminorm_error": math.sqrt(max(h1_squared, 0.0)),
        "minimum_element_gradient_norm": float(
            np.min(np.linalg.norm(final_element_gradient, axis=1))
        ),
        "tangent_symmetry_defect": float(
            np.linalg.norm(symmetry.data)
            / max(np.linalg.norm(final_tangent.data), np.finfo(float).tiny)
        ),
        "history": history,
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
    levels = [_solve_level(n, relative_tolerance=args.relative_tolerance, max_iterations=args.max_iterations) for n in args.subdivisions]
    rates = [
        {
            "coarse_subdivisions": int(coarse["subdivisions"]),
            "fine_subdivisions": int(fine["subdivisions"]),
            "l2_rate": _rate(coarse, fine, "l2_error"),
            "h1_seminorm_rate": _rate(coarse, fine, "h1_seminorm_error"),
        }
        for coarse, fine in zip(levels[:-1], levels[1:], strict=True)
    ]
    last_rate = rates[-1]
    passed = bool(
        all(level["status"] == "converged" for level in levels)
        and all(level["minimum_element_gradient_norm"] > 0.5 for level in levels)
        and all(level["tangent_symmetry_defect"] <= 1.0e-12 for level in levels)
        and float(last_rate["l2_rate"]) >= float(args.minimum_l2_rate)
        and float(last_rate["h1_seminorm_rate"]) >= float(args.minimum_h1_rate)
    )
    repo_root = Path(__file__).resolve().parents[2]
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "experiment_id": "EXP-VAL-001-PLAPLACE-MANUFACTURED",
        "status": "passed" if passed else "failed",
        "publication_evidence": False,
        "problem": {
            "domain": "unit_square",
            "element": "P1 triangles",
            "p": P_EXPONENT,
            "exact_solution": "x + y + 0.1 sin(pi x) sin(pi y)",
            "boundary_condition": "exact nonhomogeneous Dirichlet data",
            "source": "analytic -div(|grad u| grad u)",
            "load_quadrature": "degree-five seven-point Dunavant rule",
            "gradient_lower_bound": float(1.0 - AMPLITUDE * np.pi),
        },
        "solver_contract": {
            "method": "independently assembled damped Newton",
            "relative_residual_tolerance": float(args.relative_tolerance),
            "maximum_iterations": int(args.max_iterations),
            "armijo_c1": 1.0e-4,
            "backtracking_factor": 0.5,
            "maximum_backtracks": 40,
        },
        "acceptance_contract": {
            "minimum_last_l2_rate": float(args.minimum_l2_rate),
            "minimum_last_h1_seminorm_rate": float(args.minimum_h1_rate),
            "maximum_symmetry_defect": 1.0e-12,
            "minimum_discrete_gradient_norm": 0.5,
        },
        "levels": levels,
        "rates": rates,
        "limitations": [
            "This validates a smooth manufactured scalar problem, not the L-shaped production load case.",
            "The verifier uses an independent NumPy/SciPy P1 assembly but the same mathematical weak form.",
            "The dirty-worktree result is diagnostic until rerun from a clean frozen commit.",
        ],
        "provenance": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "python": sys.version.split()[0],
            "runner_sha256": source_hash,
            "git": _git_metadata(repo_root),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--minimum-l2-rate", type=float, default=1.75)
    parser.add_argument("--minimum-h1-rate", type=float, default=0.85)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if len(args.subdivisions) < 3 or any(n < 2 for n in args.subdivisions):
        raise SystemExit("provide at least three subdivision counts, each >= 2")
    if any(fine <= coarse for coarse, fine in zip(args.subdivisions[:-1], args.subdivisions[1:], strict=True)):
        raise SystemExit("subdivision counts must be strictly increasing")
    payload = run(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output)
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

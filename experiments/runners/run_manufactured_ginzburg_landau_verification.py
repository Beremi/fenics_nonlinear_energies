#!/usr/bin/env python3
"""Manufactured P1 verification for the smooth Ginzburg--Landau weak form."""

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

from experiments.runners.run_manufactured_plaplace_verification import (
    _geometry,
    _mesh,
    _quadrature_rule,
)


EPSILON = 0.04
BASE = 0.8
AMPLITUDE = 0.1


def _solve_quadrature() -> tuple[np.ndarray, np.ndarray]:
    barycentric = np.array(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ],
        dtype=np.float64,
    )
    return barycentric, np.full(3, 1.0 / 3.0, dtype=np.float64)


def _exact_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return BASE + AMPLITUDE * np.sin(np.pi * x) * np.sin(np.pi * y)


def _exact_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            AMPLITUDE * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
            AMPLITUDE * np.pi * np.sin(np.pi * x) * np.cos(np.pi * y),
        ),
        axis=-1,
    )


def _source(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    solution = _exact_solution(x, y)
    sine_product = np.sin(np.pi * x) * np.sin(np.pi * y)
    return 2.0 * EPSILON * AMPLITUDE * np.pi**2 * sine_product + solution * (
        solution**2 - 1.0
    )


def _element_data(
    values: np.ndarray,
    nodes: np.ndarray,
    elems: np.ndarray,
    gradients: np.ndarray,
    area: np.ndarray,
    barycentric: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords_q = np.einsum("qi,eid->eqd", barycentric, nodes[elems], optimize=True)
    values_q = np.einsum("qi,ei->eq", barycentric, values[elems], optimize=True)
    source_q = _source(coords_q[..., 0], coords_q[..., 1])
    element_gradient = np.einsum("ei,eid->ed", values[elems], gradients, optimize=True)
    reaction = values_q * (values_q**2 - 1.0) - source_q
    local_residual = area[:, None] * (
        EPSILON * np.einsum("ed,eid->ei", element_gradient, gradients, optimize=True)
        + np.einsum("q,eq,qi->ei", weights, reaction, barycentric, optimize=True)
    )
    local_tangent = area[:, None, None] * (
        EPSILON * np.einsum("eia,eja->eij", gradients, gradients, optimize=True)
        + np.einsum(
            "q,eq,qi,qj->eij",
            weights,
            3.0 * values_q**2 - 1.0,
            barycentric,
            barycentric,
            optimize=True,
        )
    )
    return coords_q, values_q, source_q, local_residual, local_tangent


def _assemble(
    values: np.ndarray,
    nodes: np.ndarray,
    elems: np.ndarray,
    gradients: np.ndarray,
    area: np.ndarray,
    barycentric: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, sparse.csr_matrix]:
    _, _, _, local_residual, local_tangent = _element_data(
        values, nodes, elems, gradients, area, barycentric, weights
    )
    residual = np.zeros(values.size, dtype=np.float64)
    np.add.at(residual, elems.ravel(), local_residual.ravel())
    rows = np.repeat(elems, 3, axis=1).ravel()
    cols = np.tile(elems, (1, 3)).ravel()
    tangent = sparse.coo_matrix(
        (local_tangent.ravel(), (rows, cols)), shape=(values.size, values.size)
    ).tocsr()
    return residual, tangent


def _energy(
    values: np.ndarray,
    nodes: np.ndarray,
    elems: np.ndarray,
    gradients: np.ndarray,
    area: np.ndarray,
    barycentric: np.ndarray,
    weights: np.ndarray,
) -> float:
    _, values_q, source_q, _, _ = _element_data(
        values, nodes, elems, gradients, area, barycentric, weights
    )
    element_gradient = np.einsum("ei,eid->ed", values[elems], gradients, optimize=True)
    density_q = 0.25 * (values_q**2 - 1.0) ** 2 - source_q * values_q
    return float(
        np.sum(0.5 * EPSILON * area * np.sum(element_gradient**2, axis=1))
        + np.sum(area[:, None] * weights[None, :] * density_q)
    )


def _solve_level(subdivisions: int, tolerance: float, max_iterations: int) -> dict[str, Any]:
    nodes, elems, boundary = _mesh(subdivisions)
    area, gradients = _geometry(nodes, elems)
    barycentric, weights = _solve_quadrature()
    free = np.flatnonzero(~boundary)
    values = np.full(nodes.shape[0], BASE, dtype=np.float64)
    values[boundary] = _exact_solution(nodes[boundary, 0], nodes[boundary, 1])
    initial_norm: float | None = None
    history: list[dict[str, Any]] = []
    status = "iteration_cap"
    for iteration in range(max_iterations + 1):
        residual, tangent = _assemble(
            values, nodes, elems, gradients, area, barycentric, weights
        )
        residual_free = residual[free]
        residual_norm = float(np.linalg.norm(residual_free))
        if initial_norm is None:
            initial_norm = residual_norm
        relative = residual_norm / max(initial_norm, np.finfo(float).tiny)
        if relative <= tolerance:
            status = "converged"
            history.append(
                {
                    "iteration": iteration,
                    "relative_residual": relative,
                    "energy": _energy(values, nodes, elems, gradients, area, barycentric, weights),
                    "alpha": None,
                }
            )
            break
        if iteration == max_iterations:
            break
        step = sparse_linalg.spsolve(tangent[free][:, free], -residual_free)
        directional = float(np.dot(residual_free, step))
        if not np.all(np.isfinite(step)) or not np.isfinite(directional) or directional >= 0.0:
            status = "invalid_newton_direction"
            break
        old_energy = _energy(values, nodes, elems, gradients, area, barycentric, weights)
        alpha = 1.0
        accepted = False
        for _ in range(40):
            trial = values.copy()
            trial[free] += alpha * step
            trial_energy = _energy(trial, nodes, elems, gradients, area, barycentric, weights)
            if np.isfinite(trial_energy) and trial_energy <= old_energy + 1.0e-4 * alpha * directional:
                values = trial
                accepted = True
                break
            alpha *= 0.5
        history.append(
            {
                "iteration": iteration,
                "relative_residual": relative,
                "energy": old_energy,
                "alpha": alpha if accepted else None,
            }
        )
        if not accepted:
            status = "line_search_failure"
            break

    residual, tangent = _assemble(values, nodes, elems, gradients, area, barycentric, weights)
    error_barycentric, error_weights = _quadrature_rule()
    coords_q = np.einsum("qi,eid->eqd", error_barycentric, nodes[elems], optimize=True)
    numerical_q = np.einsum("qi,ei->eq", error_barycentric, values[elems], optimize=True)
    exact_q = _exact_solution(coords_q[..., 0], coords_q[..., 1])
    exact_gradient_q = _exact_gradient(coords_q[..., 0], coords_q[..., 1])
    numerical_gradient = np.einsum("ei,eid->ed", values[elems], gradients, optimize=True)
    l2 = math.sqrt(
        float(np.sum(area[:, None] * error_weights[None, :] * (numerical_q - exact_q) ** 2))
    )
    h1 = math.sqrt(
        float(
            np.sum(
                area[:, None]
                * error_weights[None, :]
                * np.sum((numerical_gradient[:, None, :] - exact_gradient_q) ** 2, axis=-1)
            )
        )
    )
    symmetry = tangent - tangent.T
    return {
        "subdivisions": int(subdivisions),
        "h": 1.0 / subdivisions,
        "free_dofs": int(free.size),
        "status": status,
        "newton_iterations": int(sum(entry["alpha"] is not None for entry in history)),
        "final_relative_residual": float(
            np.linalg.norm(residual[free]) / max(float(initial_norm or 0.0), np.finfo(float).tiny)
        ),
        "l2_error": l2,
        "h1_seminorm_error": h1,
        "tangent_symmetry_defect": float(
            np.linalg.norm(symmetry.data) / max(np.linalg.norm(tangent.data), np.finfo(float).tiny)
        ),
        "minimum_nodal_value": float(np.min(values)),
        "maximum_nodal_value": float(np.max(values)),
        "history": history,
    }


def _rate(coarse: dict[str, Any], fine: dict[str, Any], field: str) -> float:
    return float(
        math.log(float(coarse[field]) / float(fine[field]))
        / math.log(float(coarse["h"]) / float(fine["h"]))
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    levels = [
        _solve_level(n, float(args.relative_tolerance), int(args.max_iterations))
        for n in args.subdivisions
    ]
    rates = [
        {
            "coarse_subdivisions": int(coarse["subdivisions"]),
            "fine_subdivisions": int(fine["subdivisions"]),
            "l2_rate": _rate(coarse, fine, "l2_error"),
            "h1_seminorm_rate": _rate(coarse, fine, "h1_seminorm_error"),
        }
        for coarse, fine in zip(levels[:-1], levels[1:], strict=True)
    ]
    passed = bool(
        all(level["status"] == "converged" for level in levels)
        and all(level["tangent_symmetry_defect"] <= 1.0e-12 for level in levels)
        and all(level["minimum_nodal_value"] > 1.0 / math.sqrt(3.0) for level in levels)
        and rates[-1]["l2_rate"] >= float(args.minimum_l2_rate)
        and rates[-1]["h1_seminorm_rate"] >= float(args.minimum_h1_rate)
    )
    repo_root = Path(__file__).resolve().parents[2]
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
    return {
        "schema_version": 1,
        "experiment_id": "EXP-VAL-001-GINZBURG-LANDAU-MANUFACTURED",
        "status": "passed" if passed else "failed",
        "publication_evidence": False,
        "problem": {
            "domain": "unit_square",
            "element": "P1 triangles",
            "epsilon": EPSILON,
            "exact_solution": "0.8 + 0.1 sin(pi x) sin(pi y)",
            "source": "analytic -epsilon Laplacian(u) + u(u^2-1)",
            "solve_quadrature": "production-style symmetric three-point triangle rule",
            "branch_control": "exact and computed nodal values stay above 1/sqrt(3)",
        },
        "solver_contract": {
            "method": "independently assembled damped Newton",
            "relative_residual_tolerance": float(args.relative_tolerance),
            "maximum_iterations": int(args.max_iterations),
        },
        "levels": levels,
        "rates": rates,
        "limitations": [
            "The manufactured source extends the zero-source production benchmark.",
            "The positive branch is controlled; this does not test competing nonconvex basins.",
            "The dirty-worktree result is diagnostic until rerun from a clean frozen commit.",
        ],
        "provenance": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "git": {"commit": commit, "dirty": dirty},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--minimum-l2-rate", type=float, default=1.75)
    parser.add_argument("--minimum-h1-rate", type=float, default=0.85)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.subdivisions) < 3 or any(
        fine <= coarse for coarse, fine in zip(args.subdivisions[:-1], args.subdivisions[1:], strict=True)
    ):
        raise SystemExit("provide at least three strictly increasing subdivision counts")
    payload = run(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output)
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

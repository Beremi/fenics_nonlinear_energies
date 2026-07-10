#!/usr/bin/env python3
"""Independent affine-patch verification of the production hyperelastic energy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from src.problems.hyperelasticity.jax.jax_energy import J as production_energy


jax.config.update("jax_enable_x64", True)
C1 = 0.5
D1 = 5.0


def _cube_mesh() -> tuple[np.ndarray, np.ndarray]:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    candidates = np.array(
        [
            [0, 1, 3, 7],
            [0, 3, 2, 7],
            [0, 2, 6, 7],
            [0, 6, 4, 7],
            [0, 4, 5, 7],
            [0, 5, 1, 7],
        ],
        dtype=np.int64,
    )
    elems = candidates.copy()
    for index, elem in enumerate(elems):
        matrix = np.column_stack(
            (nodes[elem[1]] - nodes[elem[0]], nodes[elem[2]] - nodes[elem[0]], nodes[elem[3]] - nodes[elem[0]])
        )
        if np.linalg.det(matrix) < 0.0:
            elems[index, [2, 3]] = elems[index, [3, 2]]
    return nodes, elems


def _element_geometry(nodes: np.ndarray, elems: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gradients = np.empty((elems.shape[0], 4, 3), dtype=np.float64)
    volumes = np.empty(elems.shape[0], dtype=np.float64)
    for index, elem in enumerate(elems):
        coords = nodes[elem]
        reference = np.column_stack(
            (coords[1] - coords[0], coords[2] - coords[0], coords[3] - coords[0])
        )
        determinant = float(np.linalg.det(reference))
        if determinant <= 0.0:
            raise ValueError("patch mesh contains a non-positive tetrahedron")
        gradients[index, 1:, :] = np.linalg.inv(reference).T.T
        gradients[index, 0, :] = -np.sum(gradients[index, 1:, :], axis=0)
        volumes[index] = determinant / 6.0
    return gradients, volumes


def _density_piola_tangent(deformation: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    determinant = float(np.linalg.det(deformation))
    if determinant <= 0.0:
        raise ValueError("patch deformation must preserve orientation")
    inverse_transpose = np.linalg.inv(deformation).T
    scalar = -2.0 * C1 + 2.0 * D1 * (determinant - 1.0) * determinant
    density = float(
        C1 * (np.sum(deformation**2) - 3.0 - 2.0 * np.log(determinant))
        + D1 * (determinant - 1.0) ** 2
    )
    piola = 2.0 * C1 * deformation + scalar * inverse_transpose
    derivative_scalar = 2.0 * D1 * (2.0 * determinant - 1.0) * determinant * inverse_transpose
    tangent = np.empty((3, 3, 3, 3), dtype=np.float64)
    for i in range(3):
        for axis in range(3):
            for k in range(3):
                for direction in range(3):
                    tangent[i, axis, k, direction] = (
                        2.0 * C1 * float(i == k and axis == direction)
                        + derivative_scalar[k, direction] * inverse_transpose[i, axis]
                        - scalar * inverse_transpose[i, direction] * inverse_transpose[k, axis]
                    )
    return density, piola, tangent.reshape((9, 9))


def _element_b_matrix(gradients: np.ndarray) -> np.ndarray:
    bmat = np.zeros((9, 12), dtype=np.float64)
    for node in range(4):
        for component in range(3):
            for axis in range(3):
                bmat[3 * component + axis, 3 * node + component] = gradients[node, axis]
    return bmat


def _independent_assembly(
    nodes: np.ndarray,
    elems: np.ndarray,
    gradients: np.ndarray,
    volumes: np.ndarray,
    deformation: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    density, piola, tangent = _density_piola_tangent(deformation)
    residual = np.zeros(3 * nodes.shape[0], dtype=np.float64)
    hessian = np.zeros((residual.size, residual.size), dtype=np.float64)
    for elem, element_gradients, volume in zip(elems, gradients, volumes, strict=True):
        bmat = _element_b_matrix(element_gradients)
        dofs = np.asarray(
            [3 * int(node) + component for node in elem for component in range(3)],
            dtype=np.int64,
        )
        residual[dofs] += float(volume) * bmat.T @ piola.reshape(-1)
        hessian[np.ix_(dofs, dofs)] += float(volume) * bmat.T @ tangent @ bmat
    return float(np.sum(volumes) * density), residual, hessian, piola


def _boundary_traction_forces(
    nodes: np.ndarray,
    elems: np.ndarray,
    piola: np.ndarray,
) -> np.ndarray:
    face_counts: dict[tuple[int, int, int], int] = {}
    for elem in elems:
        for face in (
            (elem[1], elem[2], elem[3]),
            (elem[0], elem[3], elem[2]),
            (elem[0], elem[1], elem[3]),
            (elem[0], elem[2], elem[1]),
        ):
            key = tuple(sorted(int(node) for node in face))
            face_counts[key] = face_counts.get(key, 0) + 1
    force = np.zeros((nodes.shape[0], 3), dtype=np.float64)
    center = np.mean(nodes, axis=0)
    for face, count in face_counts.items():
        if count != 1:
            continue
        a, b, c = nodes[np.asarray(face, dtype=np.int64)]
        area_vector = 0.5 * np.cross(b - a, c - a)
        centroid = (a + b + c) / 3.0
        if float(np.dot(area_vector, centroid - center)) < 0.0:
            area_vector *= -1.0
        nodal = piola @ area_vector / 3.0
        for node in face:
            force[int(node)] += nodal
    return force.reshape(-1)


def _relative_error(left: np.ndarray | float, right: np.ndarray | float) -> float:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    return float(
        np.linalg.norm(lhs - rhs)
        / max(np.linalg.norm(lhs), np.linalg.norm(rhs), np.finfo(float).tiny)
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


def run() -> dict[str, Any]:
    nodes, elems = _cube_mesh()
    gradients, volumes = _element_geometry(nodes, elems)
    deformation = np.array(
        [[1.15, 0.08, 0.0], [0.02, 0.92, 0.05], [0.0, 0.03, 1.08]],
        dtype=np.float64,
    )
    translation = np.array([0.11, -0.07, 0.04], dtype=np.float64)
    current = nodes @ deformation.T + translation
    state = current.reshape(-1)
    all_dofs = np.arange(state.size, dtype=np.int64)
    zero = np.zeros_like(state)

    def energy(values: jnp.ndarray) -> jnp.ndarray:
        return production_energy(
            values,
            jnp.asarray(zero),
            jnp.asarray(all_dofs),
            jnp.asarray(elems),
            jnp.asarray(gradients[:, :, 0]),
            jnp.asarray(gradients[:, :, 1]),
            jnp.asarray(gradients[:, :, 2]),
            jnp.asarray(volumes),
            C1,
            D1,
        )

    state_jax = jnp.asarray(state, dtype=jnp.float64)
    production_value = float(energy(state_jax))
    production_gradient = np.asarray(jax.grad(energy)(state_jax), dtype=np.float64)
    production_hessian = np.asarray(jax.hessian(energy)(state_jax), dtype=np.float64)
    independent_value, independent_residual, independent_hessian, piola = _independent_assembly(
        nodes, elems, gradients, volumes, deformation
    )
    traction_force = _boundary_traction_forces(nodes, elems, piola)

    angle = 0.47
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotated_density, rotated_piola, _ = _density_piola_tangent(rotation @ deformation)
    density, _, _ = _density_piola_tangent(deformation)
    translation_modes = []
    for component in range(3):
        mode = np.zeros_like(state)
        mode[component::3] = 1.0
        translation_modes.append(
            float(np.linalg.norm(production_hessian @ mode) / np.linalg.norm(mode))
        )

    metrics = {
        "energy_relative_error": _relative_error(production_value, independent_value),
        "residual_relative_error": _relative_error(production_gradient, independent_residual),
        "hessian_relative_error": _relative_error(production_hessian, independent_hessian),
        "hessian_symmetry_defect": _relative_error(production_hessian, production_hessian.T),
        "traction_balance_relative_error": _relative_error(independent_residual, traction_force),
        "net_internal_force_norm": float(np.linalg.norm(independent_residual.reshape((-1, 3)).sum(axis=0))),
        "objectivity_energy_relative_error": _relative_error(rotated_density, density),
        "piola_rotation_covariance_relative_error": _relative_error(rotated_piola, rotation @ piola),
        "translation_mode_hessian_action_norms": translation_modes,
    }
    tolerance = 2.0e-11
    passed = bool(
        all(
            float(metrics[field]) <= tolerance
            for field in (
                "energy_relative_error",
                "residual_relative_error",
                "hessian_relative_error",
                "hessian_symmetry_defect",
                "traction_balance_relative_error",
                "objectivity_energy_relative_error",
                "piola_rotation_covariance_relative_error",
            )
        )
        and float(metrics["net_internal_force_norm"]) <= tolerance
        and max(translation_modes) <= tolerance
    )
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": 1,
        "experiment_id": "EXP-VAL-001-HYPERELASTIC-AFFINE-PATCH",
        "status": "passed" if passed else "failed",
        "publication_evidence": False,
        "case": {
            "mesh": "unit cube split into six P1 tetrahedra",
            "deformation_gradient": deformation.tolist(),
            "determinant": float(np.linalg.det(deformation)),
            "translation": translation.tolist(),
            "C1": C1,
            "D1": D1,
            "volume": float(np.sum(volumes)),
        },
        "contract": {
            "relative_tolerance": tolerance,
            "independent_reference": "analytic Piola/tangent and boundary-traction assembly",
        },
        "metrics": metrics,
        "limitations": [
            "The affine patch is exactly representable and does not establish spatial convergence for nonaffine deformation.",
            "The production kernel uses absolute determinant, but this test remains strictly orientation preserving.",
            "The dirty-worktree result is diagnostic until rerun from a clean frozen commit.",
        ],
        "provenance": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "git": _git_metadata(repo_root),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output)
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

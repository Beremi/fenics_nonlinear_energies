#!/usr/bin/env python3
"""Verify smooth element derivatives against independent contractions and FD."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from src.problems.ginzburg_landau.jax_petsc.reordered_element_assembler import (
    _gl_integrand,
)
from src.problems.hyperelasticity.jax_petsc.parallel_hessian_dof import (
    _he_energy_density,
)
from src.problems.plaplace.jax_petsc.reordered_element_assembler import (
    _plaplace_integrand,
)


jax.config.update("jax_enable_x64", True)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _relative_error(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), np.finfo(float).tiny)
    return float(np.linalg.norm(left - right) / scale)


def _symmetry_defect(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    return float(np.linalg.norm(matrix - matrix.T) / max(np.linalg.norm(matrix), np.finfo(float).tiny))


def _slopes(step_sizes: np.ndarray, errors: list[float]) -> list[float | None]:
    out: list[float | None] = []
    for left_h, right_h, left_e, right_e in zip(
        step_sizes[:-1], step_sizes[1:], errors[:-1], errors[1:]
    ):
        if left_e <= 0.0 or right_e <= 0.0:
            out.append(None)
        else:
            out.append(float(np.log(left_e / right_e) / np.log(left_h / right_h)))
    return out


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _plaplace_case() -> tuple[Callable, Callable, np.ndarray, dict[str, object]]:
    dvx = np.array([-1.0, 1.0, 0.0], dtype=np.float64)
    dvy = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    volume = 0.5
    exponent = 3.0
    state = np.array([0.15, 0.9, -0.2], dtype=np.float64)

    def energy(v: jnp.ndarray) -> jnp.ndarray:
        return _plaplace_integrand(v, dvx, dvy, exponent) * volume

    def independent(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        bmat = np.vstack((dvx, dvy))
        q = bmat @ np.asarray(v, dtype=np.float64)
        radius = float(np.linalg.norm(q))
        stress = radius ** (exponent - 2.0) * q
        tangent = radius ** (exponent - 2.0) * np.eye(2)
        tangent += (exponent - 2.0) * radius ** (exponent - 4.0) * np.outer(q, q)
        return (
            volume * bmat.T @ stress,
            volume * bmat.T @ tangent @ bmat,
            {"gradient_magnitude": radius, "exponent": exponent},
        )

    return energy, independent, state, {"problem": "p-laplace", "state": "regular"}


def _gl_case(*, indefinite: bool) -> tuple[Callable, Callable, np.ndarray, dict[str, object]]:
    dvx = np.array([-1.0, 1.0, 0.0], dtype=np.float64)
    dvy = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    interpolation = np.array(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ],
        dtype=np.float64,
    )
    weights = np.full(3, 1.0 / 3.0, dtype=np.float64)
    volume = 0.5
    epsilon = 0.04
    state = (
        np.array([0.02, -0.01, 0.015], dtype=np.float64)
        if indefinite
        else np.array([0.8, 1.05, 0.9], dtype=np.float64)
    )

    def energy(v: jnp.ndarray) -> jnp.ndarray:
        return _gl_integrand(v, dvx, dvy, interpolation, weights, epsilon) * volume

    def independent(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        values = np.asarray(v, dtype=np.float64)
        bmat = np.vstack((dvx, dvy))
        quadrature_values = values @ interpolation
        gradient = volume * (
            epsilon * bmat.T @ (bmat @ values)
            + interpolation
            @ (weights * quadrature_values * (quadrature_values**2 - 1.0))
        )
        hessian = volume * (
            epsilon * bmat.T @ bmat
            + (interpolation * (weights * (3.0 * quadrature_values**2 - 1.0)))
            @ interpolation.T
        )
        return gradient, hessian, {"minimum_hessian_eigenvalue": float(np.linalg.eigvalsh(hessian)[0])}

    label = "indefinite" if indefinite else "regular"
    return energy, independent, state, {"problem": "ginzburg-landau", "state": label}


def _deformation_gradient_matrix(
    dphix: np.ndarray,
    dphiy: np.ndarray,
    dphiz: np.ndarray,
) -> np.ndarray:
    gradients = np.column_stack((dphix, dphiy, dphiz))
    bmat = np.zeros((9, 3 * gradients.shape[0]), dtype=np.float64)
    for node, grad in enumerate(gradients):
        for component in range(3):
            for axis in range(3):
                bmat[3 * component + axis, 3 * node + component] = grad[axis]
    return bmat


def _he_case(*, scale: float) -> tuple[Callable, Callable, np.ndarray, dict[str, object]]:
    nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dphix = np.array([-1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    dphiy = np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    dphiz = np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    volume = np.asarray([1.0 / 6.0], dtype=np.float64)
    c1 = 0.5
    d1 = 5.0
    perturbation = np.array(
        [[0.0, 0.0, 0.0], [0.4, 0.1, -0.1], [0.05, -0.2, 0.1], [-0.1, 0.15, 0.3]],
        dtype=np.float64,
    )
    state = (nodes + float(scale) * perturbation).reshape(-1)
    bmat = _deformation_gradient_matrix(dphix, dphiy, dphiz)

    def energy(v: jnp.ndarray) -> jnp.ndarray:
        density = _he_energy_density(v, dphix, dphiy, dphiz, c1, d1, False)
        return jnp.sum(density * volume)

    def density_flat(flat_f: jnp.ndarray) -> jnp.ndarray:
        matrix = flat_f.reshape((3, 3))
        determinant = jnp.linalg.det(matrix)
        return c1 * (jnp.sum(matrix**2) - 3.0 - 2.0 * jnp.log(determinant)) + d1 * (
            determinant - 1.0
        ) ** 2

    density_grad = jax.grad(density_flat)
    density_hessian = jax.hessian(density_flat)

    def independent(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        values = np.asarray(v, dtype=np.float64)
        flat_f = bmat @ values
        matrix = flat_f.reshape((3, 3))
        determinant = float(np.linalg.det(matrix))
        inverse_transpose = np.linalg.inv(matrix).T
        piola_analytic = (
            2.0 * c1 * matrix
            - 2.0 * c1 * inverse_transpose
            + 2.0 * d1 * (determinant - 1.0) * determinant * inverse_transpose
        ).reshape(-1)
        piola_ad = np.asarray(density_grad(jnp.asarray(flat_f)), dtype=np.float64)
        constitutive_tangent = np.asarray(
            density_hessian(jnp.asarray(flat_f)), dtype=np.float64
        )
        return (
            float(volume[0]) * bmat.T @ piola_analytic,
            float(volume[0]) * bmat.T @ constitutive_tangent @ bmat,
            {
                "determinant": determinant,
                "analytic_vs_constitutive_ad_stress_error": _relative_error(
                    piola_analytic, piola_ad
                ),
            },
        )

    return energy, independent, state, {"problem": "hyperelasticity", "state": f"scale-{scale:g}"}


def _verify_case(
    name: str,
    energy: Callable,
    independent: Callable,
    state: np.ndarray,
    metadata: dict[str, object],
    *,
    seed: int,
    step_sizes: np.ndarray,
    fd_step_sizes: np.ndarray,
    fd_gate_index: int,
) -> dict[str, object]:
    state = np.asarray(state, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    direction = rng.standard_normal(state.size)
    direction /= np.linalg.norm(direction)
    energy_grad = jax.grad(energy)
    energy_hessian = jax.hessian(energy)
    state_jax = jnp.asarray(state, dtype=jnp.float64)
    value = float(energy(state_jax))
    gradient_ad = np.asarray(energy_grad(state_jax), dtype=np.float64)
    hessian_ad = np.asarray(energy_hessian(state_jax), dtype=np.float64)
    gradient_reference, hessian_reference, diagnostics = independent(state)

    energy_remainders: list[float] = []
    gradient_remainders: list[float] = []
    for step in step_sizes:
        trial = jnp.asarray(state + float(step) * direction, dtype=jnp.float64)
        trial_value = float(energy(trial))
        trial_gradient = np.asarray(energy_grad(trial), dtype=np.float64)
        energy_remainders.append(
            abs(trial_value - value - float(step) * float(gradient_ad @ direction))
        )
        gradient_remainders.append(
            float(
                np.linalg.norm(
                    trial_gradient
                    - gradient_ad
                    - float(step) * (hessian_ad @ direction)
                )
            )
        )

    fd_gradient_errors: list[float] = []
    fd_hvp_errors: list[float] = []
    exact_directional = float(gradient_ad @ direction)
    exact_hvp = hessian_ad @ direction
    for step in fd_step_sizes:
        plus = jnp.asarray(state + float(step) * direction, dtype=jnp.float64)
        minus = jnp.asarray(state - float(step) * direction, dtype=jnp.float64)
        fd_directional = (float(energy(plus)) - float(energy(minus))) / (2.0 * float(step))
        fd_hvp = (
            np.asarray(energy_grad(plus), dtype=np.float64)
            - np.asarray(energy_grad(minus), dtype=np.float64)
        ) / (2.0 * float(step))
        fd_gradient_errors.append(
            float(
                abs(fd_directional - exact_directional)
                / max(abs(fd_directional), abs(exact_directional), np.linalg.norm(gradient_ad), np.finfo(float).tiny)
            )
        )
        fd_hvp_errors.append(_relative_error(fd_hvp, exact_hvp))

    return {
        "case": name,
        **metadata,
        "degrees_of_freedom": int(state.size),
        "seed": int(seed),
        "state": [float(value) for value in state],
        "direction": [float(value) for value in direction],
        "energy": value,
        "independent_diagnostics": diagnostics,
        "gradient_relative_error": _relative_error(gradient_ad, gradient_reference),
        "hessian_relative_error": _relative_error(hessian_ad, hessian_reference),
        "hessian_symmetry_defect": _symmetry_defect(hessian_ad),
        "step_sizes": [float(value) for value in step_sizes],
        "energy_taylor_remainders": energy_remainders,
        "gradient_taylor_remainders": gradient_remainders,
        "energy_taylor_slopes": _slopes(step_sizes, energy_remainders),
        "gradient_taylor_slopes": _slopes(step_sizes, gradient_remainders),
        "fd_step_sizes": [float(value) for value in fd_step_sizes],
        "fd_gate_index": int(fd_gate_index),
        "fd_gradient_relative_errors": fd_gradient_errors,
        "fd_hvp_relative_errors": fd_hvp_errors,
        "fd_gradient_error_at_gate": float(fd_gradient_errors[int(fd_gate_index)]),
        "fd_hvp_error_at_gate": float(fd_hvp_errors[int(fd_gate_index)]),
        "finite": bool(
            np.isfinite(value)
            and np.all(np.isfinite(gradient_ad))
            and np.all(np.isfinite(hessian_ad))
            and np.all(np.isfinite(gradient_reference))
            and np.all(np.isfinite(hessian_reference))
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    step_sizes = np.asarray(
        [float(part) for part in str(args.step_sizes).split(",") if part.strip()],
        dtype=np.float64,
    )
    fd_step_sizes = np.asarray(
        [float(part) for part in str(args.fd_step_sizes).split(",") if part.strip()],
        dtype=np.float64,
    )
    if step_sizes.size < 3 or np.any(step_sizes <= 0.0):
        raise ValueError("step_sizes must contain at least three positive values")
    if fd_step_sizes.size < 3 or np.any(fd_step_sizes <= 0.0):
        raise ValueError("fd_step_sizes must contain at least three positive values")
    if not 0 <= int(args.fd_gate_index) < int(fd_step_sizes.size):
        raise ValueError("fd_gate_index must select one fd_step_sizes entry")

    cases = [
        ("plaplace_regular", *_plaplace_case()),
        ("ginzburg_landau_regular", *_gl_case(indefinite=False)),
        ("ginzburg_landau_indefinite", *_gl_case(indefinite=True)),
        ("hyperelasticity_near_identity", *_he_case(scale=0.02)),
        ("hyperelasticity_intermediate", *_he_case(scale=0.08)),
    ]
    records = [
        _verify_case(
            name,
            energy,
            independent,
            state,
            metadata,
            seed=int(args.seed) + idx,
            step_sizes=step_sizes,
            fd_step_sizes=fd_step_sizes,
            fd_gate_index=int(args.fd_gate_index),
        )
        for idx, (name, energy, independent, state, metadata) in enumerate(cases)
    ]
    maxima = {
        "maximum_gradient_relative_error": max(float(row["gradient_relative_error"]) for row in records),
        "maximum_hessian_relative_error": max(float(row["hessian_relative_error"]) for row in records),
        "maximum_hessian_symmetry_defect": max(float(row["hessian_symmetry_defect"]) for row in records),
        "maximum_fd_gradient_error_at_gate": max(float(row["fd_gradient_error_at_gate"]) for row in records),
        "maximum_fd_hvp_error_at_gate": max(float(row["fd_hvp_error_at_gate"]) for row in records),
    }
    passed = bool(
        all(bool(row["finite"]) for row in records)
        and maxima["maximum_gradient_relative_error"] <= float(args.route_tolerance)
        and maxima["maximum_hessian_relative_error"] <= float(args.route_tolerance)
        and maxima["maximum_hessian_symmetry_defect"] <= float(args.symmetry_tolerance)
        and maxima["maximum_fd_gradient_error_at_gate"] <= float(args.fd_tolerance)
        and maxima["maximum_fd_hvp_error_at_gate"] <= float(args.fd_tolerance)
    )
    return {
        "experiment_id": "EXP-DERIV-001-SMOOTH-FIXED-ELEMENT",
        "status": "passed" if passed else "failed",
        "contract": {
            "route_relative_tolerance": float(args.route_tolerance),
            "symmetry_tolerance": float(args.symmetry_tolerance),
            "centered_fd_tolerance": float(args.fd_tolerance),
            "centered_fd_gate_index": int(args.fd_gate_index),
            "centered_fd_gate_step": float(fd_step_sizes[int(args.fd_gate_index)]),
        },
        "summary": {"cases": len(records), **maxima},
        "records": records,
        "provenance": {
            "git_commit": _git_commit(),
            "git_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT)
            ),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "jax_enable_x64": bool(jax.config.x64_enabled),
            "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
            "command": " ".join(sys.argv),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--step-sizes", default="1e-2,3e-3,1e-3,3e-4,1e-4,3e-5")
    parser.add_argument("--fd-step-sizes", default="1e-3,3e-4,1e-4,3e-5,1e-5")
    parser.add_argument("--fd-gate-index", type=int, default=3)
    parser.add_argument("--route-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--symmetry-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--fd-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = run(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

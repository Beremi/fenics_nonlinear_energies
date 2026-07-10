#!/usr/bin/env python3
"""Run deterministic material-point checks for the Plasticity3D potential.

This runner deliberately avoids meshes, DOLFINx, PETSc, and MPI.  It calls the
production scalar JAX kernel and differentiates that scalar with JAX.  Branch
predicates and tensor rotations are evaluated separately in NumPy.  The NumPy
branch formulas are a transcription of the production/published algebra, not
an independent constitutive model or a proof of generalized differentiability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from src.core.benchmark.run_record import (
    RUN_RECORD_SCHEMA_ID,
    RUN_RECORD_SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_run_record,
    check_experiment_preflight,
    sha256_file,
    utc_now_iso,
)
from src.problems.slope_stability_3d.jax.jax_energy_3d import (
    mc_potential_density_3d,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BRANCHES = ("elastic", "shear", "left_edge", "right_edge", "apex")
ADJACENT_INTERFACES = (
    ("elastic", "shear"),
    ("shear", "left_edge"),
    ("shear", "right_edge"),
    ("left_edge", "apex"),
    ("right_edge", "apex"),
)
TIE_BREAK_TINY = 1.0e-15


@dataclass(frozen=True)
class Material:
    c_bar: float = 1.0
    sin_phi: float = 0.5
    shear: float = 10.0
    bulk: float = 20.0

    @property
    def lame(self) -> float:
        return self.bulk - 2.0 * self.shear / 3.0

    def as_json(self) -> dict[str, float]:
        return {
            "c_bar": float(self.c_bar),
            "sin_phi": float(self.sin_phi),
            "shear": float(self.shear),
            "bulk": float(self.bulk),
            "lame": float(self.lame),
        }


# Deterministic principal-strain anchors.  They were selected by a seeded
# dimensionless search and are frozen here so the campaign does not silently
# change when search settings change.  Every row is reclassified and subjected
# to a positive active-margin gate at runtime.
PRINCIPAL_ANCHORS: dict[str, np.ndarray] = {
    "elastic": np.array([-0.13631082, -0.16680545, -0.19711281]),
    "shear": np.array([0.05755777, -0.03796704, -0.06961258]),
    "left_edge": np.array([0.06851800, 0.05688190, -0.06930851]),
    "right_edge": np.array([0.39253952, -0.13488312, -0.19385268]),
    "apex": np.array([0.37517304, 0.31871759, 0.14611402]),
}


DEGENERACY_CASES: dict[str, np.ndarray] = {
    "exact_hydrostatic_elastic": np.array([-0.01, -0.01, -0.01]),
    "exact_hydrostatic_apex": np.array([0.10, 0.10, 0.10]),
    "exact_repeated_largest": np.array([0.10, 0.10, -0.05]),
    "exact_repeated_smallest": np.array([0.20, -0.05, -0.05]),
    "near_repeated_largest": np.array([0.10, 0.10 - 1.0e-10, -0.05]),
    "near_repeated_smallest": np.array([0.20, -0.05 + 1.0e-10, -0.05]),
    "near_hydrostatic": np.array([0.10 + 2.0e-10, 0.10 + 1.0e-10, 0.10]),
}


def _strain6_to_tensor(eps6: np.ndarray) -> np.ndarray:
    """Map engineering strain components to a symmetric tensor in NumPy."""

    e11, e22, e33, g12, g23, g13 = np.asarray(eps6, dtype=np.float64)
    return np.array(
        [
            [e11, 0.5 * g12, 0.5 * g13],
            [0.5 * g12, e22, 0.5 * g23],
            [0.5 * g13, 0.5 * g23, e33],
        ],
        dtype=np.float64,
    )


def _tensor_to_strain6(tensor: np.ndarray) -> np.ndarray:
    """Map a symmetric tensor to engineering strain components in NumPy."""

    value = np.asarray(tensor, dtype=np.float64)
    return np.array(
        [
            value[0, 0],
            value[1, 1],
            value[2, 2],
            2.0 * value[0, 1],
            2.0 * value[1, 2],
            2.0 * value[0, 2],
        ],
        dtype=np.float64,
    )


def _gradient6_to_stress(gradient: np.ndarray) -> np.ndarray:
    """Map the engineering-coordinate energy gradient to a stress tensor."""

    g11, g22, g33, g12, g23, g13 = np.asarray(gradient, dtype=np.float64)
    return np.array(
        [[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]],
        dtype=np.float64,
    )


def _safe_signed_denom(value: float, tiny: float) -> float:
    sign = 1.0 if value >= 0.0 else -1.0
    return sign * max(abs(float(value)), float(tiny))


def _branch_diagnostics(
    eps6: np.ndarray,
    material: Material,
    *,
    tiny: float = TIE_BREAK_TINY,
) -> dict[str, object]:
    """Replay branch predicates in NumPy and return scale-aware diagnostics.

    This implementation is kept outside the JAX kernel to detect wiring and
    branch-selection mistakes.  It remains an algebraic transcription of the
    same model; it is not an independently derived return-mapping reference.
    """

    eps = np.asarray(eps6, dtype=np.float64)
    tensor = _strain6_to_tensor(eps)
    raw_eigvals = np.linalg.eigvalsh(tensor)
    eigvals = np.linalg.eigvalsh(tensor + float(tiny) * np.diag([0.0, 1.0, 2.0]))
    eig_3, eig_2, eig_1 = (float(value) for value in eigvals)
    invariant_1 = float(np.trace(tensor))
    c_bar = float(material.c_bar)
    sin_phi = float(material.sin_phi)
    shear = float(material.shear)
    bulk = float(material.bulk)
    lame = float(material.lame)

    f_tr = (
        2.0 * shear * ((1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * eig_3)
        + 2.0 * lame * sin_phi * invariant_1
        - c_bar
    )
    gamma_sl = (eig_1 - eig_2) / max(tiny, 1.0 + sin_phi)
    gamma_sr = (eig_2 - eig_3) / max(tiny, 1.0 - sin_phi)
    gamma_la = (eig_1 + eig_2 - 2.0 * eig_3) / max(tiny, 3.0 - sin_phi)
    gamma_ra = (2.0 * eig_1 - eig_2 - eig_3) / max(tiny, 3.0 + sin_phi)

    denom_s = 4.0 * lame * sin_phi**2 + 4.0 * shear * (1.0 + sin_phi**2)
    denom_l = (
        4.0 * lame * sin_phi**2
        + shear * (1.0 + sin_phi) ** 2
        + 2.0 * shear * (1.0 - sin_phi) ** 2
    )
    denom_r = (
        4.0 * lame * sin_phi**2
        + 2.0 * shear * (1.0 + sin_phi) ** 2
        + shear * (1.0 - sin_phi) ** 2
    )
    denom_a = 4.0 * bulk * sin_phi**2
    lambda_s = f_tr / _safe_signed_denom(denom_s, tiny)
    lambda_l = (
        shear
        * ((1.0 + sin_phi) * (eig_1 + eig_2) - 2.0 * (1.0 - sin_phi) * eig_3)
        + 2.0 * lame * sin_phi * invariant_1
        - c_bar
    ) / _safe_signed_denom(denom_l, tiny)
    lambda_r = (
        shear
        * (2.0 * (1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * (eig_2 + eig_3))
        + 2.0 * lame * sin_phi * invariant_1
        - c_bar
    ) / _safe_signed_denom(denom_r, tiny)
    lambda_a = (2.0 * bulk * sin_phi * invariant_1 - c_bar) / _safe_signed_denom(
        denom_a, tiny
    )

    if f_tr <= 0.0:
        branch = "elastic"
    elif lambda_s <= min(gamma_sl, gamma_sr):
        branch = "shear"
    elif gamma_sl < gamma_sr and gamma_sl <= lambda_l <= gamma_la:
        branch = "left_edge"
    elif gamma_sl > gamma_sr and gamma_sr <= lambda_r <= gamma_ra:
        branch = "right_edge"
    else:
        branch = "apex"

    stress_scale = max(
        abs(c_bar),
        abs(f_tr),
        2.0 * abs(shear) * max(abs(eig_1), abs(eig_3)),
        2.0 * abs(lame * sin_phi * invariant_1),
        np.finfo(float).tiny,
    )
    modulus_scale = max(abs(shear), abs(bulk), abs(lame), np.finfo(float).tiny)
    strain_scale = max(
        abs(eig_1),
        abs(eig_2),
        abs(eig_3),
        abs(lambda_s),
        abs(lambda_l),
        abs(lambda_r),
        abs(lambda_a),
        abs(gamma_sl),
        abs(gamma_sr),
        abs(gamma_la),
        abs(gamma_ra),
        abs(c_bar) / modulus_scale,
        np.finfo(float).tiny,
    )
    normalized = {
        "yield": float(f_tr / stress_scale),
        "shear_exclusion": float((lambda_s - min(gamma_sl, gamma_sr)) / strain_scale),
        "edge_order": float((gamma_sl - gamma_sr) / strain_scale),
        "left_lower": float((lambda_l - gamma_sl) / strain_scale),
        "left_upper": float((gamma_la - lambda_l) / strain_scale),
        "right_lower": float((lambda_r - gamma_sr) / strain_scale),
        "right_upper": float((gamma_ra - lambda_r) / strain_scale),
        "left_apex": float((lambda_l - gamma_la) / strain_scale),
        "right_apex": float((lambda_r - gamma_ra) / strain_scale),
    }
    if branch == "elastic":
        active_margin = -normalized["yield"]
    elif branch == "shear":
        active_margin = min(normalized["yield"], -normalized["shear_exclusion"])
    elif branch == "left_edge":
        active_margin = min(
            normalized["yield"],
            normalized["shear_exclusion"],
            -normalized["edge_order"],
            normalized["left_lower"],
            normalized["left_upper"],
        )
    elif branch == "right_edge":
        active_margin = min(
            normalized["yield"],
            normalized["shear_exclusion"],
            normalized["edge_order"],
            normalized["right_lower"],
            normalized["right_upper"],
        )
    elif normalized["edge_order"] < 0.0:
        active_margin = min(
            normalized["yield"],
            normalized["shear_exclusion"],
            -normalized["edge_order"],
            normalized["left_apex"],
        )
    elif normalized["edge_order"] > 0.0:
        active_margin = min(
            normalized["yield"],
            normalized["shear_exclusion"],
            normalized["edge_order"],
            normalized["right_apex"],
        )
    else:
        active_margin = 0.0

    e11, e22, e33, g12, g23, g13 = (float(value) for value in eps)
    elastic_quadratic = (
        e11**2
        + e22**2
        + e33**2
        + 0.5 * (g12**2 + g23**2 + g13**2)
    )
    psi_el = 0.5 * lame * invariant_1**2 + shear * elastic_quadratic
    psi_s = psi_el - 0.5 * denom_s * lambda_s**2
    psi_l = (
        0.5 * lame * invariant_1**2
        + shear * (eig_3**2 + 0.5 * (eig_1 + eig_2) ** 2)
        - 0.5 * denom_l * lambda_l**2
    )
    psi_r = (
        0.5 * lame * invariant_1**2
        + shear * (eig_1**2 + 0.5 * (eig_2 + eig_3) ** 2)
        - 0.5 * denom_r * lambda_r**2
    )
    psi_a = 0.5 * bulk * invariant_1**2 - 0.5 * denom_a * lambda_a**2
    branch_energies = {
        "elastic": psi_el,
        "shear": psi_s,
        "left_edge": psi_l,
        "right_edge": psi_r,
        "apex": psi_a,
    }

    raw_gaps = np.diff(raw_eigvals)
    perturbed_gaps = np.diff(eigvals)
    eigen_scale = max(float(np.max(np.abs(raw_eigvals))), tiny)
    tensor_norm = float(np.linalg.norm(tensor))
    return {
        "branch": branch,
        "principal_values_descending_with_tie_break": [eig_1, eig_2, eig_3],
        "raw_principal_values_ascending": [float(value) for value in raw_eigvals],
        "invariant_1": invariant_1,
        "predicate_values": {
            "trial_yield": float(f_tr),
            "gamma_sl": float(gamma_sl),
            "gamma_sr": float(gamma_sr),
            "gamma_la": float(gamma_la),
            "gamma_ra": float(gamma_ra),
            "lambda_s": float(lambda_s),
            "lambda_l": float(lambda_l),
            "lambda_r": float(lambda_r),
            "lambda_a": float(lambda_a),
        },
        "normalized_predicate_coordinates": normalized,
        "normalized_active_branch_margin": float(active_margin),
        "minimum_raw_principal_gap": float(np.min(raw_gaps)),
        "minimum_normalized_raw_principal_gap": float(np.min(raw_gaps) / eigen_scale),
        "minimum_tie_broken_principal_gap": float(np.min(perturbed_gaps)),
        "minimum_normalized_denominator": float(
            min(abs(denom_s), abs(denom_l), abs(denom_r), abs(denom_a))
            / modulus_scale
        ),
        "relative_tie_break_scale": float(
            tiny * np.sqrt(5.0) / max(tensor_norm, tiny)
        ),
        "selected_numpy_energy_transcription": float(branch_energies[branch]),
        "reference_scope": (
            "NumPy predicate and selected-energy algebra transcription; not an "
            "independent constitutive reference"
        ),
    }


def _relative_error(left: np.ndarray | float, right: np.ndarray | float) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    scale = max(
        float(np.linalg.norm(left_array)),
        float(np.linalg.norm(right_array)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left_array - right_array) / scale)


def _scaled_error(
    left: np.ndarray | float,
    right: np.ndarray | float,
    *,
    scale_floor: float,
) -> tuple[float, float]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    absolute = float(np.linalg.norm(left_array - right_array))
    scale = max(
        float(np.linalg.norm(left_array)),
        float(np.linalg.norm(right_array)),
        float(scale_floor),
    )
    return absolute, float(absolute / scale)


def _evaluate(
    eps6: np.ndarray,
    energy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    gradient_fn: Callable[[jnp.ndarray], jnp.ndarray],
    hessian_fn: Callable[[jnp.ndarray], jnp.ndarray],
    material: Material,
) -> dict[str, object]:
    eps = np.asarray(eps6, dtype=np.float64)
    value = jnp.asarray(eps, dtype=jnp.float64)
    energy = float(energy_fn(value))
    gradient = np.asarray(gradient_fn(value), dtype=np.float64)
    hessian = np.asarray(hessian_fn(value), dtype=np.float64)
    diagnostics = _branch_diagnostics(eps, material)
    finite = bool(
        np.isfinite(energy)
        and np.all(np.isfinite(gradient))
        and np.all(np.isfinite(hessian))
    )
    symmetry = float(
        np.linalg.norm(hessian - hessian.T)
        / max(np.linalg.norm(hessian), np.finfo(float).tiny)
    )
    return {
        "strain6": [float(value) for value in eps],
        "energy": energy,
        "gradient": gradient,
        "hessian": hessian,
        "gradient_norm": float(np.linalg.norm(gradient)),
        "hessian_frobenius_norm": float(np.linalg.norm(hessian)),
        "hessian_symmetry_defect": symmetry,
        "finite_energy_gradient_hessian": finite,
        "numpy_selected_energy_relative_error": _relative_error(
            energy, float(diagnostics["selected_numpy_energy_transcription"])
        ),
        "branch_diagnostics": diagnostics,
    }


def _without_arrays(record: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"gradient", "hessian"}
    }


def _directional_check(
    eps6: np.ndarray,
    evaluation: dict[str, object],
    *,
    seed: int,
    steps: tuple[float, ...],
    gate_index: int,
    energy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    gradient_fn: Callable[[jnp.ndarray], jnp.ndarray],
    material: Material,
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed))
    direction = rng.standard_normal(6)
    direction /= np.linalg.norm(direction)
    center = np.asarray(eps6, dtype=np.float64)
    gradient = np.asarray(evaluation["gradient"], dtype=np.float64)
    hessian = np.asarray(evaluation["hessian"], dtype=np.float64)
    exact_directional = float(np.dot(gradient, direction))
    exact_hvp = hessian @ direction
    energy_errors: list[float] = []
    hvp_errors: list[float] = []
    labels_plus: list[str] = []
    labels_minus: list[str] = []
    for step in steps:
        plus = center + float(step) * direction
        minus = center - float(step) * direction
        energy_plus = float(energy_fn(jnp.asarray(plus, dtype=jnp.float64)))
        energy_minus = float(energy_fn(jnp.asarray(minus, dtype=jnp.float64)))
        gradient_plus = np.asarray(
            gradient_fn(jnp.asarray(plus, dtype=jnp.float64)), dtype=np.float64
        )
        gradient_minus = np.asarray(
            gradient_fn(jnp.asarray(minus, dtype=jnp.float64)), dtype=np.float64
        )
        centered_directional = (energy_plus - energy_minus) / (2.0 * float(step))
        centered_hvp = (gradient_plus - gradient_minus) / (2.0 * float(step))
        energy_errors.append(
            _scaled_error(
                centered_directional,
                exact_directional,
                # The dimensionless material normalization makes one the
                # natural absolute floor.  This also avoids reporting an
                # infinite/meaningless relative error on the affine apex
                # branch, whose exact Hessian action is zero.
                scale_floor=1.0,
            )[1]
        )
        hvp_errors.append(
            _scaled_error(centered_hvp, exact_hvp, scale_floor=1.0)[1]
        )
        labels_plus.append(str(_branch_diagnostics(plus, material)["branch"]))
        labels_minus.append(str(_branch_diagnostics(minus, material)["branch"]))
    center_label = str(evaluation["branch_diagnostics"]["branch"])
    gate = int(gate_index)
    return {
        "direction_seed": int(seed),
        "direction": [float(value) for value in direction],
        "steps": [float(value) for value in steps],
        "gate_index": gate,
        "gate_step": float(steps[gate]),
        "centered_energy_directional_scaled_errors": energy_errors,
        "centered_hvp_scaled_errors": hvp_errors,
        "energy_error_at_gate": float(energy_errors[gate]),
        "hvp_error_at_gate": float(hvp_errors[gate]),
        "plus_branch_labels": labels_plus,
        "minus_branch_labels": labels_minus,
        "branch_stable_for_all_steps": bool(
            all(label == center_label for label in labels_plus + labels_minus)
        ),
        "scope": "fixed-branch centered differences only",
    }


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    candidate = rng.standard_normal((3, 3))
    orthogonal, triangular = np.linalg.qr(candidate)
    signs = np.sign(np.diag(triangular))
    signs[signs == 0.0] = 1.0
    orthogonal = orthogonal @ np.diag(signs)
    if np.linalg.det(orthogonal) < 0.0:
        orthogonal[:, -1] *= -1.0
    return np.asarray(orthogonal, dtype=np.float64)


def _rotation_checks(
    branch: str,
    principal_values: np.ndarray,
    *,
    rotations: int,
    seed: int,
    energy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    gradient_fn: Callable[[jnp.ndarray], jnp.ndarray],
    hessian_fn: Callable[[jnp.ndarray], jnp.ndarray],
    material: Material,
) -> list[dict[str, object]]:
    base_tensor = np.diag(np.asarray(principal_values, dtype=np.float64))
    base_eps = _tensor_to_strain6(base_tensor)
    base = _evaluate(base_eps, energy_fn, gradient_fn, hessian_fn, material)
    base_gradient = np.asarray(base["gradient"], dtype=np.float64)
    base_hessian = np.asarray(base["hessian"], dtype=np.float64)
    base_stress = _gradient6_to_stress(base_gradient)
    rng = np.random.default_rng(int(seed))
    records: list[dict[str, object]] = []
    for rotation_index in range(int(rotations)):
        rotation = _random_rotation(rng)
        direction_tensor = rng.standard_normal((3, 3))
        direction_tensor = 0.5 * (direction_tensor + direction_tensor.T)
        direction_tensor /= np.linalg.norm(direction_tensor)
        direction = _tensor_to_strain6(direction_tensor)
        rotated_tensor = rotation @ base_tensor @ rotation.T
        rotated_eps = _tensor_to_strain6(rotated_tensor)
        rotated = _evaluate(
            rotated_eps, energy_fn, gradient_fn, hessian_fn, material
        )
        rotated_gradient = np.asarray(rotated["gradient"], dtype=np.float64)
        rotated_hessian = np.asarray(rotated["hessian"], dtype=np.float64)
        expected_stress = rotation @ base_stress @ rotation.T
        actual_stress = _gradient6_to_stress(rotated_gradient)
        rotated_direction_tensor = rotation @ direction_tensor @ rotation.T
        rotated_direction = _tensor_to_strain6(rotated_direction_tensor)
        expected_action = rotation @ _gradient6_to_stress(base_hessian @ direction) @ rotation.T
        actual_action = _gradient6_to_stress(rotated_hessian @ rotated_direction)
        stress_absolute, stress_scaled = _scaled_error(
            actual_stress, expected_stress, scale_floor=1.0e-12
        )
        action_absolute, action_scaled = _scaled_error(
            actual_action, expected_action, scale_floor=1.0e-10
        )
        energy_absolute, energy_scaled = _scaled_error(
            float(rotated["energy"]), float(base["energy"]), scale_floor=1.0e-12
        )
        records.append(
            {
                "branch": str(branch),
                "rotation_index": int(rotation_index),
                "rotation_seed": int(seed),
                "rotation_matrix": [
                    [float(value) for value in row] for row in rotation
                ],
                "orthogonality_defect": float(
                    np.linalg.norm(rotation.T @ rotation - np.eye(3))
                ),
                "determinant": float(np.linalg.det(rotation)),
                "rotated_branch": str(
                    rotated["branch_diagnostics"]["branch"]
                ),
                "energy_invariance_absolute_error": energy_absolute,
                "energy_invariance_scaled_error": energy_scaled,
                "stress_covariance_absolute_error": stress_absolute,
                "stress_covariance_scaled_error": stress_scaled,
                "tangent_action_covariance_absolute_error": action_absolute,
                "tangent_action_covariance_scaled_error": action_scaled,
                "finite_energy_gradient_hessian": bool(
                    rotated["finite_energy_gradient_hessian"]
                ),
                "numpy_selected_energy_relative_error": float(
                    rotated["numpy_selected_energy_relative_error"]
                ),
                "hessian_symmetry_defect": float(
                    rotated["hessian_symmetry_defect"]
                ),
                "minimum_normalized_raw_principal_gap": float(
                    rotated["branch_diagnostics"][
                        "minimum_normalized_raw_principal_gap"
                    ]
                ),
                "relative_tie_break_scale": float(
                    rotated["branch_diagnostics"]["relative_tie_break_scale"]
                ),
                "scope": (
                    "NumPy tensor rotation against derivatives of the production JAX "
                    "scalar; evaluated only at distinct-principal-value branch interiors"
                ),
            }
        )
    return records


def _classify_principal(principal: np.ndarray, material: Material) -> str:
    eps6 = _tensor_to_strain6(np.diag(np.asarray(principal, dtype=np.float64)))
    return str(_branch_diagnostics(eps6, material)["branch"])


def _bisect_interface(
    left_branch: str,
    right_branch: str,
    left: np.ndarray,
    right: np.ndarray,
    material: Material,
) -> float:
    lo = 0.0
    hi = 1.0
    if _classify_principal(left, material) != left_branch:
        raise RuntimeError(f"left endpoint is not in {left_branch}")
    if _classify_principal(right, material) != right_branch:
        raise RuntimeError(f"right endpoint is not in {right_branch}")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        label = _classify_principal((1.0 - mid) * left + mid * right, material)
        if label == left_branch:
            lo = mid
        elif label == right_branch:
            hi = mid
        else:
            raise RuntimeError(
                f"segment {left_branch}->{right_branch} crosses intermediate branch {label}"
            )
    return 0.5 * (lo + hi)


def _interface_sweeps(
    material: Material,
    *,
    offsets: tuple[float, ...],
    energy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    gradient_fn: Callable[[jnp.ndarray], jnp.ndarray],
    hessian_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> list[dict[str, object]]:
    sweeps: list[dict[str, object]] = []
    for left_branch, right_branch in ADJACENT_INTERFACES:
        left = PRINCIPAL_ANCHORS[left_branch]
        right = PRINCIPAL_ANCHORS[right_branch]
        boundary = _bisect_interface(
            left_branch, right_branch, left, right, material
        )
        pairs: list[dict[str, object]] = []
        for offset in offsets:
            left_t = boundary - float(offset)
            right_t = boundary + float(offset)
            if left_t <= 0.0 or right_t >= 1.0:
                raise RuntimeError("interface offset leaves the anchor segment")
            left_principal = (1.0 - left_t) * left + left_t * right
            right_principal = (1.0 - right_t) * left + right_t * right
            left_eval = _evaluate(
                _tensor_to_strain6(np.diag(left_principal)),
                energy_fn,
                gradient_fn,
                hessian_fn,
                material,
            )
            right_eval = _evaluate(
                _tensor_to_strain6(np.diag(right_principal)),
                energy_fn,
                gradient_fn,
                hessian_fn,
                material,
            )
            left_energy = float(left_eval["energy"])
            right_energy = float(right_eval["energy"])
            left_gradient = np.asarray(left_eval["gradient"], dtype=np.float64)
            right_gradient = np.asarray(right_eval["gradient"], dtype=np.float64)
            left_hessian = np.asarray(left_eval["hessian"], dtype=np.float64)
            right_hessian = np.asarray(right_eval["hessian"], dtype=np.float64)
            pairs.append(
                {
                    "normalized_offset_fraction_of_anchor_segment": float(offset),
                    "left_parameter": float(left_t),
                    "right_parameter": float(right_t),
                    "left": _without_arrays(left_eval),
                    "right": _without_arrays(right_eval),
                    "paired_energy_scaled_difference": _scaled_error(
                        left_energy, right_energy, scale_floor=1.0e-12
                    )[1],
                    "paired_gradient_scaled_difference": _scaled_error(
                        left_gradient, right_gradient, scale_floor=1.0e-12
                    )[1],
                    "paired_hessian_scaled_difference": _scaled_error(
                        left_hessian, right_hessian, scale_floor=1.0e-10
                    )[1],
                    "derivative_gate_applied": False,
                    "exclusion_reason": (
                        "the pair deliberately approaches a branch interface; no "
                        "generalized-differentiability claim is made"
                    ),
                }
            )
        sweeps.append(
            {
                "interface": f"{left_branch}--{right_branch}",
                "left_branch": left_branch,
                "right_branch": right_branch,
                "boundary_parameter_on_anchor_segment": float(boundary),
                "anchor_segment_norm": float(np.linalg.norm(right - left)),
                "pairs": pairs,
            }
        )
    return sweeps


def _git_metadata() -> dict[str, object]:
    def command(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    return {
        "commit": command("rev-parse", "HEAD"),
        "dirty": bool(command("status", "--short")),
    }


def _dirty_snapshot_sha256() -> str:
    """Hash tracked changes plus untracked file contents at pilot preflight."""

    digest = hashlib.sha256()
    tracked = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--binary", "HEAD", "--", "."],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    digest.update(b"tracked-diff\0")
    digest.update(tracked)
    untracked_raw = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    for raw_path in sorted(part for part in untracked_raw.split(b"\0") if part):
        relative = os.fsdecode(raw_path)
        path = REPO_ROOT / relative
        digest.update(b"untracked-path\0")
        digest.update(raw_path)
        digest.update(b"\0")
        if path.is_file():
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        else:
            digest.update(b"missing-or-non-file")
    return digest.hexdigest()


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "local CPU; model unavailable"


def _memory_model() -> str:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                return line.replace("MemTotal:", "host MemTotal", 1).strip()
    return "local host memory; capacity unavailable"


def _artifact_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _build_run_record(
    *,
    payload: dict[str, object],
    args: argparse.Namespace,
    preflight: object,
    dirty_patch_sha256: str | None,
    started_at_utc: str,
    finished_at_utc: str,
    total_seconds: float,
    detailed_output: Path,
    report_path: Path | None,
) -> dict[str, object]:
    status = "success" if payload["status"] == "passed" else "failure"
    configuration_bytes = json.dumps(
        {
            "configuration": payload["configuration"],
            "contract": payload["contract"],
        },
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    configuration_digest = hashlib.sha256(configuration_bytes).hexdigest()
    runner_path = Path(__file__).resolve()
    production_path = (
        REPO_ROOT
        / "src"
        / "problems"
        / "slope_stability_3d"
        / "jax"
        / "jax_energy_3d.py"
    )
    protocol_path = REPO_ROOT / "paper" / "protocols" / "EXP-MC-001.md"
    code_hashes = {
        _artifact_label(runner_path): sha256_file(runner_path),
        _artifact_label(production_path): sha256_file(production_path),
    }
    if protocol_path.is_file():
        code_hashes[_artifact_label(protocol_path)] = sha256_file(protocol_path)

    interior_count = len(payload["branch_interiors"])
    rotation_count = len(payload["rotation_covariance"])
    degeneracy_count = len(payload["repeated_principal_value_cases"])
    interface_point_count = 2 * sum(
        len(sweep["pairs"]) for sweep in payload["interface_sweeps"]
    )
    base_evaluation_count = (
        interior_count
        + interior_count  # principal-frame bases repeated by covariance block
        + rotation_count
        + degeneracy_count
        + interface_point_count
    )
    centered_evaluation_count = 2 * interior_count * len(
        payload["configuration"]["fd_steps"]
    )
    function_evaluations = 1 + base_evaluation_count + centered_evaluation_count
    gradient_evaluations = 1 + base_evaluation_count + centered_evaluation_count
    hessian_evaluations = 1 + base_evaluation_count

    try:
        threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
    except ValueError:
        threads = 1
    summary = payload["summary"]
    artifacts_reports = [] if report_path is None else [_artifact_label(report_path)]
    return {
        "schema": {
            "id": RUN_RECORD_SCHEMA_ID,
            "version": RUN_RECORD_SCHEMA_VERSION,
        },
        "record_id": "paper-revision-2026-07-10-exp-mc-001-material-point-r01",
        "run_kind": str(args.run_kind),
        "identifiers": {
            "campaign": "paper_revision_2026_07_10",
            "experiment": "EXP-MC-001",
            "case": "dimensionless-five-branch-material-point-matrix",
            "method": "jax-scalar-autodiff",
            "route": "production-mohr-coulomb-scalar",
            "repetition": 1,
        },
        "problem": {
            "name": "Plasticity3D Mohr-Coulomb endpoint potential",
            "mesh": "not-applicable (material-point campaign)",
            "degree": None,
            "quadrature": "not-applicable (one constitutive point per case)",
            "total_degrees_of_freedom": 6,
            "free_degrees_of_freedom": 6,
            "notes": (
                "The six coordinates are engineering-strain components, not finite-element "
                "unknowns. No mesh or external data input is used."
            ),
        },
        "solver": {
            "algorithm": "deterministic material-point derivative verification",
            "implementation": _artifact_label(runner_path),
            "parameters": dict(payload["configuration"]),
            "preconditioner": {"type": "not-applicable"},
            "stopping_contract": "EXP-MC-001-v1",
        },
        "termination": {
            "status": status,
            "reason": (
                "all preregistered EXP-MC-001 gates passed"
                if status == "success"
                else "one or more preregistered EXP-MC-001 gates failed"
            ),
            "exit_code": 0 if status == "success" else 2,
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "limit_kind": None,
            "limit_value": None,
            "censored": False,
        },
        "accuracy": {
            "contract_id": "EXP-MC-001-v1",
            "gate_passed": status == "success",
            "absolute_residual": None,
            "relative_residual": None,
            "scaled_residual": None,
            "relative_correction": None,
            "energy_change": None,
            "custom_metrics": dict(summary),
            "notes": (
                "Residual and correction fields are not applicable to a material-point "
                "verification. All numerical gates are retained in custom_metrics."
            ),
        },
        "counts": {
            "nonlinear_iterations": 0,
            "krylov_iterations": 0,
            "function_evaluations": int(function_evaluations),
            "gradient_evaluations": int(gradient_evaluations),
            "hessian_evaluations": int(hessian_evaluations),
            "hvp_evaluations": 0,
            "preconditioner_setups": 0,
            "notes": (
                "Counts include one JIT warm-up evaluation of each derivative order, all "
                "recorded points, and both sides of every centered stencil. Tangent actions "
                "multiply stored 6-by-6 Hessians and are not separate HVP-kernel calls."
            ),
        },
        "timing": {
            "aggregation": "single-process wall clock",
            "cold_process": True,
            "barrier_policy": "not-applicable (serial process)",
            "synchronization_policy": (
                "JAX warm-up values are blocked before recorded evaluations; NumPy "
                "conversion synchronizes device results copied to host"
            ),
            "phases_overlap": False,
            "relation_to_total": "only end-to-end time is attributed; phase fields are null",
            "process_startup_s": None,
            "jit_compilation_s": None,
            "coloring_s": None,
            "derivative_evaluation_s": None,
            "constitutive_contraction_s": None,
            "assembly_s": None,
            "communication_s": None,
            "preconditioner_setup_s": None,
            "krylov_solve_s": None,
            "globalization_s": None,
            "state_output_s": None,
            "total_s": float(total_seconds),
            "notes": "Small correctness pilot; timing is provenance, not performance evidence.",
        },
        "resources": {
            "nodes": 1,
            "ranks": 1,
            "threads_per_rank": threads,
            "peak_memory_per_rank_bytes": None,
            "peak_memory_per_node_bytes": None,
            "tracked_allocations_bytes": None,
            "measurement_method": "not measured for this sub-second correctness campaign",
            "notes": "CPU-only local process; memory metrics are not used as evidence.",
        },
        "diagnostics": {
            "state": {
                "principal_anchors": {
                    key: [float(value) for value in values]
                    for key, values in PRINCIPAL_ANCHORS.items()
                },
                "repeated_spectrum_case_count": degeneracy_count,
            },
            "branch": {
                "interior_counts": dict(summary["branch_interior_counts"]),
                "adjacent_interface_count": int(summary["interface_count"]),
                "two_sided_pair_count": int(summary["interface_pair_count"]),
            },
            "feasibility": {},
            "kkt": {},
        },
        "environment": {
            "python": sys.version.split()[0],
            "packages": {
                "jax": jax.__version__,
                "jaxlib": jaxlib.__version__,
                "numpy": np.__version__,
            },
            "platform": platform.platform(),
            "jax": jax.__version__,
            "xla": jaxlib.__version__,
            "jax_enable_x64": bool(jax.config.x64_enabled),
            "petsc": "not-applicable",
            "mpi": "not-applicable",
            "compiler": "not-applicable (prebuilt Python/JAX runtime)",
            "blas": "NumPy linked BLAS; provider not separately captured",
            "cpu_model": _cpu_model(),
            "node_model": platform.node() or "local host",
            "memory_model": _memory_model(),
            "scheduler": "local",
            "scheduler_job_id": None,
            "affinity": f"OMP_NUM_THREADS={threads}; one serial process",
        },
        "provenance": {
            "git_commit": str(preflight.git_commit),
            "git_clean": bool(preflight.git_clean),
            "git_status_porcelain": list(preflight.git_status_porcelain),
            "pilot_override": bool(preflight.pilot_override),
            "pilot_override_reason": preflight.pilot_override_reason,
            "command_argv": [str(sys.executable), *sys.argv],
            "working_directory": str(REPO_ROOT),
            "code_hashes": code_hashes,
            "configuration_hashes": {
                "EXP-MC-001-configuration-v1": configuration_digest
            },
            "input_hashes": {},
            "dirty_patch_sha256": dirty_patch_sha256,
            "seed": int(args.seed),
            "deterministic_policy": (
                "Frozen branch anchors, interface adjacency, repeated spectra, offsets, "
                "and seed-derived rotations/directions; CPU FP64."
            ),
            "recorded_at_utc": finished_at_utc,
            "preflight_checked_at_utc": str(preflight.checked_at_utc),
        },
        "artifacts": {
            "raw_outputs": [_artifact_label(detailed_output)],
            "states": [],
            "logs": [],
            "tables": [],
            "figures": [],
            "reports": artifacts_reports,
        },
    }


def _all_floats_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_all_floats_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_floats_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(float(value)))
    return True


def run(args: argparse.Namespace) -> dict[str, object]:
    material = Material()

    def energy(eps6: jnp.ndarray) -> jnp.ndarray:
        return mc_potential_density_3d(
            eps6,
            material.c_bar,
            material.sin_phi,
            material.shear,
            material.bulk,
            material.lame,
            tiny=TIE_BREAK_TINY,
        )

    energy_fn = jax.jit(energy)
    gradient_fn = jax.jit(jax.grad(energy))
    hessian_fn = jax.jit(jax.hessian(energy))
    # Compile all three functions before recording values.
    warmup = jnp.zeros(6, dtype=jnp.float64)
    energy_fn(warmup).block_until_ready()
    gradient_fn(warmup).block_until_ready()
    hessian_fn(warmup).block_until_ready()

    steps = tuple(float(value) for value in str(args.fd_steps).split(","))
    offsets = tuple(float(value) for value in str(args.interface_offsets).split(","))
    if len(steps) < 3 or any(value <= 0.0 for value in steps):
        raise ValueError("fd-steps must contain at least three positive values")
    if not 0 <= int(args.fd_gate_index) < len(steps):
        raise ValueError("fd-gate-index is out of range")
    if len(offsets) < 3 or any(value <= 0.0 for value in offsets):
        raise ValueError("interface-offsets must contain at least three positive values")
    if tuple(sorted(offsets, reverse=True)) != offsets:
        raise ValueError("interface-offsets must be strictly ordered largest to smallest")

    interior_records: list[dict[str, object]] = []
    rotation_records: list[dict[str, object]] = []
    for branch_index, branch in enumerate(BRANCHES):
        principal = PRINCIPAL_ANCHORS[branch]
        eps6 = _tensor_to_strain6(np.diag(principal))
        evaluation = _evaluate(
            eps6, energy_fn, gradient_fn, hessian_fn, material
        )
        directional = _directional_check(
            eps6,
            evaluation,
            seed=int(args.seed) + 100 * branch_index,
            steps=steps,
            gate_index=int(args.fd_gate_index),
            energy_fn=energy_fn,
            gradient_fn=gradient_fn,
            material=material,
        )
        interior_records.append(
            {
                "expected_branch": branch,
                "principal_anchor": [float(value) for value in principal],
                "evaluation": _without_arrays(evaluation),
                "directional_check": directional,
            }
        )
        rotation_records.extend(
            _rotation_checks(
                branch,
                principal,
                rotations=int(args.rotations_per_branch),
                seed=int(args.seed) + 10_000 + 100 * branch_index,
                energy_fn=energy_fn,
                gradient_fn=gradient_fn,
                hessian_fn=hessian_fn,
                material=material,
            )
        )

    degeneracy_records: list[dict[str, object]] = []
    for name, principal in DEGENERACY_CASES.items():
        evaluation = _evaluate(
            _tensor_to_strain6(np.diag(principal)),
            energy_fn,
            gradient_fn,
            hessian_fn,
            material,
        )
        degeneracy_records.append(
            {
                "name": name,
                "principal_values": [float(value) for value in principal],
                "evaluation": _without_arrays(evaluation),
                "derivative_gate_applied": False,
                "rotation_gate_applied": False,
                "exclusion_reason": (
                    "repeated or nearly repeated spectrum; finite values are audited, "
                    "but spectral smoothness and rotation covariance are not inferred"
                ),
            }
        )

    interface_sweeps = _interface_sweeps(
        material,
        offsets=offsets,
        energy_fn=energy_fn,
        gradient_fn=gradient_fn,
        hessian_fn=hessian_fn,
    )

    interior_branch_counts = {
        branch: sum(
            str(record["evaluation"]["branch_diagnostics"]["branch"]) == branch
            for record in interior_records
        )
        for branch in BRANCHES
    }
    maximum_symmetry = max(
        float(record["evaluation"]["hessian_symmetry_defect"])
        for record in interior_records
    )
    maximum_fd_energy = max(
        float(record["directional_check"]["energy_error_at_gate"])
        for record in interior_records
    )
    maximum_fd_hvp = max(
        float(record["directional_check"]["hvp_error_at_gate"])
        for record in interior_records
    )
    minimum_active_margin = min(
        float(
            record["evaluation"]["branch_diagnostics"][
                "normalized_active_branch_margin"
            ]
        )
        for record in interior_records
    )
    maximum_energy_transcription = max(
        float(record["evaluation"]["numpy_selected_energy_relative_error"])
        for record in interior_records
    )
    maximum_rotation_energy = max(
        float(record["energy_invariance_scaled_error"])
        for record in rotation_records
    )
    maximum_rotation_stress = max(
        float(record["stress_covariance_scaled_error"])
        for record in rotation_records
    )
    maximum_rotation_action = max(
        float(record["tangent_action_covariance_scaled_error"])
        for record in rotation_records
    )
    maximum_rotation_action_absolute = max(
        float(record["tangent_action_covariance_absolute_error"])
        for record in rotation_records
    )
    maximum_interface_energy_transcription = max(
        max(
            float(pair["left"]["numpy_selected_energy_relative_error"]),
            float(pair["right"]["numpy_selected_energy_relative_error"]),
        )
        for sweep in interface_sweeps
        for pair in sweep["pairs"]
    )
    maximum_degeneracy_energy_transcription = max(
        float(record["evaluation"]["numpy_selected_energy_relative_error"])
        for record in degeneracy_records
    )

    interior_passed = bool(
        all(value == 1 for value in interior_branch_counts.values())
        and all(
            bool(record["evaluation"]["finite_energy_gradient_hessian"])
            and str(record["evaluation"]["branch_diagnostics"]["branch"])
            == str(record["expected_branch"])
            and bool(record["directional_check"]["branch_stable_for_all_steps"])
            for record in interior_records
        )
        and minimum_active_margin >= float(args.minimum_active_margin)
        and maximum_symmetry <= float(args.symmetry_tolerance)
        and maximum_fd_energy <= float(args.fd_tolerance)
        and maximum_fd_hvp <= float(args.fd_tolerance)
        and maximum_energy_transcription <= float(args.transcription_tolerance)
    )
    rotations_passed = bool(
        all(
            bool(record["finite_energy_gradient_hessian"])
            and str(record["rotated_branch"]) == str(record["branch"])
            and float(record["orthogonality_defect"])
            <= float(args.rotation_orthogonality_tolerance)
            and abs(float(record["determinant"]) - 1.0)
            <= float(args.rotation_orthogonality_tolerance)
            and float(record["energy_invariance_scaled_error"])
            <= float(args.rotation_tolerance)
            and float(record["stress_covariance_scaled_error"])
            <= float(args.rotation_tolerance)
            and float(record["numpy_selected_energy_relative_error"])
            <= float(args.transcription_tolerance)
            and float(record["hessian_symmetry_defect"])
            <= float(args.symmetry_tolerance)
            and (
                float(record["tangent_action_covariance_scaled_error"])
                <= float(args.rotation_tolerance)
                or float(record["tangent_action_covariance_absolute_error"])
                <= float(args.rotation_absolute_tolerance)
            )
            for record in rotation_records
        )
    )
    degeneracies_passed = bool(
        all(
            bool(record["evaluation"]["finite_energy_gradient_hessian"])
            and float(record["evaluation"]["hessian_symmetry_defect"])
            <= float(args.symmetry_tolerance)
            and float(
                record["evaluation"]["numpy_selected_energy_relative_error"]
            )
            <= float(args.transcription_tolerance)
            for record in degeneracy_records
        )
    )
    interfaces_passed = bool(
        len(interface_sweeps) == len(ADJACENT_INTERFACES)
        and all(
            str(pair["left"]["branch_diagnostics"]["branch"])
            == str(sweep["left_branch"])
            and str(pair["right"]["branch_diagnostics"]["branch"])
            == str(sweep["right_branch"])
            and bool(pair["left"]["finite_energy_gradient_hessian"])
            and bool(pair["right"]["finite_energy_gradient_hessian"])
            and float(pair["left"]["hessian_symmetry_defect"])
            <= float(args.symmetry_tolerance)
            and float(pair["right"]["hessian_symmetry_defect"])
            <= float(args.symmetry_tolerance)
            and float(pair["left"]["numpy_selected_energy_relative_error"])
            <= float(args.transcription_tolerance)
            and float(pair["right"]["numpy_selected_energy_relative_error"])
            <= float(args.transcription_tolerance)
            for sweep in interface_sweeps
            for pair in sweep["pairs"]
        )
    )
    execution_contract_passed = bool(
        jax.config.x64_enabled and str(jax.default_backend()) == "cpu"
    )

    payload: dict[str, object] = {
        "schema_name": "plasticity3d_material_point_verification",
        "schema_version": 1,
        "experiment_id": "EXP-MC-001",
        "status": "pending",
        "configuration": {
            "material": material.as_json(),
            "engineering_strain_order": ["xx", "yy", "zz", "xy", "yz", "xz"],
            "tie_break_tiny": TIE_BREAK_TINY,
            "seed": int(args.seed),
            "rotations_per_branch": int(args.rotations_per_branch),
            "fd_steps": [float(value) for value in steps],
            "fd_gate_index": int(args.fd_gate_index),
            "interface_offsets": [float(value) for value in offsets],
        },
        "contract": {
            "minimum_normalized_active_branch_margin": float(
                args.minimum_active_margin
            ),
            "hessian_symmetry_tolerance": float(args.symmetry_tolerance),
            "centered_fd_scaled_error_tolerance": float(args.fd_tolerance),
            "numpy_energy_transcription_relative_tolerance": float(
                args.transcription_tolerance
            ),
            "rotation_scaled_tolerance": float(args.rotation_tolerance),
            "rotation_absolute_tolerance_for_near_zero_tangent_actions": float(
                args.rotation_absolute_tolerance
            ),
            "required_branches": list(BRANCHES),
            "required_nondegenerate_adjacent_interfaces": [
                f"{left}--{right}" for left, right in ADJACENT_INTERFACES
            ],
        },
        "summary": {
            "branch_interior_counts": interior_branch_counts,
            "interface_count": len(interface_sweeps),
            "interface_pair_count": sum(
                len(sweep["pairs"]) for sweep in interface_sweeps
            ),
            "rotation_check_count": len(rotation_records),
            "degeneracy_case_count": len(degeneracy_records),
            "minimum_normalized_active_branch_margin": minimum_active_margin,
            "maximum_hessian_symmetry_defect": maximum_symmetry,
            "maximum_centered_energy_directional_error_at_gate": maximum_fd_energy,
            "maximum_centered_hvp_error_at_gate": maximum_fd_hvp,
            "maximum_numpy_energy_transcription_relative_error": maximum_energy_transcription,
            "maximum_interface_numpy_energy_transcription_relative_error": maximum_interface_energy_transcription,
            "maximum_degeneracy_numpy_energy_transcription_relative_error": maximum_degeneracy_energy_transcription,
            "maximum_rotation_energy_invariance_scaled_error": maximum_rotation_energy,
            "maximum_rotation_stress_covariance_scaled_error": maximum_rotation_stress,
            "maximum_rotation_tangent_action_covariance_scaled_error": maximum_rotation_action,
            "maximum_rotation_tangent_action_covariance_absolute_error": maximum_rotation_action_absolute,
            "interior_checks_passed": interior_passed,
            "rotation_checks_passed": rotations_passed,
            "interface_sweeps_passed": interfaces_passed,
            "degeneracy_finiteness_checks_passed": degeneracies_passed,
            "cpu_fp64_execution_passed": execution_contract_passed,
        },
        "branch_interiors": interior_records,
        "rotation_covariance": rotation_records,
        "interface_sweeps": interface_sweeps,
        "repeated_principal_value_cases": degeneracy_records,
        "method_scope": {
            "production_object": (
                "src.problems.slope_stability_3d.jax.jax_energy_3d."
                "mc_potential_density_3d"
            ),
            "independent_parts": (
                "engineering-tensor conversion, rotations, eigendecomposition, and "
                "predicate evaluation are implemented separately in NumPy"
            ),
            "transcribed_parts": (
                "branch inequalities and candidate-energy algebra transcribe the same "
                "Mohr-Coulomb formulas; they are not an independent constitutive reference"
            ),
            "excluded_claims": [
                "no generalized differentiability at a branch switch",
                "no path-consistent incremental-plasticity validation",
                "no physical validation of the endpoint surrogate",
                "no rotation-covariance inference at repeated or nearly repeated spectra",
            ],
        },
        "provenance": {
            "git": _git_metadata(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jax_enable_x64": bool(jax.config.x64_enabled),
            "jax_backend": str(jax.default_backend()),
            "numpy": np.__version__,
            "command": shlex.join(sys.argv),
            "environment": {
                "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", ""),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
                "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
            },
        },
    }
    payload["status"] = (
        "passed"
        if interior_passed
        and rotations_passed
        and interfaces_passed
        and degeneracies_passed
        and execution_contract_passed
        and _all_floats_finite(payload)
        else "failed"
    )
    return payload


def _render_report(payload: dict[str, object], output_path: Path) -> str:
    summary = payload["summary"]
    counts = summary["branch_interior_counts"]
    lines = [
        "# EXP-MC-001 Local Material-Point Pilot",
        "",
        f"- status: `{payload['status']}`",
        f"- JSON record: `{output_path.name}`",
        "- versioned terminal record: `run_record.json`",
        "- evidence class: dirty-worktree local CPU pilot unless provenance says otherwise",
        "",
        "## Coverage",
        "",
        "| item | result |",
        "| --- | ---: |",
    ]
    for branch in BRANCHES:
        lines.append(f"| `{branch}` interior | `{counts[branch]}` |")
    lines.extend(
        [
            f"| adjacent interfaces | `{summary['interface_count']}` |",
            f"| two-sided interface pairs | `{summary['interface_pair_count']}` |",
            f"| random-rotation checks | `{summary['rotation_check_count']}` |",
            f"| repeated/nearly repeated spectra | `{summary['degeneracy_case_count']}` |",
            "",
            "## Maximum Errors And Minimum Margins",
            "",
            "| quantity | value |",
            "| --- | ---: |",
            "| minimum normalized active branch margin | "
            f"`{summary['minimum_normalized_active_branch_margin']:.6e}` |",
            "| Hessian symmetry defect | "
            f"`{summary['maximum_hessian_symmetry_defect']:.6e}` |",
            "| centered energy directional error | "
            f"`{summary['maximum_centered_energy_directional_error_at_gate']:.6e}` |",
            "| centered HVP error | "
            f"`{summary['maximum_centered_hvp_error_at_gate']:.6e}` |",
            "| NumPy energy-transcription error | "
            f"`{summary['maximum_numpy_energy_transcription_relative_error']:.6e}` |",
            "| interface NumPy energy-transcription error | "
            f"`{summary['maximum_interface_numpy_energy_transcription_relative_error']:.6e}` |",
            "| degeneracy NumPy energy-transcription error | "
            f"`{summary['maximum_degeneracy_numpy_energy_transcription_relative_error']:.6e}` |",
            "| rotation energy-invariance error | "
            f"`{summary['maximum_rotation_energy_invariance_scaled_error']:.6e}` |",
            "| rotation stress-covariance error | "
            f"`{summary['maximum_rotation_stress_covariance_scaled_error']:.6e}` |",
            "| rotation tangent-action covariance error | "
            f"`{summary['maximum_rotation_tangent_action_covariance_scaled_error']:.6e}` |",
            "",
            "## Interface Diagnostics At The Smallest Offset",
            "",
            "These differences are descriptive only; the Hessian is allowed to jump.",
            "",
            "| interface | offset | labels | scaled energy difference | scaled gradient difference | scaled Hessian difference |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for sweep in payload["interface_sweeps"]:
        pair = sweep["pairs"][-1]
        left_label = pair["left"]["branch_diagnostics"]["branch"]
        right_label = pair["right"]["branch_diagnostics"]["branch"]
        lines.append(
            f"| `{sweep['interface']}` | "
            f"`{pair['normalized_offset_fraction_of_anchor_segment']:.1e}` | "
            f"`{left_label}` / `{right_label}` | "
            f"`{pair['paired_energy_scaled_difference']:.6e}` | "
            f"`{pair['paired_gradient_scaled_difference']:.6e}` | "
            f"`{pair['paired_hessian_scaled_difference']:.6e}` |"
        )
    lines.extend(
        [
            "",
            "## Repeated-Spectrum Diagnostics",
            "",
            "| case | selected branch | raw minimum gap | tie-broken minimum gap | finite derivatives |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for record in payload["repeated_principal_value_cases"]:
        evaluation = record["evaluation"]
        diagnostic = evaluation["branch_diagnostics"]
        lines.append(
            f"| `{record['name']}` | `{diagnostic['branch']}` | "
            f"`{diagnostic['minimum_raw_principal_gap']:.6e}` | "
            f"`{diagnostic['minimum_tie_broken_principal_gap']:.6e}` | "
            f"`{evaluation['finite_energy_gradient_hessian']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The pilot exercises all five strict branch interiors, all five "
            "nondegenerate adjacent branch interfaces from both sides, deterministic "
            "random rotations, and exact/nearly repeated principal values. Centered "
            "derivative checks and rotation covariance are gated only away from branch "
            "switches and spectral degeneracies.",
            "",
            "The NumPy predicate and candidate-energy calculations are an independent "
            "implementation of the algebra, but they transcribe the same constitutive "
            "formulas. They are therefore not an independent physical constitutive "
            "reference. This pilot makes no generalized-differentiability claim at "
            "switches and no path-consistent plasticity claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--run-record",
        type=Path,
        help="versioned terminal run record (default: run_record.json beside --output)",
    )
    parser.add_argument("--run-kind", choices=("pilot", "publication"), default="pilot")
    parser.add_argument("--pilot-dirty-override", action="store_true")
    parser.add_argument("--pilot-override-reason")
    parser.add_argument("--seed", type=int, default=240513)
    parser.add_argument("--rotations-per-branch", type=int, default=3)
    parser.add_argument("--fd-steps", default="3e-5,1e-5,3e-6,1e-6,3e-7")
    parser.add_argument("--fd-gate-index", type=int, default=3)
    parser.add_argument("--interface-offsets", default="1e-2,1e-4,1e-6")
    parser.add_argument("--minimum-active-margin", type=float, default=1.0e-3)
    parser.add_argument("--symmetry-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--fd-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--transcription-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--rotation-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--rotation-absolute-tolerance", type=float, default=1.0e-9)
    parser.add_argument(
        "--rotation-orthogonality-tolerance", type=float, default=1.0e-12
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    preflight = check_experiment_preflight(
        REPO_ROOT,
        run_kind=str(args.run_kind),
        pilot_dirty_override=bool(args.pilot_dirty_override),
        pilot_override_reason=args.pilot_override_reason,
    )
    dirty_patch_sha256 = (
        None if bool(preflight.git_clean) else _dirty_snapshot_sha256()
    )
    started_at_utc = utc_now_iso()
    started = time.perf_counter()
    payload = run(args)
    if not _all_floats_finite(payload):
        raise RuntimeError("EXP-MC-001 payload contains a nonfinite float")
    output = Path(args.output).resolve()
    atomic_write_json(output, payload)
    report: Path | None = None
    if args.report is not None:
        report = Path(args.report).resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(_render_report(payload, output), encoding="utf-8")
    finished_at_utc = utc_now_iso()
    total_seconds = float(time.perf_counter() - started)
    run_record_path = (
        Path(args.run_record).resolve()
        if args.run_record is not None
        else output.with_name("run_record.json")
    )
    if run_record_path == output:
        raise ValueError("--run-record and --output must name different files")
    run_record = _build_run_record(
        payload=payload,
        args=args,
        preflight=preflight,
        dirty_patch_sha256=dirty_patch_sha256,
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        total_seconds=total_seconds,
        detailed_output=output,
        report_path=report,
    )
    atomic_write_run_record(run_record_path, run_record)
    print(json.dumps(payload["summary"], indent=2, allow_nan=False), flush=True)
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify Plasticity3D element and constitutive derivative routes at fixed states."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from src.problems.slope_stability_3d.jax.jax_energy_3d import (
    constitutive_element_hessian_3d,
    constitutive_element_residual_3d,
    element_energy_3d,
    element_hessian_3d,
    element_residual_3d,
)
from src.problems.slope_stability_3d.support.mesh import ensure_same_mesh_case_hdf5
from src.problems.slope_stability_3d.support.reduction import davis_b_reduction_qp


REPO_ROOT = Path(__file__).resolve().parents[2]


def _relative_error(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), np.finfo(float).tiny)
    return float(np.linalg.norm(left - right) / scale)


def _symmetry_defect(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    return float(np.linalg.norm(matrix - matrix.T) / max(np.linalg.norm(matrix), np.finfo(float).tiny))


def _directional_error(value: float, reference: float, gradient: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(gradient)), abs(float(value)), abs(float(reference)), np.finfo(float).tiny)
    return float(abs(float(value) - float(reference)) / scale)


def _strain_batch(u: np.ndarray, data: dict[str, np.ndarray]) -> np.ndarray:
    ux = np.asarray(u[0::3], dtype=np.float64)
    uy = np.asarray(u[1::3], dtype=np.float64)
    uz = np.asarray(u[2::3], dtype=np.float64)
    dphix = np.asarray(data["dphix"], dtype=np.float64)
    dphiy = np.asarray(data["dphiy"], dtype=np.float64)
    dphiz = np.asarray(data["dphiz"], dtype=np.float64)
    return np.column_stack(
        (
            dphix @ ux,
            dphiy @ uy,
            dphiz @ uz,
            dphiy @ ux + dphix @ uy,
            dphiz @ uy + dphiy @ uz,
            dphiz @ ux + dphix @ uz,
        )
    )


def _branch_diagnostics(u: np.ndarray, data: dict[str, np.ndarray], *, tiny: float = 1.0e-15) -> dict[str, object]:
    """Re-evaluate production branch predicates without differentiating them.

    This is an implementation-predicate diagnostic, not an independent
    constitutive reference or a proof of regularity.
    """
    strains = _strain_batch(u, data)
    labels: list[str] = []
    normalized_margins: list[float] = []
    normalized_active_margins: list[float] = []
    normalized_gaps: list[float] = []
    normalized_denominators: list[float] = []
    tie_break_scales: list[float] = []
    for idx, eps6 in enumerate(strains):
        e11, e22, e33, g12, g23, g13 = (float(value) for value in eps6)
        matrix = np.array(
            [
                [e11, 0.5 * g12, 0.5 * g13],
                [0.5 * g12, e22, 0.5 * g23],
                [0.5 * g13, 0.5 * g23, e33],
            ],
            dtype=np.float64,
        )
        perturbed = matrix + tiny * np.diag([0.0, 1.0, 2.0])
        eig_3, eig_2, eig_1 = np.linalg.eigvalsh(perturbed)
        invariant_1 = e11 + e22 + e33
        c_bar = float(data["c_bar"][idx])
        sin_phi = float(data["sin_phi"][idx])
        shear = float(data["shear"][idx])
        bulk = float(data["bulk"][idx])
        lame = float(data["lame"][idx])

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
        lambda_s = f_tr / np.copysign(max(abs(denom_s), tiny), denom_s)
        lambda_l = (
            shear * ((1.0 + sin_phi) * (eig_1 + eig_2) - 2.0 * (1.0 - sin_phi) * eig_3)
            + 2.0 * lame * sin_phi * invariant_1
            - c_bar
        ) / np.copysign(max(abs(denom_l), tiny), denom_l)
        lambda_r = (
            shear * (2.0 * (1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * (eig_2 + eig_3))
            + 2.0 * lame * sin_phi * invariant_1
            - c_bar
        ) / np.copysign(max(abs(denom_r), tiny), denom_r)

        if f_tr <= 0.0:
            label = "elastic"
        elif lambda_s <= min(gamma_sl, gamma_sr):
            label = "shear"
        elif gamma_sl < gamma_sr and gamma_sl <= lambda_l <= gamma_la:
            label = "left_edge"
        elif gamma_sl > gamma_sr and gamma_sr <= lambda_r <= gamma_ra:
            label = "right_edge"
        else:
            label = "apex"
        labels.append(label)

        switch_quantities = np.asarray(
            [
                f_tr,
                lambda_s - min(gamma_sl, gamma_sr),
                gamma_sl - gamma_sr,
                lambda_l - gamma_sl,
                gamma_la - lambda_l,
                lambda_r - gamma_sr,
                gamma_ra - lambda_r,
            ],
            dtype=np.float64,
        )
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
            abs(gamma_sl),
            abs(gamma_sr),
            abs(gamma_la),
            abs(gamma_ra),
            abs(c_bar) / modulus_scale,
            np.finfo(float).tiny,
        )
        normalized_switches = np.concatenate(
            (
                np.asarray([switch_quantities[0] / stress_scale]),
                switch_quantities[1:] / strain_scale,
            )
        )
        normalized_margins.append(float(np.min(np.abs(normalized_switches))))
        if label == "elastic":
            active_margin = -normalized_switches[0]
        elif label == "shear":
            active_margin = min(
                normalized_switches[0],
                -normalized_switches[1],
            )
        elif label == "left_edge":
            active_margin = min(
                normalized_switches[0],
                -normalized_switches[2],
                normalized_switches[3],
                normalized_switches[4],
            )
        elif label == "right_edge":
            active_margin = min(
                normalized_switches[0],
                normalized_switches[2],
                normalized_switches[5],
                normalized_switches[6],
            )
        else:
            active_margin = float(np.min(np.abs(normalized_switches)))
        normalized_active_margins.append(float(active_margin))
        eigen_scale = max(abs(eig_1), abs(eig_2), abs(eig_3), np.finfo(float).tiny)
        normalized_gaps.append(float(min(eig_1 - eig_2, eig_2 - eig_3) / eigen_scale))
        normalized_denominators.append(
            float(min(abs(denom_s), abs(denom_l), abs(denom_r), abs(denom_a)) / modulus_scale)
        )
        tie_break_scales.append(float(tiny * np.sqrt(5.0) / max(np.linalg.norm(matrix), tiny)))

    counts = {label: labels.count(label) for label in sorted(set(labels))}
    return {
        "labels": labels,
        "counts": counts,
        "minimum_normalized_switch_margin": float(min(normalized_margins)),
        "minimum_normalized_active_branch_margin": float(min(normalized_active_margins)),
        "minimum_normalized_principal_gap": float(min(normalized_gaps)),
        "minimum_normalized_denominator": float(min(normalized_denominators)),
        "maximum_relative_tie_break_scale": float(max(tie_break_scales)),
        "interpretation": "production predicate replay only; not an independent material reference",
    }


def _slopes(step_sizes: np.ndarray, errors: np.ndarray) -> list[float]:
    steps = np.asarray(step_sizes, dtype=np.float64)
    values = np.asarray(errors, dtype=np.float64)
    out: list[float] = []
    for idx in range(len(steps) - 1):
        if values[idx] <= 0.0 or values[idx + 1] <= 0.0:
            out.append(float("nan"))
            continue
        out.append(float(np.log(values[idx] / values[idx + 1]) / np.log(steps[idx] / steps[idx + 1])))
    return out


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _load_element(mesh_name: str, degree: int, element_index: int, lambda_target: float) -> dict[str, np.ndarray]:
    path = ensure_same_mesh_case_hdf5(mesh_name, degree, constraint_variant="glued_bottom")
    with h5py.File(path, "r") as handle:
        n_elements = int(handle["elems"].shape[0])
        if not 0 <= int(element_index) < n_elements:
            raise ValueError(f"element_index must lie in [0, {n_elements})")
        row = int(element_index)
        payload = {
            "dphix": np.asarray(handle["dphix"][row], dtype=np.float64),
            "dphiy": np.asarray(handle["dphiy"][row], dtype=np.float64),
            "dphiz": np.asarray(handle["dphiz"][row], dtype=np.float64),
            "quad_weight": np.asarray(handle["quad_weight"][row], dtype=np.float64),
            "c0": np.asarray(handle["c0_q"][row], dtype=np.float64),
            "phi": np.asarray(handle["phi_q"][row], dtype=np.float64),
            "psi": np.asarray(handle["psi_q"][row], dtype=np.float64),
            "shear": np.asarray(handle["shear_q"][row], dtype=np.float64),
            "bulk": np.asarray(handle["bulk_q"][row], dtype=np.float64),
            "lame": np.asarray(handle["lame_q"][row], dtype=np.float64),
            "element_dofs": np.asarray(handle["elems"][row], dtype=np.int64),
        }
    c_bar, sin_phi = davis_b_reduction_qp(
        payload.pop("c0"), payload.pop("phi"), payload.pop("psi"), float(lambda_target)
    )
    payload["c_bar"] = np.asarray(c_bar, dtype=np.float64)
    payload["sin_phi"] = np.asarray(sin_phi, dtype=np.float64)
    payload["source_hdf5"] = np.asarray([str(path)], dtype=object)
    return payload


def _jax_args(data: dict[str, np.ndarray]) -> tuple[jnp.ndarray, ...]:
    return tuple(
        jnp.asarray(data[key], dtype=jnp.float64)
        for key in (
            "dphix",
            "dphiy",
            "dphiz",
            "quad_weight",
            "c_bar",
            "sin_phi",
            "shear",
            "bulk",
            "lame",
        )
    )


def _canonical_assembled_state(n_free: int, state_scale: float) -> np.ndarray:
    """Return a deterministic, nonzero state in reordered free-DOF coordinates."""
    indices = np.arange(int(n_free), dtype=np.float64) + 1.0
    return float(state_scale) * (
        np.sin(0.017 * indices) + 0.5 * np.cos(0.031 * indices)
    )


def _assembled_elastic_branch_diagnostics(
    assembler: object,
    u_owned: np.ndarray,
    *,
    tiny: float = 1.0e-15,
) -> dict[str, object]:
    """Replay the elastic/plastic trial predicate on every local quadrature point."""
    v_local, _exchange = assembler._owned_to_local(  # noqa: SLF001 - diagnostic runner
        np.asarray(u_owned, dtype=np.float64),
        zero_dirichlet=False,
    )
    elems = np.asarray(assembler.local_data.elems_local_np, dtype=np.int64)
    u_elem = np.asarray(v_local, dtype=np.float64)[elems]
    ux = u_elem[:, 0::3]
    uy = u_elem[:, 1::3]
    uz = u_elem[:, 2::3]
    local = assembler.local_data.local_elem_data
    dphix = np.asarray(local["dphix"], dtype=np.float64)
    dphiy = np.asarray(local["dphiy"], dtype=np.float64)
    dphiz = np.asarray(local["dphiz"], dtype=np.float64)

    e11 = np.einsum("eqn,en->eq", dphix, ux)
    e22 = np.einsum("eqn,en->eq", dphiy, uy)
    e33 = np.einsum("eqn,en->eq", dphiz, uz)
    g12 = np.einsum("eqn,en->eq", dphiy, ux) + np.einsum(
        "eqn,en->eq", dphix, uy
    )
    g23 = np.einsum("eqn,en->eq", dphiz, uy) + np.einsum(
        "eqn,en->eq", dphiy, uz
    )
    g13 = np.einsum("eqn,en->eq", dphiz, ux) + np.einsum(
        "eqn,en->eq", dphix, uz
    )
    strain = np.empty(e11.shape + (3, 3), dtype=np.float64)
    strain[..., 0, 0] = e11
    strain[..., 1, 1] = e22
    strain[..., 2, 2] = e33
    strain[..., 0, 1] = strain[..., 1, 0] = 0.5 * g12
    strain[..., 1, 2] = strain[..., 2, 1] = 0.5 * g23
    strain[..., 0, 2] = strain[..., 2, 0] = 0.5 * g13
    strain += float(tiny) * np.diag([0.0, 1.0, 2.0])
    eigenvalues = np.linalg.eigvalsh(strain)
    eig_3 = eigenvalues[..., 0]
    eig_1 = eigenvalues[..., 2]
    invariant_1 = e11 + e22 + e33
    c_bar = np.asarray(local["c_bar_q"], dtype=np.float64)
    sin_phi = np.asarray(local["sin_phi_q"], dtype=np.float64)
    shear = np.asarray(local["shear_q"], dtype=np.float64)
    lame = np.asarray(local["lame_q"], dtype=np.float64)
    trial_yield = (
        2.0 * shear * ((1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * eig_3)
        + 2.0 * lame * sin_phi * invariant_1
        - c_bar
    )
    stress_scale = np.maximum.reduce(
        (
            np.abs(c_bar),
            np.abs(trial_yield),
            2.0 * np.abs(shear) * np.maximum(np.abs(eig_1), np.abs(eig_3)),
            2.0 * np.abs(lame * sin_phi * invariant_1),
            np.full_like(trial_yield, np.finfo(float).tiny),
        )
    )
    elastic = trial_yield <= 0.0
    return {
        "quadrature_points": int(trial_yield.size),
        "elastic_quadrature_points": int(np.count_nonzero(elastic)),
        "plastic_quadrature_points": int(np.count_nonzero(~elastic)),
        "minimum_trial_yield": float(np.min(trial_yield)),
        "maximum_trial_yield": float(np.max(trial_yield)),
        "minimum_normalized_elastic_margin": float(
            np.min(-trial_yield / stress_scale)
        ),
        "all_quadrature_points_strictly_elastic": bool(np.all(trial_yield < 0.0)),
        "interpretation": (
            "production trial-yield predicate replay over the assembled local case; "
            "not an independent constitutive reference"
        ),
    }


def verify_assembled_route_equivalence(
    *,
    mesh_name: str,
    degree: int,
    lambda_target: float,
    state_scale: float,
    value_atol: float,
    value_rtol: float,
    gradient_atol: float,
    gradient_rtol: float,
    hessian_atol: float,
    hessian_rtol: float,
    symmetry_tolerance: float,
) -> dict[str, object]:
    """Compare all maintained local derivative routes without solving a system."""
    from mpi4py import MPI
    from scipy import sparse

    from src.problems.slope_stability_3d.jax_petsc.reordered_element_assembler import (
        SlopeStability3DReorderedElementAssembler,
    )
    from src.problems.slope_stability_3d.support.mesh import (
        build_same_mesh_lagrange_case_data,
        ownership_block_size_3d,
        select_reordered_perm_3d,
    )

    comm = MPI.COMM_SELF
    case = build_same_mesh_lagrange_case_data(
        str(mesh_name),
        degree=int(degree),
        constraint_variant="glued_bottom",
        build_mode="replicated",
        comm=comm,
    )
    params = dict(case.__dict__)
    params["elem_type"] = f"P{int(degree)}"
    params["element_degree"] = int(degree)
    c_bar_q, sin_phi_q = davis_b_reduction_qp(
        np.asarray(params["c0_q"], dtype=np.float64),
        np.asarray(params["phi_q"], dtype=np.float64),
        np.asarray(params["psi_q"], dtype=np.float64),
        float(lambda_target),
    )
    params["c_bar_q"] = np.asarray(c_bar_q, dtype=np.float64)
    params["sin_phi_q"] = np.asarray(sin_phi_q, dtype=np.float64)
    perm = select_reordered_perm_3d(
        "block_xyz",
        adjacency=case.adjacency,
        coords_all=np.asarray(params["nodes"], dtype=np.float64),
        freedofs=np.asarray(params["freedofs"], dtype=np.int64),
        n_parts=1,
    )
    n_free = int(np.asarray(params["freedofs"], dtype=np.int64).size)
    state = _canonical_assembled_state(n_free, float(state_scale))
    common = {
        "params": params,
        "comm": comm,
        "adjacency": case.adjacency,
        "ksp_type": "cg",
        "pc_type": "none",
        "ksp_max_it": 2,
        "reorder_mode": "block_xyz",
        "perm_override": perm,
        "block_size_override": ownership_block_size_3d(
            np.asarray(params["freedofs"], dtype=np.int64)
        ),
        "distribution_strategy": "overlap_allgather",
        "use_near_nullspace": False,
        "assembly_backend": "coo",
    }
    route_specs = (
        ("element_ad", "element", "element"),
        ("local_sfd", "sfd_local", "element"),
        ("constitutive_ad", "element", "constitutive"),
    )
    snapshots: dict[str, dict[str, object]] = {}
    branch_diagnostics: dict[str, object] | None = None
    for name, local_hessian_mode, autodiff_tangent_mode in route_specs:
        assembler = SlopeStability3DReorderedElementAssembler(
            **common,
            local_hessian_mode=str(local_hessian_mode),
            autodiff_tangent_mode=str(autodiff_tangent_mode),
        )
        vec = assembler.create_vec(state)
        grad = vec.duplicate()
        try:
            energy = float(assembler.energy_fn(vec))
            assembler.gradient_fn(vec, grad)
            gradient = np.asarray(grad.array[:], dtype=np.float64).copy()
            timing = assembler.assemble_hessian(state)
            indptr, indices, values = assembler.A.getValuesCSR()
            indptr = np.asarray(indptr, dtype=np.int64).copy()
            indices = np.asarray(indices, dtype=np.int64).copy()
            values = np.asarray(values, dtype=np.float64).copy()
            matrix = sparse.csr_matrix(
                (values, indices, indptr), shape=(n_free, n_free)
            )
            skew = matrix - matrix.T
            symmetry_defect = float(
                np.linalg.norm(skew.data)
                / max(np.linalg.norm(values), np.finfo(float).tiny)
            )
            if branch_diagnostics is None:
                branch_diagnostics = _assembled_elastic_branch_diagnostics(
                    assembler, state
                )
            snapshots[name] = {
                "energy": energy,
                "gradient": gradient,
                "indptr": indptr,
                "indices": indices,
                "hessian_values": values,
                "gradient_norm": float(np.linalg.norm(gradient)),
                "hessian_frobenius_norm": float(np.linalg.norm(values)),
                "hessian_symmetry_defect": symmetry_defect,
                "assembly_mode": str(timing.get("assembly_mode", "")),
                "hessian_nonzeros": int(values.size),
            }
        finally:
            grad.destroy()
            vec.destroy()
            assembler.cleanup()

    comparisons: list[dict[str, object]] = []
    route_names = [spec[0] for spec in route_specs]
    for left_idx, left_name in enumerate(route_names):
        for right_name in route_names[left_idx + 1 :]:
            left = snapshots[left_name]
            right = snapshots[right_name]
            energy_left = float(left["energy"])
            energy_right = float(right["energy"])
            energy_absolute_error = abs(energy_left - energy_right)
            energy_scale = max(abs(energy_left), abs(energy_right), np.finfo(float).tiny)
            gradient_left = np.asarray(left["gradient"], dtype=np.float64)
            gradient_right = np.asarray(right["gradient"], dtype=np.float64)
            gradient_absolute_error = float(np.linalg.norm(gradient_left - gradient_right))
            gradient_scale = max(
                float(np.linalg.norm(gradient_left)),
                float(np.linalg.norm(gradient_right)),
                np.finfo(float).tiny,
            )
            structure_equal = bool(
                np.array_equal(left["indptr"], right["indptr"])
                and np.array_equal(left["indices"], right["indices"])
            )
            if structure_equal:
                hessian_left = np.asarray(left["hessian_values"], dtype=np.float64)
                hessian_right = np.asarray(right["hessian_values"], dtype=np.float64)
                hessian_difference = hessian_left - hessian_right
                hessian_absolute_error = float(np.linalg.norm(hessian_difference))
                hessian_maximum_entry_error = float(np.max(np.abs(hessian_difference)))
                hessian_scale = max(
                    float(np.linalg.norm(hessian_left)),
                    float(np.linalg.norm(hessian_right)),
                    np.finfo(float).tiny,
                )
                hessian_relative_error: float | None = float(
                    hessian_absolute_error / hessian_scale
                )
            else:
                hessian_absolute_error = None
                hessian_maximum_entry_error = None
                hessian_relative_error = None
            comparison_passed = bool(
                energy_absolute_error
                <= float(value_atol) + float(value_rtol) * energy_scale
                and gradient_absolute_error
                <= float(gradient_atol) + float(gradient_rtol) * gradient_scale
                and structure_equal
                and hessian_relative_error is not None
                and hessian_relative_error <= float(hessian_rtol)
                and hessian_maximum_entry_error is not None
                and hessian_maximum_entry_error <= float(hessian_atol)
            )
            comparisons.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "energy_absolute_error": float(energy_absolute_error),
                    "energy_relative_error": float(energy_absolute_error / energy_scale),
                    "gradient_absolute_error": gradient_absolute_error,
                    "gradient_relative_error": float(
                        gradient_absolute_error / gradient_scale
                    ),
                    "hessian_csr_structure_equal": structure_equal,
                    "hessian_absolute_error": hessian_absolute_error,
                    "hessian_relative_error": hessian_relative_error,
                    "hessian_maximum_entry_error": hessian_maximum_entry_error,
                    "passed": comparison_passed,
                }
            )

    routes = {
        name: {
            key: value
            for key, value in snapshot.items()
            if key not in {"gradient", "indptr", "indices", "hessian_values"}
        }
        for name, snapshot in snapshots.items()
    }
    all_finite = bool(
        all(
            np.isfinite(float(route["energy"]))
            and np.isfinite(float(route["gradient_norm"]))
            and np.isfinite(float(route["hessian_frobenius_norm"]))
            and np.isfinite(float(route["hessian_symmetry_defect"]))
            for route in routes.values()
        )
    )
    symmetry_passed = bool(
        all(
            float(route["hessian_symmetry_defect"]) <= float(symmetry_tolerance)
            for route in routes.values()
        )
    )
    passed = bool(
        all_finite
        and symmetry_passed
        and branch_diagnostics is not None
        and bool(branch_diagnostics["all_quadrature_points_strictly_elastic"])
        and all(bool(comparison["passed"]) for comparison in comparisons)
    )
    return {
        "status": "passed" if passed else "failed",
        "case": {
            "mesh_name": str(mesh_name),
            "degree": int(degree),
            "constraint_variant": "glued_bottom",
            "lambda_target": float(lambda_target),
            "free_dofs": n_free,
            "elements": int(np.asarray(case.elems).shape[0]),
            "state_definition": (
                "u_i = state_scale * (sin(0.017*(i+1)) + "
                "0.5*cos(0.031*(i+1))) in reordered free-DOF coordinates"
            ),
            "state_scale": float(state_scale),
            "state_norm": float(np.linalg.norm(state)),
        },
        "contract": {
            "value_atol": float(value_atol),
            "value_rtol": float(value_rtol),
            "gradient_norm_atol": float(gradient_atol),
            "gradient_norm_rtol": float(gradient_rtol),
            "hessian_maximum_entry_atol": float(hessian_atol),
            "hessian_frobenius_rtol": float(hessian_rtol),
            "hessian_symmetry_tolerance": float(symmetry_tolerance),
            "branch_gate": "every quadrature point must satisfy trial_yield < 0",
        },
        "branch_diagnostics": branch_diagnostics,
        "routes": routes,
        "pairwise_comparisons": comparisons,
        "all_values_finite": all_finite,
        "all_hessians_symmetric_within_tolerance": symmetry_passed,
        "algebraic_scope": {
            "linear_solver_called": False,
            "nonlinear_solver_called": False,
            "ksp_tolerance_used_for_comparison": None,
            "local_sfd_meaning": (
                "sparse forward differentiation assembled from exact JAX JVPs; "
                "no finite-difference step is used"
            ),
            "interpretation": (
                "This is a fixed-state algebraic comparison. It isolates derivative-route "
                "agreement from the KSP-tolerance sensitivity of independently solved states. "
                "Energy and gradient use the shared production kernel; the three Hessians are "
                "generated through distinct production assembly routes."
            ),
        },
    }


def verify_state(
    data: dict[str, np.ndarray],
    *,
    seed: int,
    state_scale: float,
    step_multipliers: np.ndarray,
    fd_step_sizes: np.ndarray,
    fd_gate_index: int,
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed))
    n_dofs = int(np.asarray(data["element_dofs"]).size)
    u = float(state_scale) * rng.standard_normal(n_dofs)
    direction = rng.standard_normal(n_dofs)
    direction /= np.linalg.norm(direction)
    args = _jax_args(data)

    u_jax = jnp.asarray(u, dtype=jnp.float64)
    element_residual = np.asarray(element_residual_3d(u_jax, *args), dtype=np.float64)
    constitutive_residual = np.asarray(
        constitutive_element_residual_3d(u_jax, *args), dtype=np.float64
    )
    element_hessian = np.asarray(element_hessian_3d(u_jax, *args), dtype=np.float64)
    constitutive_hessian = np.asarray(
        constitutive_element_hessian_3d(u_jax, *args), dtype=np.float64
    )

    energy_0 = float(element_energy_3d(u_jax, *args))
    steps = np.asarray(step_multipliers, dtype=np.float64) * max(abs(float(state_scale)), 1.0e-12)
    energy_remainders: list[float] = []
    gradient_remainders: list[float] = []
    for step in steps:
        trial = jnp.asarray(u + float(step) * direction, dtype=jnp.float64)
        energy_trial = float(element_energy_3d(trial, *args))
        residual_trial = np.asarray(element_residual_3d(trial, *args), dtype=np.float64)
        energy_remainders.append(
            float(abs(energy_trial - energy_0 - float(step) * np.dot(element_residual, direction)))
        )
        gradient_remainders.append(
            float(
                np.linalg.norm(
                    residual_trial
                    - element_residual
                    - float(step) * (element_hessian @ direction)
                )
            )
        )

    centered_energy_errors: list[float] = []
    centered_hvp_errors: list[float] = []
    exact_directional_derivative = float(np.dot(element_residual, direction))
    exact_hvp = element_hessian @ direction
    for step in np.asarray(fd_step_sizes, dtype=np.float64):
        plus = jnp.asarray(u + float(step) * direction, dtype=jnp.float64)
        minus = jnp.asarray(u - float(step) * direction, dtype=jnp.float64)
        energy_plus = float(element_energy_3d(plus, *args))
        energy_minus = float(element_energy_3d(minus, *args))
        residual_plus = np.asarray(element_residual_3d(plus, *args), dtype=np.float64)
        residual_minus = np.asarray(element_residual_3d(minus, *args), dtype=np.float64)
        centered_directional = (energy_plus - energy_minus) / (2.0 * float(step))
        centered_hvp = (residual_plus - residual_minus) / (2.0 * float(step))
        centered_energy_errors.append(
            _directional_error(centered_directional, exact_directional_derivative, element_residual)
        )
        centered_hvp_errors.append(_relative_error(centered_hvp, exact_hvp))

    branch_center = _branch_diagnostics(u, data)
    gate_step = float(fd_step_sizes[int(fd_gate_index)])
    branch_plus = _branch_diagnostics(u + gate_step * direction, data)
    branch_minus = _branch_diagnostics(u - gate_step * direction, data)
    branch_stable = bool(
        branch_center["labels"] == branch_plus["labels"] == branch_minus["labels"]
    )

    residual_error = _relative_error(element_residual, constitutive_residual)
    hessian_error = _relative_error(element_hessian, constitutive_hessian)
    symmetry_element = _symmetry_defect(element_hessian)
    symmetry_constitutive = _symmetry_defect(constitutive_hessian)
    energy_slopes = _slopes(steps, np.asarray(energy_remainders))
    gradient_slopes = _slopes(steps, np.asarray(gradient_remainders))

    finite_slopes = [value for value in energy_slopes[:4] + gradient_slopes[:4] if np.isfinite(value)]
    return {
        "seed": int(seed),
        "state_scale": float(state_scale),
        "state_norm": float(np.linalg.norm(u)),
        "energy": energy_0,
        "residual_relative_error": residual_error,
        "hessian_relative_error": hessian_error,
        "element_hessian_symmetry_defect": symmetry_element,
        "constitutive_hessian_symmetry_defect": symmetry_constitutive,
        "step_sizes": [float(value) for value in steps],
        "energy_taylor_remainders": energy_remainders,
        "gradient_taylor_remainders": gradient_remainders,
        "energy_taylor_slopes": energy_slopes,
        "gradient_taylor_slopes": gradient_slopes,
        "centered_fd_step_sizes": [float(value) for value in fd_step_sizes],
        "centered_fd_gate_index": int(fd_gate_index),
        "centered_fd_gate_step": gate_step,
        "centered_fd_energy_directional_relative_errors": centered_energy_errors,
        "centered_fd_hvp_relative_errors": centered_hvp_errors,
        "centered_fd_energy_error_at_gate": float(centered_energy_errors[int(fd_gate_index)]),
        "centered_fd_hvp_error_at_gate": float(centered_hvp_errors[int(fd_gate_index)]),
        "branch_diagnostics": branch_center,
        "branch_stable_across_fd_gate": branch_stable,
        "minimum_early_taylor_slope": float(min(finite_slopes)) if finite_slopes else float("nan"),
        "finite": bool(
            np.all(np.isfinite(element_residual))
            and np.all(np.isfinite(constitutive_residual))
            and np.all(np.isfinite(element_hessian))
            and np.all(np.isfinite(constitutive_hessian))
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    data = _load_element(
        str(args.mesh_name), int(args.degree), int(args.element_index), float(args.lambda_target)
    )
    step_multipliers = np.asarray(
        [float(part) for part in str(args.step_multipliers).split(",") if part.strip()],
        dtype=np.float64,
    )
    if step_multipliers.size < 3 or np.any(step_multipliers <= 0.0):
        raise ValueError("step_multipliers must contain at least three positive values")
    fd_step_sizes = np.asarray(
        [float(part) for part in str(args.fd_step_sizes).split(",") if part.strip()],
        dtype=np.float64,
    )
    if fd_step_sizes.size < 3 or np.any(fd_step_sizes <= 0.0):
        raise ValueError("fd_step_sizes must contain at least three positive values")
    if not 0 <= int(args.fd_gate_index) < int(fd_step_sizes.size):
        raise ValueError("fd_gate_index must select one fd_step_sizes entry")
    records = [
        verify_state(
            data,
            seed=int(args.seed) + idx,
            state_scale=float(args.state_scale),
            step_multipliers=step_multipliers,
            fd_step_sizes=fd_step_sizes,
            fd_gate_index=int(args.fd_gate_index),
        )
        for idx in range(int(args.states))
    ]
    residual_max = max(float(row["residual_relative_error"]) for row in records)
    hessian_max = max(float(row["hessian_relative_error"]) for row in records)
    symmetry_max = max(
        max(
            float(row["element_hessian_symmetry_defect"]),
            float(row["constitutive_hessian_symmetry_defect"]),
        )
        for row in records
    )
    fd_energy_max = max(float(row["centered_fd_energy_error_at_gate"]) for row in records)
    fd_hvp_max = max(float(row["centered_fd_hvp_error_at_gate"]) for row in records)
    branch_stable = all(bool(row["branch_stable_across_fd_gate"]) for row in records)
    fixed_element_passed = bool(
        all(bool(row["finite"]) for row in records)
        and residual_max <= float(args.route_tolerance)
        and hessian_max <= float(args.route_tolerance)
        and symmetry_max <= float(args.symmetry_tolerance)
        and fd_energy_max <= float(args.fd_tolerance)
        and fd_hvp_max <= float(args.fd_tolerance)
        and branch_stable
    )
    assembled_routes = None
    if bool(args.assembled_route_equivalence):
        assembled_routes = verify_assembled_route_equivalence(
            mesh_name=str(args.mesh_name),
            degree=int(args.degree),
            lambda_target=float(args.lambda_target),
            state_scale=float(args.assembled_state_scale),
            value_atol=float(args.assembled_value_atol),
            value_rtol=float(args.assembled_value_rtol),
            gradient_atol=float(args.assembled_gradient_atol),
            gradient_rtol=float(args.assembled_gradient_rtol),
            hessian_atol=float(args.assembled_hessian_atol),
            hessian_rtol=float(args.assembled_hessian_rtol),
            symmetry_tolerance=float(args.assembled_symmetry_tolerance),
        )
    passed = bool(
        fixed_element_passed
        and (
            assembled_routes is None
            or str(assembled_routes["status"]) == "passed"
        )
    )
    return {
        "experiment_id": "EXP-DERIV-001-P3D-FIXED-ELEMENT",
        "status": "passed" if passed else "failed",
        "case": {
            "mesh_name": str(args.mesh_name),
            "degree": int(args.degree),
            "element_index": int(args.element_index),
            "element_dofs": int(np.asarray(data["element_dofs"]).size),
            "quadrature_points": int(np.asarray(data["quad_weight"]).size),
            "lambda_target": float(args.lambda_target),
            "source_hdf5": str(np.asarray(data["source_hdf5"], dtype=object)[0]),
        },
        "contract": {
            "route_relative_tolerance": float(args.route_tolerance),
            "symmetry_tolerance": float(args.symmetry_tolerance),
            "centered_fd_tolerance": float(args.fd_tolerance),
            "centered_fd_gate_index": int(args.fd_gate_index),
            "centered_fd_gate_step": float(fd_step_sizes[int(args.fd_gate_index)]),
            "taylor_slopes": (
                "diagnostic; the production predicate margin and fixed-branch FD gate are "
                "recorded, but this pilot does not establish interface regularity"
            ),
            "branch_gate": "production branch labels must be unchanged at both centered-FD states",
        },
        "summary": {
            "states": len(records),
            "maximum_residual_relative_error": residual_max,
            "maximum_hessian_relative_error": hessian_max,
            "maximum_hessian_symmetry_defect": symmetry_max,
            "maximum_centered_fd_energy_error_at_gate": fd_energy_max,
            "maximum_centered_fd_hvp_error_at_gate": fd_hvp_max,
            "all_states_branch_stable_at_fd_gate": branch_stable,
            "fixed_element_status": "passed" if fixed_element_passed else "failed",
            "assembled_route_equivalence_status": (
                "not_requested"
                if assembled_routes is None
                else str(assembled_routes["status"])
            ),
        },
        "records": records,
        "assembled_route_equivalence": assembled_routes,
        "provenance": {
            "git_commit": _git_commit(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT)),
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
    parser.add_argument("--mesh-name", default="hetero_ssr_L1")
    parser.add_argument("--degree", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--element-index", type=int, default=0)
    parser.add_argument("--lambda-target", type=float, default=1.5)
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--state-scale", type=float, default=1.0e-6)
    parser.add_argument("--step-multipliers", default="1e-1,3e-2,1e-2,3e-3,1e-3,3e-4")
    parser.add_argument("--fd-step-sizes", default="1e-6,3e-7,1e-7,3e-8,1e-8")
    parser.add_argument("--fd-gate-index", type=int, default=2)
    parser.add_argument("--fd-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--route-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--symmetry-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--assembled-route-equivalence",
        action="store_true",
        help=(
            "also compare assembled element-AD, exact local-SFD/JVP, and "
            "constitutive-AD routes at a deterministic fixed elastic state"
        ),
    )
    parser.add_argument("--assembled-state-scale", type=float, default=1.0e-8)
    parser.add_argument("--assembled-value-atol", type=float, default=1.0e-12)
    parser.add_argument("--assembled-value-rtol", type=float, default=1.0e-12)
    parser.add_argument("--assembled-gradient-atol", type=float, default=1.0e-10)
    parser.add_argument("--assembled-gradient-rtol", type=float, default=1.0e-12)
    parser.add_argument("--assembled-hessian-atol", type=float, default=1.0e-8)
    parser.add_argument("--assembled-hessian-rtol", type=float, default=1.0e-12)
    parser.add_argument("--assembled-symmetry-tolerance", type=float, default=1.0e-12)
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

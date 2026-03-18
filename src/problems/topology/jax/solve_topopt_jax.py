#!/usr/bin/env python3
"""Pure-JAX staggered SIMP topology optimisation on a 2D cantilever."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.problems.topology.support.policy import (
    constitutive_plane_stress,
    message_is_converged,
    relative_state_change,
    staircase_p_step,
)
from src.core.serial.jax_diff import EnergyDerivator
from src.core.serial.minimizers import gradient_descent, newton
from src.core.serial.sparse_solvers import HessSolverGenerator
from src.problems.topology.jax.jax_energy import (
    compliance,
    design_energy,
    frozen_design_density,
    material_scale_from_design,
    mechanics_energy,
    theta_from_latent,
    volume_fraction,
)
from src.problems.topology.jax.mesh import CantileverTopologyMesh


LINESEARCH_INTERVAL = (-0.25, 1.5)
LINESEARCH_TOL = 1e-1
DESIGN_SOLVER = "direct"
USE_TRUST_REGION = True
TRUST_RADIUS_INIT = 1.0
TRUST_RADIUS_MIN = 1e-8
TRUST_RADIUS_MAX = 1e6
TRUST_SHRINK = 0.5
TRUST_EXPAND = 1.5
TRUST_ETA_SHRINK = 0.05
TRUST_ETA_EXPAND = 0.75
TRUST_MAX_REJECT = 6
TRUST_SUBPROBLEM_LINE_SEARCH = True


def _summarise_newton_work(result: dict) -> dict:
    history = result.get("history", [])
    linear_timing = result.get("linear_timing", [])
    accepted = [row for row in history if bool(row.get("accepted_step", False))]
    alpha_values = [float(row.get("alpha", np.nan)) for row in accepted]
    linear_iters = int(sum(int(row.get("ksp_its", 0)) for row in linear_timing))
    linear_time = float(sum(float(row.get("linear_total_time", 0.0)) for row in linear_timing))
    maxit_hits = int(sum(bool(row.get("hit_ksp_max_it", False)) for row in linear_timing))
    return {
        "linear_iters": linear_iters,
        "linear_time": linear_time,
        "ksp_maxit_hits": maxit_hits,
        "accepted_steps": int(len(accepted)),
        "alpha_mean": float(np.mean(alpha_values)) if alpha_values else np.nan,
        "alpha_min": float(np.min(alpha_values)) if alpha_values else np.nan,
        "alpha_max": float(np.max(alpha_values)) if alpha_values else np.nan,
        "last_alpha": float(history[-1]["alpha"]) if history else np.nan,
        "ls_evals": int(sum(int(row.get("ls_evals", 0)) for row in history)),
    }


def _run_newton(
    fun,
    grad,
    hess_solver,
    x0,
    *,
    maxit: int,
    tolf: float,
    tolg: float,
    linesearch_tol: float,
    verbose: bool,
    linesearch_interval: tuple[float, float],
    use_trust_region: bool,
):
    return newton(
        fun,
        grad,
        hess_solver,
        x0,
        maxit=maxit,
        tolf=tolf,
        tolg=tolg,
        linesearch_tol=linesearch_tol,
        linesearch_interval=linesearch_interval,
        verbose=verbose,
        trust_region=use_trust_region,
        trust_radius_init=TRUST_RADIUS_INIT,
        trust_radius_min=TRUST_RADIUS_MIN,
        trust_radius_max=TRUST_RADIUS_MAX,
        trust_shrink=TRUST_SHRINK,
        trust_expand=TRUST_EXPAND,
        trust_eta_shrink=TRUST_ETA_SHRINK,
        trust_eta_expand=TRUST_ETA_EXPAND,
        trust_max_reject=TRUST_MAX_REJECT,
        trust_subproblem_line_search=TRUST_SUBPROBLEM_LINE_SEARCH,
        save_history=True,
        save_linear_timing=True,
    )


def _run_gradient_descent(
    fun,
    grad,
    x0,
    *,
    maxit: int,
    tolf: float,
    tolg: float,
    linesearch_tol: float,
    verbose: bool,
    line_search: str,
    adaptive_alpha0: float,
    adaptive_window_scale: float,
    adaptive_nonnegative: bool,
):
    return gradient_descent(
        fun,
        grad,
        x0,
        maxit=maxit,
        tolf=tolf,
        tolg=tolg,
        linesearch_tol=linesearch_tol,
        line_search=line_search,
        adaptive_alpha0=adaptive_alpha0,
        adaptive_window_scale=adaptive_window_scale,
        adaptive_nonnegative=adaptive_nonnegative,
        linesearch_interval=(0.0, 2.0),
        verbose=verbose,
        save_history=True,
    )


def run_topology_optimisation(
    *,
    nx: int = 192,
    ny: int = 96,
    length: float = 2.0,
    height: float = 1.0,
    traction: float = 1.0,
    load_fraction: float = 0.2,
    fixed_pad_cells: int = 16,
    load_pad_cells: int = 16,
    volume_fraction_target: float = 0.4,
    theta_min: float = 1e-3,
    solid_latent: float = 10.0,
    young: float = 1.0,
    poisson: float = 0.3,
    alpha_reg: float = 5e-3,
    ell_pf: float = 0.08,
    mu_move: float = 0.01,
    lambda_init: float = 0.0,
    beta_lambda: float = 12.0,
    volume_penalty: float = 10.0,
    p_start: float = 1.0,
    p_max: float = 4.0,
    p_increment: float = 0.5,
    continuation_interval: int = 20,
    outer_maxit: int = 180,
    outer_tol: float = 2e-2,
    volume_tol: float = 1e-3,
    mechanics_maxit: int = 200,
    design_maxit: int = 400,
    tolf: float = 1e-6,
    tolg: float = 1e-3,
    ksp_rtol: float = 1e-2,
    ksp_max_it: int = 80,
    linesearch_tol: float = LINESEARCH_TOL,
    linesearch_interval: tuple[float, float] = LINESEARCH_INTERVAL,
    use_trust_region: bool = USE_TRUST_REGION,
    mechanics_solver_type: str = "petsc_gamg",
    mechanics_ksp_type: str = "cg",
    mechanics_use_near_nullspace: bool = True,
    design_solver_type: str = DESIGN_SOLVER,
    design_nonlinear_method: str = "gd_golden_adaptive",
    design_gd_adaptive_alpha0: float = 1.0,
    design_gd_adaptive_window_scale: float = 2.0,
    design_gd_adaptive_nonnegative: bool = True,
    design_ksp_type: str = "gmres",
    design_pc_type: str = "ilu",
    design_norm_type: str = "unpreconditioned",
    design_ksp_rtol: float | None = None,
    design_ksp_max_it: int | None = None,
    design_pc_factor_levels: int = 1,
    gamg_threshold: float = 0.05,
    gamg_agg_nsmooths: int = 1,
    gamg_set_coordinates: bool = False,
    save_outer_state_history: bool = False,
    capture_last_design_snapshot: bool = False,
    capture_design_snapshot_iters: tuple[int, ...] | None = None,
    capture_design_solver_history_iters: tuple[int, ...] | None = None,
    verbose: bool = False,
) -> tuple[dict, dict]:
    total_start = time.perf_counter()
    setup_start = time.perf_counter()

    mesh = CantileverTopologyMesh(
        nx=nx,
        ny=ny,
        length=length,
        height=height,
        traction=traction,
        load_fraction=load_fraction,
        fixed_pad_cells=fixed_pad_cells,
        load_pad_cells=load_pad_cells,
    )
    constitutive = constitutive_plane_stress(young, poisson)
    z_template, z_full, z_free = mesh.build_design_state(
        target_volume_fraction=volume_fraction_target,
        theta_min=theta_min,
        solid_latent=solid_latent,
    )
    u_free = np.zeros(mesh.freedofs_u.size, dtype=np.float64)

    scalar_elems_jax = jnp.asarray(mesh.scalar_elems, dtype=jnp.int32)
    vector_elems_jax = jnp.asarray(mesh.vector_elems, dtype=jnp.int32)
    elem_grad_phi_jax = jnp.asarray(mesh.elem_grad_phi, dtype=jnp.float64)
    elem_B_jax = jnp.asarray(mesh.elem_B, dtype=jnp.float64)
    elem_area_jax = jnp.asarray(mesh.elem_area, dtype=jnp.float64)
    constitutive_jax = jnp.asarray(constitutive, dtype=jnp.float64)
    force_jax = jnp.asarray(mesh.force, dtype=jnp.float64)
    nodal_weights_jax = jnp.asarray(mesh.nodal_volume_weights, dtype=jnp.float64)
    u0_jax = jnp.asarray(mesh.u_0, dtype=jnp.float64)
    z_template_jax = jnp.asarray(z_template, dtype=jnp.float64)
    freedofs_u_jax = jnp.asarray(mesh.freedofs_u, dtype=jnp.int32)
    freedofs_z_jax = jnp.asarray(mesh.freedofs_z, dtype=jnp.int32)

    p_penal = float(p_start)
    lambda_volume = float(lambda_init)
    design_linear_tol = float(ksp_rtol if design_ksp_rtol is None else design_ksp_rtol)
    design_linear_maxit = int(ksp_max_it if design_ksp_max_it is None else design_ksp_max_it)
    design_adaptive_alpha = float(max(abs(design_gd_adaptive_alpha0), linesearch_tol))

    material_scale0 = np.asarray(
        material_scale_from_design(
            jnp.asarray(z_full, dtype=jnp.float64),
            scalar_elems_jax,
            theta_min,
            p_penal,
        )
    )
    mechanics_params = {
        "u_0": u0_jax,
        "freedofs": freedofs_u_jax,
        "elems": vector_elems_jax,
        "elem_B": elem_B_jax,
        "elem_area": elem_area_jax,
        "material_scale": jnp.asarray(material_scale0, dtype=jnp.float64),
        "constitutive": constitutive_jax,
        "force": force_jax,
    }
    mechanics_drv = EnergyDerivator(
        mechanics_energy,
        mechanics_params,
        mesh.adjacency_u,
        jnp.asarray(u_free, dtype=jnp.float64),
    )
    mechanics_F, mechanics_dF, mechanics_ddF = mechanics_drv.get_derivatives()
    mechanics_hess_solver = HessSolverGenerator(
        ddf=mechanics_ddF,
        solver_type=mechanics_solver_type,
        elastic_kernel=mesh.elastic_kernel,
        verbose=verbose,
        tol=ksp_rtol,
        maxiter=ksp_max_it,
        coordinates=mesh.coords[mesh.freedofs_u[0::2] // 2],
        block_size=2,
        ksp_type=mechanics_ksp_type,
        gamg_threshold=gamg_threshold,
        gamg_agg_nsmooths=gamg_agg_nsmooths,
        gamg_set_coordinates=gamg_set_coordinates,
        use_near_nullspace=mechanics_use_near_nullspace,
    )

    design_params = {
        "z_0": z_template_jax,
        "freedofs": freedofs_z_jax,
        "elems": scalar_elems_jax,
        "elem_grad_phi": elem_grad_phi_jax,
        "elem_area": elem_area_jax,
        "e_frozen": jnp.full(mesh.scalar_elems.shape[0], 1e-6, dtype=jnp.float64),
        "z_old_full": jnp.asarray(z_full, dtype=jnp.float64),
        "lambda_volume": float(lambda_volume),
        "alpha_reg": float(alpha_reg),
        "ell_pf": float(ell_pf),
        "mu_move": float(mu_move),
        "theta_min": float(theta_min),
        "p_penal": float(p_penal),
    }
    design_drv = EnergyDerivator(
        design_energy,
        design_params,
        mesh.adjacency_z,
        jnp.asarray(z_free, dtype=jnp.float64),
    )
    design_F, design_dF, design_ddF = design_drv.get_derivatives()
    design_hess_solver = None
    if design_nonlinear_method == "newton_trust":
        design_hess_solver = HessSolverGenerator(
            ddf=design_ddF,
            solver_type=design_solver_type,
            verbose=verbose,
            tol=design_linear_tol,
            maxiter=design_linear_maxit,
            ksp_type=design_ksp_type,
            pc_type=design_pc_type,
            norm_type=design_norm_type,
            mat_symmetric=True,
            petsc_options={"pc_factor_levels": int(design_pc_factor_levels)},
        )

    setup_time = time.perf_counter() - setup_start
    history = []
    status = "completed"
    outer_converged = False
    prev_compliance = np.nan
    prev_theta_state = None
    last_design_snapshot = None
    capture_snapshot_set = set() if capture_design_snapshot_iters is None else set(capture_design_snapshot_iters)
    capture_design_solver_history_set = (
        set() if capture_design_solver_history_iters is None else set(capture_design_solver_history_iters)
    )
    design_snapshots = {}
    design_solver_histories = {}

    final_theta = np.asarray(theta_from_latent(jnp.asarray(z_full, dtype=jnp.float64), theta_min))
    final_u = mesh.expand_u(u_free)
    theta_history = []
    if save_outer_state_history:
        theta_history.append(
            {
                "outer_iter": 0,
                "p_penal": float(p_penal),
                "theta": final_theta.copy(),
                "volume_fraction": float(
                    volume_fraction(
                        jnp.asarray(final_theta, dtype=jnp.float64),
                        nodal_weights_jax,
                        mesh.domain_area,
                    )
                ),
            }
        )

    for outer_it in range(1, outer_maxit + 1):
        theta_before = np.asarray(theta_from_latent(jnp.asarray(z_full, dtype=jnp.float64), theta_min))
        theta_state_change = relative_state_change(
            theta_before,
            prev_theta_state,
            mesh.freedofs_z,
        )
        volume_before = float(
            volume_fraction(
                jnp.asarray(theta_before, dtype=jnp.float64),
                nodal_weights_jax,
                mesh.domain_area,
            )
        )
        volume_residual_before = float(volume_before - volume_fraction_target)

        material_scale = np.asarray(
            material_scale_from_design(
                jnp.asarray(z_full, dtype=jnp.float64),
                scalar_elems_jax,
                theta_min,
                p_penal,
            )
        )
        mechanics_params["material_scale"] = jnp.asarray(material_scale, dtype=jnp.float64)
        mechanics_res = _run_newton(
            mechanics_F,
            mechanics_dF,
            mechanics_hess_solver,
            u_free,
            maxit=mechanics_maxit,
            tolf=tolf,
            tolg=tolg,
            linesearch_tol=linesearch_tol,
            verbose=verbose,
            linesearch_interval=linesearch_interval,
            use_trust_region=use_trust_region,
        )
        u_free = np.asarray(mechanics_res["x"], dtype=np.float64)
        u_full = mesh.expand_u(u_free)
        mechanics_work = _summarise_newton_work(mechanics_res)

        if not message_is_converged(str(mechanics_res["message"])):
            status = "failed_mechanics"
            final_u = u_full
            final_theta = theta_before
            history.append(
                {
                    "outer_iter": outer_it,
                    "p_penal": float(p_penal),
                    "mechanics_iters": int(mechanics_res["nit"]),
                    "mechanics_linear_iters": int(mechanics_work["linear_iters"]),
                    "mechanics_linear_time": float(mechanics_work["linear_time"]),
                    "mechanics_ksp_maxit_hits": int(mechanics_work["ksp_maxit_hits"]),
                    "mechanics_alpha_mean": float(mechanics_work["alpha_mean"]),
                    "mechanics_alpha_last": float(mechanics_work["last_alpha"]),
                    "mechanics_message": str(mechanics_res["message"]),
                    "design_message": "not_run",
                }
            )
            break

        e_frozen = np.asarray(
            frozen_design_density(
                jnp.asarray(u_full, dtype=jnp.float64),
                jnp.asarray(z_full, dtype=jnp.float64),
                scalar_elems_jax,
                vector_elems_jax,
                elem_B_jax,
                constitutive_jax,
                theta_min,
                p_penal,
            )
        )
        theta_elem_before = theta_before[mesh.scalar_elems].mean(axis=1)
        sensitivity_scale = p_penal * e_frozen / np.maximum(theta_elem_before, theta_min) ** (p_penal + 1.0)
        lambda_reference = float(
            np.quantile(
                sensitivity_scale,
                float(np.clip(1.0 - volume_fraction_target, 0.0, 1.0)),
            )
        )
        lambda_penalty = float(volume_penalty * volume_residual_before)
        lambda_effective = float(lambda_reference + lambda_volume + lambda_penalty)

        design_params["e_frozen"] = jnp.asarray(e_frozen, dtype=jnp.float64)
        design_params["z_old_full"] = jnp.asarray(z_full, dtype=jnp.float64)
        design_params["lambda_volume"] = lambda_effective
        design_params["p_penal"] = float(p_penal)

        if design_nonlinear_method == "newton_trust":
            design_res = _run_newton(
                design_F,
                design_dF,
                design_hess_solver,
                z_free,
                maxit=design_maxit,
                tolf=tolf,
                tolg=tolg,
                linesearch_tol=linesearch_tol,
                verbose=verbose,
                linesearch_interval=linesearch_interval,
                use_trust_region=use_trust_region,
            )
            if outer_it in capture_design_solver_history_set:
                design_solver_histories[int(outer_it)] = {
                    "outer_iter": int(outer_it),
                    "method": str(design_nonlinear_method),
                    "history": list(design_res.get("history", [])),
                }
        elif design_nonlinear_method == "gd_golden_adaptive":
            design_adaptive_alpha_before = float(design_adaptive_alpha)
            design_res = _run_gradient_descent(
                design_F,
                design_dF,
                z_free,
                maxit=design_maxit,
                tolf=tolf,
                tolg=tolg,
                linesearch_tol=linesearch_tol,
                verbose=verbose,
                line_search="golden_adaptive",
                adaptive_alpha0=design_adaptive_alpha,
                adaptive_window_scale=design_gd_adaptive_window_scale,
                adaptive_nonnegative=design_gd_adaptive_nonnegative,
            )
            design_adaptive_alpha_after = float(
                max(abs(design_res.get("last_alpha_abs", design_adaptive_alpha)), linesearch_tol)
            )
            if outer_it in capture_design_solver_history_set:
                design_solver_histories[int(outer_it)] = {
                    "outer_iter": int(outer_it),
                    "method": str(design_nonlinear_method),
                    "adaptive_alpha_before": float(design_adaptive_alpha_before),
                    "adaptive_alpha_after": float(design_adaptive_alpha_after),
                    "adaptive_window_scale": float(design_gd_adaptive_window_scale),
                    "adaptive_nonnegative": bool(design_gd_adaptive_nonnegative),
                    "history": list(design_res.get("history", [])),
                }
            design_adaptive_alpha = float(design_adaptive_alpha_after)
        else:
            raise ValueError(f"Unknown design_nonlinear_method {design_nonlinear_method!r}")
        z_free_new = np.asarray(design_res["x"], dtype=np.float64)
        z_full_new = mesh.expand_z(z_free_new, z_template)
        theta_after = np.asarray(theta_from_latent(jnp.asarray(z_full_new, dtype=jnp.float64), theta_min))
        design_work = _summarise_newton_work(design_res)

        final_u = u_full
        final_theta = theta_after
        if not message_is_converged(str(design_res["message"])):
            status = "failed_design"
            z_full = z_full_new
            history.append(
                {
                    "outer_iter": outer_it,
                    "p_penal": float(p_penal),
                    "lambda_correction": float(lambda_volume),
                    "lambda_penalty": lambda_penalty,
                    "lambda_reference": lambda_reference,
                    "lambda_effective": lambda_effective,
                    "volume_fraction_before": volume_before,
                    "volume_residual_before": volume_residual_before,
                    "theta_state_change": theta_state_change,
                    "mechanics_iters": int(mechanics_res["nit"]),
                    "design_iters": int(design_res["nit"]),
                    "mechanics_linear_iters": int(mechanics_work["linear_iters"]),
                    "design_linear_iters": int(design_work["linear_iters"]),
                    "mechanics_linear_time": float(mechanics_work["linear_time"]),
                    "design_linear_time": float(design_work["linear_time"]),
                    "mechanics_ksp_maxit_hits": int(mechanics_work["ksp_maxit_hits"]),
                    "design_ksp_maxit_hits": int(design_work["ksp_maxit_hits"]),
                    "mechanics_alpha_mean": float(mechanics_work["alpha_mean"]),
                    "design_alpha_mean": float(design_work["alpha_mean"]),
                    "mechanics_alpha_last": float(mechanics_work["last_alpha"]),
                    "design_alpha_last": float(design_work["last_alpha"]),
                    "mechanics_ls_evals": int(mechanics_work["ls_evals"]),
                    "design_ls_evals": int(design_work["ls_evals"]),
                    "mechanics_energy": float(mechanics_res["fun"]),
                    "design_energy": float(design_res["fun"]),
                    "mechanics_message": str(mechanics_res["message"]),
                    "design_message": str(design_res["message"]),
                }
            )
            break

        compliance_value = float(compliance(jnp.asarray(u_full, dtype=jnp.float64), force_jax))
        volume_value = float(
            volume_fraction(
                jnp.asarray(theta_after, dtype=jnp.float64),
                nodal_weights_jax,
                mesh.domain_area,
            )
        )
        design_change = float(
            np.linalg.norm(theta_after[mesh.freedofs_z] - theta_before[mesh.freedofs_z])
            / max(1.0, np.linalg.norm(theta_before[mesh.freedofs_z]))
        )
        compliance_change = (
            float(abs(compliance_value - prev_compliance) / max(1.0, abs(prev_compliance)))
            if np.isfinite(prev_compliance)
            else np.inf
        )
        volume_residual = float(volume_value - volume_fraction_target)
        p_step = staircase_p_step(
            p_penal,
            p_max=p_max,
            p_increment=p_increment,
            continuation_interval=continuation_interval,
            outer_it=outer_it,
        )

        history.append(
            {
                "outer_iter": outer_it,
                "p_penal": float(p_penal),
                "p_step": float(p_step),
                "lambda_correction": float(lambda_volume),
                "lambda_penalty": lambda_penalty,
                "lambda_reference": lambda_reference,
                "lambda_effective": lambda_effective,
                "compliance": compliance_value,
                "volume_fraction_before": volume_before,
                "volume_residual_before": volume_residual_before,
                "volume_fraction": volume_value,
                "volume_residual": volume_residual,
                "theta_state_change": theta_state_change,
                "design_change": design_change,
                "compliance_change": compliance_change,
                "mechanics_iters": int(mechanics_res["nit"]),
                "design_iters": int(design_res["nit"]),
                "mechanics_linear_iters": int(mechanics_work["linear_iters"]),
                "design_linear_iters": int(design_work["linear_iters"]),
                "mechanics_linear_time": float(mechanics_work["linear_time"]),
                "design_linear_time": float(design_work["linear_time"]),
                "mechanics_ksp_maxit_hits": int(mechanics_work["ksp_maxit_hits"]),
                "design_ksp_maxit_hits": int(design_work["ksp_maxit_hits"]),
                "mechanics_alpha_mean": float(mechanics_work["alpha_mean"]),
                "design_alpha_mean": float(design_work["alpha_mean"]),
                "mechanics_alpha_last": float(mechanics_work["last_alpha"]),
                "design_alpha_last": float(design_work["last_alpha"]),
                "mechanics_ls_evals": int(mechanics_work["ls_evals"]),
                "design_ls_evals": int(design_work["ls_evals"]),
                "mechanics_energy": float(mechanics_res["fun"]),
                "design_energy": float(design_res["fun"]),
                "mechanics_message": str(mechanics_res["message"]),
                "design_message": str(design_res["message"]),
            }
        )

        if capture_last_design_snapshot:
            last_design_snapshot = {
                "outer_iter": int(outer_it),
                "p_penal": float(p_penal),
                "lambda_effective": float(lambda_effective),
                "lambda_reference": float(lambda_reference),
                "lambda_penalty": float(lambda_penalty),
                "lambda_correction": float(lambda_volume),
                "z_old_full": np.asarray(z_full, dtype=np.float64).copy(),
                "z_full": np.asarray(z_full_new, dtype=np.float64).copy(),
                "e_frozen": np.asarray(e_frozen, dtype=np.float64).copy(),
                "u_full": np.asarray(u_full, dtype=np.float64).copy(),
            }
        if outer_it in capture_snapshot_set:
            design_snapshots[int(outer_it)] = {
                "outer_iter": int(outer_it),
                "p_penal": float(p_penal),
                "lambda_effective": float(lambda_effective),
                "lambda_reference": float(lambda_reference),
                "lambda_penalty": float(lambda_penalty),
                "lambda_correction": float(lambda_volume),
                "volume_fraction": float(volume_value),
                "volume_residual": float(volume_residual),
                "compliance": float(compliance_value),
                "z_old_full": np.asarray(z_full, dtype=np.float64).copy(),
                "z_full": np.asarray(z_full_new, dtype=np.float64).copy(),
                "e_frozen": np.asarray(e_frozen, dtype=np.float64).copy(),
                "u_full": np.asarray(u_full, dtype=np.float64).copy(),
            }

        z_free = z_free_new
        z_full = z_full_new
        prev_compliance = compliance_value
        prev_theta_state = theta_before.copy()
        if save_outer_state_history:
            theta_history.append(
                {
                    "outer_iter": outer_it,
                    "p_penal": float(p_penal),
                    "theta": theta_after.copy(),
                    "volume_fraction": volume_value,
                    "compliance": compliance_value,
                }
            )

        lambda_volume += beta_lambda * max(1.0, lambda_reference) * volume_residual
        if (
            p_penal >= p_max - 1e-12
            and abs(volume_residual) <= volume_tol
            and design_change <= outer_tol
            and compliance_change <= outer_tol
        ):
            outer_converged = True
            break

        p_penal = min(p_max, p_penal + p_step)

    if status == "completed" and not outer_converged and len(history) >= outer_maxit:
        status = "max_outer_iterations"

    if status != "failed_mechanics":
        final_theta = np.asarray(theta_from_latent(jnp.asarray(z_full, dtype=jnp.float64), theta_min))
        final_material_scale = np.asarray(
            material_scale_from_design(
                jnp.asarray(z_full, dtype=jnp.float64),
                scalar_elems_jax,
                theta_min,
                p_penal,
            )
        )
        mechanics_params["material_scale"] = jnp.asarray(final_material_scale, dtype=jnp.float64)
        final_mechanics_res = _run_newton(
            mechanics_F,
            mechanics_dF,
            mechanics_hess_solver,
            u_free,
            maxit=mechanics_maxit,
            tolf=tolf,
            tolg=tolg,
            linesearch_tol=linesearch_tol,
            verbose=verbose,
            linesearch_interval=linesearch_interval,
            use_trust_region=use_trust_region,
        )
        if message_is_converged(str(final_mechanics_res["message"])):
            u_free = np.asarray(final_mechanics_res["x"], dtype=np.float64)
            final_u = mesh.expand_u(u_free)
        else:
            status = "failed_final_mechanics"
            final_u = mesh.expand_u(u_free)

    total_time = time.perf_counter() - total_start
    final_volume = float(
        volume_fraction(jnp.asarray(final_theta, dtype=jnp.float64), nodal_weights_jax, mesh.domain_area)
    )
    final_compliance = float(compliance(jnp.asarray(final_u, dtype=jnp.float64), force_jax))

    result = {
        "solver": "pure_jax_staggered_topopt",
        "backend": "serial",
        "result": status,
        "time": float(total_time),
        "setup_time": float(setup_time),
        "nprocs": 1,
        "mesh": {
            "nx": int(nx),
            "ny": int(ny),
            "nodes": int(mesh.n_nodes),
            "elements": int(mesh.scalar_elems.shape[0]),
            "displacement_free_dofs": int(mesh.freedofs_u.size),
            "design_free_dofs": int(mesh.freedofs_z.size),
        },
        "parameters": {
            "length": float(length),
            "height": float(height),
            "traction": float(traction),
            "load_fraction": float(load_fraction),
            "fixed_pad_cells": int(fixed_pad_cells),
            "load_pad_cells": int(load_pad_cells),
            "volume_fraction_target": float(volume_fraction_target),
            "theta_min": float(theta_min),
            "solid_latent": float(solid_latent),
            "young": float(young),
            "poisson": float(poisson),
            "alpha_reg": float(alpha_reg),
            "ell_pf": float(ell_pf),
            "mu_move": float(mu_move),
            "lambda_init": float(lambda_init),
            "beta_lambda": float(beta_lambda),
            "volume_penalty": float(volume_penalty),
            "p_start": float(p_start),
            "p_max": float(p_max),
            "p_increment": float(p_increment),
            "continuation_interval": int(continuation_interval),
            "outer_maxit": int(outer_maxit),
            "outer_tol": float(outer_tol),
            "volume_tol": float(volume_tol),
        },
        "solver_options": {
            "mechanics_solver_type": str(mechanics_solver_type),
            "mechanics_ksp_type": str(mechanics_ksp_type),
            "mechanics_use_near_nullspace": bool(mechanics_use_near_nullspace),
            "gamg_threshold": float(gamg_threshold),
            "gamg_agg_nsmooths": int(gamg_agg_nsmooths),
            "gamg_set_coordinates": bool(gamg_set_coordinates),
            "mechanics_maxit": int(mechanics_maxit),
            "design_maxit": int(design_maxit),
            "tolf": float(tolf),
            "tolg": float(tolg),
            "linesearch_interval": [float(linesearch_interval[0]), float(linesearch_interval[1])],
            "linesearch_tol": float(linesearch_tol),
            "design_solver": str(design_solver_type),
            "design_nonlinear_method": str(design_nonlinear_method),
            "design_gd_adaptive_alpha0": float(design_gd_adaptive_alpha0),
            "design_gd_adaptive_window_scale": float(design_gd_adaptive_window_scale),
            "design_gd_adaptive_nonnegative": bool(design_gd_adaptive_nonnegative),
            "design_ksp_type": str(design_ksp_type),
            "design_pc_type": str(design_pc_type),
            "design_norm_type": str(design_norm_type),
            "design_ksp_rtol": float(design_linear_tol),
            "design_ksp_max_it": int(design_linear_maxit),
            "design_pc_factor_levels": int(design_pc_factor_levels),
            "use_trust_region": bool(use_trust_region),
            "trust_radius_init": float(TRUST_RADIUS_INIT),
            "trust_radius_min": float(TRUST_RADIUS_MIN),
            "trust_radius_max": float(TRUST_RADIUS_MAX),
            "trust_shrink": float(TRUST_SHRINK),
            "trust_expand": float(TRUST_EXPAND),
            "trust_eta_shrink": float(TRUST_ETA_SHRINK),
            "trust_eta_expand": float(TRUST_ETA_EXPAND),
            "trust_max_reject": int(TRUST_MAX_REJECT),
            "trust_subproblem_line_search": bool(TRUST_SUBPROBLEM_LINE_SEARCH),
            "ksp_rtol": float(ksp_rtol),
            "ksp_max_it": int(ksp_max_it),
        },
        "final_metrics": {
            "outer_iterations": int(len(history)),
            "final_volume_fraction": final_volume,
            "final_compliance": final_compliance,
            "final_theta_min": float(np.min(final_theta)),
            "final_theta_max": float(np.max(final_theta)),
            "final_p_penal": float(p_penal),
        },
        "history": history,
        "jax_setup_timing": {
            "mechanics": mechanics_drv.timings,
            "design": design_drv.timings,
        },
        "jax_version": jax.__version__,
    }
    state = {
        "coords": mesh.coords,
        "triangles": mesh.scalar_elems,
        "theta": final_theta,
        "u": final_u,
        "z": z_full,
    }
    if save_outer_state_history:
        state["theta_history"] = theta_history
    if capture_last_design_snapshot:
        state["last_design_snapshot"] = last_design_snapshot
    if capture_snapshot_set:
        state["design_snapshots"] = design_snapshots
    if capture_design_solver_history_set:
        state["design_solver_histories"] = design_solver_histories
    return result, state


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=192)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--length", type=float, default=2.0)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--traction", type=float, default=1.0)
    parser.add_argument("--load_fraction", type=float, default=0.2)
    parser.add_argument("--fixed_pad_cells", type=int, default=16)
    parser.add_argument("--load_pad_cells", type=int, default=16)
    parser.add_argument("--volume_fraction_target", type=float, default=0.4)
    parser.add_argument("--theta_min", type=float, default=1e-3)
    parser.add_argument("--solid_latent", type=float, default=10.0)
    parser.add_argument("--young", type=float, default=1.0)
    parser.add_argument("--poisson", type=float, default=0.3)
    parser.add_argument("--alpha_reg", type=float, default=5e-3)
    parser.add_argument("--ell_pf", type=float, default=0.08)
    parser.add_argument("--mu_move", type=float, default=0.01)
    parser.add_argument("--lambda_init", type=float, default=0.0)
    parser.add_argument("--beta_lambda", type=float, default=12.0)
    parser.add_argument("--volume_penalty", type=float, default=10.0)
    parser.add_argument("--p_start", type=float, default=1.0)
    parser.add_argument("--p_max", type=float, default=4.0)
    parser.add_argument("--p_increment", type=float, default=0.5)
    parser.add_argument("--continuation_interval", type=int, default=20)
    parser.add_argument("--outer_maxit", type=int, default=180)
    parser.add_argument("--outer_tol", type=float, default=2e-2)
    parser.add_argument("--volume_tol", type=float, default=1e-3)
    parser.add_argument("--mechanics_maxit", type=int, default=200)
    parser.add_argument("--design_maxit", type=int, default=400)
    parser.add_argument("--tolf", type=float, default=1e-6)
    parser.add_argument("--tolg", type=float, default=1e-3)
    parser.add_argument("--ksp_rtol", type=float, default=1e-2)
    parser.add_argument("--ksp_max_it", type=int, default=80)
    parser.add_argument("--linesearch_tol", type=float, default=LINESEARCH_TOL)
    parser.add_argument("--linesearch_a", type=float, default=LINESEARCH_INTERVAL[0])
    parser.add_argument("--linesearch_b", type=float, default=LINESEARCH_INTERVAL[1])
    parser.add_argument("--use_trust_region", action=argparse.BooleanOptionalAction, default=USE_TRUST_REGION)
    parser.add_argument("--mechanics_solver_type", choices=("amg", "petsc_gamg"), default="petsc_gamg")
    parser.add_argument("--mechanics_ksp_type", type=str, default="cg")
    parser.add_argument("--mechanics_use_near_nullspace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--design_solver_type", choices=("direct", "amg", "petsc_gamg"), default=DESIGN_SOLVER)
    parser.add_argument(
        "--design_nonlinear_method",
        choices=("newton_trust", "gd_golden_adaptive"),
        default="gd_golden_adaptive",
    )
    parser.add_argument("--design_gd_adaptive_alpha0", type=float, default=1.0)
    parser.add_argument("--design_gd_adaptive_window_scale", type=float, default=2.0)
    parser.add_argument(
        "--design_gd_adaptive_nonnegative",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--design_ksp_type", type=str, default="gmres")
    parser.add_argument("--design_pc_type", type=str, default="ilu")
    parser.add_argument("--design_norm_type", choices=("default", "unpreconditioned"), default="unpreconditioned")
    parser.add_argument("--design_ksp_rtol", type=float, default=None)
    parser.add_argument("--design_ksp_max_it", type=int, default=None)
    parser.add_argument("--design_pc_factor_levels", type=int, default=1)
    parser.add_argument("--gamg_threshold", type=float, default=0.05)
    parser.add_argument("--gamg_agg_nsmooths", type=int, default=1)
    parser.add_argument("--gamg_set_coordinates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--state_out", type=str, default="")
    parser.add_argument("--json_out", type=str, default="")
    parser.add_argument("--save_outer_state_history", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    result, state = run_topology_optimisation(
        nx=args.nx,
        ny=args.ny,
        length=args.length,
        height=args.height,
        traction=args.traction,
        load_fraction=args.load_fraction,
        fixed_pad_cells=args.fixed_pad_cells,
        load_pad_cells=args.load_pad_cells,
        volume_fraction_target=args.volume_fraction_target,
        theta_min=args.theta_min,
        solid_latent=args.solid_latent,
        young=args.young,
        poisson=args.poisson,
        alpha_reg=args.alpha_reg,
        ell_pf=args.ell_pf,
        mu_move=args.mu_move,
        lambda_init=args.lambda_init,
        beta_lambda=args.beta_lambda,
        volume_penalty=args.volume_penalty,
        p_start=args.p_start,
        p_max=args.p_max,
        p_increment=args.p_increment,
        continuation_interval=args.continuation_interval,
        outer_maxit=args.outer_maxit,
        outer_tol=args.outer_tol,
        volume_tol=args.volume_tol,
        mechanics_maxit=args.mechanics_maxit,
        design_maxit=args.design_maxit,
        tolf=args.tolf,
        tolg=args.tolg,
        ksp_rtol=args.ksp_rtol,
        ksp_max_it=args.ksp_max_it,
        linesearch_tol=args.linesearch_tol,
        linesearch_interval=(args.linesearch_a, args.linesearch_b),
        use_trust_region=args.use_trust_region,
        mechanics_solver_type=args.mechanics_solver_type,
        mechanics_ksp_type=args.mechanics_ksp_type,
        mechanics_use_near_nullspace=args.mechanics_use_near_nullspace,
        design_solver_type=args.design_solver_type,
        design_nonlinear_method=args.design_nonlinear_method,
        design_gd_adaptive_alpha0=args.design_gd_adaptive_alpha0,
        design_gd_adaptive_window_scale=args.design_gd_adaptive_window_scale,
        design_gd_adaptive_nonnegative=args.design_gd_adaptive_nonnegative,
        design_ksp_type=args.design_ksp_type,
        design_pc_type=args.design_pc_type,
        design_norm_type=args.design_norm_type,
        design_ksp_rtol=args.design_ksp_rtol,
        design_ksp_max_it=args.design_ksp_max_it,
        design_pc_factor_levels=args.design_pc_factor_levels,
        gamg_threshold=args.gamg_threshold,
        gamg_agg_nsmooths=args.gamg_agg_nsmooths,
        gamg_set_coordinates=args.gamg_set_coordinates,
        save_outer_state_history=args.save_outer_state_history or bool(args.state_out),
        verbose=not args.quiet,
    )

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if args.state_out:
        np.savez(
            args.state_out,
            coords=state["coords"],
            triangles=state["triangles"],
            theta=state["theta"],
            u=state["u"],
            z=state["z"],
            theta_history=np.stack([snap["theta"] for snap in state["theta_history"]], axis=0),
            theta_history_outer=np.array([snap["outer_iter"] for snap in state["theta_history"]], dtype=np.int32),
            theta_history_p=np.array([snap["p_penal"] for snap in state["theta_history"]], dtype=np.float64),
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

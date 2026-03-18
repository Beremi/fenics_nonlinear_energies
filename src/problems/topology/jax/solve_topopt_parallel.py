#!/usr/bin/env python3
"""MPI-parallel topology optimisation with PETSc mechanics and GD design updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI
from PIL import Image

from src.problems.topology.support.policy import (
    constitutive_plane_stress,
    message_is_converged,
    relative_state_change,
    staircase_p_step,
)
from src.core.petsc.minimizers import gradient_descent
from src.problems.topology.jax.jax_energy import theta_from_latent
from src.problems.topology.jax.parallel_support import (
    build_structured_topology_partition,
    create_initial_design_vec,
    distributed_relative_change,
    gather_quantile,
    TopologyDesignEvaluator,
    TopologyMechanicsAssembler,
)


jax.config.update("jax_enable_x64", True)


def _mechanics_reached_max_it(message: str) -> bool:
    text = str(message).upper()
    return "DIVERGED_MAX_IT" in text or "ACCEPTED_MAX_IT_LINESEARCH" in text


def _design_reached_max_it(message: str) -> bool:
    return "maximum number of iterations reached" in str(message).lower()


def _continuation_gate_recent_maxit(history: list[dict[str, object]], lookback: int) -> tuple[bool, int, int]:
    if lookback <= 0 or not history:
        return True, 0, 0
    recent = history[-lookback:]
    mechanics_hits = sum(
        1 for row in recent if _mechanics_reached_max_it(str(row.get("mechanics_message", "")))
    )
    design_hits = sum(
        1 for row in recent if _design_reached_max_it(str(row.get("design_message", "")))
    )
    return mechanics_hits == 0 and design_hits == 0, mechanics_hits, design_hits


def _design_step_is_acceptable(
    message: str,
    last_grad_norm: float,
    tolg: float,
    *,
    accepted_steps: int = 0,
    last_dE: float = 0.0,
) -> bool:
    msg = message.lower()
    if message_is_converged(message):
        return True
    if "golden-section line search failed" in msg and np.isfinite(last_grad_norm):
        return float(last_grad_norm) <= 10.0 * float(tolg)
    if accepted_steps > 0 and np.isfinite(last_dE) and float(last_dE) > 0.0:
        return True
    if "maximum number of iterations reached" in msg and np.isfinite(last_grad_norm):
        return float(last_grad_norm) <= max(2.0 * float(tolg), float(tolg) + 5e-4)
    return False


def _summarise_gd_work(result: dict) -> dict[str, float]:
    history = result.get("history", [])
    accepted = [row for row in history if bool(row.get("accepted_step", False))]
    alpha_values = [float(row.get("alpha", np.nan)) for row in accepted]
    gamma_values = [
        float(row.get("alpha", np.nan)) * float(row.get("grad_inf_norm", np.nan))
        for row in accepted
        if np.isfinite(float(row.get("alpha", np.nan))) and np.isfinite(float(row.get("grad_inf_norm", np.nan)))
    ]
    grad_values = [float(row.get("grad_norm", np.nan)) for row in history]
    grad_inf_values = [float(row.get("grad_inf_norm", np.nan)) for row in history]
    return {
        "accepted_steps": int(len(accepted)),
        "alpha_mean": float(np.mean(alpha_values)) if alpha_values else np.nan,
        "alpha_min": float(np.min(alpha_values)) if alpha_values else np.nan,
        "alpha_max": float(np.max(alpha_values)) if alpha_values else np.nan,
        "last_alpha": float(history[-1]["alpha"]) if history else np.nan,
        "gamma_mean": float(np.mean(gamma_values)) if gamma_values else np.nan,
        "gamma_min": float(np.min(gamma_values)) if gamma_values else np.nan,
        "gamma_max": float(np.max(gamma_values)) if gamma_values else np.nan,
        "last_gamma": float(result.get("last_gamma_scaled", np.nan)),
        "ls_evals": int(sum(int(row.get("ls_evals", 0)) for row in history)),
        "last_grad_norm": float(grad_values[-1]) if grad_values else np.nan,
        "last_grad_inf_norm": float(grad_inf_values[-1]) if grad_inf_values else np.nan,
        "grad_time": float(sum(float(row.get("t_grad", 0.0)) for row in history)),
        "ls_time": float(sum(float(row.get("t_ls", 0.0)) for row in history)),
        "update_time": float(sum(float(row.get("t_update", 0.0)) for row in history)),
        "iter_time": float(sum(float(row.get("t_iter", 0.0)) for row in history)),
    }


def _root_theta_grid(
    partition,
    z_vec,
    *,
    theta_min: float,
    solid_latent: float,
):
    z_owned = np.asarray(z_vec.array[:], dtype=np.float64)
    z_free = partition.scalar_layout.gather_full(z_owned)
    if partition.comm.Get_rank() != 0:
        return None
    z_grid = np.full((partition.nx + 1, partition.ny + 1), float(solid_latent), dtype=np.float64)
    free_mask = partition.design_index_map >= 0
    z_grid[free_mask] = z_free[partition.design_index_map[free_mask]]
    theta_grid = np.asarray(
        theta_from_latent(jnp.asarray(z_grid, dtype=jnp.float64), theta_min),
        dtype=np.float64,
    )
    return theta_grid


def _root_displacement_grid(partition, u_vec):
    u_owned = np.asarray(u_vec.array[:], dtype=np.float64)
    u_free = partition.vector_layout.gather_full(u_owned)
    if partition.comm.Get_rank() != 0:
        return None
    u_grid = np.zeros((partition.nx + 1, partition.ny + 1, 2), dtype=np.float64)
    if u_free.size:
        u_grid[1:, :, :] = u_free.reshape(partition.nx, partition.ny + 1, 2)
    return u_grid


def _cell_density_from_theta_grid(theta_grid: np.ndarray) -> np.ndarray:
    return 0.25 * (
        theta_grid[:-1, :-1]
        + theta_grid[1:, :-1]
        + theta_grid[:-1, 1:]
        + theta_grid[1:, 1:]
    )


def _write_density_snapshot_png(path: Path, theta_grid: np.ndarray) -> None:
    cell_density = np.clip(_cell_density_from_theta_grid(theta_grid), 0.0, 1.0)
    # theta_grid is stored as (nx+1, ny+1), so cell densities are (nx, ny);
    # transpose to image layout (rows=y, cols=x) before flipping to image origin.
    image = np.flipud(np.round(255.0 * cell_density.T).astype(np.uint8))
    Image.fromarray(image, mode="L").save(path)


def run_topology_optimisation_parallel(
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
    theta_min: float = 1e-6,
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
    p_max: float = 10.0,
    p_increment: float = 0.2,
    continuation_interval: int = 1,
    outer_maxit: int = 180,
    outer_tol: float = 2e-2,
    volume_tol: float = 1e-3,
    stall_theta_tol: float = 1e-6,
    stall_p_min: float = 4.0,
    design_maxit: int = 20,
    tolf: float = 1e-6,
    tolg: float = 1e-3,
    linesearch_tol: float = 1e-1,
    mechanics_ksp_rtol: float = 1e-4,
    mechanics_ksp_max_it: int = 100,
    mechanics_ksp_type: str = "fgmres",
    mechanics_pc_type: str = "gamg",
    mechanics_use_near_nullspace: bool = True,
    mechanics_reorder_mode: str = "block_xy",
    mechanics_gamg_threshold: float = 0.05,
    mechanics_gamg_agg_nsmooths: int = 1,
    mechanics_gamg_repartition: bool = True,
    mechanics_gamg_reuse_interpolation: bool = True,
    mechanics_gamg_aggressive_coarsening: int = 1,
    mechanics_gamg_set_coordinates: bool = True,
    mechanics_enable_fallback: bool = True,
    mechanics_fallback_max_it: int = 400,
    mechanics_fallback_sub_pc_factor_levels: int = 1,
    design_reorder: bool = True,
    design_gd_adaptive_alpha0: float = 1.0,
    design_gd_adaptive_window_scale: float = 2.0,
    design_gd_adaptive_nonnegative: bool = True,
    design_gd_line_search: str = "golden_adaptive",
    linesearch_relative_to_bound: bool = True,
    print_outer_iterations: bool = False,
    outer_log_path: str = "",
    save_outer_state_history: bool = False,
    save_design_iteration_history: bool = False,
    outer_snapshot_stride: int = 1,
    outer_snapshot_dir: str = "",
    verbose: bool = False,
) -> tuple[dict, dict]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    total_start = time.perf_counter()
    setup_start = time.perf_counter()

    def _log(msg: str) -> None:
        if verbose and rank == 0:
            elapsed = time.perf_counter() - total_start
            print(f"[parallel-topopt +{elapsed:8.3f}s] {msg}", flush=True)

    if mechanics_reorder_mode != "block_xy":
        raise ValueError("The distributed structured implementation currently supports mechanics_reorder_mode='block_xy' only.")

    partition = build_structured_topology_partition(
        nx=nx,
        ny=ny,
        length=length,
        height=height,
        traction=traction,
        load_fraction=load_fraction,
        fixed_pad_cells=fixed_pad_cells,
        load_pad_cells=load_pad_cells,
        theta_min=theta_min,
        solid_latent=solid_latent,
        comm=comm,
    )
    _log("partition built")
    constitutive = constitutive_plane_stress(young, poisson)

    mechanics_pc_options = {}
    if mechanics_pc_type == "gamg":
        mechanics_pc_options = {
            "pc_gamg_threshold": float(mechanics_gamg_threshold),
            "pc_gamg_agg_nsmooths": int(mechanics_gamg_agg_nsmooths),
            "pc_gamg_repartition": str(bool(mechanics_gamg_repartition)).lower(),
            "pc_gamg_reuse_interpolation": str(bool(mechanics_gamg_reuse_interpolation)).lower(),
            "pc_gamg_aggressive_coarsening": int(mechanics_gamg_aggressive_coarsening),
        }
    mechanics_fallback_solvers: list[dict[str, object]] = []

    mechanics_assembler = TopologyMechanicsAssembler(
        partition=partition,
        constitutive=constitutive,
        comm=comm,
        ksp_rtol=mechanics_ksp_rtol,
        ksp_type=mechanics_ksp_type,
        pc_type=mechanics_pc_type,
        ksp_max_it=mechanics_ksp_max_it,
        use_near_nullspace=mechanics_use_near_nullspace,
        pc_options=mechanics_pc_options,
        gamg_set_coordinates=mechanics_gamg_set_coordinates,
        fallback_solvers=mechanics_fallback_solvers,
    )
    _log("mechanics assembler created")
    design_eval = TopologyDesignEvaluator(
        partition=partition,
        constitutive=constitutive,
        alpha_reg=alpha_reg,
        ell_pf=ell_pf,
        mu_move=mu_move,
        comm=comm,
    )
    _log("design evaluator created")

    u_vec = mechanics_assembler.create_vec()
    z_vec, _ = create_initial_design_vec(
        partition=partition,
        target_volume_fraction=volume_fraction_target,
        theta_min=theta_min,
        solid_latent=solid_latent,
    )
    _log("initial vectors created")

    setup_time = time.perf_counter() - setup_start
    _log(f"setup complete in {setup_time:.3f}s")
    p_penal = float(p_start)
    lambda_volume = float(lambda_init)
    design_adaptive_alpha = float(max(abs(design_gd_adaptive_alpha0), linesearch_tol))
    prev_compliance = np.nan
    prev_theta_state: np.ndarray | None = None
    history: list[dict[str, object]] = []
    design_iteration_history: list[dict[str, object]] = []
    status = "completed"
    outer_converged = False
    outer_stall_converged = False
    final_theta_min = float("inf")
    final_theta_max = float(theta_min)
    snapshot_records: list[dict[str, object]] = []
    last_snapshot_outer = -1
    snapshot_dir = Path(outer_snapshot_dir) if outer_snapshot_dir else None
    if rank == 0 and save_outer_state_history and snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for stale in sorted(snapshot_dir.glob("theta_*.png")):
            stale.unlink()
    outer_log_file = Path(outer_log_path) if outer_log_path else None
    if rank == 0 and outer_log_file is not None:
        outer_log_file.parent.mkdir(parents=True, exist_ok=True)
        outer_log_file.write_text("")

    def _emit_outer_line(line: str) -> None:
        if rank != 0:
            return
        if print_outer_iterations:
            print(line, flush=True)
        if outer_log_file is not None:
            with outer_log_file.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _maybe_save_snapshot(outer_it: int, p_value: float, volume_value: float) -> None:
        nonlocal last_snapshot_outer
        if not save_outer_state_history:
            return
        stride = max(1, int(outer_snapshot_stride))
        if outer_it != 0 and (outer_it % stride) != 0:
            return
        theta_grid = _root_theta_grid(
            partition,
            z_vec,
            theta_min=theta_min,
            solid_latent=solid_latent,
        )
        last_snapshot_outer = int(outer_it)
        if rank != 0 or theta_grid is None:
            return
        frame_name = f"theta_{len(snapshot_records):04d}_outer_{outer_it:04d}.png"
        if snapshot_dir is not None:
            _write_density_snapshot_png(snapshot_dir / frame_name, theta_grid)
        snapshot_records.append(
            {
                "outer_iter": int(outer_it),
                "p_penal": float(p_value),
                "volume_fraction": float(volume_value),
                "file": frame_name,
            }
        )

    initial_volume = float(design_eval.volume_fraction(z_vec))
    _maybe_save_snapshot(0, p_penal, initial_volume)
    for outer_it in range(1, outer_maxit + 1):
        _log(f"outer {outer_it}: start")
        theta_before_owned = design_eval.theta_owned(z_vec)
        _log(f"outer {outer_it}: theta_before ready")
        theta_state_change = distributed_relative_change(theta_before_owned, prev_theta_state, comm)
        _log(f"outer {outer_it}: theta_state_change ready")
        volume_before = float(design_eval.volume_fraction(z_vec))
        _log(f"outer {outer_it}: volume_before ready")
        volume_residual_before = float(volume_before - volume_fraction_target)
        _log(f"outer {outer_it}: pre-checks done")

        mechanics_assembler.update_material_scale_from_design(z_vec, p_penal)
        _log(f"outer {outer_it}: mechanics material updated")
        mechanics_res = mechanics_assembler.solve(
            u_vec,
            capture_residual_history=False,
        )
        _log(
            f"outer {outer_it}: mechanics solved "
            f"(its={int(mechanics_res['ksp_its'])}, "
            f"assemble={float(mechanics_res['assemble_time']):.3f}s, "
            f"solve={float(mechanics_res['solve_time']):.3f}s)"
        )
        if mechanics_res["reason_code"] <= 0:
            status = "failed_mechanics"
            _emit_outer_line(f"[outer {outer_it:04d}] mechanics failed: {mechanics_res['reason']}")
            history.append(
                {
                    "outer_iter": outer_it,
                    "p_penal": float(p_penal),
                    "mechanics_ksp_its": int(mechanics_res["ksp_its"]),
                    "mechanics_assemble_time": float(mechanics_res["assemble_time"]),
                    "mechanics_solve_time": float(mechanics_res["solve_time"]),
                    "mechanics_solver_label": str(mechanics_res.get("solver_label", "")),
                    "mechanics_attempt_count": int(len(mechanics_res.get("attempts", []))),
                    "mechanics_message": str(mechanics_res["reason"]),
                    "design_message": "not_run",
                }
            )
            break

        u_vec.destroy()
        u_vec = mechanics_res["x"]
        design_eval.update_state_from_current(
            z_vec=z_vec,
            u_vec=u_vec,
            lambda_volume=0.0,
            p_penal=p_penal,
        )
        _log(f"outer {outer_it}: design frozen state updated (lambda=0)")
        lambda_reference = gather_quantile(
            design_eval.sensitivity_scale_owned_cells(),
            float(np.clip(1.0 - volume_fraction_target, 0.0, 1.0)),
            comm,
        )
        lambda_penalty = float(volume_penalty * volume_residual_before)
        lambda_effective = float(lambda_reference + lambda_volume + lambda_penalty)

        design_eval.update_state_from_current(
            z_vec=z_vec,
            u_vec=u_vec,
            lambda_volume=lambda_effective,
            p_penal=p_penal,
        )
        _log(
            f"outer {outer_it}: design state updated "
            f"(lambda_ref={lambda_reference:.6e}, lambda_eff={lambda_effective:.6e})"
        )
        design_res = gradient_descent(
            design_eval.energy_fn,
            design_eval.gradient_fn,
            z_vec,
            tolf=tolf,
            tolg=tolg,
            linesearch_tol=linesearch_tol,
            line_search=design_gd_line_search,
            adaptive_alpha0=design_adaptive_alpha,
            adaptive_window_scale=design_gd_adaptive_window_scale,
            adaptive_nonnegative=design_gd_adaptive_nonnegative,
            linesearch_relative_to_bound=linesearch_relative_to_bound,
            maxit=design_maxit,
            comm=comm,
            save_history=True,
            verbose=verbose,
        )
        _log(
            f"outer {outer_it}: design GD finished "
            f"(nit={int(design_res['nit'])}, message={design_res['message']})"
        )
        design_work = _summarise_gd_work(design_res)
        if design_gd_line_search == "golden_gamma_beta":
            design_adaptive_alpha = float(
                max(abs(design_res.get("last_gamma_scaled", design_adaptive_alpha)), linesearch_tol)
            )
        elif design_gd_line_search == "golden_adaptive":
            design_adaptive_alpha = float(
                max(abs(design_res.get("last_alpha_abs", design_adaptive_alpha)), linesearch_tol)
            )
        if save_design_iteration_history:
            for row in design_res.get("history", []):
                design_iteration_history.append(
                    {
                        "outer_iter": int(outer_it),
                        "gd_iter": int(row.get("it", 0)),
                        "energy": float(row.get("energy", np.nan)),
                        "dE": float(row.get("dE", np.nan)),
                        "grad_norm": float(row.get("grad_norm", np.nan)),
                        "grad_inf_norm": float(row.get("grad_inf_norm", np.nan)),
                        "alpha": float(row.get("alpha", np.nan)),
                        "ls_a": float(row.get("ls_a", np.nan)),
                        "ls_b": float(row.get("ls_b", np.nan)),
                        "ls_evals": int(row.get("ls_evals", 0)),
                        "accepted_step": bool(row.get("accepted_step", False)),
                        "message": str(row.get("message", "")),
                    }
                )
        last_dE = 0.0
        if design_res.get("history"):
            last_dE = float(design_res["history"][-1].get("dE", 0.0))
        if not _design_step_is_acceptable(
            str(design_res["message"]),
            float(design_work["last_grad_norm"]),
            tolg,
            accepted_steps=int(design_work["accepted_steps"]),
            last_dE=last_dE,
        ):
            status = "failed_design"
            z_vec = design_res["x"]
            _emit_outer_line(
                "[outer "
                f"{outer_it:04d}] design stopped: "
                f"{design_res['message']} "
                f"(gd={int(design_res['nit'])}, "
                f"g={float(design_work['last_grad_norm']):.3e})"
            )
            history.append(
                {
                    "outer_iter": outer_it,
                    "p_penal": float(p_penal),
                    "mechanics_ksp_its": int(mechanics_res["ksp_its"]),
                    "mechanics_assemble_time": float(mechanics_res["assemble_time"]),
                    "mechanics_build_v_local_time": float(mechanics_res["build_v_local_time"]),
                    "mechanics_elem_hessian_time": float(mechanics_res["elem_hessian_time"]),
                    "mechanics_scatter_time": float(mechanics_res["scatter_time"]),
                    "mechanics_coo_assembly_time": float(mechanics_res["coo_assembly_time"]),
                    "mechanics_solve_time": float(mechanics_res["solve_time"]),
                    "mechanics_solver_label": str(mechanics_res.get("solver_label", "")),
                    "mechanics_attempt_count": int(len(mechanics_res.get("attempts", []))),
                    "design_iters": int(design_res["nit"]),
                    "design_ls_evals": int(design_work["ls_evals"]),
                    "design_grad_norm_last": float(design_work["last_grad_norm"]),
                    "design_grad_time": float(design_work["grad_time"]),
                    "design_ls_time": float(design_work["ls_time"]),
                    "design_update_time": float(design_work["update_time"]),
                    "design_iter_time": float(design_work["iter_time"]),
                    "mechanics_message": str(mechanics_res["reason"]),
                    "design_message": str(design_res["message"]),
                }
            )
            break

        z_vec = design_res["x"]
        theta_after_owned = design_eval.theta_owned(z_vec)

        compliance_value = float(mechanics_assembler.compliance(u_vec))
        volume_value = float(design_eval.volume_fraction(z_vec))
        design_change = distributed_relative_change(theta_after_owned, theta_before_owned, comm)
        compliance_change = (
            float(abs(compliance_value - prev_compliance) / max(1.0, abs(prev_compliance)))
            if np.isfinite(prev_compliance)
            else np.inf
        )
        volume_residual = float(volume_value - volume_fraction_target)
        scheduled_p_step = staircase_p_step(
            p_penal,
            p_max=p_max,
            p_increment=p_increment,
            continuation_interval=continuation_interval,
            outer_it=outer_it,
        )
        recent_gate_history = history[-9:] + [
            {
                "mechanics_message": str(mechanics_res["reason"]),
                "design_message": str(design_res["message"]),
            }
        ]
        continuation_gate_ok, recent_mech_maxit, recent_design_maxit = _continuation_gate_recent_maxit(
            recent_gate_history, 10
        )
        p_step = float(scheduled_p_step if (scheduled_p_step <= 0.0 or continuation_gate_ok) else 0.0)

        history.append(
            {
                "outer_iter": int(outer_it),
                "p_penal": float(p_penal),
                "p_step": float(p_step),
                "scheduled_p_step": float(scheduled_p_step),
                "continuation_gate_ok": bool(continuation_gate_ok),
                "recent_mechanics_maxit_count": int(recent_mech_maxit),
                "recent_design_maxit_count": int(recent_design_maxit),
                "lambda_correction": float(lambda_volume),
                "lambda_penalty": float(lambda_penalty),
                "lambda_reference": float(lambda_reference),
                "lambda_effective": float(lambda_effective),
                "compliance": float(compliance_value),
                "volume_fraction_before": float(volume_before),
                "volume_residual_before": float(volume_residual_before),
                "volume_fraction": float(volume_value),
                "volume_residual": float(volume_residual),
                "theta_state_change": float(theta_state_change),
                "design_change": float(design_change),
                "compliance_change": float(compliance_change),
                "mechanics_ksp_its": int(mechanics_res["ksp_its"]),
                "mechanics_assemble_time": float(mechanics_res["assemble_time"]),
                "mechanics_build_v_local_time": float(mechanics_res["build_v_local_time"]),
                "mechanics_elem_hessian_time": float(mechanics_res["elem_hessian_time"]),
                "mechanics_scatter_time": float(mechanics_res["scatter_time"]),
                "mechanics_coo_assembly_time": float(mechanics_res["coo_assembly_time"]),
                "mechanics_solve_time": float(mechanics_res["solve_time"]),
                "mechanics_solver_label": str(mechanics_res.get("solver_label", "")),
                "mechanics_attempt_count": int(len(mechanics_res.get("attempts", []))),
                "design_iters": int(design_res["nit"]),
                "design_ls_evals": int(design_work["ls_evals"]),
                "design_alpha_mean": float(design_work["alpha_mean"]),
                "design_alpha_last": float(design_work["last_alpha"]),
                "design_gamma_mean": float(design_work["gamma_mean"]),
                "design_gamma_last": float(design_work["last_gamma"]),
                "design_grad_norm_last": float(design_work["last_grad_norm"]),
                "design_grad_inf_norm_last": float(design_work["last_grad_inf_norm"]),
                "design_grad_time": float(design_work["grad_time"]),
                "design_ls_time": float(design_work["ls_time"]),
                "design_update_time": float(design_work["update_time"]),
                "design_iter_time": float(design_work["iter_time"]),
                "design_energy": float(design_res["fun"]),
                "design_time": float(design_res["time"]),
                "mechanics_message": str(mechanics_res["reason"]),
                "design_message": str(design_res["message"]),
            }
        )

        _emit_outer_line(
            "[outer "
            f"{outer_it:04d}] p={p_penal:.2f} "
            f"C={compliance_value:.6f} "
            f"V={volume_value:.6f} "
            f"|V-V*|={abs(volume_residual):.3e} "
            f"dtheta={design_change:.3e} "
            f"dC={compliance_change:.3e} "
            f"ksp={int(mechanics_res['ksp_its'])} "
            f"gd={int(design_res['nit'])} "
            f"ls={int(design_work['ls_evals'])} "
            f"g={float(design_work['last_grad_norm']):.3e}"
        )

        local_theta_min = np.min(theta_after_owned) if theta_after_owned.size else np.inf
        local_theta_max = np.max(theta_after_owned) if theta_after_owned.size else -np.inf
        final_theta_min = min(final_theta_min, float(comm.allreduce(local_theta_min, op=MPI.MIN)))
        final_theta_max = max(final_theta_max, float(comm.allreduce(local_theta_max, op=MPI.MAX)))
        prev_compliance = compliance_value
        prev_theta_state = theta_before_owned.copy()
        lambda_volume += beta_lambda * max(1.0, lambda_reference) * volume_residual
        _maybe_save_snapshot(outer_it, p_penal, volume_value)

        if (
            p_penal >= p_max - 1e-12
            and abs(volume_residual) <= volume_tol
            and design_change <= outer_tol
            and compliance_change <= outer_tol
        ):
            outer_converged = True
            break

        if (
            stall_theta_tol > 0.0
            and p_penal >= stall_p_min
            and design_change <= stall_theta_tol
            and theta_state_change <= stall_theta_tol
        ):
            outer_converged = True
            outer_stall_converged = True
            break

        p_penal = min(p_max, p_penal + p_step)

    if status == "completed" and not outer_converged and len(history) >= outer_maxit:
        status = "max_outer_iterations"

    if status != "failed_mechanics":
        mechanics_assembler.update_material_scale_from_design(z_vec, p_penal)
        final_mechanics_res = mechanics_assembler.solve(u_vec)
        if final_mechanics_res["reason_code"] > 0:
            u_vec.destroy()
            u_vec = final_mechanics_res["x"]
        else:
            status = "failed_final_mechanics"
            final_mechanics_res["x"].destroy()

    final_theta_owned = design_eval.theta_owned(z_vec)
    local_theta_min = np.min(final_theta_owned) if final_theta_owned.size else np.inf
    local_theta_max = np.max(final_theta_owned) if final_theta_owned.size else -np.inf
    final_theta_min = float(comm.allreduce(local_theta_min, op=MPI.MIN))
    final_theta_max = float(comm.allreduce(local_theta_max, op=MPI.MAX))
    total_time = time.perf_counter() - total_start
    final_volume = float(design_eval.volume_fraction(z_vec))
    final_compliance = float(mechanics_assembler.compliance(u_vec))
    theta_grid = _root_theta_grid(
        partition,
        z_vec,
        theta_min=theta_min,
        solid_latent=solid_latent,
    )
    u_grid = _root_displacement_grid(partition, u_vec)
    if save_outer_state_history and history:
        if last_snapshot_outer != int(len(history)):
            _maybe_save_snapshot(int(len(history)), p_penal, final_volume)
    result = {
        "solver": "parallel_jax_petsc_topopt",
        "backend": "mpi",
        "result": status,
        "time": float(total_time),
        "setup_time": float(setup_time),
        "nprocs": int(comm.size),
        "rank": int(rank),
        "mesh": {
            "nx": int(nx),
            "ny": int(ny),
            "nodes": int((nx + 1) * (ny + 1)),
            "elements": int(2 * nx * ny),
            "displacement_free_dofs": int(partition.n_vector_free),
            "design_free_dofs": int(partition.n_scalar_free),
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
            "stall_theta_tol": float(stall_theta_tol),
            "stall_p_min": float(stall_p_min),
        },
        "solver_options": {
            "mechanics_ksp_type": str(mechanics_ksp_type),
            "mechanics_pc_type": str(mechanics_pc_type),
            "mechanics_ksp_rtol": float(mechanics_ksp_rtol),
            "mechanics_ksp_max_it": int(mechanics_ksp_max_it),
            "mechanics_use_near_nullspace": bool(mechanics_use_near_nullspace),
            "mechanics_reorder_mode": str(mechanics_reorder_mode),
            "mechanics_gamg_threshold": float(mechanics_gamg_threshold),
            "mechanics_gamg_agg_nsmooths": int(mechanics_gamg_agg_nsmooths),
            "mechanics_gamg_repartition": bool(mechanics_gamg_repartition),
            "mechanics_gamg_reuse_interpolation": bool(mechanics_gamg_reuse_interpolation),
            "mechanics_gamg_aggressive_coarsening": int(mechanics_gamg_aggressive_coarsening),
            "mechanics_gamg_set_coordinates": bool(mechanics_gamg_set_coordinates),
            "mechanics_enable_fallback": bool(mechanics_enable_fallback),
            "mechanics_fallback_max_it": int(mechanics_fallback_max_it),
            "mechanics_fallback_sub_pc_factor_levels": int(mechanics_fallback_sub_pc_factor_levels),
            "design_reorder": bool(design_reorder),
            "design_maxit": int(design_maxit),
            "tolf": float(tolf),
            "tolg": float(tolg),
            "linesearch_tol": float(linesearch_tol),
            "design_gd_adaptive_alpha0": float(design_gd_adaptive_alpha0),
            "design_gd_adaptive_window_scale": float(design_gd_adaptive_window_scale),
            "design_gd_adaptive_nonnegative": bool(design_gd_adaptive_nonnegative),
            "design_gd_line_search": str(design_gd_line_search),
            "linesearch_relative_to_bound": bool(linesearch_relative_to_bound),
        },
        "final_metrics": {
            "outer_iterations": int(len(history)),
            "final_volume_fraction": float(final_volume),
            "final_compliance": float(final_compliance),
            "final_theta_min": float(final_theta_min),
            "final_theta_max": float(max(final_theta_max, float(theta_from_latent(jnp.asarray([solid_latent], dtype=jnp.float64), theta_min)[0]))),
            "final_p_penal": float(p_penal),
            "outer_stall_converged": bool(outer_stall_converged),
        },
        "history": history,
        "design_iteration_history": design_iteration_history if save_design_iteration_history else [],
        "jax_version": jax.__version__,
    }
    state = {}
    if rank == 0:
        state = {
            "theta_grid": theta_grid,
            "u_grid": u_grid,
            "snapshot_records": snapshot_records,
            "snapshot_dir": str(snapshot_dir) if snapshot_dir is not None else "",
        }
    u_vec.destroy()
    z_vec.destroy()
    mechanics_assembler.cleanup()

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
    parser.add_argument("--theta_min", type=float, default=1e-6)
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
    parser.add_argument("--p_max", type=float, default=10.0)
    parser.add_argument("--p_increment", type=float, default=0.2)
    parser.add_argument("--continuation_interval", type=int, default=1)
    parser.add_argument("--outer_maxit", type=int, default=180)
    parser.add_argument("--outer_tol", type=float, default=2e-2)
    parser.add_argument("--volume_tol", type=float, default=1e-3)
    parser.add_argument("--stall_theta_tol", type=float, default=1e-6)
    parser.add_argument("--stall_p_min", type=float, default=4.0)
    parser.add_argument("--design_maxit", type=int, default=20)
    parser.add_argument("--tolf", type=float, default=1e-6)
    parser.add_argument("--tolg", type=float, default=1e-3)
    parser.add_argument("--linesearch_tol", type=float, default=1e-1)
    parser.add_argument("--mechanics_ksp_rtol", type=float, default=1e-4)
    parser.add_argument("--mechanics_ksp_max_it", type=int, default=100)
    parser.add_argument("--mechanics_ksp_type", type=str, default="fgmres")
    parser.add_argument("--mechanics_pc_type", type=str, default="gamg")
    parser.add_argument("--mechanics_use_near_nullspace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mechanics_reorder_mode",
        choices=("none", "block_rcm", "block_xy", "block_metis"),
        default="block_xy",
    )
    parser.add_argument("--mechanics_gamg_threshold", type=float, default=0.05)
    parser.add_argument("--mechanics_gamg_agg_nsmooths", type=int, default=1)
    parser.add_argument("--mechanics_gamg_repartition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mechanics_gamg_reuse_interpolation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mechanics_gamg_aggressive_coarsening", type=int, default=1)
    parser.add_argument("--mechanics_gamg_set_coordinates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mechanics_enable_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mechanics_fallback_max_it", type=int, default=400)
    parser.add_argument("--mechanics_fallback_sub_pc_factor_levels", type=int, default=1)
    parser.add_argument("--design_reorder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--design_gd_adaptive_alpha0", type=float, default=1.0)
    parser.add_argument("--design_gd_adaptive_window_scale", type=float, default=2.0)
    parser.add_argument("--design_gd_adaptive_nonnegative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--linesearch_relative_to_bound", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--design_gd_line_search",
        type=str,
        choices=("golden_adaptive", "golden_linf", "golden_gamma_beta"),
        default="golden_adaptive",
    )
    parser.add_argument("--print_outer_iterations", action="store_true")
    parser.add_argument("--outer_log_path", type=str, default="")
    parser.add_argument("--save_outer_state_history", action="store_true")
    parser.add_argument("--save_design_iteration_history", action="store_true")
    parser.add_argument("--outer_snapshot_stride", type=int, default=1)
    parser.add_argument("--outer_snapshot_dir", type=str, default="")
    parser.add_argument("--json_out", type=str, default="")
    parser.add_argument("--state_out", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    result, state = run_topology_optimisation_parallel(
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
        stall_theta_tol=args.stall_theta_tol,
        stall_p_min=args.stall_p_min,
        design_maxit=args.design_maxit,
        tolf=args.tolf,
        tolg=args.tolg,
        linesearch_tol=args.linesearch_tol,
        mechanics_ksp_rtol=args.mechanics_ksp_rtol,
        mechanics_ksp_max_it=args.mechanics_ksp_max_it,
        mechanics_ksp_type=args.mechanics_ksp_type,
        mechanics_pc_type=args.mechanics_pc_type,
        mechanics_use_near_nullspace=args.mechanics_use_near_nullspace,
        mechanics_reorder_mode=args.mechanics_reorder_mode,
        mechanics_gamg_threshold=args.mechanics_gamg_threshold,
        mechanics_gamg_agg_nsmooths=args.mechanics_gamg_agg_nsmooths,
        mechanics_gamg_repartition=args.mechanics_gamg_repartition,
        mechanics_gamg_reuse_interpolation=args.mechanics_gamg_reuse_interpolation,
        mechanics_gamg_aggressive_coarsening=args.mechanics_gamg_aggressive_coarsening,
        mechanics_gamg_set_coordinates=args.mechanics_gamg_set_coordinates,
        mechanics_enable_fallback=args.mechanics_enable_fallback,
        mechanics_fallback_max_it=args.mechanics_fallback_max_it,
        mechanics_fallback_sub_pc_factor_levels=args.mechanics_fallback_sub_pc_factor_levels,
        design_reorder=args.design_reorder,
        design_gd_adaptive_alpha0=args.design_gd_adaptive_alpha0,
        design_gd_adaptive_window_scale=args.design_gd_adaptive_window_scale,
        design_gd_adaptive_nonnegative=args.design_gd_adaptive_nonnegative,
        design_gd_line_search=args.design_gd_line_search,
        linesearch_relative_to_bound=args.linesearch_relative_to_bound,
        print_outer_iterations=args.print_outer_iterations,
        outer_log_path=args.outer_log_path,
        save_outer_state_history=args.save_outer_state_history or bool(args.state_out) or bool(args.outer_snapshot_dir),
        save_design_iteration_history=args.save_design_iteration_history,
        outer_snapshot_stride=args.outer_snapshot_stride,
        outer_snapshot_dir=args.outer_snapshot_dir,
        verbose=not args.quiet,
    )

    if comm.rank == 0:
        if args.json_out:
            path = Path(args.json_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2))
        if args.state_out:
            path = Path(args.state_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            theta_grid = state.get("theta_grid")
            u_grid = state.get("u_grid")
            snapshot_records = state.get("snapshot_records", [])
            np.savez_compressed(
                path,
                theta_grid=np.asarray(theta_grid, dtype=np.float32) if theta_grid is not None else np.zeros((0,), dtype=np.float32),
                u_grid=np.asarray(u_grid, dtype=np.float32) if u_grid is not None else np.zeros((0,), dtype=np.float32),
                snapshot_outer=np.array([snap["outer_iter"] for snap in snapshot_records], dtype=np.int32),
                snapshot_p=np.array([snap["p_penal"] for snap in snapshot_records], dtype=np.float32),
                snapshot_volume=np.array([snap["volume_fraction"] for snap in snapshot_records], dtype=np.float32),
            )
        if not args.quiet:
            print(json.dumps(result["final_metrics"], indent=2))


if __name__ == "__main__":
    main()

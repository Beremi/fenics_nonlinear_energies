#!/usr/bin/env python3
"""
HyperElasticity 3D solver — custom Newton (JAX-version) CLI entry point.

Solver logic is in ``HyperElasticity3D_fenics/solver_custom_newton.py``.

Usage:
  Serial:   python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py
  Parallel: mpirun -n <nprocs> python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py
"""
import json
import argparse

from mpi4py import MPI

from HyperElasticity3D_fenics.solver_custom_newton import run_level


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1, help="Mesh level (1-4)")
    parser.add_argument("--steps", type=int, default=1, help="Number of time steps")
    parser.add_argument("--start_step", type=int, default=1, help="Global step index to start from")
    parser.add_argument("--maxit", type=int, default=100, help="Maximum Newton iterations")
    parser.add_argument("--init_npz", type=str, default="", help="NPZ from JAX test data with coords and u_full_steps")
    parser.add_argument("--init_step", type=int, default=0, help="Step index in init_npz to use as initial guess")
    parser.add_argument("--linesearch_a", type=float, default=-0.5, help="Line-search interval lower bound")
    parser.add_argument("--linesearch_b", type=float, default=2.0, help="Line-search interval upper bound")
    parser.add_argument("--use_abs_det", action="store_true", help="Use abs(det(F)) in energy (JAX-compatible)")
    parser.add_argument("--ksp_type", type=str, default="gmres", help="PETSc KSP type")
    parser.add_argument("--pc_type", type=str, default="hypre", help="PETSc PC type")
    parser.add_argument("--ksp_rtol", type=float, default=1e-3, help="KSP relative tolerance")
    parser.add_argument("--ksp_max_it", type=int, default=10000, help="KSP maximum iterations per Newton step")
    parser.add_argument("--tolf", type=float, default=1e-4, help="Newton energy-change tolerance")
    parser.add_argument("--tolg", type=float, default=1e-3, help="Newton gradient-norm tolerance")
    parser.add_argument("--tolg_rel", type=float, default=0.0,
                        help="Newton relative gradient tolerance (scaled by initial gradient)")
    parser.add_argument("--tolx_rel", type=float, default=1e-6, help="Newton relative step-size tolerance")
    parser.add_argument("--tolx_abs", type=float, default=1e-10, help="Newton absolute step-size tolerance")
    parser.add_argument("--require_all_convergence", action="store_true",
                        help="Require energy, step, and gradient convergence together")
    parser.add_argument("--no_near_nullspace", action="store_true", help="Disable elasticity near-nullspace on Hessian")
    parser.add_argument("--hypre_nodal_coarsen", type=int, default=6,
                        help="BoomerAMG nodal coarsen (-1 to skip setting)")
    parser.add_argument("--hypre_vec_interp_variant", type=int, default=3,
                        help="BoomerAMG vector interpolation variant (-1 to skip setting)")
    parser.add_argument("--hypre_strong_threshold", type=float, default=None, help="BoomerAMG strong threshold")
    parser.add_argument("--hypre_coarsen_type", type=str, default="", help="BoomerAMG coarsen type (e.g. HMIS, PMIS)")
    parser.add_argument("--save_history", action="store_true",
                        help="Include per-iteration Newton profile in output JSON")
    parser.add_argument("--save_linear_timing", action="store_true",
                        help="Include per-Newton linear timing breakdown in output JSON")
    parser.add_argument("--pc_setup_on_ksp_cap", action="store_true",
                        help="Only run KSP/PC setup when previous linear solve hit ksp_max_it")
    parser.add_argument("--out", type=str, default="", help="Output JSON file")
    parser.add_argument("--total_steps", type=int, default=24,
                        help="Total steps that span the full 4×2π rotation (controls step size)")
    parser.add_argument("--gamg_threshold", type=float, default=-1.0,
                        help="GAMG threshold for filtering graph (-1 = keep all, most robust)")
    parser.add_argument("--gamg_agg_nsmooths", type=int, default=1,
                        help="GAMG number of smoothing steps for SA prolongation")
    parser.add_argument("--no_gamg_coordinates", action="store_true",
                        help="Disable PCSetCoordinates for GAMG")
    parser.add_argument("--no_fail_fast", action="store_true",
                        help="Do not stop trajectory early on Newton non-convergence")
    parser.add_argument("--no_retry_on_nonfinite", action="store_true",
                        help="Disable non-finite repair attempt with tighter settings")
    parser.add_argument("--nonfinite_retry_rtol_factor", type=float, default=0.1,
                        help="Multiplier for KSP rtol in non-finite repair attempt")
    parser.add_argument("--nonfinite_retry_linesearch_b", type=float, default=1.0,
                        help="Upper bound of line-search interval in non-finite repair attempt")
    parser.add_argument("--no_retry_on_maxit", action="store_true",
                        help="Disable repair attempt when Newton reaches max iterations")
    parser.add_argument("--retry_ksp_max_it_factor", type=float, default=2.0,
                        help="Multiplier for KSP max_it in repair attempt")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    res = run_level(
        args.level,
        num_steps=args.steps,
        verbose=not args.quiet,
        maxit=args.maxit,
        start_step=args.start_step,
        init_npz=args.init_npz,
        init_step=args.init_step,
        linesearch_interval=(args.linesearch_a, args.linesearch_b),
        use_abs_det=args.use_abs_det,
        ksp_type=args.ksp_type,
        pc_type=args.pc_type,
        ksp_rtol=args.ksp_rtol,
        ksp_max_it=args.ksp_max_it,
        tolf=args.tolf,
        tolg=args.tolg,
        tolg_rel=args.tolg_rel,
        tolx_rel=args.tolx_rel,
        tolx_abs=args.tolx_abs,
        require_all_convergence=args.require_all_convergence,
        use_near_nullspace=not args.no_near_nullspace,
        total_steps=args.total_steps,
        hypre_nodal_coarsen=args.hypre_nodal_coarsen,
        hypre_vec_interp_variant=args.hypre_vec_interp_variant,
        hypre_strong_threshold=args.hypre_strong_threshold,
        hypre_coarsen_type=args.hypre_coarsen_type,
        save_history=args.save_history,
        save_linear_timing=args.save_linear_timing,
        pc_setup_on_ksp_cap=args.pc_setup_on_ksp_cap,
        gamg_threshold=args.gamg_threshold,
        gamg_agg_nsmooths=args.gamg_agg_nsmooths,
        gamg_set_coordinates=not args.no_gamg_coordinates,
        fail_fast=not args.no_fail_fast,
        retry_on_nonfinite=not args.no_retry_on_nonfinite,
        retry_on_maxit=not args.no_retry_on_maxit,
        nonfinite_retry_rtol_factor=args.nonfinite_retry_rtol_factor,
        nonfinite_retry_linesearch_b=args.nonfinite_retry_linesearch_b,
        retry_ksp_max_it_factor=args.retry_ksp_max_it_factor,
    )

    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(res, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(res, f, indent=2)

#!/usr/bin/env python3
"""Run one trust-region/line-search benchmark case and write JSON output."""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

from mpi4py import MPI

from src.core.benchmark.run_record import atomic_write_json, strict_json_dumps


def _configure_thread_env(nproc_threads: int) -> None:
    threads = max(1, int(nproc_threads))
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false "
        "--xla_force_host_platform_device_count=1"
    )
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one trust-region benchmark case",
    )
    parser.add_argument("--problem", choices=("plaplace", "gl", "he"), required=True)
    parser.add_argument("--backend", choices=("fenics", "sfd", "element"), required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument(
        "--state-in",
        type=str,
        default="",
        help="Optional canonical NPZ initial state loaded before the nonlinear solve",
    )
    parser.add_argument(
        "--state-out",
        type=str,
        default="",
        help="Optional NPZ path for the final JAX/PETSc state",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-history", action="store_true")
    parser.add_argument("--save-linear-timing", action="store_true")

    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=24)

    parser.add_argument("--ksp-type", type=str, default=None)
    parser.add_argument("--pc-type", type=str, default="hypre")
    parser.add_argument("--ksp-rtol", type=float, default=1e-3)
    parser.add_argument("--ksp-max-it", type=int, default=10000)
    parser.add_argument("--profile", type=str, default="reference")
    parser.add_argument("--coloring-trials", type=int, default=10)
    parser.add_argument("--nproc-threads", type=int, default=1)
    parser.add_argument("--hvp-eval-mode", choices=("batched", "sequential"), default="sequential")
    parser.add_argument(
        "--element-reorder-mode",
        choices=("none", "block_rcm", "block_xyz", "block_metis"),
        default=None,
    )
    parser.add_argument(
        "--local-hessian-mode",
        choices=("element", "sfd_local", "sfd_local_vmap"),
        default=None,
    )
    parser.add_argument(
        "--problem-build-mode",
        choices=("rank_local", "replicated"),
        default=None,
    )
    parser.add_argument(
        "--he-mesh-source",
        choices=("procedural", "hdf5"),
        default=None,
        help="HE rank-local element mode mesh source.",
    )
    parser.add_argument(
        "--he-element-degree",
        "--elem-degree",
        dest="he_element_degree",
        choices=(1, 4),
        type=int,
        default=1,
        help="HE rank-local procedural element degree.",
    )
    parser.add_argument(
        "--distribution-strategy",
        choices=("overlap_p2p", "overlap_allgather"),
        default=None,
    )
    parser.add_argument(
        "--assembly-backend",
        choices=("coo_local", "coo"),
        default=None,
    )
    parser.add_argument(
        "--he-pmg-coarsest-level",
        type=str,
        default="1",
        help="HE PCMG coarsest level, or 'auto' to choose from MPI rank count.",
    )
    parser.add_argument("--he-pmg-auto-min-dofs-per-rank", type=int, default=128)
    parser.add_argument("--he-pmg-smoother-ksp-type", type=str, default="chebyshev")
    parser.add_argument("--he-pmg-smoother-pc-type", type=str, default="jacobi")
    parser.add_argument("--he-pmg-smoother-steps", type=int, default=2)
    parser.add_argument("--he-pmg-coarse-ksp-type", type=str, default="")
    parser.add_argument("--he-pmg-coarse-pc-type", type=str, default="hypre")
    parser.add_argument("--he-pmg-coarse-redundant-number", type=int, default=0)
    parser.add_argument("--he-pmg-coarse-telescope-reduction-factor", type=int, default=0)
    parser.add_argument("--he-pmg-coarse-factor-solver-type", type=str, default="")
    parser.add_argument("--he-pmg-coarse-hypre-nodal-coarsen", type=int, default=6)
    parser.add_argument("--he-pmg-coarse-hypre-vec-interp-variant", type=int, default=3)
    parser.add_argument("--he-pmg-coarse-hypre-strong-threshold", type=float, default=None)
    parser.add_argument("--he-pmg-coarse-hypre-coarsen-type", type=str, default="")
    parser.add_argument("--he-pmg-coarse-hypre-max-iter", type=int, default=2)
    parser.add_argument("--he-pmg-coarse-hypre-tol", type=float, default=0.0)
    parser.add_argument(
        "--he-pmg-coarse-hypre-relax-type-all",
        type=str,
        default="symmetric-SOR/Jacobi",
    )
    parser.add_argument(
        "--he-pmg-galerkin",
        choices=("both", "pmat", "mat"),
        default="both",
    )

    parser.add_argument("--tolf", type=float, default=1e-4)
    parser.add_argument("--tolg", type=float, default=1e-3)
    parser.add_argument("--tolg-rel", type=float, default=1e-3)
    parser.add_argument("--tolx-rel", type=float, default=1e-3)
    parser.add_argument("--tolx-abs", type=float, default=1e-10)
    parser.add_argument(
        "--convergence-metric",
        choices=("coefficient_l2", "lumped_l2", "reference_elastic_energy"),
        default="coefficient_l2",
        help=(
            "Nonlinear stopping metric: lumped_l2 is the scalar P1 map and "
            "reference_elastic_energy is the JAX/PETSc HE mechanics map."
        ),
    )
    parser.add_argument(
        "--convergence-state-scale",
        type=float,
        default=None,
        help="Positive absolute state scale in the selected primal norm.",
    )
    parser.add_argument("--riesz-ksp-type", type=str, default="cg")
    parser.add_argument("--riesz-pc-type", type=str, default="jacobi")
    parser.add_argument("--riesz-ksp-rtol", type=float, default=1.0e-10)
    parser.add_argument("--riesz-ksp-atol", type=float, default=1.0e-14)
    parser.add_argument("--riesz-ksp-max-it", type=int, default=5000)
    parser.add_argument("--riesz-true-residual-rtol", type=float, default=1.0e-8)
    parser.add_argument(
        "--riesz-spd-factor-solver-type",
        type=str,
        default="mumps",
    )
    parser.add_argument("--riesz-symmetry-tol", type=float, default=1.0e-12)
    parser.add_argument("--maxit", type=int, default=100)
    parser.add_argument("--step-time-limit-s", type=float, default=None)

    parser.add_argument("--linesearch-a", type=float, default=-0.5)
    parser.add_argument("--linesearch-b", type=float, default=2.0)
    parser.add_argument("--linesearch-tol", type=float, default=1e-3)
    parser.add_argument(
        "--line-search",
        choices=("golden_fixed", "armijo"),
        default="golden_fixed",
        help="Line search used inside Newton/trust-region globalization.",
    )

    parser.add_argument(
        "--use-trust-region",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--trust-radius-init", type=float, default=1.0)
    parser.add_argument("--trust-radius-min", type=float, default=1e-8)
    parser.add_argument("--trust-radius-max", type=float, default=1e6)
    parser.add_argument("--trust-shrink", type=float, default=0.5)
    parser.add_argument("--trust-expand", type=float, default=1.5)
    parser.add_argument("--trust-eta-shrink", type=float, default=0.05)
    parser.add_argument("--trust-eta-expand", type=float, default=0.75)
    parser.add_argument("--trust-max-reject", type=int, default=6)
    parser.add_argument(
        "--trust-subproblem-line-search",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--local-coloring",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--retry-on-failure",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--pc-setup-on-ksp-cap",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use-near-nullspace",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reorder",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--gamg-threshold", type=float, default=-1.0)
    parser.add_argument("--gamg-agg-nsmooths", type=int, default=1)
    parser.add_argument(
        "--gamg-set-coordinates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _run_plaplace(args):
    if args.backend == "fenics":
        from src.problems.plaplace.fenics.solver_custom_newton import run
    else:
        _configure_thread_env(args.nproc_threads)
        from src.problems.plaplace.jax_petsc.solver import run

    ns = SimpleNamespace(
        level=args.level,
        quiet=args.quiet,
        save_history=args.save_history,
        save_linear_timing=args.save_linear_timing,
        profile=args.profile,
        ksp_type=args.ksp_type or "cg",
        pc_type=args.pc_type,
        ksp_rtol=args.ksp_rtol,
        ksp_max_it=args.ksp_max_it,
        pc_setup_on_ksp_cap=args.pc_setup_on_ksp_cap,
        gamg_threshold=args.gamg_threshold,
        gamg_agg_nsmooths=args.gamg_agg_nsmooths,
        gamg_set_coordinates=args.gamg_set_coordinates,
        reorder=args.reorder,
        local_coloring=bool(args.local_coloring),
        hvp_eval_mode=args.hvp_eval_mode,
        coloring_trials=args.coloring_trials,
        assembly_mode=args.backend,
        element_reorder_mode=args.element_reorder_mode,
        local_hessian_mode=args.local_hessian_mode,
        tolf=args.tolf,
        tolg=args.tolg,
        tolg_rel=args.tolg_rel,
        tolx_rel=args.tolx_rel,
        tolx_abs=args.tolx_abs,
        maxit=args.maxit,
        linesearch_a=args.linesearch_a,
        linesearch_b=args.linesearch_b,
        linesearch_tol=args.linesearch_tol,
        line_search=args.line_search,
        retry_on_failure=bool(args.retry_on_failure),
        nproc=args.nproc_threads,
        out="",
        state_in=args.state_in,
        state_out=args.state_out,
        use_trust_region=args.use_trust_region,
        trust_radius_init=args.trust_radius_init,
        trust_radius_min=args.trust_radius_min,
        trust_radius_max=args.trust_radius_max,
        trust_shrink=args.trust_shrink,
        trust_expand=args.trust_expand,
        trust_eta_shrink=args.trust_eta_shrink,
        trust_eta_expand=args.trust_eta_expand,
        trust_max_reject=args.trust_max_reject,
        trust_subproblem_line_search=args.trust_subproblem_line_search,
        step_time_limit_s=args.step_time_limit_s,
        convergence_metric=args.convergence_metric,
        convergence_state_scale=args.convergence_state_scale,
    )
    return run(ns)


def _run_gl(args):
    if args.backend == "fenics":
        from src.problems.ginzburg_landau.fenics.solver_custom_newton import run
    else:
        _configure_thread_env(args.nproc_threads)
        from src.problems.ginzburg_landau.jax_petsc.solver import run

    ns = SimpleNamespace(
        level=args.level,
        quiet=args.quiet,
        save_history=args.save_history,
        save_linear_timing=args.save_linear_timing,
        profile=args.profile,
        ksp_type=args.ksp_type or "gmres",
        pc_type=args.pc_type,
        ksp_rtol=args.ksp_rtol,
        ksp_max_it=args.ksp_max_it,
        pc_setup_on_ksp_cap=args.pc_setup_on_ksp_cap,
        gamg_threshold=args.gamg_threshold,
        gamg_agg_nsmooths=args.gamg_agg_nsmooths,
        gamg_set_coordinates=args.gamg_set_coordinates,
        reorder=args.reorder,
        local_coloring=bool(args.local_coloring),
        hvp_eval_mode=args.hvp_eval_mode,
        coloring_trials=args.coloring_trials,
        assembly_mode=args.backend,
        element_reorder_mode=args.element_reorder_mode,
        local_hessian_mode=args.local_hessian_mode,
        tolf=args.tolf,
        tolg=args.tolg,
        tolg_rel=args.tolg_rel,
        tolx_rel=args.tolx_rel,
        tolx_abs=args.tolx_abs,
        maxit=args.maxit,
        linesearch_a=args.linesearch_a,
        linesearch_b=args.linesearch_b,
        linesearch_tol=args.linesearch_tol,
        line_search=args.line_search,
        retry_on_failure=bool(args.retry_on_failure),
        nproc=args.nproc_threads,
        out="",
        state_in=args.state_in,
        state_out=args.state_out,
        use_trust_region=args.use_trust_region,
        trust_radius_init=args.trust_radius_init,
        trust_radius_min=args.trust_radius_min,
        trust_radius_max=args.trust_radius_max,
        trust_shrink=args.trust_shrink,
        trust_expand=args.trust_expand,
        trust_eta_shrink=args.trust_eta_shrink,
        trust_eta_expand=args.trust_eta_expand,
        trust_max_reject=args.trust_max_reject,
        trust_subproblem_line_search=args.trust_subproblem_line_search,
        step_time_limit_s=args.step_time_limit_s,
        convergence_metric=args.convergence_metric,
        convergence_state_scale=args.convergence_state_scale,
    )
    return run(ns)


def _run_he(args):
    if args.convergence_metric == "lumped_l2":
        raise ValueError(
            "Hyperelasticity does not use the scalar lumped_l2 metric; select "
            "coefficient_l2 or reference_elastic_energy."
        )
    if (
        args.backend == "fenics"
        and args.convergence_metric == "reference_elastic_energy"
    ):
        raise ValueError(
            "reference_elastic_energy is currently certified only for the "
            "JAX/PETSc HyperElasticity backends, not the FEniCS runner."
        )
    linesearch_interval = (args.linesearch_a, args.linesearch_b)
    common = dict(
        mesh_level=args.level,
        num_steps=args.steps,
        verbose=(not args.quiet),
        maxit=args.maxit,
        start_step=args.start_step,
        linesearch_interval=linesearch_interval,
        linesearch_tol=args.linesearch_tol,
        ksp_type=args.ksp_type or "gmres",
        pc_type=args.pc_type,
        ksp_rtol=args.ksp_rtol,
        ksp_max_it=args.ksp_max_it,
        total_steps=args.total_steps,
        tolf=args.tolf,
        tolg=args.tolg,
        tolg_rel=args.tolg_rel,
        tolx_rel=args.tolx_rel,
        tolx_abs=args.tolx_abs,
        save_history=args.save_history,
        save_linear_timing=args.save_linear_timing,
        pc_setup_on_ksp_cap=args.pc_setup_on_ksp_cap,
        gamg_threshold=args.gamg_threshold,
        gamg_agg_nsmooths=args.gamg_agg_nsmooths,
        gamg_set_coordinates=args.gamg_set_coordinates,
        use_near_nullspace=args.use_near_nullspace,
        fail_fast=False,
        retry_on_nonfinite=bool(args.retry_on_failure),
        retry_on_maxit=bool(args.retry_on_failure),
        use_trust_region=args.use_trust_region,
        trust_radius_init=args.trust_radius_init,
        trust_radius_min=args.trust_radius_min,
        trust_radius_max=args.trust_radius_max,
        trust_shrink=args.trust_shrink,
        trust_expand=args.trust_expand,
        trust_eta_shrink=args.trust_eta_shrink,
        trust_eta_expand=args.trust_eta_expand,
        trust_max_reject=args.trust_max_reject,
        trust_subproblem_line_search=args.trust_subproblem_line_search,
        step_time_limit_s=args.step_time_limit_s,
    )

    if args.backend == "fenics":
        from src.problems.hyperelasticity.fenics.solver_custom_newton import run_level

        return run_level(**common)

    _configure_thread_env(args.nproc_threads)
    from src.problems.hyperelasticity.jax_petsc.solver import run

    ns = SimpleNamespace(
        level=args.level,
        steps=args.steps,
        total_steps=args.total_steps,
        start_step=args.start_step,
        profile=args.profile,
        ksp_type=args.ksp_type,
        pc_type=args.pc_type,
        ksp_rtol=args.ksp_rtol,
        ksp_max_it=args.ksp_max_it,
        pc_setup_on_ksp_cap=args.pc_setup_on_ksp_cap,
        gamg_threshold=args.gamg_threshold,
        gamg_agg_nsmooths=args.gamg_agg_nsmooths,
        gamg_set_coordinates=args.gamg_set_coordinates,
        use_near_nullspace=args.use_near_nullspace,
        reorder=args.reorder,
        local_coloring=bool(args.local_coloring),
        hvp_eval_mode=args.hvp_eval_mode,
        coloring_trials=args.coloring_trials,
        assembly_mode=args.backend,
        element_reorder_mode=args.element_reorder_mode,
        local_hessian_mode=args.local_hessian_mode,
        problem_build_mode=args.problem_build_mode,
        mesh_source=args.he_mesh_source,
        he_element_degree=args.he_element_degree,
        distribution_strategy=args.distribution_strategy,
        assembly_backend=args.assembly_backend,
        he_pmg_coarsest_level=args.he_pmg_coarsest_level,
        he_pmg_auto_min_dofs_per_rank=args.he_pmg_auto_min_dofs_per_rank,
        he_pmg_smoother_ksp_type=args.he_pmg_smoother_ksp_type,
        he_pmg_smoother_pc_type=args.he_pmg_smoother_pc_type,
        he_pmg_smoother_steps=args.he_pmg_smoother_steps,
        he_pmg_coarse_ksp_type=args.he_pmg_coarse_ksp_type,
        he_pmg_coarse_pc_type=args.he_pmg_coarse_pc_type,
        he_pmg_coarse_redundant_number=args.he_pmg_coarse_redundant_number,
        he_pmg_coarse_telescope_reduction_factor=args.he_pmg_coarse_telescope_reduction_factor,
        he_pmg_coarse_factor_solver_type=args.he_pmg_coarse_factor_solver_type,
        he_pmg_coarse_hypre_nodal_coarsen=args.he_pmg_coarse_hypre_nodal_coarsen,
        he_pmg_coarse_hypre_vec_interp_variant=args.he_pmg_coarse_hypre_vec_interp_variant,
        he_pmg_coarse_hypre_strong_threshold=args.he_pmg_coarse_hypre_strong_threshold,
        he_pmg_coarse_hypre_coarsen_type=args.he_pmg_coarse_hypre_coarsen_type,
        he_pmg_coarse_hypre_max_iter=args.he_pmg_coarse_hypre_max_iter,
        he_pmg_coarse_hypre_tol=args.he_pmg_coarse_hypre_tol,
        he_pmg_coarse_hypre_relax_type_all=args.he_pmg_coarse_hypre_relax_type_all,
        he_pmg_galerkin=args.he_pmg_galerkin,
        tolf=args.tolf,
        tolg=args.tolg,
        tolg_rel=args.tolg_rel,
        tolx_rel=args.tolx_rel,
        tolx_abs=args.tolx_abs,
        maxit=args.maxit,
        linesearch_a=args.linesearch_a,
        linesearch_b=args.linesearch_b,
        linesearch_tol=args.linesearch_tol,
        line_search=args.line_search,
        retry_on_failure=bool(args.retry_on_failure),
        stop_on_fail=False,
        use_abs_det=False,
        nproc=args.nproc_threads,
        save_history=args.save_history,
        save_linear_timing=args.save_linear_timing,
        quiet=args.quiet,
        out="",
        state_in=args.state_in,
        state_out=args.state_out,
        use_trust_region=args.use_trust_region,
        trust_radius_init=args.trust_radius_init,
        trust_radius_min=args.trust_radius_min,
        trust_radius_max=args.trust_radius_max,
        trust_shrink=args.trust_shrink,
        trust_expand=args.trust_expand,
        trust_eta_shrink=args.trust_eta_shrink,
        trust_eta_expand=args.trust_eta_expand,
        trust_max_reject=args.trust_max_reject,
        trust_subproblem_line_search=args.trust_subproblem_line_search,
        step_time_limit_s=args.step_time_limit_s,
        convergence_metric=args.convergence_metric,
        convergence_state_scale=args.convergence_state_scale,
        riesz_ksp_type=args.riesz_ksp_type,
        riesz_pc_type=args.riesz_pc_type,
        riesz_ksp_rtol=args.riesz_ksp_rtol,
        riesz_ksp_atol=args.riesz_ksp_atol,
        riesz_ksp_max_it=args.riesz_ksp_max_it,
        riesz_true_residual_rtol=args.riesz_true_residual_rtol,
        riesz_spd_factor_solver_type=args.riesz_spd_factor_solver_type,
        riesz_symmetry_tol=args.riesz_symmetry_tol,
    )
    return run(ns)


def _write_payload(path: str, payload: dict[str, object]) -> None:
    """Write one durable RFC-compliant case result.

    Iteration histories use non-finite floating-point values internally for
    diagnostics that do not apply to a given algorithm.  JSON has no NaN or
    infinity literals, so those optional sentinels are represented as null at
    the serialization boundary.
    """

    atomic_write_json(path, payload, nonfinite_as_null=True)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.problem == "plaplace":
        result = _run_plaplace(args)
    elif args.problem == "gl":
        result = _run_gl(args)
    else:
        result = _run_he(args)

    payload = {
        "case": vars(args),
        "result": result,
    }

    if MPI.COMM_WORLD.rank == 0:
        _write_payload(args.out, payload)
        print(strict_json_dumps(payload, indent=2, nonfinite_as_null=True))


if __name__ == "__main__":
    main()

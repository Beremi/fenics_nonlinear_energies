#!/usr/bin/env python3
"""Experimental JAX+PETSc slope-stability solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpi4py import MPI

from src.core.cli.threading import configure_jax_cpu_threading
from src.problems.slope_stability.support import DEFAULT_LEVEL


def _build_parser(profile_defaults):
    parser = argparse.ArgumentParser(
        description="Experimental slope-stability DOF-partitioned JAX + PETSc solver"
    )
    parser.add_argument("--case", type=str, default="")
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    parser.add_argument(
        "--elem_degree",
        type=int,
        choices=(1, 2, 4),
        default=2,
        help="Fine-space Lagrange degree for the operator",
    )
    parser.add_argument("--lambda-target", type=float, default=1.2)

    parser.add_argument(
        "--profile",
        choices=sorted(profile_defaults.keys()),
        default="performance",
        help="Linear solver profile",
    )

    parser.add_argument("--ksp_type", type=str, default=None, help="PETSc KSP type")
    parser.add_argument("--pc_type", type=str, default=None, help="PETSc PC type")
    parser.add_argument("--ksp_rtol", type=float, default=None, help="KSP relative tolerance")
    parser.add_argument("--ksp_max_it", type=int, default=None, help="KSP maximum iterations")
    parser.add_argument(
        "--ksp_accept_true_rel",
        type=float,
        default=None,
        help="Accept a KSP MAX_IT solve when the measured true relative residual stays below this cap",
    )
    parser.add_argument(
        "--pc_setup_on_ksp_cap",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--pc_reuse_preconditioner",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ask PETSc to reuse the preconditioner setup across changing operators",
    )
    parser.add_argument("--gamg_threshold", type=float, default=None)
    parser.add_argument("--gamg_agg_nsmooths", type=int, default=None)
    parser.add_argument(
        "--hypre_nodal_coarsen",
        type=int,
        default=None,
        help="BoomerAMG nodal coarsen (-1 to skip setting)",
    )
    parser.add_argument(
        "--hypre_vec_interp_variant",
        type=int,
        default=None,
        help="BoomerAMG vector interpolation variant (-1 to skip setting)",
    )
    parser.add_argument(
        "--hypre_strong_threshold",
        type=float,
        default=None,
        help="BoomerAMG strong threshold",
    )
    parser.add_argument(
        "--hypre_coarsen_type",
        type=str,
        default=None,
        help="BoomerAMG coarsen type, e.g. HMIS or PMIS",
    )
    parser.add_argument(
        "--hypre_max_iter",
        type=int,
        default=None,
        help="BoomerAMG V-cycles per preconditioner application",
    )
    parser.add_argument(
        "--hypre_tol",
        type=float,
        default=None,
        help="BoomerAMG tolerance used inside one preconditioner application",
    )
    parser.add_argument(
        "--hypre_relax_type_all",
        type=str,
        default=None,
        help="BoomerAMG relax_type_all, e.g. symmetric-SOR/Jacobi",
    )
    parser.add_argument(
        "--gamg_set_coordinates",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use_near_nullspace",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--reorder",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument(
        "--assembly_mode",
        choices=("element",),
        default="element",
    )
    parser.add_argument(
        "--operator_mode",
        choices=("assembled", "matfree_element", "matfree_overlap"),
        default="assembled",
        help="Fine-level operator representation used by PETSc KSP",
    )
    parser.add_argument(
        "--distribution_strategy",
        choices=("overlap_allgather", "overlap_p2p"),
        default="overlap_p2p",
        help="Distributed overlap-state exchange used by the reordered assembler",
    )
    parser.add_argument(
        "--problem_build_mode",
        choices=("replicated", "root_bcast", "rank_local"),
        default="root_bcast",
        help="How same-mesh problem data is constructed across MPI ranks",
    )
    parser.add_argument(
        "--mg_level_build_mode",
        choices=("replicated", "root_bcast", "rank_local"),
        default="root_bcast",
        help="How lower multigrid levels are constructed across MPI ranks",
    )
    parser.add_argument(
        "--mg_transfer_build_mode",
        choices=("replicated", "root_bcast", "owned_rows"),
        default="owned_rows",
        help="How mixed-order prolongation/restriction operators are built across MPI ranks",
    )
    parser.add_argument(
        "--element_reorder_mode",
        choices=("none", "block_rcm", "block_xyz", "block_metis"),
        default=None,
    )
    parser.add_argument(
        "--local_hessian_mode",
        choices=("element", "sfd_local", "sfd_local_vmap"),
        default=None,
    )
    parser.add_argument(
        "--local_coloring",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--preconditioner_operator",
        choices=("same_operator", "refined_p1_same_nodes"),
        default="same_operator",
        help="Matrix used for PETSc preconditioning",
    )
    parser.add_argument(
        "--mg_coarsest_level",
        type=int,
        default=1,
        help="Coarsest geometric level used when --pc_type mg",
    )
    parser.add_argument(
        "--mg_strategy",
        choices=(
            "legacy_p2_h",
            "same_mesh_p2_p1",
            "same_mesh_p2_p1_lminus1_p1",
            "same_mesh_p4_p1",
            "same_mesh_p4_p2_p1",
            "same_mesh_p4_p1_lminus1_p1",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "custom_mixed",
        ),
        default="legacy_p2_h",
        help="Mixed-order multigrid hierarchy strategy when --pc_type mg",
    )
    parser.add_argument(
        "--mg_custom_hierarchy",
        type=str,
        default=None,
        help="Comma-separated custom MG hierarchy, coarse-to-fine, e.g. '1:1,2:1,6:2,6:4'",
    )
    parser.add_argument(
        "--mg_variant",
        choices=("auto", "legacy_pmg", "explicit_pmg", "outer_pcksp"),
        default="auto",
        help="PETSc multigrid implementation variant",
    )
    parser.add_argument(
        "--fine_pmat_policy",
        choices=(
            "same_operator",
            "elastic_frozen",
            "initial_tangent_frozen",
            "staggered_whole",
            "staggered_smoother_only",
        ),
        default="same_operator",
        help="Fine-level Pmat policy for matrix-free legacy PCMG on same-mesh P4 hierarchies",
    )
    parser.add_argument(
        "--fine_pmat_stagger_period",
        type=int,
        default=2,
        help="Newton-step cadence for staggered fine-Pmat updates",
    )
    parser.add_argument(
        "--mg_lower_operator_policy",
        choices=("refresh_each_newton", "fixed_setup", "galerkin_refresh"),
        default="refresh_each_newton",
        help="How assembled lower-level operators are updated for explicit MG variants",
    )
    parser.add_argument("--mg_p4_smoother_ksp_type", type=str, default=None)
    parser.add_argument("--mg_p4_smoother_pc_type", type=str, default=None)
    parser.add_argument("--mg_p4_smoother_steps", type=int, default=None)
    parser.add_argument("--mg_p2_smoother_ksp_type", type=str, default=None)
    parser.add_argument("--mg_p2_smoother_pc_type", type=str, default=None)
    parser.add_argument("--mg_p2_smoother_steps", type=int, default=None)
    parser.add_argument("--mg_p1_smoother_ksp_type", type=str, default=None)
    parser.add_argument("--mg_p1_smoother_pc_type", type=str, default=None)
    parser.add_argument("--mg_p1_smoother_steps", type=int, default=None)
    parser.add_argument("--mg_fine_ksp_type", type=str, default="richardson")
    parser.add_argument("--mg_fine_pc_type", type=str, default="none")
    parser.add_argument(
        "--mg_fine_python_pc_variant",
        choices=("none", "overlap_lu"),
        default="none",
    )
    parser.add_argument("--mg_fine_steps", type=int, default=2)
    parser.add_argument("--mg_fine_down_ksp_type", type=str, default=None)
    parser.add_argument("--mg_fine_down_pc_type", type=str, default=None)
    parser.add_argument("--mg_fine_down_steps", type=int, default=None)
    parser.add_argument("--mg_fine_up_ksp_type", type=str, default=None)
    parser.add_argument("--mg_fine_up_pc_type", type=str, default=None)
    parser.add_argument("--mg_fine_up_steps", type=int, default=None)
    parser.add_argument("--mg_intermediate_steps", type=int, default=3)
    parser.add_argument("--mg_intermediate_pc_type", type=str, default="jacobi")
    parser.add_argument("--mg_degree1_pc_type", type=str, default=None)
    parser.add_argument("--mg_degree2_pc_type", type=str, default=None)
    parser.add_argument("--mg_coarse_ksp_type", type=str, default=None)
    parser.add_argument("--mg_coarse_pc_type", type=str, default=None)
    parser.add_argument(
        "--mg_coarse_backend",
        choices=(
            "hypre",
            "lu",
            "jacobi",
            "redundant_lu",
            "redundant_hypre",
            "rank0_lu_broadcast",
            "rank0_hypre_broadcast",
        ),
        default=None,
        help="Convenience selector for the MG coarse solver PC backend",
    )
    parser.add_argument(
        "--mg_coarse_hypre_nodal_coarsen",
        type=int,
        default=6,
        help="BoomerAMG nodal coarsen used on the MG coarse solve",
    )
    parser.add_argument(
        "--mg_coarse_hypre_vec_interp_variant",
        type=int,
        default=3,
        help="BoomerAMG vector interpolation variant used on the MG coarse solve",
    )
    parser.add_argument(
        "--mg_coarse_hypre_strong_threshold",
        type=float,
        default=0.5,
        help="BoomerAMG strong-threshold override for the MG coarse solve",
    )
    parser.add_argument(
        "--mg_coarse_hypre_coarsen_type",
        type=str,
        default="HMIS",
        help="BoomerAMG coarsen type used on the MG coarse solve",
    )
    parser.add_argument(
        "--mg_coarse_hypre_max_iter",
        type=int,
        default=2,
        help="BoomerAMG V-cycles per MG coarse preconditioner application",
    )
    parser.add_argument(
        "--mg_coarse_hypre_tol",
        type=float,
        default=0.0,
        help="BoomerAMG tolerance used inside each MG coarse preconditioner application",
    )
    parser.add_argument(
        "--mg_coarse_hypre_relax_type_all",
        type=str,
        default="symmetric-SOR/Jacobi",
        help="BoomerAMG relax_type_all used on the MG coarse solve",
    )
    parser.add_argument("--outer_pcksp_inner_ksp_type", type=str, default="fgmres")
    parser.add_argument("--outer_pcksp_inner_ksp_rtol", type=float, default=1e-2)
    parser.add_argument("--outer_pcksp_inner_ksp_max_it", type=int, default=20)
    parser.add_argument(
        "--python_pc_variant",
        choices=("none", "overlap_lu"),
        default="none",
        help="Custom PETSc Python preconditioner used when --pc_type python",
    )

    parser.add_argument("--tolf", type=float, default=1e-4)
    parser.add_argument("--tolg", type=float, default=1e-3)
    parser.add_argument("--tolg_rel", type=float, default=1e-3)
    parser.add_argument("--tolx_rel", type=float, default=1e-3)
    parser.add_argument("--tolx_abs", type=float, default=1e-10)
    parser.add_argument("--maxit", type=int, default=100)
    parser.add_argument(
        "--line_search",
        choices=("golden_fixed", "armijo"),
        default="golden_fixed",
        help="Newton line-search policy",
    )
    parser.add_argument("--armijo_alpha0", type=float, default=1.0)
    parser.add_argument("--armijo_c1", type=float, default=1e-4)
    parser.add_argument("--armijo_shrink", type=float, default=0.5)
    parser.add_argument("--armijo_max_ls", type=int, default=40)
    parser.add_argument(
        "--step_time_limit_s",
        type=float,
        default=None,
        help="Optional per-step wall-time limit passed into the Newton loop",
    )
    parser.add_argument(
        "--benchmark_mode",
        choices=("end_to_end", "warmup_once_then_solve"),
        default="end_to_end",
        help="Report either the full end-to-end timing or a steady-state timing with warmup separated",
    )

    parser.add_argument("--linesearch_a", type=float, default=-0.5)
    parser.add_argument("--linesearch_b", type=float, default=2.0)
    parser.add_argument("--linesearch_tol", type=float, default=1e-1)
    parser.add_argument(
        "--trust_subproblem_line_search",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use_trust_region",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--trust_radius_init", type=float, default=0.5)
    parser.add_argument("--trust_radius_min", type=float, default=1e-8)
    parser.add_argument("--trust_radius_max", type=float, default=1e6)
    parser.add_argument("--trust_shrink", type=float, default=0.5)
    parser.add_argument("--trust_expand", type=float, default=1.5)
    parser.add_argument("--trust_eta_shrink", type=float, default=0.05)
    parser.add_argument("--trust_eta_expand", type=float, default=0.75)
    parser.add_argument("--trust_max_reject", type=int, default=6)

    parser.add_argument(
        "--retry_on_failure",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--accept_ksp_maxit_direction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When KSP hits DIVERGED_MAX_IT, keep the capped linear step as the Newton direction instead of aborting the nonlinear step",
    )
    parser.add_argument(
        "--guard_ksp_maxit_direction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require extra quality checks before accepting a DIVERGED_MAX_IT linear step as the Newton direction",
    )
    parser.add_argument(
        "--ksp_maxit_direction_true_rel_cap",
        type=float,
        default=6.0e-2,
        help="Maximum true relative residual allowed when accepting a capped DIVERGED_MAX_IT direction",
    )
    parser.add_argument("--reg", type=float, default=1.0e-12)
    parser.add_argument(
        "--reuse_hessian_value_buffers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse persistent COO-value buffers on the assembled Hessian hot path",
    )
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--save-history", action="store_true")
    parser.add_argument("--save-linear-timing", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--state-out", type=str, default="")
    return parser


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--nproc", type=int, default=1)
    pre_args, _ = pre_parser.parse_known_args()
    configure_jax_cpu_threading(pre_args.nproc)

    from src.problems.slope_stability.jax_petsc.solver import PROFILE_DEFAULTS, run

    parser = _build_parser(PROFILE_DEFAULTS)
    args = parser.parse_args()
    result = run(args)

    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(result, indent=2))
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

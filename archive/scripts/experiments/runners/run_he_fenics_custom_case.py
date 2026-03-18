#!/usr/bin/env python3
"""Run one FEniCS custom HE case and write JSON output."""

from __future__ import annotations

import argparse
import json

from mpi4py import MPI

from src.problems.hyperelasticity.fenics.solver_custom_newton import run_level


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--maxit", type=int, default=100)

    parser.add_argument("--linesearch-a", type=float, default=-0.5)
    parser.add_argument("--linesearch-b", type=float, default=2.0)
    parser.add_argument("--linesearch-tol", type=float, default=1e-3)

    parser.add_argument("--ksp-type", type=str, default="gmres")
    parser.add_argument("--pc-type", type=str, default="gamg")
    parser.add_argument("--ksp-rtol", type=float, default=1e-1)
    parser.add_argument("--ksp-max-it", type=int, default=30)
    parser.add_argument("--gamg-threshold", type=float, default=0.05)
    parser.add_argument("--gamg-agg-nsmooths", type=int, default=1)
    parser.add_argument(
        "--pc-setup-on-ksp-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gamg-set-coordinates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-near-nullspace",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--tolf", type=float, default=1e-4)
    parser.add_argument("--tolg", type=float, default=1e-3)
    parser.add_argument("--tolg-rel", type=float, default=1e-3)
    parser.add_argument("--tolx-rel", type=float, default=1e-3)
    parser.add_argument("--tolx-abs", type=float, default=1e-10)
    parser.add_argument(
        "--require-all-convergence",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        "--retry-on-nonfinite",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--retry-on-maxit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-history", action="store_true")
    parser.add_argument("--save-linear-timing", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--step-time-limit-s", type=float, default=None)
    parser.add_argument("--out", type=str, required=True)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    result = run_level(
        mesh_level=int(args.level),
        num_steps=int(args.steps),
        verbose=(not args.quiet),
        maxit=int(args.maxit),
        start_step=int(args.start_step),
        linesearch_interval=(float(args.linesearch_a), float(args.linesearch_b)),
        linesearch_tol=float(args.linesearch_tol),
        ksp_type=str(args.ksp_type),
        pc_type=str(args.pc_type),
        ksp_rtol=float(args.ksp_rtol),
        ksp_max_it=int(args.ksp_max_it),
        use_near_nullspace=bool(args.use_near_nullspace),
        total_steps=int(args.total_steps),
        tolf=float(args.tolf),
        tolg=float(args.tolg),
        tolg_rel=float(args.tolg_rel),
        tolx_rel=float(args.tolx_rel),
        tolx_abs=float(args.tolx_abs),
        save_history=bool(args.save_history),
        save_linear_timing=bool(args.save_linear_timing),
        pc_setup_on_ksp_cap=bool(args.pc_setup_on_ksp_cap),
        gamg_threshold=float(args.gamg_threshold),
        gamg_agg_nsmooths=int(args.gamg_agg_nsmooths),
        gamg_set_coordinates=bool(args.gamg_set_coordinates),
        require_all_convergence=bool(args.require_all_convergence),
        fail_fast=bool(args.fail_fast),
        retry_on_nonfinite=bool(args.retry_on_nonfinite),
        retry_on_maxit=bool(args.retry_on_maxit),
        use_trust_region=bool(args.use_trust_region),
        trust_radius_init=float(args.trust_radius_init),
        trust_radius_min=float(args.trust_radius_min),
        trust_radius_max=float(args.trust_radius_max),
        trust_shrink=float(args.trust_shrink),
        trust_expand=float(args.trust_expand),
        trust_eta_shrink=float(args.trust_eta_shrink),
        trust_eta_expand=float(args.trust_eta_expand),
        trust_max_reject=int(args.trust_max_reject),
        trust_subproblem_line_search=bool(args.trust_subproblem_line_search),
        step_time_limit_s=(
            None if args.step_time_limit_s is None else float(args.step_time_limit_s)
        ),
    )

    if MPI.COMM_WORLD.rank == 0:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

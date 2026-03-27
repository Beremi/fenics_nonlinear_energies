#!/usr/bin/env python3
"""MPI-capable certified JAX + PETSc solve for the arctan-resonance family."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mpi4py import MPI

from src.core.cli.threading import configure_jax_cpu_threading


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--lambda1", type=float, required=True)
    parser.add_argument("--lambda-level", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=1.0e-8)
    parser.add_argument("--maxit", type=int, default=80)
    parser.add_argument("--init-state", type=str, default="")
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument("--handoff-source", type=str, default="init_state")
    parser.add_argument("--pc-type", choices=("mg", "hypre", "gamg"), default="mg")
    parser.add_argument("--ksp-type", type=str, default="fgmres")
    parser.add_argument("--ksp-rtol", type=float, default=1.0e-8)
    parser.add_argument("--ksp-max-it", type=int, default=400)
    parser.add_argument("--merit-ksp-type", type=str, default="cg")
    parser.add_argument("--merit-ksp-rtol", type=float, default=1.0e-10)
    parser.add_argument("--merit-ksp-max-it", type=int, default=400)
    parser.add_argument("--mg-coarsest-level", type=int, default=2)
    parser.add_argument("--mg-smoother-ksp-type", type=str, default="richardson")
    parser.add_argument("--mg-smoother-pc-type", type=str, default="sor")
    parser.add_argument("--mg-smoother-steps", type=int, default=2)
    parser.add_argument("--mg-coarse-ksp-type", type=str, default="auto")
    parser.add_argument("--mg-coarse-pc-type", type=str, default="auto")
    parser.add_argument("--element-reorder-mode", choices=("none", "block_rcm", "block_xyz", "block_metis"), default="block_xyz")
    parser.add_argument("--local-hessian-mode", choices=("element", "sfd_local", "sfd_local_vmap"), default="element")
    parser.add_argument("--distribution-strategy", choices=("overlap_allgather", "overlap_p2p"), default="overlap_p2p")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--state-out", type=str, default="")
    parser.add_argument("--out", "--json", dest="out", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    return parser


args = build_parser().parse_args()
# This PETSc CLI is intended for one CPU thread per MPI rank.
threads = configure_jax_cpu_threading(1)

from jax import config  # noqa: E402

config.update("jax_enable_x64", True)

from src.problems.plaplace_up_arctan.jax_petsc.solver import (  # noqa: E402
    solve_certified_stationary_petsc,
    solve_from_state_path,
)
from src.problems.plaplace_up_arctan.solver_common import build_problem  # noqa: E402


def _coarse_solver_defaults(pc_type: str, comm_size: int) -> tuple[str, str]:
    if str(pc_type) != "mg":
        return "preonly", str(pc_type)
    if int(comm_size) == 1:
        return "preonly", "lu"
    return "cg", "hypre"


def run_from_args(parsed: argparse.Namespace) -> dict[str, object]:
    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    lambda_level = int(parsed.lambda_level) if int(parsed.lambda_level) > 0 else int(parsed.level)
    problem = build_problem(
        level=int(parsed.level),
        p=float(parsed.p),
        geometry="square_unit",
        init_mode="sine",
        lambda1=float(parsed.lambda1),
        lambda_level=lambda_level,
        seed=0,
    )
    coarse_ksp_type, coarse_pc_type = _coarse_solver_defaults(str(parsed.pc_type), int(comm.Get_size()))
    if str(parsed.mg_coarse_ksp_type) != "auto":
        coarse_ksp_type = str(parsed.mg_coarse_ksp_type)
    if str(parsed.mg_coarse_pc_type) != "auto":
        coarse_pc_type = str(parsed.mg_coarse_pc_type)

    kwargs = dict(
        epsilon=float(parsed.epsilon),
        maxit=int(parsed.maxit),
        state_out=str(parsed.state_out),
        handoff_source=str(parsed.handoff_source),
        ksp_type=str(parsed.ksp_type),
        ksp_rtol=float(parsed.ksp_rtol),
        ksp_max_it=int(parsed.ksp_max_it),
        merit_ksp_type=str(parsed.merit_ksp_type),
        merit_ksp_rtol=float(parsed.merit_ksp_rtol),
        merit_ksp_max_it=int(parsed.merit_ksp_max_it),
        pc_type=str(parsed.pc_type),
        reorder_mode=str(parsed.element_reorder_mode),
        local_hessian_mode=str(parsed.local_hessian_mode),
        distribution_strategy=str(parsed.distribution_strategy),
        mg_coarsest_level=int(parsed.mg_coarsest_level),
        mg_smoother_ksp_type=str(parsed.mg_smoother_ksp_type),
        mg_smoother_pc_type=str(parsed.mg_smoother_pc_type),
        mg_smoother_steps=int(parsed.mg_smoother_steps),
        mg_coarse_ksp_type=str(coarse_ksp_type),
        mg_coarse_pc_type=str(coarse_pc_type),
    )
    if parsed.init_state:
        result = solve_from_state_path(
            problem,
            init_state_path=str(parsed.init_state),
            init_scale=float(parsed.init_scale),
            **kwargs,
        )
    else:
        result = solve_certified_stationary_petsc(
            problem,
            init_free=float(parsed.init_scale) * problem.u_init,
            **kwargs,
        )

    payload = {
        "solver": "jax_petsc",
        "problem": {
            "name": "pLaplaceUPArctan",
            "geometry": "square_unit",
            "p": float(parsed.p),
            "level": int(parsed.level),
            "lambda1": float(parsed.lambda1),
            "lambda_level": int(lambda_level),
            "free_dofs": int(problem.free_dofs),
            "total_dofs": int(problem.total_dofs),
            "h": float(problem.h),
        },
        "metadata": {
            "nprocs": int(comm.Get_size()),
            "nproc_threads": int(threads),
            "linear_solver": dict(result.get("linear_solver", {})),
        },
        "result": result,
        "timings": dict(result.get("timings", {})),
    }
    if not parsed.quiet and rank == 0:
        sys.stdout.write(
            "pLaplace_up_arctan PETSc | "
            f"p={parsed.p:g} L{parsed.level} np={comm.Get_size()} "
            f"status={result['status']} residual={result['residual_norm']:.3e} "
            f"time={float(result['timings']['total_time']):.3f}s\n"
        )
        sys.stdout.flush()
    return payload


def main() -> None:
    payload = run_from_args(args)
    if MPI.COMM_WORLD.Get_rank() != 0:
        return
    text = json.dumps(payload, indent=2)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if (not args.quiet) or (not args.out):
        print(text)


if __name__ == "__main__":
    main()

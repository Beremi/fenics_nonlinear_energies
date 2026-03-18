#!/usr/bin/env python3
"""
p-Laplace 2D solver — DOF-partitioned parallel JAX + PETSc (CLI entry point).

Solver logic is in ``src/problems/plaplace/jax_petsc/solver.py``.

Thread control:
  - XLA multi-thread Eigen is disabled (memory-bandwidth limited workload)
  - OMP_NUM_THREADS=1 by default (critical for Hypre AMG — prevents internal
    thread oversubscription with MPI)

Usage:
  mpiexec -n 4 python3 src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 9
  mpiexec -n 16 python3 src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 9 --pc-type hypre
"""

import sys
import os
import json
import argparse
from pathlib import Path

from mpi4py import MPI
from src.core.cli.threading import configure_jax_cpu_threading

# ---- Parse args before setting env vars (critical for JAX init order) ----
parser = argparse.ArgumentParser(
    description="p-Laplace 2D — DOF-partitioned parallel JAX + PETSc solver"
)
parser.add_argument("--level", type=int, default=9, help="Mesh level (default: 9)")
parser.add_argument("--levels", type=int, nargs="+", default=None,
                    help="Multiple mesh levels (overrides --level)")
parser.add_argument("--repeats", type=int, default=1,
                    help="Number of solve repetitions (default: 1)")
parser.add_argument("--nproc", type=int, default=1,
                    help="XLA/OMP thread count per rank (default: 1)")
parser.add_argument("--coloring-trials", type=int, default=10,
                    help="Graph coloring trials per rank (default: 10)")
parser.add_argument("--profile", type=str, default="reference",
                    choices=["reference", "performance"],
                    help="Linear-solver profile defaults (default: reference)")
parser.add_argument("--ksp-type", type=str, default="cg",
                    help="PETSc KSP type (default: cg)")
parser.add_argument("--ksp-rtol", type=float, default=1e-3,
                    help="KSP relative tolerance (default: 1e-3)")
parser.add_argument("--ksp-max-it", type=int, default=200,
                    help="PETSc KSP max iterations (default: 200)")
parser.add_argument("--pc-type", type=str, default="hypre",
                    choices=["gamg", "hypre"],
                    help="PETSc PC type (default: hypre)")
parser.add_argument("--quiet", action="store_true",
                    help="Suppress per-iteration Newton output")
parser.add_argument("--json", type=str, default=None,
                    help="Output JSON file (rank 0 only)")
parser.add_argument("--tolf", type=float, default=1e-5,
                    help="Energy change tolerance (default: 1e-5)")
parser.add_argument("--tolg", type=float, default=1e-3,
                    help="Gradient norm tolerance (default: 1e-3)")
parser.add_argument("--linesearch-tol", type=float, default=1e-3,
                    help="Line-search tolerance (default: 1e-3)")
parser.add_argument("--local-coloring", action="store_true",
                    help="Use local per-rank graph coloring + vmap (Variant B)")
parser.add_argument("--assembly-mode", choices=("sfd", "element"), default="sfd",
                    help="Hessian assembly mode: 'sfd' (graph coloring) or "
                         "'element' (analytical element Hessians via jax.hessian)")
parser.add_argument("--local-hessian-mode", choices=("element", "sfd_local", "sfd_local_vmap"),
                    default="element",
                    help="Local Hessian mode for --assembly-mode element (default: element)")
parser.add_argument("--element-reorder-mode",
                    choices=("none", "block_rcm", "block_xyz", "block_metis"),
                    default="block_xyz",
                    help="DOF reorder for the reordered element assembler (default: block_xyz)")
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# ---- Set environment variables BEFORE importing JAX/PETSc ----
_threads = configure_jax_cpu_threading(args.nproc)

from jax import config  # noqa: E402
config.update("jax_enable_x64", True)

from src.problems.plaplace.jax_petsc.solver import run_level  # noqa: E402


def main():
    levels = args.levels if args.levels is not None else [args.level]

    if rank == 0:
        sys.stdout.write(
            f"p-Laplace 2D DOF-partitioned solver | "
            f"{nprocs} MPI rank(s) | NPROC={_threads} | "
            f"PC={args.pc_type}\n"
        )
        sys.stdout.write("=" * 80 + "\n")
        sys.stdout.flush()

    all_results = []

    for mesh_lvl in levels:
        level_results = []
        for rep in range(args.repeats):
            if rank == 0:
                sys.stdout.write(
                    f"\n  --- Level {mesh_lvl}"
                    f"{f', rep {rep + 1}/{args.repeats}' if args.repeats > 1 else ''}"
                    f" ---\n"
                )
                sys.stdout.flush()

            result = run_level(
                mesh_lvl, comm,
                verbose=(not args.quiet),
                coloring_trials=args.coloring_trials,
                profile=args.profile,
                ksp_type=args.ksp_type,
                ksp_rtol=args.ksp_rtol,
                ksp_max_it=args.ksp_max_it,
                pc_type=args.pc_type,
                tolf=args.tolf,
                tolg=args.tolg,
                local_coloring=args.local_coloring,
                assembly_mode=args.assembly_mode,
                nproc_threads=_threads,
                linesearch_tol=args.linesearch_tol,
                local_hessian_mode=args.local_hessian_mode,
                element_reorder_mode=args.element_reorder_mode,
            )
            level_results.append(result)

            if rank == 0:
                sys.stdout.write(
                    f"  [RESULT] level={result['mesh_level']} "
                    f"dofs={result['dofs']} np={nprocs} "
                    f"solve={result['solve_time']:.3f}s "
                    f"setup={result['setup_time']:.3f}s "
                    f"iters={result['iters']} "
                    f"ksp_its_total={result['total_ksp_its']} "
                    f"J={result['energy']:.6f} "
                    f"[{result['message']}]\n"
                )
                sys.stdout.flush()

        # Use the best (min solve_time) if multiple repeats
        best = min(level_results, key=lambda r: r["solve_time"])
        all_results.append(best)

        comm.Barrier()

    # ---- Summary table (rank 0) ----
    if rank == 0:
        sys.stdout.write("\n" + "=" * 80 + "\n")
        sys.stdout.write("Summary:\n")
        sys.stdout.write(
            f"  {'level':>5s} {'dofs':>8s} {'np':>3s} {'setup':>7s} "
            f"{'solve':>7s} {'total':>7s} {'its':>4s} "
            f"{'ksp':>6s} {'energy':>14s}\n"
        )
        for r in all_results:
            sys.stdout.write(
                f"  {r['mesh_level']:5d} {r['dofs']:8d} {nprocs:3d} "
                f"{r['setup_time']:7.3f} {r['solve_time']:7.3f} "
                f"{r['total_time']:7.3f} {r['iters']:4d} "
                f"{r['total_ksp_its']:6d} {r['energy']:14.6f}\n"
            )
        sys.stdout.write("Done.\n")
        sys.stdout.flush()

        if args.json:
            metadata = {
                "solver": "jax_petsc_dof_partitioned",
                "description": (
                    f"DOF-partitioned JAX + PETSc: P2P ghost exchange, "
                    f"local SFD Hessian, CG + {args.pc_type.upper()} AMG"
                ),
                "nprocs": nprocs,
                "nproc_threads": _threads,
                "coloring_trials_per_rank": args.coloring_trials,
                "linear_solver": {
                    "ksp_type": args.ksp_type,
                    "pc_type": args.pc_type,
                    "ksp_rtol": args.ksp_rtol,
                    "ksp_max_it": args.ksp_max_it,
                },
                "newton_params": {
                    "tolf": args.tolf,
                    "tolg": args.tolg,
                    "linesearch_interval": [-0.5, 2.0],
                    "linesearch_tol": args.linesearch_tol,
                    "maxit": 100,
                },
            }
            output = {"metadata": metadata, "results": all_results}
            path = Path(args.json)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fp:
                json.dump(output, fp, indent=2, default=str)
            sys.stdout.write(f"Results saved to {args.json}\n")


if __name__ == "__main__":
    main()

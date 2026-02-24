#!/usr/bin/env python3
"""
p-Laplace 2D solver — parallel JAX + PETSc (MPI).

Uses JAX for automatic differentiation and energy evaluation, with
MPI-parallel sparse-finite-difference (SFD) Hessian assembly and
PETSc KSP linear solves.  The Newton iteration uses the same algorithm
as the FEniCS custom_jaxversion solver:

  • Golden-section line search on [−0.5, 2] with tol = 1e-3
  • Stopping:  ‖∇J‖₂ < 1e-3  or  |ΔJ| < 1e-5
  • CG + HYPRE AMG linear solver with rtol = 1e-3

Usage:
  Serial:   python3 pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py
  Parallel: mpirun -n 4 python3 pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py

Requires: jax, jaxlib, h5py, scipy, numpy, mpi4py, petsc4py
"""

from tools_petsc4py.minimizers import newton
from pLaplace2D_jax_petsc.parallel_sfd import ParallelSFDSolver
from pLaplace2D_jax_petsc.jax_energy import J
from pLaplace2D_jax_petsc.mesh import MeshpLaplace2D
import sys
import time
import json
import argparse
import numpy as np
from mpi4py import MPI
from jax import config

config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Timing breakdown printer
# ---------------------------------------------------------------------------

def _print_timing_breakdown(report, solve_time, solver):
    """Print detailed per-iteration timing breakdown (rank 0)."""
    sys.stdout.write("\n  Timing Breakdown (hessian_solve_fn per Newton iteration):\n")
    sys.stdout.write(
        f"  {'It':>3s} {'allgath':>8s} {'HVP':>8s} {'nHVP':>5s} "
        f"{'allred':>8s} {'assem':>8s} {'KSP':>8s} {'KSP it':>7s} "
        f"{'total':>8s}\n"
    )
    sys.stdout.write("  " + "-" * 72 + "\n")
    for i, d in enumerate(report["iteration_details"]):
        sys.stdout.write(
            f"  {i:3d} {d['allgather']:8.4f} {d['hvp']:8.4f} "
            f"{d['n_hvps']:5d} {d['allreduce']:8.4f} {d['assembly']:8.4f} "
            f"{d['ksp']:8.4f} {d['ksp_its']:7d} {d['total']:8.4f}\n"
        )

    totals = report["totals"]
    sys.stdout.write("  " + "-" * 72 + "\n")
    sys.stdout.write(
        f"  {'SUM':>3s} {totals['allgather']:8.4f} {totals['hvp']:8.4f} "
        f"{'':>5s} {totals['allreduce']:8.4f} {totals['assembly']:8.4f} "
        f"{totals['ksp']:8.4f} {'':>7s} {totals['total']:8.4f}\n"
    )
    hess_total = totals["total"]
    overhead = solve_time - hess_total
    sys.stdout.write(
        f"\n  Hessian-solve total: {hess_total:.4f}s  |  "
        f"Other (energy+grad+LS): {overhead:.4f}s  |  "
        f"Solve wall: {solve_time:.4f}s\n"
    )
    sys.stdout.write(
        f"  n_colors={report['n_colors']}  "
        f"active_ranks={report['active_ranks']}/{solver.size}  "
        f"my_HVPs/iter={report['my_n_colors']}  "
        f"dofs={report['n_dofs']}  "
        f"nnz={report['nnz']}\n"
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Solver for a single mesh level
# ---------------------------------------------------------------------------

def run_level(mesh_level, comm, verbose=True, coloring_trials=10,
              ksp_rtol=1e-3, pc_type="gamg"):
    """Run parallel JAX+PETSc Newton solver for one mesh level.

    Returns dict with: mesh_level, dofs, setup_time, time, iters, energy,
    message, n_colors, timings.
    """
    _ = comm.Get_rank()  # available for future use

    # ---- load mesh (every rank reads the same HDF5 file) ----
    setup_start = time.perf_counter()
    mesh_obj = MeshpLaplace2D(mesh_level=mesh_level)
    params, adjacency, u_init = mesh_obj.get_data_jax()
    n_dofs = len(u_init)

    # ---- build parallel SFD solver ----
    solver = ParallelSFDSolver(
        energy_jax_fn=J,
        params=params,
        adjacency=adjacency,
        u_init=u_init,
        comm=comm,
        coloring_trials_per_rank=coloring_trials,
        ksp_rtol=ksp_rtol,
        ksp_type="cg",
        pc_type=pc_type,
    )
    setup_time = time.perf_counter() - setup_start

    # ---- initial guess as distributed PETSc Vec ----
    x = solver.create_vec(np.array(u_init))

    # ---- solve ----
    solve_start = time.perf_counter()
    result = newton(
        energy_fn=solver.energy_fn,
        gradient_fn=solver.gradient_fn,
        hessian_solve_fn=solver.hessian_solve_fn,
        x=x,
        tolf=1e-5,
        tolg=1e-3,
        linesearch_tol=1e-3,
        linesearch_interval=(-0.5, 2.0),
        maxit=100,
        verbose=verbose,
        comm=comm,
        ghost_update_fn=None,      # no ghosts — standard MPI Vec
        save_history=True,
    )
    solve_time = time.perf_counter() - solve_start

    # ---- collect results ----
    n_colors = solver.n_colors
    timings = {k: round(v, 4) for k, v in solver.timings.items()}
    timing_report = solver.get_timing_report()

    # ---- print per-iteration timing breakdown (rank 0 only) ----
    rank = comm.Get_rank()
    if verbose and rank == 0 and timing_report.get("iteration_details"):
        _print_timing_breakdown(timing_report, solve_time, solver)

    # ---- cleanup PETSc objects ----
    x.destroy()
    solver.cleanup()

    return {
        "mesh_level": mesh_level,
        "dofs": n_dofs,
        "setup_time": round(setup_time, 4),
        "time": round(solve_time, 4),
        "iters": result["nit"],
        "energy": round(float(result["fun"]), 10),
        "message": result["message"],
        "n_colors": n_colors,
        "timings": timings,
        "timing_report": timing_report,
        "history": result.get("history", []),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="p-Laplace 2D — parallel JAX + PETSc (SFD Hessian) benchmark"
    )
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[5, 6, 7, 8, 9],
        help="Mesh levels to run (default: 5 6 7 8 9)",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Output JSON file path (only written by rank 0)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-iteration output",
    )
    parser.add_argument(
        "--coloring-trials", type=int, default=10,
        help="Multi-start coloring trials per MPI rank (default: 10)",
    )
    parser.add_argument(
        "--ksp-rtol", type=float, default=1e-3,
        help="PETSc KSP relative tolerance (default: 1e-3)",
    )
    parser.add_argument(
        "--pc-type", type=str, default="gamg",
        help="PETSc PC type: gamg (default, recommended) or hypre",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    nprocs = comm.size

    if rank == 0:
        sys.stdout.write(
            f"p-Laplace 2D JAX+PETSc (SFD Hessian) | {nprocs} MPI process(es)\n"
        )
        sys.stdout.write("=" * 80 + "\n")
        sys.stdout.flush()

    all_results = []
    for mesh_lvl in args.levels:
        if rank == 0:
            sys.stdout.write(f"\n  --- Mesh level {mesh_lvl} ---\n")
            sys.stdout.flush()

        result = run_level(
            mesh_lvl,
            comm,
            verbose=(not args.quiet),
            coloring_trials=args.coloring_trials,
            ksp_rtol=args.ksp_rtol,
            pc_type=args.pc_type,
        )
        all_results.append(result)

        if rank == 0:
            sys.stdout.write(
                f"  RESULT mesh_level={result['mesh_level']} "
                f"dofs={result['dofs']} "
                f"time={result['time']:.3f}s "
                f"setup={result['setup_time']:.3f}s "
                f"iters={result['iters']} "
                f"n_colors={result['n_colors']} "
                f"J(u)={result['energy']:.6f} [{result['message']}]\n"
            )
            sys.stdout.flush()
        comm.Barrier()

    if rank == 0:
        sys.stdout.write("\n" + "=" * 80 + "\n")
        sys.stdout.write("Done.\n")
        sys.stdout.flush()

        if args.json:
            metadata = {
                "solver": "jax_petsc_sfd",
                "description": (
                    f"JAX + PETSc parallel SFD Hessian: "
                    f"golden-section line search [-0.5, 2], CG + {args.pc_type.upper()} AMG"
                ),
                "nprocs": nprocs,
                "coloring_trials_per_rank": args.coloring_trials,
                "linear_solver": {
                    "ksp_type": "cg",
                    "pc_type": args.pc_type,
                    "ksp_rtol": args.ksp_rtol,
                },
                "newton_params": {
                    "tolf": 1e-5,
                    "tolg": 1e-3,
                    "linesearch_interval": [-0.5, 2.0],
                    "linesearch_tol": 1e-3,
                    "maxit": 100,
                },
                "p": 3,
                "rhs_f": -10.0,
            }
            output = {"metadata": metadata, "results": all_results}
            with open(args.json, "w") as fp:
                json.dump(output, fp, indent=2)
            sys.stdout.write(f"Results saved to {args.json}\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

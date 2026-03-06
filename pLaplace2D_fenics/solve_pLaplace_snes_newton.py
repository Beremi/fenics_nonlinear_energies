#!/usr/bin/env python3
"""
p-Laplace 2D SNES Newton solver — CLI entry point.

Solver logic is in ``pLaplace2D_fenics/solver_snes.py``.

Usage:
  Serial:   python3 pLaplace2D_fenics/solve_pLaplace_snes_newton.py
  Parallel: mpirun -n <nprocs> python3 pLaplace2D_fenics/solve_pLaplace_snes_newton.py

Requires: DOLFINx >= 0.10, PETSc, mpi4py
"""
import sys
import json
import argparse

import dolfinx
from mpi4py import MPI

from pLaplace2D_fenics.solver_snes import run_level


def main():
    parser = argparse.ArgumentParser(description="p-Laplace 2D SNES Newton benchmark")
    parser.add_argument("--levels", type=int, nargs="+", default=[5, 6, 7, 8, 9],
                        help="Mesh levels to run (default: 5 6 7 8 9)")
    parser.add_argument("--json", type=str, default=None,
                        help="Output JSON file path (only written by rank 0)")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    nprocs = comm.size

    if rank == 0:
        sys.stdout.write(f"p-Laplace 2D SNES Newton | {nprocs} MPI process(es)\n")
        sys.stdout.write("=" * 80 + "\n")
        sys.stdout.flush()

    all_results = []
    for mesh_lvl in args.levels:
        result = run_level(mesh_lvl)
        all_results.append(result)
        if rank == 0:
            sys.stdout.write(
                f"  mesh_level={result['mesh_level']} dofs={result['total_dofs']} "
                f"time={result['time']:.3f}s iters={result['iters']} "
                f"J(u)={result['energy']:.4f} reason={result['converged_reason']}\n"
            )
            sys.stdout.flush()
        comm.Barrier()

    if rank == 0:
        sys.stdout.write("=" * 80 + "\n")
        sys.stdout.write("Done.\n")
        sys.stdout.flush()

        if args.json:
            metadata = {
                "solver": "snes_newton",
                "description": "Built-in PETSc SNES Newton solver with CG + HYPRE AMG",
                "dolfinx_version": dolfinx.__version__,
                "nprocs": nprocs,
                "petsc_options": {
                    "snes_type": "newtonls",
                    "snes_linesearch_type": "basic",
                    "snes_atol": 1e-6,
                    "snes_rtol": 1e-8,
                    "snes_max_it": 20,
                    "ksp_type": "cg",
                    "pc_type": "hypre",
                    "ksp_rtol": 1e-1,
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

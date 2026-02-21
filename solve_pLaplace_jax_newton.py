#!/usr/bin/env python3
"""
p-Laplace 2D solver using JAX-based energy minimization with custom Newton method.

Solves the p-Laplacian problem (p=3) on a unit square with homogeneous
Dirichlet BCs and constant RHS f=-10. Uses JAX for automatic differentiation,
sparse finite differences for Hessian assembly, and PyAMG for the linear solver.

This solver is single-process only (no MPI parallelism).

Runs mesh levels 5-9 (table levels 4-8) and reports dofs, time, iterations, energy.

Usage:
  python3 solve_pLaplace_jax_newton.py
  python3 solve_pLaplace_jax_newton.py --levels 5 6 7 8 9
  python3 solve_pLaplace_jax_newton.py --json results/jax_run.json

Requires: jax, jaxlib, h5py, pyamg, scipy, numpy
"""
from pLaplace2D.mesh import MeshpLaplace2D
from pLaplace2D.jax_energy import J
from tools.jax_diff import EnergyDerivator
from tools.sparse_solvers import HessSolverGenerator
from tools.minimizers import newton
import sys
import time
import json
import argparse
import numpy as np

from jax import config
config.update("jax_enable_x64", True)


def run_level(mesh_level, verbose=False):
    """Run JAX Newton solver for a single mesh level.

    Returns dict with: mesh_level, dofs, time, iters, energy, setup_time, message
    """
    # Setup: load mesh, compile JAX derivatives
    setup_start = time.perf_counter()
    mesh = MeshpLaplace2D(mesh_level=mesh_level)
    params, adjacency, u_init = mesh.get_data_jax()
    energy = EnergyDerivator(J, params, adjacency, u_init)
    F, dF, ddF = energy.get_derivatives()
    ddf_with_solver = HessSolverGenerator(ddf=ddF, solver_type="amg", verbose=False, tol=1e-3)
    setup_time = time.perf_counter() - setup_start

    n_dofs = len(u_init)

    # Solve
    solve_start = time.perf_counter()
    res = newton(F, dF, ddf_with_solver, u_init, verbose=verbose, tolf=1e-5, linesearch_tol=1e-3)
    solve_time = time.perf_counter() - solve_start

    return {
        "mesh_level": mesh_level,
        "dofs": n_dofs,
        "setup_time": round(setup_time, 4),
        "time": round(solve_time, 4),
        "iters": res["nit"],
        "energy": round(float(res["fun"]), 4),
        "message": res["message"],
    }


def main():
    parser = argparse.ArgumentParser(description="p-Laplace 2D JAX Newton benchmark")
    parser.add_argument("--levels", type=int, nargs="+", default=[5, 6, 7, 8, 9],
                        help="Mesh levels to run (default: 5 6 7 8 9)")
    parser.add_argument("--json", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-iteration output")
    args = parser.parse_args()

    sys.stdout.write("p-Laplace 2D JAX Newton (single process)\n")
    sys.stdout.write("=" * 80 + "\n")
    sys.stdout.flush()

    all_results = []
    for mesh_lvl in args.levels:
        sys.stdout.write(f"  --- Mesh level {mesh_lvl} ---\n")
        sys.stdout.flush()
        result = run_level(mesh_lvl, verbose=not args.quiet)
        all_results.append(result)
        sys.stdout.write(
            f"  RESULT mesh_level={result['mesh_level']} dofs={result['dofs']} "
            f"setup={result['setup_time']:.3f}s solve={result['time']:.3f}s "
            f"iters={result['iters']} J(u)={result['energy']:.4f} [{result['message']}]\n"
        )
        sys.stdout.flush()

    sys.stdout.write("=" * 80 + "\n")
    sys.stdout.write("Done.\n")
    sys.stdout.flush()

    if args.json:
        import jax
        metadata = {
            "solver": "jax_newton",
            "description": "JAX Newton with golden-section line search, PyAMG CG solver",
            "jax_version": jax.__version__,
            "nprocs": 1,
            "solver_options": {
                "tolf": 1e-5,
                "linesearch_tol": 1e-3,
                "hess_solver": "amg",
                "hess_tol": 1e-3,
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

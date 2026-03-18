#!/usr/bin/env python3
"""
HyperElasticity 3D solver — SNES Newton CLI entry point.

Solver logic is in ``src/problems/hyperelasticity/fenics/solver_snes.py``.

Usage:
  Serial:   python3 src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py
  Parallel: mpiexec -n <nprocs> python3 src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py
"""
import json
import argparse
from pathlib import Path

from mpi4py import MPI

from src.problems.hyperelasticity.fenics.solver_snes import run_level


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--snes_type", type=str, default="newtonls")
    parser.add_argument("--linesearch", type=str, default="basic")
    parser.add_argument("--ksp_type", type=str, default="gmres")
    parser.add_argument("--pc_type", type=str, default="hypre")
    parser.add_argument("--ksp_rtol", type=float, default=1e-3)
    parser.add_argument("--ksp_max_it", type=int, default=10000)
    parser.add_argument("--snes_atol", type=float, default=1e-5)
    parser.add_argument("--use_objective", action="store_true")
    parser.add_argument("--no_near_nullspace", action="store_true")
    parser.add_argument("--hypre_nodal_coarsen", type=int, default=-1,
                        help="BoomerAMG nodal coarsen (-1 to skip setting)")
    parser.add_argument("--hypre_vec_interp_variant", type=int, default=-1,
                        help="BoomerAMG vec interp variant (-1 to skip setting)")
    parser.add_argument("--nullspace_gram_schmidt", action="store_true",
                        help="Apply centering + Gram-Schmidt to RBM nullspace vectors")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--stop_on_fail", action="store_true",
                        help="Stop loading sequence at first diverged step")
    parser.add_argument("--total_steps", type=int, default=None,
                        help="Total steps spanning full 4*2pi rotation (controls step size); "
                             "default: same as --steps")
    args, _ = parser.parse_known_args()

    res = run_level(args.level, num_steps=args.steps, total_steps=args.total_steps,
                    snes_type=args.snes_type, linesearch=args.linesearch,
                    ksp_type=args.ksp_type, pc_type=args.pc_type,
                    ksp_rtol=args.ksp_rtol, ksp_max_it=args.ksp_max_it,
                    snes_atol=args.snes_atol,
                    use_objective=args.use_objective, verbose=not args.quiet,
                    use_near_nullspace=not args.no_near_nullspace,
                    hypre_nodal_coarsen=args.hypre_nodal_coarsen,
                    hypre_vec_interp_variant=args.hypre_vec_interp_variant,
                    nullspace_gram_schmidt=args.nullspace_gram_schmidt,
                    stop_on_fail=args.stop_on_fail)

    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(res, indent=2))
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)

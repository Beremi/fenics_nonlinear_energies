"""
p-Laplace 2D solver using DOLFINx built-in SNES Newton solver.

Solves the p-Laplacian problem (p=3) on a unit square with homogeneous
Dirichlet BCs and constant RHS f=-10, using PETSc SNES with CG + HYPRE AMG.

Runs mesh levels 5-9 (table levels 4-8) and reports dofs, time, iterations, energy.

Usage:
  Serial:   python3 solve_pLaplace_snes_newton.py
  Parallel: mpirun -n <nprocs> python3 solve_pLaplace_snes_newton.py

Requires: DOLFINx >= 0.10, PETSc, mpi4py
"""
import sys
import time
import json
import argparse
import ufl
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.io import XDMFFile
from petsc4py import PETSc
from dolfinx.fem.petsc import NonlinearProblem, set_bc
from petsc4py.PETSc import ScalarType


def run_level(mesh_level):
    """Run p-Laplace solver for a single mesh level.

    Returns dict with: mesh_level, total_dofs, time, iters, energy, converged_reason
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    with XDMFFile(comm, f"mesh_data/pLaplace/mesh_level_{mesh_level}.xdmf", "r") as xdmf_file:
        msh = xdmf_file.read_mesh(name="mesh")

    V = fem.functionspace(msh, ("Lagrange", 1))
    total_dofs = V.dofmap.index_map.size_global

    msh.topology.create_connectivity(1, 2)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    bc_dofs = fem.locate_dofs_topological(V, 1, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0), bc_dofs, V)

    p = 3
    f_const = fem.Constant(msh, ScalarType(-10.0))

    u = fem.Function(V)

    # Initial guess (small random, fixed seed)
    np.random.seed(42)
    vec = u.x.petsc_vec
    idx_local = range(*vec.getOwnershipRange())
    vec.setValues(idx_local, 1e-2 * np.random.rand(len(idx_local)))
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    vec.assemble()
    set_bc(vec, [bc])

    v = ufl.TestFunction(V)
    F_energy = (1 / p) * ufl.inner(ufl.grad(u), ufl.grad(u))**(p / 2) * ufl.dx - f_const * u * ufl.dx
    J_form = ufl.derivative(F_energy, u, v)

    petsc_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "basic",
        "snes_atol": 1e-6,
        "snes_rtol": 1e-8,
        "snes_max_it": 20,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-1,
    }

    problem = NonlinearProblem(
        J_form, u,
        petsc_options_prefix=f"lvl{mesh_level}_",
        bcs=[bc],
        petsc_options=petsc_opts,
    )

    start_time = time.time()
    problem.solve()
    total_time = time.time() - start_time

    snes = problem.solver
    n_iters = snes.getIterationNumber()
    reason = snes.getConvergedReason()
    snes.destroy()

    final_energy = fem.assemble_scalar(fem.form(F_energy))
    final_energy = msh.comm.allreduce(final_energy, op=MPI.SUM)

    return {
        "mesh_level": mesh_level,
        "total_dofs": total_dofs,
        "time": round(total_time, 4),
        "iters": n_iters,
        "energy": round(final_energy, 4),
        "converged_reason": int(reason),
    }


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
            import dolfinx
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

"""
Ginzburg-Landau 2D solver using DOLFINx built-in SNES Newton solver.

Solves the Ginzburg-Landau energy minimisation problem on [-1,1]² with
homogeneous Dirichlet BCs:

    min_u  J(u) = ∫_Ω  ε/2 |∇u|² + 1/4 (u² − 1)²  dx

with ε = 0.01.  Uses PETSc SNES with basic line search, GMRES + HYPRE AMG.

Note: The Ginzburg-Landau Hessian is indefinite (non-convex problem), so GMRES
is used instead of CG.

WARNING: SNES basic line search (full Newton step) is unreliable for non-convex
problems.  Convergence to the correct minimum depends on mesh level and MPI
decomposition.  For a reliable solver see solve_GL_custom_jaxversion.py which
uses a golden-section energy line search.

Runs mesh levels 5-9 by default and reports dofs, time, iterations, energy.

Usage:
  Serial:   python3 GinzburgLandau2D_fenics/solve_GL_snes_newton.py
  Parallel: mpirun -n <nprocs> python3 GinzburgLandau2D_fenics/solve_GL_snes_newton.py

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
from petsc4py import PETSc
from dolfinx.fem.petsc import NonlinearProblem, set_bc
from petsc4py.PETSc import ScalarType


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
EPS = 0.01


def run_level(mesh_level):
    """Run GL SNES solver for a single mesh level.

    Returns dict with: mesh_level, total_dofs, time, iters, energy, converged_reason
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # ---- mesh: structured grid on [-1,1]², 2^(level+1) divisions per side ----
    N = 2 ** (mesh_level + 1)
    msh = mesh.create_rectangle(
        comm, [[-1.0, -1.0], [1.0, 1.0]], [N, N],
        cell_type=mesh.CellType.triangle,
    )

    V = fem.functionspace(msh, ("Lagrange", 1))
    total_dofs = V.dofmap.index_map.size_global

    # ---- Dirichlet BC (u = 0 on ∂Ω) ----
    msh.topology.create_connectivity(1, 2)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    bc_dofs = fem.locate_dofs_topological(V, 1, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0), bc_dofs, V)

    # ---- variational forms ----
    u = fem.Function(V)
    v = ufl.TestFunction(V)

    eps = fem.Constant(msh, ScalarType(EPS))

    # Energy functional
    F_energy = (
        (eps / 2) * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx
        + (1.0 / 4.0) * (u**2 - 1)**2 * ufl.dx
    )
    # Residual (gradient of energy)
    J_form = ufl.derivative(F_energy, u, v)

    # ---- initial guess: sin(π(x-1)/2) * sin(π(y-1)/2) ----
    def initial_guess(x):
        return np.sin(np.pi * (x[0] - 1) / 2) * np.sin(np.pi * (x[1] - 1) / 2)

    u.interpolate(initial_guess)
    vec = u.x.petsc_vec
    set_bc(vec, [bc])
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # ---- SNES solver options ----
    # Note: GMRES is needed because the Hessian is indefinite (non-convex energy).
    # snes_divergence_tolerance = -1 disables the DTOL check which can trigger
    # spuriously for non-convex problems.
    petsc_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "basic",
        "snes_atol": 1e-6,
        "snes_rtol": 1e-8,
        "snes_max_it": 500,
        "snes_divergence_tolerance": -1.0,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-1,
    }

    problem = NonlinearProblem(
        J_form, u,
        petsc_options_prefix=f"gl{mesh_level}_",
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
    parser = argparse.ArgumentParser(description="Ginzburg-Landau 2D SNES Newton benchmark")
    parser.add_argument("--levels", type=int, nargs="+", default=[5, 6, 7, 8, 9],
                        help="Mesh levels to run (default: 5 6 7 8 9)")
    parser.add_argument("--json", type=str, default=None,
                        help="Output JSON file path (only written by rank 0)")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    nprocs = comm.size

    if rank == 0:
        sys.stdout.write(f"Ginzburg-Landau 2D SNES Newton | {nprocs} MPI process(es)\n")
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
                "description": "Built-in PETSc SNES Newton solver with basic line search, GMRES + HYPRE AMG",
                "dolfinx_version": dolfinx.__version__,
                "nprocs": nprocs,
                "petsc_options": {
                    "snes_type": "newtonls",
                    "snes_linesearch_type": "basic",
                    "snes_atol": 1e-6,
                    "snes_rtol": 1e-8,
                    "snes_max_it": 500,
                    "snes_divergence_tolerance": -1.0,
                    "ksp_type": "gmres",
                    "pc_type": "hypre",
                    "ksp_rtol": 1e-1,
                },
                "eps": EPS,
            }
            output = {"metadata": metadata, "results": all_results}
            with open(args.json, "w") as fp:
                json.dump(output, fp, indent=2)
            sys.stdout.write(f"Results saved to {args.json}\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

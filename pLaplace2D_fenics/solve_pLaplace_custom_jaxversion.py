"""
p-Laplace 2D solver — custom Newton (JAX-version algorithm) with PETSc linear algebra.

Re-implements the JAX Newton minimiser (tools/minimizers.py) on top of PETSc
objects so it runs under MPI with DOLFINx assembly.  The algorithm matches the
JAX version exactly:

  • Golden-section line search on [−0.5, 2] with tol = 1e-3
  • Stopping:  ‖∇J‖₂ < 1e-3  or  |ΔJ| < 1e-5
  • CG + HYPRE AMG linear solver with rtol = 1e-3

This should reproduce the JAX iteration counts while benefiting from PETSc's
MPI-parallel matrix/vector operations and HYPRE preconditioning.

Usage:
  Serial:   python3 pLaplace2D_fenics/solve_pLaplace_custom_jaxversion.py
  Parallel: mpirun -n <nprocs> python3 pLaplace2D_fenics/solve_pLaplace_custom_jaxversion.py

Requires: DOLFINx >= 0.10, PETSc, mpi4py
"""
import sys
import time
import json
import argparse
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.io import XDMFFile
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_matrix,
    set_bc,
)

from tools_petsc4py.minimizers import newton


# ---------------------------------------------------------------------------
# Helper: PETSc ghost update for DOLFINx vectors
# ---------------------------------------------------------------------------

def _ghost_update(v):
    """INSERT-mode forward scatter (owned → ghosts)."""
    v.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


# ---------------------------------------------------------------------------
# Solver for a single mesh level
# ---------------------------------------------------------------------------

def run_level(mesh_level, verbose=True):
    """Run JAX-version Newton solver for one mesh level.

    Returns dict with: mesh_level, total_dofs, time, iters, energy, message
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # ---- mesh + function space ----
    with XDMFFile(comm, f"mesh_data/pLaplace/mesh_level_{mesh_level}.xdmf", "r") as f:
        msh = f.read_mesh(name="mesh")

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
    w = ufl.TrialFunction(V)

    p = 3.0
    f_rhs = fem.Constant(msh, ScalarType(-10.0))

    J_energy = (
        (1.0 / p) * ufl.inner(ufl.grad(u), ufl.grad(u)) ** (p / 2) * ufl.dx
        - f_rhs * u * ufl.dx
    )
    dJ = ufl.derivative(J_energy, u, v)
    ddJ = ufl.derivative(dJ, u, w)

    energy_form = fem.form(J_energy)
    grad_form = fem.form(dJ)
    hessian_form = fem.form(ddJ)

    # ---- form for energy evaluation at arbitrary point ----
    u_ls = fem.Function(V)
    energy_ls = ufl.replace(J_energy, {u: u_ls})
    energy_ls_form = fem.form(energy_ls)

    # ---- initial guess  (small random, same as other FEniCS benchmarks) ----
    np.random.seed(42)
    x = u.x.petsc_vec
    lo, hi = x.getOwnershipRange()
    x.setValues(range(lo, hi), 1e-2 * np.random.rand(hi - lo))
    _ghost_update(x)
    x.assemble()
    set_bc(x, [bc])

    # ---- pre-allocate Hessian matrix and KSP ----
    A = create_matrix(hessian_form)
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.HYPRE)
    ksp.setTolerances(rtol=1e-3)

    # ------------------------------------------------------------------
    # Callbacks for tools_petsc4py.minimizers.newton
    #
    # The Newton solver only sees PETSc Vec / scalar — all DOLFINx / UFL
    # assembly is hidden inside these three closures.
    # ------------------------------------------------------------------

    def energy_fn(vec):
        """J(u)  at an arbitrary PETSc Vec  (globally reduced scalar)."""
        vec.copy(u_ls.x.petsc_vec)
        u_ls.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        local_val = fem.assemble_scalar(energy_ls_form)
        return comm.allreduce(local_val, op=MPI.SUM)

    def gradient_fn(vec, g):
        """Assemble  ∇J  into *g*  (BCs applied, ghost-updated)."""
        # vec IS u.x.petsc_vec  (newton modifies it in-place)
        # → forms that reference `u` see the current iterate automatically.
        with g.localForm() as g_loc:
            g_loc.set(0.0)
        assemble_vector(g, grad_form)
        apply_lifting(g, [hessian_form], [[bc]], x0=[vec])
        g.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(g, [bc], vec)

    def hessian_solve_fn(vec, rhs, sol):
        """Assemble Hessian, solve  H · sol = rhs.  Return KSP iters."""
        A.zeroEntries()
        assemble_matrix(A, hessian_form, bcs=[bc])
        A.assemble()
        ksp.setOperators(A)
        ksp.solve(rhs, sol)
        return ksp.getIterationNumber()

    # ---- solve ----
    t_start = time.perf_counter()
    result = newton(
        energy_fn,
        gradient_fn,
        hessian_solve_fn,
        x,
        tolf=1e-5,
        tolg=1e-3,
        linesearch_tol=1e-3,
        linesearch_interval=(-0.5, 2.0),
        maxit=100,
        verbose=verbose,
        comm=comm,
        ghost_update_fn=_ghost_update,
    )
    total_time = time.perf_counter() - t_start

    # ---- final energy (from the solver result, already computed) ----
    final_energy = result["fun"]

    # ---- clean up PETSc objects ----
    ksp.destroy()
    A.destroy()

    return {
        "mesh_level": mesh_level,
        "total_dofs": total_dofs,
        "time": round(total_time, 4),
        "iters": result["nit"],
        "energy": round(final_energy, 4),
        "message": result["message"],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="p-Laplace 2D custom Newton (JAX-version algorithm) benchmark"
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
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    nprocs = comm.size

    if rank == 0:
        sys.stdout.write(
            f"p-Laplace 2D Custom Newton (JAX-version) | {nprocs} MPI process(es)\n"
        )
        sys.stdout.write("=" * 80 + "\n")
        sys.stdout.flush()

    all_results = []
    for mesh_lvl in args.levels:
        if rank == 0:
            sys.stdout.write(f"  --- Mesh level {mesh_lvl} ---\n")
            sys.stdout.flush()

        result = run_level(mesh_lvl, verbose=(not args.quiet))
        all_results.append(result)

        if rank == 0:
            sys.stdout.write(
                f"  RESULT mesh_level={result['mesh_level']} "
                f"dofs={result['total_dofs']} "
                f"time={result['time']:.3f}s iters={result['iters']} "
                f"J(u)={result['energy']:.4f} [{result['message']}]\n"
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
                "solver": "custom_jaxversion",
                "description": (
                    "Custom Newton (JAX-version algorithm): "
                    "golden-section line search [-0.5, 2], CG + HYPRE AMG"
                ),
                "dolfinx_version": dolfinx.__version__,
                "nprocs": nprocs,
                "linear_solver": {
                    "ksp_type": "cg",
                    "pc_type": "hypre",
                    "ksp_rtol": 1e-3,
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

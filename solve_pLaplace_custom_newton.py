"""
p-Laplace 2D solver using a custom Newton method with golden-section line search.

Solves the p-Laplacian problem (p=3) on a unit square with homogeneous
Dirichlet BCs and constant RHS f=-10. Uses manually assembled Hessian and
gradient with CG + HYPRE AMG linear solver and golden-section line search.

Runs mesh levels 5-9 (table levels 4-8) and reports dofs, time, iterations, energy.

Usage:
  Serial:   python3 solve_pLaplace_custom_newton.py
  Parallel: mpirun -n <nprocs> python3 solve_pLaplace_custom_newton.py

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
    apply_lifting, assemble_matrix, assemble_vector,
    set_bc, create_matrix, create_vector,
)


def golden_section_search(f, a, b, tol):
    """Find the minimum of f on [a,b] using golden section search.

    Parameters
    ----------
    f : callable
        Univariate function to minimize.
    a, b : float
        Search interval endpoints.
    tol : float
        Stopping tolerance on interval width.

    Returns
    -------
    t : float
        Approximate minimizer.
    it : int
        Number of iterations.
    """
    gamma = 0.5 + np.sqrt(5) / 2

    an, bn = a, b
    dn = (bn - an) / gamma + an
    cn = an + bn - dn

    fcn = f(cn)
    fdn = f(dn)
    it = 0

    while bn - an > tol:
        if fcn < fdn:
            an, bn = an, dn
            dn, cn = cn, an + bn - cn
            fdn, fcn = fcn, f(cn)
        else:
            an, bn = cn, bn
            cn, dn = dn, an + bn - dn
            fcn, fdn = fdn, f(dn)
        it += 1

    return (an + bn) / 2, it


def run_level(mesh_level, max_iterations=40, grad_tol=1e-6, verbose=True):
    """Run custom Newton solver for a single mesh level.

    Returns dict with: mesh_level, total_dofs, time, iters, energy
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    with XDMFFile(comm, f"pLaplace_fenics_mesh/mesh_level_{mesh_level}.xdmf", "r") as xdmf_file:
        msh = xdmf_file.read_mesh(name="mesh")

    V = fem.functionspace(msh, ("Lagrange", 1))
    total_dofs = V.dofmap.index_map.size_global

    # Boundary conditions
    msh.topology.create_connectivity(1, 2)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    bc_dofs_idx = fem.locate_dofs_topological(V, 1, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0), bc_dofs_idx, V)

    # Functions
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    u_trial = ufl.TrialFunction(V)

    # Initial guess
    np.random.seed(42)
    vec = u.x.petsc_vec
    idx_local = range(*vec.getOwnershipRange())
    vec.setValues(idx_local, 1e-2 * np.random.rand(len(idx_local)))
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    vec.assemble()
    set_bc(vec, [bc])

    # Problem definition
    p = 3.0
    f_const = fem.Constant(msh, ScalarType(-10.0))
    energy = (1 / p) * ufl.inner(ufl.grad(u), ufl.grad(u))**(p / 2) * ufl.dx - f_const * u * ufl.dx
    grad_energy = ufl.derivative(energy, u, v)
    hessian = ufl.derivative(grad_energy, u, u_trial)

    # Forms
    energy_form = fem.form(energy)
    grad_form = fem.form(grad_energy)
    hessian_form = fem.form(hessian)

    # Auxiliary function for line search energy evaluation
    u_new = fem.Function(V)
    energy_new = ufl.replace(energy, {u: u_new})
    energy_form_new = fem.form(energy_new)

    # Linear solver setup
    du = fem.Function(V)
    A = create_matrix(hessian_form)
    L = create_vector(V)
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.HYPRE)
    ksp.setTolerances(rtol=1e-1)

    def compute_energy_alpha(alpha):
        u_new.x.petsc_vec.waxpy(alpha, du.x.petsc_vec, u.x.petsc_vec)
        u_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        val = fem.assemble_scalar(energy_form_new)
        return msh.comm.allreduce(val, op=MPI.SUM)

    # Newton iteration loop
    all_start = time.time()
    converged = False
    n_iters = 0

    for i in range(max_iterations):
        # Assemble Hessian
        A.zeroEntries()
        assemble_matrix(A, hessian_form, bcs=[bc])
        A.assemble()

        # Assemble negative gradient
        with L.localForm() as loc_L:
            loc_L.set(0)
        assemble_vector(L, grad_form)
        apply_lifting(L, [hessian_form], [[bc]], x0=[u.x.petsc_vec])  # type: ignore
        L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        L.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        set_bc(L, [bc])
        L.assemble()
        L.scale(-1)

        # Solve linear system
        ksp.solve(L, du.x.petsc_vec)
        du.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        grad_inf_norm = L.norm(PETSc.NormType.NORM_INFINITY)

        # Line search
        alpha, _ = golden_section_search(compute_energy_alpha, 0, 1, 1e-1)

        # Update solution
        u.x.petsc_vec.axpy(alpha, du.x.petsc_vec)
        u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        n_iters = i + 1

        if verbose and rank == 0:
            sys.stdout.write(
                f"    IT {n_iters}: |grad|_inf = {grad_inf_norm:.3e}, alpha = {alpha:.3e}, "
                f"ksp_its = {ksp.getIterationNumber()}\n"
            )
            sys.stdout.flush()

        if grad_inf_norm < grad_tol:
            converged = True
            break

    total_time = time.time() - all_start

    final_energy = fem.assemble_scalar(energy_form)
    final_energy = msh.comm.allreduce(final_energy, op=MPI.SUM)

    ksp.destroy()
    A.destroy()
    L.destroy()

    return {
        "mesh_level": mesh_level,
        "total_dofs": total_dofs,
        "time": round(total_time, 4),
        "iters": n_iters,
        "energy": round(final_energy, 4),
        "converged": converged,
    }


def main():
    parser = argparse.ArgumentParser(description="p-Laplace 2D custom Newton benchmark")
    parser.add_argument("--levels", type=int, nargs="+", default=[5, 6, 7, 8, 9],
                        help="Mesh levels to run (default: 5 6 7 8 9)")
    parser.add_argument("--json", type=str, default=None,
                        help="Output JSON file path (only written by rank 0)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-iteration output")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    nprocs = comm.size

    if rank == 0:
        sys.stdout.write(f"p-Laplace 2D Custom Newton | {nprocs} MPI process(es)\n")
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
                f"  RESULT mesh_level={result['mesh_level']} dofs={result['total_dofs']} "
                f"time={result['time']:.3f}s iters={result['iters']} "
                f"J(u)={result['energy']:.4f} converged={result['converged']}\n"
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
                "solver": "custom_newton_golden_section",
                "description": "Custom Newton with golden-section line search, CG + HYPRE AMG",
                "dolfinx_version": dolfinx.__version__,
                "nprocs": nprocs,
                "linear_solver": {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-1},
                "newton_params": {"max_it": 40, "grad_tol": 1e-6, "linesearch_tol": 1e-1},
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

#!/usr/bin/env python3
"""Test GL custom Newton solver."""
import sys
sys.path.insert(0, '/work')
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import (
    apply_lifting, assemble_matrix, assemble_vector, create_matrix, set_bc,
)
import ufl
import time
from tools_petsc4py.minimizers import newton

EPS = 0.01
comm = MPI.COMM_WORLD

def _ghost_update(v):
    v.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

for mesh_level in [5, 6, 7]:
    N = 2 ** (mesh_level + 1)
    msh = mesh.create_rectangle(
        comm, [[-1.0, -1.0], [1.0, 1.0]], [N, N],
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(msh, ("Lagrange", 1))
    total_dofs = V.dofmap.index_map.size_global

    msh.topology.create_connectivity(1, 2)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    bc_dofs = fem.locate_dofs_topological(V, 1, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0), bc_dofs, V)

    u = fem.Function(V)
    v = ufl.TestFunction(V)
    w = ufl.TrialFunction(V)
    eps_c = fem.Constant(msh, ScalarType(EPS))

    J_energy = (
        (eps_c / 2) * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx
        + (1.0 / 4.0) * (u**2 - 1)**2 * ufl.dx
    )
    dJ = ufl.derivative(J_energy, u, v)
    ddJ = ufl.derivative(dJ, u, w)

    energy_form = fem.form(J_energy)
    grad_form = fem.form(dJ)
    hessian_form = fem.form(ddJ)

    u_ls = fem.Function(V)
    energy_ls = ufl.replace(J_energy, {u: u_ls})
    energy_ls_form = fem.form(energy_ls)

    def initial_guess(x):
        return np.sin(np.pi * (x[0] - 1) / 2) * np.sin(np.pi * (x[1] - 1) / 2)
    u.interpolate(initial_guess)
    x = u.x.petsc_vec
    set_bc(x, [bc])
    _ghost_update(x)
    x.assemble()

    A = create_matrix(hessian_form)
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setType(PETSc.KSP.Type.GMRES)  # GMRES because Hessian can be indefinite
    ksp.getPC().setType(PETSc.PC.Type.ILU)
    ksp.setTolerances(rtol=1e-3)

    def energy_fn(vec):
        vec.copy(u_ls.x.petsc_vec)
        u_ls.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        local_val = fem.assemble_scalar(energy_ls_form)
        return comm.allreduce(local_val, op=MPI.SUM)

    def gradient_fn(vec, g):
        with g.localForm() as g_loc:
            g_loc.set(0.0)
        assemble_vector(g, grad_form)
        apply_lifting(g, [hessian_form], [[bc]], x0=[vec])
        g.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(g, [bc], vec)

    def hessian_solve_fn(vec, rhs, sol):
        A.zeroEntries()
        assemble_matrix(A, hessian_form, bcs=[bc])
        A.assemble()
        ksp.setOperators(A)
        ksp.solve(rhs, sol)
        return ksp.getIterationNumber()

    t_start = time.perf_counter()
    result = newton(
        energy_fn, gradient_fn, hessian_solve_fn, x,
        tolf=1e-6, tolg=1e-5,
        linesearch_tol=1e-3, linesearch_interval=(-0.5, 2.0),
        maxit=200, verbose=True, comm=comm, ghost_update_fn=_ghost_update,
    )
    total_time = time.perf_counter() - t_start

    sys.stdout.write(
        "level={} dofs={}: time={:.3f}s iters={} J={:.6f} [{}]\n".format(
            mesh_level, total_dofs, total_time, result['nit'], result['fun'], result['message']
        )
    )
    sys.stdout.flush()

    ksp.destroy()
    A.destroy()

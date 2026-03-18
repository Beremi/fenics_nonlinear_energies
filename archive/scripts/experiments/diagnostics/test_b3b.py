#!/usr/bin/env python3
"""Test B3 variant with ksp_rtol=1e-3."""
import sys
import numpy as np
import time
from mpi4py import MPI
from dolfinx import fem, mesh
from petsc4py import PETSc
from dolfinx.fem.petsc import NonlinearProblem, set_bc
from petsc4py.PETSc import ScalarType
import ufl

comm = MPI.COMM_WORLD
rank = comm.rank


def run_test(mesh_level):
    N = 2 ** (mesh_level + 1)
    msh = mesh.create_rectangle(comm, [[-1.0, -1.0], [1.0, 1.0]], [N, N], cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", 1))
    msh.topology.create_connectivity(1, 2)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    bc_dofs = fem.locate_dofs_topological(V, 1, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0), bc_dofs, V)
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    eps_c = fem.Constant(msh, ScalarType(0.01))
    F_energy = (eps_c / 2) * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx + 0.25 * (u**2 - 1)**2 * ufl.dx
    J_form = ufl.derivative(F_energy, u, v)
    energy_form = fem.form(F_energy)
    u.interpolate(lambda x: np.sin(np.pi * (x[0] - 1) / 2) * np.sin(np.pi * (x[1] - 1) / 2))
    vec = u.x.petsc_vec
    set_bc(vec, [bc])
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    opts = {
        "snes_type": "newtonls", "snes_linesearch_type": "l2",
        "snes_linesearch_max_it": 20,
        "snes_atol": 1e-10, "snes_rtol": 1e-8, "snes_max_it": 500,
        "snes_divergence_tolerance": -1.0,
        "ksp_type": "fgmres", "ksp_rtol": 1e-3, "ksp_max_it": 200,
        "pc_type": "hypre", "pc_hypre_type": "boomeramg",
    }
    pfx = "b3b{}_".format(mesh_level)
    problem = NonlinearProblem(J_form, u, petsc_options_prefix=pfx, bcs=[bc], petsc_options=opts)
    t0 = time.time()
    try:
        problem.solve()
    except BaseException:
        pass
    dt = time.time() - t0
    snes = problem.solver
    n_iters = snes.getIterationNumber()
    reason = snes.getConvergedReason()
    snes.destroy()
    final_energy = fem.assemble_scalar(energy_form)
    final_energy = msh.comm.allreduce(final_energy, op=MPI.SUM)
    ok = reason > 0 and abs(final_energy - 0.3456) < 0.01
    return dt, n_iters, reason, final_energy, ok


for lvl in [5, 6, 7, 8]:
    dt, iters, reason, energy, ok = run_test(lvl)
    if rank == 0:
        print("L{}: {:.3f}s  iters={:<4d} reason={:<3d} J={:.6f} {}".format(
            lvl, dt, iters, reason, energy, "OK" if ok else "FAIL"))

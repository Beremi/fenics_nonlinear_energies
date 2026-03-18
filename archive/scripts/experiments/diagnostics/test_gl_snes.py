#!/usr/bin/env python3
"""
Test GL SNES solver with more aggressive settings.
Try bt line search with larger steptol, or l2 line search.
Also compare with JAX energy value.
"""
import sys
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
from petsc4py import PETSc
from dolfinx.fem.petsc import NonlinearProblem, set_bc
from petsc4py.PETSc import ScalarType
import ufl

EPS = 0.01
comm = MPI.COMM_WORLD

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
    eps_c = fem.Constant(msh, ScalarType(EPS))

    F_energy = (
        (eps_c / 2) * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx
        + (1.0 / 4.0) * (u**2 - 1)**2 * ufl.dx
    )
    J_form = ufl.derivative(F_energy, u, v)

    # Initial guess
    def initial_guess(x):
        return np.sin(np.pi * (x[0] - 1) / 2) * np.sin(np.pi * (x[1] - 1) / 2)
    u.interpolate(initial_guess)
    vec = u.x.petsc_vec
    set_bc(vec, [bc])
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Try with bt line search (more robust)
    petsc_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_atol": 1e-6,
        "snes_rtol": 1e-8,
        "snes_max_it": 100,
        "snes_linesearch_max_it": 20,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-1,
    }

    problem = NonlinearProblem(
        J_form, u,
        petsc_options_prefix="gl{}_".format(mesh_level),
        bcs=[bc],
        petsc_options=petsc_opts,
    )

    import time
    t0 = time.time()
    problem.solve()
    dt = time.time() - t0

    snes = problem.solver
    n_iters = snes.getIterationNumber()
    reason = snes.getConvergedReason()
    snes.destroy()

    final_energy = fem.assemble_scalar(fem.form(F_energy))
    final_energy = msh.comm.allreduce(final_energy, op=MPI.SUM)

    # Count free dofs
    n_free = total_dofs - len(bc_dofs)

    sys.stdout.write(
        "level={} N={} total_dofs={} free_dofs~={} time={:.3f}s iters={} reason={} J={:.6f}\n".format(
            mesh_level, N, total_dofs, n_free, dt, n_iters, reason, final_energy
        )
    )
    sys.stdout.flush()

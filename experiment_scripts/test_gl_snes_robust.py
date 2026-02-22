#!/usr/bin/env python3
"""
Test robust SNES configs for GL - trying to find one that works at all levels and processor counts.
"""
import sys
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
from petsc4py import PETSc
from dolfinx.fem.petsc import NonlinearProblem, set_bc
from petsc4py.PETSc import ScalarType
import ufl
import time

EPS = 0.01
comm = MPI.COMM_WORLD

configs = [
    ("basic+gmres+ilu", {
        "snes_type": "newtonls", "snes_linesearch_type": "basic",
        "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
        "ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-1,
    }),
    ("basic+gmres+bjacobi_ilu", {
        "snes_type": "newtonls", "snes_linesearch_type": "basic",
        "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
        "ksp_type": "gmres", "pc_type": "bjacobi",
        "sub_pc_type": "ilu",
        "ksp_rtol": 1e-1,
    }),
    ("basic+gmres+asm_lu", {
        "snes_type": "newtonls", "snes_linesearch_type": "basic",
        "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
        "ksp_type": "gmres", "pc_type": "asm",
        "sub_pc_type": "lu",
        "ksp_rtol": 1e-1,
    }),
    ("basic+gmres+hypre+ksp001", {
        "snes_type": "newtonls", "snes_linesearch_type": "basic",
        "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
        "ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-3,
    }),
]

for mesh_level in [5, 6, 7, 8]:
    for name, petsc_opts in configs:
        N = 2 ** (mesh_level + 1)
        msh = mesh.create_rectangle(
            comm, [[-1.0, -1.0], [1.0, 1.0]], [N, N],
            cell_type=mesh.CellType.triangle,
        )
        V = fem.functionspace(msh, ("Lagrange", 1))
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

        def initial_guess(x):
            return np.sin(np.pi * (x[0] - 1) / 2) * np.sin(np.pi * (x[1] - 1) / 2)
        u.interpolate(initial_guess)
        vec = u.x.petsc_vec
        set_bc(vec, [bc])
        vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        pfx = "l{}_{}_".format(mesh_level, name.replace("+", "_"))
        problem = NonlinearProblem(
            J_form, u, petsc_options_prefix=pfx, bcs=[bc], petsc_options=petsc_opts,
        )

        t0 = time.time()
        try:
            problem.solve()
        except Exception as e:
            sys.stdout.write("level={} {} : EXCEPTION {}\n".format(mesh_level, name, e))
            sys.stdout.flush()
            continue
        dt = time.time() - t0

        snes = problem.solver
        n_iters = snes.getIterationNumber()
        reason = snes.getConvergedReason()
        snes.destroy()

        final_energy = fem.assemble_scalar(fem.form(F_energy))
        final_energy = msh.comm.allreduce(final_energy, op=MPI.SUM)

        converged = reason > 0 and final_energy < 1.0
        sys.stdout.write(
            "level={} {}: time={:.3f}s iters={} reason={} J={:.6f} {}\n".format(
                mesh_level, name, dt, n_iters, reason, final_energy,
                "OK" if converged else "FAIL"
            )
        )
        sys.stdout.flush()

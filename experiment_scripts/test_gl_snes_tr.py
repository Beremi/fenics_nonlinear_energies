#!/usr/bin/env python3
"""Test SNES with trust region and other non-convex-friendly configs."""
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
rank = comm.rank

configs = {
    "newtontr+gmres+hypre": {
        "snes_type": "newtontr", "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
        "ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-1,
    },
    "newtonls+bt+gmres+hypre": {
        "snes_type": "newtonls", "snes_linesearch_type": "bt",
        "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
        "snes_divergence_tolerance": -1.0,
        "ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-1,
    },
    "newtonls+basic+gmres+hypre+divtol": {
        "snes_type": "newtonls", "snes_linesearch_type": "basic",
        "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
        "snes_divergence_tolerance": -1.0,
        "ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-1,
    },
    "nrichardson+gmres+hypre": {
        "snes_type": "nrichardson",
        "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 2000,
        "ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-1,
    },
}

for cname, opts in configs.items():
    if rank == 0:
        sys.stdout.write(f"\n=== Config: {cname} | {comm.size} procs ===\n")
        sys.stdout.flush()
    
    for mesh_level in [5, 6, 7, 8]:
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

        pfx = f"t{hash(cname) % 1000}l{mesh_level}_"
        try:
            problem = NonlinearProblem(J_form, u, petsc_options_prefix=pfx, bcs=[bc], petsc_options=opts)
            t0 = time.time()
            problem.solve()
            dt = time.time() - t0
            snes = problem.solver
            n_iters = snes.getIterationNumber()
            reason = snes.getConvergedReason()
            snes.destroy()
            final_energy = fem.assemble_scalar(fem.form(F_energy))
            final_energy = msh.comm.allreduce(final_energy, op=MPI.SUM)
            ok = reason > 0 and final_energy < 1.0
            if rank == 0:
                sys.stdout.write(f"  L{mesh_level}: {dt:.2f}s iters={n_iters} reason={reason} J={final_energy:.6f} {'OK' if ok else 'FAIL'}\n")
                sys.stdout.flush()
        except Exception as e:
            if rank == 0:
                sys.stdout.write(f"  L{mesh_level}: EXCEPTION {e}\n")
                sys.stdout.flush()
        comm.Barrier()

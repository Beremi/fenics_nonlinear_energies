#!/usr/bin/env python3
"""
Benchmark B4 config: l2 + fgmres + HYPRE, ksp_rtol=1e-3, snes_atol=1e-5.
Run at all levels (5-9), report iters, time, J(u).
"""
import sys, json, numpy as np, time
from mpi4py import MPI
from dolfinx import fem, mesh
from petsc4py import PETSc
from dolfinx.fem.petsc import NonlinearProblem, set_bc
from petsc4py.PETSc import ScalarType
import ufl

comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size


def solve_gl(mesh_level):
    N = 2 ** (mesh_level + 1)
    msh = mesh.create_rectangle(comm, [[-1.0, -1.0], [1.0, 1.0]], [N, N],
                                cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", 1))
    msh.topology.create_connectivity(1, 2)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    bc_dofs = fem.locate_dofs_topological(V, 1, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0), bc_dofs, V)
    u = fem.Function(V); v = ufl.TestFunction(V)
    eps_c = fem.Constant(msh, ScalarType(0.01))
    F_energy = (eps_c / 2) * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx \
               + 0.25 * (u**2 - 1)**2 * ufl.dx
    J_form = ufl.derivative(F_energy, u, v)
    energy_form = fem.form(F_energy)
    u.interpolate(lambda x: np.sin(np.pi * (x[0] - 1) / 2) * np.sin(np.pi * (x[1] - 1) / 2))
    vec = u.x.petsc_vec; set_bc(vec, [bc])
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    ndofs = V.dofmap.index_map.size_global
    opts = {
        "snes_type": "newtonls", "snes_linesearch_type": "l2",
        "snes_linesearch_max_it": 20,
        "snes_atol": 1e-5, "snes_rtol": 1e-8, "snes_max_it": 500,
        "snes_divergence_tolerance": -1.0,
        "ksp_type": "fgmres", "ksp_rtol": 1e-3, "ksp_max_it": 200,
        "pc_type": "hypre", "pc_hypre_type": "boomeramg",
    }
    pfx = "b4_{}{}_".format(nprocs, mesh_level)
    problem = NonlinearProblem(J_form, u, petsc_options_prefix=pfx, bcs=[bc],
                               petsc_options=opts)
    # warmup / compile
    # solve 3 times, take median
    results = []
    for run in range(3):
        u.interpolate(lambda x: np.sin(np.pi * (x[0] - 1) / 2) * np.sin(np.pi * (x[1] - 1) / 2))
        vec = u.x.petsc_vec; set_bc(vec, [bc])
        vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        comm.Barrier()
        t0 = time.time()
        try:
            problem.solve()
        except:
            pass
        comm.Barrier()
        dt = time.time() - t0
        snes = problem.solver
        n_iters = snes.getIterationNumber()
        reason = snes.getConvergedReason()
        final_energy = fem.assemble_scalar(energy_form)
        final_energy = msh.comm.allreduce(final_energy, op=MPI.SUM)
        results.append((dt, n_iters, reason, final_energy))
    snes.destroy()
    # pick median by time
    results.sort(key=lambda x: x[0])
    return ndofs, results[1]  # median


for lvl in [5, 6, 7, 8, 9]:
    ndofs, (dt, iters, reason, energy) = solve_gl(lvl)
    if rank == 0:
        ok = "OK" if reason > 0 else "FAIL(r={})".format(reason)
        line = "L{}: np={} dofs={} time={:.4f} iters={} J={:.7f} {}".format(
            lvl, nprocs, ndofs, dt, iters, energy, ok)
        print(line)
        sys.stdout.flush()
        with open("/work/tmp_work/b4_results_np{}.txt".format(nprocs), "a") as f:
            f.write(line + "\n")

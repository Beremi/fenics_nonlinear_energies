#!/usr/bin/env python3
"""
Benchmark newtontr + fgmres + HYPRE with ksp_rtol from env.
Levels 5-9, smart reps: 3 for L5-L7, 1 for L8-L9 (known failures).
Early stop: if first rep fails, skip remaining reps.
"""
import sys, os, numpy as np, time
from mpi4py import MPI
from dolfinx import fem, mesh
from petsc4py import PETSc
from dolfinx.fem.petsc import NonlinearProblem, set_bc
from petsc4py.PETSc import ScalarType
import ufl

comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size
KSP_RTOL = float(os.environ.get("KSP_RTOL", "1e-3"))


def solve_gl(mesh_level, max_reps=3):
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

    def init_guess(x):
        return np.sin(np.pi * (x[0] - 1) / 2) * np.sin(np.pi * (x[1] - 1) / 2)

    ndofs = V.dofmap.index_map.size_global
    max_it = 300 if mesh_level >= 8 else 500
    opts = {
        "snes_type": "newtontr",
        "snes_atol": 1e-5, "snes_rtol": 1e-8, "snes_max_it": max_it,
        "ksp_type": "fgmres", "ksp_rtol": KSP_RTOL, "ksp_max_it": 200,
        "pc_type": "hypre", "pc_hypre_type": "boomeramg",
    }
    pfx = "tr{}_{}{}_".format(str(KSP_RTOL).replace('.', ''), nprocs, mesh_level)
    problem = NonlinearProblem(J_form, u, petsc_options_prefix=pfx, bcs=[bc],
                               petsc_options=opts)

    def objective(snes, x_vec):
        x_vec.copy(u.x.petsc_vec)
        u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        local_val = fem.assemble_scalar(energy_form)
        return comm.allreduce(local_val, op=MPI.SUM)
    problem.solver.setObjective(objective)

    results = []
    for run in range(max_reps):
        u.interpolate(init_guess)
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
        # Early stop: if first run fails, don't waste time on more reps
        if run == 0 and reason <= 0:
            break
    snes.destroy()
    results.sort(key=lambda x: x[0])
    idx = len(results) // 2  # median
    return ndofs, results[idx]


for lvl in [5, 6, 7, 8, 9]:
    max_reps = 1 if lvl >= 8 else 3
    ndofs, (dt, iters, reason, energy) = solve_gl(lvl, max_reps=max_reps)
    if rank == 0:
        ok = "OK" if reason > 0 else "FAIL(r={})".format(reason)
        line = "L{}: np={} dofs={} time={:.4f} iters={} J={:.7f} {}".format(
            lvl, nprocs, ndofs, dt, iters, energy, ok)
        print(line, flush=True)
        outfile = "/work/tmp_work/tr2_ksp{}_np{}.txt".format(
            str(KSP_RTOL).replace('.', ''), nprocs)
        with open(outfile, "a") as f:
            f.write(line + "\n")

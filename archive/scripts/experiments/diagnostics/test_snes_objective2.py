#!/usr/bin/env python3
"""
Test SNES with various strategies to handle the indefinite Hessian:
1. bt + objective + allow negative alpha (snes_linesearch_minlambda)
2. Modify the Hessian with a shift to make it positive definite
3. Use SNES trust region with objective
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
rank = comm.rank
nprocs = comm.size


def run_test(mesh_level, config_name, petsc_opts, set_objective=False):
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
    energy_form = fem.form(F_energy)

    def initial_guess(x):
        return np.sin(np.pi * (x[0] - 1) / 2) * np.sin(np.pi * (x[1] - 1) / 2)
    u.interpolate(initial_guess)
    vec = u.x.petsc_vec
    set_bc(vec, [bc])
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    pfx = "s{}{}_".format(abs(hash(config_name)) % 10000, mesh_level)
    problem = NonlinearProblem(J_form, u, petsc_options_prefix=pfx, bcs=[bc],
                               petsc_options=petsc_opts)

    if set_objective:
        def objective(snes, x_vec):
            x_vec.copy(u.x.petsc_vec)
            u.x.petsc_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )
            local_val = fem.assemble_scalar(energy_form)
            return comm.allreduce(local_val, op=MPI.SUM)
        problem.solver.setObjective(objective)

    t0 = time.time()
    problem.solve()
    dt = time.time() - t0
    snes = problem.solver
    n_iters = snes.getIterationNumber()
    reason = snes.getConvergedReason()
    snes.destroy()
    final_energy = fem.assemble_scalar(energy_form)
    final_energy = msh.comm.allreduce(final_energy, op=MPI.SUM)
    ok = reason > 0 and abs(final_energy - 0.3456) < 0.01
    return dt, n_iters, reason, final_energy, ok


configs = {
    # Try bt with very small minlambda (allow very small steps)
    "bt+obj+minlambda=1e-20": (
        {"snes_type": "newtonls", "snes_linesearch_type": "bt",
         "snes_linesearch_minlambda": 1e-20,
         "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
         "ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-1},
        True,
    ),
    # Try l2 with objective
    "l2+obj": (
        {"snes_type": "newtonls", "snes_linesearch_type": "l2",
         "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
         "snes_divergence_tolerance": -1.0,
         "ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-1},
        True,
    ),
    # Try basic with objective — basic always takes alpha=1, but let's see if
    # objective monitoring changes behavior
    "basic+obj": (
        {"snes_type": "newtonls", "snes_linesearch_type": "basic",
         "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
         "snes_divergence_tolerance": -1.0,
         "ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-1},
        True,
    ),
    # Trust region with objective
    "newtontr+obj": (
        {"snes_type": "newtontr",
         "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
         "ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-1},
        True,
    ),
    # NGMRES — nonlinear GMRES, a different approach
    "ngmres+basic": (
        {"snes_type": "ngmres",
         "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
         "snes_divergence_tolerance": -1.0,
         "npc_snes_type": "newtonls", "npc_snes_linesearch_type": "basic",
         "npc_snes_max_it": 1, "npc_ksp_type": "gmres", "npc_pc_type": "hypre",
         "npc_ksp_rtol": 1e-1},
        False,
    ),
}

for cname, (opts, set_obj) in configs.items():
    if rank == 0:
        sys.stdout.write("\n=== {} | {} procs ===\n".format(cname, nprocs))
        sys.stdout.flush()
    for lvl in [5, 6, 7, 8]:
        try:
            dt, iters, reason, energy, ok = run_test(lvl, cname, opts, set_obj)
            if rank == 0:
                sys.stdout.write("  L{}: {:.3f}s iters={} reason={} J={:.6f} {}\n".format(
                    lvl, dt, iters, reason, energy, "OK" if ok else "FAIL"))
                sys.stdout.flush()
        except Exception as e:
            if rank == 0:
                sys.stdout.write("  L{}: EXCEPTION {}\n".format(lvl, e))
                sys.stdout.flush()
        comm.Barrier()

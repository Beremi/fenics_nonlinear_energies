#!/usr/bin/env python3
"""
Sweep nonlinear solver tolerances for B4 variant (l2 + fgmres + HYPRE, ksp_rtol=1e-3).
Find the loosest snes_atol/snes_rtol that still matches the reference energy (rel error < 1e-5).

First compute reference energy at tight tolerance, then test progressively looser tolerances.
"""
import sys, numpy as np, time
from mpi4py import MPI
from dolfinx import fem, mesh
from petsc4py import PETSc
from dolfinx.fem.petsc import NonlinearProblem, set_bc
from petsc4py.PETSc import ScalarType
import ufl

comm = MPI.COMM_WORLD
rank = comm.rank


def solve_gl(mesh_level, snes_atol, snes_rtol, ksp_rtol=1e-3):
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
    opts = {
        "snes_type": "newtonls", "snes_linesearch_type": "l2",
        "snes_linesearch_max_it": 20,
        "snes_atol": snes_atol, "snes_rtol": snes_rtol, "snes_max_it": 500,
        "snes_divergence_tolerance": -1.0,
        "ksp_type": "fgmres", "ksp_rtol": ksp_rtol, "ksp_max_it": 200,
        "pc_type": "hypre", "pc_hypre_type": "boomeramg",
    }
    pfx = "tol{}{}_{}_".format(abs(hash(str(snes_atol))) % 1000,
                                abs(hash(str(snes_rtol))) % 1000, mesh_level)
    problem = NonlinearProblem(J_form, u, petsc_options_prefix=pfx, bcs=[bc],
                               petsc_options=opts)
    t0 = time.time()
    try:
        problem.solve()
    except:
        pass
    dt = time.time() - t0
    snes = problem.solver
    n_iters = snes.getIterationNumber(); reason = snes.getConvergedReason()
    snes.destroy()
    final_energy = fem.assemble_scalar(energy_form)
    final_energy = msh.comm.allreduce(final_energy, op=MPI.SUM)
    return dt, n_iters, reason, final_energy


# ---------- Phase 1: reference energies at tight tolerance ----------
if rank == 0:
    print("=== Phase 1: Reference energies (snes_atol=1e-12, snes_rtol=1e-12) ===")
ref_energies = {}
for lvl in [5, 6, 7, 8]:
    dt, iters, reason, energy = solve_gl(lvl, 1e-12, 1e-12, ksp_rtol=1e-6)
    ref_energies[lvl] = energy
    if rank == 0:
        print(f"  L{lvl}: {dt:.3f}s  iters={iters:<4d} reason={reason:<3d} J={energy:.12f}")

# ---------- Phase 2: sweep tolerances ----------
# We test combinations of snes_atol and snes_rtol
atol_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
rtol_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

if rank == 0:
    print("\n=== Phase 2: Tolerance sweep (ksp_rtol=1e-3) ===")
    print(f"{'atol':>10s} {'rtol':>10s} | ", end="")
    for lvl in [5, 6, 7, 8]:
        print(f"{'L'+str(lvl)+' iters':>10s} {'J':>14s} {'relErr':>10s} | ", end="")
    print("all OK?")
    print("-" * 140)

for atol in atol_values:
    for rtol in rtol_values:
        results = {}
        all_ok = True
        for lvl in [5, 6, 7, 8]:
            dt, iters, reason, energy = solve_gl(lvl, atol, rtol)
            rel_err = abs(energy - ref_energies[lvl]) / abs(ref_energies[lvl])
            ok = reason > 0 and rel_err < 1e-5
            results[lvl] = (iters, energy, rel_err, ok)
            if not ok:
                all_ok = False
        if rank == 0:
            line = f"{atol:>10.0e} {rtol:>10.0e} | "
            for lvl in [5, 6, 7, 8]:
                iters, energy, rel_err, ok = results[lvl]
                tag = "" if ok else " FAIL"
                line += f"{iters:>10d} {energy:>14.10f} {rel_err:>10.2e}{tag:>5s} | "
            line += "YES" if all_ok else "NO"
            print(line)
            sys.stdout.flush()

if rank == 0:
    print("\n=== DONE ===")

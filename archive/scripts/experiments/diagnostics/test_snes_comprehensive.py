#!/usr/bin/env python3
"""
Comprehensive SNES configuration survey for the Ginzburg-Landau 2D problem.

Tests all configurations suggested in the advisory note:
  - newtontr variants (fgmres, ksp_ew, ASM+ILU)
  - newtonls + l2 variants (fgmres+GAMG, ASM+ILU)
  - Eisenstat-Walker, lag preconditioner
  - epsilon continuation
  - tanh initial guess

All serial, levels 5-8.
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

comm = MPI.COMM_WORLD
rank = comm.rank

TARGET_J = 0.3456  # correct minimum energy (approx)


def solve_gl(mesh_level, petsc_opts, set_objective=False, eps_val=0.01,
             init_guess="sine", u_prev_array=None, pfx_tag="x"):
    """Solve GL at given mesh level with given SNES options.
    
    init_guess: "sine" | "tanh" | "from_array"
    """
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
    eps_c = fem.Constant(msh, ScalarType(eps_val))
    F_energy = (
        (eps_c / 2) * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx
        + (1.0 / 4.0) * (u**2 - 1)**2 * ufl.dx
    )
    J_form = ufl.derivative(F_energy, u, v)
    energy_form = fem.form(F_energy)

    if init_guess == "sine":
        u.interpolate(lambda x: np.sin(np.pi * (x[0] - 1) / 2) *
                                 np.sin(np.pi * (x[1] - 1) / 2))
    elif init_guess == "tanh":
        sqrt_eps = np.sqrt(eps_val)
        def tanh_guess(x):
            # distance to boundary of [-1,1]^2
            dist = np.minimum(
                np.minimum(x[0] - (-1.0), 1.0 - x[0]),
                np.minimum(x[1] - (-1.0), 1.0 - x[1])
            )
            return np.tanh(dist / sqrt_eps)
        u.interpolate(tanh_guess)
    elif init_guess == "from_array" and u_prev_array is not None:
        u.x.array[:] = u_prev_array
    
    vec = u.x.petsc_vec
    set_bc(vec, [bc])
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    pfx = "{}{}{}_".format(pfx_tag, abs(hash(str(petsc_opts))) % 10000, mesh_level)
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
    try:
        problem.solve()
    except Exception:
        pass
    dt = time.time() - t0
    snes = problem.solver
    n_iters = snes.getIterationNumber()
    reason = snes.getConvergedReason()
    snes.destroy()
    final_energy = fem.assemble_scalar(energy_form)
    final_energy = msh.comm.allreduce(final_energy, op=MPI.SUM)
    ok = reason > 0 and abs(final_energy - TARGET_J) < 0.01
    return dt, n_iters, reason, final_energy, ok, u.x.array.copy()


def run_config(label, petsc_opts, levels, set_objective=False,
               init_guess="sine", pfx_tag="x"):
    """Run a config over multiple levels and print results."""
    if rank == 0:
        sys.stdout.write("\n=== {} | {} procs ===\n".format(label, comm.size))
        sys.stdout.flush()
    for lvl in levels:
        try:
            dt, iters, reason, energy, ok, _ = solve_gl(
                lvl, petsc_opts, set_objective=set_objective,
                init_guess=init_guess, pfx_tag=pfx_tag)
            if rank == 0:
                tag = "OK" if ok else "FAIL"
                sys.stdout.write(
                    "  L{}: {:.3f}s  iters={:<4d} reason={:<3d} J={:.6f} {}\n"
                    .format(lvl, dt, iters, reason, energy, tag))
                sys.stdout.flush()
        except Exception as e:
            if rank == 0:
                sys.stdout.write("  L{}: EXCEPTION {}\n".format(lvl, str(e)[:80]))
                sys.stdout.flush()
        comm.Barrier()


def run_continuation(label, petsc_opts, levels, set_objective=False, pfx_tag="c"):
    """Run with epsilon continuation: 0.1 -> 0.05 -> 0.02 -> 0.01."""
    eps_sequence = [0.1, 0.05, 0.02, 0.01]
    if rank == 0:
        sys.stdout.write("\n=== {} | {} procs ===\n".format(label, comm.size))
        sys.stdout.flush()
    for lvl in levels:
        total_t = 0
        total_iters = 0
        u_array = None
        ig = "sine"
        final_energy = 0.0
        final_ok = False
        final_reason = 0
        failed = False
        eps_details = []
        for eps_val in eps_sequence:
            try:
                dt, iters, reason, energy, ok, u_arr = solve_gl(
                    lvl, petsc_opts, set_objective=set_objective,
                    eps_val=eps_val, init_guess=ig, u_prev_array=u_array,
                    pfx_tag=pfx_tag)
                total_t += dt
                total_iters += iters
                u_array = u_arr
                ig = "from_array"
                final_energy = energy
                final_reason = reason
                eps_details.append("e{:.2f}:{}it".format(eps_val, iters))
                if reason <= 0:
                    failed = True
                    break
            except Exception as e:
                failed = True
                eps_details.append("e{:.2f}:EXC".format(eps_val))
                break
        final_ok = (not failed) and abs(final_energy - TARGET_J) < 0.01
        if rank == 0:
            tag = "OK" if final_ok else "FAIL"
            detail = " | ".join(eps_details)
            sys.stdout.write(
                "  L{}: {:.3f}s  iters={:<4d} reason={:<3d} J={:.6f} {} [{}]\n"
                .format(lvl, total_t, total_iters, final_reason, final_energy,
                        tag, detail))
            sys.stdout.flush()
        comm.Barrier()


LEVELS = [5, 6, 7, 8]

# =====================================================================
# Group A: newtontr variants
# =====================================================================

# A1: newtontr + fgmres + HYPRE (the "good default pack" from the note)
run_config("A1: newtontr + fgmres + HYPRE", {
    "snes_type": "newtontr",
    "snes_atol": 1e-10, "snes_rtol": 1e-8, "snes_max_it": 500,
    "ksp_type": "fgmres", "ksp_rtol": 1e-6, "ksp_max_it": 200,
    "pc_type": "hypre", "pc_hypre_type": "boomeramg",
}, LEVELS, set_objective=True, pfx_tag="a1")

# A2: newtontr + ksp_ew + fgmres + HYPRE (the "what I'd try first" from the note)
run_config("A2: newtontr + ksp_ew + fgmres + HYPRE", {
    "snes_type": "newtontr",
    "snes_ksp_ew": None,
    "snes_atol": 1e-10, "snes_rtol": 1e-8, "snes_max_it": 500,
    "ksp_type": "fgmres", "ksp_rtol": 1e-6, "ksp_max_it": 200,
    "pc_type": "hypre", "pc_hypre_type": "boomeramg",
}, LEVELS, set_objective=True, pfx_tag="a2")

# A3: newtontr + ASM+ILU fallback
run_config("A3: newtontr + fgmres + ASM+ILU", {
    "snes_type": "newtontr",
    "snes_atol": 1e-10, "snes_rtol": 1e-8, "snes_max_it": 500,
    "ksp_type": "fgmres", "ksp_rtol": 1e-6, "ksp_max_it": 200,
    "pc_type": "asm", "pc_asm_overlap": 2,
    "sub_pc_type": "ilu", "sub_pc_factor_levels": 1,
}, LEVELS, set_objective=True, pfx_tag="a3")

# A4: newtontr + gmres + HYPRE + ksp_rtol=1e-1 (our previous test settings)
run_config("A4: newtontr + gmres + HYPRE (rtol=1e-1)", {
    "snes_type": "newtontr",
    "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
    "ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-1,
}, LEVELS, set_objective=True, pfx_tag="a4")

# =====================================================================
# Group B: newtonls + l2 variants
# =====================================================================

# B1: newtonls + l2 + fgmres + GAMG (from the note)
run_config("B1: newtonls + l2 + fgmres + GAMG", {
    "snes_type": "newtonls", "snes_linesearch_type": "l2",
    "snes_linesearch_max_it": 20,
    "snes_atol": 1e-10, "snes_rtol": 1e-8, "snes_max_it": 500,
    "snes_divergence_tolerance": -1.0,
    "ksp_type": "fgmres", "ksp_rtol": 1e-6, "ksp_max_it": 200,
    "pc_type": "gamg",
}, LEVELS, set_objective=False, pfx_tag="b1")

# B2: newtonls + l2 + fgmres + ASM+ILU
run_config("B2: newtonls + l2 + fgmres + ASM+ILU", {
    "snes_type": "newtonls", "snes_linesearch_type": "l2",
    "snes_linesearch_max_it": 20,
    "snes_atol": 1e-10, "snes_rtol": 1e-8, "snes_max_it": 500,
    "snes_divergence_tolerance": -1.0,
    "ksp_type": "fgmres", "ksp_rtol": 1e-6, "ksp_max_it": 200,
    "pc_type": "asm", "pc_asm_overlap": 2,
    "sub_pc_type": "ilu", "sub_pc_factor_levels": 1,
}, LEVELS, set_objective=False, pfx_tag="b2")

# B3: newtonls + l2 + fgmres + HYPRE (like our prev but fgmres+tighter ksp)
run_config("B3: newtonls + l2 + fgmres + HYPRE", {
    "snes_type": "newtonls", "snes_linesearch_type": "l2",
    "snes_linesearch_max_it": 20,
    "snes_atol": 1e-10, "snes_rtol": 1e-8, "snes_max_it": 500,
    "snes_divergence_tolerance": -1.0,
    "ksp_type": "fgmres", "ksp_rtol": 1e-6, "ksp_max_it": 200,
    "pc_type": "hypre", "pc_hypre_type": "boomeramg",
}, LEVELS, set_objective=False, pfx_tag="b3")

# =====================================================================
# Group C: Eisenstat-Walker and lag preconditioner
# =====================================================================

# C1: basic + ksp_ew 
run_config("C1: basic + ksp_ew + gmres + HYPRE", {
    "snes_type": "newtonls", "snes_linesearch_type": "basic",
    "snes_ksp_ew": None,
    "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
    "snes_divergence_tolerance": -1.0,
    "ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-1,
}, LEVELS, set_objective=False, pfx_tag="c1")

# C2: basic + lag_preconditioner=2
run_config("C2: basic + lag_pc=2 + gmres + HYPRE", {
    "snes_type": "newtonls", "snes_linesearch_type": "basic",
    "snes_lag_preconditioner": 2,
    "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
    "snes_divergence_tolerance": -1.0,
    "ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-1,
}, LEVELS, set_objective=False, pfx_tag="c2")

# C3: basic + ksp_ew + lag_pc + fgmres + HYPRE
run_config("C3: basic + ksp_ew + lag_pc=2 + fgmres + HYPRE", {
    "snes_type": "newtonls", "snes_linesearch_type": "basic",
    "snes_ksp_ew": None, "snes_lag_preconditioner": 2,
    "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
    "snes_divergence_tolerance": -1.0,
    "ksp_type": "fgmres", "ksp_rtol": 1e-6, "ksp_max_it": 200,
    "pc_type": "hypre", "pc_hypre_type": "boomeramg",
}, LEVELS, set_objective=False, pfx_tag="c3")

# =====================================================================
# Group D: Initial guess variations
# =====================================================================

# D1: basic + tanh initial guess
run_config("D1: basic + tanh init + gmres + HYPRE", {
    "snes_type": "newtonls", "snes_linesearch_type": "basic",
    "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
    "snes_divergence_tolerance": -1.0,
    "ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-1,
}, LEVELS, set_objective=False, init_guess="tanh", pfx_tag="d1")

# D2: newtontr + tanh initial guess
run_config("D2: newtontr + tanh init + fgmres + HYPRE", {
    "snes_type": "newtontr",
    "snes_ksp_ew": None,
    "snes_atol": 1e-10, "snes_rtol": 1e-8, "snes_max_it": 500,
    "ksp_type": "fgmres", "ksp_rtol": 1e-6, "ksp_max_it": 200,
    "pc_type": "hypre", "pc_hypre_type": "boomeramg",
}, LEVELS, set_objective=True, init_guess="tanh", pfx_tag="d2")

# D3: bt + objective + tanh init
run_config("D3: bt + obj + tanh init + fgmres + HYPRE", {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_atol": 1e-10, "snes_rtol": 1e-8, "snes_max_it": 500,
    "ksp_type": "fgmres", "ksp_rtol": 1e-6, "ksp_max_it": 200,
    "pc_type": "hypre", "pc_hypre_type": "boomeramg",
}, LEVELS, set_objective=True, init_guess="tanh", pfx_tag="d3")

# =====================================================================
# Group E: Epsilon continuation
# =====================================================================

# E1: basic + continuation (0.1 -> 0.05 -> 0.02 -> 0.01)
run_continuation("E1: basic + eps-continuation + gmres + HYPRE", {
    "snes_type": "newtonls", "snes_linesearch_type": "basic",
    "snes_atol": 1e-6, "snes_rtol": 1e-8, "snes_max_it": 500,
    "snes_divergence_tolerance": -1.0,
    "ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-1,
}, LEVELS, set_objective=False, pfx_tag="e1")

# E2: newtontr + continuation
run_continuation("E2: newtontr + eps-continuation + fgmres + HYPRE", {
    "snes_type": "newtontr",
    "snes_ksp_ew": None,
    "snes_atol": 1e-10, "snes_rtol": 1e-8, "snes_max_it": 500,
    "ksp_type": "fgmres", "ksp_rtol": 1e-6, "ksp_max_it": 200,
    "pc_type": "hypre", "pc_hypre_type": "boomeramg",
}, LEVELS, set_objective=True, pfx_tag="e2")

# E3: bt + objective + continuation
run_continuation("E3: bt + obj + eps-continuation + fgmres + HYPRE", {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_atol": 1e-10, "snes_rtol": 1e-8, "snes_max_it": 500,
    "ksp_type": "fgmres", "ksp_rtol": 1e-6, "ksp_max_it": 200,
    "pc_type": "hypre", "pc_hypre_type": "boomeramg",
}, LEVELS, set_objective=True, pfx_tag="e3")

if rank == 0:
    print("\n=== ALL DONE ===")

"""
p-Laplace 2D — FEniCS custom Newton solver logic.

Provides ``run_level()`` using the JAX-version Newton algorithm
(golden-section line search) on top of DOLFINx assembly.
CLI entry point is in ``solve_pLaplace_custom_jaxversion.py``.
"""

import sys
import time

import h5py
import numpy as np
import ufl
import basix.ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_matrix,
    set_bc,
)

from tools_petsc4py.minimizers import newton
from tools_petsc4py.fenics_tools import ghost_update as _ghost_update


def run_level(mesh_level, verbose=True, pc_type="hypre", ksp_rtol=1e-3,
              linesearch_interval=(-0.5, 2.0), linesearch_tol=1e-3,
              maxit=100, use_trust_region=False, trust_radius_init=1.0,
              trust_radius_min=1e-8, trust_radius_max=1e6,
              trust_shrink=0.5, trust_expand=1.5,
              trust_eta_shrink=0.05, trust_eta_expand=0.75,
              trust_max_reject=6):
    """Run JAX-version Newton solver for one mesh level.

    Returns dict with timing breakdown matching the JAX-PETSc benchmark format.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # ---- mesh + function space ----
    if rank == 0:
        with h5py.File(f"mesh_data/pLaplace/pLaplace_level{mesh_level}.h5", "r",
                       driver="core", backing_store=False) as f:
            points = f["nodes"][:]
            triangles = f["elems"][:].astype(np.int64)
    else:
        points = np.empty((0, 2), dtype=np.float64)
        triangles = np.empty((0, 3), dtype=np.int64)
    c_el = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    msh = mesh.create_mesh(comm, triangles, c_el, points)

    V = fem.functionspace(msh, ("Lagrange", 1))
    total_dofs = V.dofmap.index_map.size_global

    # ---- Dirichlet BC (u = 0 on ∂Ω) ----
    msh.topology.create_connectivity(1, 2)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    bc_dofs = fem.locate_dofs_topological(V, 1, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0), bc_dofs, V)

    t_setup_start = time.perf_counter()

    # ---- variational forms ----
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    w = ufl.TrialFunction(V)

    p = 3.0
    f_rhs = fem.Constant(msh, ScalarType(-10.0))

    J_energy = (
        (1.0 / p) * ufl.inner(ufl.grad(u), ufl.grad(u)) ** (p / 2) * ufl.dx
        - f_rhs * u * ufl.dx
    )
    dJ = ufl.derivative(J_energy, u, v)
    ddJ = ufl.derivative(dJ, u, w)

    grad_form = fem.form(dJ)
    hessian_form = fem.form(ddJ)

    # ---- form for energy evaluation at arbitrary point ----
    u_ls = fem.Function(V)
    energy_ls = ufl.replace(J_energy, {u: u_ls})
    energy_ls_form = fem.form(energy_ls)

    # ---- initial guess  (small random, same as other FEniCS benchmarks) ----
    np.random.seed(42)
    x = u.x.petsc_vec
    lo, hi = x.getOwnershipRange()
    x.setValues(range(lo, hi), 1e-2 * np.random.rand(hi - lo))
    _ghost_update(x)
    x.assemble()
    set_bc(x, [bc])

    # ---- pre-allocate Hessian matrix and KSP ----
    A = create_matrix(hessian_form)
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setType(PETSc.KSP.Type.CG)
    if pc_type == "gamg":
        ksp.getPC().setType(PETSc.PC.Type.GAMG)
    else:
        ksp.getPC().setType(PETSc.PC.Type.HYPRE)
    ksp.setTolerances(rtol=ksp_rtol)

    setup_time = time.perf_counter() - t_setup_start

    # ------------------------------------------------------------------
    # Callbacks for tools_petsc4py.minimizers.newton
    # ------------------------------------------------------------------

    def energy_fn(vec):
        """J(u) at an arbitrary PETSc Vec (globally reduced scalar)."""
        vec.copy(u_ls.x.petsc_vec)
        u_ls.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        local_val = fem.assemble_scalar(energy_ls_form)
        return comm.allreduce(local_val, op=MPI.SUM)

    def gradient_fn(vec, g):
        """Assemble ∇J into *g* (BCs applied, ghost-updated)."""
        with g.localForm() as g_loc:
            g_loc.set(0.0)
        assemble_vector(g, grad_form)
        apply_lifting(g, [hessian_form], [[bc]], x0=[vec])
        g.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(g, [bc], vec)

    _hess_timings = []

    def hessian_solve_fn(vec, rhs, sol):
        """Assemble Hessian, solve H · sol = rhs. Return KSP iters."""
        t0 = time.perf_counter()
        A.zeroEntries()
        assemble_matrix(A, hessian_form, bcs=[bc])
        A.assemble()
        t_asm = time.perf_counter() - t0

        t0 = time.perf_counter()
        ksp.setOperators(A)
        t_setop = time.perf_counter() - t0

        t0 = time.perf_counter()
        ksp.setUp()
        t_pc = time.perf_counter() - t0

        t0 = time.perf_counter()
        ksp.solve(rhs, sol)
        ksp_its = ksp.getIterationNumber()
        t_solve = time.perf_counter() - t0

        _hess_timings.append({
            "assembly": t_asm,
            "setop": t_setop,
            "pc_setup": t_pc,
            "solve": t_solve,
            "ksp": t_pc + t_solve,
            "ksp_its": ksp_its,
            "total": t_asm + t_setop + t_pc + t_solve,
        })
        return ksp_its

    # ---- solve ----
    t_start = time.perf_counter()
    result = newton(
        energy_fn,
        gradient_fn,
        hessian_solve_fn,
        x,
        tolf=1e-5,
        tolg=1e-3,
        linesearch_tol=linesearch_tol,
        linesearch_interval=linesearch_interval,
        maxit=maxit,
        verbose=verbose,
        comm=comm,
        ghost_update_fn=_ghost_update,
        project_fn=lambda vec: set_bc(vec, [bc]),
        hessian_matvec_fn=lambda _x, vin, vout: A.mult(vin, vout),
        trust_region=use_trust_region,
        trust_radius_init=trust_radius_init,
        trust_radius_min=trust_radius_min,
        trust_radius_max=trust_radius_max,
        trust_shrink=trust_shrink,
        trust_expand=trust_expand,
        trust_eta_shrink=trust_eta_shrink,
        trust_eta_expand=trust_eta_expand,
        trust_max_reject=trust_max_reject,
    )
    total_time = time.perf_counter() - t_start

    final_energy = result["fun"]

    # ---- print per-iteration breakdown (rank 0 only) ----
    if verbose and rank == 0 and _hess_timings:
        other_time = total_time - sum(d["total"] for d in _hess_timings)
        sys.stdout.write("\n  Timing Breakdown (hessian_solve_fn per Newton iteration):\n")
        sys.stdout.write(
            f"  {'It':>3s} {'assembly':>10s} {'PC':>10s} {'solve':>10s} {'KSP it':>7s} {'total':>10s}\n"
        )
        sys.stdout.write("  " + "-" * 58 + "\n")
        for i, d in enumerate(_hess_timings):
            sys.stdout.write(
                f"  {i:3d} {d['assembly']:10.4f} {d['pc_setup']:10.4f} {d['solve']:10.4f}"
                f" {d['ksp_its']:7d} {d['total']:10.4f}\n"
            )
        sys.stdout.write("  " + "-" * 58 + "\n")
        asm_sum = sum(d["assembly"] for d in _hess_timings)
        pc_sum = sum(d["pc_setup"] for d in _hess_timings)
        solve_sum = sum(d["solve"] for d in _hess_timings)
        tot_sum = sum(d["total"] for d in _hess_timings)
        sys.stdout.write(
            f"  {'SUM':>3s} {asm_sum:10.4f} {pc_sum:10.4f} {solve_sum:10.4f}"
            f" {'':>7s} {tot_sum:10.4f}\n"
        )
        sys.stdout.write(
            f"\n  Hessian total: {tot_sum:.4f}s  |  "
            f"Other (energy+grad+LS): {other_time:.4f}s  |  "
            f"Solve wall: {total_time:.4f}s\n"
        )
        sys.stdout.flush()

    # ---- clean up PETSc objects ----
    ksp.destroy()
    A.destroy()

    asm_cumulative = sum(d["assembly"] for d in _hess_timings)
    pc_setup_cumulative = sum(d["pc_setup"] for d in _hess_timings)
    linear_solve_cumulative = sum(d["solve"] for d in _hess_timings)
    ksp_cumulative = sum(d["ksp"] for d in _hess_timings)
    total_ksp_its = sum(d["ksp_its"] for d in _hess_timings)

    return {
        "mesh_level": mesh_level,
        "total_dofs": total_dofs,
        "dofs": total_dofs,
        "nprocs": comm.size,
        "pc_type": pc_type,
        "ksp_rtol": ksp_rtol,
        "assembly_mode": "fenics",
        "setup_time": round(setup_time, 4),
        "solve_time": round(total_time, 4),
        "total_time": round(setup_time + total_time, 4),
        "iters": result["nit"],
        "energy": round(final_energy, 10),
        "message": result["message"],
        "total_ksp_its": total_ksp_its,
        "asm_time_cumulative": round(asm_cumulative, 4),
        "pc_setup_time_cumulative": round(pc_setup_cumulative, 4),
        "linear_solve_time_cumulative": round(linear_solve_cumulative, 4),
        "ksp_time_cumulative": round(ksp_cumulative, 4),
        "hess_timings": list(_hess_timings),
    }

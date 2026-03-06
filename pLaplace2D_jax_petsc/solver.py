"""
p-Laplace 2D — solver logic (DOF-partitioned JAX + PETSc).

Provides ``run_level()`` which runs the Newton solver for one mesh level
and returns a result dict. Import ``run_level`` from here; CLI entry point
is in ``solve_pLaplace_dof.py``.
"""

import sys
import time

import numpy as np
from mpi4py import MPI

from pLaplace2D_petsc_support.mesh import MeshpLaplace2D
from pLaplace2D_jax_petsc.parallel_hessian_dof import (
    LocalColoringAssembler,
    ParallelDOFHessianAssembler,
)
from tools_petsc4py.minimizers import newton


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_newton_breakdown(history, solve_time):
    """Print per-iteration Newton-level timing breakdown (rank 0)."""
    sys.stdout.write("\n  Newton Iteration Breakdown:\n")
    sys.stdout.write(
        f"  {'It':>3s} {'grad':>8s} {'hess':>8s} {'LS':>8s} "
        f"{'update':>8s} {'ls_ev':>6s} {'KSP it':>7s} {'iter':>8s}\n"
    )
    sys.stdout.write("  " + "-" * 58 + "\n")

    s_grad = s_hess = s_ls = s_update = 0.0
    for h in history:
        s_grad += h["t_grad"]
        s_hess += h["t_hess"]
        s_ls += h["t_ls"]
        s_update += h["t_update"]
        sys.stdout.write(
            f"  {h['it']:3d} {h['t_grad']:8.4f} {h['t_hess']:8.4f} "
            f"{h['t_ls']:8.4f} {h['t_update']:8.4f} {h['ls_evals']:6d} "
            f"{h['ksp_its']:7d} {h['t_iter']:8.4f}\n"
        )

    sys.stdout.write("  " + "-" * 58 + "\n")
    s_total = s_grad + s_hess + s_ls + s_update
    sys.stdout.write(
        f"  {'SUM':>3s} {s_grad:8.4f} {s_hess:8.4f} "
        f"{s_ls:8.4f} {s_update:8.4f} {'':>6s} {'':>7s} {s_total:8.4f}\n"
    )
    overhead = solve_time - s_total
    sys.stdout.write(
        f"\n  grad={s_grad:.4f}s  hess={s_hess:.4f}s  "
        f"LS={s_ls:.4f}s  update={s_update:.4f}s  "
        f"overhead={overhead:.4f}s  solve={solve_time:.4f}s\n"
    )
    sys.stdout.flush()


def _print_assembly_breakdown(assembler_timings, n_iters):
    """Print Hessian assembly timing breakdown."""
    if not assembler_timings:
        return
    sys.stdout.write("\n  Hessian Assembly Breakdown (per Newton iteration):\n")
    sys.stdout.write(
        f"  {'It':>3s} {'allgath':>8s} {'HVP':>8s} {'COO':>8s} "
        f"{'KSP':>8s} {'KSP it':>7s} {'total':>8s}\n"
    )
    sys.stdout.write("  " + "-" * 52 + "\n")
    for i, d in enumerate(assembler_timings):
        sys.stdout.write(
            f"  {i:3d} {d.get('allgatherv', 0):8.4f} "
            f"{d.get('hvp_compute', 0):8.4f} "
            f"{d.get('coo_assembly', 0):8.4f} "
            f"{d.get('ksp', 0):8.4f} {d.get('ksp_its', 0):7d} "
            f"{d.get('total_with_ksp', d.get('total', 0)):8.4f}\n"
        )
    sys.stdout.write("  " + "-" * 52 + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Solver for a single mesh level
# ---------------------------------------------------------------------------

def run_level(mesh_level, comm, verbose=True, coloring_trials=10,
              ksp_rtol=1e-3, pc_type="hypre", tolf=1e-5, tolg=1e-3,
              local_coloring=False, assembly_mode="sfd", nproc_threads=1,
              linesearch_interval=(-0.5, 2.0), linesearch_tol=1e-3,
              maxit=100, use_trust_region=False, trust_radius_init=1.0,
              trust_radius_min=1e-8, trust_radius_max=1e6,
              trust_shrink=0.5, trust_expand=1.5,
              trust_eta_shrink=0.05, trust_eta_expand=0.75,
              trust_max_reject=6):
    """Run DOF-partitioned parallel solver for one mesh level.

    Returns dict with timing and convergence info.
    """
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # ---- load mesh ----
    setup_start = time.perf_counter()
    mesh_obj = MeshpLaplace2D(mesh_level=mesh_level)
    params, adjacency, u_init = mesh_obj.get_data_jax()
    n_dofs = len(u_init)
    t_mesh = time.perf_counter() - setup_start

    # ---- build DOF-partitioned assembler ----
    t0 = time.perf_counter()
    AssemblerClass = LocalColoringAssembler if local_coloring else ParallelDOFHessianAssembler
    assembler = AssemblerClass(
        params=params,
        comm=comm,
        adjacency=adjacency,
        coloring_trials_per_rank=coloring_trials,
        ksp_rtol=ksp_rtol,
        ksp_type="cg",
        pc_type=pc_type,
    )
    t_assembler = time.perf_counter() - t0

    use_element_assembly = (assembly_mode == "element")
    if use_element_assembly:
        if not local_coloring:
            raise ValueError("--assembly-mode element requires --local-coloring")
        assembler.setup_element_hessian()

    setup_time = time.perf_counter() - setup_start

    if verbose and rank == 0:
        n_colors = assembler.n_colors
        overlap = assembler._sum_local_elems / (params["elems"].shape[0]) - 1.0
        sys.stdout.write(
            f"  DOFs={n_dofs}  elements={params['elems'].shape[0]}  "
            f"np={nprocs}  NPROC={nproc_threads}  pc={pc_type}\n"
        )
        sys.stdout.write(
            f"  Setup: {setup_time:.3f}s  (mesh={t_mesh:.3f}s, "
            f"assembler={t_assembler:.3f}s)  "
            f"n_colors={n_colors}  overlap={overlap:.1%}\n"
        )
        sys.stdout.flush()

    # ---- initial guess as distributed PETSc Vec ----
    u_init_reord = np.asarray(u_init, dtype=np.float64)[assembler.part.perm]
    x = assembler.create_vec(u_init_reord)

    # ---- hessian_solve_fn: dispatch sfd vs element ----
    _linear_iters = []
    _pc_setup_times = []
    _linear_solve_times = []

    def hessian_solve_fn(vec, rhs, sol):
        u_owned = np.array(vec.array[:], dtype=np.float64)
        if use_element_assembly:
            assembler.assemble_hessian_element(u_owned)
        else:
            assembler.assemble_hessian(u_owned, variant=2)
        assembler.ksp.setOperators(assembler.A)

        t0 = time.perf_counter()
        assembler.ksp.setUp()
        _pc_setup_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        assembler.ksp.solve(rhs, sol)
        _linear_solve_times.append(time.perf_counter() - t0)
        ksp_its = int(assembler.ksp.getIterationNumber())
        _linear_iters.append(ksp_its)

        if assembler.iter_timings:
            assembler.iter_timings[-1]["pc_setup"] = float(_pc_setup_times[-1])
            assembler.iter_timings[-1]["solve"] = float(_linear_solve_times[-1])
            assembler.iter_timings[-1]["ksp"] = float(_pc_setup_times[-1] + _linear_solve_times[-1])
            assembler.iter_timings[-1]["ksp_its"] = int(ksp_its)
            assembler.iter_timings[-1]["total_with_ksp"] = float(
                assembler.iter_timings[-1].get("total", 0.0)
                + _pc_setup_times[-1]
                + _linear_solve_times[-1]
            )
        return ksp_its

    # ---- solve ----
    comm.Barrier()
    solve_start = time.perf_counter()
    result = newton(
        energy_fn=assembler.energy_fn,
        gradient_fn=assembler.gradient_fn,
        hessian_solve_fn=hessian_solve_fn,
        x=x,
        tolf=tolf,
        tolg=tolg,
        linesearch_tol=linesearch_tol,
        linesearch_interval=linesearch_interval,
        maxit=maxit,
        verbose=verbose,
        comm=comm,
        ghost_update_fn=None,
        hessian_matvec_fn=lambda _x, vin, vout: assembler.A.mult(vin, vout),
        save_history=True,
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
    solve_time = time.perf_counter() - solve_start

    # ---- print breakdown (rank 0) ----
    if verbose and rank == 0:
        if result.get("history"):
            _print_newton_breakdown(result["history"], solve_time)
        if assembler.iter_timings:
            _print_assembly_breakdown(assembler.iter_timings, result["nit"])

    # ---- extract timing totals from history ----
    total_grad = sum(h["t_grad"] for h in result.get("history", []))
    total_hess = sum(h["t_hess"] for h in result.get("history", []))
    total_ls = sum(h["t_ls"] for h in result.get("history", []))
    total_ksp_its = sum(_linear_iters)

    asm_cumulative = sum(d.get("total", 0.0) for d in assembler.iter_timings)
    pc_setup_cumulative = sum(_pc_setup_times)
    linear_solve_cumulative = sum(_linear_solve_times)
    ksp_cumulative = sum(d.get("ksp", 0.0) for d in assembler.iter_timings)

    # ---- cleanup ----
    x.destroy()
    assembler.cleanup()

    return {
        "mesh_level": mesh_level,
        "dofs": n_dofs,
        "nprocs": nprocs,
        "nproc_threads": nproc_threads,
        "pc_type": pc_type,
        "ksp_rtol": ksp_rtol,
        "assembly_mode": assembly_mode,
        "n_colors": assembler.n_colors,
        "setup_time": round(setup_time, 4),
        "solve_time": round(solve_time, 4),
        "total_time": round(setup_time + solve_time, 4),
        "iters": result["nit"],
        "energy": round(float(result["fun"]), 10),
        "message": result["message"],
        "grad_time": round(total_grad, 4),
        "hess_time": round(total_hess, 4),
        "ls_time": round(total_ls, 4),
        "total_ksp_its": total_ksp_its,
        "asm_time_cumulative": round(asm_cumulative, 4),
        "pc_setup_time_cumulative": round(pc_setup_cumulative, 4),
        "linear_solve_time_cumulative": round(linear_solve_cumulative, 4),
        "ksp_time_cumulative": round(ksp_cumulative, 4),
        "history": result.get("history", []),
        "assembly_details": list(assembler.iter_timings),
    }

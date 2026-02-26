#!/usr/bin/env python3
"""
p-Laplace 2D solver — DOF-partitioned parallel JAX + PETSc.

Uses DOF-based overlapping domain decomposition for all operations:
  - Energy/gradient via P2P ghost exchange (no Allgatherv)
  - Sparse Hessian assembly via local SFD + COO fast-path (no Allreduce)
  - PETSc KSP linear solves (CG + Hypre BoomerAMG or GAMG)
  - Newton iteration with golden-section line search

This replaces the replicated-data SFD approach (solve_pLaplace_jax_petsc.py)
with the DOF-partitioned approach for improved parallel scaling.

Thread control:
  - XLA multi-thread Eigen is disabled (memory-bandwidth limited workload)
  - OMP_NUM_THREADS=1 by default (critical for Hypre AMG — prevents internal
    thread oversubscription with MPI)

Usage:
  mpirun -n 4 python3 pLaplace2D_jax_petsc/solve_pLaplace_dof.py --level 9
  mpirun -n 16 python3 pLaplace2D_jax_petsc/solve_pLaplace_dof.py --level 9 --pc-type hypre
"""

import sys
import os
import time
import json
import argparse
import numpy as np
from mpi4py import MPI

# ---- Parse args before setting env vars ----
parser = argparse.ArgumentParser(
    description="p-Laplace 2D — DOF-partitioned parallel JAX + PETSc solver"
)
parser.add_argument("--level", type=int, default=9, help="Mesh level (default: 9)")
parser.add_argument("--levels", type=int, nargs="+", default=None,
                    help="Multiple mesh levels (overrides --level)")
parser.add_argument("--repeats", type=int, default=1,
                    help="Number of solve repetitions (default: 1)")
parser.add_argument("--nproc", type=int, default=1,
                    help="XLA/OMP thread count per rank (default: 1)")
parser.add_argument("--coloring-trials", type=int, default=10,
                    help="Graph coloring trials per rank (default: 10)")
parser.add_argument("--ksp-rtol", type=float, default=1e-3,
                    help="KSP relative tolerance (default: 1e-3)")
parser.add_argument("--pc-type", type=str, default="hypre",
                    choices=["gamg", "hypre"],
                    help="PETSc PC type (default: hypre)")
parser.add_argument("--quiet", action="store_true",
                    help="Suppress per-iteration Newton output")
parser.add_argument("--json", type=str, default=None,
                    help="Output JSON file (rank 0 only)")
parser.add_argument("--tolf", type=float, default=1e-5,
                    help="Energy change tolerance (default: 1e-5)")
parser.add_argument("--tolg", type=float, default=1e-3,
                    help="Gradient norm tolerance (default: 1e-3)")
parser.add_argument("--local-coloring", action="store_true",
                    help="Use local per-rank graph coloring + vmap (Variant B)")
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# ---- Set environment variables BEFORE importing JAX/PETSc ----
_threads = args.nproc
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false "
    "--xla_force_host_platform_device_count=1"
)
os.environ["OMP_NUM_THREADS"] = str(_threads)
os.environ["MKL_NUM_THREADS"] = str(_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(_threads)

from jax import config  # noqa: E402
config.update("jax_enable_x64", True)

from pLaplace2D_jax_petsc.mesh import MeshpLaplace2D  # noqa: E402
from pLaplace2D_jax_petsc.parallel_hessian_dof import ParallelDOFHessianAssembler  # noqa: E402
from pLaplace2D_jax_petsc.parallel_hessian_dof import LocalColoringAssembler  # noqa: E402
from tools_petsc4py.minimizers import newton  # noqa: E402


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
              local_coloring=False):
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

    setup_time = time.perf_counter() - setup_start

    if verbose and rank == 0:
        n_colors = assembler.n_colors
        overlap = assembler._sum_local_elems / (params["elems"].shape[0]) - 1.0
        sys.stdout.write(
            f"  DOFs={n_dofs}  elements={params['elems'].shape[0]}  "
            f"np={nprocs}  NPROC={_threads}  pc={pc_type}\n"
        )
        sys.stdout.write(
            f"  Setup: {setup_time:.3f}s  (mesh={t_mesh:.3f}s, "
            f"assembler={t_assembler:.3f}s)  "
            f"n_colors={n_colors}  overlap={overlap:.1%}\n"
        )
        sys.stdout.flush()

    # ---- initial guess as distributed PETSc Vec ----
    # u_init is in original DOF ordering → convert to reordered
    u_init_reord = np.asarray(u_init, dtype=np.float64)[assembler.part.perm]
    x = assembler.create_vec(u_init_reord)

    # ---- solve ----
    comm.Barrier()
    solve_start = time.perf_counter()
    result = newton(
        energy_fn=assembler.energy_fn,
        gradient_fn=assembler.gradient_fn,
        hessian_solve_fn=assembler.hessian_solve_fn,
        x=x,
        tolf=tolf,
        tolg=tolg,
        linesearch_tol=1e-3,
        linesearch_interval=(-0.5, 2.0),
        maxit=100,
        verbose=verbose,
        comm=comm,
        ghost_update_fn=None,
        save_history=True,
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
    total_ksp_its = sum(h["ksp_its"] for h in result.get("history", []))

    # ---- cleanup ----
    x.destroy()
    assembler.cleanup()

    return {
        "mesh_level": mesh_level,
        "dofs": n_dofs,
        "nprocs": nprocs,
        "nproc_threads": _threads,
        "pc_type": pc_type,
        "ksp_rtol": ksp_rtol,
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
        "history": result.get("history", []),
        "assembly_details": list(assembler.iter_timings),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    levels = args.levels if args.levels is not None else [args.level]

    if rank == 0:
        sys.stdout.write(
            f"p-Laplace 2D DOF-partitioned solver | "
            f"{nprocs} MPI rank(s) | NPROC={_threads} | "
            f"PC={args.pc_type}\n"
        )
        sys.stdout.write("=" * 80 + "\n")
        sys.stdout.flush()

    all_results = []

    for mesh_lvl in levels:
        level_results = []
        for rep in range(args.repeats):
            if rank == 0:
                sys.stdout.write(
                    f"\n  --- Level {mesh_lvl}"
                    f"{f', rep {rep + 1}/{args.repeats}' if args.repeats > 1 else ''}"
                    f" ---\n"
                )
                sys.stdout.flush()

            result = run_level(
                mesh_lvl, comm,
                verbose=(not args.quiet),
                coloring_trials=args.coloring_trials,
                ksp_rtol=args.ksp_rtol,
                pc_type=args.pc_type,
                tolf=args.tolf,
                tolg=args.tolg,
                local_coloring=args.local_coloring,
            )
            level_results.append(result)

            if rank == 0:
                sys.stdout.write(
                    f"  [RESULT] level={result['mesh_level']} "
                    f"dofs={result['dofs']} np={nprocs} "
                    f"solve={result['solve_time']:.3f}s "
                    f"setup={result['setup_time']:.3f}s "
                    f"iters={result['iters']} "
                    f"ksp_its_total={result['total_ksp_its']} "
                    f"J={result['energy']:.6f} "
                    f"[{result['message']}]\n"
                )
                sys.stdout.flush()

        # Use the best (min solve_time) if multiple repeats
        best = min(level_results, key=lambda r: r["solve_time"])
        all_results.append(best)

        comm.Barrier()

    # ---- Summary table (rank 0) ----
    if rank == 0:
        sys.stdout.write("\n" + "=" * 80 + "\n")
        sys.stdout.write("Summary:\n")
        sys.stdout.write(
            f"  {'level':>5s} {'dofs':>8s} {'np':>3s} {'setup':>7s} "
            f"{'solve':>7s} {'total':>7s} {'its':>4s} "
            f"{'ksp':>6s} {'energy':>14s}\n"
        )
        for r in all_results:
            sys.stdout.write(
                f"  {r['mesh_level']:5d} {r['dofs']:8d} {nprocs:3d} "
                f"{r['setup_time']:7.3f} {r['solve_time']:7.3f} "
                f"{r['total_time']:7.3f} {r['iters']:4d} "
                f"{r['total_ksp_its']:6d} {r['energy']:14.6f}\n"
            )
        sys.stdout.write("Done.\n")
        sys.stdout.flush()

        if args.json:
            metadata = {
                "solver": "jax_petsc_dof_partitioned",
                "description": (
                    f"DOF-partitioned JAX + PETSc: P2P ghost exchange, "
                    f"local SFD Hessian, CG + {args.pc_type.upper()} AMG"
                ),
                "nprocs": nprocs,
                "nproc_threads": _threads,
                "coloring_trials_per_rank": args.coloring_trials,
                "linear_solver": {
                    "ksp_type": "cg",
                    "pc_type": args.pc_type,
                    "ksp_rtol": args.ksp_rtol,
                },
                "newton_params": {
                    "tolf": args.tolf,
                    "tolg": args.tolg,
                    "linesearch_interval": [-0.5, 2.0],
                    "linesearch_tol": 1e-3,
                    "maxit": 100,
                },
            }
            output = {"metadata": metadata, "results": all_results}
            with open(args.json, "w") as fp:
                json.dump(output, fp, indent=2, default=str)
            sys.stdout.write(f"Results saved to {args.json}\n")


if __name__ == "__main__":
    main()

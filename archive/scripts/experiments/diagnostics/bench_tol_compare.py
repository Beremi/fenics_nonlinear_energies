#!/usr/bin/env python3
"""Benchmark: compare linesearch_tol=1e-3/ksp_rtol=1e-3 vs 1e-1/1e-1.

Usage:
  docker exec bench_container mpirun -np 16 python -m experiment_scripts.bench_tol_compare --level 9
"""
from tools_petsc4py.minimizers import newton
from pLaplace2D_jax_petsc.parallel_hessian_dof import LocalColoringAssembler
from pLaplace2D_jax_petsc.mesh import MeshpLaplace2D
import argparse
from mpi4py import MPI
import jax
import sys
import os
import time
import gc
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("--level", type=int, default=9)
parser.add_argument("--pc", type=str, default="gamg")
parser.add_argument("--repeats", type=int, default=3)
args, _ = parser.parse_known_args()

# ── Load mesh once ──
mesh_obj = MeshpLaplace2D(mesh_level=args.level)
params, adjacency, u_init = mesh_obj.get_data_jax()


def run_one(label, ksp_rtol, ls_tol, trial_idx):
    gc.collect()
    comm.Barrier()

    t_setup_start = time.perf_counter()
    assembler = LocalColoringAssembler(
        params=params, comm=comm, adjacency=adjacency,
        coloring_trials_per_rank=1,
        ksp_rtol=ksp_rtol, ksp_type="cg", pc_type=args.pc)
    t_setup = time.perf_counter() - t_setup_start
    t_coloring = assembler.timings.get("coloring", 0)
    n_colors = assembler.n_colors

    u_init_reord = np.asarray(u_init, dtype=np.float64)[assembler.part.perm]
    x = assembler.create_vec(u_init_reord)

    comm.Barrier()
    t_solve_start = time.perf_counter()
    result = newton(
        energy_fn=assembler.energy_fn,
        gradient_fn=assembler.gradient_fn,
        hessian_solve_fn=assembler.hessian_solve_fn,
        x=x, tolf=1e-5, tolg=1e-3, linesearch_tol=ls_tol,
        linesearch_interval=(-0.5, 2.0), maxit=100,
        verbose=False, comm=comm, ghost_update_fn=None, save_history=True)
    t_solve = time.perf_counter() - t_solve_start

    iters = result["nit"]
    energy = result.get("fun", 0)
    history = result.get("history", [])

    total_ls_evals = sum(h.get("ls_evals", 0) for h in history)
    total_ksp_its = sum(h.get("ksp_its", 0) for h in history)

    t_ksp = sum(d.get("ksp", 0) for d in assembler.iter_timings)
    t_hvp = sum(d.get("hvp_compute", 0) for d in assembler.iter_timings)
    t_ghost = sum(d.get("allgatherv", d.get("p2p_exchange", 0))
                  for d in assembler.iter_timings)

    # Line search time from newton history
    t_ls = sum(h.get("t_ls", 0) for h in history)
    t_grad = sum(h.get("t_grad", 0) for h in history)
    t_hess = sum(h.get("t_hess", 0) for h in history)

    assembler.cleanup()

    return {
        "label": label, "n_colors": n_colors,
        "setup": t_setup, "coloring": t_coloring,
        "solve": t_solve, "total": t_setup + t_solve,
        "ksp_rtol": ksp_rtol, "ls_tol": ls_tol,
        "newton_its": iters, "ksp_its": total_ksp_its,
        "ls_evals": total_ls_evals, "energy": energy,
        "t_ksp": t_ksp, "t_hvp": t_hvp, "t_ghost": t_ghost,
        "t_ls": t_ls, "t_grad": t_grad, "t_hess": t_hess,
    }


configs = [
    ("ksp=1e-3, ls=1e-3", 1e-3, 1e-3),
    ("ksp=1e-1, ls=1e-1", 1e-1, 1e-1),
]

all_results = {}
for label, ksp_rtol, ls_tol in configs:
    runs = []
    for i in range(args.repeats):
        r = run_one(label, ksp_rtol, ls_tol, i)
        runs.append(r)
        if rank == 0:
            sys.stdout.write(
                f"  [{label}] rep {i + 1}/{args.repeats}  "
                f"setup={r['setup']:.3f}  solve={r['solve']:.3f}  "
                f"total={r['total']:.3f}  newton={r['newton_its']}  "
                f"ksp={r['ksp_its']}  ls={r['ls_evals']}  J={r['energy']:.6f}\n"
            )
            sys.stdout.flush()
    all_results[label] = runs

if rank == 0:
    sys.stdout.write("\n" + "=" * 100 + "\n")
    sys.stdout.write(
        f"{'Config':<22s} {'setup':>7s} {'solve':>7s} {'total':>7s} "
        f"{'N.its':>5s} {'KSP':>5s} {'LS':>5s} {'t_ksp':>7s} "
        f"{'t_hvp':>7s} {'t_ls':>7s} {'t_grad':>7s} {'J(u)':>12s}\n"
    )
    sys.stdout.write("-" * 100 + "\n")
    for label, runs in all_results.items():
        # Pick median by total time
        runs_sorted = sorted(runs, key=lambda r: r["total"])
        r = runs_sorted[len(runs_sorted) // 2]
        sys.stdout.write(
            f"{r['label']:<22s} {r['setup']:7.3f} {r['solve']:7.3f} {r['total']:7.3f} "
            f"{r['newton_its']:5d} {r['ksp_its']:5d} {r['ls_evals']:5d} {r['t_ksp']:7.3f} "
            f"{r['t_hvp']:7.3f} {r['t_ls']:7.3f} {r['t_grad']:7.3f} {r['energy']:12.6f}\n"
        )
    sys.stdout.write("=" * 100 + "\n")

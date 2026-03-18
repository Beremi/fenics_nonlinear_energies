#!/usr/bin/env python3
"""Benchmark: local coloring without A broadcast (new) at np=16,32.

Usage:
  docker exec bench_container mpirun -np 16 python -m experiment_scripts.bench_no_bcast --level 9
"""
import scipy.sparse as sp  # noqa: F401 (kept for consistency with bench scripts)
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
t0 = time.perf_counter()
mesh_obj = MeshpLaplace2D(mesh_level=args.level)
params, adjacency, u_init = mesh_obj.get_data_jax()
t_mesh = time.perf_counter() - t0


def run_one(label, trial_idx):
    """Run solver, return dict of timings."""
    gc.collect()
    comm.Barrier()

    # Build assembler
    t_setup_start = time.perf_counter()
    assembler = LocalColoringAssembler(
        params=params, comm=comm, adjacency=adjacency,
        coloring_trials_per_rank=1,
        ksp_rtol=1e-3, ksp_type="cg", pc_type=args.pc)
    t_setup = time.perf_counter() - t_setup_start
    t_coloring = assembler.timings.get("coloring", 0)

    n_colors = assembler.n_colors

    # Solve
    u_init_reord = np.asarray(u_init, dtype=np.float64)[assembler.part.perm]
    x = assembler.create_vec(u_init_reord)

    comm.Barrier()
    t_solve_start = time.perf_counter()
    result = newton(
        energy_fn=assembler.energy_fn,
        gradient_fn=assembler.gradient_fn,
        hessian_solve_fn=assembler.hessian_solve_fn,
        x=x, tolf=1e-5, tolg=1e-3, linesearch_tol=1e-3,
        linesearch_interval=(-0.5, 2.0), maxit=100,
        verbose=False, comm=comm, ghost_update_fn=None, save_history=True)
    t_solve = time.perf_counter() - t_solve_start

    iters = result["nit"]
    energy = result.get("fun", 0)
    history = result.get("history", [])

    # Breakdown from assembler
    t_ksp2 = 0.0
    for d in assembler.iter_timings:
        t_ksp2 += d.get("ksp", 0)

    assembler.cleanup()

    return {
        "label": label, "n_colors": n_colors,
        "setup": t_setup, "coloring": t_coloring,
        "solve": t_solve, "total": t_setup + t_solve,
        "ksp": t_ksp2,
        "iters": iters, "energy": energy,
    }


# ── Run repeats ──
results = []
for i in range(args.repeats):
    r = run_one(f"igraph-no-bcast #{i + 1}", i)
    results.append(r)
    if rank == 0:
        sys.stdout.write(
            f"  [{i + 1}/{args.repeats}] colors={r['n_colors']}  "
            f"setup={r['setup']:.3f}  coloring={r['coloring']:.3f}  "
            f"solve={r['solve']:.3f}  total={r['total']:.3f}  "
            f"iters={r['iters']}  J={r['energy']:.6f}\n"
        )
        sys.stdout.flush()

if rank == 0:
    # Best of repeats (min total)
    best = min(results, key=lambda r: r["total"])
    sys.stdout.write(
        f"\n  BEST: colors={best['n_colors']}  "
        f"setup={best['setup']:.3f}  coloring={best['coloring']:.3f}  "
        f"solve={best['solve']:.3f}  total={best['total']:.3f}\n"
    )

    # Also median
    setups = sorted(r["setup"] for r in results)
    colorings = sorted(r["coloring"] for r in results)
    solves = sorted(r["solve"] for r in results)
    totals = sorted(r["total"] for r in results)
    n = len(results)
    mid = n // 2
    sys.stdout.write(
        f"  MEDIAN: setup={setups[mid]:.3f}  coloring={colorings[mid]:.3f}  "
        f"solve={solves[mid]:.3f}  total={totals[mid]:.3f}\n"
    )

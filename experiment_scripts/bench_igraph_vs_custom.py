#!/usr/bin/env python3
"""Quick benchmark: igraph vs custom-C local coloring (1/5/10 trials).
np=16, level=9.  Prints one comparison table.

Usage:
  mpirun -np 16 python -m experiment_scripts.bench_igraph_vs_custom --level 9
"""
import scipy.sparse as sp
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
args, _ = parser.parse_known_args()


# ── Load mesh once ──
t0 = time.perf_counter()
mesh_obj = MeshpLaplace2D(mesh_level=args.level)
params, adjacency, u_init = mesh_obj.get_data_jax()
t_mesh = time.perf_counter() - t0

# ── Monkey-patch _setup_coloring to allow method selection ──


def _setup_coloring_custom(self, adjacency, comm, trials_per_rank):
    """Local coloring using custom C multi-start."""
    from graph_coloring.coloring_custom import color_custom_random
    part = self.part
    lo, hi = part.lo, part.hi
    perm = part.perm
    n_free = part.n_free

    if self.rank == 0:
        adj_csr = adjacency.tocsr()
        row_adj, col_adj = adj_csr.nonzero()
        row_adj = np.ascontiguousarray(row_adj, dtype=np.int64)
        col_adj = np.ascontiguousarray(col_adj, dtype=np.int64)
        nnz = np.int64(len(row_adj))
    else:
        nnz = np.int64(0)
    nnz = int(comm.bcast(int(nnz), root=0))
    if self.rank != 0:
        row_adj = np.empty(nnz, dtype=np.int64)
        col_adj = np.empty(nnz, dtype=np.int64)
    comm.Bcast(row_adj, root=0)
    comm.Bcast(col_adj, root=0)

    self.nnz_global = nnz
    self._row_adj = row_adj
    self._col_adj = col_adj

    A_csr = sp.csr_matrix(
        (np.ones(nnz, dtype=np.float64), (row_adj, col_adj)),
        shape=(n_free, n_free))

    owned_orig = perm[lo:hi]
    slices = [A_csr.indices[A_csr.indptr[d]:A_csr.indptr[d + 1]] for d in owned_orig]
    all_nbrs = np.unique(np.concatenate(slices)) if slices else np.array([], dtype=np.int64)
    J_arr = np.union1d(owned_orig, all_nbrs).astype(np.int64)
    n_J = len(J_arr)

    J_to_idx = np.full(n_free, -1, dtype=np.int64)
    J_to_idx[J_arr] = np.arange(n_J, dtype=np.int64)

    mask = (J_to_idx[row_adj] >= 0) & (J_to_idx[col_adj] >= 0)
    A_J = sp.csr_matrix(
        (np.ones(int(mask.sum()), dtype=np.float64),
         (J_to_idx[row_adj[mask]], J_to_idx[col_adj[mask]])),
        shape=(n_J, n_J))

    A2_J = sp.csr_matrix(A_J @ A_J)

    # Multi-start custom C coloring
    best_nc = n_J + 1
    best_coloring = None
    for trial in range(trials_per_rank):
        nc, cols = color_custom_random(A2_J, seed=self.rank * 1000 + trial, is_A2=True)
        if nc < best_nc:
            best_nc = nc
            best_coloring = cols.copy()

    self.n_colors = best_nc
    self._local_coloring = best_coloring
    self._J_dofs = J_arr
    self._J_to_idx = J_to_idx
    self.coloring = None
    self.color_info = {"method": f"local_custom_c_{trials_per_rank}", "n_J": n_J,
                       "n_owned": hi - lo, "n_colors": best_nc, "trials": trials_per_rank}


def run_one(label, coloring_method, trials=1):
    """Run solver, return dict of timings."""
    gc.collect()
    comm.Barrier()

    # Patch coloring method
    if coloring_method == "igraph":
        # Use the default igraph _setup_coloring (no patching needed)
        if hasattr(LocalColoringAssembler, '_orig_setup_coloring'):
            LocalColoringAssembler._setup_coloring = LocalColoringAssembler._orig_setup_coloring
    else:
        if not hasattr(LocalColoringAssembler, '_orig_setup_coloring'):
            LocalColoringAssembler._orig_setup_coloring = LocalColoringAssembler._setup_coloring
        LocalColoringAssembler._setup_coloring = _setup_coloring_custom

    # Build assembler
    t_setup_start = time.perf_counter()
    assembler = LocalColoringAssembler(
        params=params, comm=comm, adjacency=adjacency,
        coloring_trials_per_rank=trials,
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

    # Collect per-iteration timings
    iters = result["nit"]
    history = result.get("history", [])
    t_grad = sum(h.get("t_grad", 0) for h in history)
    t_hess = sum(h.get("t_hess", 0) for h in history)
    t_ls = sum(h.get("t_ls", 0) for h in history)

    # Get breakdown from assembler iter_timings
    t_hvp_sum = 0.0
    t_allg_sum = 0.0
    t_coo_sum = 0.0
    t_ksp2 = 0.0
    for d in assembler.iter_timings:
        t_hvp_sum += d.get("hvp_compute", 0)
        t_allg_sum += d.get("allgatherv", d.get("p2p_exchange", 0))
        t_coo_sum += d.get("coo_assembly", 0)
        t_ksp2 += d.get("ksp", 0)

    energy = result.get("fun", 0)
    ksp_total = sum(h.get("ksp_its", 0) for h in history)

    assembler.cleanup()

    return {
        "label": label, "n_colors": n_colors,
        "setup": t_setup, "coloring": t_coloring,
        "solve": t_solve, "total": t_setup + t_solve,
        "grad": t_grad, "hess": t_hess, "ls": t_ls,
        "hvp": t_hvp_sum, "ghost": t_allg_sum, "coo": t_coo_sum, "ksp": t_ksp2,
        "iters": iters, "ksp_total": ksp_total, "energy": energy,
    }


# ── Run all configs ──
configs = [
    ("custom-C ×1", "custom", 1),
    ("custom-C ×5", "custom", 5),
    ("custom-C ×10", "custom", 10),
    ("igraph", "igraph", 1),
]

results = []
for label, method, trials in configs:
    r = run_one(label, method, trials)
    results.append(r)
    if rank == 0:
        sys.stdout.write(f"  done: {label}  colors={r['n_colors']}  setup={r['setup']:.3f}  solve={r['solve']:.3f}\n")
        sys.stdout.flush()

# ── Print table ──
if rank == 0:
    print(f"\n{'=' * 100}")
    print(f"  Level {args.level} | np={comm.Get_size()} | PC={args.pc}")
    print(f"{'=' * 100}")
    print(f"  {'Method':<14} {'colors':>6} {'setup':>7} {'color':>7} | "
          f"{'ghost':>6} {'HVP':>7} {'COO':>6} {'KSP':>7} | "
          f"{'grad':>6} {'hess':>7} {'LS':>7} {'solve':>7} {'total':>7}")
    print(f"  {'─' * 96}")
    for r in results:
        print(f"  {r['label']:<14} {r['n_colors']:>6} {r['setup']:>7.3f} {r['coloring']:>7.3f} | "
              f"{r['ghost']:>6.3f} {r['hvp']:>7.3f} {r['coo']:>6.3f} {r['ksp']:>7.3f} | "
              f"{r['grad']:>6.3f} {r['hess']:>7.3f} {r['ls']:>7.3f} {r['solve']:>7.3f} {r['total']:>7.3f}")
    print(f"  {'─' * 96}")
    print(
        f"  All converged to J={
            results[0]['energy']:.6f}  iters={
            results[0]['iters']}  ksp_total={
                results[0]['ksp_total']}")
    print()

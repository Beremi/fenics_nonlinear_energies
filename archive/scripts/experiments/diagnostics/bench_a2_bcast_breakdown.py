#!/usr/bin/env python3
"""
Detailed timing breakdown: A² compute vs Bcast vs coloring.

Measures each phase separately with MPI barriers:
  Phase 1: A² computation on rank 0 (others idle at barrier)
  Phase 2: Bcast of CSR arrays to all ranks
  Phase 3: Greedy coloring on each rank (independent, random seed = rank)

Also compares:
  - scipy CSR on rank 0 + Bcast
  - C direct 2-hop on rank 0 + Bcast
  - scipy CSR on ALL ranks (redundant, no Bcast)
  - C direct 2-hop on ALL ranks (redundant, no Bcast)

Usage:
    mpirun -n 16 python bench_a2_bcast_breakdown.py
"""
from graph_coloring.coloring_custom import _get_lib, _i32, _ptr
from graph_coloring.mesh_loader import PROBLEMS, load_adjacency
import ctypes
import scipy.sparse as sp
import numpy as np
from mpi4py import MPI
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Serialise .so compilation
if rank == 0:
    _get_lib()
comm.Barrier()
lib = _get_lib()


def load_A(h5_path):
    """Load adjacency as CSR on this rank."""
    A = load_adjacency(h5_path)
    A_csr = sp.csr_matrix(A)
    return A_csr


def compute_A2_scipy(A_csr):
    """Compute A² = A@A with scipy, return CSR arrays."""
    A2 = sp.csr_matrix(A_csr @ A_csr)
    n = np.int32(A2.shape[0])
    ip = np.ascontiguousarray(A2.indptr, dtype=np.int32)
    ix = np.ascontiguousarray(A2.indices, dtype=np.int32)
    return n, ip, ix


def compute_A2_c_direct(A_csr):
    """Compute A² pattern with C build_A2_pattern, return CSR arrays."""
    n = np.int32(A_csr.shape[0])
    ai = _i32(A_csr.indptr)
    aj = _i32(A_csr.indices)

    # Pass 1: get nnz
    a2_ip = np.zeros(n + 1, dtype=np.int32)
    nnz_out = ctypes.c_int(0)
    lib.build_A2_pattern(
        ctypes.c_int(n), _ptr(ai), _ptr(aj),
        _ptr(a2_ip),
        ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_int)),
        ctypes.byref(nnz_out),
    )
    nnz = nnz_out.value

    # Pass 2: fill indices
    a2_ix = np.zeros(nnz, dtype=np.int32)
    lib.build_A2_pattern(
        ctypes.c_int(n), _ptr(ai), _ptr(aj),
        _ptr(a2_ip), _ptr(a2_ix),
        ctypes.byref(nnz_out),
    )
    return n, a2_ip, a2_ix


def bcast_csr_arrays(n, ip, ix, comm):
    """Broadcast CSR arrays from rank 0."""
    rk = comm.Get_rank()
    n = comm.bcast(int(n), root=0)
    nnz = comm.bcast(len(ix) if rk == 0 else 0, root=0)
    if rk != 0:
        ip = np.empty(n + 1, dtype=np.int32)
        ix = np.empty(nnz, dtype=np.int32)
    comm.Bcast(ip, root=0)
    comm.Bcast(ix, root=0)
    return n, ip, ix


def greedy_coloring(n, ip, ix, seed):
    """Run randomised greedy coloring, return n_colors."""
    colors = np.zeros(n, dtype=np.int32)
    nc = lib.custom_greedy_color_random(
        ctypes.c_int(n), _ptr(ip), _ptr(ix),
        _ptr(colors), ctypes.c_uint(seed),
    )
    return int(nc)


# ────────────────────────────────────────────────────────────────
# Strategy A: rank 0 computes A², Bcast, each rank colors
# ────────────────────────────────────────────────────────────────
def strategy_rank0_bcast(h5_path, compute_fn, label, comm, silent=False):
    rk = comm.Get_rank()

    # Warm up: load adjacency on rank 0
    if rk == 0:
        A_csr = load_A(h5_path)
    comm.Barrier()

    # Phase 1: A² on rank 0 only
    comm.Barrier()
    t1_start = time.perf_counter()
    if rk == 0:
        n, ip, ix = compute_fn(A_csr)
    else:
        n = np.int32(0)
        ip = ix = np.empty(0, dtype=np.int32)
    comm.Barrier()
    t1_end = time.perf_counter()

    # Phase 2: Bcast
    comm.Barrier()
    t2_start = time.perf_counter()
    n, ip, ix = bcast_csr_arrays(n, ip, ix, comm)
    comm.Barrier()
    t2_end = time.perf_counter()

    # Phase 3: Greedy coloring (each rank, seed=rank)
    comm.Barrier()
    t3_start = time.perf_counter()
    nc = greedy_coloring(n, ip, ix, seed=rk)
    comm.Barrier()
    t3_end = time.perf_counter()

    # Gather colors
    all_nc = comm.gather(nc, root=0)

    dt_a2 = t1_end - t1_start
    dt_bcast = t2_end - t2_start
    dt_color = t3_end - t3_start
    dt_total = dt_a2 + dt_bcast + dt_color

    nnz = len(ix)

    if rk == 0 and not silent:
        best_nc = min(all_nc)
        print(f"  {label:<36}  {dt_a2:>7.4f}  {dt_bcast:>7.4f}  "
              f"{dt_color:>7.4f}  {dt_total:>7.4f}  {best_nc:>5}  {nnz:>12,}")

    return dt_a2, dt_bcast, dt_color


# ────────────────────────────────────────────────────────────────
# Strategy B: ALL ranks compute A² independently (no Bcast)
# ────────────────────────────────────────────────────────────────
def strategy_all_ranks(h5_path, compute_fn, label, comm, silent=False):
    rk = comm.Get_rank()

    # Warm up: load adjacency on ALL ranks
    A_csr = load_A(h5_path)
    comm.Barrier()

    # Phase 1: A² on ALL ranks
    comm.Barrier()
    t1_start = time.perf_counter()
    n, ip, ix = compute_fn(A_csr)
    comm.Barrier()
    t1_end = time.perf_counter()

    # Phase 2: no Bcast needed
    dt_bcast = 0.0

    # Phase 3: Greedy coloring (each rank, seed=rank)
    comm.Barrier()
    t3_start = time.perf_counter()
    nc = greedy_coloring(n, ip, ix, seed=rk)
    comm.Barrier()
    t3_end = time.perf_counter()

    all_nc = comm.gather(nc, root=0)

    dt_a2 = t1_end - t1_start
    dt_color = t3_end - t3_start
    dt_total = dt_a2 + dt_color

    nnz = len(ix)

    if rk == 0 and not silent:
        best_nc = min(all_nc)
        print(f"  {label:<36}  {dt_a2:>7.4f}  {'—':>7}  "
              f"{dt_color:>7.4f}  {dt_total:>7.4f}  {best_nc:>5}  {nnz:>12,}")

    return dt_a2, dt_bcast, dt_color


# ────────────────────────────────────────────────────────────────
# Run benchmarks
# ────────────────────────────────────────────────────────────────
BENCHMARKS = [
    ("pLaplace2D", 9),
    ("GinzburgLandau2D", 9),
    ("HyperElasticity3D", 4),
]

NREPS = 1  # warmup reps before timed run

for prob_name, lvl in BENCHMARKS:
    h5 = PROBLEMS[prob_name]["path"](lvl)
    if rank == 0:
        A_ref = load_adjacency(h5)
        n_ref = A_ref.shape[0]
        print(f"\n{'=' * 90}")
        print(f"  {prob_name} level {lvl}  (N={n_ref:,}, np={size})")
        print(f"{'=' * 90}")
        print(f"  {'Strategy':<36}  {'A²(s)':>7}  {'Bcast':>7}  "
              f"{'Color':>7}  {'Total':>7}  {'#col':>5}  {'nnz(A²)':>12}")
        print(f"  {'-' * 36}  {'-' * 7}  {'-' * 7}  "
              f"{'-' * 7}  {'-' * 7}  {'-' * 5}  {'-' * 12}")
        sys.stdout.flush()
    comm.Barrier()

    strategies = [
        ("scipy rank0 + Bcast", strategy_rank0_bcast, compute_A2_scipy),
        ("C direct rank0 + Bcast", strategy_rank0_bcast, compute_A2_c_direct),
        ("scipy ALL ranks", strategy_all_ranks, compute_A2_scipy),
        ("C direct ALL ranks", strategy_all_ranks, compute_A2_c_direct),
    ]

    for label, strategy_fn, compute_fn in strategies:
        # 1 warmup rep (silent), then timed run (printed)
        strategy_fn(h5, compute_fn, label, comm, silent=True)
        strategy_fn(h5, compute_fn, label, comm, silent=False)
        comm.Barrier()
        sys.stdout.flush()

    if rank == 0:
        print()

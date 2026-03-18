#!/usr/bin/env python3
"""
MPI multi-start randomised custom coloring benchmark.

Each rank receives the same A² and runs the full greedy algorithm
independently with a different random seed (= rank).  The best
(fewest-colour) result across all ranks is selected.

Two A² strategies are benchmarked:
  1. scipy: rank 0 computes A², broadcasts CSC arrays
  2. petsc: parallel distributed matmult → getRedundantMatrix

Usage:
    mpirun -n 16 python bench_custom_multistart.py
"""
from graph_coloring.mesh_loader import PROBLEMS, load_adjacency
from graph_coloring.coloring_custom import _get_lib, _i32, _ptr
import ctypes
import numpy as np
from mpi4py import MPI
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Serialise .so compilation: only rank 0 triggers build, then barrier
if rank == 0:
    _get_lib()
comm.Barrier()
lib = _get_lib()


def build_A2_scipy(h5_path, comm):
    """Compute A² on rank 0 with scipy and broadcast CSC arrays."""
    rank = comm.Get_rank()
    if rank == 0:
        import scipy.sparse as sp
        A = load_adjacency(h5_path)
        A2 = sp.csr_matrix(A @ A)  # CSR — A² is symmetric, skip CSC conversion
        n = np.int32(A2.shape[0])
        indptr = _i32(A2.indptr)
        indices = _i32(A2.indices)
        nnz = np.int32(len(indices))
    else:
        n = np.int32(0)
        nnz = np.int32(0)
        indptr = None
        indices = None

    n = comm.bcast(n, root=0)
    nnz = comm.bcast(nnz, root=0)

    if rank != 0:
        indptr = np.empty(n + 1, dtype=np.int32)
        indices = np.empty(nnz, dtype=np.int32)

    comm.Bcast(indptr, root=0)
    comm.Bcast(indices, root=0)
    return int(n), indptr, indices


def build_A2_petsc(h5_path, comm):
    """Compute A² with PETSc parallel matmult, then getRedundantMatrix."""
    from petsc4py import PETSc
    import scipy.sparse as sp

    rank = comm.Get_rank()
    size = comm.Get_size()

    # Only rank 0 loads the raw data
    if rank == 0:
        A_sp = load_adjacency(h5_path)
        A_csr = sp.csr_matrix(A_sp)
        n = A_csr.shape[0]
        ai = _i32(A_csr.indptr)
        aj = _i32(A_csr.indices)
        av = np.ones(len(aj), dtype=np.float64)
    else:
        n = 0
        ai = aj = av = None

    n = comm.bcast(n, root=0)

    # Compute ownership ranges
    local_n = n // size + (1 if rank < n % size else 0)
    rstart = rank * (n // size) + min(rank, n % size)
    rend = rstart + local_n

    # Scatter row data from rank 0
    if rank == 0:
        local_ai = ai[rstart:rend + 1].copy()
        local_aj = aj[ai[rstart]:ai[rend]].copy()
        local_av = av[ai[rstart]:ai[rend]].copy()
        local_ai -= local_ai[0]

        for r in range(1, size):
            rs = r * (n // size) + min(r, n % size)
            re = rs + n // size + (1 if r < n % size else 0)
            r_ai = ai[rs:re + 1].copy()
            r_aj = aj[ai[rs]:ai[re]].copy()
            r_av = av[ai[rs]:ai[re]].copy()
            r_ai -= r_ai[0]
            comm.send((r_ai, r_aj, r_av), dest=r, tag=0)
    else:
        local_ai, local_aj, local_av = comm.recv(source=0, tag=0)

    # Create distributed PETSc matrix
    A_pet = PETSc.Mat().createAIJ(
        size=((local_n, n), (local_n, n)),
        csr=(local_ai, local_aj, local_av),
        comm=comm,
    )
    A_pet.assemblyBegin()
    A_pet.assemblyEnd()

    # A² = A * A (parallel)
    A2_pet = A_pet.matMult(A_pet)

    # Gather full A² to every rank
    A2_red = A2_pet.getRedundantMatrix(size)

    # Extract CSR directly — A² is symmetric, no CSC conversion needed
    ai2, aj2, av2 = A2_red.getValuesCSR()
    A2_csr = sp.csr_matrix((np.ones(len(aj2)), aj2, ai2), shape=(n, n))

    indptr = _i32(A2_csr.indptr)
    indices = _i32(A2_csr.indices)

    A_pet.destroy()
    A2_pet.destroy()
    A2_red.destroy()

    return int(n), indptr, indices


# ---------------------------------------------------------------------------
BENCHMARKS = [
    ("pLaplace2D", 9),
    ("GinzburgLandau2D", 9),
    ("HyperElasticity3D", 4),
]

for prob_name, lvl in BENCHMARKS:
    h5 = PROBLEMS[prob_name]["path"](lvl)

    # --- Method 1: scipy + broadcast ---
    comm.Barrier()
    t0 = time.perf_counter()
    n, indptr, indices = build_A2_scipy(h5, comm)
    t_scipy = time.perf_counter() - t0

    comm.Barrier()
    t0 = time.perf_counter()
    colors = np.zeros(n, dtype=np.int32)
    nc = lib.custom_greedy_color_random(
        ctypes.c_int(n), _ptr(indptr), _ptr(indices),
        _ptr(colors), ctypes.c_uint(rank),
    )
    t_color_scipy = time.perf_counter() - t0

    all_nc_scipy = comm.gather(nc, root=0)
    all_tc_scipy = comm.gather(t_color_scipy, root=0)

    # --- Method 2: PETSc parallel matmult ---
    comm.Barrier()
    t0 = time.perf_counter()
    n2, indptr2, indices2 = build_A2_petsc(h5, comm)
    t_petsc = time.perf_counter() - t0

    comm.Barrier()
    t0 = time.perf_counter()
    colors2 = np.zeros(n2, dtype=np.int32)
    nc2 = lib.custom_greedy_color_random(
        ctypes.c_int(n2), _ptr(indptr2), _ptr(indices2),
        _ptr(colors2), ctypes.c_uint(rank),
    )
    t_color_petsc = time.perf_counter() - t0

    all_nc_petsc = comm.gather(nc2, root=0)
    all_tc_petsc = comm.gather(t_color_petsc, root=0)

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"  {prob_name} level {lvl}  (N={n:,}, np={size})")
        print(f"{'=' * 70}")

        print(f"\n  SciPy (rank 0 compute + Bcast):")
        print(f"    A² setup: {t_scipy:.4f} s")
        print(f"    Greedy:   min={min(all_tc_scipy):.4f}s  max={max(all_tc_scipy):.4f}s")
        print(f"    Total:    {t_scipy + max(all_tc_scipy):.4f} s")
        print(f"    Colors:   {all_nc_scipy}")
        print(f"    Best:     {min(all_nc_scipy)} (rank {int(np.argmin(all_nc_scipy))})")

        print(f"\n  PETSc (parallel matmult + getRedundantMatrix):")
        print(f"    A² setup: {t_petsc:.4f} s")
        print(f"    Greedy:   min={min(all_tc_petsc):.4f}s  max={max(all_tc_petsc):.4f}s")
        print(f"    Total:    {t_petsc + max(all_tc_petsc):.4f} s")
        print(f"    Colors:   {all_nc_petsc}")
        print(f"    Best:     {min(all_nc_petsc)} (rank {int(np.argmin(all_nc_petsc))})")

        sys.stdout.flush()

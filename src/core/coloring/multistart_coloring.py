#!/usr/bin/env python3
"""
MPI multi-start randomised graph coloring pipeline.

Usage (standalone):
    mpirun -n 16 python graph_coloring/multistart_coloring.py  path/to/mesh.h5

Usage (as library):
    from graph_coloring.multistart_coloring import multistart_color
    n_colors, colors = multistart_color(adjacency, comm, trials_per_rank=5)

Pipeline:
    1. Rank 0 loads A, computes A² = A·A with scipy (CSR), broadcasts
       the CSR indptr/indices arrays to all ranks.
    2. Each rank runs `trials_per_rank` independent randomised greedy
       colorings (C backend) with distinct PRNG seeds.
    3. Allreduce determines the global best color count; the rank that
       found it broadcasts its coloring to all ranks.

The C shared library (custom_coloring.so) must be pre-built — see Makefile.
"""

import ctypes
import os
import time
import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# C library loading (no auto-compilation — use Makefile)
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_DIR, "custom_coloring.so")
_LIB = None


def _get_lib():
    """Load the pre-built C shared library."""
    global _LIB
    if _LIB is not None:
        return _LIB

    if not os.path.exists(_SO):
        # Fallback: try to compile on the fly
        _SRC = os.path.join(_DIR, "custom_coloring.c")
        if os.path.exists(_SRC):
            import subprocess
            try:
                subprocess.check_call(
                    ["gcc", "-O3", "-march=native", "-shared", "-fPIC",
                     "-o", _SO, _SRC],
                    cwd=_DIR,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError(
                    f"custom_coloring.so not found and compilation failed.\n"
                    f"Run 'make' in {_DIR} or the project root."
                )
        else:
            raise RuntimeError(
                f"custom_coloring.so not found at {_SO}.\n"
                f"Run 'make' in the project root."
            )

    _LIB = ctypes.CDLL(_SO)

    # custom_greedy_color_random(n, indptr, indices, colors, seed) -> n_colors
    _LIB.custom_greedy_color_random.restype = ctypes.c_int
    _LIB.custom_greedy_color_random.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint,
    ]

    return _LIB


def _i32(arr):
    return np.ascontiguousarray(arr, dtype=np.int32)


def _ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_A2_scipy(adjacency):
    """Compute A² = A·A as CSR arrays (indptr, indices) on this process."""
    A2 = sp.csr_matrix(adjacency @ adjacency)
    n = np.int32(A2.shape[0])
    ip = _i32(A2.indptr)
    ix = _i32(A2.indices)
    return n, ip, ix


def bcast_csr(n, indptr, indices, comm):
    """Broadcast CSR arrays from rank 0 to all ranks."""
    rk = comm.Get_rank()
    n = comm.bcast(int(n) if rk == 0 else 0, root=0)
    nnz = comm.bcast(len(indices) if rk == 0 else 0, root=0)
    if rk != 0:
        indptr = np.empty(n + 1, dtype=np.int32)
        indices = np.empty(nnz, dtype=np.int32)
    comm.Bcast(indptr, root=0)
    comm.Bcast(indices, root=0)
    return int(n), indptr, indices


def greedy_color_random(n, indptr, indices, seed):
    """
    Run one randomised greedy coloring (C backend).

    Returns (n_colors, colors_array).
    """
    lib = _get_lib()
    colors = np.zeros(n, dtype=np.int32)
    nc = lib.custom_greedy_color_random(
        ctypes.c_int(n), _ptr(indptr), _ptr(indices),
        _ptr(colors), ctypes.c_uint(seed),
    )
    return int(nc), colors


def multistart_color(adjacency, comm, trials_per_rank=1):
    """
    MPI multi-start randomised graph coloring.

    Parameters
    ----------
    adjacency : scipy.sparse matrix
        Element-DOF adjacency matrix A (available on rank 0).
        Other ranks may pass None.
    comm : MPI communicator
    trials_per_rank : int
        Number of independent random colorings each rank runs.

    Returns
    -------
    n_colors : int
        Best (minimum) color count across all ranks × trials.
    colors : ndarray of int32
        The coloring that achieved n_colors (broadcast to all ranks).
    info : dict
        Timing breakdown: 'a2_time', 'bcast_time', 'color_time',
        'total_time', 'colors_per_trial' (list on this rank).
    """
    rk = comm.Get_rank()
    sz = comm.Get_size()

    # Ensure the C library is loaded (rank 0 first to handle compilation)
    if rk == 0:
        _get_lib()
    comm.Barrier()
    _get_lib()

    # Phase 1: A² on rank 0
    comm.Barrier()
    t0 = time.perf_counter()
    if rk == 0:
        n, ip, ix = compute_A2_scipy(adjacency)
    else:
        n = np.int32(0)
        ip = ix = np.empty(0, dtype=np.int32)
    comm.Barrier()
    t_a2 = time.perf_counter() - t0

    # Phase 2: Broadcast
    comm.Barrier()
    t0 = time.perf_counter()
    n, ip, ix = bcast_csr(n, ip, ix, comm)
    comm.Barrier()
    t_bcast = time.perf_counter() - t0

    # Phase 3: Multi-trial coloring
    comm.Barrier()
    t0 = time.perf_counter()
    best_nc = n + 1  # impossibly large
    best_colors = None
    colors_per_trial = []

    for trial in range(trials_per_rank):
        seed = rk * trials_per_rank + trial
        nc, cols = greedy_color_random(n, ip, ix, seed)
        colors_per_trial.append(nc)
        if nc < best_nc:
            best_nc = nc
            best_colors = cols

    comm.Barrier()
    t_color = time.perf_counter() - t0

    # Phase 4: Global best
    from mpi4py import MPI

    local_best = np.array([best_nc, rk], dtype=np.int32)
    global_best = np.empty(2, dtype=np.int32)
    # Custom: find min color count, break ties by lowest rank
    all_bests = comm.allgather((best_nc, rk))
    global_nc, winner_rank = min(all_bests, key=lambda x: (x[0], x[1]))

    # Winner broadcasts its coloring
    if rk == winner_rank:
        result_colors = best_colors.copy()
    else:
        result_colors = np.empty(n, dtype=np.int32)
    comm.Bcast(result_colors, root=winner_rank)

    t_total = t_a2 + t_bcast + t_color

    info = {
        'a2_time': t_a2,
        'bcast_time': t_bcast,
        'color_time': t_color,
        'total_time': t_total,
        'colors_per_trial': colors_per_trial,
        'best_local_nc': best_nc,
        'winner_rank': winner_rank,
    }

    return int(global_nc), result_colors, info


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) < 2:
        if rank == 0:
            print(f"Usage: mpirun -n N python {sys.argv[0]} mesh.h5 [trials_per_rank]")
        sys.exit(1)

    h5_path = sys.argv[1]
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    # Load adjacency on rank 0
    if rank == 0:
        from graph_coloring.mesh_loader import load_adjacency
        A = load_adjacency(h5_path)
        print(f"Loaded: N={A.shape[0]:,}, nnz={A.nnz:,}, np={size}, "
              f"trials_per_rank={trials}")
    else:
        A = None

    nc, colors, info = multistart_color(A, comm, trials_per_rank=trials)

    if rank == 0:
        print(f"\nResult: {nc} colors")
        print(f"Timing: A²={info['a2_time']:.4f}s, "
              f"Bcast={info['bcast_time']:.4f}s, "
              f"Color={info['color_time']:.4f}s, "
              f"Total={info['total_time']:.4f}s")
        print(f"Winner: rank {info['winner_rank']}")

    # Gather all trial results
    all_trials = comm.gather(info['colors_per_trial'], root=0)
    if rank == 0:
        print(f"\nColors per rank (best of {trials} trials):")
        for r, trials_list in enumerate(all_trials):
            print(f"  rank {r:2d}: {trials_list}  best={min(trials_list)}")

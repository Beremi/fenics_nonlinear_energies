#!/usr/bin/env python3
"""
Benchmark alternative A² sparsity-pattern computation methods.

Instead of sparse matrix multiplication A@A, compute the distance-2 adjacency
directly using graph libraries or custom C code.

Usage:
    python bench_a2_graph_methods.py
    mpirun -n 16 python bench_a2_graph_methods.py
"""
from graph_coloring.mesh_loader import PROBLEMS, load_adjacency
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


# ────────────────────────────────────────────────────────────────
# Helper: broadcast CSR arrays from rank 0
# ────────────────────────────────────────────────────────────────
def bcast_csr(n, indptr, indices, comm):
    """Broadcast pre-built CSR arrays from rank 0 to all ranks."""
    rk = comm.Get_rank()
    n = comm.bcast(n, root=0)
    nnz = comm.bcast(len(indices) if rk == 0 else 0, root=0)
    if rk != 0:
        indptr = np.empty(n + 1, dtype=np.int32)
        indices = np.empty(nnz, dtype=np.int32)
    comm.Bcast(indptr, root=0)
    comm.Bcast(indices, root=0)
    return int(n), indptr, indices


# ────────────────────────────────────────────────────────────────
# Method 1: scipy CSR (baseline)
# ────────────────────────────────────────────────────────────────
def method_scipy_csr(h5_path, comm):
    rk = comm.Get_rank()
    if rk == 0:
        A = load_adjacency(h5_path)
        A2 = sp.csr_matrix(A @ A)
        n = np.int32(A2.shape[0])
        ip = np.ascontiguousarray(A2.indptr, dtype=np.int32)
        ix = np.ascontiguousarray(A2.indices, dtype=np.int32)
    else:
        n = np.int32(0)
        ip = ix = np.empty(0, dtype=np.int32)
    return bcast_csr(n, ip, ix, comm)


# ────────────────────────────────────────────────────────────────
# Method 2: igraph neighborhood(order=2) → CSR
# ────────────────────────────────────────────────────────────────
def method_igraph_neighborhood(h5_path, comm):
    import igraph
    rk = comm.Get_rank()
    if rk == 0:
        A = load_adjacency(h5_path)
        A_csr = sp.csr_matrix(A)
        n = A_csr.shape[0]

        # Build igraph from adjacency
        t_ig0 = time.perf_counter()
        rows, cols = A_csr.nonzero()
        # Remove self-loops and keep upper triangle for undirected
        mask = rows < cols
        edges = list(zip(rows[mask].tolist(), cols[mask].tolist()))
        g = igraph.Graph(n=n, edges=edges, directed=False)
        t_ig1 = time.perf_counter()

        # Get 2-hop neighborhoods
        t_nb0 = time.perf_counter()
        nbs = g.neighborhood(order=2)
        t_nb1 = time.perf_counter()

        # Convert to CSR
        t_csr0 = time.perf_counter()
        # Build indptr and indices
        ip = np.zeros(n + 1, dtype=np.int32)
        for i in range(n):
            ip[i + 1] = ip[i] + len(nbs[i])
        ix = np.empty(ip[n], dtype=np.int32)
        for i in range(n):
            # neighborhood includes self (distance 0), sort for consistency
            ix[ip[i]:ip[i + 1]] = np.sort(nbs[i])
        t_csr1 = time.perf_counter()

        print(f"    [igraph] graph build: {t_ig1 - t_ig0:.4f}s, "
              f"neighborhood: {t_nb1 - t_nb0:.4f}s, "
              f"CSR build: {t_csr1 - t_csr0:.4f}s")

        n = np.int32(n)
    else:
        n = np.int32(0)
        ip = ix = np.empty(0, dtype=np.int32)
    return bcast_csr(n, ip, ix, comm)


# ────────────────────────────────────────────────────────────────
# Method 3: igraph neighborhood (faster CSR via numpy)
# ────────────────────────────────────────────────────────────────
def method_igraph_neighborhood_fast(h5_path, comm):
    import igraph
    rk = comm.Get_rank()
    if rk == 0:
        A = load_adjacency(h5_path)
        A_csr = sp.csr_matrix(A)
        n = A_csr.shape[0]

        # Build igraph - try using adjacency matrix directly
        t_ig0 = time.perf_counter()
        # Use scipy sparse matrix directly
        A_upper = sp.triu(A_csr, k=1)
        sources, targets = A_upper.nonzero()
        g = igraph.Graph(n=n, edges=list(zip(sources.tolist(), targets.tolist())), directed=False)
        t_ig1 = time.perf_counter()

        # Get 2-hop neighborhoods
        t_nb0 = time.perf_counter()
        nbs = g.neighborhood(order=2)
        t_nb1 = time.perf_counter()

        # Convert to CSR using concatenation
        t_csr0 = time.perf_counter()
        lengths = np.array([len(nb) for nb in nbs], dtype=np.int32)
        ip = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(lengths, out=ip[1:])
        ix = np.concatenate([np.array(nb, dtype=np.int32) for nb in nbs])
        # Sort each row
        for i in range(n):
            ix[ip[i]:ip[i + 1]].sort()
        t_csr1 = time.perf_counter()

        print(f"    [igraph fast] graph build: {t_ig1 - t_ig0:.4f}s, "
              f"neighborhood: {t_nb1 - t_nb0:.4f}s, "
              f"CSR build: {t_csr1 - t_csr0:.4f}s")

        n = np.int32(n)
    else:
        n = np.int32(0)
        ip = ix = np.empty(0, dtype=np.int32)
    return bcast_csr(n, ip, ix, comm)


# ────────────────────────────────────────────────────────────────
# Method 4: Direct C implementation of 2-hop adjacency
# For each vertex i, collect union of neighbors of neighbors
# ────────────────────────────────────────────────────────────────
def method_c_direct(h5_path, comm):
    """Compute A² pattern directly in C: for each row, collect 2-hop neighbors."""
    import ctypes
    rk = comm.Get_rank()
    if rk == 0:
        A = load_adjacency(h5_path)
        A_csr = sp.csr_matrix(A)
        n = np.int32(A_csr.shape[0])

        ai = np.ascontiguousarray(A_csr.indptr, dtype=np.int32)
        aj = np.ascontiguousarray(A_csr.indices, dtype=np.int32)

        # Load C library
        from graph_coloring.coloring_custom import _get_lib
        lib = _get_lib()

        # Check if the function exists, otherwise fall back
        try:
            func = lib.build_A2_pattern
        except AttributeError:
            print("    [C direct] function not available in .so")
            return None

        # First call to get nnz
        nnz_out = ctypes.c_int(0)
        func.restype = ctypes.c_int
        func.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]

        # Two-pass: first get sizes, then fill
        a2_ip = np.zeros(n + 1, dtype=np.int32)
        # Pass 1: get row lengths
        ret = func(
            ctypes.c_int(n),
            ai.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            aj.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            a2_ip.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_int)),  # NULL
            ctypes.byref(nnz_out),
        )
        nnz = nnz_out.value
        a2_ix = np.zeros(nnz, dtype=np.int32)
        # Pass 2: fill indices
        ret = func(
            ctypes.c_int(n),
            ai.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            aj.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            a2_ip.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            a2_ix.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.byref(nnz_out),
        )
        ip = a2_ip
        ix = a2_ix
    else:
        n = np.int32(0)
        ip = ix = np.empty(0, dtype=np.int32)
    return bcast_csr(n, ip, ix, comm)


# ────────────────────────────────────────────────────────────────
# Method 5: scipy symbolic (just square the boolean pattern)
# Using scipy's internal _cs_matrix._mul_sparse_matrix
# ────────────────────────────────────────────────────────────────
def method_scipy_bool_csr(h5_path, comm):
    rk = comm.Get_rank()
    if rk == 0:
        A = load_adjacency(h5_path)
        A_bool = sp.csr_matrix(A, dtype=bool)
        A2 = sp.csr_matrix(A_bool @ A_bool)
        n = np.int32(A2.shape[0])
        ip = np.ascontiguousarray(A2.indptr, dtype=np.int32)
        ix = np.ascontiguousarray(A2.indices, dtype=np.int32)
    else:
        n = np.int32(0)
        ip = ix = np.empty(0, dtype=np.int32)
    return bcast_csr(n, ip, ix, comm)


# ────────────────────────────────────────────────────────────────
# Method 6: igraph neighborhood_size + neighborhood
#   (use neighborhood_size first to preallocate, then fill)
# ────────────────────────────────────────────────────────────────
def method_igraph_from_csr(h5_path, comm):
    """Build igraph directly from CSR data, use neighborhood."""
    import igraph
    rk = comm.Get_rank()
    if rk == 0:
        A = load_adjacency(h5_path)
        A_csr = sp.csr_matrix(A)
        n = A_csr.shape[0]

        # Build igraph from edge list (CSR -> edge list)
        t0 = time.perf_counter()
        ai = A_csr.indptr
        aj = A_csr.indices
        # Only upper triangle for undirected graph
        edges = []
        for i in range(n):
            for jj in range(ai[i], ai[i + 1]):
                j = aj[jj]
                if j > i:
                    edges.append((i, j))
        g = igraph.Graph(n=n, edges=edges, directed=False)
        t1 = time.perf_counter()

        # 2-hop neighborhoods
        t2 = time.perf_counter()
        nbs = g.neighborhood(order=2)
        t3 = time.perf_counter()

        # Build CSR
        t4 = time.perf_counter()
        lengths = np.array([len(nb) for nb in nbs], dtype=np.int32)
        ip = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(lengths, out=ip[1:])
        ix = np.concatenate([np.array(nb, dtype=np.int32) for nb in nbs])
        for i in range(n):
            ix[ip[i]:ip[i + 1]].sort()
        t5 = time.perf_counter()

        print(f"    [igraph from CSR] graph build: {t1 - t0:.4f}s, "
              f"neighborhood: {t3 - t2:.4f}s, CSR: {t5 - t4:.4f}s")

        n = np.int32(n)
    else:
        n = np.int32(0)
        ip = ix = np.empty(0, dtype=np.int32)
    return bcast_csr(n, ip, ix, comm)


# ────────────────────────────────────────────────────────────────
# Method 7: igraph using Graph.Adjacency (sparse)
# ────────────────────────────────────────────────────────────────
def method_igraph_adjacency(h5_path, comm):
    """Build igraph using its Adjacency constructor with sparse matrix."""
    import igraph
    rk = comm.Get_rank()
    if rk == 0:
        A = load_adjacency(h5_path)
        n = A.shape[0]

        # Build igraph from sparse adjacency
        t0 = time.perf_counter()
        try:
            g = igraph.Graph.Adjacency(A, mode="undirected")
        except TypeError:
            # Try with dense
            g = igraph.Graph.Weighted_Adjacency(A.toarray(), mode="undirected")
        t1 = time.perf_counter()

        # 2-hop neighborhoods
        t2 = time.perf_counter()
        nbs = g.neighborhood(order=2)
        t3 = time.perf_counter()

        # Build CSR
        t4 = time.perf_counter()
        lengths = np.array([len(nb) for nb in nbs], dtype=np.int32)
        ip = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(lengths, out=ip[1:])
        ix = np.concatenate([np.array(nb, dtype=np.int32) for nb in nbs])
        for i in range(n):
            ix[ip[i]:ip[i + 1]].sort()
        t5 = time.perf_counter()

        print(f"    [igraph Adjacency] graph build: {t0:.4f}->{t1:.4f} ({t1 - t0:.4f}s), "
              f"neighborhood: {t3 - t2:.4f}s, CSR: {t5 - t4:.4f}s")

        n = np.int32(n)
    else:
        n = np.int32(0)
        ip = ix = np.empty(0, dtype=np.int32)
    return bcast_csr(n, ip, ix, comm)


# ────────────────────────────────────────────────────────────────
# Run all methods
# ────────────────────────────────────────────────────────────────
METHODS = [
    ("scipy CSR + Bcast", method_scipy_csr),
    ("scipy bool CSR + Bcast", method_scipy_bool_csr),
    ("igraph neighborhood + Bcast", method_igraph_neighborhood),
    ("igraph neighborhood fast + Bcast", method_igraph_neighborhood_fast),
    ("C direct 2-hop + Bcast", method_c_direct),
]

BENCHMARKS = [
    ("pLaplace2D", 9),
    ("GinzburgLandau2D", 9),
    ("HyperElasticity3D", 4),
]

for prob_name, lvl in BENCHMARKS:
    h5 = PROBLEMS[prob_name]["path"](lvl)
    if rank == 0:
        A_ref = load_adjacency(h5)
        n_ref = A_ref.shape[0]
        print(f"\n{'=' * 72}")
        print(f"  {prob_name} level {lvl}  (N={n_ref:,}, np={size})")
        print(f"{'=' * 72}")
        print(f"  {'Method':<38}  {'Time (s)':>10}  {'nnz(A²)':>12}")
        print(f"  {'-' * 38}  {'-' * 10}  {'-' * 12}")

    for mname, mfunc in METHODS:
        comm.Barrier()
        try:
            t0 = time.perf_counter()
            result = mfunc(h5, comm)
            dt = time.perf_counter() - t0
            if result is None:
                if rank == 0:
                    print(f"  {mname:<38}  {'N/A':>10}  {'—':>12}")
                continue
            n, ip, ix = result
            if rank == 0:
                nnz = len(ix)
                print(f"  {mname:<38}  {dt:>10.4f}  {nnz:>12,}")
        except Exception as e:
            if rank == 0:
                import traceback
                print(f"  {mname:<38}  {'ERROR':>10}  {str(e)[:40]}")
                traceback.print_exc()
        sys.stdout.flush()
        comm.Barrier()

if rank == 0:
    print()

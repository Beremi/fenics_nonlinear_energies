#!/usr/bin/env python3
"""
Investigate PETSc A² sparsity-pattern computation approaches.

For graph coloring we only need the *sparsity pattern* of A², not numerical
values.  This script tests various strategies to see what's fastest.

Usage (serial):
    python bench_a2_investigation.py

Usage (parallel):
    mpirun -n 16 python bench_a2_investigation.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpi4py import MPI
import numpy as np
import scipy.sparse as sp
from graph_coloring.mesh_loader import PROBLEMS, load_adjacency

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ────────────────────────────────────────────────────────────────
# Helper: load adjacency on rank 0 and share raw CSR data
# ────────────────────────────────────────────────────────────────
def load_and_share(h5_path, comm):
    """Return (n, ai, aj) on every rank (int32 CSR arrays)."""
    rk = comm.Get_rank()
    if rk == 0:
        A = load_adjacency(h5_path)
        A = sp.csr_matrix(A)
        n = np.int32(A.shape[0])
        ai = np.ascontiguousarray(A.indptr, dtype=np.int32)
        aj = np.ascontiguousarray(A.indices, dtype=np.int32)
        nnz = np.int32(len(aj))
    else:
        n = np.int32(0); nnz = np.int32(0)
        ai = aj = None
    n = comm.bcast(n, root=0)
    nnz = comm.bcast(nnz, root=0)
    if rk != 0:
        ai = np.empty(n + 1, dtype=np.int32)
        aj = np.empty(nnz, dtype=np.int32)
    comm.Bcast(ai, root=0)
    comm.Bcast(aj, root=0)
    return int(n), ai, aj

# ────────────────────────────────────────────────────────────────
# Method 1: scipy serial on rank 0 + Bcast
# ────────────────────────────────────────────────────────────────
def method_scipy_bcast(h5_path, comm):
    rk = comm.Get_rank()
    if rk == 0:
        A = load_adjacency(h5_path)
        A2 = sp.csc_matrix(A @ A)
        n = np.int32(A2.shape[0])
        ip = np.ascontiguousarray(A2.indptr, dtype=np.int32)
        ix = np.ascontiguousarray(A2.indices, dtype=np.int32)
        nnz = np.int32(len(ix))
    else:
        n = np.int32(0); nnz = np.int32(0)
        ip = ix = None
    n = comm.bcast(n, root=0)
    nnz = comm.bcast(nnz, root=0)
    if rk != 0:
        ip = np.empty(n + 1, dtype=np.int32)
        ix = np.empty(nnz, dtype=np.int32)
    comm.Bcast(ip, root=0)
    comm.Bcast(ix, root=0)
    return int(n), ip, ix

# ────────────────────────────────────────────────────────────────
# Method 2: PETSc matMult (current approach from bench_custom_multistart)
# ────────────────────────────────────────────────────────────────
def method_petsc_matmult(h5_path, comm):
    from petsc4py import PETSc
    rk = comm.Get_rank(); sz = comm.Get_size()
    n, ai, aj = load_and_share(h5_path, comm)
    av = np.ones(len(aj), dtype=np.float64)

    # build distributed matrix
    local_n = n // sz + (1 if rk < n % sz else 0)
    rstart = rk * (n // sz) + min(rk, n % sz)
    rend = rstart + local_n
    lai = ai[rstart:rend+1].copy(); lai -= lai[0]
    laj = aj[ai[rstart]:ai[rend]].copy()
    lav = av[ai[rstart]:ai[rend]].copy()

    A_pet = PETSc.Mat().createAIJ(
        size=((local_n, n), (local_n, n)),
        csr=(lai, laj, lav), comm=comm,
    )
    A_pet.assemblyBegin(); A_pet.assemblyEnd()

    A2 = A_pet.matMult(A_pet)
    A2_red = A2.getRedundantMatrix(sz)
    ai2, aj2, _ = A2_red.getValuesCSR()
    A2_csc = sp.csc_matrix((np.ones(len(aj2)), aj2, ai2), shape=(n, n))
    ip = np.ascontiguousarray(A2_csc.indptr, dtype=np.int32)
    ix = np.ascontiguousarray(A2_csc.indices, dtype=np.int32)

    A_pet.destroy(); A2.destroy(); A2_red.destroy()
    return n, ip, ix

# ────────────────────────────────────────────────────────────────
# Method 3: PETSc matMult with fill hint
# ────────────────────────────────────────────────────────────────
def method_petsc_matmult_fill(h5_path, comm):
    from petsc4py import PETSc
    rk = comm.Get_rank(); sz = comm.Get_size()
    n, ai, aj = load_and_share(h5_path, comm)
    av = np.ones(len(aj), dtype=np.float64)

    local_n = n // sz + (1 if rk < n % sz else 0)
    rstart = rk * (n // sz) + min(rk, n % sz)
    rend = rstart + local_n
    lai = ai[rstart:rend+1].copy(); lai -= lai[0]
    laj = aj[ai[rstart]:ai[rend]].copy()
    lav = av[ai[rstart]:ai[rend]].copy()

    A_pet = PETSc.Mat().createAIJ(
        size=((local_n, n), (local_n, n)),
        csr=(lai, laj, lav), comm=comm,
    )
    A_pet.assemblyBegin(); A_pet.assemblyEnd()

    # Use fill parameter to give better estimate
    A2 = A_pet.matMult(A_pet, fill=2.0)
    A2_red = A2.getRedundantMatrix(sz)
    ai2, aj2, _ = A2_red.getValuesCSR()
    A2_csc = sp.csc_matrix((np.ones(len(aj2)), aj2, ai2), shape=(n, n))
    ip = np.ascontiguousarray(A2_csc.indptr, dtype=np.int32)
    ix = np.ascontiguousarray(A2_csc.indices, dtype=np.int32)

    A_pet.destroy(); A2.destroy(); A2_red.destroy()
    return n, ip, ix

# ────────────────────────────────────────────────────────────────
# Method 4: PETSc matTransposeMatMult (A^T * A == A * A for symmetric A)
# ────────────────────────────────────────────────────────────────
def method_petsc_AtA(h5_path, comm):
    from petsc4py import PETSc
    rk = comm.Get_rank(); sz = comm.Get_size()
    n, ai, aj = load_and_share(h5_path, comm)
    av = np.ones(len(aj), dtype=np.float64)

    local_n = n // sz + (1 if rk < n % sz else 0)
    rstart = rk * (n // sz) + min(rk, n % sz)
    rend = rstart + local_n
    lai = ai[rstart:rend+1].copy(); lai -= lai[0]
    laj = aj[ai[rstart]:ai[rend]].copy()
    lav = av[ai[rstart]:ai[rend]].copy()

    A_pet = PETSc.Mat().createAIJ(
        size=((local_n, n), (local_n, n)),
        csr=(lai, laj, lav), comm=comm,
    )
    A_pet.assemblyBegin(); A_pet.assemblyEnd()

    try:
        A2 = A_pet.transposeMatMult(A_pet)
        A2_red = A2.getRedundantMatrix(sz)
        ai2, aj2, _ = A2_red.getValuesCSR()
        A2_csc = sp.csc_matrix((np.ones(len(aj2)), aj2, ai2), shape=(n, n))
        ip = np.ascontiguousarray(A2_csc.indptr, dtype=np.int32)
        ix = np.ascontiguousarray(A2_csc.indices, dtype=np.int32)
        A2.destroy(); A2_red.destroy()
    except Exception as e:
        if rk == 0: print(f"    AtA failed: {e}")
        A_pet.destroy()
        return None
    A_pet.destroy()
    return n, ip, ix

# ────────────────────────────────────────────────────────────────
# Method 5: PETSc matMult with MPIAIJ preallocated
# Better preallocation by pre-estimating nnz per row
# ────────────────────────────────────────────────────────────────
def method_petsc_prealloc(h5_path, comm):
    from petsc4py import PETSc
    rk = comm.Get_rank(); sz = comm.Get_size()
    n, ai, aj = load_and_share(h5_path, comm)
    av = np.ones(len(aj), dtype=np.float64)

    local_n = n // sz + (1 if rk < n % sz else 0)
    rstart = rk * (n // sz) + min(rk, n % sz)
    rend = rstart + local_n
    lai = ai[rstart:rend+1].copy(); lai -= lai[0]
    laj = aj[ai[rstart]:ai[rend]].copy()
    lav = av[ai[rstart]:ai[rend]].copy()

    # Compute nnz per row for preallocation
    nnz_per_row = np.diff(lai)
    d_nnz = np.zeros(local_n, dtype=np.int32)
    o_nnz = np.zeros(local_n, dtype=np.int32)
    for i in range(local_n):
        cols = laj[lai[i]:lai[i+1]]
        d_nnz[i] = np.sum((cols >= rstart) & (cols < rend))
        o_nnz[i] = len(cols) - d_nnz[i]

    A_pet = PETSc.Mat().createAIJ(
        size=((local_n, n), (local_n, n)),
        nnz=(d_nnz, o_nnz), comm=comm,
    )
    # Set values row by row
    for i in range(local_n):
        row = rstart + i
        cols = laj[lai[i]:lai[i+1]].astype(np.int32)
        vals = lav[lai[i]:lai[i+1]]
        A_pet.setValues(row, cols, vals)
    A_pet.assemblyBegin(); A_pet.assemblyEnd()

    A2 = A_pet.matMult(A_pet)
    A2_red = A2.getRedundantMatrix(sz)
    ai2, aj2, _ = A2_red.getValuesCSR()
    A2_csc = sp.csc_matrix((np.ones(len(aj2)), aj2, ai2), shape=(n, n))
    ip = np.ascontiguousarray(A2_csc.indptr, dtype=np.int32)
    ix = np.ascontiguousarray(A2_csc.indices, dtype=np.int32)

    A_pet.destroy(); A2.destroy(); A2_red.destroy()
    return n, ip, ix

# ────────────────────────────────────────────────────────────────
# Method 6: scipy on ALL ranks (no Bcast needed, but redundant work)
# ────────────────────────────────────────────────────────────────
def method_scipy_all_ranks(h5_path, comm):
    n, ai, aj = load_and_share(h5_path, comm)
    A = sp.csr_matrix((np.ones(len(aj), dtype=np.float64), aj, ai), shape=(n, n))
    A2 = sp.csc_matrix(A @ A)
    ip = np.ascontiguousarray(A2.indptr, dtype=np.int32)
    ix = np.ascontiguousarray(A2.indices, dtype=np.int32)
    return n, ip, ix

# ────────────────────────────────────────────────────────────────
# Method 7: scipy bool (avoid numerical multiply, use bool pattern)
# ────────────────────────────────────────────────────────────────
def method_scipy_bool(h5_path, comm):
    rk = comm.Get_rank()
    if rk == 0:
        A = load_adjacency(h5_path)
        # Use bool to get just pattern  
        A_bool = A.astype(bool)
        A2 = sp.csc_matrix(A_bool @ A_bool)
        n = np.int32(A2.shape[0])
        ip = np.ascontiguousarray(A2.indptr, dtype=np.int32)
        ix = np.ascontiguousarray(A2.indices, dtype=np.int32)
        nnz = np.int32(len(ix))
    else:
        n = np.int32(0); nnz = np.int32(0)
        ip = ix = None
    n = comm.bcast(n, root=0)
    nnz = comm.bcast(nnz, root=0)
    if rk != 0:
        ip = np.empty(n + 1, dtype=np.int32)
        ix = np.empty(nnz, dtype=np.int32)
    comm.Bcast(ip, root=0)
    comm.Bcast(ix, root=0)
    return int(n), ip, ix

# ────────────────────────────────────────────────────────────────
# Method 8: scipy int8 (minimal integer type)
# ────────────────────────────────────────────────────────────────
def method_scipy_int8(h5_path, comm):
    rk = comm.Get_rank()
    if rk == 0:
        A = load_adjacency(h5_path)
        A8 = A.astype(np.int8)
        A2 = sp.csc_matrix(A8 @ A8)
        n = np.int32(A2.shape[0])
        ip = np.ascontiguousarray(A2.indptr, dtype=np.int32)
        ix = np.ascontiguousarray(A2.indices, dtype=np.int32)
        nnz = np.int32(len(ix))
    else:
        n = np.int32(0); nnz = np.int32(0)
        ip = ix = None
    n = comm.bcast(n, root=0)
    nnz = comm.bcast(nnz, root=0)
    if rk != 0:
        ip = np.empty(n + 1, dtype=np.int32)
        ix = np.empty(nnz, dtype=np.int32)
    comm.Bcast(ip, root=0)
    comm.Bcast(ix, root=0)
    return int(n), ip, ix

# ────────────────────────────────────────────────────────────────
# Method 9: PETSc matmult serial on rank 0 only + Bcast
#   (avoid distributed overhead, but use PETSc's serial matmult)
# ────────────────────────────────────────────────────────────────
def method_petsc_serial_bcast(h5_path, comm):
    from petsc4py import PETSc
    rk = comm.Get_rank()
    if rk == 0:
        A_sp = load_adjacency(h5_path)
        A_csr = sp.csr_matrix(A_sp)
        n = A_csr.shape[0]
        ai = np.ascontiguousarray(A_csr.indptr, dtype=np.int32)
        aj = np.ascontiguousarray(A_csr.indices, dtype=np.int32)
        av = np.ones(len(aj), dtype=np.float64)

        A_pet = PETSc.Mat().createAIJ(
            size=(n, n), csr=(ai, aj, av), comm=PETSc.COMM_SELF,
        )
        A_pet.assemblyBegin(); A_pet.assemblyEnd()

        A2_pet = A_pet.matMult(A_pet)
        ai2, aj2, _ = A2_pet.getValuesCSR()
        A2_csc = sp.csc_matrix((np.ones(len(aj2)), aj2, ai2), shape=(n, n))
        ip = np.ascontiguousarray(A2_csc.indptr, dtype=np.int32)
        ix = np.ascontiguousarray(A2_csc.indices, dtype=np.int32)
        nnz = np.int32(len(ix))

        A_pet.destroy(); A2_pet.destroy()
        n = np.int32(n)
    else:
        n = np.int32(0); nnz = np.int32(0)
        ip = ix = None
    n = comm.bcast(n, root=0)
    nnz = comm.bcast(nnz, root=0)
    if rk != 0:
        ip = np.empty(n + 1, dtype=np.int32)
        ix = np.empty(nnz, dtype=np.int32)
    comm.Bcast(ip, root=0)
    comm.Bcast(ix, root=0)
    return int(n), ip, ix

# ────────────────────────────────────────────────────────────────
# Method 10: PETSc symbolic product only (no numeric phase)
# ────────────────────────────────────────────────────────────────
def method_petsc_symbolic(h5_path, comm):
    from petsc4py import PETSc
    rk = comm.Get_rank()
    if rk == 0:
        A_sp = load_adjacency(h5_path)
        A_csr = sp.csr_matrix(A_sp)
        n = A_csr.shape[0]
        ai = np.ascontiguousarray(A_csr.indptr, dtype=np.int32)
        aj = np.ascontiguousarray(A_csr.indices, dtype=np.int32)
        av = np.ones(len(aj), dtype=np.float64)

        A_pet = PETSc.Mat().createAIJ(
            size=(n, n), csr=(ai, aj, av), comm=PETSc.COMM_SELF,
        )
        A_pet.assemblyBegin(); A_pet.assemblyEnd()

        # Try product symbolic only
        A2_pet = PETSc.Mat()
        try:
            A2_pet.matProductSymbolic(A_pet, A_pet, None, 
                                      product_type='AB')
        except AttributeError:
            # Try alternative API
            A2_pet = A_pet.matMult(A_pet)  # fallback
        
        ai2, aj2, _ = A2_pet.getValuesCSR()
        A2_csc = sp.csc_matrix((np.ones(len(aj2)), aj2, ai2), shape=(n, n))
        ip = np.ascontiguousarray(A2_csc.indptr, dtype=np.int32)
        ix = np.ascontiguousarray(A2_csc.indices, dtype=np.int32)
        nnz = np.int32(len(ix))

        A_pet.destroy(); A2_pet.destroy()
        n = np.int32(n)
    else:
        n = np.int32(0); nnz = np.int32(0)
        ip = ix = None
    n = comm.bcast(n, root=0)
    nnz = comm.bcast(nnz, root=0)
    if rk != 0:
        ip = np.empty(n + 1, dtype=np.int32)
        ix = np.empty(nnz, dtype=np.int32)
    comm.Bcast(ip, root=0)
    comm.Bcast(ix, root=0)
    return int(n), ip, ix

# ────────────────────────────────────────────────────────────────
# Method 11: scipy on rank 0, but keeping CSR (skip CSC conversion)
# ────────────────────────────────────────────────────────────────
def method_scipy_csr_bcast(h5_path, comm):
    rk = comm.Get_rank()
    if rk == 0:
        A = load_adjacency(h5_path)
        A2 = sp.csr_matrix(A @ A)  # keep CSR, don't convert to CSC
        n = np.int32(A2.shape[0])
        ip = np.ascontiguousarray(A2.indptr, dtype=np.int32)
        ix = np.ascontiguousarray(A2.indices, dtype=np.int32)
        nnz = np.int32(len(ix))
    else:
        n = np.int32(0); nnz = np.int32(0)
        ip = ix = None
    n = comm.bcast(n, root=0)
    nnz = comm.bcast(nnz, root=0)
    if rk != 0:
        ip = np.empty(n + 1, dtype=np.int32)
        ix = np.empty(nnz, dtype=np.int32)
    comm.Bcast(ip, root=0)
    comm.Bcast(ix, root=0)
    return int(n), ip, ix

# ────────────────────────────────────────────────────────────────
# Run all methods
# ────────────────────────────────────────────────────────────────
METHODS = [
    ("scipy (CSC) + Bcast",            method_scipy_bcast),
    ("scipy (CSR) + Bcast",            method_scipy_csr_bcast),
    ("scipy bool + Bcast",             method_scipy_bool),
    ("scipy int8 + Bcast",             method_scipy_int8),
    ("scipy all-ranks (redundant)",    method_scipy_all_ranks),
    ("PETSc serial rank0 + Bcast",     method_petsc_serial_bcast),
    ("PETSc symbolic rank0 + Bcast",   method_petsc_symbolic),
    ("PETSc par matMult + gather",     method_petsc_matmult),
    ("PETSc par matMult fill=2",       method_petsc_matmult_fill),
    ("PETSc par A^T*A + gather",       method_petsc_AtA),
]

BENCHMARKS = [
    ("pLaplace2D", 9),
    ("GinzburgLandau2D", 9),
    ("HyperElasticity3D", 4),
]

for prob_name, lvl in BENCHMARKS:
    h5 = PROBLEMS[prob_name]["path"](lvl)
    if rank == 0:
        print(f"\n{'='*72}")
        A_ref = load_adjacency(h5)
        n_ref = A_ref.shape[0]
        print(f"  {prob_name} level {lvl}  (N={n_ref:,}, np={size})")
        print(f"{'='*72}")
        print(f"  {'Method':<38}  {'Time (s)':>10}  {'nnz(A²)':>12}")
        print(f"  {'-'*38}  {'-'*10}  {'-'*12}")

    for mname, mfunc in METHODS:
        comm.Barrier()
        try:
            t0 = time.perf_counter()
            result = mfunc(h5, comm)
            dt = time.perf_counter() - t0
            if result is None:
                if rank == 0:
                    print(f"  {mname:<38}  {'FAILED':>10}  {'—':>12}")
                continue
            n, ip, ix = result
            if rank == 0:
                nnz = len(ix)
                print(f"  {mname:<38}  {dt:>10.4f}  {nnz:>12,}")
        except Exception as e:
            if rank == 0:
                print(f"  {mname:<38}  {'ERROR':>10}  {str(e)[:30]}")
        sys.stdout.flush()
        comm.Barrier()

if rank == 0:
    print()

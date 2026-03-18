"""
Custom greedy coloring: "most coloured neighbours" vertex-selection heuristic.

Provides
--------
  color_custom(adjacency)
      Serial coloring via compiled C implementation.
  color_custom_omp(adjacency, nthreads=None)
      OpenMP shared-memory parallel version: block-partition local colouring
      followed by deterministic boundary-conflict resolution.
  color_custom_mpi(adjacency, comm)
      MPI parallel version: domain-decomposition local colouring followed by
      deterministic boundary-conflict resolution.

The algorithm is a faithful translation of the MATLAB ``my_greedy_color2``
function.  It computes A² = A·A (including diagonal), then greedily colours
vertices, always picking the uncoloured vertex among the *current* vertex's
A²-neighbours that has the most already-coloured neighbours.  This produces
colouring quality comparable to DSATUR at a fraction of the cost.

The C shared library (``custom_coloring.so``) is auto-compiled from
``custom_coloring.c`` on first import if it does not exist or is older than
the source.
"""

import ctypes
import os
import subprocess
import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# C library compilation and loading
# ---------------------------------------------------------------------------
_LIB = None
_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "custom_coloring.c")
_SO = os.path.join(_DIR, "custom_coloring.so")


def _get_lib():
    """Compile (if needed) and load the C shared library."""
    global _LIB
    if _LIB is not None:
        return _LIB

    need = not os.path.exists(_SO) or os.path.getmtime(_SRC) > os.path.getmtime(_SO)
    if need:
        # Try with OpenMP first, fall back to without
        try:
            subprocess.check_call(
                ["gcc", "-O3", "-fopenmp", "-shared", "-fPIC", "-o", _SO, _SRC],
                cwd=_DIR,
            )
        except subprocess.CalledProcessError:
            subprocess.check_call(
                ["gcc", "-O3", "-shared", "-fPIC", "-o", _SO, _SRC],
                cwd=_DIR,
            )

    _LIB = ctypes.CDLL(_SO)

    # custom_greedy_color(n, indptr, indices, colors) -> n_colors
    _LIB.custom_greedy_color.restype = ctypes.c_int
    _LIB.custom_greedy_color.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]

    # fix_coloring_conflicts(n, row_ptr, col_idx, n_boundary, boundary, colors) -> n_colors
    _LIB.fix_coloring_conflicts.restype = ctypes.c_int
    _LIB.fix_coloring_conflicts.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]

    # custom_greedy_color_omp(n, csc_ip, csc_ix, csr_ip, csr_ix, colors, nthreads) -> n_colors
    _LIB.custom_greedy_color_omp.restype = ctypes.c_int
    _LIB.custom_greedy_color_omp.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]

    # custom_has_openmp() -> int (0 = no OpenMP, >0 = max threads)
    _LIB.custom_has_openmp.restype = ctypes.c_int
    _LIB.custom_has_openmp.argtypes = []

    # custom_greedy_color_random(n, indptr, indices, colors, seed) -> n_colors
    _LIB.custom_greedy_color_random.restype = ctypes.c_int
    _LIB.custom_greedy_color_random.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint,
    ]

    # build_A2_pattern(n, ai, aj, a2_indptr, a2_indices, nnz_out) -> 0
    _LIB.build_A2_pattern.restype = ctypes.c_int
    _LIB.build_A2_pattern.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]

    return _LIB


def _i32(arr):
    """Ensure *arr* is a contiguous int32 ndarray."""
    return np.ascontiguousarray(arr, dtype=np.int32)


def _ptr(arr):
    """ctypes int-pointer to a contiguous int32 array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


# ---------------------------------------------------------------------------
# Serial coloring
# ---------------------------------------------------------------------------
def color_custom(adjacency):
    """
    Custom greedy distance-2 coloring (serial, C backend).

    Parameters
    ----------
    adjacency : scipy.sparse matrix
        Element–DOF adjacency matrix *A*.

    Returns
    -------
    n_colors : int
        Number of colours used.
    coloring : ndarray of int32
        0-based colour for each vertex.
    """
    lib = _get_lib()

    # A² is symmetric → CSR and CSC have identical structure.
    # Using CSR avoids the costly CSC conversion (~15–20 % faster).
    A2 = sp.csr_matrix(adjacency @ adjacency)
    n = A2.shape[0]

    indptr = _i32(A2.indptr)
    indices = _i32(A2.indices)
    colors = np.zeros(n, dtype=np.int32)

    nc = lib.custom_greedy_color(ctypes.c_int(n), _ptr(indptr), _ptr(indices), _ptr(colors))
    return int(nc), colors


def color_custom_random(adjacency_or_A2, seed=0, *, is_A2=False):
    """
    Randomised custom greedy coloring (serial, C backend).

    Same algorithm as :func:`color_custom` but with a random starting vertex
    and random tie-breaking when multiple candidates have equal score.
    Different seeds produce different (valid) colourings.

    Parameters
    ----------
    adjacency_or_A2 : scipy.sparse matrix
        Element–DOF adjacency *A* (default) or pre-computed *A²* if
        *is_A2* is ``True``.
    seed : int
        PRNG seed (e.g. MPI rank).
    is_A2 : bool
        If ``True``, skip the A² computation.

    Returns
    -------
    n_colors : int
    coloring : ndarray of int32
    """
    lib = _get_lib()

    # A² is symmetric → CSR ≡ CSC structurally.  Keep CSR to avoid conversion.
    if is_A2:
        A2 = sp.csr_matrix(adjacency_or_A2)
    else:
        A2 = sp.csr_matrix(adjacency_or_A2 @ adjacency_or_A2)

    n = A2.shape[0]
    indptr = _i32(A2.indptr)
    indices = _i32(A2.indices)
    colors = np.zeros(n, dtype=np.int32)

    nc = lib.custom_greedy_color_random(
        ctypes.c_int(n), _ptr(indptr), _ptr(indices),
        _ptr(colors), ctypes.c_uint(seed),
    )
    return int(nc), colors


# ---------------------------------------------------------------------------
# Parallel coloring (MPI)
# ---------------------------------------------------------------------------
def color_custom_mpi(adjacency, comm=None):
    """
    Parallel custom coloring via domain decomposition + conflict resolution.

    1. Each rank colours its local sub-graph of A² independently using the
       serial C algorithm.
    2. Partial colourings are gathered with ``Allgatherv``.
    3. Boundary conflicts (edges crossing partition boundaries) are fixed
       greedily by the deterministic C routine ``fix_coloring_conflicts``
       (runs identically on every rank – no extra MPI needed).

    Parameters
    ----------
    adjacency : scipy.sparse matrix
        Full adjacency matrix (must be available on every rank).
    comm : MPI communicator, optional
        Defaults to ``MPI.COMM_WORLD``.

    Returns
    -------
    n_colors : int
    coloring : ndarray of int32
    """
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()
    n = adjacency.shape[0]

    if size == 1:
        return color_custom(adjacency)

    lib = _get_lib()

    # --- full A² on every rank (redundant but simple) ---
    A2_csc = sp.csc_matrix(adjacency @ adjacency)
    A2_csr = sp.csr_matrix(A2_csc)

    # --- partition vertices among ranks ---
    chunk, rem = divmod(n, size)
    counts = [chunk + 1 if r < rem else chunk for r in range(size)]
    offsets = [0]
    for c in counts[:-1]:
        offsets.append(offsets[-1] + c)

    rstart = offsets[rank]
    local_n = counts[rank]

    # --- extract and colour local sub-graph ---
    local_A2 = sp.csc_matrix(A2_csc[rstart: rstart + local_n, rstart: rstart + local_n])
    li = _i32(local_A2.indptr)
    lx = _i32(local_A2.indices)
    local_colors = np.zeros(local_n, dtype=np.int32)

    lib.custom_greedy_color(ctypes.c_int(local_n), _ptr(li), _ptr(lx), _ptr(local_colors))

    # --- all-gather partial colourings ---
    colors = np.zeros(n, dtype=np.int32)
    comm.Allgatherv(local_colors, [colors, counts, offsets, MPI.INT])

    # --- identify boundary vertices (numpy-vectorised) ---
    partition_of = np.empty(n, dtype=np.int32)
    for r in range(size):
        partition_of[offsets[r]: offsets[r] + counts[r]] = r

    boundary_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        p0, p1 = A2_csr.indptr[i], A2_csr.indptr[i + 1]
        nbrs = A2_csr.indices[p0:p1]
        my_part = partition_of[i]
        if np.any(partition_of[nbrs[nbrs != i]] != my_part):
            boundary_mask[i] = True

    boundary = np.where(boundary_mask)[0].astype(np.int32)

    # --- fix boundary conflicts (deterministic, same on all ranks) ---
    rp = _i32(A2_csr.indptr)
    ci = _i32(A2_csr.indices)

    if len(boundary) > 0:
        bptr = _ptr(boundary)
    else:
        _dummy = np.zeros(1, dtype=np.int32)
        bptr = _ptr(_dummy)

    nc = lib.fix_coloring_conflicts(
        ctypes.c_int(n),
        _ptr(rp),
        _ptr(ci),
        ctypes.c_int(len(boundary)),
        bptr,
        _ptr(colors),
    )

    return int(nc), colors


# ---------------------------------------------------------------------------
# OpenMP parallel coloring
# ---------------------------------------------------------------------------
def color_custom_omp(adjacency, nthreads=None, reorder=True):
    """
    OpenMP-parallel custom greedy coloring.

    Block-partitions vertices among *nthreads* OpenMP threads.  Each thread
    independently colours its local sub-graph of A², then boundary conflicts
    (edges crossing partition boundaries) are fixed greedily.

    When *reorder* is ``True`` (default), a reverse Cuthill–McKee permutation
    is applied beforehand so that block partitions correspond to spatially
    local vertex clusters, drastically reducing boundary conflicts and
    improving both speed and colour quality.

    Typical use-case: the **C coloring step** is 2–4× faster than serial with
    16 threads, but the total time (A² + RCM + coloring) is comparable to
    serial because the reordering itself costs O(nnz).  This variant is most
    useful when the same sparsity pattern is coloured repeatedly or when the
    reordering can be amortised.

    Parameters
    ----------
    adjacency : scipy.sparse matrix
        Element–DOF adjacency matrix *A*.
    nthreads : int, optional
        Number of OpenMP threads (default: ``os.cpu_count()``).
    reorder : bool, optional
        Apply reverse Cuthill–McKee reordering before partitioning (default
        ``True``).  Without reordering, block partition by DOF index gives
        poor locality and the parallel version is typically *slower* than
        serial.

    Returns
    -------
    n_colors : int
        Number of colours used.
    coloring : ndarray of int32
        0-based colour for each vertex.
    """
    lib = _get_lib()

    if nthreads is None:
        nthreads = os.cpu_count() or 1

    # Check OpenMP availability
    has_omp = lib.custom_has_openmp()
    if has_omp == 0 and nthreads > 1:
        import warnings
        warnings.warn(
            "custom_coloring.so was compiled without OpenMP; "
            "falling back to serial coloring."
        )

    A2_csc = sp.csc_matrix(adjacency @ adjacency)
    n = A2_csc.shape[0]

    # Optional bandwidth-reducing reorder for better partition locality
    perm = None
    if reorder and nthreads > 1:
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        perm = reverse_cuthill_mckee(A2_csc)
        A2_csc = sp.csc_matrix(A2_csc[perm][:, perm])

    # A² is symmetric (DOF adjacency graph), so CSC data can serve as CSR too:
    # iterating column j of CSC gives the same neighbours as iterating row j of
    # CSR.  This avoids an expensive .tocsr() conversion for large matrices.
    ip = _i32(A2_csc.indptr)
    ix = _i32(A2_csc.indices)
    colors = np.zeros(n, dtype=np.int32)

    nc = lib.custom_greedy_color_omp(
        ctypes.c_int(n),
        _ptr(ip), _ptr(ix),   # CSC
        _ptr(ip), _ptr(ix),   # reused as CSR (symmetric)
        _ptr(colors),
        ctypes.c_int(nthreads),
    )

    # Undo permutation if applied
    if perm is not None:
        inv = np.empty(n, dtype=np.int32)
        inv[perm] = np.arange(n, dtype=np.int32)
        colors = colors[inv]

    return int(nc), colors

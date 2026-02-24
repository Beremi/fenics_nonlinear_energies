"""
Custom greedy coloring: "most coloured neighbours" vertex-selection heuristic.

Provides
--------
  color_custom(adjacency)
      Serial coloring via compiled C implementation.
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

    A2 = sp.csc_matrix(adjacency @ adjacency)
    n = A2.shape[0]

    indptr = _i32(A2.indptr)
    indices = _i32(A2.indices)
    colors = np.zeros(n, dtype=np.int32)

    nc = lib.custom_greedy_color(ctypes.c_int(n), _ptr(indptr), _ptr(indices), _ptr(colors))
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
    local_A2 = sp.csc_matrix(A2_csc[rstart : rstart + local_n, rstart : rstart + local_n])
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
        partition_of[offsets[r] : offsets[r] + counts[r]] = r

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

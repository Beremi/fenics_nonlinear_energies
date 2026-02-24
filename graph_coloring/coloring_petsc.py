"""
Graph coloring via PETSc MatColoring (serial and parallel).

Uses distance-2 coloring directly on the adjacency matrix,
avoiding the explicit formation of A@A.

The PETSc ``MatColoring`` class is not exposed in petsc4py bindings,
so we call the C functions through ``ctypes``.
"""

import ctypes
import os
from typing import Optional

import numpy as np
import scipy.sparse as sps
from mpi4py import MPI

# ---------------------------------------------------------------------------
# Lazy-load libpetsc
# ---------------------------------------------------------------------------
_libpetsc: Optional[ctypes.CDLL] = None


def _get_libpetsc() -> ctypes.CDLL:
    """Return the loaded libpetsc shared library (cached)."""
    global _libpetsc
    if _libpetsc is not None:
        return _libpetsc

    petsc_dir = os.environ.get("PETSC_DIR", "/usr/local/petsc")
    petsc_arch = os.environ.get("PETSC_ARCH", "")

    search_paths = [
        os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.so"),
        os.path.join(petsc_dir, "lib", "libpetsc.so"),
        "libpetsc.so",
    ]
    for path in search_paths:
        try:
            _libpetsc = ctypes.CDLL(path)
            return _libpetsc
        except OSError:
            continue
    raise RuntimeError(
        f"Could not load libpetsc.so.  Searched: {search_paths}. "
        "Set PETSC_DIR / PETSC_ARCH environment variables."
    )


# ---------------------------------------------------------------------------
# Helpers: scipy COO → PETSc MATMPIAIJ
# ---------------------------------------------------------------------------

def _scipy_coo_to_petsc_mat(
    adjacency: sps.spmatrix,
    comm: MPI.Comm,
) -> "PETSc.Mat":
    """Convert a *replicated* scipy sparse matrix to a PETSc Mat (distributed rows)."""
    from petsc4py import PETSc

    csr = sps.csr_matrix(adjacency)
    csr.sum_duplicates()
    csr.eliminate_zeros()
    N = csr.shape[0]

    mat = PETSc.Mat().createAIJ(size=(N, N), comm=comm)
    mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    mat.setUp()

    rstart, rend = mat.getOwnershipRange()
    for i in range(rstart, rend):
        row_start = csr.indptr[i]
        row_end = csr.indptr[i + 1]
        cols = csr.indices[row_start:row_end].astype(np.int32)
        vals = csr.data[row_start:row_end].astype(np.float64)
        mat.setValues(i, cols, vals)

    mat.assemble()
    return mat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def color_petsc(
    adjacency: sps.spmatrix,
    coloring_type: str = "greedy",
    distance: int = 2,
    comm: Optional[MPI.Comm] = None,
) -> tuple[int, np.ndarray]:
    """
    Distance-*distance* coloring of *adjacency* using PETSc MatColoring.

    Parameters
    ----------
    adjacency : scipy sparse matrix
        The element–DOF adjacency / sparsity pattern *P*  (replicated on every rank).
    coloring_type : str
        PETSc coloring algorithm: ``"greedy"``, ``"jp"``, ``"power"``, ``"lf"``,
        ``"id"``, ``"sl"``, ``"natural"``.
    distance : int
        Coloring distance (2 = color the graph of P²).
    comm : MPI.Comm or None
        MPI communicator.  ``None`` → ``MPI.COMM_SELF`` (sequential).

    Returns
    -------
    n_colors : int
        Number of distinct colors.
    coloring : np.ndarray, shape (N,), dtype int64  (only meaningful on rank 0
        for the full global array; every rank gets its own local portion in
        parallel, but we gather to rank 0 for benchmarking convenience).
    """
    if comm is None:
        comm = MPI.COMM_SELF

    lib = _get_libpetsc()
    mat = _scipy_coo_to_petsc_mat(adjacency, comm)
    N = adjacency.shape[0]

    # --- MatColoring ---
    mc = ctypes.c_void_p()
    lib.MatColoringCreate(ctypes.c_void_p(mat.handle), ctypes.byref(mc))
    lib.MatColoringSetDistance(mc, ctypes.c_int(distance))
    lib.MatColoringSetType(mc, coloring_type.encode())
    lib.MatColoringSetFromOptions(mc)

    isc = ctypes.c_void_p()
    lib.MatColoringApply(mc, ctypes.byref(isc))

    # --- Extract per-vertex colors ---
    n_colors_c = ctypes.c_int()
    is_arr_ptr = ctypes.POINTER(ctypes.c_void_p)()
    comm_self_f = MPI.COMM_SELF.py2f()
    lib.ISColoringGetIS(
        isc,
        ctypes.c_int(comm_self_f),
        ctypes.byref(n_colors_c),
        ctypes.byref(is_arr_ptr),
    )

    rstart, rend = mat.getOwnershipRange()
    local_n = rend - rstart
    local_coloring = np.full(local_n, -1, dtype=np.int64)

    for c in range(n_colors_c.value):
        is_handle = is_arr_ptr[c]
        is_size = ctypes.c_int()
        lib.ISGetLocalSize(ctypes.c_void_p(is_handle), ctypes.byref(is_size))
        indices_ptr = ctypes.POINTER(ctypes.c_int)()
        lib.ISGetIndices(ctypes.c_void_p(is_handle), ctypes.byref(indices_ptr))
        for j in range(is_size.value):
            global_idx = indices_ptr[j]
            local_idx = global_idx - rstart
            if 0 <= local_idx < local_n:
                local_coloring[local_idx] = c
        lib.ISRestoreIndices(ctypes.c_void_p(is_handle), ctypes.byref(indices_ptr))

    # Gather to build global coloring on rank 0
    if comm.Get_size() > 1:
        all_colorings = comm.gather(local_coloring, root=0)
        if comm.Get_rank() == 0:
            coloring = np.concatenate(all_colorings)
        else:
            coloring = np.empty(0, dtype=np.int64)
    else:
        coloring = local_coloring

    # Global n_colors (take max across ranks)
    local_max = int(n_colors_c.value)
    n_colors = comm.allreduce(local_max, op=MPI.MAX)

    # --- Cleanup ---
    lib.ISColoringRestoreIS(
        isc, ctypes.c_int(comm_self_f), ctypes.byref(is_arr_ptr)
    )
    lib.MatColoringDestroy(ctypes.byref(mc))
    lib.ISColoringDestroy(ctypes.byref(isc))
    mat.destroy()

    return n_colors, coloring

"""
Graph coloring via PETSc MatColoring (serial and parallel).

Uses distance-2 coloring directly on the adjacency matrix,
avoiding the explicit formation of A@A.

The PETSc ``MatColoring`` class is not exposed in petsc4py bindings,
so we call the C functions through ``ctypes``.
"""

import ctypes
import os
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sps
from mpi4py import MPI

# ---------------------------------------------------------------------------
# Lazy-load libpetsc
# ---------------------------------------------------------------------------
_libpetsc: Optional[ctypes.CDLL] = None


def _petsc_search_paths() -> list[str]:
    """Return candidate PETSc libraries from the environment and petsc4py."""
    petsc_dir = str(os.environ.get("PETSC_DIR", "")).strip()
    petsc_arch = str(os.environ.get("PETSC_ARCH", "")).strip()
    if not petsc_dir:
        try:
            import petsc4py

            config = petsc4py.get_config()
            petsc_dir = str(config.get("PETSC_DIR", "")).strip()
            petsc_arch = str(config.get("PETSC_ARCH", "")).strip()
        except (ImportError, OSError, RuntimeError):
            pass

    candidates: list[str] = []
    if petsc_dir:
        root = Path(petsc_dir)
        if petsc_arch:
            candidates.append(str(root / petsc_arch / "lib" / "libpetsc.so"))
        candidates.append(str(root / "lib" / "libpetsc.so"))
    candidates.append("libpetsc.so")
    return list(dict.fromkeys(candidates))


def _petsc_int_ctype():
    from petsc4py import PETSc

    return ctypes.c_int if np.dtype(PETSc.IntType).itemsize == 4 else ctypes.c_longlong


def _configure_signatures(lib: ctypes.CDLL) -> None:
    """Declare the small PETSc C surface used by this module."""
    PetscInt = _petsc_int_ctype()
    PetscErrorCode = ctypes.c_int
    PetscObject = ctypes.c_void_p

    lib.MatColoringCreate.argtypes = [PetscObject, ctypes.POINTER(PetscObject)]
    lib.MatColoringCreate.restype = PetscErrorCode
    lib.MatColoringSetDistance.argtypes = [PetscObject, PetscInt]
    lib.MatColoringSetDistance.restype = PetscErrorCode
    lib.MatColoringSetType.argtypes = [PetscObject, ctypes.c_char_p]
    lib.MatColoringSetType.restype = PetscErrorCode
    lib.MatColoringSetWeightType.argtypes = [PetscObject, ctypes.c_int]
    lib.MatColoringSetWeightType.restype = PetscErrorCode
    lib.MatColoringSetFromOptions.argtypes = [PetscObject]
    lib.MatColoringSetFromOptions.restype = PetscErrorCode
    lib.MatColoringApply.argtypes = [PetscObject, ctypes.POINTER(PetscObject)]
    lib.MatColoringApply.restype = PetscErrorCode
    lib.MatColoringTest.argtypes = [PetscObject, PetscObject]
    lib.MatColoringTest.restype = PetscErrorCode
    lib.MatColoringDestroy.argtypes = [ctypes.POINTER(PetscObject)]
    lib.MatColoringDestroy.restype = PetscErrorCode

    lib.ISColoringGetIS.argtypes = [
        PetscObject,
        ctypes.c_int,
        ctypes.POINTER(PetscInt),
        ctypes.POINTER(ctypes.POINTER(PetscObject)),
    ]
    lib.ISColoringGetIS.restype = PetscErrorCode
    lib.ISColoringRestoreIS.argtypes = [
        PetscObject,
        ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(PetscObject)),
    ]
    lib.ISColoringRestoreIS.restype = PetscErrorCode
    lib.ISColoringDestroy.argtypes = [ctypes.POINTER(PetscObject)]
    lib.ISColoringDestroy.restype = PetscErrorCode
    lib.ISGetLocalSize.argtypes = [PetscObject, ctypes.POINTER(PetscInt)]
    lib.ISGetLocalSize.restype = PetscErrorCode
    lib.ISGetIndices.argtypes = [PetscObject, ctypes.POINTER(ctypes.POINTER(PetscInt))]
    lib.ISGetIndices.restype = PetscErrorCode
    lib.ISRestoreIndices.argtypes = [PetscObject, ctypes.POINTER(ctypes.POINTER(PetscInt))]
    lib.ISRestoreIndices.restype = PetscErrorCode


def _check_petsc_error(code: int, operation: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"PETSc {operation} failed with error code {int(code)}")


def _get_libpetsc() -> ctypes.CDLL:
    """Return the loaded libpetsc shared library (cached)."""
    global _libpetsc
    if _libpetsc is not None:
        return _libpetsc

    search_paths = _petsc_search_paths()
    for path in search_paths:
        try:
            _libpetsc = ctypes.CDLL(path)
            _configure_signatures(_libpetsc)
            return _libpetsc
        except OSError:
            continue
    raise RuntimeError(
        f"Could not load libpetsc.so. Searched: {search_paths}. "
        "Set PETSC_DIR/PETSC_ARCH or install petsc4py with a valid configuration."
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
        cols = csr.indices[row_start:row_end].astype(PETSc.IntType, copy=False)
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
    weight_type: str = "lexical",
    allow_options: bool = False,
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
        ``"id"``, or ``"sl"``. PETSc's ``"natural"`` ordering is rejected for
        distance greater than one because it does not provide a valid
        distance-2 coloring in supported PETSc builds.
    distance : int
        Coloring distance (2 = color the graph of P²).
    weight_type : str
        Deterministic vertex weighting. Publication runs use ``"lexical"``;
        ``"lf"`` and ``"sl"`` are also supported.
    allow_options : bool
        If true, permit PETSc's ambient options database to override the
        explicit contract. False by default for reproducibility.
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
    coloring_type = str(coloring_type).strip().lower()
    distance = int(distance)
    supported = {"greedy", "jp", "power", "lf", "id", "sl"}
    if coloring_type not in supported:
        raise ValueError(
            f"Unsupported PETSc coloring type {coloring_type!r}; "
            f"choose one of {sorted(supported)}"
        )
    if distance < 1:
        raise ValueError("distance must be at least one")
    weight_type = str(weight_type).strip().lower()
    weight_types = {"random": 0, "lexical": 1, "lf": 2, "sl": 3}
    if weight_type not in weight_types:
        raise ValueError(
            f"Unsupported PETSc coloring weight type {weight_type!r}; "
            f"choose one of {sorted(weight_types)}"
        )

    lib = _get_libpetsc()
    mat = _scipy_coo_to_petsc_mat(adjacency, comm)
    PetscInt = _petsc_int_ctype()
    PETSC_USE_POINTER = 2

    # --- MatColoring ---
    mc = ctypes.c_void_p()
    _check_petsc_error(
        lib.MatColoringCreate(ctypes.c_void_p(mat.handle), ctypes.byref(mc)),
        "MatColoringCreate",
    )
    _check_petsc_error(
        lib.MatColoringSetDistance(mc, PetscInt(distance)),
        "MatColoringSetDistance",
    )
    _check_petsc_error(
        lib.MatColoringSetType(mc, coloring_type.encode()),
        "MatColoringSetType",
    )
    _check_petsc_error(
        lib.MatColoringSetWeightType(mc, ctypes.c_int(weight_types[weight_type])),
        "MatColoringSetWeightType",
    )
    if bool(allow_options):
        _check_petsc_error(
            lib.MatColoringSetFromOptions(mc),
            "MatColoringSetFromOptions",
        )

    isc = ctypes.c_void_p()
    _check_petsc_error(lib.MatColoringApply(mc, ctypes.byref(isc)), "MatColoringApply")
    _check_petsc_error(lib.MatColoringTest(mc, isc), "MatColoringTest")

    # --- Extract per-vertex colors ---
    n_colors_c = PetscInt()
    is_arr_ptr = ctypes.POINTER(ctypes.c_void_p)()
    _check_petsc_error(
        lib.ISColoringGetIS(
        isc,
        ctypes.c_int(PETSC_USE_POINTER),
        ctypes.byref(n_colors_c),
        ctypes.byref(is_arr_ptr),
        ),
        "ISColoringGetIS",
    )

    rstart, rend = mat.getOwnershipRange()
    local_n = rend - rstart
    local_coloring = np.full(local_n, -1, dtype=np.int64)

    for c in range(n_colors_c.value):
        is_handle = is_arr_ptr[c]
        is_size = PetscInt()
        _check_petsc_error(
            lib.ISGetLocalSize(ctypes.c_void_p(is_handle), ctypes.byref(is_size)),
            "ISGetLocalSize",
        )
        indices_ptr = ctypes.POINTER(PetscInt)()
        _check_petsc_error(
            lib.ISGetIndices(ctypes.c_void_p(is_handle), ctypes.byref(indices_ptr)),
            "ISGetIndices",
        )
        for j in range(is_size.value):
            global_idx = indices_ptr[j]
            local_idx = global_idx - rstart
            if 0 <= local_idx < local_n:
                local_coloring[local_idx] = c
        _check_petsc_error(
            lib.ISRestoreIndices(ctypes.c_void_p(is_handle), ctypes.byref(indices_ptr)),
            "ISRestoreIndices",
        )

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
    _check_petsc_error(
        lib.ISColoringRestoreIS(
            isc, ctypes.c_int(PETSC_USE_POINTER), ctypes.byref(is_arr_ptr)
        ),
        "ISColoringRestoreIS",
    )
    _check_petsc_error(lib.MatColoringDestroy(ctypes.byref(mc)), "MatColoringDestroy")
    _check_petsc_error(lib.ISColoringDestroy(ctypes.byref(isc)), "ISColoringDestroy")
    mat.destroy()

    return n_colors, coloring

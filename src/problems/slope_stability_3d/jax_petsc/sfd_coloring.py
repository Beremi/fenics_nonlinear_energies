"""Memory-bounded structural coloring for vector-valued 3D SFD recovery."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy import sparse

from src.core.coloring.coloring_petsc import color_petsc


@dataclass(frozen=True)
class ScalarSupportColoring:
    """Validated scalar support coloring and its total-node lookup."""

    colors: np.ndarray
    total_node_to_vertex: np.ndarray
    number_of_colors: int
    number_of_vertices: int
    number_of_nonzeros: int
    raw_pair_upper_bound: int
    build_seconds: float
    coloring_seconds: float
    validation_seconds: float


def _active_scalar_vertices(
    freedofs: np.ndarray,
    *,
    number_of_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    freedofs = np.asarray(freedofs, dtype=np.int64).ravel()
    if freedofs.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.full(int(number_of_nodes), -1, dtype=np.int32),
        )
    total_nodes = freedofs // 3
    if np.any(total_nodes < 0) or np.any(total_nodes >= int(number_of_nodes)):
        raise ValueError("freedofs contain a total DOF outside the scalar node range")
    active_nodes = np.unique(total_nodes)
    index_dtype = (
        np.int32
        if active_nodes.size <= int(np.iinfo(np.int32).max)
        else np.int64
    )
    total_node_to_vertex = np.full(
        int(number_of_nodes), -1, dtype=index_dtype
    )
    total_node_to_vertex[active_nodes] = np.arange(
        active_nodes.size, dtype=index_dtype
    )
    return active_nodes, total_node_to_vertex


def build_free_scalar_support_csr(
    elems_scalar: np.ndarray,
    freedofs: np.ndarray,
    *,
    number_of_nodes: int,
    chunk_elements: int = 4096,
) -> tuple[sparse.csr_matrix, np.ndarray, int]:
    """Build the active scalar-node FE support without vector pair expansion."""
    elems_scalar = np.asarray(elems_scalar, dtype=np.int64)
    if elems_scalar.ndim != 2:
        raise ValueError("elems_scalar must be a two-dimensional array")
    if elems_scalar.size and (
        np.any(elems_scalar < 0) or np.any(elems_scalar >= int(number_of_nodes))
    ):
        raise ValueError("elems_scalar contains a node outside number_of_nodes")
    chunk_elements = max(1, int(chunk_elements))
    active_nodes, total_node_to_vertex = _active_scalar_vertices(
        freedofs,
        number_of_nodes=int(number_of_nodes),
    )
    number_of_vertices = int(active_nodes.size)
    if number_of_vertices == 0:
        return (
            sparse.csr_matrix((0, 0), dtype=np.bool_),
            total_node_to_vertex,
            0,
        )

    mapped = total_node_to_vertex[elems_scalar]
    active_per_element = np.count_nonzero(mapped >= 0, axis=1).astype(np.int64)
    raw_pair_upper_bound = int(np.dot(active_per_element, active_per_element))
    key_base = np.int64(number_of_vertices)
    key_batches: list[np.ndarray] = []
    for start in range(0, int(mapped.shape[0]), chunk_elements):
        block = mapped[start : start + chunk_elements]
        rows = block[:, :, None]
        cols = block[:, None, :]
        valid = (rows >= 0) & (cols >= 0)
        if not np.any(valid):
            continue
        row_values = np.broadcast_to(rows, valid.shape)[valid].astype(
            np.int64, copy=False
        )
        col_values = np.broadcast_to(cols, valid.shape)[valid].astype(
            np.int64, copy=False
        )
        keys = np.unique(row_values * key_base + col_values)
        if keys.size:
            key_batches.append(np.asarray(keys, dtype=np.int64))

    if not key_batches:
        return (
            sparse.csr_matrix(
                (number_of_vertices, number_of_vertices), dtype=np.bool_
            ),
            total_node_to_vertex,
            raw_pair_upper_bound,
        )
    keys = np.unique(np.concatenate(key_batches, axis=0))
    index_dtype = (
        np.int32
        if number_of_vertices <= int(np.iinfo(np.int32).max)
        else np.int64
    )
    row = np.asarray(keys // key_base, dtype=index_dtype)
    col = np.asarray(keys % key_base, dtype=index_dtype)
    support = sparse.csr_matrix(
        (np.ones(keys.size, dtype=np.bool_), (row, col)),
        shape=(number_of_vertices, number_of_vertices),
        dtype=np.bool_,
    )
    support.sum_duplicates()
    support.data[:] = True
    support.sort_indices()
    return support, total_node_to_vertex, raw_pair_upper_bound


def validate_column_intersection_coloring(
    support: sparse.csr_matrix,
    colors: np.ndarray,
) -> None:
    """Require every scalar row's structurally active columns to be orthogonal."""
    support = sparse.csr_matrix(support)
    colors = np.asarray(colors, dtype=np.int64).ravel()
    if colors.shape != (support.shape[1],):
        raise ValueError("color array does not match the scalar support graph")
    if colors.size and np.any(colors < 0):
        raise ValueError("color array contains an unassigned vertex")
    for row in range(support.shape[0]):
        lo = int(support.indptr[row])
        hi = int(support.indptr[row + 1])
        row_colors = colors[support.indices[lo:hi]]
        if np.unique(row_colors).size != row_colors.size:
            raise ValueError(
                f"invalid distance-2 scalar coloring at support row {row}"
            )


def color_free_scalar_support(
    elems_scalar: np.ndarray,
    freedofs: np.ndarray,
    *,
    number_of_nodes: int,
    chunk_elements: int = 4096,
) -> ScalarSupportColoring:
    """Build, color, and independently validate the active scalar FE support."""
    started = time.perf_counter()
    support, total_node_to_vertex, raw_pairs = build_free_scalar_support_csr(
        elems_scalar,
        freedofs,
        number_of_nodes=int(number_of_nodes),
        chunk_elements=int(chunk_elements),
    )
    build_seconds = float(time.perf_counter() - started)

    started = time.perf_counter()
    number_of_colors, colors = color_petsc(
        support,
        coloring_type="greedy",
        distance=2,
        weight_type="lexical",
        allow_options=False,
    )
    coloring_seconds = float(time.perf_counter() - started)

    started = time.perf_counter()
    validate_column_intersection_coloring(support, colors)
    validation_seconds = float(time.perf_counter() - started)
    return ScalarSupportColoring(
        colors=np.asarray(colors, dtype=np.int32),
        total_node_to_vertex=np.asarray(total_node_to_vertex),
        number_of_colors=int(number_of_colors),
        number_of_vertices=int(support.shape[0]),
        number_of_nonzeros=int(support.nnz),
        raw_pair_upper_bound=int(raw_pairs),
        build_seconds=build_seconds,
        coloring_seconds=coloring_seconds,
        validation_seconds=validation_seconds,
    )


def lift_scalar_colors_to_reordered_free_dofs(
    scalar: ScalarSupportColoring,
    *,
    freedofs: np.ndarray,
    reordered_to_original_free: np.ndarray,
    reordered_dofs: np.ndarray,
) -> np.ndarray:
    """Lift scalar distance-2 colors by displacement component and compact them."""
    freedofs = np.asarray(freedofs, dtype=np.int64).ravel()
    perm = np.asarray(reordered_to_original_free, dtype=np.int64).ravel()
    reordered_dofs = np.asarray(reordered_dofs, dtype=np.int64).ravel()
    if reordered_dofs.size == 0:
        return np.zeros(0, dtype=np.int32)
    if np.any(reordered_dofs < 0) or np.any(reordered_dofs >= perm.size):
        raise ValueError("reordered_dofs lie outside the free-DOF permutation")
    original_free = perm[reordered_dofs]
    if np.any(original_free < 0) or np.any(original_free >= freedofs.size):
        raise ValueError("permutation maps outside the original free-DOF array")
    total_dofs = freedofs[original_free]
    scalar_vertices = scalar.total_node_to_vertex[total_dofs // 3]
    if np.any(scalar_vertices < 0):
        raise ValueError("a free vector DOF maps to an inactive scalar vertex")
    lifted = (
        3 * scalar.colors[np.asarray(scalar_vertices, dtype=np.int64)]
        + total_dofs % 3
    )
    _labels, compact = np.unique(lifted, return_inverse=True)
    return np.asarray(compact, dtype=np.int32)

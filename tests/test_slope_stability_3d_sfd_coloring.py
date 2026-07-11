from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from src.problems.slope_stability_3d.jax_petsc.reordered_element_assembler import (
    SlopeStability3DReorderedElementAssembler,
)
from src.problems.slope_stability_3d.jax_petsc.sfd_coloring import (
    build_free_scalar_support_csr,
    color_free_scalar_support,
    lift_scalar_colors_to_reordered_free_dofs,
)


def _toy_connectivity() -> np.ndarray:
    return np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)


def _mixed_free_dofs() -> np.ndarray:
    full = np.arange(15, dtype=np.int64)
    constrained = np.asarray([0, 7, 11], dtype=np.int64)
    return full[~np.isin(full, constrained)]


def _reordered_vector_pattern(
    elems_scalar: np.ndarray,
    freedofs: np.ndarray,
    perm: np.ndarray,
) -> sparse.csr_matrix:
    number_of_total_dofs = int(3 * (np.max(elems_scalar) + 1))
    total_to_original_free = np.full(number_of_total_dofs, -1, dtype=np.int64)
    total_to_original_free[freedofs] = np.arange(freedofs.size, dtype=np.int64)
    original_to_reordered = np.empty(freedofs.size, dtype=np.int64)
    original_to_reordered[perm] = np.arange(freedofs.size, dtype=np.int64)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for scalar_nodes in elems_scalar:
        total = (
            3 * scalar_nodes[:, None] + np.arange(3, dtype=np.int64)[None, :]
        ).ravel()
        original = total_to_original_free[total]
        local = original_to_reordered[original[original >= 0]]
        rows.append(np.repeat(local, local.size))
        cols.append(np.tile(local, local.size))
    return sparse.coo_matrix(
        (
            np.ones(sum(row.size for row in rows), dtype=np.bool_),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(freedofs.size, freedofs.size),
    ).tocsr()


def test_scalar_component_lift_is_valid_for_mixed_vector_constraints() -> None:
    elems_scalar = _toy_connectivity()
    freedofs = _mixed_free_dofs()
    perm = np.asarray([7, 0, 10, 2, 5, 9, 1, 8, 3, 11, 4, 6], dtype=np.int64)

    scalar = color_free_scalar_support(
        elems_scalar,
        freedofs,
        number_of_nodes=5,
    )
    colors = lift_scalar_colors_to_reordered_free_dofs(
        scalar,
        freedofs=freedofs,
        reordered_to_original_free=perm,
        reordered_dofs=np.arange(freedofs.size, dtype=np.int64),
    )
    vector_pattern = _reordered_vector_pattern(elems_scalar, freedofs, perm)
    conflicts = (vector_pattern.T @ vector_pattern).tocoo()

    off_diagonal = conflicts.row != conflicts.col
    assert np.all(colors[conflicts.row[off_diagonal]] != colors[conflicts.col[off_diagonal]])
    assert scalar.number_of_nonzeros > 0
    assert scalar.number_of_colors == int(scalar.colors.max()) + 1


def test_scalar_support_builder_is_deterministic_and_keeps_diagonal() -> None:
    args = (_toy_connectivity(), _mixed_free_dofs())
    first, first_lookup, first_pairs = build_free_scalar_support_csr(
        *args,
        number_of_nodes=5,
        chunk_elements=1,
    )
    second, second_lookup, second_pairs = build_free_scalar_support_csr(
        *args,
        number_of_nodes=5,
        chunk_elements=8,
    )

    assert (first != second).nnz == 0
    np.testing.assert_array_equal(first_lookup, second_lookup)
    np.testing.assert_array_equal(first.diagonal(), np.ones(first.shape[0], dtype=bool))
    assert first_pairs == second_pairs


def test_p4_sfd_coloring_fails_closed_without_scalar_support_inputs() -> None:
    assembler = object.__new__(SlopeStability3DReorderedElementAssembler)
    assembler.block_size = 3
    assembler.params = {"element_degree": 4}
    assembler.layout = SimpleNamespace(perm=np.arange(1), n_free=1)

    with pytest.raises(MemoryError, match="memory-bounded scalar-support"):
        assembler._build_sfd_column_coloring(np.asarray([0], dtype=np.int64))

from __future__ import annotations

import numpy as np

from src.core.petsc.reordered_element_base import (
    ReorderedElementAssemblerBase,
    _sfd_column_conflict_matrix,
)


def test_sfd_column_conflicts_follow_shared_rows_for_owned_row_pattern():
    # Rank-local SFD reconstructs only owned rows, so the extracted pattern is
    # generally not symmetric. Columns 1 and 2 conflict because both contribute
    # to extracted row 0, even though there is no row-1/row-2 path in this
    # owned-row-only pattern.
    row_idx = np.asarray([0, 0, 1, 1], dtype=np.int64)
    col_idx = np.asarray([1, 2, 0, 3], dtype=np.int64)

    conflicts = _sfd_column_conflict_matrix(row_idx, col_idx, n_cols=4).toarray()

    assert conflicts[1, 2] == 1.0
    assert conflicts[2, 1] == 1.0
    assert conflicts[0, 3] == 1.0
    assert conflicts[3, 0] == 1.0
    assert conflicts[1, 3] == 0.0
    assert np.count_nonzero(np.diag(conflicts)) == 0


def test_sfd_tangent_batches_have_fixed_shape_and_zero_padded_tail():
    assembler = object.__new__(ReorderedElementAssemblerBase)
    assembler._sfd_n_colors = 3
    assembler._sfd_local_vector_size = 6
    assembler._sfd_seed_color_offsets = np.asarray([0, 2, 3, 5], dtype=np.int64)
    assembler._sfd_seed_local_dofs_by_color = np.asarray(
        [0, 4, 1, 2, 5], dtype=np.int32
    )

    first, first_active = assembler._sfd_tangent_batch(
        color_start=0,
        batch_size=2,
    )
    tail, tail_active = assembler._sfd_tangent_batch(
        color_start=2,
        batch_size=2,
    )

    np.testing.assert_array_equal(
        first,
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    np.testing.assert_array_equal(
        tail,
        np.asarray(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    assert first_active == 2
    assert tail_active == 1


def test_sfd_color_major_plan_scatters_each_batch_without_all_color_tensor():
    assembler = object.__new__(ReorderedElementAssemblerBase)
    assembler._sfd_entry_color_offsets = np.asarray([0, 2, 3, 5], dtype=np.int64)
    assembler._sfd_entry_positions_by_color = np.asarray(
        [0, 3, 2, 1, 4], dtype=np.int32
    )
    assembler._sfd_entry_local_rows_by_color = np.asarray(
        [1, 4, 0, 2, 5], dtype=np.int32
    )
    owned_values = np.zeros(5, dtype=np.float64)
    hvps = np.asarray(
        [
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
        ]
    )

    assembler._scatter_sfd_hvp_batch(
        owned_values,
        hvps,
        color_start=1,
        active=2,
    )

    np.testing.assert_array_equal(owned_values, [0.0, 22.0, 10.0, 0.0, 25.0])

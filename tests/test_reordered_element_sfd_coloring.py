from __future__ import annotations

import numpy as np

from src.core.petsc.reordered_element_base import _sfd_column_conflict_matrix


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

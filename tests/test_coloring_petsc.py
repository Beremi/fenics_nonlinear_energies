from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from src.core.coloring import coloring_petsc


def _path_pattern(size: int) -> sparse.csr_matrix:
    return sparse.diags(
        [np.ones(size - 1), np.ones(size), np.ones(size - 1)],
        offsets=[-1, 0, 1],
        shape=(size, size),
        format="csr",
    )


def test_petsc_distance_two_coloring_is_valid_without_environment_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PETSC_DIR", raising=False)
    monkeypatch.delenv("PETSC_ARCH", raising=False)
    monkeypatch.setattr(coloring_petsc, "_libpetsc", None)

    pattern = _path_pattern(12)
    number_of_colors, colors = coloring_petsc.color_petsc(
        pattern,
        coloring_type="greedy",
        distance=2,
    )

    squared = (pattern @ pattern).tocsr()
    assert number_of_colors == int(colors.max()) + 1
    assert colors.shape == (12,)
    for row in range(squared.shape[0]):
        columns = squared.indices[squared.indptr[row] : squared.indptr[row + 1]]
        distinct = columns[columns != row]
        assert np.all(colors[distinct] != colors[row])


@pytest.mark.parametrize("coloring_type", ["natural", "unknown"])
def test_petsc_coloring_rejects_unsafe_or_unknown_types(coloring_type: str) -> None:
    with pytest.raises(ValueError, match="Unsupported PETSc coloring type"):
        coloring_petsc.color_petsc(
            _path_pattern(4),
            coloring_type=coloring_type,
            distance=2,
        )


def test_petsc_coloring_rejects_nonpositive_distance() -> None:
    with pytest.raises(ValueError, match="distance must be at least one"):
        coloring_petsc.color_petsc(_path_pattern(4), distance=0)

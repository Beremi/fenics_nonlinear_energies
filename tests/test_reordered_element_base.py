from __future__ import annotations

import numpy as np
from scipy import sparse

from src.core.petsc.reordered_element_base import (
    local_vec_from_full,
    select_permutation,
)


def test_select_permutation_block_xyz_scalar_uses_coordinate_sort():
    adjacency = sparse.eye(4, format="csr")
    coords = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    freedofs = np.arange(4, dtype=np.int64)

    perm = select_permutation(
        "block_xyz",
        adjacency=adjacency,
        coords_all=coords,
        freedofs=freedofs,
        n_parts=2,
        block_size=1,
    )

    assert perm.tolist() == [2, 1, 0, 3]


def test_select_permutation_block_xyz_vector_preserves_triplets():
    adjacency = sparse.eye(6, format="csr")
    coords = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    freedofs = np.arange(6, dtype=np.int64)

    perm = select_permutation(
        "block_xyz",
        adjacency=adjacency,
        coords_all=coords,
        freedofs=freedofs,
        n_parts=2,
        block_size=3,
    )

    assert perm.tolist() == [3, 4, 5, 0, 1, 2]


def test_local_vec_from_full_merges_dirichlet_and_free_values():
    full_reordered = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    total_to_free_reord = np.array([-1, 0, 1, -1, 2], dtype=np.int64)
    local_total_nodes = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    dirichlet_full = np.array([5.0, -1.0, -1.0, 6.0, -1.0], dtype=np.float64)

    v_local = local_vec_from_full(
        full_reordered,
        total_to_free_reord,
        local_total_nodes,
        dirichlet_full,
    )

    np.testing.assert_allclose(v_local, np.array([5.0, 10.0, 20.0, 6.0, 30.0]))

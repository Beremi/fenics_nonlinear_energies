"""Shared PETSc GAMG helpers."""

from __future__ import annotations

import numpy as np


def build_gamg_coordinates(part, freedofs, nodes, *, block_size: int = 1) -> np.ndarray:
    """Return owned coordinate rows in the reordered free-DOF ordering."""
    freedofs_arr = np.asarray(freedofs, dtype=np.int64).ravel()
    nodes_arr = np.asarray(nodes, dtype=np.float64)
    owned_orig_free = np.asarray(part.perm[part.lo : part.hi], dtype=np.int64)
    owned_total_dofs = freedofs_arr[owned_orig_free]
    if owned_total_dofs.size == 0:
        spatial_dim = int(nodes_arr.shape[1]) if nodes_arr.ndim == 2 else 1
        return np.zeros((0, spatial_dim), dtype=np.float64)

    if int(block_size) <= 1:
        return np.asarray(nodes_arr[owned_total_dofs], dtype=np.float64)

    if owned_total_dofs.size % int(block_size) != 0:
        raise RuntimeError(
            f"Owned DOFs are not divisible by block_size={block_size}; "
            "cannot build block coordinates"
        )

    offsets = np.arange(int(block_size), dtype=np.int64)
    blocks = owned_total_dofs.reshape(-1, int(block_size))
    contiguous = np.all(blocks == blocks[:, :1] + offsets[None, :])
    same_node = np.all((blocks // int(block_size)) == (blocks[:, :1] // int(block_size)))
    if not (contiguous and same_node):
        raise RuntimeError(
            "DOF ordering does not preserve block triplets. "
            "Disable reordering or request block-compatible coordinates."
        )

    node_ids = blocks[:, 0] // int(block_size)
    return np.asarray(nodes_arr[node_ids], dtype=np.float64)

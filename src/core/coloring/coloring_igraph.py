"""
Graph coloring via igraph (sequential greedy).

Reproduces the coloring pipeline from tools/graph_sfd.py:
  connectivity = adjacency @ adjacency   (distance-2 sparsity)
  extract lower-triangle edges
  greedy vertex coloring via igraph
"""

import igraph
import numpy as np
import scipy.sparse as sps


def color_igraph(adjacency: sps.spmatrix) -> tuple[int, np.ndarray]:
    """
    Distance-2 greedy coloring of *adjacency* using igraph.

    Parameters
    ----------
    adjacency : sparse matrix (any format)
        The element–DOF adjacency / sparsity pattern *P*.

    Returns
    -------
    n_colors : int
        Number of distinct colors.
    coloring : np.ndarray, shape (n,), dtype int64
        Per-vertex color assignment  (0 … n_colors-1).
    """
    adjacency = sps.csr_matrix(adjacency)
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    n = adjacency.shape[0]

    connectivity = adjacency @ adjacency
    i, j = connectivity.tocoo().coords
    mask = i > j
    i = i[mask]
    j = j[mask]
    indices = np.array((i, j)).T

    graph = igraph.Graph(n, indices, directed=False)
    coloring_raw = graph.vertex_coloring_greedy()
    coloring = np.array(coloring_raw, dtype=np.int64).ravel()

    n_colors = int(coloring.max() + 1)
    return n_colors, coloring

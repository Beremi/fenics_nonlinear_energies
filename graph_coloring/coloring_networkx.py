"""
Graph coloring via NetworkX (sequential, multiple strategies).

Supports several greedy strategies including DSATUR which typically
gives the best color quality.
"""

import numpy as np
import scipy.sparse as sps

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def _sparse_to_nx_graph(mat: sps.spmatrix) -> "nx.Graph":
    """Convert a scipy sparse matrix to a NetworkX Graph (edges where nnz)."""
    coo = sps.coo_matrix(mat)
    G = nx.Graph()
    G.add_nodes_from(range(mat.shape[0]))
    # Only upper triangle to avoid duplicate edges
    mask = coo.row < coo.col
    edges = zip(coo.row[mask].tolist(), coo.col[mask].tolist())
    G.add_edges_from(edges)
    return G


def color_networkx(
    adjacency: sps.spmatrix,
    strategy: str = "DSATUR",
) -> tuple[int, np.ndarray]:
    """
    Distance-2 coloring via NetworkX greedy_color on explicit A^2.

    Parameters
    ----------
    adjacency : sparse matrix
        The element–DOF adjacency / sparsity pattern.
    strategy : str
        NetworkX greedy strategy: "DSATUR", "largest_first",
        "smallest_last", "random_sequential", "independent_set".

    Returns
    -------
    n_colors : int
    coloring : np.ndarray, shape (n,), dtype int64
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is not installed")

    csr = sps.csr_matrix(adjacency)
    csr.sum_duplicates()
    csr.eliminate_zeros()
    n = csr.shape[0]

    # Build A^2
    connectivity = csr @ csr
    G = _sparse_to_nx_graph(connectivity)

    # Map strategy name
    strategy_map = {
        "DSATUR": "DSATUR",
        "largest_first": "largest_first",
        "smallest_last": "smallest_last",
        "random_sequential": "random_sequential",
        "independent_set": "independent_set",
        "saturation_largest_first": "saturation_largest_first",
    }
    nx_strategy = strategy_map.get(strategy, strategy)

    color_dict = nx.coloring.greedy_color(G, strategy=nx_strategy)
    coloring = np.array([color_dict[i] for i in range(n)], dtype=np.int64)
    n_colors = int(coloring.max() + 1)
    return n_colors, coloring

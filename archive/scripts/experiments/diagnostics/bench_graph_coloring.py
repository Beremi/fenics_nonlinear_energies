"""
Benchmark graph coloring for all three problems (Ginzburg-Landau 2D, p-Laplace 2D, HyperElasticity 3D).

Loads adjacency matrices from mesh HDF5 files, constructs the connectivity graph,
and measures the greedy graph coloring (via igraph) — recording number of colors and wall time.
Stops progressing through mesh levels once a single coloring exceeds 10 s.

Results are written to:
  tmp_work/graph_coloring_results.json   (machine-readable)
  stdout                                 (human-readable table)
"""

import json
import os
import sys
import time

import h5py
import igraph
import numpy as np
import scipy.sparse as sps

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(REPO_ROOT, "mesh_data")
TMP_DIR = os.path.join(REPO_ROOT, "tmp_work")
os.makedirs(TMP_DIR, exist_ok=True)

TIME_LIMIT = 10.0  # seconds – stop after a level exceeds this

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_adjacency(h5_path: str) -> sps.coo_matrix:
    """Load the COO adjacency matrix stored in an HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        grp = f["adjacency"]
        data = grp["data"][:]
        row = grp["row"][:]
        col = grp["col"][:]
        shape = tuple(grp["shape"][:])
    return sps.coo_matrix((data, (row, col)), shape=shape)


def color_adjacency(adjacency: sps.coo_matrix):
    """
    Reproduce the exact coloring pipeline from tools/graph_sfd.py:
      1. Build connectivity = adjacency @ adjacency
      2. Extract lower-triangle edges
      3. Greedy vertex coloring via igraph
    Returns (n_vertices, nnz_adjacency, nnz_connectivity, n_colors, elapsed_seconds).
    """
    adjacency = adjacency.tocsr()
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    n = adjacency.shape[0]
    nnz_adj = adjacency.nnz

    t0 = time.perf_counter()

    connectivity = adjacency @ adjacency
    i, j = connectivity.tocoo().coords
    mask = i > j
    i = i[mask]
    j = j[mask]
    indices = np.array((i, j)).T

    graph = igraph.Graph(n, indices, directed=False)
    coloring = graph.vertex_coloring_greedy()
    n_colors = max(coloring) + 1

    elapsed = time.perf_counter() - t0

    nnz_conn = connectivity.nnz

    return n, nnz_adj, nnz_conn, n_colors, elapsed


# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------

PROBLEMS = [
    {
        "name": "pLaplace2D",
        "levels": list(range(1, 10)),           # 1 .. 9
        "path_template": os.path.join(MESH_DIR, "pLaplace", "pLaplace_level{level}.h5"),
    },
    {
        "name": "GinzburgLandau2D",
        "levels": list(range(2, 10)),           # 2 .. 9
        "path_template": os.path.join(MESH_DIR, "GinzburgLandau", "GL_level{level}.h5"),
    },
    {
        "name": "HyperElasticity3D",
        "levels": list(range(1, 5)),            # 1 .. 4
        "path_template": os.path.join(MESH_DIR, "HyperElasticity", "HyperElasticity_level{level}.h5"),
    },
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    all_results = {}

    for prob in PROBLEMS:
        pname = prob["name"]
        print(f"\n{'=' * 70}")
        print(f" {pname}")
        print(f"{'=' * 70}")
        header = f"{'Level':>5} | {'N':>10} | {'nnz(A)':>12} | {'nnz(A²)':>12} | {'Colors':>6} | {'Time (s)':>10}"
        print(header)
        print("-" * len(header))

        results = []
        for level in prob["levels"]:
            h5_path = prob["path_template"].format(level=level)
            if not os.path.isfile(h5_path):
                print(f"{level:>5} | {'FILE NOT FOUND':>10}")
                continue

            adj = load_adjacency(h5_path)
            n, nnz_adj, nnz_conn, n_colors, elapsed = color_adjacency(adj)
            results.append({
                "level": level,
                "n_vertices": int(n),
                "nnz_adjacency": int(nnz_adj),
                "nnz_connectivity": int(nnz_conn),
                "n_colors": int(n_colors),
                "time_s": round(elapsed, 6),
            })
            print(f"{level:>5} | {n:>10,} | {nnz_adj:>12,} | {nnz_conn:>12,} | {n_colors:>6} | {elapsed:>10.4f}")

            if elapsed > TIME_LIMIT:
                print(f"  *** Exceeded {TIME_LIMIT}s limit – skipping remaining levels for {pname}")
                break

        all_results[pname] = results

    # Save JSON
    out_path = os.path.join(TMP_DIR, "graph_coloring_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

"""Utility to load adjacency matrices from the mesh HDF5 files."""

import os
import h5py
import scipy.sparse as sps


MESH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mesh_data")

PROBLEMS = {
    "pLaplace2D": {
        "levels": list(range(1, 10)),
        "path": lambda lvl: os.path.join(MESH_DIR, "pLaplace", f"pLaplace_level{lvl}.h5"),
    },
    "GinzburgLandau2D": {
        "levels": list(range(2, 10)),
        "path": lambda lvl: os.path.join(MESH_DIR, "GinzburgLandau", f"GL_level{lvl}.h5"),
    },
    "HyperElasticity3D": {
        "levels": list(range(1, 5)),
        "path": lambda lvl: os.path.join(MESH_DIR, "HyperElasticity", f"HyperElasticity_level{lvl}.h5"),
    },
}


def load_adjacency(h5_path: str) -> sps.coo_matrix:
    """Load the COO adjacency matrix stored in an HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        grp = f["adjacency"]
        data = grp["data"][:]
        row = grp["row"][:]
        col = grp["col"][:]
        shape = tuple(grp["shape"][:])
    return sps.coo_matrix((data, (row, col)), shape=shape)

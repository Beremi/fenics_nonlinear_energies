"""Structured interval mesh support for the thesis 1D harness.

The 1D direction-study code intentionally mirrors the 2D mesh helpers, but keeps
its simpler interval-specific data in a separate file so the main 2D mesh module
stays easier to read.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp


GEOMETRY_INTERVAL_PI = "interval_pi"
SUPPORTED_1D_INIT_MODES = ("sine", "random")


def _build_adjacency(elems: np.ndarray, freedofs: np.ndarray, n_total: int) -> sp.coo_matrix:
    """Build the interval free-DOF adjacency graph used by JAX helpers."""
    freedofs = np.asarray(freedofs, dtype=np.int64)
    full_to_free = np.full(int(n_total), -1, dtype=np.int64)
    full_to_free[freedofs] = np.arange(freedofs.size, dtype=np.int64)
    local = full_to_free[np.asarray(elems, dtype=np.int64)]
    rr = np.repeat(local, local.shape[1], axis=1).reshape(-1)
    cc = np.tile(local, (1, local.shape[1])).reshape(-1)
    mask = (rr >= 0) & (cc >= 0)
    adjacency = sp.coo_matrix(
        (np.ones(int(np.count_nonzero(mask)), dtype=np.float64), (rr[mask], cc[mask])),
        shape=(freedofs.size, freedofs.size),
    )
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    return adjacency


class MeshPLaplaceU31D:
    """Uniform P1 interval mesh on ``[0, π]`` for Chapter 5 direction studies."""

    def __init__(
        self,
        mesh_level: int,
        *,
        p: float = 2.0,
        geometry: str = GEOMETRY_INTERVAL_PI,
        init_mode: str = "sine",
        seed: int = 0,
    ) -> None:
        if str(geometry) != GEOMETRY_INTERVAL_PI:
            raise ValueError(f"Unsupported 1D geometry {geometry!r}")
        if str(init_mode) not in SUPPORTED_1D_INIT_MODES:
            raise ValueError(
                f"Unsupported init_mode={init_mode!r}; expected one of {SUPPORTED_1D_INIT_MODES}"
            )
        mesh_level = int(mesh_level)
        if mesh_level < 1:
            raise ValueError("mesh_level must be >= 1")

        # The thesis uses uniform interval meshes with h = pi / 2^L.
        n_subdiv = 2**mesh_level
        h = math.pi / float(n_subdiv)
        nodes = np.linspace(0.0, math.pi, n_subdiv + 1, dtype=np.float64)
        elems = np.column_stack(
            (
                np.arange(0, n_subdiv, dtype=np.int64),
                np.arange(1, n_subdiv + 1, dtype=np.int64),
            )
        )
        freedofs = np.arange(1, n_subdiv, dtype=np.int64)
        dv = np.tile(np.array([[-1.0 / h, 1.0 / h]], dtype=np.float64), (n_subdiv, 1))
        vol = np.full(n_subdiv, h, dtype=np.float64)
        u_0 = np.zeros(n_subdiv + 1, dtype=np.float64)

        if init_mode == "sine":
            u_init = np.sin(nodes[freedofs])
        else:
            rng = np.random.default_rng(int(seed))
            u_init = rng.standard_normal(freedofs.size)

        self.mesh_level = mesh_level
        self.geometry = GEOMETRY_INTERVAL_PI
        self.init_mode = str(init_mode)
        self.seed = int(seed)
        self.h = float(h)
        self.params = {
            "topology": "interval",
            "geometry": self.geometry,
            "mesh_level": mesh_level,
            "p": float(p),
            "h": float(h),
            "nodes_1d": nodes,
            "u_0": u_0,
            "freedofs": freedofs,
            "elems": elems,
            "dv": dv,
            "vol": vol,
        }
        self.adjacency = _build_adjacency(elems, freedofs, n_total=nodes.size)
        self.u_init = np.asarray(u_init, dtype=np.float64)

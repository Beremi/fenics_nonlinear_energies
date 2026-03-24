"""Structured mesh and seed support shared by the ``plaplace_u3`` family.

This module is deliberately reusable and low-level: it builds the structured
square / square-with-hole meshes, boundary masks, adjacency, and the thesis seed
families. Solver-specific logic stays in the thesis package.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from src.core.problem_data.hdf5 import jaxify_problem_data


GEOMETRY_SQUARE_PI = "square_pi"
GEOMETRY_SQUARE_HOLE_PI = "square_hole_pi"
SUPPORTED_GEOMETRIES = (GEOMETRY_SQUARE_PI, GEOMETRY_SQUARE_HOLE_PI)

INIT_SINE = "sine"
INIT_SINE_X2 = "sine_x2"
INIT_SINE_Y2 = "sine_y2"
INIT_SKEW = "skew"
INIT_ABS_SINE_Y2 = "abs_sine_y2"
INIT_ABS_SINE_3X3 = "abs_sine_3x3"
INIT_RANDOM = "random"
SUPPORTED_INIT_MODES = (
    INIT_SINE,
    INIT_SINE_X2,
    INIT_SINE_Y2,
    INIT_SKEW,
    INIT_ABS_SINE_Y2,
    INIT_ABS_SINE_3X3,
    INIT_RANDOM,
)

DOMAIN_MAX = math.pi
HOLE_MIN = math.pi / 4.0
HOLE_MAX = 3.0 * math.pi / 4.0


def _level_subdivisions(mesh_level: int) -> int:
    mesh_level = int(mesh_level)
    if mesh_level < 1:
        raise ValueError("mesh_level must be >= 1")
    return 2**mesh_level


def _triangle_operators(
    nodes: np.ndarray,
    elems: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return element gradients of the P1 basis and the triangle volumes."""
    verts = np.asarray(nodes[np.asarray(elems, dtype=np.int64)], dtype=np.float64)
    x0 = verts[:, 0, 0]
    y0 = verts[:, 0, 1]
    x1 = verts[:, 1, 0]
    y1 = verts[:, 1, 1]
    x2 = verts[:, 2, 0]
    y2 = verts[:, 2, 1]

    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if np.any(det <= 0.0):
        raise ValueError("Encountered non-positive triangle orientation in generated mesh")

    dvx = np.stack((y1 - y2, y2 - y0, y0 - y1), axis=1) / det[:, None]
    dvy = np.stack((x2 - x1, x0 - x2, x1 - x0), axis=1) / det[:, None]
    vol = 0.5 * det
    return dvx.astype(np.float64), dvy.astype(np.float64), vol.astype(np.float64)


def _build_scalar_adjacency(elems: np.ndarray, freedofs: np.ndarray) -> sp.coo_matrix:
    """Build the free-DOF adjacency used by JAX differentiation helpers."""
    freedofs = np.asarray(freedofs, dtype=np.int64)
    elems = np.asarray(elems, dtype=np.int64)
    n_total = int(np.max(elems)) + 1 if elems.size else int(freedofs.size)
    n_free = int(freedofs.size)
    full_to_free = np.full(n_total, -1, dtype=np.int64)
    full_to_free[freedofs] = np.arange(n_free, dtype=np.int64)

    local = full_to_free[elems]
    rr = np.repeat(local, local.shape[1], axis=1).reshape(-1)
    cc = np.tile(local, (1, local.shape[1])).reshape(-1)
    mask = (rr >= 0) & (cc >= 0)
    adjacency = sp.coo_matrix(
        (np.ones(int(np.count_nonzero(mask)), dtype=np.float64), (rr[mask], cc[mask])),
        shape=(n_free, n_free),
    )
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    return adjacency


def _compact_mesh(
    nodes_full: np.ndarray,
    elems_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop unused nodes after removing cells, then renumber the element table."""
    used = np.unique(np.asarray(elems_full, dtype=np.int64).reshape(-1))
    old_to_new = np.full(nodes_full.shape[0], -1, dtype=np.int64)
    old_to_new[used] = np.arange(used.size, dtype=np.int64)
    nodes = np.asarray(nodes_full[used], dtype=np.float64)
    elems = old_to_new[np.asarray(elems_full, dtype=np.int64)]
    return nodes, elems


def _hole_cell_mask(n_subdiv: int) -> np.ndarray:
    """Mark which square cells stay in the square-with-hole geometry."""
    if int(n_subdiv) % 4 != 0:
        raise ValueError(
            "square_hole_pi requires a level with 2^level divisible by 4; use level >= 2"
        )
    lo = int(n_subdiv // 4)
    hi = int(3 * n_subdiv // 4)
    ii, jj = np.meshgrid(
        np.arange(n_subdiv, dtype=np.int64),
        np.arange(n_subdiv, dtype=np.int64),
        indexing="xy",
    )
    return ~((ii >= lo) & (ii < hi) & (jj >= lo) & (jj < hi))


def _generate_structured_triangles(
    *,
    mesh_level: int,
    geometry: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate the thesis structured triangle mesh for the chosen geometry."""
    n_subdiv = _level_subdivisions(mesh_level)
    h = DOMAIN_MAX / float(n_subdiv)
    coords = np.linspace(0.0, DOMAIN_MAX, n_subdiv + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(coords, coords, indexing="xy")
    nodes_full = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    node_ids = np.arange((n_subdiv + 1) * (n_subdiv + 1), dtype=np.int64).reshape(
        n_subdiv + 1, n_subdiv + 1
    )

    keep_cells = np.ones((n_subdiv, n_subdiv), dtype=bool)
    if geometry == GEOMETRY_SQUARE_HOLE_PI:
        keep_cells = _hole_cell_mask(n_subdiv)
    elif geometry != GEOMETRY_SQUARE_PI:
        raise ValueError(
            f"Unsupported geometry={geometry!r}; expected one of {SUPPORTED_GEOMETRIES}"
        )

    jj, ii = np.nonzero(keep_cells)
    ll = node_ids[jj, ii]
    lr = node_ids[jj, ii + 1]
    ul = node_ids[jj + 1, ii]
    ur = node_ids[jj + 1, ii + 1]

    # Every square cell is split with the same diagonal, matching the structured
    # thesis-style triangulation used throughout this packet.
    tri_lo = np.column_stack((ll, lr, ul))
    tri_hi = np.column_stack((ur, ul, lr))
    elems_full = np.vstack((tri_lo, tri_hi)).astype(np.int64)
    nodes, elems = _compact_mesh(nodes_full, elems_full)
    return nodes, elems, float(h)


def _boundary_mask(nodes: np.ndarray, *, geometry: str, h: float) -> np.ndarray:
    """Return the homogeneous Dirichlet boundary mask for the selected geometry."""
    tol = max(1.0e-12, 1.0e-10 * float(h))
    x = np.asarray(nodes[:, 0], dtype=np.float64)
    y = np.asarray(nodes[:, 1], dtype=np.float64)

    outer = (
        np.isclose(x, 0.0, atol=tol)
        | np.isclose(x, DOMAIN_MAX, atol=tol)
        | np.isclose(y, 0.0, atol=tol)
        | np.isclose(y, DOMAIN_MAX, atol=tol)
    )
    if geometry == GEOMETRY_SQUARE_PI:
        return outer

    on_hole_vertical = (
        (np.isclose(x, HOLE_MIN, atol=tol) | np.isclose(x, HOLE_MAX, atol=tol))
        & (y >= HOLE_MIN - tol)
        & (y <= HOLE_MAX + tol)
    )
    on_hole_horizontal = (
        (np.isclose(y, HOLE_MIN, atol=tol) | np.isclose(y, HOLE_MAX, atol=tol))
        & (x >= HOLE_MIN - tol)
        & (x <= HOLE_MAX + tol)
    )
    return outer | on_hole_vertical | on_hole_horizontal


def _build_initial_guess(
    nodes: np.ndarray,
    freedofs: np.ndarray,
    *,
    init_mode: str,
    seed: int,
) -> np.ndarray:
    """Return one thesis initial guess restricted to free DOFs."""
    x = np.asarray(nodes[:, 0], dtype=np.float64)
    y = np.asarray(nodes[:, 1], dtype=np.float64)

    if init_mode == INIT_SINE:
        values = np.sin(x) * np.sin(y)
    elif init_mode == INIT_SINE_X2:
        values = 10.0 * np.sin(2.0 * x) * np.sin(y)
    elif init_mode == INIT_SINE_Y2:
        values = 10.0 * np.sin(x) * np.sin(2.0 * y)
    elif init_mode == INIT_SKEW:
        values = 4.0 * (x - y) * np.sin(x) * np.sin(y)
    elif init_mode == INIT_ABS_SINE_Y2:
        values = 4.0 * np.abs(np.sin(x) * np.sin(2.0 * y))
    elif init_mode == INIT_ABS_SINE_3X3:
        values = np.abs(4.0 * np.sin(3.0 * x) * np.sin(3.0 * y))
    elif init_mode == INIT_RANDOM:
        rng = np.random.default_rng(int(seed))
        values = rng.standard_normal(nodes.shape[0])
    else:
        raise ValueError(
            f"Unsupported init_mode={init_mode!r}; expected one of {SUPPORTED_INIT_MODES}"
        )

    return np.asarray(values[np.asarray(freedofs, dtype=np.int64)], dtype=np.float64)


def build_problem_data(
    mesh_level: int,
    *,
    geometry: str,
    p: float,
) -> tuple[dict[str, object], sp.coo_matrix]:
    """Build the mesh/assembly data shared by the 2D thesis solvers."""
    nodes, elems, h = _generate_structured_triangles(
        mesh_level=int(mesh_level),
        geometry=str(geometry),
    )
    boundary = _boundary_mask(nodes, geometry=str(geometry), h=float(h))
    freedofs = np.flatnonzero(~boundary).astype(np.int64)
    u_0 = np.zeros(nodes.shape[0], dtype=np.float64)
    dvx, dvy, vol = _triangle_operators(nodes, elems)
    adjacency = _build_scalar_adjacency(elems, freedofs)

    params = {
        "nodes": nodes,
        "u_0": u_0,
        "freedofs": freedofs,
        "elems": elems.astype(np.int64),
        "dvx": dvx,
        "dvy": dvy,
        "vol": vol,
        "p": float(p),
        "h": float(h),
        "geometry": str(geometry),
        "mesh_level": int(mesh_level),
    }
    return params, adjacency


class MeshPLaplaceU32D:
    """Structured P1 thesis geometry support for the plaplace_u3 family."""

    def __init__(
        self,
        mesh_level: int,
        *,
        p: float = 2.0,
        geometry: str = GEOMETRY_SQUARE_PI,
        init_mode: str = INIT_SINE,
        seed: int = 0,
    ) -> None:
        if geometry not in SUPPORTED_GEOMETRIES:
            raise ValueError(
                f"Unsupported geometry={geometry!r}; expected one of {SUPPORTED_GEOMETRIES}"
            )
        if init_mode not in SUPPORTED_INIT_MODES:
            raise ValueError(
                f"Unsupported init_mode={init_mode!r}; expected one of {SUPPORTED_INIT_MODES}"
            )

        self.mesh_level = int(mesh_level)
        self.geometry = str(geometry)
        self.init_mode = str(init_mode)
        self.seed = int(seed)
        self.filename = None
        self.params, self.adjacency = build_problem_data(
            self.mesh_level,
            geometry=self.geometry,
            p=float(p),
        )
        self.h = float(self.params["h"])
        self.u_init = _build_initial_guess(
            np.asarray(self.params["nodes"], dtype=np.float64),
            np.asarray(self.params["freedofs"], dtype=np.int64),
            init_mode=self.init_mode,
            seed=self.seed,
        )

    def get_data_jax(self):
        """Return JAX-friendly arrays together with adjacency and the seed vector."""
        import jax.numpy as jnp

        params = jaxify_problem_data(
            self.params,
            arrays={
                "nodes": "float64",
                "u_0": "float64",
                "freedofs": "int32",
                "elems": "int32",
                "dvx": "float64",
                "dvy": "float64",
                "vol": "float64",
            },
            scalars={"p": float, "h": float},
        )
        return params, self.adjacency, jnp.asarray(self.u_init, dtype=jnp.float64)

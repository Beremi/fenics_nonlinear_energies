"""Structured unit-square mesh support for the arctan-resonance problem family."""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from src.core.problem_data.hdf5 import jaxify_problem_data


GEOMETRY_SQUARE_UNIT = "square_unit"
SUPPORTED_GEOMETRIES = (GEOMETRY_SQUARE_UNIT,)

INIT_SINE = "sine"
INIT_RANDOM = "random"
SUPPORTED_INIT_MODES = (INIT_SINE, INIT_RANDOM)

DOMAIN_MIN = 0.0
DOMAIN_MAX = 1.0


def _level_subdivisions(mesh_level: int) -> int:
    mesh_level = int(mesh_level)
    if mesh_level < 1:
        raise ValueError("mesh_level must be >= 1")
    return 2**mesh_level


def _triangle_operators(
    nodes: np.ndarray,
    elems: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _generate_structured_triangles(mesh_level: int) -> tuple[np.ndarray, np.ndarray, float]:
    n_subdiv = _level_subdivisions(mesh_level)
    h = (DOMAIN_MAX - DOMAIN_MIN) / float(n_subdiv)
    coords = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n_subdiv + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(coords, coords, indexing="xy")
    nodes = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    node_ids = np.arange((n_subdiv + 1) * (n_subdiv + 1), dtype=np.int64).reshape(
        n_subdiv + 1, n_subdiv + 1
    )

    ll = node_ids[:-1, :-1].reshape(-1)
    lr = node_ids[:-1, 1:].reshape(-1)
    ul = node_ids[1:, :-1].reshape(-1)
    ur = node_ids[1:, 1:].reshape(-1)
    tri_lo = np.column_stack((ll, lr, ul))
    tri_hi = np.column_stack((ur, ul, lr))
    elems = np.vstack((tri_lo, tri_hi)).astype(np.int64)
    return nodes.astype(np.float64), elems, float(h)


def _boundary_mask(nodes: np.ndarray, *, h: float) -> np.ndarray:
    tol = max(1.0e-12, 1.0e-10 * float(h))
    x = np.asarray(nodes[:, 0], dtype=np.float64)
    y = np.asarray(nodes[:, 1], dtype=np.float64)
    return (
        np.isclose(x, DOMAIN_MIN, atol=tol)
        | np.isclose(x, DOMAIN_MAX, atol=tol)
        | np.isclose(y, DOMAIN_MIN, atol=tol)
        | np.isclose(y, DOMAIN_MAX, atol=tol)
    )


def _build_initial_guess(
    nodes: np.ndarray,
    freedofs: np.ndarray,
    *,
    init_mode: str,
    seed: int,
) -> np.ndarray:
    x = np.asarray(nodes[:, 0], dtype=np.float64)
    y = np.asarray(nodes[:, 1], dtype=np.float64)
    if init_mode == INIT_SINE:
        values = np.sin(math.pi * x) * np.sin(math.pi * y)
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
    if str(geometry) != GEOMETRY_SQUARE_UNIT:
        raise ValueError(
            f"Unsupported geometry={geometry!r}; expected one of {SUPPORTED_GEOMETRIES}"
        )
    nodes, elems, h = _generate_structured_triangles(int(mesh_level))
    boundary = _boundary_mask(nodes, h=float(h))
    freedofs = np.flatnonzero(~boundary).astype(np.int64)
    u_0 = np.zeros(nodes.shape[0], dtype=np.float64)
    dvx, dvy, vol = _triangle_operators(nodes, elems)
    adjacency = _build_scalar_adjacency(elems, freedofs)
    params = {
        "topology": "triangle",
        "geometry": str(geometry),
        "mesh_level": int(mesh_level),
        "nodes": nodes,
        "u_0": u_0,
        "freedofs": freedofs,
        "elems": elems.astype(np.int64),
        "dvx": dvx,
        "dvy": dvy,
        "vol": vol,
        "p": float(p),
        "h": float(h),
    }
    return params, adjacency


class MeshPLaplaceUPArctan2D:
    """Structured P1 unit-square mesh support for the arctan-resonance family."""

    def __init__(
        self,
        mesh_level: int,
        *,
        p: float = 2.0,
        geometry: str = GEOMETRY_SQUARE_UNIT,
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


"""Procedural P2 mesh and frozen snapshot helpers for the slope-stability prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import basix
import h5py
from mpi4py import MPI
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.core.problem_data.hdf5 import (
    MESH_DATA_ROOT,
    load_problem_hdf5,
    load_problem_hdf5_fields,
)
from src.core.petsc.dof_partition import petsc_ownership_range
from src.core.petsc.reordered_element_base import inverse_permutation, select_permutation


LEGACY_DEFAULT_CASE = "ssr_homo_capture_p2_h1"
DEFAULT_LEVEL = 3
LEVEL_TO_H = {
    1: 4.0,
    2: 2.0,
    3: 1.0,
    4: 0.5,
    5: 0.25,
    6: 0.125,
    7: 0.0625,
}
LEVEL_TO_CASE = {level: f"ssr_homo_capture_p2_level{level}" for level in LEVEL_TO_H}
DEFAULT_CASE = LEGACY_DEFAULT_CASE
COORD_DECIMALS = 12


@dataclass(frozen=True)
class SlopeStabilityCaseData:
    case_name: str
    level: int
    nodes: np.ndarray
    elems_scalar: np.ndarray
    elems: np.ndarray
    surf: np.ndarray
    q_mask: np.ndarray
    freedofs: np.ndarray
    elem_B: np.ndarray
    quad_weight: np.ndarray
    force: np.ndarray
    u_0: np.ndarray
    eps_p_old: np.ndarray
    adjacency: sp.coo_matrix
    h: float
    x1: float
    x2: float
    x3: float
    y1: float
    y2: float
    beta_deg: float
    E: float
    nu: float
    c0: float
    phi_deg: float
    psi_deg: float
    gamma: float
    davis_type: str
    lambda_target_default: float

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.elems_scalar.shape[0])

    @property
    def n_q(self) -> int:
        return int(self.quad_weight.shape[1])


def supported_levels() -> tuple[int, ...]:
    return tuple(sorted(LEVEL_TO_H))


def h_for_level(level: int) -> float:
    try:
        return float(LEVEL_TO_H[int(level)])
    except KeyError as exc:
        raise ValueError(f"Unsupported slope-stability level {level!r}") from exc


def case_name_for_level(level: int) -> str:
    try:
        return str(LEVEL_TO_CASE[int(level)])
    except KeyError as exc:
        raise ValueError(f"Unsupported slope-stability level {level!r}") from exc


def level_for_case_name(case_name: str) -> int:
    case_name = str(case_name)
    if case_name == LEGACY_DEFAULT_CASE:
        return DEFAULT_LEVEL
    for level, canonical in LEVEL_TO_CASE.items():
        if case_name == canonical:
            return int(level)
    raise ValueError(f"Unsupported slope-stability case {case_name!r}")


def canonical_case_name(case_name: str) -> str:
    return case_name_for_level(level_for_case_name(case_name))


def same_mesh_case_name(level: int, degree: int) -> str:
    return f"{case_name_for_level(int(level))}_p{int(degree)}_same_mesh"


def build_scalar_mesh_counts(
    *,
    h: float,
    x1: float,
    x2: float,
    x3: float,
    y1: float,
    y2: float,
) -> dict[str, int]:
    nx12 = int(round((x1 + x2) / h))
    nx3 = int(round(x3 / h))
    nx = nx12 + nx3
    ny1 = int(round(y1 / h))
    ny2 = int(round(y2 / h))
    ny = ny1 + ny2
    n_nodes = (2 * ny1 + 1) * (2 * nx + 1) + 2 * ny2 * (2 * nx12 + 1)
    n_elem = 2 * nx * ny1 + 2 * nx12 * ny2
    return {
        "nx12": nx12,
        "nx3": nx3,
        "nx": nx,
        "ny1": ny1,
        "ny2": ny2,
        "ny": ny,
        "n_nodes": n_nodes,
        "n_elem": n_elem,
    }


def _quadrature_volume_p2() -> tuple[np.ndarray, np.ndarray]:
    xi = np.array(
        [
            [
                0.1012865073235,
                0.7974269853531,
                0.1012865073235,
                0.4701420641051,
                0.4701420641051,
                0.0597158717898,
                1.0 / 3.0,
            ],
            [
                0.1012865073235,
                0.1012865073235,
                0.7974269853531,
                0.0597158717898,
                0.4701420641051,
                0.4701420641051,
                1.0 / 3.0,
            ],
        ],
        dtype=np.float64,
    )
    wf = (
        np.array(
            [
                0.1259391805448,
                0.1259391805448,
                0.1259391805448,
                0.1323941527885,
                0.1323941527885,
                0.1323941527885,
                0.225,
            ],
            dtype=np.float64,
        )
        / 2.0
    )
    return xi, wf


def _quadrature_volume_p1() -> tuple[np.ndarray, np.ndarray]:
    xi = np.array([[1.0 / 3.0], [1.0 / 3.0]], dtype=np.float64)
    wf = np.array([0.5], dtype=np.float64)
    return xi, wf


def _quadrature_degree_for_lagrange(degree: int) -> int:
    degree = int(degree)
    if degree <= 1:
        return 1
    return 2 * degree + 1


def _local_basis_volume_p2(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xi1 = xi[0, :]
    xi2 = xi[1, :]
    xi0 = 1.0 - xi1 - xi2
    n_q = xi.shape[1]
    hatp = np.array(
        [
            xi0 * (2.0 * xi0 - 1.0),
            xi1 * (2.0 * xi1 - 1.0),
            xi2 * (2.0 * xi2 - 1.0),
            4.0 * xi1 * xi2,
            4.0 * xi0 * xi2,
            4.0 * xi0 * xi1,
        ],
        dtype=np.float64,
    )
    dhat1 = np.array(
        [
            -4.0 * xi0 + 1.0,
            4.0 * xi1 - 1.0,
            np.zeros(n_q, dtype=np.float64),
            4.0 * xi2,
            -4.0 * xi2,
            4.0 * (xi0 - xi1),
        ],
        dtype=np.float64,
    )
    dhat2 = np.array(
        [
            -4.0 * xi0 + 1.0,
            np.zeros(n_q, dtype=np.float64),
            4.0 * xi2 - 1.0,
            4.0 * xi1,
            4.0 * (xi0 - xi2),
            -4.0 * xi1,
        ],
        dtype=np.float64,
    )
    return hatp, dhat1, dhat2


def _local_basis_volume_p1(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_q = xi.shape[1]
    hatp = np.array(
        [
            1.0 - xi[0, :] - xi[1, :],
            xi[0, :],
            xi[1, :],
        ],
        dtype=np.float64,
    )
    dhat1 = np.array(
        [
            -np.ones(n_q, dtype=np.float64),
            np.ones(n_q, dtype=np.float64),
            np.zeros(n_q, dtype=np.float64),
        ],
        dtype=np.float64,
    )
    dhat2 = np.array(
        [
            -np.ones(n_q, dtype=np.float64),
            np.zeros(n_q, dtype=np.float64),
            np.ones(n_q, dtype=np.float64),
        ],
        dtype=np.float64,
    )
    return hatp, dhat1, dhat2


def _triangle_lagrange_element(degree: int) -> basix.finite_element.FiniteElement:
    return basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.triangle,
        int(degree),
        basix.LagrangeVariant.equispaced,
    )


def _assemble_triangle_operators_lagrange(
    nodes: np.ndarray,
    elems_scalar: np.ndarray,
    *,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    degree = int(degree)
    element = _triangle_lagrange_element(degree)
    qpts, wf = basix.make_quadrature(
        basix.CellType.triangle,
        _quadrature_degree_for_lagrange(degree),
    )
    tab = element.tabulate(1, np.asarray(qpts, dtype=np.float64))
    hatp = np.asarray(tab[0, :, :, 0], dtype=np.float64).T
    dhat1 = np.asarray(tab[1, :, :, 0], dtype=np.float64).T
    dhat2 = np.asarray(tab[2, :, :, 0], dtype=np.float64).T

    n_elem = elems_scalar.shape[0]
    n_q = hatp.shape[1]
    n_p = elems_scalar.shape[1]
    elem_B = np.zeros((n_elem, n_q, 3, 2 * n_p), dtype=np.float64)
    quad_weight = np.zeros((n_elem, n_q), dtype=np.float64)

    for e in range(n_elem):
        verts = np.asarray(nodes[elems_scalar[e, :3]], dtype=np.float64)
        x_v = verts[:, 0]
        y_v = verts[:, 1]
        j11 = float(x_v[1] - x_v[0])
        j12 = float(y_v[1] - y_v[0])
        j21 = float(x_v[2] - x_v[0])
        j22 = float(y_v[2] - y_v[0])
        det_j = j11 * j22 - j12 * j21
        inv = np.array([[j22, -j12], [-j21, j11]], dtype=np.float64) / det_j
        for q in range(n_q):
            grads = inv @ np.vstack((dhat1[:, q], dhat2[:, q]))
            dphix = grads[0, :]
            dphiy = grads[1, :]
            B = np.zeros((3, 2 * n_p), dtype=np.float64)
            for a in range(n_p):
                B[0, 2 * a] = dphix[a]
                B[1, 2 * a + 1] = dphiy[a]
                B[2, 2 * a] = dphiy[a]
                B[2, 2 * a + 1] = dphix[a]
            elem_B[e, q, :, :] = B
            quad_weight[e, q] = abs(det_j) * float(wf[q])

    return elem_B, quad_weight, hatp


def _generate_homogeneous_slope_mesh_p2(
    *,
    h: float,
    x1: float,
    x2: float,
    x3: float,
    y1: float,
    y2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = build_scalar_mesh_counts(h=h, x1=x1, x2=x2, x3=x3, y1=y1, y2=y2)
    nx12 = counts["nx12"]
    nx3 = counts["nx3"]
    nx = counts["nx"]
    ny1 = counts["ny1"]
    ny2 = counts["ny2"]
    ny = counts["ny"]

    coord_x12 = np.linspace(0.0, x1 + x2, 2 * nx12 + 1, dtype=np.float64)
    coord_x3 = np.linspace(x1 + x2, x1 + x2 + x3, 2 * nx3 + 1, dtype=np.float64)
    coord_x = np.concatenate((coord_x12, coord_x3[1:]))

    coord_y1 = np.linspace(0.0, y1, 2 * ny1 + 1, dtype=np.float64)
    coord_y2 = np.linspace(y1, y1 + y2, 2 * ny2 + 1, dtype=np.float64)
    coord_y = np.concatenate((coord_y1, coord_y2[1:]))

    coord = np.zeros((2, counts["n_nodes"]), dtype=np.float64)
    C = np.zeros((2 * nx + 1, 2 * ny + 1), dtype=np.int64)
    n_n = 0

    for j in range(2 * ny1 + 1):
        for i in range(2 * nx + 1):
            C[i, j] = n_n
            coord[:, n_n] = np.array([coord_x[i], coord_y[j]], dtype=np.float64)
            n_n += 1

    for j in range(2 * ny1 + 1, 2 * ny + 1):
        x_max = x1 + x2 * (y1 + y2 - coord_y[j]) / y2
        local_x = np.linspace(0.0, x_max, 2 * nx12 + 1, dtype=np.float64)
        for i in range(2 * nx12 + 1):
            C[i, j] = n_n
            coord[:, n_n] = np.array([local_x[i], coord_y[j]], dtype=np.float64)
            n_n += 1

    elem = np.zeros((6, 2 * nx * ny), dtype=np.int64)
    n_e = 0
    for j in range(ny1):
        for i in range(nx):
            elem[:, n_e] = np.array(
                [
                    C[2 * i, 2 * j],
                    C[2 * i + 2, 2 * j],
                    C[2 * i, 2 * j + 2],
                    C[2 * i + 1, 2 * j + 1],
                    C[2 * i, 2 * j + 1],
                    C[2 * i + 1, 2 * j],
                ],
                dtype=np.int64,
            )
            n_e += 1
            elem[:, n_e] = np.array(
                [
                    C[2 * i + 2, 2 * j + 2],
                    C[2 * i, 2 * j + 2],
                    C[2 * i + 2, 2 * j],
                    C[2 * i + 1, 2 * j + 1],
                    C[2 * i + 2, 2 * j + 1],
                    C[2 * i + 1, 2 * j + 2],
                ],
                dtype=np.int64,
            )
            n_e += 1
    n1_e = n_e
    for j in range(ny1, ny):
        for i in range(nx12):
            elem[:, n_e] = np.array(
                [
                    C[2 * i, 2 * j],
                    C[2 * i + 2, 2 * j],
                    C[2 * i, 2 * j + 2],
                    C[2 * i + 1, 2 * j + 1],
                    C[2 * i, 2 * j + 1],
                    C[2 * i + 1, 2 * j],
                ],
                dtype=np.int64,
            )
            n_e += 1
            elem[:, n_e] = np.array(
                [
                    C[2 * i + 2, 2 * j + 2],
                    C[2 * i, 2 * j + 2],
                    C[2 * i + 2, 2 * j],
                    C[2 * i + 1, 2 * j + 1],
                    C[2 * i + 2, 2 * j + 1],
                    C[2 * i + 1, 2 * j + 2],
                ],
                dtype=np.int64,
            )
            n_e += 1
    elem = elem[:, :n_e]
    n2_e = n_e - n1_e

    n_ed = 0
    n1_ed = nx * (ny1 + 1)
    Eh1 = np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx, ny1 + 1, order="F")
    n_ed += n1_ed
    n1_ed = (nx + 1) * ny1
    Ev1 = np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx + 1, ny1, order="F")
    n_ed += n1_ed
    n1_ed = nx * ny1
    Ed1 = np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx, ny1, order="F")
    n_ed += n1_ed

    E12 = Eh1[:, :ny1].reshape(-1, order="F")
    E23 = Ev1[1:, :ny1].reshape(-1, order="F")
    E34 = Eh1[:, 1 : ny1 + 1].reshape(-1, order="F")
    E14 = Ev1[:nx, :ny1].reshape(-1, order="F")
    E24 = Ed1.reshape(-1, order="F")
    aux_elem_ed = np.vstack((E12, E24, E14, E34, E24, E23))
    elem1_ed = aux_elem_ed.reshape(3, n1_e, order="F")

    n1_ed = nx12 * ny2
    Eh2 = np.concatenate(
        (
            Eh1[:nx12, -1][:, None],
            np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx12, ny2, order="F"),
        ),
        axis=1,
    )
    n_ed += n1_ed
    n1_ed = (nx12 + 1) * ny2
    Ev2 = np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx12 + 1, ny2, order="F")
    n_ed += n1_ed
    n1_ed = nx12 * ny2
    Ed2 = np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx12, ny2, order="F")

    E12 = Eh2[:, :ny2].reshape(-1, order="F")
    E23 = Ev2[1:, :ny2].reshape(-1, order="F")
    E34 = Eh2[:, 1 : ny2 + 1].reshape(-1, order="F")
    E14 = Ev2[:nx12, :ny2].reshape(-1, order="F")
    E24 = Ed2.reshape(-1, order="F")
    aux_elem_ed = np.vstack((E12, E24, E14, E34, E24, E23))
    elem2_ed = aux_elem_ed.reshape(3, n2_e, order="F")
    elem_ed = np.hstack((elem1_ed, elem2_ed))

    Tlb1 = np.zeros((nx + 1, ny1 + 1), dtype=np.int64)
    Trt1 = np.zeros((nx + 1, ny1 + 1), dtype=np.int64)
    if n1_e:
        Tlb1[:nx, :ny1] = np.arange(0, n1_e, 2, dtype=np.int64).reshape(nx, ny1, order="F") + 1
        Tlb1[:nx12, -1] = np.arange(0, 2 * nx12, 2, dtype=np.int64) + n1_e + 1
        Trt1[1:, 1 : ny1 + 1] = np.arange(1, n1_e, 2, dtype=np.int64).reshape(nx, ny1, order="F") + 1

    Tlb2 = np.zeros((nx12 + 1, ny2 + 1), dtype=np.int64)
    Trt2 = np.zeros((nx12 + 1, ny2 + 1), dtype=np.int64)
    if n2_e:
        Tlb2[:nx12, :ny2] = np.arange(n1_e, n_e, 2, dtype=np.int64).reshape(nx12, ny2, order="F") + 1
        Trt2[1:, 1 : ny2 + 1] = np.arange(n1_e + 1, n_e, 2, dtype=np.int64).reshape(nx12, ny2, order="F") + 1
        Trt2[1:, 0] = np.arange(2, 2 * nx12 + 1, 2, dtype=np.int64) + n1_e - 2 * nx

    edge1_el_h = np.vstack((Tlb1[:nx, :].reshape(-1, order="F"), Trt1[1:, :].reshape(-1, order="F")))
    edge2_el_h = np.vstack((Tlb2[:nx12, 1:].reshape(-1, order="F"), Trt2[1:, 1:].reshape(-1, order="F")))
    edge1_el_v = np.vstack((Trt1[:, 1:].reshape(-1, order="F"), Tlb1[:, :ny1].reshape(-1, order="F")))
    edge2_el_v = np.vstack((Trt2[:, 1:].reshape(-1, order="F"), Tlb2[:, :ny2].reshape(-1, order="F")))
    edge1_el_d = np.vstack((Tlb1[:nx, :ny1].reshape(-1, order="F"), Trt1[1:, 1:].reshape(-1, order="F")))
    edge2_el_d = np.vstack((Tlb2[:nx12, :ny2].reshape(-1, order="F"), Trt2[1:, 1:].reshape(-1, order="F")))
    edge_el = np.hstack((edge1_el_h, edge1_el_v, edge1_el_d, edge2_el_h, edge2_el_v, edge2_el_d))

    edge_nodes = np.zeros((3, edge_el.shape[1]), dtype=np.int64)
    for e in range(elem_ed.shape[1]):
        tri = elem[:, e]
        for local_edge, edge_id in enumerate(elem_ed[:, e]):
            if np.any(edge_nodes[:, edge_id] != 0) or (tri[0] == 0 and tri[1] == 0 and tri[2] == 0):
                continue
            if local_edge == 0:
                edge_nodes[:, edge_id] = np.array([tri[0], tri[1], tri[5]], dtype=np.int64)
            elif local_edge == 1:
                edge_nodes[:, edge_id] = np.array([tri[1], tri[2], tri[3]], dtype=np.int64)
            else:
                edge_nodes[:, edge_id] = np.array([tri[0], tri[2], tri[4]], dtype=np.int64)

    boundary = np.any(edge_el == 0, axis=0)
    surf = edge_nodes[:, boundary].T.copy()

    x_max = float(np.max(coord[0, :])) - 1.0e-9
    q_mask = np.zeros((2, coord.shape[1]), dtype=bool)
    q_mask[0, :] = (coord[0, :] > 0.0) & (coord[1, :] > 0.0) & (coord[0, :] < x_max)
    q_mask[1, :] = coord[1, :] > 0.0

    return coord.T.copy(), elem.T.copy(), surf, q_mask.T.copy()


def _expand_triangle_connectivity_to_dofs(elems_scalar: np.ndarray) -> np.ndarray:
    elems_scalar = np.asarray(elems_scalar, dtype=np.int64)
    offsets = np.array([0, 1], dtype=np.int64)
    return (2 * elems_scalar[:, :, None] + offsets[None, None, :]).reshape(elems_scalar.shape[0], -1)


def _assemble_triangle_operators(
    nodes: np.ndarray,
    elems_scalar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xi, wf = _quadrature_volume_p2()
    hatp, dhat1, dhat2 = _local_basis_volume_p2(xi)
    n_elem = elems_scalar.shape[0]
    n_q = xi.shape[1]
    n_p = elems_scalar.shape[1]

    elem_B = np.zeros((n_elem, n_q, 3, 2 * n_p), dtype=np.float64)
    quad_weight = np.zeros((n_elem, n_q), dtype=np.float64)

    for e in range(n_elem):
        x_e = nodes[elems_scalar[e], 0]
        y_e = nodes[elems_scalar[e], 1]
        for q in range(n_q):
            dh1 = dhat1[:, q]
            dh2 = dhat2[:, q]
            j11 = float(np.dot(x_e, dh1))
            j12 = float(np.dot(y_e, dh1))
            j21 = float(np.dot(x_e, dh2))
            j22 = float(np.dot(y_e, dh2))
            det_j = j11 * j22 - j12 * j21
            inv = np.array([[j22, -j12], [-j21, j11]], dtype=np.float64) / det_j
            grads = inv @ np.vstack((dh1, dh2))
            dphix = grads[0, :]
            dphiy = grads[1, :]
            B = np.zeros((3, 2 * n_p), dtype=np.float64)
            for a in range(n_p):
                B[0, 2 * a] = dphix[a]
                B[1, 2 * a + 1] = dphiy[a]
                B[2, 2 * a] = dphiy[a]
                B[2, 2 * a + 1] = dphix[a]
            elem_B[e, q, :, :] = B
            quad_weight[e, q] = abs(det_j) * wf[q]

    return elem_B, quad_weight, hatp


def _assemble_triangle_operators_p1(
    nodes: np.ndarray,
    elems_scalar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xi, wf = _quadrature_volume_p1()
    hatp, dhat1, dhat2 = _local_basis_volume_p1(xi)
    n_elem = elems_scalar.shape[0]
    n_q = xi.shape[1]
    n_p = elems_scalar.shape[1]
    elem_B = np.zeros((n_elem, n_q, 3, 2 * n_p), dtype=np.float64)
    quad_weight = np.zeros((n_elem, n_q), dtype=np.float64)

    for e in range(n_elem):
        x_e = nodes[elems_scalar[e], 0]
        y_e = nodes[elems_scalar[e], 1]
        for q in range(n_q):
            dh1 = dhat1[:, q]
            dh2 = dhat2[:, q]
            j11 = float(np.dot(x_e, dh1))
            j12 = float(np.dot(y_e, dh1))
            j21 = float(np.dot(x_e, dh2))
            j22 = float(np.dot(y_e, dh2))
            det_j = j11 * j22 - j12 * j21
            inv = np.array([[j22, -j12], [-j21, j11]], dtype=np.float64) / det_j
            grads = inv @ np.vstack((dh1, dh2))
            dphix = grads[0, :]
            dphiy = grads[1, :]
            B = np.zeros((3, 2 * n_p), dtype=np.float64)
            for a in range(n_p):
                B[0, 2 * a] = dphix[a]
                B[1, 2 * a + 1] = dphiy[a]
                B[2, 2 * a] = dphiy[a]
                B[2, 2 * a + 1] = dphix[a]
            elem_B[e, q, :, :] = B
            quad_weight[e, q] = abs(det_j) * wf[q]

    return elem_B, quad_weight, hatp


def _assemble_gravity_load(
    nodes: np.ndarray,
    elems_scalar: np.ndarray,
    quad_weight: np.ndarray,
    hatp: np.ndarray,
    gamma: float,
) -> np.ndarray:
    n_nodes = nodes.shape[0]
    n_q = quad_weight.shape[1]
    force = np.zeros(2 * n_nodes, dtype=np.float64)
    body_force = np.array([0.0, -float(gamma)], dtype=np.float64)

    for e, elem in enumerate(elems_scalar):
        f_local = np.zeros(2 * elem.shape[0], dtype=np.float64)
        for q in range(n_q):
            for a, node in enumerate(elem):
                f_local[2 * a : 2 * a + 2] += quad_weight[e, q] * hatp[a, q] * body_force
        dofs = _expand_triangle_connectivity_to_dofs(elem[None, :])[0]
        force[dofs] += f_local
    return force


def _build_free_dofs(q_mask: np.ndarray) -> np.ndarray:
    q_mask = np.asarray(q_mask, dtype=bool)
    n_nodes = q_mask.shape[0]
    dof_ids = np.arange(2 * n_nodes, dtype=np.int64).reshape(n_nodes, 2)
    return dof_ids[q_mask].astype(np.int64)


def _build_dof_adjacency(elems: np.ndarray, freedofs: np.ndarray) -> sp.coo_matrix:
    n_free = int(freedofs.size)
    full_to_free = np.full(int(np.max(elems)) + 1, -1, dtype=np.int64)
    full_to_free[freedofs] = np.arange(n_free, dtype=np.int64)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for elem_dofs in np.asarray(elems, dtype=np.int64):
        local = full_to_free[elem_dofs]
        local = local[local >= 0]
        if local.size == 0:
            continue
        rr = np.repeat(local, local.size)
        cc = np.tile(local, local.size)
        rows.append(rr)
        cols.append(cc)

    if rows:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        data = np.ones(row.size, dtype=np.float64)
    else:
        row = np.empty(0, dtype=np.int64)
        col = np.empty(0, dtype=np.int64)
        data = np.empty(0, dtype=np.float64)

    adjacency = sp.coo_matrix((data, (row, col)), shape=(n_free, n_free))
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    return adjacency


def _point_key(point: np.ndarray) -> tuple[float, float]:
    return tuple(np.round(np.asarray(point, dtype=np.float64), COORD_DECIMALS).tolist())


def _build_q_mask_from_nodes(nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.float64)
    x_max = float(np.max(nodes[:, 0])) - 1.0e-9
    q_mask = np.zeros((nodes.shape[0], 2), dtype=bool)
    q_mask[:, 0] = (
        (nodes[:, 0] > 0.0)
        & (nodes[:, 1] > 0.0)
        & (nodes[:, 0] < x_max)
    )
    q_mask[:, 1] = nodes[:, 1] > 0.0
    return q_mask


def _reference_to_physical_triangle(verts: np.ndarray, ref_point: np.ndarray) -> np.ndarray:
    xi = float(ref_point[0])
    eta = float(ref_point[1])
    l0 = 1.0 - xi - eta
    return (
        l0 * np.asarray(verts[0], dtype=np.float64)
        + xi * np.asarray(verts[1], dtype=np.float64)
        + eta * np.asarray(verts[2], dtype=np.float64)
    )


def _build_macro_mesh_from_case(case_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p2_case = build_case_data(case_name)
    p2_nodes = np.asarray(p2_case.nodes, dtype=np.float64)
    p2_elems = np.asarray(p2_case.elems_scalar[:, :3], dtype=np.int64)
    vertex_ids = np.unique(p2_elems.reshape(-1))
    old_to_new = np.full(p2_nodes.shape[0], -1, dtype=np.int64)
    old_to_new[vertex_ids] = np.arange(vertex_ids.size, dtype=np.int64)
    macro_nodes = p2_nodes[vertex_ids]
    macro_elems = old_to_new[p2_elems]
    macro_surf = old_to_new[np.asarray(p2_case.surf[:, :2], dtype=np.int64)]
    return macro_nodes, macro_elems, macro_surf


def _build_same_mesh_lagrange_connectivity(
    macro_nodes: np.ndarray,
    macro_elems: np.ndarray,
    *,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    element = _triangle_lagrange_element(degree)
    ref_points = np.asarray(element.points, dtype=np.float64)
    node_lookup: dict[tuple[float, float], int] = {}
    node_coords: list[np.ndarray] = []
    elems_scalar = np.zeros((macro_elems.shape[0], ref_points.shape[0]), dtype=np.int64)

    for e, macro_elem in enumerate(np.asarray(macro_elems, dtype=np.int64)):
        verts = np.asarray(macro_nodes[macro_elem], dtype=np.float64)
        for a, ref_point in enumerate(ref_points):
            phys = _reference_to_physical_triangle(verts, ref_point)
            key = _point_key(phys)
            node_id = node_lookup.get(key)
            if node_id is None:
                node_id = len(node_coords)
                node_lookup[key] = node_id
                node_coords.append(np.asarray(phys, dtype=np.float64))
            elems_scalar[e, a] = int(node_id)

    nodes = np.vstack(node_coords).astype(np.float64)
    return nodes, elems_scalar


def _plane_strain_constitutive(E: float, nu: float) -> np.ndarray:
    lam = float(E) * float(nu) / ((1.0 + float(nu)) * (1.0 - 2.0 * float(nu)))
    mu = float(E) / (2.0 * (1.0 + float(nu)))
    return np.array(
        [
            [lam + 2.0 * mu, lam, 0.0],
            [lam, lam + 2.0 * mu, 0.0],
            [0.0, 0.0, mu],
        ],
        dtype=np.float64,
    )


def assemble_plane_strain_stiffness(
    elem_B: np.ndarray,
    quad_weight: np.ndarray,
    elems: np.ndarray,
    E: float,
    nu: float,
) -> sp.csr_matrix:
    C = _plane_strain_constitutive(E, nu)
    n_dofs = int(np.max(elems)) + 1
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []

    for e, elem_dofs in enumerate(np.asarray(elems, dtype=np.int64)):
        k_local = np.zeros((elem_dofs.size, elem_dofs.size), dtype=np.float64)
        for q in range(quad_weight.shape[1]):
            Bq = elem_B[e, q, :, :]
            k_local += quad_weight[e, q] * (Bq.T @ C @ Bq)
        rr = np.repeat(elem_dofs, elem_dofs.size)
        cc = np.tile(elem_dofs, elem_dofs.size)
        rows.append(rr)
        cols.append(cc)
        data.append(k_local.reshape(-1))

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    val = np.concatenate(data)
    K = sp.coo_matrix((val, (row, col)), shape=(n_dofs, n_dofs)).tocsr()
    K.sum_duplicates()
    K = 0.5 * (K + K.T)
    return K.tocsr()


def build_elastic_initial_guess(
    *,
    elems: np.ndarray,
    elem_B: np.ndarray,
    quad_weight: np.ndarray,
    force: np.ndarray,
    freedofs: np.ndarray,
    E: float,
    nu: float,
) -> np.ndarray:
    K = assemble_plane_strain_stiffness(elem_B, quad_weight, elems, E, nu)
    K_free = K[freedofs][:, freedofs].tocsr()
    f_free = np.asarray(force[freedofs], dtype=np.float64)
    u_free = spla.spsolve(K_free, f_free)
    return np.asarray(u_free, dtype=np.float64)


def build_near_nullspace_modes(nodes: np.ndarray, freedofs: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.float64)
    n_nodes = nodes.shape[0]
    rigid = np.zeros((2 * n_nodes, 3), dtype=np.float64)
    rigid[0::2, 0] = 1.0
    rigid[1::2, 1] = 1.0
    center = np.mean(nodes, axis=0)
    x = nodes[:, 0] - center[0]
    y = nodes[:, 1] - center[1]
    rigid[0::2, 2] = -y
    rigid[1::2, 2] = x
    return rigid[np.asarray(freedofs, dtype=np.int64), :]


def build_case_data(case_name: str = DEFAULT_CASE) -> SlopeStabilityCaseData:
    case_name = str(case_name)
    level = level_for_case_name(case_name)
    h = h_for_level(level)
    x1 = 15.0
    x2 = 10.0
    x3 = 15.0
    y1 = 10.0
    y2 = 10.0
    beta_deg = 45.0
    E = 40000.0
    nu = 0.30
    c0 = 6.0
    phi_deg = 45.0
    psi_deg = 0.0
    gamma = 20.0
    lambda_target_default = 1.21

    nodes, elems_scalar, surf, q_mask = _generate_homogeneous_slope_mesh_p2(
        h=h,
        x1=x1,
        x2=x2,
        x3=x3,
        y1=y1,
        y2=y2,
    )
    elems = _expand_triangle_connectivity_to_dofs(elems_scalar)
    elem_B, quad_weight, hatp = _assemble_triangle_operators(nodes, elems_scalar)
    force = _assemble_gravity_load(nodes, elems_scalar, quad_weight, hatp, gamma)
    freedofs = _build_free_dofs(q_mask)
    adjacency = _build_dof_adjacency(elems, freedofs)

    return SlopeStabilityCaseData(
        case_name=case_name,
        level=level,
        nodes=nodes,
        elems_scalar=elems_scalar,
        elems=elems,
        surf=surf,
        q_mask=q_mask,
        freedofs=freedofs,
        elem_B=elem_B,
        quad_weight=quad_weight,
        force=force,
        u_0=np.zeros(2 * nodes.shape[0], dtype=np.float64),
        eps_p_old=np.zeros((elems_scalar.shape[0], quad_weight.shape[1], 3), dtype=np.float64),
        adjacency=adjacency,
        h=h,
        x1=x1,
        x2=x2,
        x3=x3,
        y1=y1,
        y2=y2,
        beta_deg=beta_deg,
        E=E,
        nu=nu,
        c0=c0,
        phi_deg=phi_deg,
        psi_deg=psi_deg,
        gamma=gamma,
        davis_type="B",
        lambda_target_default=lambda_target_default,
    )


def build_refined_p1_case_data(case_name: str = DEFAULT_CASE) -> SlopeStabilityCaseData:
    """Refine each P2 triangle into four P1 triangles on the same node set."""

    p2_case = build_case_data(case_name)
    base_elem = np.asarray(p2_case.elems_scalar, dtype=np.int64)
    refined_elem = np.vstack(
        (
            base_elem[:, [0, 5, 4]],
            base_elem[:, [5, 1, 3]],
            base_elem[:, [4, 3, 2]],
            base_elem[:, [5, 3, 4]],
        )
    )
    surf = np.vstack(
        (
            np.asarray(p2_case.surf, dtype=np.int64)[:, [0, 2]],
            np.asarray(p2_case.surf, dtype=np.int64)[:, [2, 1]],
        )
    )
    elems = _expand_triangle_connectivity_to_dofs(refined_elem)
    elem_B, quad_weight, hatp = _assemble_triangle_operators_p1(
        np.asarray(p2_case.nodes, dtype=np.float64),
        refined_elem,
    )
    force = _assemble_gravity_load(
        np.asarray(p2_case.nodes, dtype=np.float64),
        refined_elem,
        quad_weight,
        hatp,
        float(p2_case.gamma),
    )
    freedofs = np.asarray(p2_case.freedofs, dtype=np.int64)
    adjacency = _build_dof_adjacency(elems, freedofs)
    return SlopeStabilityCaseData(
        case_name=f"{canonical_case_name(case_name)}_refined_p1",
        level=int(p2_case.level),
        nodes=np.asarray(p2_case.nodes, dtype=np.float64),
        elems_scalar=refined_elem,
        elems=elems,
        surf=surf,
        q_mask=np.asarray(p2_case.q_mask, dtype=bool),
        freedofs=freedofs,
        elem_B=elem_B,
        quad_weight=quad_weight,
        force=force,
        u_0=np.asarray(p2_case.u_0, dtype=np.float64),
        eps_p_old=np.zeros((refined_elem.shape[0], quad_weight.shape[1], 3), dtype=np.float64),
        adjacency=adjacency,
        h=float(p2_case.h),
        x1=float(p2_case.x1),
        x2=float(p2_case.x2),
        x3=float(p2_case.x3),
        y1=float(p2_case.y1),
        y2=float(p2_case.y2),
        beta_deg=float(p2_case.beta_deg),
        E=float(p2_case.E),
        nu=float(p2_case.nu),
        c0=float(p2_case.c0),
        phi_deg=float(p2_case.phi_deg),
        psi_deg=float(p2_case.psi_deg),
        gamma=float(p2_case.gamma),
        davis_type=str(p2_case.davis_type),
        lambda_target_default=float(p2_case.lambda_target_default),
    )


def _broadcast_case_data(
    builder,
    *,
    build_mode: str,
    comm: MPI.Comm | None,
):
    build_mode = str(build_mode)
    if build_mode not in {"replicated", "root_bcast"}:
        raise ValueError(f"Unsupported case-data build mode {build_mode!r}")
    if comm is None or int(comm.size) <= 1 or build_mode == "replicated":
        return builder()
    case_data = builder() if int(comm.rank) == 0 else None
    return comm.bcast(case_data, root=0)


def build_same_mesh_lagrange_case_data(
    case_name: str = DEFAULT_CASE,
    *,
    degree: int,
    build_mode: str = "replicated",
    comm: MPI.Comm | None = None,
) -> SlopeStabilityCaseData:
    canonical = canonical_case_name(case_name)
    level = level_for_case_name(canonical)
    asset_path = same_mesh_case_hdf5_path(level, int(degree))

    # Large same-mesh level assets can exceed practical pickle/bcast limits on MPI.
    # When a canonical HDF5 asset exists, load it directly on each rank instead of
    # root-building a Python object and broadcasting that object verbatim.
    if (
        asset_path.exists()
        and build_mode == "root_bcast"
        and comm is not None
        and int(comm.size) > 1
    ):
        return load_case_hdf5(asset_path)

    return _broadcast_case_data(
        lambda: (
            load_case_hdf5(asset_path)
            if asset_path.exists()
            else _build_same_mesh_lagrange_case_data_impl(canonical, degree=degree)
        ),
        build_mode=build_mode,
        comm=comm,
    )


def _build_same_mesh_lagrange_case_data_impl(
    case_name: str = DEFAULT_CASE,
    *,
    degree: int,
) -> SlopeStabilityCaseData:
    degree = int(degree)
    if degree not in {1, 2, 4}:
        raise ValueError(f"Unsupported same-mesh Lagrange degree {degree!r}; expected 1, 2, or 4")
    if degree == 2:
        case = build_case_data(case_name)
        return SlopeStabilityCaseData(
            case_name=f"{canonical_case_name(case_name)}_p2_same_mesh",
            level=int(case.level),
            nodes=np.asarray(case.nodes, dtype=np.float64),
            elems_scalar=np.asarray(case.elems_scalar, dtype=np.int64),
            elems=np.asarray(case.elems, dtype=np.int64),
            surf=np.asarray(case.surf, dtype=np.int64),
            q_mask=np.asarray(case.q_mask, dtype=bool),
            freedofs=np.asarray(case.freedofs, dtype=np.int64),
            elem_B=np.asarray(case.elem_B, dtype=np.float64),
            quad_weight=np.asarray(case.quad_weight, dtype=np.float64),
            force=np.asarray(case.force, dtype=np.float64),
            u_0=np.asarray(case.u_0, dtype=np.float64),
            eps_p_old=np.asarray(case.eps_p_old, dtype=np.float64),
            adjacency=case.adjacency.copy(),
            h=float(case.h),
            x1=float(case.x1),
            x2=float(case.x2),
            x3=float(case.x3),
            y1=float(case.y1),
            y2=float(case.y2),
            beta_deg=float(case.beta_deg),
            E=float(case.E),
            nu=float(case.nu),
            c0=float(case.c0),
            phi_deg=float(case.phi_deg),
            psi_deg=float(case.psi_deg),
            gamma=float(case.gamma),
            davis_type=str(case.davis_type),
            lambda_target_default=float(case.lambda_target_default),
        )

    p2_case = build_case_data(case_name)
    macro_nodes, macro_elems, macro_surf = _build_macro_mesh_from_case(case_name)
    nodes, elems_scalar = _build_same_mesh_lagrange_connectivity(
        macro_nodes,
        macro_elems,
        degree=degree,
    )
    surf = np.asarray(macro_surf, dtype=np.int64)
    q_mask = _build_q_mask_from_nodes(nodes)
    elems = _expand_triangle_connectivity_to_dofs(elems_scalar)
    elem_B, quad_weight, hatp = _assemble_triangle_operators_lagrange(
        nodes,
        elems_scalar,
        degree=degree,
    )
    force = _assemble_gravity_load(
        nodes,
        elems_scalar,
        quad_weight,
        hatp,
        float(p2_case.gamma),
    )
    freedofs = _build_free_dofs(q_mask)
    adjacency = _build_dof_adjacency(elems, freedofs)
    return SlopeStabilityCaseData(
        case_name=f"{canonical_case_name(case_name)}_p{degree}_same_mesh",
        level=int(p2_case.level),
        nodes=nodes,
        elems_scalar=elems_scalar,
        elems=elems,
        surf=surf,
        q_mask=q_mask,
        freedofs=freedofs,
        elem_B=elem_B,
        quad_weight=quad_weight,
        force=force,
        u_0=np.zeros(2 * nodes.shape[0], dtype=np.float64),
        eps_p_old=np.zeros((elems_scalar.shape[0], quad_weight.shape[1], 3), dtype=np.float64),
        adjacency=adjacency,
        h=float(p2_case.h),
        x1=float(p2_case.x1),
        x2=float(p2_case.x2),
        x3=float(p2_case.x3),
        y1=float(p2_case.y1),
        y2=float(p2_case.y2),
        beta_deg=float(p2_case.beta_deg),
        E=float(p2_case.E),
        nu=float(p2_case.nu),
        c0=float(p2_case.c0),
        phi_deg=float(p2_case.phi_deg),
        psi_deg=float(p2_case.psi_deg),
        gamma=float(p2_case.gamma),
        davis_type=str(p2_case.davis_type),
        lambda_target_default=float(p2_case.lambda_target_default),
    )


def _mesh_data_path(case_name: str) -> Path:
    return MESH_DATA_ROOT / "SlopeStability" / f"{case_name}.h5"


def same_mesh_case_hdf5_path(level: int, degree: int) -> Path:
    return _mesh_data_path(same_mesh_case_name(int(level), int(degree)))


_SAME_MESH_LIGHT_FIELDS = (
    "case_name",
    "level",
    "nodes",
    "elems_scalar",
    "elems",
    "surf",
    "q_mask",
    "freedofs",
    "force",
    "u_0",
    "h",
    "x1",
    "x2",
    "x3",
    "y1",
    "y2",
    "beta_deg",
    "E",
    "nu",
    "c0",
    "phi_deg",
    "psi_deg",
    "gamma",
    "davis_type",
    "lambda_target_default",
)


def _decode_hdf5_string(value: object) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return str(value)


def load_case_hdf5_fields(
    path: str | Path,
    *,
    fields: list[str] | tuple[str, ...] | set[str],
    load_adjacency: bool = False,
) -> tuple[dict[str, object], sp.coo_matrix | None]:
    raw, adjacency = load_problem_hdf5_fields(
        str(path),
        fields=fields,
        load_adjacency=bool(load_adjacency),
    )
    for key in ("case_name", "davis_type"):
        if key in raw:
            raw[key] = _decode_hdf5_string(raw[key])
    return raw, adjacency


def load_same_mesh_case_hdf5_light(
    level: int,
    degree: int,
) -> dict[str, object]:
    path = same_mesh_case_hdf5_path(int(level), int(degree))
    raw, _ = load_case_hdf5_fields(path, fields=_SAME_MESH_LIGHT_FIELDS, load_adjacency=False)
    raw["elem_type"] = f"P{int(degree)}"
    raw["element_degree"] = int(degree)
    return raw


def load_same_mesh_case_hdf5_rank_local(
    level: int,
    degree: int,
    *,
    reorder_mode: str,
    comm: MPI.Comm,
    block_size: int = 2,
) -> dict[str, object]:
    degree = int(degree)
    path = same_mesh_case_hdf5_path(int(level), degree)
    raw, _ = load_case_hdf5_fields(path, fields=_SAME_MESH_LIGHT_FIELDS, load_adjacency=False)

    nodes = np.asarray(raw["nodes"], dtype=np.float64)
    elems = np.asarray(raw["elems"], dtype=np.int64)
    freedofs = np.asarray(raw["freedofs"], dtype=np.int64)
    u_0 = np.asarray(raw["u_0"], dtype=np.float64)
    n_free = int(freedofs.size)

    perm = select_permutation(
        str(reorder_mode),
        adjacency=None,
        coords_all=nodes,
        freedofs=freedofs,
        n_parts=int(comm.size),
        block_size=int(block_size),
    )
    iperm = inverse_permutation(np.asarray(perm, dtype=np.int64))
    lo, hi = petsc_ownership_range(
        n_free,
        int(comm.rank),
        int(comm.size),
        block_size=int(block_size),
    )

    total_to_free_orig = np.full(len(u_0), -1, dtype=np.int64)
    total_to_free_orig[freedofs] = np.arange(n_free, dtype=np.int64)
    total_to_free_reord = np.full(len(u_0), -1, dtype=np.int64)
    free_mask = total_to_free_orig >= 0
    total_to_free_reord[free_mask] = iperm[total_to_free_orig[free_mask]]

    elems_reordered = total_to_free_reord[elems]
    local_elem_mask = np.any((elems_reordered >= lo) & (elems_reordered < hi), axis=1)
    local_elem_idx = np.where(local_elem_mask)[0].astype(np.int64)

    with h5py.File(path, "r") as handle:
        local_elem_B = np.asarray(handle["elem_B"][local_elem_idx], dtype=np.float64)
        local_quad_weight = np.asarray(handle["quad_weight"][local_elem_idx], dtype=np.float64)
        local_eps_p_old = np.asarray(handle["eps_p_old"][local_elem_idx], dtype=np.float64)

    raw["case_name"] = _decode_hdf5_string(raw["case_name"])
    raw["davis_type"] = _decode_hdf5_string(raw["davis_type"])
    raw["elem_type"] = f"P{degree}"
    raw["element_degree"] = degree
    raw["_distributed_local_elem_idx"] = local_elem_idx
    raw["_distributed_elem_B"] = local_elem_B
    raw["_distributed_quad_weight"] = local_quad_weight
    raw["_distributed_eps_p_old"] = local_eps_p_old
    raw["_distributed_local_data"] = True
    raw["_distributed_perm"] = np.asarray(perm, dtype=np.int64)
    return raw


def load_case_hdf5(path: str | Path) -> SlopeStabilityCaseData:
    raw, adjacency = load_problem_hdf5(str(path))
    if adjacency is None:
        raise RuntimeError(f"Slope-stability HDF5 snapshot {path!s} is missing adjacency data")

    def _string_value(key: str) -> str:
        value = raw[key]
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8")
        return str(value)

    return SlopeStabilityCaseData(
        case_name=_string_value("case_name"),
        level=int(raw["level"]),
        nodes=np.asarray(raw["nodes"], dtype=np.float64),
        elems_scalar=np.asarray(raw["elems_scalar"], dtype=np.int64),
        elems=np.asarray(raw["elems"], dtype=np.int64),
        surf=np.asarray(raw["surf"], dtype=np.int64),
        q_mask=np.asarray(raw["q_mask"], dtype=bool),
        freedofs=np.asarray(raw["freedofs"], dtype=np.int64),
        elem_B=np.asarray(raw["elem_B"], dtype=np.float64),
        quad_weight=np.asarray(raw["quad_weight"], dtype=np.float64),
        force=np.asarray(raw["force"], dtype=np.float64),
        u_0=np.asarray(raw["u_0"], dtype=np.float64),
        eps_p_old=np.asarray(raw["eps_p_old"], dtype=np.float64),
        adjacency=adjacency.tocoo(),
        h=float(raw["h"]),
        x1=float(raw["x1"]),
        x2=float(raw["x2"]),
        x3=float(raw["x3"]),
        y1=float(raw["y1"]),
        y2=float(raw["y2"]),
        beta_deg=float(raw["beta_deg"]),
        E=float(raw["E"]),
        nu=float(raw["nu"]),
        c0=float(raw["c0"]),
        phi_deg=float(raw["phi_deg"]),
        psi_deg=float(raw["psi_deg"]),
        gamma=float(raw["gamma"]),
        davis_type=_string_value("davis_type"),
        lambda_target_default=float(raw["lambda_target_default"]),
    )


def write_case_hdf5(
    path: str | Path,
    case_data: SlopeStabilityCaseData,
    *,
    include_u_init_free: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    u_init_free = None
    if include_u_init_free:
        u_init_free = build_elastic_initial_guess(
            elems=case_data.elems,
            elem_B=case_data.elem_B,
            quad_weight=case_data.quad_weight,
            force=case_data.force,
            freedofs=case_data.freedofs,
            E=case_data.E,
            nu=case_data.nu,
        )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("nodes", data=case_data.nodes)
        handle.create_dataset("elems_scalar", data=case_data.elems_scalar)
        handle.create_dataset("elems", data=case_data.elems)
        handle.create_dataset("surf", data=case_data.surf)
        handle.create_dataset("q_mask", data=case_data.q_mask.astype(np.uint8))
        handle.create_dataset("freedofs", data=case_data.freedofs)
        handle.create_dataset("elem_B", data=case_data.elem_B)
        handle.create_dataset("quad_weight", data=case_data.quad_weight)
        handle.create_dataset("force", data=case_data.force)
        handle.create_dataset("u_0", data=case_data.u_0)
        if u_init_free is not None:
            handle.create_dataset("u_init_free", data=u_init_free)
        handle.create_dataset("eps_p_old", data=case_data.eps_p_old)
        handle.create_dataset("level", data=int(case_data.level))
        handle.create_dataset("h", data=float(case_data.h))
        handle.create_dataset("x1", data=float(case_data.x1))
        handle.create_dataset("x2", data=float(case_data.x2))
        handle.create_dataset("x3", data=float(case_data.x3))
        handle.create_dataset("y1", data=float(case_data.y1))
        handle.create_dataset("y2", data=float(case_data.y2))
        handle.create_dataset("beta_deg", data=float(case_data.beta_deg))
        handle.create_dataset("E", data=float(case_data.E))
        handle.create_dataset("nu", data=float(case_data.nu))
        handle.create_dataset("c0", data=float(case_data.c0))
        handle.create_dataset("phi_deg", data=float(case_data.phi_deg))
        handle.create_dataset("psi_deg", data=float(case_data.psi_deg))
        handle.create_dataset("gamma", data=float(case_data.gamma))
        handle.create_dataset("lambda_target_default", data=float(case_data.lambda_target_default))
        handle.create_dataset("case_name", data=np.bytes_(case_data.case_name))
        handle.create_dataset("davis_type", data=np.bytes_(case_data.davis_type))
        grp = handle.create_group("adjacency")
        adjacency = case_data.adjacency.tocoo()
        grp.create_dataset("data", data=adjacency.data)
        grp.create_dataset("row", data=adjacency.row)
        grp.create_dataset("col", data=adjacency.col)
        grp.create_dataset("shape", data=np.asarray(adjacency.shape, dtype=np.int64))


def level_case_hdf5_path(level: int) -> Path:
    return _mesh_data_path(case_name_for_level(level))


def default_case_hdf5_path() -> Path:
    return _mesh_data_path(DEFAULT_CASE)


def ensure_level_case_hdf5(level: int) -> Path:
    path = level_case_hdf5_path(level)
    needs_rewrite = False
    if path.exists():
        with h5py.File(path, "r") as handle:
            needs_rewrite = "u_init_free" not in handle
    if not path.exists() or needs_rewrite:
        write_case_hdf5(path, build_case_data(case_name_for_level(level)))
    return path


def ensure_default_case_hdf5() -> Path:
    path = default_case_hdf5_path()
    needs_rewrite = False
    if path.exists():
        with h5py.File(path, "r") as handle:
            needs_rewrite = "u_init_free" not in handle
    if not path.exists() or needs_rewrite:
        write_case_hdf5(path, build_case_data(DEFAULT_CASE))
    return path


def ensure_all_level_case_hdf5() -> list[Path]:
    paths = [ensure_level_case_hdf5(level) for level in supported_levels()]
    paths.append(ensure_default_case_hdf5())
    return paths


def ensure_same_mesh_case_hdf5(level: int, degree: int) -> Path:
    level = int(level)
    degree = int(degree)
    path = same_mesh_case_hdf5_path(level, degree)
    if not path.exists():
        write_case_hdf5(
            path,
            _build_same_mesh_lagrange_case_data_impl(case_name_for_level(level), degree=degree),
            include_u_init_free=False,
        )
    return path

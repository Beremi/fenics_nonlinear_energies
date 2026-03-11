from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


def _logit(x: np.ndarray) -> np.ndarray:
    return np.log(x / (1.0 - x))


def _sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _theta_from_latent_numpy(z: np.ndarray, theta_min: float) -> np.ndarray:
    return theta_min + (1.0 - theta_min) * _sigmoid_numpy(z)


def _build_free_adjacency(elems: np.ndarray, freedofs: np.ndarray, n_full: int) -> sp.coo_matrix:
    n_free = int(freedofs.size)
    free_index = -np.ones(n_full, dtype=np.int32)
    free_index[freedofs] = np.arange(n_free, dtype=np.int32)

    row_parts = [np.arange(n_free, dtype=np.int32)]
    col_parts = [np.arange(n_free, dtype=np.int32)]

    for conn in elems:
        free_conn = free_index[conn]
        free_conn = free_conn[free_conn >= 0]
        if free_conn.size == 0:
            continue
        row_parts.append(np.repeat(free_conn, free_conn.size))
        col_parts.append(np.tile(free_conn, free_conn.size))

    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    data = np.ones(rows.shape[0], dtype=np.float64)
    adjacency = sp.coo_matrix((data, (rows, cols)), shape=(n_free, n_free))
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    return adjacency


@dataclass
class CantileverTopologyMesh:
    nx: int = 24
    ny: int = 12
    length: float = 2.0
    height: float = 1.0
    traction: float = 1.0
    load_fraction: float = 0.2
    fixed_pad_cells: int = 2
    load_pad_cells: int = 2

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 2:
            raise ValueError("Need at least a 2x2 cell mesh.")
        if not (0.0 < self.load_fraction <= 1.0):
            raise ValueError("load_fraction must lie in (0, 1].")

        self.hx = self.length / float(self.nx)
        self.hy = self.height / float(self.ny)

        self.coords = self._build_nodes()
        self.scalar_elems = self._build_triangles()
        self.vector_elems = self._expand_vector_connectivity(self.scalar_elems)
        self.elem_area, self.elem_grad_phi, self.elem_B = self._compute_element_geometry()

        self.domain_area = float(np.sum(self.elem_area))
        self.n_nodes = int(self.coords.shape[0])
        self.n_disp_dofs = 2 * self.n_nodes

        self.u_0 = np.zeros(self.n_disp_dofs, dtype=np.float64)
        self.force = self._build_force_vector()
        self.freedofs_u = self._build_displacement_freedofs()
        self.elastic_kernel = self._build_elastic_kernel()

        self.fixed_design_mask = self._build_fixed_design_mask()
        self.freedofs_z = np.flatnonzero(~self.fixed_design_mask).astype(np.int32)
        if self.freedofs_z.size == 0:
            raise ValueError("No free design DOFs remain; reduce the solid pad sizes or refine the mesh.")
        self.nodal_volume_weights = self._build_nodal_volume_weights()

        self.adjacency_u = _build_free_adjacency(
            self.vector_elems, self.freedofs_u, self.n_disp_dofs
        )
        self.adjacency_z = _build_free_adjacency(
            self.scalar_elems, self.freedofs_z, self.n_nodes
        )

    def _node_id(self, ix: int, iy: int) -> int:
        return iy * (self.nx + 1) + ix

    def _build_nodes(self) -> np.ndarray:
        xs = np.linspace(0.0, self.length, self.nx + 1)
        ys = np.linspace(0.0, self.height, self.ny + 1)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        return np.column_stack((grid_x.ravel(), grid_y.ravel()))

    def _build_triangles(self) -> np.ndarray:
        triangles = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                n00 = self._node_id(ix, iy)
                n10 = self._node_id(ix + 1, iy)
                n01 = self._node_id(ix, iy + 1)
                n11 = self._node_id(ix + 1, iy + 1)
                if (ix + iy) % 2 == 0:
                    triangles.append((n00, n10, n11))
                    triangles.append((n00, n11, n01))
                else:
                    triangles.append((n00, n10, n01))
                    triangles.append((n10, n11, n01))
        return np.asarray(triangles, dtype=np.int32)

    def _expand_vector_connectivity(self, elems: np.ndarray) -> np.ndarray:
        vector_elems = np.empty((elems.shape[0], 2 * elems.shape[1]), dtype=np.int32)
        for local_node in range(elems.shape[1]):
            vector_elems[:, 2 * local_node] = 2 * elems[:, local_node]
            vector_elems[:, 2 * local_node + 1] = 2 * elems[:, local_node] + 1
        return vector_elems

    def _compute_element_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xy = self.coords[self.scalar_elems]
        x0 = xy[:, 0, 0]
        y0 = xy[:, 0, 1]
        x1 = xy[:, 1, 0]
        y1 = xy[:, 1, 1]
        x2 = xy[:, 2, 0]
        y2 = xy[:, 2, 1]

        twice_area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if np.any(np.isclose(twice_area, 0.0)):
            raise ValueError("Degenerate triangle found in generated mesh.")

        elem_area = 0.5 * np.abs(twice_area)
        grad_x = np.column_stack((y1 - y2, y2 - y0, y0 - y1)) / twice_area[:, None]
        grad_y = np.column_stack((x2 - x1, x0 - x2, x1 - x0)) / twice_area[:, None]
        elem_grad_phi = np.stack((grad_x, grad_y), axis=2)

        elem_B = np.zeros((self.scalar_elems.shape[0], 3, 6), dtype=np.float64)
        for local_node in range(3):
            dphix = elem_grad_phi[:, local_node, 0]
            dphiy = elem_grad_phi[:, local_node, 1]
            elem_B[:, 0, 2 * local_node] = dphix
            elem_B[:, 1, 2 * local_node + 1] = dphiy
            elem_B[:, 2, 2 * local_node] = dphiy
            elem_B[:, 2, 2 * local_node + 1] = dphix

        return elem_area, elem_grad_phi, elem_B

    def _build_force_vector(self) -> np.ndarray:
        force = np.zeros(self.n_disp_dofs, dtype=np.float64)
        load_center = 0.5 * self.height
        load_half_height = 0.5 * self.load_fraction * self.height
        load_min = load_center - load_half_height - 1e-12
        load_max = load_center + load_half_height + 1e-12
        traction_vec = np.array([0.0, -abs(self.traction)], dtype=np.float64)
        any_overlap = False

        for iy in range(self.ny):
            n0 = self._node_id(self.nx, iy)
            n1 = self._node_id(self.nx, iy + 1)
            y0 = float(self.coords[n0, 1])
            y1 = float(self.coords[n1, 1])
            overlap = max(0.0, min(max(y0, y1), load_max) - max(min(y0, y1), load_min))
            if overlap <= 0.0:
                continue
            any_overlap = True
            nodal_load = 0.5 * overlap * traction_vec
            force[2 * n0: 2 * n0 + 2] += nodal_load
            force[2 * n1: 2 * n1 + 2] += nodal_load

        if not any_overlap:
            raise ValueError("The selected load patch does not hit any boundary edge.")
        return force

    def _build_displacement_freedofs(self) -> np.ndarray:
        left_nodes = np.flatnonzero(np.isclose(self.coords[:, 0], 0.0))
        fixed = np.empty(2 * left_nodes.size, dtype=np.int32)
        fixed[0::2] = 2 * left_nodes
        fixed[1::2] = 2 * left_nodes + 1
        all_dofs = np.arange(self.n_disp_dofs, dtype=np.int32)
        mask = np.ones(self.n_disp_dofs, dtype=bool)
        mask[fixed] = False
        return all_dofs[mask]

    def _build_elastic_kernel(self) -> np.ndarray:
        rigid = np.zeros((self.n_disp_dofs, 3), dtype=np.float64)
        rigid[0::2, 0] = 1.0
        rigid[1::2, 1] = 1.0
        rigid[0::2, 2] = -self.coords[:, 1]
        rigid[1::2, 2] = self.coords[:, 0]
        return rigid[self.freedofs_u]

    def _load_patch_bounds(self) -> tuple[float, float]:
        load_center = 0.5 * self.height
        load_half_height = 0.5 * self.load_fraction * self.height
        return load_center - load_half_height, load_center + load_half_height

    def _build_fixed_design_mask(self) -> np.ndarray:
        x = self.coords[:, 0]
        y = self.coords[:, 1]
        load_min, load_max = self._load_patch_bounds()
        pad_x = self.fixed_pad_cells * self.hx + 1e-12
        pad_y = self.load_pad_cells * self.hy + 1e-12

        left_solid = x <= pad_x
        right_load_pad = (
            (x >= self.length - pad_x)
            & (y >= load_min - pad_y)
            & (y <= load_max + pad_y)
        )
        return left_solid | right_load_pad

    def _build_nodal_volume_weights(self) -> np.ndarray:
        weights = np.zeros(self.n_nodes, dtype=np.float64)
        contrib = self.elem_area / 3.0
        for local_node in range(3):
            np.add.at(weights, self.scalar_elems[:, local_node], contrib)
        return weights

    def expand_u(self, u_free: np.ndarray) -> np.ndarray:
        u_full = self.u_0.copy()
        u_full[self.freedofs_u] = np.asarray(u_free, dtype=np.float64)
        return u_full

    def build_design_state(
        self,
        target_volume_fraction: float,
        theta_min: float,
        solid_latent: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not (0.0 < target_volume_fraction < 1.0):
            raise ValueError("target_volume_fraction must lie in (0, 1).")
        if not (0.0 < theta_min < 1.0):
            raise ValueError("theta_min must lie in (0, 1).")

        z_template = np.zeros(self.n_nodes, dtype=np.float64)
        z_template[self.fixed_design_mask] = solid_latent

        theta_solid = float(_theta_from_latent_numpy(np.array([solid_latent]), theta_min)[0])
        fixed_volume = float(
            theta_solid * np.sum(self.nodal_volume_weights[self.fixed_design_mask])
        )
        free_weight = float(np.sum(self.nodal_volume_weights[~self.fixed_design_mask]))
        min_volume = fixed_volume + theta_min * free_weight
        if target_volume_fraction * self.domain_area < min_volume - 1e-12:
            raise ValueError(
                "Target volume fraction is below the minimum possible volume once fixed solid "
                "regions and theta_min are enforced."
            )

        theta_free = (target_volume_fraction * self.domain_area - fixed_volume) / free_weight
        theta_free = float(np.clip(theta_free, theta_min + 1e-6, 1.0 - 1e-6))
        latent_free = float(
            _logit(np.array([(theta_free - theta_min) / (1.0 - theta_min)], dtype=np.float64))[0]
        )

        z_init = z_template.copy()
        z_init[self.freedofs_z] = latent_free
        z_free_init = z_init[self.freedofs_z]
        return z_template, z_init, z_free_init

    def expand_z(self, z_free: np.ndarray, z_template: np.ndarray) -> np.ndarray:
        z_full = np.asarray(z_template, dtype=np.float64).copy()
        z_full[self.freedofs_z] = np.asarray(z_free, dtype=np.float64)
        return z_full

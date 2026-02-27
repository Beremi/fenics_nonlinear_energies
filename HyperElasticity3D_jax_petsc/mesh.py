import h5py
import numpy as np
import scipy.sparse as sp


def expand_tet_connectivity_to_dofs(elems2nodes):
    """Expand scalar-node tetra connectivity to flat 3-DOF connectivity.

    scalar tet: [n0, n1, n2, n3]
    dof tet:    [3*n0,3*n0+1,3*n0+2, ..., 3*n3+2]
    """
    elems2nodes = np.asarray(elems2nodes, dtype=np.int64)
    dof_offsets = np.arange(3, dtype=np.int64)
    return (3 * elems2nodes[:, :, None] + dof_offsets[None, None, :]).reshape(
        elems2nodes.shape[0], 12
    )


class MeshHyperElasticity3D:
    """Load HyperElasticity 3D mesh/problem data from HDF5."""

    def __init__(self, mesh_level):
        self.mesh_level = int(mesh_level)
        self.filename = (
            f"mesh_data/HyperElasticity/HyperElasticity_level{self.mesh_level}.h5"
        )
        self._load_data(self.filename)
        self._build_problem_data()

    def _load_data(self, filename):
        self._raw = {}
        self.adjacency = None
        with h5py.File(filename, "r") as f:
            for key in f:
                if key == "adjacency":
                    grp = f[key]
                    data = grp["data"][:]
                    row = grp["row"][:]
                    col = grp["col"][:]
                    shape = tuple(grp["shape"][:])
                    self.adjacency = sp.coo_matrix((data, (row, col)), shape=shape)
                else:
                    if f[key].shape == ():
                        self._raw[key] = f[key][()]
                    else:
                        self._raw[key] = f[key][:]

        if self.adjacency is None:
            raise RuntimeError("Mesh file is missing required 'adjacency' group")

    def _build_problem_data(self):
        nodes2coord = np.asarray(self._raw["nodes2coord"], dtype=np.float64)
        elems_scalar = np.asarray(self._raw["elems2nodes"], dtype=np.int64)
        elems_dof = expand_tet_connectivity_to_dofs(elems_scalar)

        u0_ref = np.asarray(self._raw["u0"], dtype=np.float64).ravel()
        freedofs = np.asarray(self._raw["dofsMinim"], dtype=np.int64).ravel()

        right_x = np.max(nodes2coord[:, 0])
        right_nodes = np.where(np.isclose(nodes2coord[:, 0], right_x))[0].astype(np.int64)

        # Rigid-body near-nullspace in full DOF space, then restricted to free DOFs.
        n_nodes = nodes2coord.shape[0]
        rigid_modes = np.zeros((3 * n_nodes, 6), dtype=np.float64)
        rigid_modes[0::3, 0] = 1.0
        rigid_modes[1::3, 1] = 1.0
        rigid_modes[2::3, 2] = 1.0

        rigid_modes[1::3, 3] = -nodes2coord[:, 2]
        rigid_modes[2::3, 3] = nodes2coord[:, 1]

        rigid_modes[0::3, 4] = nodes2coord[:, 2]
        rigid_modes[2::3, 4] = -nodes2coord[:, 0]

        rigid_modes[0::3, 5] = -nodes2coord[:, 1]
        rigid_modes[1::3, 5] = nodes2coord[:, 0]

        elastic_kernel = rigid_modes[freedofs, :]

        self.params = {
            "u_0": u0_ref.copy(),
            "u_0_ref": u0_ref.copy(),
            "freedofs": freedofs,
            "elems": elems_dof,
            "elems_scalar": elems_scalar,
            "dphix": np.asarray(self._raw["dphix"], dtype=np.float64),
            "dphiy": np.asarray(self._raw["dphiy"], dtype=np.float64),
            "dphiz": np.asarray(self._raw["dphiz"], dtype=np.float64),
            "vol": np.asarray(self._raw["vol"], dtype=np.float64),
            "C1": float(self._raw["C1"]),
            "D1": float(self._raw["D1"]),
            "nodes2coord": nodes2coord,
            "right_nodes": right_nodes,
            "elastic_kernel": elastic_kernel,
        }

        self.u_init = u0_ref[freedofs].copy()

    def get_data(self):
        return self.params, self.adjacency, self.u_init.copy()

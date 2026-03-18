import numpy as np

from src.core.problem_data.hdf5 import (
    jaxify_problem_data,
    load_problem_hdf5,
    mesh_data_path,
)


class MeshHyperElasticity3D:
    def __init__(self, mesh_level):
        self.mesh_level = mesh_level
        self.filename = mesh_data_path(
            "HyperElasticity", f"HyperElasticity_level{mesh_level}.h5"
        )
        self.load_data(self.filename)
        self.compute_initial_guess()
        self.compute_elastic_nullspace()

    def load_data(self, filename):
        self.params, self.adjacency = load_problem_hdf5(filename)

    def compute_initial_guess(self):
        coords = self.params["nodes2coord"].ravel()
        self.u_init = coords[self.params["dofsMinim"].ravel()]

    def compute_elastic_nullspace(self):
        coords = self.params["nodes2coord"]

        # Number of nodes
        N = coords.shape[0]

        # Initialize rigid body modes matrix with 6 modes
        rigid_modes = np.zeros((3 * N, 6))

        # Translational modes
        rigid_modes[::3, 0] = 1  # Translation in X
        rigid_modes[1::3, 1] = 1  # Translation in Y
        rigid_modes[2::3, 2] = 1  # Translation in Z

        # Rotational modes about the X, Y, Z axes
        rigid_modes[1::3, 3] = -coords[:, 2]
        rigid_modes[2::3, 3] = coords[:, 1]

        rigid_modes[::3, 4] = coords[:, 2]
        rigid_modes[2::3, 4] = -coords[:, 0]

        rigid_modes[::3, 5] = -coords[:, 1]
        rigid_modes[1::3, 5] = coords[:, 0]

        self.elastic_kernel = rigid_modes[self.params["dofsMinim"].ravel(), :]

    def get_data_jax(self):
        import jax.numpy as jnp

        params = jaxify_problem_data(
            self.params,
            arrays={
                "u0": "float64",
                "dofsMinim": "int32",
                "elems2nodes": "int32",
                "dphix": "float64",
                "dphiy": "float64",
                "dphiz": "float64",
                "vol": "float64",
            },
            scalars={"C1": float, "D1": float},
        )
        return {
            "u_0": params["u0"],
            "freedofs": params["dofsMinim"],
            "elems": params["elems2nodes"],
            "dvx": params["dphix"],
            "dvy": params["dphiy"],
            "dvz": params["dphiz"],
            "vol": params["vol"],
            "C1": params["C1"],
            "D1": params["D1"],
        }, self.adjacency, jnp.asarray(self.u_init, dtype=jnp.float64)

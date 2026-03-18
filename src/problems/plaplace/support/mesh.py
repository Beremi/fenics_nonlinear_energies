import numpy as np

from src.core.problem_data.hdf5 import (
    jaxify_problem_data,
    load_problem_hdf5,
    mesh_data_path,
)


class MeshpLaplace2D:
    def __init__(self, mesh_level):
        self.mesh_level = mesh_level
        self.filename = mesh_data_path("pLaplace", f"pLaplace_level{mesh_level}.h5")
        self.load_data(self.filename)
        self.compute_initial_guess()

    def load_data(self, filename):
        self.params, self.adjacency = load_problem_hdf5(filename)

    def compute_initial_guess(self):
        np.random.seed(0)
        self.u_init = np.random.rand(self.params["freedofs"].size)

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
                "f": "float64",
            },
            scalars={"p": float},
        )
        return params, self.adjacency, jnp.asarray(self.u_init, dtype=jnp.float64)

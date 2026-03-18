import numpy as np

from src.core.problem_data.hdf5 import (
    jaxify_problem_data,
    load_problem_hdf5,
    mesh_data_path,
)


class MeshGL2D:
    def __init__(self, mesh_level):
        self.mesh_level = mesh_level
        self.filename = mesh_data_path("GinzburgLandau", f"GL_level{mesh_level}.h5")
        self.load_data(self.filename)
        self.compute_initial_guess()

    def load_data(self, filename):
        self.params, self.adjacency = load_problem_hdf5(filename)

    def compute_initial_guess(self):
        def f(x, y): return np.sin(np.pi * (x - 1) / 2) * np.sin(np.pi * (y - 1) / 2)
        self.u_init = f(self.params["nodes"][:, 0], self.params["nodes"][:, 1])
        self.u_init = self.u_init[self.params["freedofs"]]

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
                "ip": "float64",
                "w": "float64",
            },
            scalars={"eps": float},
        )
        return params, self.adjacency, jnp.asarray(self.u_init, dtype=jnp.float64)

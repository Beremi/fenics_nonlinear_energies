"""HDF5-backed loader for the experimental slope-stability P2 case."""

from __future__ import annotations

import numpy as np

from src.core.problem_data.hdf5 import jaxify_problem_data, load_problem_hdf5, mesh_data_path
from src.problems.slope_stability.support.mesh import (
    DEFAULT_CASE,
    DEFAULT_LEVEL,
    build_elastic_initial_guess,
    build_near_nullspace_modes,
    case_name_for_level,
    ensure_default_case_hdf5,
    ensure_level_case_hdf5,
    level_for_case_name,
)


class MeshSlopeStability2D:
    """Load the frozen P2 slope-stability snapshot and derived helpers."""

    def __init__(self, level: int | None = None, case: str | None = None):
        if level is None and case is None:
            level = DEFAULT_LEVEL
        if case is None:
            assert level is not None
            case = case_name_for_level(int(level))
        resolved_level = level_for_case_name(str(case))
        if level is not None and int(level) != resolved_level:
            raise ValueError(
                f"Requested level={level!r} does not match case={case!r}"
            )

        self.level = int(resolved_level)
        self.case = str(case)
        if self.case == DEFAULT_CASE:
            ensure_default_case_hdf5()
        else:
            ensure_level_case_hdf5(self.level)
        self.filename = mesh_data_path("SlopeStability", f"{self.case}.h5")
        self.load_data(self.filename)
        self.compute_initial_guess()
        self.compute_near_nullspace()

    def load_data(self, filename: str) -> None:
        self.params, self.adjacency = load_problem_hdf5(filename)
        if self.adjacency is None:
            raise RuntimeError("SlopeStability snapshot is missing required adjacency data")
        for key in ("case_name", "davis_type"):
            value = self.params.get(key)
            if isinstance(value, (bytes, bytearray)):
                self.params[key] = value.decode("utf-8")

    def compute_initial_guess(self) -> None:
        if "u_init_free" in self.params:
            self.u_init = np.asarray(self.params["u_init_free"], dtype=np.float64)
            return
        self.u_init = build_elastic_initial_guess(
            elems=np.asarray(self.params["elems"], dtype=np.int64),
            elem_B=np.asarray(self.params["elem_B"], dtype=np.float64),
            quad_weight=np.asarray(self.params["quad_weight"], dtype=np.float64),
            force=np.asarray(self.params["force"], dtype=np.float64),
            freedofs=np.asarray(self.params["freedofs"], dtype=np.int64),
            E=float(self.params["E"]),
            nu=float(self.params["nu"]),
        )

    def compute_near_nullspace(self) -> None:
        self.elastic_kernel = build_near_nullspace_modes(
            np.asarray(self.params["nodes"], dtype=np.float64),
            np.asarray(self.params["freedofs"], dtype=np.int64),
        )

    def get_data(self):
        params = dict(self.params)
        params["elastic_kernel"] = np.asarray(self.elastic_kernel, dtype=np.float64)
        return params, self.adjacency, np.asarray(self.u_init, dtype=np.float64)

    def get_data_jax(self):
        import jax.numpy as jnp

        params = jaxify_problem_data(
            self.params,
            arrays={
                "nodes": "float64",
                "elems_scalar": "int32",
                "elems": "int32",
                "surf": "int32",
                "q_mask": "bool",
                "freedofs": "int32",
                "elem_B": "float64",
                "quad_weight": "float64",
                "force": "float64",
                "u_0": "float64",
                "eps_p_old": "float64",
            },
            scalars={
                "h": float,
                "x1": float,
                "x2": float,
                "x3": float,
                "y1": float,
                "y2": float,
                "beta_deg": float,
                "E": float,
                "nu": float,
                "c0": float,
                "phi_deg": float,
                "psi_deg": float,
                "gamma": float,
                "lambda_target_default": float,
                "level": int,
            },
        )
        params["cohesion"] = float(self.params["c0"])
        params["elastic_kernel"] = jnp.asarray(self.elastic_kernel, dtype=jnp.float64)
        return params, self.adjacency, jnp.asarray(self.u_init, dtype=jnp.float64)

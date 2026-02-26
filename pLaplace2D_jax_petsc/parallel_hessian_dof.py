"""
pLaplace-specific parallel Hessian assemblers.

Thin subclasses of the generic assemblers from
``tools_petsc4py.parallel_assembler``, providing the pLaplace energy
function via ``_make_local_energy_fns()``.

Usage
-----
>>> from pLaplace2D_jax_petsc.parallel_hessian_dof import LocalColoringAssembler
>>> assembler = LocalColoringAssembler(params=params, comm=comm, adjacency=adj)
"""

import numpy as np
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from tools_petsc4py.parallel_assembler import (  # noqa: E402
    DOFHessianAssemblerBase,
    LocalColoringAssemblerBase,
)


class _PLaplaceMixin:
    """Mixin providing pLaplace energy function via _make_local_energy_fns().

    Expects ``self._p`` to be set before ``_compile_jax()`` is called,
    and ``self.part`` to have ``elems_local_np``, ``local_elem_data``
    with keys "dvx", "dvy", "vol", and ``elem_weights``.
    """

    def _make_local_energy_fns(self):
        """Return (energy_weighted, energy_full) for pLaplace."""
        p = self._p
        elems = jnp.array(self.part.elems_local_np, dtype=jnp.int32)
        dvx = jnp.array(self.part.local_elem_data["dvx"], dtype=jnp.float64)
        dvy = jnp.array(self.part.local_elem_data["dvy"], dtype=jnp.float64)
        vol = jnp.array(self.part.local_elem_data["vol"], dtype=jnp.float64)
        vol_w = jnp.array(
            self.part.local_elem_data["vol"] * self.part.elem_weights,
            dtype=jnp.float64,
        )

        def energy_weighted(v_local):
            v_e = v_local[elems]
            Fx = jnp.sum(v_e * dvx, axis=1)
            Fy = jnp.sum(v_e * dvy, axis=1)
            intgrds = (1.0 / p) * (Fx**2 + Fy**2) ** (p / 2.0)
            return jnp.sum(intgrds * vol_w)

        def energy_full(v_local):
            v_e = v_local[elems]
            Fx = jnp.sum(v_e * dvx, axis=1)
            Fy = jnp.sum(v_e * dvy, axis=1)
            intgrds = (1.0 / p) * (Fx**2 + Fy**2) ** (p / 2.0)
            return jnp.sum(intgrds * vol)

        return energy_weighted, energy_full


class ParallelDOFHessianAssembler(_PLaplaceMixin, DOFHessianAssemblerBase):
    """pLaplace assembler with global multi-start coloring.

    Accepts the pLaplace ``params`` dict for backward compatibility.
    """

    def __init__(
        self,
        params,
        comm,
        adjacency=None,
        coloring_trials_per_rank=10,
        ksp_rtol=1e-3,
        ksp_type="cg",
        pc_type="gamg",
    ):
        self._p = float(params["p"])
        super().__init__(
            freedofs=np.asarray(params["freedofs"]),
            elems=np.asarray(params["elems"]),
            u_0=np.asarray(params["u_0"]),
            comm=comm,
            adjacency=adjacency,
            f=np.asarray(params["f"]),
            elem_data={
                "dvx": np.asarray(params["dvx"]),
                "dvy": np.asarray(params["dvy"]),
                "vol": np.asarray(params["vol"]),
            },
            coloring_trials_per_rank=coloring_trials_per_rank,
            ksp_rtol=ksp_rtol,
            ksp_type=ksp_type,
            pc_type=pc_type,
        )


class LocalColoringAssembler(_PLaplaceMixin, LocalColoringAssemblerBase):
    """pLaplace assembler with per-rank local coloring + vmap.

    Accepts the pLaplace ``params`` dict for backward compatibility.
    """

    def __init__(
        self,
        params,
        comm,
        adjacency=None,
        coloring_trials_per_rank=10,
        ksp_rtol=1e-3,
        ksp_type="cg",
        pc_type="gamg",
    ):
        self._p = float(params["p"])
        super().__init__(
            freedofs=np.asarray(params["freedofs"]),
            elems=np.asarray(params["elems"]),
            u_0=np.asarray(params["u_0"]),
            comm=comm,
            adjacency=adjacency,
            f=np.asarray(params["f"]),
            elem_data={
                "dvx": np.asarray(params["dvx"]),
                "dvy": np.asarray(params["dvy"]),
                "vol": np.asarray(params["vol"]),
            },
            coloring_trials_per_rank=coloring_trials_per_rank,
            ksp_rtol=ksp_rtol,
            ksp_type=ksp_type,
            pc_type=pc_type,
        )

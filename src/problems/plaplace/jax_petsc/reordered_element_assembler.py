"""Production pLaplace element assembler using reordered PETSc ownership."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import config

from src.core.petsc.reordered_element_base import ReorderedElementAssemblerBase


config.update("jax_enable_x64", True)


def _plaplace_integrand(v_e, dvx_e, dvy_e, p):
    fx = jnp.sum(v_e * dvx_e, axis=-1)
    fy = jnp.sum(v_e * dvy_e, axis=-1)
    return (1.0 / p) * (fx**2 + fy**2) ** (p / 2.0)


class PLaplaceReorderedElementAssembler(ReorderedElementAssemblerBase):
    """Production pLaplace assembler using overlap domains and reordered ownership."""

    block_size = 1
    coordinate_key = "nodes"
    dirichlet_key = "u_0"
    local_elem_data_keys = ("dvx", "dvy", "vol")

    def _make_local_element_kernels(self):
        elems = jnp.asarray(self.local_data.elems_local_np, dtype=jnp.int32)
        dvx = jnp.asarray(self.local_data.local_elem_data["dvx"], dtype=jnp.float64)
        dvy = jnp.asarray(self.local_data.local_elem_data["dvy"], dtype=jnp.float64)
        vol = jnp.asarray(self.local_data.local_elem_data["vol"], dtype=jnp.float64)
        energy_weights = jnp.asarray(self.local_data.energy_weights, dtype=jnp.float64)
        p = float(self.params["p"])

        def element_energy(v_e, dvx_e, dvy_e, vol_e):
            return _plaplace_integrand(v_e, dvx_e, dvy_e, p) * vol_e

        hess_elem = jax.vmap(jax.hessian(element_energy), in_axes=(0, 0, 0, 0))

        def local_full_energy(v_local):
            v_e = v_local[elems]
            e = jax.vmap(element_energy, in_axes=(0, 0, 0, 0))(v_e, dvx, dvy, vol)
            return jnp.sum(e)

        grad_local = jax.grad(local_full_energy)

        @jax.jit
        def energy_fn(v_local):
            v_e = v_local[elems]
            e = jax.vmap(element_energy, in_axes=(0, 0, 0, 0))(v_e, dvx, dvy, vol)
            return jnp.sum(e * energy_weights)

        @jax.jit
        def local_grad_fn(v_local):
            return grad_local(v_local)

        @jax.jit
        def elem_hess_fn(v_local):
            v_e = v_local[elems]
            return hess_elem(v_e, dvx, dvy, vol)

        return energy_fn, local_grad_fn, elem_hess_fn, grad_local

    def _build_rhs_owned(self) -> np.ndarray:
        rhs = np.asarray(self.params["f"], dtype=np.float64)
        freedofs = np.asarray(self.params["freedofs"], dtype=np.int64)
        rhs_free = rhs[freedofs]
        rhs_reordered = rhs_free[self.layout.perm]
        return np.asarray(rhs_reordered[self.layout.lo : self.layout.hi], dtype=np.float64)

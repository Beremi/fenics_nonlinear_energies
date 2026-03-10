"""Production GL element assembler using reordered PETSc ownership."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import config

from pLaplace2D_jax_petsc.reordered_element_assembler import (
    PLaplaceReorderedElementAssembler,
)


config.update("jax_enable_x64", True)


def _gl_integrand(v_e, dvx_e, dvy_e, ip, w, eps):
    fx = jnp.sum(v_e * dvx_e, axis=-1)
    fy = jnp.sum(v_e * dvy_e, axis=-1)
    grad_term = 0.5 * eps * (fx**2 + fy**2)
    nodal_vals = v_e @ ip
    potential_term = 0.25 * (((nodal_vals**2) - 1.0) ** 2) @ w
    return grad_term + potential_term


class GLReorderedElementAssembler(PLaplaceReorderedElementAssembler):
    """GL overlap assembler with exact element Hessians or local SFD."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dirichlet_only_energy_local = self._compute_dirichlet_only_energy_local()

    def _compute_dirichlet_only_energy_local(self) -> float:
        if self.rank != 0:
            return 0.0

        elems = np.asarray(self.params["elems"], dtype=np.int64)
        elems_reordered = self.layout.total_to_free_reord[elems]
        mask = np.all(elems_reordered < 0, axis=1)
        if not np.any(mask):
            return 0.0

        v_e = jnp.asarray(np.asarray(self.params["u_0"], dtype=np.float64)[elems[mask]])
        dvx = jnp.asarray(np.asarray(self.params["dvx"], dtype=np.float64)[mask])
        dvy = jnp.asarray(np.asarray(self.params["dvy"], dtype=np.float64)[mask])
        vol = jnp.asarray(np.asarray(self.params["vol"], dtype=np.float64)[mask])
        ip = jnp.asarray(self.params["ip"], dtype=jnp.float64)
        w = jnp.asarray(self.params["w"], dtype=jnp.float64)
        eps = float(self.params["eps"])

        return float(jnp.sum(_gl_integrand(v_e, dvx, dvy, ip, w, eps) * vol))

    def _make_local_element_kernels(self):
        elems = jnp.asarray(self.local_data.elems_local_np, dtype=jnp.int32)
        dvx = jnp.asarray(self.local_data.local_elem_data["dvx"], dtype=jnp.float64)
        dvy = jnp.asarray(self.local_data.local_elem_data["dvy"], dtype=jnp.float64)
        vol = jnp.asarray(self.local_data.local_elem_data["vol"], dtype=jnp.float64)
        energy_weights = jnp.asarray(self.local_data.energy_weights, dtype=jnp.float64)
        ip = jnp.asarray(self.params["ip"], dtype=jnp.float64)
        w = jnp.asarray(self.params["w"], dtype=jnp.float64)
        eps = float(self.params["eps"])

        def element_energy(v_e, dvx_e, dvy_e, vol_e):
            return _gl_integrand(v_e, dvx_e, dvy_e, ip, w, eps) * vol_e

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
        return np.zeros(self.layout.hi - self.layout.lo, dtype=np.float64)

    def energy_fn(self, vec):
        return super().energy_fn(vec) + float(self._dirichlet_only_energy_local)

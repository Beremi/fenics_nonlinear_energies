"""PETSc reordered-element assembler for the arctan-resonance problem."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import config

from src.core.petsc.reordered_element_base import ReorderedElementAssemblerBase
from src.problems.plaplace_up_arctan.common import (
    G_arctan_shifted,
    TRI_QUAD_BARY,
    TRI_QUAD_WEIGHTS,
    gradient_components,
)


config.update("jax_enable_x64", True)

_HESSIAN_SMOOTH_EPS = 1.0e-12


def _smooth_abs_power(values, exponent: float):
    eps_sq = _HESSIAN_SMOOTH_EPS * _HESSIAN_SMOOTH_EPS
    return (values * values + eps_sq) ** (0.5 * float(exponent)) - (_HESSIAN_SMOOTH_EPS ** float(exponent))


def _smooth_grad_power(v_e, dvx_e, dvy_e, p: float):
    fx, fy = gradient_components(v_e, dvx_e, dvy_e)
    g2 = fx * fx + fy * fy
    eps_sq = _HESSIAN_SMOOTH_EPS * _HESSIAN_SMOOTH_EPS
    return (g2 + eps_sq) ** (0.5 * float(p)) - (_HESSIAN_SMOOTH_EPS ** float(p))


def _quad_values(v_e):
    return v_e @ TRI_QUAD_BARY.T


def _smooth_lq_integrand(v_e, exponent: float):
    values = _quad_values(v_e)
    return jnp.sum(TRI_QUAD_WEIGHTS * _smooth_abs_power(values, float(exponent)), axis=-1)


def _g_integrand(v_e):
    values = _quad_values(v_e)
    return jnp.sum(TRI_QUAD_WEIGHTS * G_arctan_shifted(values), axis=-1)


class PLaplaceUPArctanReorderedElementAssembler(ReorderedElementAssemblerBase):
    """Distributed scalar assembler for the arctan-resonance energy."""

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
        lambda1 = float(self.params["lambda1"])

        def element_energy(v_e, dvx_e, dvy_e, vol_e):
            density = (1.0 / p) * _smooth_grad_power(v_e, dvx_e, dvy_e, p) - (lambda1 / p) * _smooth_lq_integrand(v_e, p) - _g_integrand(v_e)
            return density * vol_e

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

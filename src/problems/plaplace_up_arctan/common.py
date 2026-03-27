"""Shared formulas for the unit-square arctan-resonance p-Laplacian."""

from __future__ import annotations

import math

from jax import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)


# Six-point Dunavant rule, scaled so the weights sum to one on each physical
# triangle after multiplication by the triangle area.
TRI_QUAD_BARY = jnp.asarray(
    [
        [0.816847572980459, 0.091576213509771, 0.091576213509771],
        [0.091576213509771, 0.816847572980459, 0.091576213509771],
        [0.091576213509771, 0.091576213509771, 0.816847572980459],
        [0.108103018168070, 0.445948490915965, 0.445948490915965],
        [0.445948490915965, 0.108103018168070, 0.445948490915965],
        [0.445948490915965, 0.445948490915965, 0.108103018168070],
    ],
    dtype=jnp.float64,
)
TRI_QUAD_WEIGHTS = jnp.asarray(
    [
        0.109951743655322,
        0.109951743655322,
        0.109951743655322,
        0.223381589678011,
        0.223381589678011,
        0.223381589678011,
    ],
    dtype=jnp.float64,
)

G_ZERO_CONSTANT = (math.pi / 4.0) - 0.5 * math.log(2.0)


def gradient_components(v_e, dvx_e, dvy_e):
    fx = (v_e * dvx_e).sum(axis=-1)
    fy = (v_e * dvy_e).sum(axis=-1)
    return fx, fy


def grad_norm_sq(v_e, dvx_e, dvy_e):
    fx, fy = gradient_components(v_e, dvx_e, dvy_e)
    return fx * fx + fy * fy


def arctan_shifted(value):
    return jnp.arctan(value + 1.0)


def arctan_shifted_prime(value):
    shifted = value + 1.0
    return 1.0 / (1.0 + shifted * shifted)


def G_arctan_shifted(value):
    shifted = value + 1.0
    return shifted * jnp.arctan(shifted) - 0.5 * jnp.log1p(shifted * shifted) - G_ZERO_CONSTANT


def F_helper(value, p):
    safe = jnp.where(jnp.abs(value) > 1.0e-12, value, 1.0)
    raw = (p / safe) * G_arctan_shifted(safe) - arctan_shifted(safe)
    limit_zero = p * (math.pi / 4.0) - (math.pi / 4.0)
    return jnp.where(jnp.abs(value) > 1.0e-12, raw, limit_zero)


def a_integrand(v_e, dvx_e, dvy_e, p):
    return grad_norm_sq(v_e, dvx_e, dvy_e) ** (p / 2.0)


def _triangle_quadrature_values(v_e):
    return v_e @ TRI_QUAD_BARY.T


def lq_integrand(v_e, exponent):
    values = _triangle_quadrature_values(v_e)
    return jnp.sum(TRI_QUAD_WEIGHTS * jnp.abs(values) ** exponent, axis=-1)


def g_integrand(v_e):
    values = _triangle_quadrature_values(v_e)
    return jnp.sum(TRI_QUAD_WEIGHTS * G_arctan_shifted(values), axis=-1)


def energy_integrand(v_e, dvx_e, dvy_e, *, p, lambda1):
    return (
        (1.0 / p) * a_integrand(v_e, dvx_e, dvy_e, p)
        - (lambda1 / p) * lq_integrand(v_e, p)
        - g_integrand(v_e)
    )

"""Plane-strain Mohr-Coulomb energy kernels used by the slope benchmark implementations."""

from __future__ import annotations

import jax.numpy as jnp
from jax import config, lax, vmap

config.update("jax_enable_x64", True)


def _elastic_tensors(E, nu):
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = jnp.array(
        [
            [lam + 2.0 * mu, lam, 0.0],
            [lam, lam + 2.0 * mu, 0.0],
            [0.0, 0.0, mu],
        ],
        dtype=jnp.float64,
    )
    S = jnp.array(
        [
            [(1.0 - nu * nu) / E, -(nu * (1.0 + nu)) / E, 0.0],
            [-(nu * (1.0 + nu)) / E, (1.0 - nu * nu) / E, 0.0],
            [0.0, 0.0, 1.0 / mu],
        ],
        dtype=jnp.float64,
    )
    return C, S


def _elastic_energy_density(
    strain,
    eps_p_old,
    E,
    nu,
):
    C, S = _elastic_tensors(E, nu)
    eps_e = strain - eps_p_old
    sig = C @ eps_e
    return sig @ eps_e - 0.5 * sig @ (S @ sig)


def _mc_energy_density(
    strain,
    eps_p_old,
    E,
    nu,
    phi_deg,
    cohesion,
    reg,
):
    C, S = _elastic_tensors(E, nu)
    eps_e = strain - eps_p_old
    sig_tr = C @ eps_e

    sxx, syy, txy = sig_tr
    m = 0.5 * (sxx + syy)
    d = 0.5 * (sxx - syy)
    r = jnp.sqrt(d * d + txy * txy + reg * reg)
    s1 = m + r
    s2 = m - r

    phi = jnp.deg2rad(phi_deg)
    alpha = 1.0 + jnp.sin(phi)
    beta = 1.0 - jnp.sin(phi)
    kappa = 2.0 * cohesion * jnp.cos(phi)
    f_tr = alpha * s1 - beta * s2 - kappa

    def elastic(_):
        sig = sig_tr
        return sig @ eps_e - 0.5 * sig @ (S @ sig)

    def plastic(_):
        A11, A12, A22 = S[0, 0], S[0, 1], S[1, 1]
        n1, n2 = alpha, -beta
        det = A11 * A22 - A12 * A12
        z1 = (A22 * n1 - A12 * n2) / det
        z2 = (-A12 * n1 + A11 * n2) / det
        lam_mc = (n1 * s1 + n2 * s2 - kappa) / (n1 * z1 + n2 * z2)
        s1_line = s1 - lam_mc * z1
        s2_line = s2 - lam_mc * z2

        safe_den = jnp.where(jnp.abs(alpha - beta) > 1.0e-14, alpha - beta, 1.0)
        s_apex = kappa / safe_den

        def line_return(_):
            return s1_line, s2_line

        def apex_return(_):
            return s_apex, s_apex

        s1p, s2p = lax.cond(s1_line >= s2_line, line_return, apex_return, operand=None)
        mp = 0.5 * (s1p + s2p)
        rp = 0.5 * (s1p - s2p)
        two_theta = jnp.arctan2(2.0 * txy, sxx - syy)
        sig = jnp.array(
            [
                mp + rp * jnp.cos(two_theta),
                mp - rp * jnp.cos(two_theta),
                rp * jnp.sin(two_theta),
            ],
            dtype=jnp.float64,
        )
        return sig @ eps_e - 0.5 * sig @ (S @ sig)

    return lax.cond(f_tr <= 0.0, elastic, plastic, operand=None)


_quadrature_density = vmap(
    _mc_energy_density,
    in_axes=(0, 0, None, None, None, None, None),
)
_elastic_quadrature_density = vmap(
    _elastic_energy_density,
    in_axes=(0, 0, None, None),
)


def element_energy(
    u_elem,
    elem_B_elem,
    quad_weight_elem,
    eps_p_old_elem,
    E,
    nu,
    phi_deg,
    cohesion,
    reg=1.0e-12,
):
    """Return the scalar internal energy of one triangle in the active Lagrange space.

    The contract is degree-agnostic: the current repository uses it for the
    original P2 bring-up and for the same-mesh P4 Mohr-Coulomb runs, with the
    element size carried entirely by ``u_elem`` / ``elem_B_elem`` /
    ``quad_weight_elem``.
    """

    strain = jnp.einsum("qij,j->qi", elem_B_elem, u_elem)
    density = _quadrature_density(strain, eps_p_old_elem, E, nu, phi_deg, cohesion, reg)
    return jnp.sum(quad_weight_elem * density)


def elastic_element_energy(
    u_elem,
    elem_B_elem,
    quad_weight_elem,
    eps_p_old_elem,
    E,
    nu,
    reg=1.0e-12,
):
    del reg
    strain = jnp.einsum("qij,j->qi", elem_B_elem, u_elem)
    density = _elastic_quadrature_density(strain, eps_p_old_elem, E, nu)
    return jnp.sum(quad_weight_elem * density)


_element_energy_vmapped = vmap(
    element_energy,
    in_axes=(0, 0, 0, 0, None, None, None, None, None),
)


def J(
    u_free,
    u_0,
    freedofs,
    elems,
    elem_B,
    quad_weight,
    force,
    eps_p_old,
    E,
    nu,
    phi_deg,
    cohesion,
    reg=1.0e-12,
):
    u_full = u_0.at[freedofs].set(u_free)
    u_elem = u_full[elems]
    internal = jnp.sum(
        _element_energy_vmapped(
            u_elem,
            elem_B,
            quad_weight,
            eps_p_old,
            E,
            nu,
            phi_deg,
            cohesion,
            reg,
        )
    )
    external = jnp.dot(force, u_full)
    return internal - external

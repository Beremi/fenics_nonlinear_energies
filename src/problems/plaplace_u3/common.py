"""Shared exact P1 formulas for the scalar p-Laplacian with cubic reaction.

These helpers are intentionally tiny and backend-agnostic: the thesis layer uses
the same expressions both in JAX-differentiated objectives and in NumPy
post-processing/reporting. Keeping the polynomial factors in one place makes it
much easier to audit that those two paths are computing the same quantities.
"""

from __future__ import annotations


def gradient_components(v_e, dvx_e, dvy_e):
    """Return the constant P1 gradient on each triangle element."""
    fx = (v_e * dvx_e).sum(axis=-1)
    fy = (v_e * dvy_e).sum(axis=-1)
    return fx, fy


def grad_norm_sq(v_e, dvx_e, dvy_e):
    """Return ``|∇u_h|^2`` on each triangle element."""
    fx, fy = gradient_components(v_e, dvx_e, dvy_e)
    return fx * fx + fy * fy


def degree4_complete_homogeneous(v_e):
    """Return the exact quartic triangle factor ``h_4(v0, v1, v2)``.

    For a P1 function on one triangle, ``∫_K u_h^4`` equals ``|K| * h_4 / 15``.
    Writing the polynomial out explicitly is verbose, but it is also the most
    transparent representation to compare against the thesis derivation.
    """
    v0 = v_e[..., 0]
    v1 = v_e[..., 1]
    v2 = v_e[..., 2]
    return (
        v0**4
        + v1**4
        + v2**4
        + v0**3 * v1
        + v0**3 * v2
        + v1**3 * v0
        + v1**3 * v2
        + v2**3 * v0
        + v2**3 * v1
        + v0**2 * v1**2
        + v0**2 * v2**2
        + v1**2 * v2**2
        + v0**2 * v1 * v2
        + v0 * v1**2 * v2
        + v0 * v1 * v2**2
    )


def quartic_p1_integrand(v_e):
    """Return the exact quartic density factor for one triangle element."""
    return degree4_complete_homogeneous(v_e) / 15.0


def a_integrand(v_e, dvx_e, dvy_e, p):
    """Return the elementwise density for ``A(u) = ∫ |∇u|^p``."""
    return grad_norm_sq(v_e, dvx_e, dvy_e) ** (p / 2.0)


def b_integrand(v_e):
    """Return the elementwise density for ``B(u) = ∫ u^4``."""
    return quartic_p1_integrand(v_e)


def energy_integrand(v_e, dvx_e, dvy_e, p):
    """Return the thesis energy density ``J(u) = A(u)/p - B(u)/4``."""
    return (1.0 / p) * a_integrand(v_e, dvx_e, dvy_e, p) - 0.25 * b_integrand(v_e)

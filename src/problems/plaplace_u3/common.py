"""Shared exact formulas for the scalar p-Laplacian with cubic reaction."""

from __future__ import annotations


def gradient_components(v_e, dvx_e, dvy_e):
    """Return elementwise piecewise-constant P1 gradient components."""
    fx = (v_e * dvx_e).sum(axis=-1)
    fy = (v_e * dvy_e).sum(axis=-1)
    return fx, fy


def grad_norm_sq(v_e, dvx_e, dvy_e):
    fx, fy = gradient_components(v_e, dvx_e, dvy_e)
    return fx * fx + fy * fy


def degree4_complete_homogeneous(v_e):
    """Return h_4(v1, v2, v3) for one or many P1 triangle nodal vectors."""
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
    """Exact per-element density factor for int_K u_h^4 dx = |K| * q(v)."""
    return degree4_complete_homogeneous(v_e) / 15.0


def a_integrand(v_e, dvx_e, dvy_e, p):
    """Per-element density for A(u) = int |grad u|^p."""
    return grad_norm_sq(v_e, dvx_e, dvy_e) ** (p / 2.0)


def b_integrand(v_e):
    """Per-element density for B(u) = int u^4."""
    return quartic_p1_integrand(v_e)


def energy_integrand(v_e, dvx_e, dvy_e, p):
    """Per-element density for J(u) = A(u)/p - B(u)/4."""
    return (1.0 / p) * a_integrand(v_e, dvx_e, dvy_e, p) - 0.25 * b_integrand(v_e)

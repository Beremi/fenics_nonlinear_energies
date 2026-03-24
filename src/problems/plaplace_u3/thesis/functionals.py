"""Discrete thesis functionals and helper operators.

This file is the main "math dictionary" for the thesis layer:

- exact FE evaluations of ``A(u)``, ``B(u)``, ``J(u)``, and ``I(u)``
- rescaling from a raw iterate to the physical weak-solution representative
- the standard Laplace stiffness matrix used only by the auxiliary ``d^{V_h}``
  direction solve
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from src.problems.plaplace_u3.common import a_integrand, b_integrand, energy_integrand


@dataclass(frozen=True)
class StateStats:
    """Discrete scalar state and scaling information for one FE vector."""

    a: float
    b: float
    seminorm_p: float
    l4_norm: float
    J: float
    I: float
    c: float
    scale_to_solution: float


def conjugate_exponent(p: float) -> float:
    """Return ``q = p / (p - 1)`` for ``p > 1``."""
    p = float(p)
    if p <= 1.0:
        raise ValueError("p must be > 1 for the conjugate exponent")
    return p / (p - 1.0)


def expand_free_vector(u_free: np.ndarray, u_0: np.ndarray, freedofs: np.ndarray) -> np.ndarray:
    """Expand one free-DOF vector into the full nodal vector."""
    u_full = np.asarray(u_0, dtype=np.float64).copy()
    u_full[np.asarray(freedofs, dtype=np.int64)] = np.asarray(u_free, dtype=np.float64)
    return u_full


def quartic_interval_integrand(v_e):
    """Exact per-element density factor for ``∫_K u_h^4 dx`` on one P1 interval."""
    v0 = v_e[..., 0]
    v1 = v_e[..., 1]
    return (v0**4 + v0**3 * v1 + v0**2 * v1**2 + v0 * v1**3 + v1**4) / 5.0


def a_integrand_interval(v_e, dv_e, exponent):
    """Per-element density for ``∫ |u'|^exponent`` on one P1 interval."""
    grad = jnp.sum(v_e * dv_e, axis=-1)
    return jnp.abs(grad) ** exponent


def energy_integrand_interval(v_e, dv_e, p):
    """Per-element density for the thesis energy on one P1 interval."""
    return (1.0 / p) * a_integrand_interval(v_e, dv_e, p) - 0.25 * quartic_interval_integrand(v_e)


def triangle_terms_full(
    u_full: np.ndarray,
    *,
    elems: np.ndarray,
    dvx: np.ndarray,
    dvy: np.ndarray,
    vol: np.ndarray,
    exponent: float,
) -> float:
    """Return ``∫ |∇u_h|^exponent`` on the structured P1 triangle mesh."""
    u_e = np.asarray(u_full[np.asarray(elems, dtype=np.int64)], dtype=np.float64)
    value = np.asarray(
        a_integrand(
            jnp.asarray(u_e),
            jnp.asarray(np.asarray(dvx, dtype=np.float64)),
            jnp.asarray(np.asarray(dvy, dtype=np.float64)),
            float(exponent),
        ),
        dtype=np.float64,
    )
    return float(np.sum(value * np.asarray(vol, dtype=np.float64)))


def interval_terms_full(
    u_full: np.ndarray,
    *,
    elems: np.ndarray,
    dv: np.ndarray,
    vol: np.ndarray,
    exponent: float,
) -> float:
    """Return ``∫ |u_h'|^exponent`` on the structured P1 interval mesh."""
    u_e = np.asarray(u_full[np.asarray(elems, dtype=np.int64)], dtype=np.float64)
    grad = np.sum(u_e * np.asarray(dv, dtype=np.float64), axis=1)
    return float(np.sum(np.abs(grad) ** float(exponent) * np.asarray(vol, dtype=np.float64)))


def triangle_b_full(
    u_full: np.ndarray,
    *,
    elems: np.ndarray,
    vol: np.ndarray,
) -> float:
    """Return ``∫ u_h^4`` on the structured triangle mesh."""
    u_e = np.asarray(u_full[np.asarray(elems, dtype=np.int64)], dtype=np.float64)
    value = np.asarray(b_integrand(jnp.asarray(u_e)), dtype=np.float64)
    return float(np.sum(value * np.asarray(vol, dtype=np.float64)))


def interval_b_full(
    u_full: np.ndarray,
    *,
    elems: np.ndarray,
    vol: np.ndarray,
) -> float:
    """Return ``∫ u_h^4`` on the structured interval mesh."""
    u_e = np.asarray(u_full[np.asarray(elems, dtype=np.int64)], dtype=np.float64)
    value = np.asarray(quartic_interval_integrand(jnp.asarray(u_e)), dtype=np.float64)
    return float(np.sum(value * np.asarray(vol, dtype=np.float64)))


def seminorm_full(params: dict[str, object], u_full: np.ndarray, *, exponent: float | None = None) -> float:
    """Return the discrete ``|u|_{1,exponent,0}`` seminorm."""
    exponent_eff = float(params["p"] if exponent is None else exponent)
    topology = str(params["topology"])
    if topology == "triangle":
        a = triangle_terms_full(
            u_full,
            elems=np.asarray(params["elems"], dtype=np.int64),
            dvx=np.asarray(params["dvx"], dtype=np.float64),
            dvy=np.asarray(params["dvy"], dtype=np.float64),
            vol=np.asarray(params["vol"], dtype=np.float64),
            exponent=exponent_eff,
        )
    elif topology == "interval":
        a = interval_terms_full(
            u_full,
            elems=np.asarray(params["elems"], dtype=np.int64),
            dv=np.asarray(params["dv"], dtype=np.float64),
            vol=np.asarray(params["vol"], dtype=np.float64),
            exponent=exponent_eff,
        )
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported topology {topology!r}")
    return float(a ** (1.0 / exponent_eff))


def l4_norm_full(params: dict[str, object], u_full: np.ndarray) -> float:
    """Return the discrete ``L^4`` norm."""
    topology = str(params["topology"])
    if topology == "triangle":
        b = triangle_b_full(
            u_full,
            elems=np.asarray(params["elems"], dtype=np.int64),
            vol=np.asarray(params["vol"], dtype=np.float64),
        )
    elif topology == "interval":
        b = interval_b_full(
            u_full,
            elems=np.asarray(params["elems"], dtype=np.int64),
            vol=np.asarray(params["vol"], dtype=np.float64),
        )
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported topology {topology!r}")
    return float(b ** 0.25)


def compute_state_stats_full(params: dict[str, object], u_full: np.ndarray) -> StateStats:
    """Evaluate the thesis functionals on one full nodal vector.

    ``a`` and ``b`` are the raw integral terms from the thesis notation, while
    ``scale_to_solution`` is the analytic ray projection that maps a nonzero
    iterate onto the associated weak solution on its positive ray.
    """
    p = float(params["p"])
    topology = str(params["topology"])

    if topology == "triangle":
        a = triangle_terms_full(
            u_full,
            elems=np.asarray(params["elems"], dtype=np.int64),
            dvx=np.asarray(params["dvx"], dtype=np.float64),
            dvy=np.asarray(params["dvy"], dtype=np.float64),
            vol=np.asarray(params["vol"], dtype=np.float64),
            exponent=p,
        )
        b = triangle_b_full(
            u_full,
            elems=np.asarray(params["elems"], dtype=np.int64),
            vol=np.asarray(params["vol"], dtype=np.float64),
        )
    elif topology == "interval":
        a = interval_terms_full(
            u_full,
            elems=np.asarray(params["elems"], dtype=np.int64),
            dv=np.asarray(params["dv"], dtype=np.float64),
            vol=np.asarray(params["vol"], dtype=np.float64),
            exponent=p,
        )
        b = interval_b_full(
            u_full,
            elems=np.asarray(params["elems"], dtype=np.int64),
            vol=np.asarray(params["vol"], dtype=np.float64),
        )
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported topology {topology!r}")

    seminorm_p = float(a ** (1.0 / p)) if a > 0.0 else 0.0
    l4 = float(b ** 0.25) if b > 0.0 else 0.0
    J_value = float(a / p - 0.25 * b)
    I_value = float(seminorm_p / l4) if seminorm_p > 0.0 and l4 > 0.0 else float("inf")
    c = float(1.0 / I_value) if np.isfinite(I_value) and I_value > 0.0 else 0.0
    scale = float((a / b) ** (1.0 / (4.0 - p))) if a > 0.0 and b > 0.0 else float("nan")
    return StateStats(
        a=float(a),
        b=float(b),
        seminorm_p=seminorm_p,
        l4_norm=l4,
        J=J_value,
        I=I_value,
        c=c,
        scale_to_solution=scale,
    )


def compute_state_stats_free(params: dict[str, object], u_free: np.ndarray) -> StateStats:
    """Evaluate the thesis functionals on one free-DOF vector."""
    u_full = expand_free_vector(
        np.asarray(u_free, dtype=np.float64),
        np.asarray(params["u_0"], dtype=np.float64),
        np.asarray(params["freedofs"], dtype=np.int64),
    )
    return compute_state_stats_full(params, u_full)


def rescale_free_to_solution(params: dict[str, object], u_free: np.ndarray) -> tuple[np.ndarray, np.ndarray, StateStats]:
    """Return the thesis weak-solution scaling ``u_tilde`` together with its stats."""
    raw_stats = compute_state_stats_free(params, u_free)
    scaled = np.asarray(u_free, dtype=np.float64) * float(raw_stats.scale_to_solution)
    scaled_full = expand_free_vector(
        scaled,
        np.asarray(params["u_0"], dtype=np.float64),
        np.asarray(params["freedofs"], dtype=np.int64),
    )
    return scaled, scaled_full, compute_state_stats_full(params, scaled_full)


def _append_local_matrix_entries(
    loc: np.ndarray,
    ke: np.ndarray,
    *,
    rows: list[np.ndarray],
    cols: list[np.ndarray],
    data: list[np.ndarray],
) -> None:
    """Append the free-DOF part of one element matrix in COO triplet form."""
    mask = np.asarray(loc, dtype=np.int64) >= 0
    if not np.any(mask):
        return
    free_loc = np.asarray(loc[mask], dtype=np.int64)
    rows.append(np.repeat(free_loc, free_loc.size))
    cols.append(np.tile(free_loc, free_loc.size))
    data.append(np.asarray(ke[np.ix_(mask, mask)], dtype=np.float64).reshape(-1))


def assemble_stiffness_matrix(params: dict[str, object]) -> sp.csr_matrix:
    """Assemble the standard scalar Laplace stiffness matrix on free DOFs.

    This is not the nonlinear p-Laplacian Hessian. It is the cheap linear helper
    matrix used by the approximate thesis descent direction ``d^{V_h}``.
    """
    topology = str(params["topology"])
    freedofs = np.asarray(params["freedofs"], dtype=np.int64)
    elems = np.asarray(params["elems"], dtype=np.int64)
    full_to_free = np.full(int(np.asarray(params["u_0"]).size), -1, dtype=np.int64)
    full_to_free[freedofs] = np.arange(freedofs.size, dtype=np.int64)
    local = full_to_free[elems]

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []

    if topology == "triangle":
        dvx = np.asarray(params["dvx"], dtype=np.float64)
        dvy = np.asarray(params["dvy"], dtype=np.float64)
        vol = np.asarray(params["vol"], dtype=np.float64)
        for elem_idx in range(elems.shape[0]):
            # Standard P1 Laplace local stiffness on one triangle.
            ke = vol[elem_idx] * (
                np.outer(dvx[elem_idx], dvx[elem_idx]) + np.outer(dvy[elem_idx], dvy[elem_idx])
            )
            _append_local_matrix_entries(local[elem_idx], ke, rows=rows, cols=cols, data=data)
    elif topology == "interval":
        dv = np.asarray(params["dv"], dtype=np.float64)
        vol = np.asarray(params["vol"], dtype=np.float64)
        for elem_idx in range(elems.shape[0]):
            # In 1D the same idea reduces to the interval derivative matrix.
            ke = vol[elem_idx] * np.outer(dv[elem_idx], dv[elem_idx])
            _append_local_matrix_entries(local[elem_idx], ke, rows=rows, cols=cols, data=data)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported topology {topology!r}")

    if not rows:
        return sp.csr_matrix((freedofs.size, freedofs.size))
    mat = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(freedofs.size, freedofs.size),
    ).tocsr()
    mat.sum_duplicates()
    return mat


def J_triangle_free(u_free, *, u_0, freedofs, elems, dvx, dvy, vol, p):
    """JAX energy callback for the structured triangle mesh."""
    u_full = u_0.at[freedofs].set(u_free)
    u_e = u_full[elems]
    return jnp.sum(energy_integrand(u_e, dvx, dvy, p) * vol)


def I_triangle_free(u_free, *, u_0, freedofs, elems, dvx, dvy, vol, p):
    """JAX quotient callback for the structured triangle mesh."""
    u_full = u_0.at[freedofs].set(u_free)
    u_e = u_full[elems]
    a = jnp.sum(a_integrand(u_e, dvx, dvy, p) * vol)
    b = jnp.sum(b_integrand(u_e) * vol)
    return a ** (1.0 / p) / (b ** 0.25)


def direction_triangle_free(u_free, *, rhs, u_0, freedofs, elems, dvx, dvy, vol, p):
    """Auxiliary convex energy whose minimizer solves the exact direction subproblem."""
    u_full = u_0.at[freedofs].set(u_free)
    u_e = u_full[elems]
    return jnp.sum((1.0 / p) * a_integrand(u_e, dvx, dvy, p) * vol) - jnp.dot(rhs, u_free)


def J_interval_free(u_free, *, u_0, freedofs, elems, dv, vol, p):
    """JAX energy callback for the structured interval mesh."""
    u_full = u_0.at[freedofs].set(u_free)
    u_e = u_full[elems]
    return jnp.sum(energy_integrand_interval(u_e, dv, p) * vol)


def I_interval_free(u_free, *, u_0, freedofs, elems, dv, vol, p):
    """JAX quotient callback for the structured interval mesh."""
    u_full = u_0.at[freedofs].set(u_free)
    u_e = u_full[elems]
    a = jnp.sum(a_integrand_interval(u_e, dv, p) * vol)
    b = jnp.sum(quartic_interval_integrand(u_e) * vol)
    return a ** (1.0 / p) / (b ** 0.25)


def direction_interval_free(u_free, *, rhs, u_0, freedofs, elems, dv, vol, p):
    """Auxiliary convex energy for the exact 1D descent direction."""
    u_full = u_0.at[freedofs].set(u_free)
    u_e = u_full[elems]
    return jnp.sum((1.0 / p) * a_integrand_interval(u_e, dv, p) * vol) - jnp.dot(rhs, u_free)

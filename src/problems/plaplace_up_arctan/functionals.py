"""Discrete functionals for the arctan-resonance p-Laplacian family."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from src.problems.plaplace_up_arctan.common import (
    a_integrand,
    energy_integrand,
    g_integrand,
    lq_integrand,
)


@dataclass(frozen=True)
class StateStats:
    """Discrete state diagnostics for one FE vector."""

    a: float
    b_p: float
    g_term: float
    seminorm_p: float
    lp_norm: float
    J: float


def conjugate_exponent(p: float) -> float:
    p = float(p)
    if p <= 1.0:
        raise ValueError("p must be > 1 for the conjugate exponent")
    return p / (p - 1.0)


def expand_free_vector(u_free: np.ndarray, u_0: np.ndarray, freedofs: np.ndarray) -> np.ndarray:
    u_full = np.asarray(u_0, dtype=np.float64).copy()
    u_full[np.asarray(freedofs, dtype=np.int64)] = np.asarray(u_free, dtype=np.float64)
    return u_full


def triangle_a_full(
    u_full: np.ndarray,
    *,
    elems: np.ndarray,
    dvx: np.ndarray,
    dvy: np.ndarray,
    vol: np.ndarray,
    exponent: float,
) -> float:
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


def triangle_lq_full(
    u_full: np.ndarray,
    *,
    elems: np.ndarray,
    vol: np.ndarray,
    exponent: float,
) -> float:
    u_e = np.asarray(u_full[np.asarray(elems, dtype=np.int64)], dtype=np.float64)
    value = np.asarray(lq_integrand(jnp.asarray(u_e), float(exponent)), dtype=np.float64)
    return float(np.sum(value * np.asarray(vol, dtype=np.float64)))


def triangle_g_full(
    u_full: np.ndarray,
    *,
    elems: np.ndarray,
    vol: np.ndarray,
) -> float:
    u_e = np.asarray(u_full[np.asarray(elems, dtype=np.int64)], dtype=np.float64)
    value = np.asarray(g_integrand(jnp.asarray(u_e)), dtype=np.float64)
    return float(np.sum(value * np.asarray(vol, dtype=np.float64)))


def seminorm_full(params: dict[str, object], u_full: np.ndarray, *, exponent: float | None = None) -> float:
    exponent_eff = float(params["p"] if exponent is None else exponent)
    a = triangle_a_full(
        u_full,
        elems=np.asarray(params["elems"], dtype=np.int64),
        dvx=np.asarray(params["dvx"], dtype=np.float64),
        dvy=np.asarray(params["dvy"], dtype=np.float64),
        vol=np.asarray(params["vol"], dtype=np.float64),
        exponent=exponent_eff,
    )
    return float(a ** (1.0 / exponent_eff)) if a > 0.0 else 0.0


def lp_norm_full(params: dict[str, object], u_full: np.ndarray, *, exponent: float | None = None) -> float:
    exponent_eff = float(params["p"] if exponent is None else exponent)
    b = triangle_lq_full(
        u_full,
        elems=np.asarray(params["elems"], dtype=np.int64),
        vol=np.asarray(params["vol"], dtype=np.float64),
        exponent=exponent_eff,
    )
    return float(b ** (1.0 / exponent_eff)) if b > 0.0 else 0.0


def compute_state_stats_full(params: dict[str, object], u_full: np.ndarray) -> StateStats:
    p = float(params["p"])
    lambda1 = float(params["lambda1"])
    a = triangle_a_full(
        u_full,
        elems=np.asarray(params["elems"], dtype=np.int64),
        dvx=np.asarray(params["dvx"], dtype=np.float64),
        dvy=np.asarray(params["dvy"], dtype=np.float64),
        vol=np.asarray(params["vol"], dtype=np.float64),
        exponent=p,
    )
    b_p = triangle_lq_full(
        u_full,
        elems=np.asarray(params["elems"], dtype=np.int64),
        vol=np.asarray(params["vol"], dtype=np.float64),
        exponent=p,
    )
    g_term = triangle_g_full(
        u_full,
        elems=np.asarray(params["elems"], dtype=np.int64),
        vol=np.asarray(params["vol"], dtype=np.float64),
    )
    seminorm_p = float(a ** (1.0 / p)) if a > 0.0 else 0.0
    lp_norm = float(b_p ** (1.0 / p)) if b_p > 0.0 else 0.0
    J_value = float(a / p - (lambda1 / p) * b_p - g_term)
    return StateStats(
        a=float(a),
        b_p=float(b_p),
        g_term=float(g_term),
        seminorm_p=seminorm_p,
        lp_norm=lp_norm,
        J=J_value,
    )


def compute_state_stats_free(params: dict[str, object], u_free: np.ndarray) -> StateStats:
    u_full = expand_free_vector(
        np.asarray(u_free, dtype=np.float64),
        np.asarray(params["u_0"], dtype=np.float64),
        np.asarray(params["freedofs"], dtype=np.int64),
    )
    return compute_state_stats_full(params, u_full)


def arctan_energy_free(
    u_free,
    *,
    u_0,
    freedofs,
    elems,
    dvx,
    dvy,
    vol,
    p,
    lambda1,
):
    u_full = u_0.at[freedofs].set(u_free)
    u_e = u_full[elems]
    density = energy_integrand(u_e, dvx, dvy, p=p, lambda1=lambda1)
    return jnp.sum(density * vol)


def eigen_quotient_free(
    u_free,
    *,
    u_0,
    freedofs,
    elems,
    dvx,
    dvy,
    vol,
    p,
):
    u_full = u_0.at[freedofs].set(u_free)
    u_e = u_full[elems]
    a = jnp.sum(a_integrand(u_e, dvx, dvy, p) * vol)
    b = jnp.sum(lq_integrand(u_e, p) * vol)
    safe_b = jnp.maximum(b, 1.0e-30)
    return a / safe_b


def neg_arctan_energy_free(
    u_free,
    *,
    u_0,
    freedofs,
    elems,
    dvx,
    dvy,
    vol,
    p,
    lambda1,
):
    return -arctan_energy_free(
        u_free,
        u_0=u_0,
        freedofs=freedofs,
        elems=elems,
        dvx=dvx,
        dvy=dvy,
        vol=vol,
        p=p,
        lambda1=lambda1,
    )


def assemble_stiffness_matrix(params: dict[str, object]) -> sp.csr_matrix:
    elems = np.asarray(params["elems"], dtype=np.int64)
    dvx = np.asarray(params["dvx"], dtype=np.float64)
    dvy = np.asarray(params["dvy"], dtype=np.float64)
    vol = np.asarray(params["vol"], dtype=np.float64)
    freedofs = np.asarray(params["freedofs"], dtype=np.int64)
    n_total = int(np.asarray(params["u_0"], dtype=np.float64).size)
    n_free = int(freedofs.size)
    full_to_free = np.full(n_total, -1, dtype=np.int64)
    full_to_free[freedofs] = np.arange(n_free, dtype=np.int64)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for e_idx in range(elems.shape[0]):
        nodes = elems[e_idx]
        grads = np.column_stack((dvx[e_idx], dvy[e_idx]))
        ke = vol[e_idx] * (grads @ grads.T)
        local = full_to_free[nodes]
        mask = local >= 0
        if not np.any(mask):
            continue
        free_rows = local[mask]
        free_cols = local[mask]
        block = ke[np.ix_(mask, mask)]
        rows.append(np.repeat(free_rows, free_cols.size))
        cols.append(np.tile(free_cols, free_rows.size))
        data.append(block.reshape(-1))
    if not rows:
        return sp.csr_matrix((n_free, n_free))
    matrix = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_free, n_free),
    )
    matrix.sum_duplicates()
    return matrix.tocsr()

"""Validation helpers for the plaplace_u3 thesis benchmark."""

from __future__ import annotations

import numpy as np


def w1p_seminorm_error(
    u_full: np.ndarray,
    u_ref_full: np.ndarray,
    *,
    elems: np.ndarray,
    dvx: np.ndarray,
    dvy: np.ndarray,
    vol: np.ndarray,
    p: float,
) -> float:
    """Return the discrete |u-u_ref|_{1,p,0} seminorm used in the thesis."""
    diff_elem = (
        np.asarray(u_full, dtype=np.float64) - np.asarray(u_ref_full, dtype=np.float64)
    )[np.asarray(elems, dtype=np.int64)]
    grad_x = np.sum(diff_elem * np.asarray(dvx, dtype=np.float64), axis=1)
    grad_y = np.sum(diff_elem * np.asarray(dvy, dtype=np.float64), axis=1)
    grad_norm = (grad_x**2 + grad_y**2) ** (0.5 * float(p))
    return float(np.sum(np.asarray(vol, dtype=np.float64) * grad_norm) ** (1.0 / float(p)))

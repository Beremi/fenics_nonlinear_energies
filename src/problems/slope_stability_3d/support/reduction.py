"""Strength-reduction helpers for the 3D heterogeneous slope-stability benchmark."""

from __future__ import annotations

import numpy as np


def davis_reduction_qp(
    c0_q: np.ndarray,
    phi_q: np.ndarray,
    psi_q: np.ndarray,
    lam: float,
    Davis_type: str = "B",
    *,
    tiny: float = 1.0e-15,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reduced ``c_bar`` and ``sin(phi)`` arrays at quadrature points."""

    c0 = np.asarray(c0_q, dtype=np.float64)
    phi = np.asarray(phi_q, dtype=np.float64)
    psi = np.asarray(psi_q, dtype=np.float64)

    if c0.shape != phi.shape or c0.shape != psi.shape:
        raise ValueError("c0_q, phi_q, and psi_q must have matching shapes")
    if float(lam) <= 0.0:
        raise ValueError("lambda must be positive")

    typ = str(Davis_type).upper()
    if typ == "A":
        beta = np.cos(phi) * np.cos(psi) / np.maximum(
            tiny,
            1.0 - np.sin(phi) * np.sin(psi),
        )
        c0_lambda = beta * c0 / float(lam)
        phi_lambda = np.arctan(beta * np.tan(phi) / float(lam))
        return 2.0 * c0_lambda * np.cos(phi_lambda), np.sin(phi_lambda)

    if typ == "B":
        c01 = c0 / float(lam)
        phi1 = np.arctan(np.tan(phi) / float(lam))
        psi1 = np.arctan(np.tan(psi) / float(lam))
        beta = np.cos(phi1) * np.cos(psi1) / np.maximum(
            tiny,
            1.0 - np.sin(phi1) * np.sin(psi1),
        )
        c0_lambda = beta * c01
        phi_lambda = np.arctan(beta * np.tan(phi1))
        return 2.0 * c0_lambda * np.cos(phi_lambda), np.sin(phi_lambda)

    if typ == "C":
        c01 = c0 / float(lam)
        phi1 = np.arctan(np.tan(phi) / float(lam))
        beta = np.where(
            phi1 > psi,
            np.cos(phi1) * np.cos(psi) / np.maximum(
                tiny,
                1.0 - np.sin(phi1) * np.sin(psi),
            ),
            1.0,
        )
        c0_lambda = beta * c01
        phi_lambda = np.arctan(beta * np.tan(phi1))
        return 2.0 * c0_lambda * np.cos(phi_lambda), np.sin(phi_lambda)

    raise ValueError(f"Unsupported Davis_type {Davis_type!r}")


def davis_b_reduction_qp(
    c0_q: np.ndarray,
    phi_q: np.ndarray,
    psi_q: np.ndarray,
    lam: float,
    *,
    tiny: float = 1.0e-15,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper for the benchmark's Davis-B reduction."""

    return davis_reduction_qp(
        c0_q,
        phi_q,
        psi_q,
        lam,
        Davis_type="B",
        tiny=tiny,
    )

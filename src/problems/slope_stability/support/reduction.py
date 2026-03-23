"""Strength-reduction helpers for the experimental slope-stability prototype."""

from __future__ import annotations

import math


def davis_b_reduction(c0: float, phi_deg: float, psi_deg: float, lam: float) -> tuple[float, float]:
    """Return Davis-B reduced raw cohesion and friction angle in degrees.

    The pure-JAX prototype expects Mohr-Coulomb cohesion ``c`` and later forms
    ``kappa = 2 c cos(phi)`` internally.  The source PETSc code reduces to
    ``c_bar = 2 c_lambda cos(phi_lambda)`` directly, so this helper returns the
    intermediate reduced cohesion ``c_lambda`` instead of ``c_bar``.
    """
    if lam <= 0.0:
        raise ValueError("lambda must be positive")

    phi = math.radians(float(phi_deg))
    psi = math.radians(float(psi_deg))

    c01 = float(c0) / float(lam)
    phi1 = math.atan(math.tan(phi) / float(lam))
    psi1 = math.atan(math.tan(psi) / float(lam))
    beta = math.cos(phi1) * math.cos(psi1) / (1.0 - math.sin(phi1) * math.sin(psi1))
    reduced_cohesion = beta * c01
    phi_lambda = math.atan(beta * math.tan(phi1))
    phi_lambda_deg = math.degrees(phi_lambda)
    return reduced_cohesion, phi_lambda_deg

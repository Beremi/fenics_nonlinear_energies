"""Shared nonlinear-solver repair policies."""

from __future__ import annotations

import numpy as np


def needs_solver_repair(
    result: dict,
    *,
    retry_on_nonfinite: bool = True,
    retry_on_maxit: bool = True,
) -> bool:
    """Return True when a nonlinear result should trigger a repair attempt."""
    msg = str(result.get("message", "")).lower()
    has_nonfinite = (
        not np.isfinite(float(result.get("fun", np.nan)))
        or "non-finite" in msg
        or "nonfinite" in msg
        or "nan" in msg
    )
    hit_newton_maxit = "maximum number of iterations reached" in msg
    if retry_on_nonfinite and has_nonfinite:
        return True
    return bool(retry_on_maxit and hit_newton_maxit)


def build_retry_attempts(
    *,
    retry_on_failure: bool | None = None,
    retry_on_nonfinite: bool | None = None,
    retry_on_maxit: bool | None = None,
    linesearch_interval: tuple[float, float],
    ksp_rtol: float,
    ksp_max_it: int,
    retry_rtol_factor: float = 0.1,
    retry_linesearch_b: float = 1.0,
    retry_ksp_max_it_factor: float = 2.0,
    min_rtol: float = 1e-12,
) -> list[tuple[str, tuple[float, float], float, int]]:
    """Build the standard primary + repair attempt schedule."""
    attempts = [
        (
            "primary",
            (float(linesearch_interval[0]), float(linesearch_interval[1])),
            float(ksp_rtol),
            int(ksp_max_it),
        )
    ]
    wants_repair = bool(retry_on_failure)
    if retry_on_nonfinite is not None or retry_on_maxit is not None:
        wants_repair = bool(retry_on_nonfinite) or bool(retry_on_maxit)
    if not wants_repair:
        return attempts

    ls_a = float(linesearch_interval[0])
    ls_b = float(linesearch_interval[1])
    repair_ls_b = min(ls_b, float(retry_linesearch_b))
    if repair_ls_b <= ls_a:
        repair_ls_b = ls_b
    repair_rtol = max(float(min_rtol), float(ksp_rtol) * float(retry_rtol_factor))
    repair_ksp_max_it = max(
        int(ksp_max_it) + 1,
        int(round(float(ksp_max_it) * float(retry_ksp_max_it_factor))),
    )
    if (
        repair_ls_b < ls_b
        or repair_rtol < float(ksp_rtol)
        or repair_ksp_max_it > int(ksp_max_it)
    ):
        attempts.append(
            (
                "repair",
                (ls_a, repair_ls_b),
                repair_rtol,
                repair_ksp_max_it,
            )
        )
    return attempts

from __future__ import annotations

import numpy as np
import pytest
from petsc4py import PETSc

from src.core.petsc.metrics import DiagonalRieszMetric
from src.core.petsc.minimizers import gradient_descent


def _quadratic_problem():
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, -2.0], dtype=np.float64)
    x_star = np.linalg.solve(A, b)
    return A, b, x_star


def _make_callbacks(A: np.ndarray, b: np.ndarray):
    def energy_fn(x: PETSc.Vec) -> float:
        xa = np.asarray(x.getArray(readonly=True), dtype=np.float64)
        return float(0.5 * xa @ A @ xa - b @ xa)

    def gradient_fn(x: PETSc.Vec, g: PETSc.Vec) -> None:
        xa = np.asarray(x.getArray(readonly=True), dtype=np.float64)
        ga = g.getArray(readonly=False)
        ga[:] = A @ xa - b
        del ga

    return energy_fn, gradient_fn


def _vec_from_array(x0: np.ndarray) -> PETSc.Vec:
    vec = PETSc.Vec().createSeq(len(x0), comm=PETSc.COMM_SELF)
    arr = vec.getArray(readonly=False)
    arr[:] = x0
    del arr
    return vec


def test_petsc_gradient_descent_new_golden_modes_converge_on_quadratic():
    A, b, x_star = _quadratic_problem()
    energy_fn, gradient_fn = _make_callbacks(A, b)

    for mode in ("golden_adaptive", "golden_linf", "golden_gamma_beta"):
        x = _vec_from_array(np.zeros(2, dtype=np.float64))
        res = gradient_descent(
            energy_fn,
            gradient_fn,
            x,
            line_search=mode,
            adaptive_nonnegative=True,
            adaptive_window_scale=2.0,
            maxit=200,
            tolg=1e-8,
            tolf=1e-12,
            linesearch_tol=1e-6,
            save_history=True,
            comm=PETSc.COMM_SELF.tompi4py(),
        )
        xa = np.asarray(res["x"].getArray(readonly=True), dtype=np.float64)
        assert np.linalg.norm(xa - x_star) < 1e-5
        assert any(abs(float(row["alpha"])) > 0.0 for row in res["history"])
        assert all(float(row["ls_a"]) >= -1e-12 for row in res["history"])
        assert all(float(row["ls_b"]) >= float(row["ls_a"]) - 1e-12 for row in res["history"])
        res["x"].destroy()


def test_gradient_descent_uses_configured_dual_norm_for_gradient_gate():
    x = _vec_from_array(np.array([1.0], dtype=np.float64))
    weights = _vec_from_array(np.array([100.0], dtype=np.float64))
    metric = DiagonalRieszMetric(weights, name="design_mass")

    def energy_fn(v: PETSc.Vec) -> float:
        value = float(v.getArray(readonly=True)[0])
        return 0.5 * value * value

    def gradient_fn(v: PETSc.Vec, g: PETSc.Vec) -> None:
        v.copy(g)

    try:
        res = gradient_descent(
            energy_fn,
            gradient_fn,
            x,
            tolg=0.2,
            maxit=2,
            convergence_metric=metric,
            comm=PETSc.COMM_SELF.tompi4py(),
        )
        assert res["message"] == "Gradient norm converged"
        assert res["nit"] == 0
        assert res["grad_norm_coefficient_l2"] == pytest.approx(1.0)
        assert res["dual_residual_norm"] == pytest.approx(0.1)
        assert res["convergence_metric"]["name"] == "design_mass"
    finally:
        metric.destroy()
        weights.destroy()
        x.destroy()


def test_gradient_descent_terminal_audit_and_legacy_correction_ratio():
    def energy_fn(v: PETSc.Vec) -> float:
        value = float(v.getArray(readonly=True)[0])
        return 0.5 * value * value

    def gradient_fn(v: PETSc.Vec, g: PETSc.Vec) -> None:
        v.copy(g)

    legacy_x = _vec_from_array(np.array([2.0], dtype=np.float64))
    metric_x = _vec_from_array(np.array([2.0], dtype=np.float64))
    try:
        legacy = gradient_descent(
            energy_fn,
            gradient_fn,
            legacy_x,
            line_search="armijo",
            maxit=1,
            save_history=True,
            comm=PETSc.COMM_SELF.tompi4py(),
        )
        metric_current = gradient_descent(
            energy_fn,
            gradient_fn,
            metric_x,
            line_search="armijo",
            maxit=1,
            save_history=True,
            convergence_correction_mode="metric_current_state",
            comm=PETSc.COMM_SELF.tompi4py(),
        )
        assert legacy["initial_dual_residual_norm"] == pytest.approx(2.0)
        assert legacy["dual_residual_norm"] == pytest.approx(0.0)
        assert legacy["dual_residual_relative"] == pytest.approx(0.0)
        assert legacy["grad_norm_coefficient_l2"] == pytest.approx(0.0)
        assert legacy["convergence_correction_mode"] == "legacy_coefficient"
        assert legacy["relative_correction"] == pytest.approx(1.0)
        assert metric_current["convergence_correction_mode"] == "metric_current_state"
        assert metric_current["relative_correction"] == pytest.approx(2.0)
    finally:
        legacy_x.destroy()
        metric_x.destroy()

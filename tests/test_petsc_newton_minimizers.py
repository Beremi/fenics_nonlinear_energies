from __future__ import annotations

import numpy as np
import pytest
from petsc4py import PETSc

from src.core.petsc.metrics import DiagonalRieszMetric
from src.core.petsc.minimizers import newton


def _vec_from_array(x0: np.ndarray) -> PETSc.Vec:
    vec = PETSc.Vec().createSeq(len(x0), comm=PETSc.COMM_SELF)
    arr = vec.getArray(readonly=False)
    arr[:] = x0
    del arr
    return vec


def test_hybrid_trust_accepts_roundoff_limited_converged_step():
    x = _vec_from_array(np.array([1.0e-8], dtype=np.float64))

    def energy_fn(v: PETSc.Vec) -> float:
        xa = np.asarray(v.getArray(readonly=True), dtype=np.float64)
        return -2.0e-20 if abs(float(xa[0])) < 0.5e-8 else 0.0

    def gradient_fn(v: PETSc.Vec, g: PETSc.Vec) -> None:
        xa = np.asarray(v.getArray(readonly=True), dtype=np.float64)
        ga = g.getArray(readonly=False)
        ga[:] = xa
        del ga

    def hessian_matvec_fn(_x: PETSc.Vec, vin: PETSc.Vec, vout: PETSc.Vec) -> None:
        vin.copy(vout)

    def trust_solve_fn(_x: PETSc.Vec, rhs: PETSc.Vec, p: PETSc.Vec, _radius: float) -> int:
        rhs.copy(p)
        return 1

    res = newton(
        energy_fn,
        gradient_fn,
        lambda _x, _p, _h: 0,
        x,
        tolf=1e-12,
        tolg=1e-10,
        line_search="armijo",
        maxit=4,
        tolx_rel=1e-6,
        tolx_abs=1e-14,
        require_all_convergence=True,
        hessian_matvec_fn=hessian_matvec_fn,
        trust_subproblem_solve_fn=trust_solve_fn,
        trust_subproblem_line_search=True,
        trust_region=True,
        trust_eta_shrink=0.05,
        save_history=True,
        comm=PETSc.COMM_SELF.tompi4py(),
    )

    assert res["success"]
    assert res["message"] == "Converged (energy, step, gradient)"
    assert res["history"][-1]["used_roundoff_acceptance"] is True
    assert res["history"][-1]["accepted_step"] is True
    assert res["history"][-1]["trust_ratio"] < 0.05
    x.destroy()


def test_newton_uses_configured_dual_norm_for_gradient_gate():
    x = _vec_from_array(np.array([1.0], dtype=np.float64))
    weights = _vec_from_array(np.array([100.0], dtype=np.float64))
    metric = DiagonalRieszMetric(weights, name="scaled_mass")

    def energy_fn(v: PETSc.Vec) -> float:
        value = float(v.getArray(readonly=True)[0])
        return 0.5 * value * value

    def gradient_fn(v: PETSc.Vec, g: PETSc.Vec) -> None:
        v.copy(g)

    try:
        res = newton(
            energy_fn,
            gradient_fn,
            lambda _x, _p, _h: 0,
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
        assert res["convergence_metric"]["name"] == "scaled_mass"
    finally:
        metric.destroy()
        weights.destroy()
        x.destroy()


def test_default_euclidean_newton_preserves_legacy_prestate_correction_ratio():
    def energy_fn(v: PETSc.Vec) -> float:
        value = float(v.getArray(readonly=True)[0])
        return 0.5 * value * value

    def gradient_fn(v: PETSc.Vec, g: PETSc.Vec) -> None:
        v.copy(g)

    def identity_solve(_x: PETSc.Vec, rhs: PETSc.Vec, step: PETSc.Vec) -> int:
        rhs.copy(step)
        return 1

    legacy_x = _vec_from_array(np.array([2.0], dtype=np.float64))
    metric_x = _vec_from_array(np.array([2.0], dtype=np.float64))
    try:
        legacy = newton(
            energy_fn,
            gradient_fn,
            identity_solve,
            legacy_x,
            line_search="armijo",
            maxit=1,
            save_history=True,
            comm=PETSc.COMM_SELF.tompi4py(),
        )
        metric_current = newton(
            energy_fn,
            gradient_fn,
            identity_solve,
            metric_x,
            line_search="armijo",
            maxit=1,
            save_history=True,
            convergence_correction_mode="metric_current_state",
            comm=PETSc.COMM_SELF.tompi4py(),
        )
        assert legacy["convergence_correction_mode"] == "legacy_coefficient"
        assert legacy["history"][0]["step_norm"] == pytest.approx(2.0)
        assert legacy["relative_correction"] == pytest.approx(1.0)
        assert metric_current["convergence_correction_mode"] == "metric_current_state"
        assert metric_current["relative_correction"] == pytest.approx(2.0)
    finally:
        legacy_x.destroy()
        metric_x.destroy()


def test_armijo_failure_is_terminal_and_retains_the_last_accepted_state():
    x = _vec_from_array(np.array([1.0], dtype=np.float64))

    def energy_fn(v: PETSc.Vec) -> float:
        value = float(v.getArray(readonly=True)[0])
        return 0.5 * value * value

    def gradient_fn(v: PETSc.Vec, g: PETSc.Vec) -> None:
        v.copy(g)

    def ascent_solve(_x: PETSc.Vec, rhs: PETSc.Vec, step: PETSc.Vec) -> int:
        rhs.copy(step)
        step.scale(-1.0)
        return 1

    try:
        result = newton(
            energy_fn,
            gradient_fn,
            ascent_solve,
            x,
            line_search="armijo",
            armijo_max_ls=4,
            maxit=8,
            save_history=True,
            comm=PETSc.COMM_SELF.tompi4py(),
        )

        assert not result["success"]
        assert result["nit"] == 1
        assert result["message"] == "armijo line search failed at Newton iteration 1"
        assert len(result["history"]) == 1
        assert result["history"][0]["accepted_step"] is False
        assert result["history"][0]["ls_evals"] == 4
        np.testing.assert_allclose(x.getArray(readonly=True), np.array([1.0]))
    finally:
        x.destroy()


def test_vanishing_newton_direction_is_terminal_before_gradient_convergence():
    x = _vec_from_array(np.array([1.0], dtype=np.float64))

    def energy_fn(v: PETSc.Vec) -> float:
        value = float(v.getArray(readonly=True)[0])
        return 0.5 * value * value

    def gradient_fn(v: PETSc.Vec, g: PETSc.Vec) -> None:
        v.copy(g)

    def zero_solve(_x: PETSc.Vec, _rhs: PETSc.Vec, step: PETSc.Vec) -> int:
        step.set(0.0)
        return 1

    try:
        result = newton(
            energy_fn,
            gradient_fn,
            zero_solve,
            x,
            line_search="armijo",
            maxit=8,
            save_history=True,
            comm=PETSc.COMM_SELF.tompi4py(),
        )

        assert not result["success"]
        assert result["nit"] == 1
        assert result["message"] == (
            "Newton direction vanished before gradient convergence at Newton iteration 1"
        )
        assert len(result["history"]) == 1
        assert result["history"][0]["accepted_step"] is False
        np.testing.assert_allclose(x.getArray(readonly=True), np.array([1.0]))
    finally:
        x.destroy()


@pytest.mark.parametrize(
    ("eta", "accepted", "message"),
    [
        (0.05, True, "Gradient norm converged"),
        (0.0500001, False, "Trust-region rejected all candidate steps at Newton iteration 1"),
    ],
)
def test_trust_acceptance_threshold_is_inclusive(
    eta: float,
    accepted: bool,
    message: str,
):
    x = _vec_from_array(np.array([1.0], dtype=np.float64))

    def energy_fn(v: PETSc.Vec) -> float:
        value = float(v.getArray(readonly=True)[0])
        return 0.5 * value * value

    def gradient_fn(v: PETSc.Vec, g: PETSc.Vec) -> None:
        v.copy(g)

    def negative_curvature_action(
        _x: PETSc.Vec,
        vin: PETSc.Vec,
        vout: PETSc.Vec,
    ) -> None:
        vin.copy(vout)
        vout.scale(-18.0)

    def trust_step(
        _x: PETSc.Vec,
        rhs: PETSc.Vec,
        step: PETSc.Vec,
        _radius: float,
    ) -> int:
        rhs.copy(step)
        return 1

    try:
        result = newton(
            energy_fn,
            gradient_fn,
            lambda _x, _rhs, _step: 0,
            x,
            tolg=1.0e-12,
            maxit=2,
            require_all_convergence=False,
            hessian_matvec_fn=negative_curvature_action,
            trust_subproblem_solve_fn=trust_step,
            trust_subproblem_line_search=False,
            trust_region=True,
            trust_radius_init=1.0,
            trust_eta_shrink=eta,
            trust_eta_expand=0.75,
            trust_max_reject=0,
            save_history=True,
            comm=PETSc.COMM_SELF.tompi4py(),
        )

        assert result["message"] == message
        assert result["history"][0]["accepted_step"] is accepted
        assert result["history"][0]["trust_ratio"] == pytest.approx(0.05)
        assert result["history"][0]["trust_qp"] < 0.0
        expected = np.array([0.0 if accepted else 1.0])
        np.testing.assert_allclose(x.getArray(readonly=True), expected)
    finally:
        x.destroy()


def test_reduced_subspace_trust_honors_armijo_policy():
    x = _vec_from_array(np.array([1.0], dtype=np.float64))
    evaluated: list[float] = []

    def energy_fn(v: PETSc.Vec) -> float:
        value = float(v.getArray(readonly=True)[0])
        evaluated.append(value)
        return 0.5 * value * value

    def gradient_fn(v: PETSc.Vec, g: PETSc.Vec) -> None:
        v.copy(g)

    def hessian_solve(_x: PETSc.Vec, rhs: PETSc.Vec, step: PETSc.Vec) -> int:
        rhs.copy(step)
        return 1

    def hessian_action(
        _x: PETSc.Vec,
        direction: PETSc.Vec,
        output: PETSc.Vec,
    ) -> None:
        direction.copy(output)

    try:
        result = newton(
            energy_fn,
            gradient_fn,
            hessian_solve,
            x,
            line_search="armijo",
            armijo_alpha0=0.25,
            maxit=1,
            trust_region=True,
            trust_radius_init=1.0,
            hessian_matvec_fn=hessian_action,
            save_history=True,
            comm=PETSc.COMM_SELF.tompi4py(),
        )

        assert result["history"][0]["alpha"] == pytest.approx(0.25)
        assert result["history"][0]["ls_evals"] == 1
        assert result["history"][0]["accepted_step"] is True
        np.testing.assert_allclose(x.getArray(readonly=True), np.array([0.75]))
        # Initial energy, trial energy, and final bookkeeping energy are enough;
        # golden-section search would add many interior evaluations.
        assert len(evaluated) <= 4
    finally:
        x.destroy()

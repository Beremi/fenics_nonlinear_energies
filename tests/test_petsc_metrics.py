from __future__ import annotations

import numpy as np
import pytest
from petsc4py import PETSc

from src.core.petsc.metrics import DiagonalRieszMetric, EuclideanMetric, MatrixRieszMetric


def _vec(values: list[float]) -> PETSc.Vec:
    vec = PETSc.Vec().createSeq(len(values), comm=PETSc.COMM_SELF)
    vec.setArray(np.asarray(values, dtype=np.float64))
    return vec


def _diag(values: list[float]) -> PETSc.Mat:
    mat = PETSc.Mat().createAIJ([len(values), len(values)], comm=PETSc.COMM_SELF)
    mat.setUp()
    for idx, value in enumerate(values):
        mat.setValue(idx, idx, float(value))
    mat.assemblyBegin()
    mat.assemblyEnd()
    return mat


def test_euclidean_metric_matches_petsc_norm() -> None:
    vec = _vec([3.0, 4.0])
    other = _vec([0.0, 0.0])
    metric = EuclideanMetric()
    try:
        assert metric.primal_norm(vec).value == pytest.approx(5.0)
        assert metric.dual_norm(vec).value == pytest.approx(5.0)
        assert metric.distance(vec, other).value == pytest.approx(5.0)
    finally:
        vec.destroy()
        other.destroy()


def test_diagonal_riesz_metric_primal_and_dual_norms() -> None:
    weights = _vec([2.0, 8.0])
    vec = _vec([1.0, 2.0])
    zero = _vec([0.0, 0.0])
    metric = DiagonalRieszMetric(weights, name="lumped_mass")
    try:
        assert metric.primal_norm(vec).value == pytest.approx(np.sqrt(34.0))
        assert metric.dual_norm(vec).value == pytest.approx(1.0)
        assert metric.distance(vec, zero).value == pytest.approx(np.sqrt(34.0))
    finally:
        metric.destroy()
        weights.destroy()
        vec.destroy()
        zero.destroy()


@pytest.mark.parametrize("bad_weight", [0.0, np.inf, np.nan])
def test_diagonal_riesz_metric_rejects_invalid_free_weight(bad_weight: float) -> None:
    weights = _vec([1.0, bad_weight])
    try:
        with pytest.raises(ValueError, match="strictly positive"):
            DiagonalRieszMetric(weights, name="invalid")
    finally:
        weights.destroy()


def test_matrix_riesz_petsc_options_are_disabled_by_default_and_namespaced() -> None:
    operator = _diag([2.0, 8.0])
    options = PETSc.Options()
    unprefixed_key = "ksp_rtol"
    prefixed_key = "unit_riesz_ksp_rtol"
    previous_unprefixed = (
        (unprefixed_key in options),
        options[unprefixed_key] if unprefixed_key in options else None,
    )
    previous_prefixed = (
        (prefixed_key in options),
        options[prefixed_key] if prefixed_key in options else None,
    )
    default_metric = None
    configured_metric = None
    try:
        options[unprefixed_key] = 0.25
        default_metric = MatrixRieszMetric(operator, name="default_options")
        default_description = default_metric.describe()
        assert default_description["set_from_petsc_options"] is False
        assert default_description["effective_rtol"] == pytest.approx(1.0e-10)

        options[prefixed_key] = 0.125
        configured_metric = MatrixRieszMetric(
            operator,
            name="configured_options",
            set_from_options=True,
            options_prefix="unit_riesz",
        )
        configured_description = configured_metric.describe()
        assert configured_description["petsc_options_prefix"] == "unit_riesz_"
        assert configured_description["requested_rtol"] == pytest.approx(1.0e-10)
        assert configured_description["effective_rtol"] == pytest.approx(0.125)
    finally:
        if configured_metric is not None:
            configured_metric.destroy()
        if default_metric is not None:
            default_metric.destroy()
        for key, (present, value) in {
            unprefixed_key: previous_unprefixed,
            prefixed_key: previous_prefixed,
        }.items():
            if present:
                options[key] = value
            elif key in options:
                del options[key]
        operator.destroy()


def test_matrix_riesz_metric_reports_certified_dual_solve() -> None:
    operator = _diag([2.0, 8.0])
    vec = _vec([1.0, 2.0])
    zero = _vec([0.0, 0.0])
    metric = MatrixRieszMetric(operator, name="reference_energy", pc_type="jacobi")
    try:
        assert metric.primal_norm(vec).value == pytest.approx(np.sqrt(34.0))
        dual = metric.dual_norm(vec)
        assert dual.value == pytest.approx(1.0)
        assert dual.metadata["reason"] > 0
        assert dual.metadata["relative_true_residual"] <= 1.0e-12
        assert metric.first_dual_evaluation is not None
        assert metric.first_dual_evaluation.value == pytest.approx(1.0)
        assert metric.distance(vec, zero).value == pytest.approx(np.sqrt(34.0))
    finally:
        metric.destroy()
        operator.destroy()
        vec.destroy()
        zero.destroy()


def test_riesz_metrics_reject_nonfinite_quadratic_forms() -> None:
    weights = _vec([2.0, 8.0])
    operator = _diag([2.0, 8.0])
    vec = _vec([np.nan, 1.0])
    diagonal = DiagonalRieszMetric(weights, name="diagonal_nonfinite")
    matrix = MatrixRieszMetric(operator, name="matrix_nonfinite", pc_type="jacobi")
    try:
        with pytest.raises(ValueError, match="nonfinite"):
            diagonal.primal_norm(vec)
        with pytest.raises(ValueError, match="nonfinite"):
            diagonal.dual_norm(vec)
        with pytest.raises(ValueError, match="nonfinite"):
            matrix.primal_norm(vec)
    finally:
        matrix.destroy()
        diagonal.destroy()
        vec.destroy()
        operator.destroy()
        weights.destroy()

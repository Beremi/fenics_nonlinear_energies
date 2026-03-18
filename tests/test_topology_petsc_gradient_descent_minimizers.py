from __future__ import annotations

import numpy as np
from petsc4py import PETSc

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

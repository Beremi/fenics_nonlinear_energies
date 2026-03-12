from __future__ import annotations

import numpy as np

from tools.minimizers import gradient_descent


def _quadratic_problem():
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, -2.0], dtype=np.float64)

    def f(x):
        x = np.asarray(x, dtype=np.float64)
        return 0.5 * x @ A @ x - b @ x

    def df(x):
        x = np.asarray(x, dtype=np.float64)
        return A @ x - b

    x_star = np.linalg.solve(A, b)
    return f, df, x_star


def test_gradient_descent_armijo_converges_on_quadratic():
    f, df, x_star = _quadratic_problem()
    res = gradient_descent(
        f,
        df,
        np.zeros(2, dtype=np.float64),
        line_search="armijo",
        maxit=200,
        tolg=1e-8,
        tolf=1e-12,
        save_history=True,
    )
    assert np.linalg.norm(res["x"] - x_star) < 1e-5
    assert res["nit"] > 0
    assert res["last_alpha_abs"] > 0.0


def test_gradient_descent_golden_modes_converge_on_quadratic():
    f, df, x_star = _quadratic_problem()
    for mode in ("golden_fixed", "golden_adaptive"):
        res = gradient_descent(
            f,
            df,
            np.zeros(2, dtype=np.float64),
            line_search=mode,
            maxit=200,
            tolg=1e-8,
            tolf=1e-12,
            linesearch_tol=1e-6,
            save_history=True,
        )
        assert np.linalg.norm(res["x"] - x_star) < 1e-5
        assert any(abs(float(row["alpha"])) > 0.0 for row in res["history"])
        assert res["last_alpha_abs"] > 0.0


def test_gradient_descent_positive_adaptive_bracket_stays_nonnegative():
    f, df, x_star = _quadratic_problem()
    res = gradient_descent(
        f,
        df,
        np.zeros(2, dtype=np.float64),
        line_search="golden_adaptive",
        adaptive_nonnegative=True,
        maxit=200,
        tolg=1e-8,
        tolf=1e-12,
        linesearch_tol=1e-6,
        save_history=True,
    )
    assert np.linalg.norm(res["x"] - x_star) < 1e-5
    assert all(float(row["alpha"]) >= 0.0 for row in res["history"])

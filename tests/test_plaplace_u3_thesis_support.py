from __future__ import annotations

import math

import numpy as np

from src.problems.plaplace_u3.thesis.directions import DirectionContext
from src.problems.plaplace_u3.thesis.functionals import (
    compute_state_stats_free,
    rescale_free_to_solution,
)
from src.problems.plaplace_u3.thesis.solver_common import build_objective_bundle, build_problem
from src.problems.plaplace_u3.thesis.transfer import (
    nested_w1p_error,
    prolong_free_to_problem,
    same_mesh_w1p_error,
)


def test_thesis_I_is_scale_invariant():
    problem = build_problem(
        dimension=2,
        level=2,
        p=2.0,
        geometry="square_pi",
        init_mode="sine",
        seed=0,
    )
    stats = compute_state_stats_free(problem.params, problem.u_init)
    scaled = 3.7 * np.asarray(problem.u_init, dtype=np.float64)
    scaled_stats = compute_state_stats_free(problem.params, scaled)
    assert math.isclose(stats.I, scaled_stats.I, rel_tol=1e-10, abs_tol=1e-10)


def test_thesis_u_tilde_balances_a_and_b():
    problem = build_problem(
        dimension=2,
        level=2,
        p=2.5,
        geometry="square_pi",
        init_mode="sine",
        seed=0,
    )
    scaled_free, _, stats = rescale_free_to_solution(problem.params, problem.u_init)
    assert np.linalg.norm(scaled_free) > 0.0
    assert math.isclose(stats.a, stats.b, rel_tol=1e-10, abs_tol=1e-10)


def test_nested_transfer_preserves_prolonged_target_state():
    coarse = build_problem(
        dimension=2,
        level=2,
        p=2.0,
        geometry="square_pi",
        init_mode="sine",
        seed=0,
    )
    fine = build_problem(
        dimension=2,
        level=3,
        p=2.0,
        geometry="square_pi",
        init_mode="sine",
        seed=0,
    )
    coarse_scaled, _, _ = rescale_free_to_solution(coarse.params, coarse.u_init)
    coarse_on_fine = prolong_free_to_problem(coarse.params, coarse_scaled, fine.params)
    err = nested_w1p_error(coarse.params, coarse_scaled, fine.params, coarse_on_fine)
    assert math.isclose(err, 0.0, abs_tol=1e-12)


def test_same_mesh_error_is_zero_for_identical_state():
    problem = build_problem(
        dimension=2,
        level=2,
        p=2.0,
        geometry="square_pi",
        init_mode="sine",
        seed=0,
    )
    scaled_free, _, _ = rescale_free_to_solution(problem.params, problem.u_init)
    err = same_mesh_w1p_error(problem.params, scaled_free, scaled_free)
    assert math.isclose(err, 0.0, abs_tol=1e-12)


def test_all_thesis_directions_are_descents():
    problem = build_problem(
        dimension=2,
        level=2,
        p=2.0,
        geometry="square_pi",
        init_mode="sine",
        seed=0,
    )
    objective = build_objective_bundle(problem, "J")
    ctx = DirectionContext(problem, objective)

    for direction in ("d_vh", "d_rn", "d"):
        result = ctx.compute(np.asarray(problem.u_init, dtype=np.float64), direction)
        assert np.isfinite(result.stop_measure)
        assert result.descent_value <= 1.0e-10

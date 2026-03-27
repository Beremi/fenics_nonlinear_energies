from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.problems.plaplace_up_arctan.common import (
    F_helper,
    G_arctan_shifted,
    arctan_shifted,
    arctan_shifted_prime,
)
from src.problems.plaplace_up_arctan.directions import DIRECTION_MODEL_DVH
from src.problems.plaplace_up_arctan.eigen import compute_lambda1_cached
from src.problems.plaplace_up_arctan.ray_audit import audit_ray_profile
from src.problems.plaplace_up_arctan.solver_common import build_objective_bundle
from src.problems.plaplace_up_arctan.solver_common import build_problem
from src.problems.plaplace_up_arctan.solver_mpa import run_mpa, run_mpa_symmetric
from src.problems.plaplace_up_arctan.solver_rmpa import run_rmpa, run_rmpa_shifted
from src.problems.plaplace_up_arctan.transfer import nested_w1p_error, prolong_free_to_problem, same_mesh_w1p_error


def test_arctan_shifted_formulas_and_asymptotics() -> None:
    assert abs(float(arctan_shifted(0.0)) - (np.pi / 4.0)) < 1.0e-12
    assert abs(float(arctan_shifted_prime(0.0)) - 0.5) < 1.0e-12
    assert abs(float(G_arctan_shifted(0.0))) < 1.0e-12

    large = 1.0e6
    for p in (2.0, 3.0):
        expected = 0.5 * (p - 1.0) * np.pi
        assert abs(float(F_helper(large, p)) - expected) < 5.0e-5
        assert abs(float(F_helper(-large, p)) + expected) < 5.0e-5


def test_nested_transfer_and_w1p_errors_are_consistent() -> None:
    coarse = build_problem(
        level=1,
        p=2.0,
        geometry="square_unit",
        init_mode="sine",
        lambda1=float(2.0 * np.pi**2),
        lambda_level=1,
        seed=0,
    )
    fine = build_problem(
        level=2,
        p=2.0,
        geometry="square_unit",
        init_mode="sine",
        lambda1=float(2.0 * np.pi**2),
        lambda_level=2,
        seed=0,
    )
    prolonged = prolong_free_to_problem(coarse.params, coarse.u_init, fine.params)
    assert prolonged.shape == fine.u_init.shape
    assert np.all(np.isfinite(prolonged))
    assert same_mesh_w1p_error(fine.params, prolonged, prolonged) == 0.0
    assert nested_w1p_error(coarse.params, coarse.u_init, fine.params, prolonged) < 1.0e-12


class _SyntheticRayObjective:
    def __init__(self, fn, name: str) -> None:
        self._fn = fn
        self.name = name

    def value(self, u: np.ndarray) -> float:
        x = float(np.asarray(u, dtype=np.float64)[0])
        return float(self._fn(x))


def test_ray_audit_detects_synthetic_minimum_and_maximum_profiles() -> None:
    base = np.asarray([1.0], dtype=np.float64)
    min_objective = _SyntheticRayObjective(lambda x: (x - 2.0) ** 2 + 1.0, "synthetic_min")
    max_objective = _SyntheticRayObjective(lambda x: -((x - 2.0) ** 2) + 4.0, "synthetic_max")

    min_payload = audit_ray_profile(None, min_objective, base, t_values=np.linspace(0.0, 4.0, 41))
    max_payload = audit_ray_profile(None, max_objective, base, t_values=np.linspace(0.0, 4.0, 41))

    assert min_payload["status"] == "ok"
    assert max_payload["status"] == "ok"
    assert min_payload["best_kind"] == "minimum"
    assert max_payload["best_kind"] == "maximum"
    assert min_payload["stable_interior_extremum"] is True
    assert max_payload["stable_interior_extremum"] is True
    assert 1 < int(min_payload["best_index"]) < 39
    assert 1 < int(max_payload["best_index"]) < 39
    assert abs(float(min_payload["best_t"]) - 2.0) < 0.15
    assert abs(float(max_payload["best_t"]) - 2.0) < 0.15
    json.dumps(min_payload)
    json.dumps(max_payload)


def test_ray_audit_detects_small_problem_branch_geometry() -> None:
    problem = build_problem(
        level=2,
        p=2.0,
        geometry="square_unit",
        init_mode="sine",
        lambda1=float(2.0 * np.pi**2),
        lambda_level=2,
        seed=0,
    )
    objective = build_objective_bundle(problem, "J")
    payload = audit_ray_profile(problem, objective, problem.u_init, num_samples=61, t_max=6.0)

    assert payload["status"] == "ok"
    assert payload["best_kind"] == "minimum"
    assert payload["stable_interior_extremum"] is True
    assert payload["best_index"] not in (None, 0, 60)
    assert 0.0 < float(payload["best_t"]) < 6.0
    assert float(payload["best_value"]) < float(payload["base_value"])
    assert float(payload["endpoint_gap"]) > 0.0
    json.dumps(payload)


def test_p2_solver_smokes_emit_state_files(tmp_path: Path) -> None:
    problem = build_problem(
        level=2,
        p=2.0,
        geometry="square_unit",
        init_mode="sine",
        lambda1=float(2.0 * np.pi**2),
        lambda_level=2,
        seed=0,
    )
    for method, runner in (
        ("rmpa", lambda out: run_rmpa(problem, epsilon=1.0e-3, maxit=3, delta0=1.0, state_out=str(out))),
        (
            "rmpa_shifted",
            lambda out: run_rmpa_shifted(problem, epsilon=1.0e-3, maxit=3, delta0=1.0, state_out=str(out)),
        ),
        (
            "mpa",
            lambda out: run_mpa(
                problem,
                epsilon=1.0e-3,
                maxit=2,
                num_nodes=10,
                rho=1.0,
                segment_tol_factor=0.125,
                state_out=str(out),
            ),
        ),
        (
            "mpa_symmetric",
            lambda out: run_mpa_symmetric(
                problem,
                epsilon=1.0e-3,
                maxit=2,
                num_nodes=10,
                rho=1.0,
                segment_tol_factor=0.125,
                state_out=str(out),
            ),
        ),
    ):
        state_path = tmp_path / f"{method}.npz"
        payload = runner(state_path)
        assert payload["method"] == method
        assert payload["status"] in {"completed", "maxit", "failed"}
        assert np.isfinite(payload["J"])
        assert np.isfinite(payload["residual_norm"])
        assert payload["direction_model"] == DIRECTION_MODEL_DVH
        if payload["history"]:
            assert "direction_model" in payload["history"][0]
        assert state_path.exists()


def test_p3_eigen_stage_and_solver_smoke(tmp_path: Path) -> None:
    cache_path = tmp_path / "lambda_p3_l2.json"
    state_path = tmp_path / "lambda_p3_l2_state.npz"
    lambda_payload = compute_lambda1_cached(
        cache_path=cache_path,
        state_out=state_path,
        level=2,
        geometry="square_unit",
        init_mode="sine",
        seed=0,
        force=True,
    )
    eigenfunction = np.asarray(lambda_payload["eigenfunction_free"], dtype=np.float64)
    assert lambda_payload["status"] in {"completed", "maxit", "failed"}
    assert float(lambda_payload["lambda1"]) > 0.0
    assert abs(float(lambda_payload["normalization_error"])) < 1.0e-8
    assert np.min(eigenfunction) > 0.0
    assert state_path.exists()

    problem = build_problem(
        level=2,
        p=3.0,
        geometry="square_unit",
        init_mode="sine",
        lambda1=float(lambda_payload["lambda1"]),
        lambda_level=int(lambda_payload["lambda_level"]),
        seed=0,
    )
    result = run_rmpa(
        problem,
        epsilon=1.0e-3,
        maxit=3,
        delta0=1.0,
        init_free=eigenfunction,
        state_out=str(tmp_path / "p3_rmpa.npz"),
    )
    assert result["status"] in {"completed", "maxit", "failed"}
    assert np.isfinite(result["J"])
    assert np.isfinite(result["residual_norm"])
    assert result["direction_model"] == DIRECTION_MODEL_DVH
    assert (tmp_path / "p3_rmpa.npz").exists()

    shifted = run_rmpa_shifted(
        problem,
        epsilon=1.0e-3,
        maxit=3,
        delta0=1.0,
        init_free=eigenfunction,
        state_out=str(tmp_path / "p3_rmpa_shifted.npz"),
    )
    assert shifted["status"] in {"completed", "maxit", "failed"}
    assert np.isfinite(shifted["J"])
    assert np.isfinite(shifted["residual_norm"])
    assert shifted["direction_model"] == DIRECTION_MODEL_DVH
    assert (tmp_path / "p3_rmpa_shifted.npz").exists()


def test_compute_lambda1_cached_reuses_completed_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "lambda_p3_l2.json"
    state_path = tmp_path / "lambda_p3_l2_state.npz"
    first = compute_lambda1_cached(
        cache_path=cache_path,
        state_out=state_path,
        level=2,
        geometry="square_unit",
        init_mode="sine",
        seed=0,
        force=True,
    )
    second = compute_lambda1_cached(
        cache_path=cache_path,
        state_out=state_path,
        level=2,
        geometry="square_unit",
        init_mode="sine",
        seed=0,
        force=False,
    )
    assert first["lambda1"] == second["lambda1"]
    assert json.loads(cache_path.read_text(encoding="utf-8"))["lambda1"] == second["lambda1"]

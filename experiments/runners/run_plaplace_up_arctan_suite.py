#!/usr/bin/env python3
"""Run the arctan-resonance p-Laplacian study and write one summary."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.core.serial.minimizers import newton
from src.core.serial.sparse_solvers import HessSolverGenerator
from src.problems.plaplace_up_arctan.directions import DIRECTION_MODEL_DVH
from src.problems.plaplace_up_arctan.eigen import compute_lambda1_cached
from src.problems.plaplace_up_arctan.ray_audit import audit_ray_profile
from src.problems.plaplace_up_arctan.seeds import candidate_start_seed_names, named_start_seed
from src.problems.plaplace_up_arctan.solver_common import (
    ARCTAN_SOLVER_REVISION,
    build_objective_bundle,
    build_problem,
    residual_metrics,
)
from src.problems.plaplace_up_arctan.transfer import (
    nested_w1p_error,
    prolong_free_to_problem,
    same_mesh_w1p_error,
)
from src.problems.plaplace_up_arctan.workflow import certify_from_iterate, run_raw_method


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "raw_results" / "plaplace_up_arctan_full"
DEFAULT_SUMMARY = DEFAULT_OUT_DIR / "summary.json"
GEOMETRY = "square_unit"
SUITE_MPA_MAXIT = 360
SUITE_RMPA_MAXIT = 240
DEFAULT_CERTIFIED_TOL = 1.0e-8
DEFAULT_POLISH_MAXIT = 80
DEFAULT_CONTINUATION_STEP = 0.2
DEFAULT_CONTINUATION_STEP_FLOOR = 0.025
REF_LEVEL = 7
REFINEMENT_LEVELS = (4, 5, 6)
TOL_LEVEL = 6
EPSILONS = (1.0e-4, 1.0e-5, 1.0e-6)
P_VALUES = (2.0, 3.0)
METHODS = ("mpa", "rmpa")
RUNNER_REVISION = "rawJ_dvh_certified_continuation_v2"


@dataclass(frozen=True)
class Case:
    study: str
    method: str
    p: float
    level: int
    epsilon: float


def _case_name(case: Case) -> str:
    p_tag = f"p{int(case.p)}"
    eps_tag = str(case.epsilon).replace(".", "p").replace("-", "m")
    return f"{case.study}_{case.method}_{p_tag}_l{case.level}_eps{eps_tag}"


def _p_tag(p: float) -> str:
    value = float(p)
    if abs(value - round(value)) <= 1.0e-12:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _lambda_cache_dir(out_dir: Path) -> Path:
    return out_dir / "lambda_cache"


def _lambda_cache_path(out_dir: Path, level: int, p: float) -> Path:
    tag = _p_tag(p)
    if abs(float(p) - 3.0) <= 1.0e-12:
        return _lambda_cache_dir(out_dir) / f"lambda_p3_l{int(level)}.json"
    return _lambda_cache_dir(out_dir) / f"lambda_p{tag}_l{int(level)}.json"


def _lambda_state_path(out_dir: Path, level: int, p: float) -> Path:
    tag = _p_tag(p)
    if abs(float(p) - 3.0) <= 1.0e-12:
        return _lambda_cache_dir(out_dir) / f"lambda_p3_l{int(level)}_state.npz"
    return _lambda_cache_dir(out_dir) / f"lambda_p{tag}_l{int(level)}_state.npz"


def _reference_path(out_dir: Path, name: str) -> Path:
    return out_dir / "references" / name


def _payload_has_residual_diagnostics(payload: dict[str, object]) -> bool:
    if "gradient_residual_norm" not in payload or "residual_norm" not in payload:
        return False
    history = list(payload.get("history", []))
    if history and "dual_residual_norm" not in history[-1]:
        return False
    direction_model = str(payload.get("direction_model", ""))
    return direction_model in {DIRECTION_MODEL_DVH, "jax_autodiff_stationary_newton", "jax_autodiff_newton_reference"}


def _case_payload_valid(payload: dict[str, object]) -> bool:
    if str(payload.get("runner_revision")) != RUNNER_REVISION:
        return False
    raw = payload.get("raw")
    if not isinstance(raw, dict) or not _payload_has_residual_diagnostics(raw):
        return False
    certified = payload.get("certified")
    if certified is not None and (not isinstance(certified, dict) or not _payload_has_residual_diagnostics(certified)):
        return False
    return True


def _lambda_payload(
    out_dir: Path,
    level: int,
    *,
    p: float = 3.0,
    init_free: np.ndarray | None = None,
    force: bool = False,
) -> dict[str, object]:
    if abs(float(p) - 2.0) <= 1.0e-12:
        return {
            "status": "completed",
            "message": "exact",
            "level": int(level),
            "p": 2.0,
            "geometry": GEOMETRY,
            "lambda1": float(2.0 * np.pi**2),
            "lambda_level": int(level),
        }
    cache_path = _lambda_cache_path(out_dir, level, p)
    state_out = _lambda_state_path(out_dir, level, p)
    if cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if str(cached.get("status")) == "completed" and cached.get("state_out") and Path(str(cached["state_out"])).exists():
            return cached
    return compute_lambda1_cached(
        cache_path=cache_path,
        state_out=state_out,
        p=float(p),
        level=int(level),
        geometry=GEOMETRY,
        init_mode="sine",
        seed=0,
        maxit=120,
        epsilon=2.0e-6,
        force=True,
        init_free=None if init_free is None else np.asarray(init_free, dtype=np.float64),
    )


def _select_init_candidates(problem, case: Case, lambda_payload: dict[str, object] | None) -> list[tuple[str, np.ndarray]]:
    init_candidates: list[tuple[str, np.ndarray]] = []
    has_eigenfunction = lambda_payload is not None and "eigenfunction_free" in lambda_payload
    for seed_name in candidate_start_seed_names(
        p=float(case.p),
        method=str(case.method),
        has_eigenfunction=has_eigenfunction,
    ):
        init_candidates.append(
            (
                seed_name,
                named_start_seed(
                    problem,
                    seed_name,
                    eigenfunction_free=None if lambda_payload is None else np.asarray(lambda_payload["eigenfunction_free"], dtype=np.float64),
                ),
            )
        )
    return init_candidates


def _seed_attempt_score(result: dict[str, object]) -> tuple[float, int, int]:
    status = str(result.get("status", "failed"))
    status_rank = {"completed": 0, "maxit": 1, "failed": 2}.get(status, 3)
    residual = float(result.get("residual_norm", float("inf")))
    iterations = int(result.get("outer_iterations", 10**9))
    return residual, status_rank, iterations


def _run_with_seed_candidates(
    case: Case,
    *,
    problem,
    init_candidates: list[tuple[str, np.ndarray]],
    raw_state_out: str,
) -> dict[str, object]:
    attempts: list[dict[str, object]] = []
    best_result: dict[str, object] | None = None
    best_seed_name = ""
    raw_maxit = SUITE_MPA_MAXIT if str(case.method) == "mpa" else SUITE_RMPA_MAXIT

    for idx, (seed_name, init_free) in enumerate(init_candidates):
        result = run_raw_method(
            problem,
            method=str(case.method),
            epsilon=float(case.epsilon),
            maxit=raw_maxit,
            init_free=init_free,
            state_out=raw_state_out if idx == 0 else "",
        )
        result["start_seed_name"] = str(seed_name)
        attempts.append(
            {
                "seed_name": str(seed_name),
                "status": str(result["status"]),
                "residual_norm": float(result["residual_norm"]),
                "gradient_residual_norm": float(result.get("gradient_residual_norm", result["residual_norm"])),
                "outer_iterations": int(result["outer_iterations"]),
                "J": float(result["J"]),
            }
        )
        if best_result is None or _seed_attempt_score(result) < _seed_attempt_score(best_result):
            best_result = result
            best_seed_name = str(seed_name)

    assert best_result is not None
    if best_result.get("state_out") is None and raw_state_out:
        chosen_seed = next(seed for name, seed in init_candidates if name == best_seed_name)
        best_result = run_raw_method(
            problem,
            method=str(case.method),
            epsilon=float(case.epsilon),
            maxit=raw_maxit,
            init_free=chosen_seed,
            state_out=raw_state_out,
        )
        best_result["start_seed_name"] = best_seed_name

    best_result["start_seed_name"] = best_seed_name
    best_result["seed_attempts"] = attempts
    return best_result


def _direct_or_warmstart_certified(
    problem,
    *,
    init_free: np.ndarray,
    certified_tol: float,
    polish_maxit: int,
    warm_method: str = "mpa",
    raw_epsilon: float = 1.0e-5,
    raw_maxit: int = SUITE_MPA_MAXIT,
    cache_dir: Path | None = None,
    name: str = "certified",
) -> dict[str, object]:
    payload: dict[str, object] = {"raw": None, "certified": None}
    direct_state_out = "" if cache_dir is None else str(cache_dir / f"{name}_direct_state.npz")
    certified = certify_from_iterate(
        problem,
        iterate_free=np.asarray(init_free, dtype=np.float64),
        epsilon=float(certified_tol),
        maxit=int(polish_maxit),
        state_out=direct_state_out,
        handoff_source="direct_init",
    )
    payload["certified"] = certified
    if str(certified["status"]) == "completed":
        payload["path"] = "direct"
        return payload

    raw_state_out = "" if cache_dir is None else str(cache_dir / f"{name}_warm_raw_state.npz")
    raw = run_raw_method(
        problem,
        method=str(warm_method),
        epsilon=float(raw_epsilon),
        maxit=int(raw_maxit),
        init_free=np.asarray(init_free, dtype=np.float64),
        state_out=raw_state_out,
    )
    payload["raw"] = raw
    cert_state_out = "" if cache_dir is None else str(cache_dir / f"{name}_warm_certified_state.npz")
    certified = certify_from_iterate(
        problem,
        iterate_free=np.asarray(raw["iterate_free"], dtype=np.float64),
        epsilon=float(certified_tol),
        maxit=int(polish_maxit),
        state_out=cert_state_out,
        handoff_source=f"{warm_method}:{raw.get('reported_iterate_source', 'reported')}",
    )
    payload["certified"] = certified
    payload["path"] = "warmstart"
    return payload


def _p2_reference(out_dir: Path, *, certified_tol: float, polish_maxit: int) -> dict[str, object]:
    ref_dir = _reference_path(out_dir, "p2_newton_l7")
    result_path = ref_dir / "output.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if str(cached.get("runner_revision")) == RUNNER_REVISION and _payload_has_residual_diagnostics(cached):
            return cached

    ref_dir.mkdir(parents=True, exist_ok=True)
    lambda1 = 2.0 * (np.pi**2)
    problem = build_problem(
        level=REF_LEVEL,
        p=2.0,
        geometry=GEOMETRY,
        init_mode="sine",
        lambda1=float(lambda1),
        lambda_level=REF_LEVEL,
        seed=0,
    )
    objective = build_objective_bundle(problem, "J")
    x0 = np.asarray(problem.u_init, dtype=np.float64).copy()
    t0 = time.perf_counter()
    res = newton(
        lambda x: float(objective.value(np.asarray(x, dtype=np.float64))),
        lambda x: np.asarray(objective.grad(np.asarray(x, dtype=np.float64)), dtype=np.float64),
        HessSolverGenerator(
            lambda x: objective.ddf(np.asarray(x, dtype=np.float64)),
            solver_type="direct",
            tol=1.0e-10,
            maxiter=200,
        ),
        x0,
        tolf=0.0,
        tolg=1.0e-10,
        tolg_rel=0.0,
        maxit=40,
        linesearch_interval=(0.0, 1.0),
        linesearch_tol=1.0e-5,
        verbose=False,
        fail_on_nonfinite=False,
    )
    solve_time_s = time.perf_counter() - t0
    iterate = np.asarray(res["x"], dtype=np.float64)
    state_out = ref_dir / "state.npz"
    from src.problems.plaplace_up_arctan.solver_common import export_state_if_requested

    payload = {
        "method": "newton_reference",
        "status": "completed" if "converged" in str(res["message"]).lower() else "maxit",
        "message": str(res["message"]),
        "solve_time_s": float(solve_time_s),
        "reference_level": REF_LEVEL,
        "p": 2.0,
        "lambda1": float(lambda1),
        "lambda_level": REF_LEVEL,
        "direction_model": "jax_autodiff_newton_reference",
        "solver_revision": ARCTAN_SOLVER_REVISION,
        "runner_revision": RUNNER_REVISION,
        "iterate_free": iterate.tolist(),
        "certified_tol": float(certified_tol),
        "polish_maxit": int(polish_maxit),
    }
    problem_stats = problem.stats(iterate)
    payload["J"] = float(problem_stats.J)
    gradient_residual_norm, dual_residual_norm = residual_metrics(problem, objective, iterate)
    payload["gradient_residual_norm"] = float(gradient_residual_norm)
    payload["residual_norm"] = float(dual_residual_norm)
    payload["history"] = []
    payload["outer_iterations"] = int(res.get("nit", 0))
    payload["accepted_step_count"] = int(res.get("nit", 0))
    payload["reported_iterate_source"] = "final"
    payload["best_residual_norm"] = float(payload["residual_norm"])
    payload["best_gradient_residual_norm"] = float(payload["gradient_residual_norm"])
    payload["best_residual_outer_it"] = int(payload["outer_iterations"])
    payload["state_out"] = export_state_if_requested(
        str(state_out),
        problem=problem,
        energy=float(problem_stats.J),
        u_full=problem.expand_free(iterate),
        metadata={"method": "newton_reference", "epsilon": 1.0e-10, "seed_name": "sine"},
    )
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _p2_certified_level(out_dir: Path, level: int, *, certified_tol: float, polish_maxit: int) -> dict[str, object]:
    ref_dir = _reference_path(out_dir, f"p2_certified_l{int(level)}")
    result_path = ref_dir / "output.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        certified = cached.get("certified")
        if isinstance(certified, dict) and str(cached.get("runner_revision")) == RUNNER_REVISION and _payload_has_residual_diagnostics(certified):
            return cached

    ref_dir.mkdir(parents=True, exist_ok=True)
    problem = build_problem(
        level=int(level),
        p=2.0,
        geometry=GEOMETRY,
        init_mode="sine",
        lambda1=float(2.0 * np.pi**2),
        lambda_level=int(level),
        seed=0,
    )
    t0 = time.perf_counter()
    raw = run_raw_method(
        problem,
        method="mpa",
        epsilon=1.0e-5,
        maxit=SUITE_MPA_MAXIT,
        init_free=np.asarray(problem.u_init, dtype=np.float64),
        state_out=str(ref_dir / "raw_state.npz"),
    )
    certified = certify_from_iterate(
        problem,
        iterate_free=np.asarray(raw["iterate_free"], dtype=np.float64),
        epsilon=float(certified_tol),
        maxit=int(polish_maxit),
        state_out=str(ref_dir / "certified_state.npz"),
        handoff_source=f"mpa:{raw.get('reported_iterate_source', 'reported')}",
    )
    payload = {
        "runner_revision": RUNNER_REVISION,
        "level": int(level),
        "p": 2.0,
        "solve_time_s": float(time.perf_counter() - t0),
        "raw": raw,
        "certified": certified,
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _p3_certified_level(out_dir: Path, level: int, *, certified_tol: float, polish_maxit: int) -> dict[str, object]:
    ref_dir = _reference_path(out_dir, f"p3_certified_l{int(level)}")
    result_path = ref_dir / "output.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if str(cached.get("runner_revision")) == RUNNER_REVISION and _payload_has_residual_diagnostics(dict(cached.get("certified", {}))):
            return cached

    ref_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    p2_base = _p2_certified_level(out_dir, level, certified_tol=certified_tol, polish_maxit=polish_maxit)
    prev_p = 2.0
    prev_iterate = np.asarray(p2_base["certified"]["iterate_free"], dtype=np.float64)
    step = float(DEFAULT_CONTINUATION_STEP)
    continuation_steps: list[dict[str, object]] = []
    final_stage: dict[str, object] | None = None
    final_lambda: dict[str, object] | None = None

    while prev_p < 3.0 - 1.0e-12:
        target_p = min(3.0, prev_p + step)
        target_lambda = _lambda_payload(out_dir, level, p=target_p, init_free=np.maximum(prev_iterate, 1.0e-8), force=True)
        target_problem = build_problem(
            level=int(level),
            p=float(target_p),
            geometry=GEOMETRY,
            init_mode="sine",
            lambda1=float(target_lambda["lambda1"]),
            lambda_level=int(target_lambda["lambda_level"]),
            seed=0,
        )
        stage = _direct_or_warmstart_certified(
            target_problem,
            init_free=np.asarray(prev_iterate, dtype=np.float64),
            certified_tol=certified_tol,
            polish_maxit=polish_maxit,
            warm_method="mpa",
            raw_epsilon=1.0e-5,
            raw_maxit=SUITE_MPA_MAXIT,
            cache_dir=ref_dir,
            name=f"cont_p{_p_tag(target_p)}",
        )
        certified = dict(stage["certified"])
        continuation_steps.append(
            {
                "from_p": float(prev_p),
                "to_p": float(target_p),
                "step": float(step),
                "path": str(stage["path"]),
                "status": str(certified["status"]),
                "residual_norm": float(certified["residual_norm"]),
            }
        )
        final_stage = stage
        final_lambda = target_lambda
        if str(certified["status"]) == "completed":
            prev_p = float(target_p)
            prev_iterate = np.asarray(certified["iterate_free"], dtype=np.float64)
            continue
        if step * 0.5 >= DEFAULT_CONTINUATION_STEP_FLOOR - 1.0e-15:
            step *= 0.5
            continue
        break

    assert final_stage is not None
    assert final_lambda is not None
    payload = {
        "runner_revision": RUNNER_REVISION,
        "level": int(level),
        "p": 3.0,
        "solve_time_s": float(time.perf_counter() - t0),
        "continuation_steps": continuation_steps,
        "raw": final_stage.get("raw"),
        "certified": final_stage["certified"],
        "lambda1": float(final_lambda["lambda1"]),
        "lambda_level": int(final_lambda["lambda_level"]),
        "iterate_free": list(final_stage["certified"]["iterate_free"]),
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _p3_reference(out_dir: Path, *, certified_tol: float, polish_maxit: int) -> dict[str, object]:
    ref_dir = _reference_path(out_dir, "p3_certified_l7")
    result_path = ref_dir / "output.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if str(cached.get("runner_revision")) == RUNNER_REVISION and _payload_has_residual_diagnostics(dict(cached.get("certified", {}))):
            return cached

    ref_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    base_l6 = _p3_certified_level(out_dir, TOL_LEVEL, certified_tol=certified_tol, polish_maxit=polish_maxit)
    prev_problem = build_problem(
        level=TOL_LEVEL,
        p=3.0,
        geometry=GEOMETRY,
        init_mode="sine",
        lambda1=float(base_l6["lambda1"]),
        lambda_level=int(base_l6["lambda_level"]),
        seed=0,
    )
    prev_iterate = np.asarray(base_l6["certified"]["iterate_free"], dtype=np.float64)
    continuation_steps = list(base_l6["continuation_steps"])

    lambda_l7 = _lambda_payload(out_dir, REF_LEVEL, p=3.0, init_free=np.maximum(prev_iterate, 1.0e-8))
    problem_l7 = build_problem(
        level=REF_LEVEL,
        p=3.0,
        geometry=GEOMETRY,
        init_mode="sine",
        lambda1=float(lambda_l7["lambda1"]),
        lambda_level=int(lambda_l7["lambda_level"]),
        seed=0,
    )
    prolonged = prolong_free_to_problem(prev_problem.params, prev_iterate, problem_l7.params)
    final_stage = _direct_or_warmstart_certified(
        problem_l7,
        init_free=prolonged,
        certified_tol=certified_tol,
        polish_maxit=polish_maxit,
        warm_method="mpa",
        raw_epsilon=1.0e-5,
        raw_maxit=SUITE_MPA_MAXIT,
        cache_dir=ref_dir,
        name="reference_l7",
    )
    payload = {
        "runner_revision": RUNNER_REVISION,
        "reference_level": REF_LEVEL,
        "p": 3.0,
        "solve_time_s": float(time.perf_counter() - t0),
        "continuation_steps": continuation_steps,
        "raw": final_stage.get("raw"),
        "certified": final_stage["certified"],
        "lambda1": float(lambda_l7["lambda1"]),
        "lambda_level": int(lambda_l7["lambda_level"]),
        "iterate_free": list(final_stage["certified"]["iterate_free"]),
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _reference_payload(out_dir: Path, p: float, *, certified_tol: float, polish_maxit: int) -> dict[str, object]:
    if abs(float(p) - 2.0) <= 1.0e-12:
        return _p2_reference(out_dir, certified_tol=certified_tol, polish_maxit=polish_maxit)
    return _p3_reference(out_dir, certified_tol=certified_tol, polish_maxit=polish_maxit)


def _reference_iterate(out_dir: Path, p: float, *, certified_tol: float, polish_maxit: int) -> tuple[dict[str, object], dict[str, object], np.ndarray]:
    ref = _reference_payload(out_dir, p, certified_tol=certified_tol, polish_maxit=polish_maxit)
    if abs(float(p) - 2.0) <= 1.0e-12:
        problem = build_problem(
            level=REF_LEVEL,
            p=2.0,
            geometry=GEOMETRY,
            init_mode="sine",
            lambda1=float(ref["lambda1"]),
            lambda_level=REF_LEVEL,
            seed=0,
        )
        iterate = np.asarray(ref["iterate_free"], dtype=np.float64)
        return ref, problem.params, iterate

    lambda_payload = _lambda_payload(out_dir, REF_LEVEL, p=3.0)
    problem = build_problem(
        level=REF_LEVEL,
        p=3.0,
        geometry=GEOMETRY,
        init_mode="sine",
        lambda1=float(lambda_payload["lambda1"]),
        lambda_level=int(lambda_payload["lambda_level"]),
        seed=0,
    )
    iterate = np.asarray(ref["certified"]["iterate_free"], dtype=np.float64)
    return ref, problem.params, iterate


def _result_error(problem, iterate: np.ndarray, ref_params: dict[str, object], ref_iter: np.ndarray) -> float:
    if int(problem.mesh_level) == REF_LEVEL and int(problem.free_dofs) == int(np.asarray(ref_iter, dtype=np.float64).size):
        return same_mesh_w1p_error(problem.params, iterate, ref_iter)
    return nested_w1p_error(problem.params, iterate, ref_params, ref_iter)


def _run_case(
    case: Case,
    *,
    out_dir: Path,
    certified_tol: float,
    polish_maxit: int,
) -> dict[str, object]:
    case_dir = out_dir / _case_name(case)
    result_path = case_dir / "output.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if _case_payload_valid(cached):
            return cached

    case_dir.mkdir(parents=True, exist_ok=True)
    if abs(float(case.p) - 2.0) <= 1.0e-12:
        lambda_payload = None
        lambda1 = 2.0 * (np.pi**2)
        lambda_level = int(case.level)
    else:
        lambda_payload = _lambda_payload(out_dir, case.level, p=3.0)
        lambda1 = float(lambda_payload["lambda1"])
        lambda_level = int(lambda_payload["lambda_level"])

    problem = build_problem(
        level=int(case.level),
        p=float(case.p),
        geometry=GEOMETRY,
        init_mode="sine",
        lambda1=float(lambda1),
        lambda_level=int(lambda_level),
        seed=0,
    )
    init_candidates = _select_init_candidates(problem, case, lambda_payload)

    t0 = time.perf_counter()
    raw_result = _run_with_seed_candidates(
        case,
        problem=problem,
        init_candidates=init_candidates,
        raw_state_out=str(case_dir / "raw_state.npz"),
    )
    raw_result["solve_time_s"] = float(time.perf_counter() - t0)

    objective = build_objective_bundle(problem, "J")
    ray_audit = audit_ray_profile(
        problem,
        objective,
        np.asarray(raw_result["iterate_free"], dtype=np.float64),
        ray_mode="maximum",
        t_max=6.0,
        num_samples=81,
    )
    allow_rmpa_certification = str(case.method) == "mpa" or (
        str(ray_audit.get("best_kind")) == "maximum" and bool(ray_audit.get("stable_interior_extremum"))
    )

    certified_result = None
    certification_message = ""
    t1 = time.perf_counter()
    if allow_rmpa_certification:
        certified_result = certify_from_iterate(
            problem,
            iterate_free=np.asarray(raw_result["iterate_free"], dtype=np.float64),
            epsilon=float(certified_tol),
            maxit=int(polish_maxit),
            state_out=str(case_dir / "certified_state.npz"),
            handoff_source=f"{str(case.method)}:{raw_result.get('reported_iterate_source', 'reported')}",
        )
        certified_result["solve_time_s"] = float(time.perf_counter() - t1)
    else:
        certification_message = "Certification skipped because the ray audit did not show a stable interior maximum."

    if abs(float(case.p) - 3.0) <= 1.0e-12 and str(case.method) == "mpa":
        continued = _p3_certified_level(out_dir, int(case.level), certified_tol=certified_tol, polish_maxit=polish_maxit)
        certified_result = dict(continued["certified"])
        certified_result["solve_time_s"] = float(continued.get("solve_time_s", certified_result.get("solve_time_s", 0.0)))
        certification_message = "Certified via adaptive p-continuation from the p=2 branch."

    _, ref_params, ref_iter = _reference_iterate(out_dir, case.p, certified_tol=certified_tol, polish_maxit=polish_maxit)
    raw_result["reference_error_w1p"] = float(_result_error(problem, np.asarray(raw_result["iterate_free"], dtype=np.float64), ref_params, ref_iter))
    if certified_result is not None:
        certified_result["reference_error_w1p"] = float(
            _result_error(problem, np.asarray(certified_result["iterate_free"], dtype=np.float64), ref_params, ref_iter)
        )

    result = {
        "runner_revision": RUNNER_REVISION,
        "study": str(case.study),
        "method": str(case.method),
        "p": float(case.p),
        "level": int(case.level),
        "epsilon": float(case.epsilon),
        "result_path": str(result_path),
        "raw": raw_result,
        "certified": certified_result,
        "ray_audit": ray_audit,
        "certification_message": certification_message,
        "solution_track": "certified" if certified_result is not None else "raw",
        "start_seed_name": raw_result.get("start_seed_name"),
        "seed_attempts": list(raw_result.get("seed_attempts", [])),
        "solve_time_s": float(raw_result.get("solve_time_s", 0.0)) + float(0.0 if certified_result is None else certified_result.get("solve_time_s", 0.0)),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _row_from_result(case: Case, result: dict[str, object]) -> dict[str, object]:
    if "raw" not in result:
        result = {
            "raw": dict(result),
            "certified": None,
            "solution_track": "raw",
            "start_seed_name": result.get("start_seed_name"),
            "result_path": result.get("result_path"),
            "solve_time_s": result.get("solve_time_s", 0.0),
            "ray_audit": {},
            "certification_message": "",
        }
    raw = dict(result["raw"])
    certified = None if result.get("certified") is None else dict(result["certified"])
    preferred = certified if certified is not None else raw

    row = {
        **asdict(case),
        "status": str(preferred["status"]),
        "message": str(preferred["message"]),
        "lambda1": float(preferred["lambda1"]),
        "lambda_level": int(preferred["lambda_level"]),
        "J": float(preferred["J"]),
        "residual_norm": float(preferred["residual_norm"]),
        "gradient_residual_norm": float(preferred.get("gradient_residual_norm", preferred["residual_norm"])),
        "outer_iterations": int(preferred["outer_iterations"]),
        "accepted_step_count": int(preferred["accepted_step_count"]),
        "reference_error_w1p": None if preferred.get("reference_error_w1p") is None else float(preferred["reference_error_w1p"]),
        "solve_time_s": float(result.get("solve_time_s", 0.0)),
        "state_path": preferred.get("state_out"),
        "result_path": result.get("result_path"),
        "history": list(preferred.get("history", [])),
        "solution_track": str(result.get("solution_track", "raw")),
        "start_seed_name": result.get("start_seed_name"),
        "raw_status": str(raw["status"]),
        "raw_message": str(raw["message"]),
        "raw_J": float(raw["J"]),
        "raw_residual_norm": float(raw["residual_norm"]),
        "raw_gradient_residual_norm": float(raw.get("gradient_residual_norm", raw["residual_norm"])),
        "raw_best_residual_norm": float(raw.get("best_residual_norm", raw["residual_norm"])),
        "raw_best_gradient_residual_norm": float(raw.get("best_gradient_residual_norm", raw.get("gradient_residual_norm", raw["residual_norm"]))),
        "raw_best_residual_outer_it": None if raw.get("best_residual_outer_it") is None else int(raw["best_residual_outer_it"]),
        "raw_final_history_residual_norm": None if not raw.get("history") else float(raw["history"][-1].get("dual_residual_norm", raw["residual_norm"])),
        "raw_final_history_gradient_residual_norm": None if not raw.get("history") else float(raw["history"][-1].get("gradient_residual_norm", raw.get("gradient_residual_norm", raw["residual_norm"]))),
        "raw_reported_iterate_source": raw.get("reported_iterate_source"),
        "raw_outer_iterations": int(raw["outer_iterations"]),
        "raw_accepted_step_count": int(raw["accepted_step_count"]),
        "raw_reference_error_w1p": None if raw.get("reference_error_w1p") is None else float(raw["reference_error_w1p"]),
        "raw_state_path": raw.get("state_out"),
        "raw_history": list(raw.get("history", [])),
        "raw_configured_maxit": raw.get("configured_maxit"),
        "raw_direction_model": raw.get("direction_model"),
        "certified_status": None if certified is None else str(certified["status"]),
        "certified_message": None if certified is None else str(certified["message"]),
        "certified_J": None if certified is None else float(certified["J"]),
        "certified_residual_norm": None if certified is None else float(certified["residual_norm"]),
        "certified_gradient_residual_norm": None if certified is None else float(certified.get("gradient_residual_norm", certified["residual_norm"])),
        "certified_best_residual_norm": None if certified is None else float(certified.get("best_residual_norm", certified["residual_norm"])),
        "certified_best_gradient_residual_norm": None if certified is None else float(certified.get("best_gradient_residual_norm", certified.get("gradient_residual_norm", certified["residual_norm"]))),
        "certified_best_residual_outer_it": None if certified is None or certified.get("best_residual_outer_it") is None else int(certified["best_residual_outer_it"]),
        "certified_reported_iterate_source": None if certified is None else certified.get("reported_iterate_source"),
        "certified_outer_iterations": None if certified is None else int(certified["outer_iterations"]),
        "certified_newton_iters": None if certified is None else int(certified.get("certified_newton_iters", certified["outer_iterations"])),
        "certified_reference_error_w1p": None if certified is None or certified.get("reference_error_w1p") is None else float(certified["reference_error_w1p"]),
        "certified_state_path": None if certified is None else certified.get("state_out"),
        "certified_history": [] if certified is None else list(certified.get("history", [])),
        "certified_handoff_source": None if certified is None else certified.get("handoff_source"),
        "ray_best_kind": result.get("ray_audit", {}).get("best_kind"),
        "ray_stable_interior_extremum": bool(result.get("ray_audit", {}).get("stable_interior_extremum", False)),
        "certification_message": result.get("certification_message"),
        "configured_maxit": raw.get("configured_maxit"),
    }
    return row


def build_cases() -> list[Case]:
    cases: list[Case] = []
    for p in P_VALUES:
        for method in METHODS:
            for level in REFINEMENT_LEVELS:
                cases.append(Case("mesh_refinement", method, p, level, 1.0e-5))
            for epsilon in EPSILONS:
                cases.append(Case("tolerance_sweep", method, p, TOL_LEVEL, epsilon))
    return cases


def run_suite(*, out_dir: Path, certified_tol: float = DEFAULT_CERTIFIED_TOL, polish_maxit: int = DEFAULT_POLISH_MAXIT) -> dict[str, object]:
    rows = []
    for level in range(4, REF_LEVEL + 1):
        _lambda_payload(out_dir, level, p=3.0)
    _reference_payload(out_dir, 2.0, certified_tol=certified_tol, polish_maxit=polish_maxit)
    _reference_payload(out_dir, 3.0, certified_tol=certified_tol, polish_maxit=polish_maxit)
    for case in build_cases():
        rows.append(_row_from_result(case, _run_case(case, out_dir=out_dir, certified_tol=certified_tol, polish_maxit=polish_maxit)))
    payload = {
        "suite": "plaplace_up_arctan_full",
        "geometry": GEOMETRY,
        "reference_level": REF_LEVEL,
        "runner_revision": RUNNER_REVISION,
        "solver_revision": ARCTAN_SOLVER_REVISION,
        "certified_tol": float(certified_tol),
        "polish_maxit": int(polish_maxit),
        "rows": rows,
        "lambda_cache_dir": str(_lambda_cache_dir(out_dir)),
        "generated_case_count": len(rows),
        "status_counts": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in sorted({row["status"] for row in rows})
        },
        "raw_status_counts": {
            status: sum(1 for row in rows if row["raw_status"] == status)
            for status in sorted({row["raw_status"] for row in rows})
        },
        "certified_status_counts": {
            status: sum(1 for row in rows if row["certified_status"] == status)
            for status in sorted({row["certified_status"] for row in rows if row["certified_status"] is not None})
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--certified-tol", type=float, default=DEFAULT_CERTIFIED_TOL)
    parser.add_argument("--polish-maxit", type=int, default=DEFAULT_POLISH_MAXIT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = run_suite(out_dir=args.out_dir, certified_tol=float(args.certified_tol), polish_maxit=int(args.polish_maxit))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(args.summary), "rows": len(payload["rows"])}, indent=2))


if __name__ == "__main__":
    main()

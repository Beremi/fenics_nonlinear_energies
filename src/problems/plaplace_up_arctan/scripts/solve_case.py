#!/usr/bin/env python3
"""CLI entrypoint for one arctan-resonance p-Laplacian solve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.problems.plaplace_up_arctan.eigen import compute_lambda1_cached
from src.problems.plaplace_up_arctan.seeds import (
    SEED_BUBBLE,
    SEED_EIGENFUNCTION,
    SEED_SINE,
    SEED_TILTED,
    named_start_seed,
)
from src.problems.plaplace_up_arctan.solver_common import build_problem
from src.problems.plaplace_up_arctan.support.mesh import GEOMETRY_SQUARE_UNIT, SUPPORTED_INIT_MODES
from src.problems.plaplace_up_arctan.workflow import certify_from_iterate, run_raw_method


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("mpa", "rmpa"), default="rmpa")
    parser.add_argument("--track", choices=("raw", "certified", "both"), default="both")
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--level", type=int, default=6)
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    parser.add_argument("--certified-tol", type=float, default=1.0e-8)
    parser.add_argument("--polish-maxit", type=int, default=80)
    parser.add_argument("--p-continuation", action="store_true")
    parser.add_argument("--geometry", choices=(GEOMETRY_SQUARE_UNIT,), default=GEOMETRY_SQUARE_UNIT)
    parser.add_argument("--init-mode", choices=SUPPORTED_INIT_MODES, default="sine")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maxit", type=int, default=0)
    parser.add_argument("--delta0", type=float, default=1.0)
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--segment-tol-factor", type=float, default=0.125)
    parser.add_argument("--lambda-cache", type=str, default="")
    parser.add_argument("--lambda-value", type=float, default=float("nan"))
    parser.add_argument("--start-seed", choices=("auto", SEED_SINE, SEED_BUBBLE, SEED_TILTED, SEED_EIGENFUNCTION), default="auto")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--json", type=str, default="")
    parser.add_argument("--state-out", type=str, default="")
    return parser


def _resolve_output_path(args: argparse.Namespace) -> str:
    if args.out and args.json and Path(args.out) != Path(args.json):
        raise ValueError("--out and --json must match when both are provided")
    return args.out or args.json


def _default_lambda_cache(args: argparse.Namespace, p: float | None = None) -> Path:
    p_eff = float(args.p if p is None else p)
    if abs(p_eff - 3.0) <= 1.0e-12:
        return Path("artifacts/raw_results/plaplace_up_arctan_lambda") / f"lambda_p3_l{int(args.level)}.json"
    tag = f"{p_eff:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return Path("artifacts/raw_results/plaplace_up_arctan_lambda") / f"lambda_p{tag}_l{int(args.level)}.json"


def _resolve_lambda_payload(args: argparse.Namespace, *, p: float | None = None, init_free: np.ndarray | None = None) -> dict[str, object] | None:
    p_eff = float(args.p if p is None else p)
    if abs(p_eff - 2.0) <= 1.0e-12:
        return None
    if np.isfinite(float(args.lambda_value)) and abs(p_eff - float(args.p)) <= 1.0e-12:
        return None
    cache_path = Path(args.lambda_cache) if args.lambda_cache and abs(p_eff - float(args.p)) <= 1.0e-12 else _default_lambda_cache(args, p_eff)
    return compute_lambda1_cached(
        cache_path=cache_path,
        p=float(p_eff),
        level=int(args.level),
        geometry=str(args.geometry),
        init_mode=str(args.init_mode),
        seed=int(args.seed),
        force=False,
        init_free=None if init_free is None else np.asarray(init_free, dtype=np.float64),
    )


def _resolve_lambda(args: argparse.Namespace, lambda_payload: dict[str, object] | None = None, *, p: float | None = None) -> tuple[float, int]:
    p_eff = float(args.p if p is None else p)
    if abs(p_eff - 2.0) <= 1.0e-12:
        return 2.0 * (np.pi**2), int(args.level)
    if np.isfinite(float(args.lambda_value)) and abs(p_eff - float(args.p)) <= 1.0e-12:
        return float(args.lambda_value), int(args.level)
    payload = lambda_payload if lambda_payload is not None else _resolve_lambda_payload(args, p=p_eff)
    assert payload is not None
    return float(payload["lambda1"]), int(payload["lambda_level"])


def _build_problem_for_p(args: argparse.Namespace, p: float, *, init_free: np.ndarray | None = None) -> tuple[object, dict[str, object] | None]:
    lambda_payload = _resolve_lambda_payload(args, p=p, init_free=init_free)
    lambda1, lambda_level = _resolve_lambda(args, lambda_payload, p=p)
    problem = build_problem(
        level=int(args.level),
        p=float(p),
        geometry=str(args.geometry),
        init_mode=str(args.init_mode),
        lambda1=float(lambda1),
        lambda_level=int(lambda_level),
        seed=int(args.seed),
    )
    return problem, lambda_payload


def _raw_maxit(args: argparse.Namespace) -> int:
    default_maxit = 600 if str(args.method) == "mpa" else 300
    return int(args.maxit) if int(args.maxit) > 0 else int(default_maxit)


def _resolve_start_seed(problem, args: argparse.Namespace, lambda_payload: dict[str, object] | None) -> tuple[np.ndarray | None, str]:
    if str(args.start_seed) == "auto":
        init_free = None if lambda_payload is None else np.asarray(lambda_payload["eigenfunction_free"], dtype=np.float64)
        return init_free, ("eigenfunction" if lambda_payload is not None else "sine")
    init_free = named_start_seed(
        problem,
        str(args.start_seed),
        eigenfunction_free=None if lambda_payload is None else np.asarray(lambda_payload["eigenfunction_free"], dtype=np.float64),
    )
    return init_free, str(args.start_seed)


def _state_path_with_suffix(state_out: str, suffix: str) -> str:
    if not state_out:
        return ""
    path = Path(state_out)
    return str(path.with_name(f"{path.stem}_{suffix}{path.suffix}"))


def _run_fixed_level_continuation(args: argparse.Namespace) -> dict[str, object]:
    level = int(args.level)
    current_p = 2.0
    problem, _ = _build_problem_for_p(args, current_p)
    raw = run_raw_method(
        problem,
        method="mpa",
        epsilon=float(args.epsilon),
        maxit=_raw_maxit(args),
        init_free=np.asarray(problem.u_init, dtype=np.float64),
        state_out=_state_path_with_suffix(str(args.state_out), "p2_raw"),
        delta0=float(args.delta0),
        num_nodes=int(args.num_nodes),
        rho=float(args.rho),
        segment_tol_factor=float(args.segment_tol_factor),
    )
    certified = certify_from_iterate(
        problem,
        iterate_free=np.asarray(raw["iterate_free"], dtype=np.float64),
        epsilon=float(args.certified_tol),
        maxit=int(args.polish_maxit),
        state_out=_state_path_with_suffix(str(args.state_out), "p2_certified"),
        handoff_source=f"p2_mpa:{raw.get('reported_iterate_source', 'reported')}",
    )
    continuation_steps = [
        {
            "from_p": 2.0,
            "to_p": 2.0,
            "path": "p2_base",
            "status": str(certified["status"]),
            "residual_norm": float(certified["residual_norm"]),
        }
    ]
    prev = np.asarray(certified["iterate_free"], dtype=np.float64)
    step = 0.2
    final_lambda = {"lambda1": problem.lambda1, "lambda_level": level}

    while current_p < float(args.p) - 1.0e-12:
        target_p = min(float(args.p), current_p + step)
        target_problem, target_lambda_payload = _build_problem_for_p(args, target_p, init_free=np.maximum(prev, 1.0e-8))
        direct = certify_from_iterate(
            target_problem,
            iterate_free=prev,
            epsilon=float(args.certified_tol),
            maxit=int(args.polish_maxit),
            state_out=_state_path_with_suffix(str(args.state_out), f"p{str(target_p).replace('.', 'p')}_direct"),
            handoff_source="direct_continuation",
        )
        raw_status = "direct"
        raw_residual = None
        stage_raw = None
        if str(direct["status"]) != "completed":
            stage_raw = run_raw_method(
                target_problem,
                method="mpa",
                epsilon=float(args.epsilon),
                maxit=_raw_maxit(args),
                init_free=prev,
                state_out=_state_path_with_suffix(str(args.state_out), f"p{str(target_p).replace('.', 'p')}_raw"),
            )
            direct = certify_from_iterate(
                target_problem,
                iterate_free=np.asarray(stage_raw["iterate_free"], dtype=np.float64),
                epsilon=float(args.certified_tol),
                maxit=int(args.polish_maxit),
                state_out=_state_path_with_suffix(str(args.state_out), f"p{str(target_p).replace('.', 'p')}_certified"),
                handoff_source=f"mpa:{stage_raw.get('reported_iterate_source', 'reported')}",
            )
            raw_status = str(stage_raw["status"])
            raw_residual = float(stage_raw["residual_norm"])
        continuation_steps.append(
            {
                "from_p": float(current_p),
                "to_p": float(target_p),
                "step": float(step),
                "path": raw_status,
                "raw_residual_norm": raw_residual,
                "status": str(direct["status"]),
                "residual_norm": float(direct["residual_norm"]),
            }
        )
        if str(direct["status"]) != "completed":
            if step / 2.0 >= 0.025 - 1.0e-15:
                step *= 0.5
                continue
            break
        current_p = float(target_p)
        prev = np.asarray(direct["iterate_free"], dtype=np.float64)
        certified = direct
        raw = stage_raw if stage_raw is not None else raw
        if target_lambda_payload is not None:
            final_lambda = {
                "lambda1": float(target_lambda_payload["lambda1"]),
                "lambda_level": int(target_lambda_payload["lambda_level"]),
            }

    return {
        "track": "certified",
        "used_p_continuation": True,
        "method": str(args.method),
        "p": float(args.p),
        "level": level,
        "lambda1": float(final_lambda["lambda1"]),
        "lambda_level": int(final_lambda["lambda_level"]),
        "raw": raw,
        "certified": certified,
        "continuation_steps": continuation_steps,
    }


def run_case_from_args(args: argparse.Namespace) -> dict[str, object]:
    if bool(args.p_continuation) and float(args.p) > 2.0 and str(args.track) != "raw":
        result = _run_fixed_level_continuation(args)
        result["reference"] = bool(args.reference)
        return result

    problem, lambda_payload = _build_problem_for_p(args, float(args.p))
    init_free, start_seed_name = _resolve_start_seed(problem, args, lambda_payload)
    raw = run_raw_method(
        problem,
        method=str(args.method),
        epsilon=float(args.epsilon),
        maxit=_raw_maxit(args),
        init_free=init_free,
        state_out=str(args.state_out) if str(args.track) == "raw" else _state_path_with_suffix(str(args.state_out), "raw"),
        delta0=float(args.delta0),
        num_nodes=int(args.num_nodes),
        rho=float(args.rho),
        segment_tol_factor=float(args.segment_tol_factor),
    )
    raw["start_seed_name"] = start_seed_name
    if str(args.track) == "raw":
        raw["reference"] = bool(args.reference)
        raw["track"] = "raw"
        raw["used_p_continuation"] = False
        return raw

    certified = certify_from_iterate(
        problem,
        iterate_free=np.asarray(raw["iterate_free"], dtype=np.float64),
        epsilon=float(args.certified_tol),
        maxit=int(args.polish_maxit),
        state_out=str(args.state_out) if str(args.track) == "certified" else _state_path_with_suffix(str(args.state_out), "certified"),
        handoff_source=f"{str(args.method)}:{raw.get('reported_iterate_source', 'reported')}",
    )
    payload = {
        "track": "certified" if str(args.track) == "certified" else "both",
        "used_p_continuation": False,
        "method": str(args.method),
        "p": float(args.p),
        "level": int(args.level),
        "lambda1": float(raw["lambda1"]),
        "lambda_level": int(raw["lambda_level"]),
        "raw": raw,
        "certified": certified,
        "reference": bool(args.reference),
        "start_seed_name": start_seed_name,
    }
    return payload


def main() -> None:
    args = build_parser().parse_args()
    out_path = _resolve_output_path(args)
    result = run_case_from_args(args)
    text = json.dumps(result, indent=2)
    print(text)
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

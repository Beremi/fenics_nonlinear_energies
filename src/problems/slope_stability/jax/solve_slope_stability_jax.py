#!/usr/bin/env python3
"""Experimental pure-JAX slope-stability bring-up on the P2 homogeneous SSR mesh."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from src.core.benchmark.state_export import export_planestrain_state_npz
from src.core.serial.jax_diff import EnergyDerivator
from src.core.serial.minimizers import newton
from src.core.serial.sparse_solvers import HessSolverGenerator
from src.problems.slope_stability.jax.jax_energy import J
from src.problems.slope_stability.jax.mesh import MeshSlopeStability2D
from src.problems.slope_stability.support import DEFAULT_CASE, DEFAULT_LEVEL, davis_b_reduction


SUCCESS_MESSAGE_PREFIXES = (
    "Converged",
    "Gradient norm converged",
    "Stopping condition for f is satisfied",
    "Stopping condition for step size is satisfied",
    "Trust-region step converged",
)


def _reduced_material(mesh: MeshSlopeStability2D, lambda_target: float) -> tuple[float, float]:
    return davis_b_reduction(
        float(mesh.params["c0"]),
        float(mesh.params["phi_deg"]),
        float(mesh.params["psi_deg"]),
        float(lambda_target),
    )


def _is_success_message(message: str) -> bool:
    return any(str(message).startswith(prefix) for prefix in SUCCESS_MESSAGE_PREFIXES)


def build_solver_context(
    case: str | None = None,
    level: int | None = None,
    *,
    ksp_rtol: float = 1e-1,
    ksp_max_it: int = 30,
    reg: float = 1.0e-12,
    verbose: bool = False,
) -> dict[str, object]:
    mesh = MeshSlopeStability2D(level=level, case=case)
    params, adjacency, u_init = mesh.get_data_jax()
    params["reg"] = float(reg)
    energy = EnergyDerivator(J, params, adjacency, u_init)
    F, dF, ddF = energy.get_derivatives()
    solver = HessSolverGenerator(
        ddf=ddF,
        solver_type="amg",
        elastic_kernel=mesh.elastic_kernel,
        verbose=verbose,
        tol=ksp_rtol,
        maxiter=ksp_max_it,
    )
    return {
        "mesh": mesh,
        "energy": energy,
        "F": F,
        "dF": dF,
        "ddf_with_solver": solver,
        "u_init_default": np.asarray(u_init, dtype=np.float64),
        "reg": float(reg),
        "ksp_rtol": float(ksp_rtol),
        "ksp_max_it": int(ksp_max_it),
    }


def solve_lambda_step(
    context: dict[str, object],
    *,
    lambda_target: float,
    u_guess: np.ndarray | None = None,
    maxit: int = 100,
    linesearch_interval: tuple[float, float] = (-0.5, 2.0),
    linesearch_tol: float = 1e-1,
    tolf: float = 1e-4,
    tolg: float = 1e-3,
    tolg_rel: float = 1e-3,
    tolx_rel: float = 1e-3,
    tolx_abs: float = 1e-10,
    require_all_convergence: bool = True,
    use_trust_region: bool = True,
    trust_radius_init: float = 0.5,
    trust_radius_min: float = 1e-8,
    trust_radius_max: float = 1e6,
    trust_shrink: float = 0.5,
    trust_expand: float = 1.5,
    trust_eta_shrink: float = 0.05,
    trust_eta_expand: float = 0.75,
    trust_max_reject: int = 6,
    trust_subproblem_line_search: bool = True,
    state_out: str = "",
    verbose: bool = False,
) -> dict[str, object]:
    mesh = context["mesh"]
    energy = context["energy"]
    F = context["F"]
    dF = context["dF"]
    ddf_with_solver = context["ddf_with_solver"]
    default_guess = context["u_init_default"]

    reduced_cohesion, reduced_phi_deg = _reduced_material(mesh, float(lambda_target))
    energy.params["phi_deg"] = float(reduced_phi_deg)
    energy.params["cohesion"] = float(reduced_cohesion)
    energy.params["reg"] = float(context["reg"])

    guess = np.asarray(default_guess if u_guess is None else u_guess, dtype=np.float64)
    solve_start = time.perf_counter()
    res = newton(
        F,
        dF,
        ddf_with_solver,
        guess,
        tolf=tolf,
        tolg=tolg,
        tolg_rel=tolg_rel,
        linesearch_tol=linesearch_tol,
        linesearch_interval=linesearch_interval,
        maxit=maxit,
        tolx_rel=tolx_rel,
        tolx_abs=tolx_abs,
        require_all_convergence=require_all_convergence,
        fail_on_nonfinite=True,
        verbose=verbose,
        trust_region=use_trust_region,
        trust_radius_init=trust_radius_init,
        trust_radius_min=trust_radius_min,
        trust_radius_max=trust_radius_max,
        trust_shrink=trust_shrink,
        trust_expand=trust_expand,
        trust_eta_shrink=trust_eta_shrink,
        trust_eta_expand=trust_eta_expand,
        trust_max_reject=trust_max_reject,
        trust_subproblem_line_search=trust_subproblem_line_search,
        save_history=True,
        save_linear_timing=True,
    )
    solve_time = time.perf_counter() - solve_start

    u_full = np.asarray(mesh.params["u_0"], dtype=np.float64).copy()
    freedofs = np.asarray(mesh.params["freedofs"], dtype=np.int64)
    u_full[freedofs] = np.asarray(res["x"], dtype=np.float64)
    coords_ref = np.asarray(mesh.params["nodes"], dtype=np.float64)
    coords_final = coords_ref + u_full.reshape((-1, 2))
    displacement = coords_final - coords_ref
    u_max = float(np.max(np.linalg.norm(displacement, axis=1)))
    omega = float(np.dot(np.asarray(mesh.params["force"], dtype=np.float64), u_full))
    total_linear_iters = int(sum(int(rec.get("ksp_its", 0)) for rec in res.get("linear_timing", [])))

    message = str(res["message"])
    solver_success = bool(
        _is_success_message(message)
        and np.isfinite(float(res["fun"]))
        and np.all(np.isfinite(np.asarray(res["x"], dtype=np.float64)))
    )
    result_status = "completed" if solver_success else "failed"

    if state_out:
        export_planestrain_state_npz(
            state_out,
            coords_ref=coords_ref,
            x_final=coords_final,
            triangles=np.asarray(mesh.params["elems_scalar"], dtype=np.int32),
            case_name=str(mesh.params["case_name"]),
            lambda_target=float(lambda_target),
            energy=float(res["fun"]),
            metadata={
                "solver_family": "pure_jax",
                "prototype_mode": "zero_history_endpoint",
                "davis_type": str(mesh.params["davis_type"]),
            },
        )

    return {
        "success": solver_success,
        "u_free": np.asarray(res["x"], dtype=np.float64),
        "u_full": u_full,
        "coords_ref": coords_ref,
        "coords_final": coords_final,
        "displacement": displacement,
        "omega": omega,
        "solver": "pure_jax",
        "family": "slope_stability",
        "prototype_mode": "zero_history_endpoint",
        "case": {
            "name": str(mesh.params["case_name"]),
            "level": int(mesh.params["level"]),
            "analysis": "ssr_endpoint",
            "elem_type": "P2",
            "lambda_target": float(lambda_target),
            "davis_type": str(mesh.params["davis_type"]),
        },
        "mesh": {
            "level": int(mesh.params["level"]),
            "h": float(mesh.params["h"]),
            "nodes": int(mesh.params["nodes"].shape[0]),
            "elements": int(mesh.params["elems_scalar"].shape[0]),
            "free_dofs": int(freedofs.size),
            "free_x_dofs": int(np.asarray(mesh.params["q_mask"], dtype=bool)[:, 0].sum()),
            "free_y_dofs": int(np.asarray(mesh.params["q_mask"], dtype=bool)[:, 1].sum()),
        },
        "material": {
            "raw": {
                "c0": float(mesh.params["c0"]),
                "phi_deg": float(mesh.params["phi_deg"]),
                "psi_deg": float(mesh.params["psi_deg"]),
                "E": float(mesh.params["E"]),
                "nu": float(mesh.params["nu"]),
                "gamma": float(mesh.params["gamma"]),
            },
            "reduced": {
                "cohesion": float(reduced_cohesion),
                "phi_deg": float(reduced_phi_deg),
            },
        },
        "timings": {
            "setup_time": 0.0,
            "solve_time": float(solve_time),
            "total_time": float(solve_time),
            "jax_setup_timing": energy.timings,
        },
        "result": {
            "status": result_status,
            "final_energy": float(res["fun"]),
            "omega": omega,
            "u_max": u_max,
            "newton_iters": int(res["nit"]),
            "linear_iters": total_linear_iters,
            "message": message,
            "solver_success": solver_success,
            "solve_time": float(solve_time),
        },
        "history": res.get("history", []),
        "linear_timing": res.get("linear_timing", []),
    }


def run_case(
    case: str | None = None,
    level: int | None = None,
    *,
    lambda_target: float = 1.21,
    maxit: int = 100,
    linesearch_interval: tuple[float, float] = (-0.5, 2.0),
    linesearch_tol: float = 1e-1,
    ksp_rtol: float = 1e-1,
    ksp_max_it: int = 30,
    tolf: float = 1e-4,
    tolg: float = 1e-3,
    tolg_rel: float = 1e-3,
    tolx_rel: float = 1e-3,
    tolx_abs: float = 1e-10,
    require_all_convergence: bool = True,
    use_trust_region: bool = True,
    trust_radius_init: float = 0.5,
    trust_radius_min: float = 1e-8,
    trust_radius_max: float = 1e6,
    trust_shrink: float = 0.5,
    trust_expand: float = 1.5,
    trust_eta_shrink: float = 0.05,
    trust_eta_expand: float = 0.75,
    trust_max_reject: int = 6,
    trust_subproblem_line_search: bool = True,
    reg: float = 1.0e-12,
    state_out: str = "",
    verbose: bool = False,
) -> dict:
    total_start = time.perf_counter()
    setup_start = time.perf_counter()
    context = build_solver_context(
        case=case,
        level=level,
        ksp_rtol=ksp_rtol,
        ksp_max_it=ksp_max_it,
        reg=reg,
        verbose=verbose,
    )
    setup_time = time.perf_counter() - setup_start
    payload = solve_lambda_step(
        context,
        lambda_target=lambda_target,
        u_guess=np.asarray(context["u_init_default"], dtype=np.float64),
        maxit=maxit,
        linesearch_interval=linesearch_interval,
        linesearch_tol=linesearch_tol,
        tolf=tolf,
        tolg=tolg,
        tolg_rel=tolg_rel,
        tolx_rel=tolx_rel,
        tolx_abs=tolx_abs,
        require_all_convergence=require_all_convergence,
        use_trust_region=use_trust_region,
        trust_radius_init=trust_radius_init,
        trust_radius_min=trust_radius_min,
        trust_radius_max=trust_radius_max,
        trust_shrink=trust_shrink,
        trust_expand=trust_expand,
        trust_eta_shrink=trust_eta_shrink,
        trust_eta_expand=trust_eta_expand,
        trust_max_reject=trust_max_reject,
        trust_subproblem_line_search=trust_subproblem_line_search,
        state_out=state_out,
        verbose=verbose,
    )
    payload["timings"]["setup_time"] = float(setup_time)
    payload["timings"]["total_time"] = float(time.perf_counter() - total_start)
    for key in ("u_free", "u_full", "coords_ref", "coords_final", "displacement"):
        payload.pop(key, None)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=str, default="")
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    parser.add_argument("--lambda-target", type=float, default=1.21)
    parser.add_argument("--json", type=str, default="")
    parser.add_argument("--state-out", type=str, default="")
    parser.add_argument("--maxit", type=int, default=100)
    parser.add_argument("--linesearch_a", type=float, default=-0.5)
    parser.add_argument("--linesearch_b", type=float, default=2.0)
    parser.add_argument("--linesearch_tol", type=float, default=1e-1)
    parser.add_argument("--ksp_rtol", type=float, default=1e-1)
    parser.add_argument("--ksp_max_it", type=int, default=30)
    parser.add_argument("--tolf", type=float, default=1e-4)
    parser.add_argument("--tolg", type=float, default=1e-3)
    parser.add_argument("--tolg_rel", type=float, default=1e-3)
    parser.add_argument("--tolx_rel", type=float, default=1e-3)
    parser.add_argument("--tolx_abs", type=float, default=1e-10)
    parser.add_argument("--require_all_convergence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-trust-region", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_radius_init", type=float, default=0.5)
    parser.add_argument("--trust_radius_min", type=float, default=1e-8)
    parser.add_argument("--trust_radius_max", type=float, default=1e6)
    parser.add_argument("--trust_shrink", type=float, default=0.5)
    parser.add_argument("--trust_expand", type=float, default=1.5)
    parser.add_argument("--trust_eta_shrink", type=float, default=0.05)
    parser.add_argument("--trust_eta_expand", type=float, default=0.75)
    parser.add_argument("--trust_max_reject", type=int, default=6)
    parser.add_argument("--trust_subproblem_line_search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reg", type=float, default=1.0e-12)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    payload = run_case(
        case=(args.case or None),
        level=args.level,
        lambda_target=args.lambda_target,
        maxit=args.maxit,
        linesearch_interval=(float(args.linesearch_a), float(args.linesearch_b)),
        linesearch_tol=args.linesearch_tol,
        ksp_rtol=args.ksp_rtol,
        ksp_max_it=args.ksp_max_it,
        tolf=args.tolf,
        tolg=args.tolg,
        tolg_rel=args.tolg_rel,
        tolx_rel=args.tolx_rel,
        tolx_abs=args.tolx_abs,
        require_all_convergence=args.require_all_convergence,
        use_trust_region=args.use_trust_region,
        trust_radius_init=args.trust_radius_init,
        trust_radius_min=args.trust_radius_min,
        trust_radius_max=args.trust_radius_max,
        trust_shrink=args.trust_shrink,
        trust_expand=args.trust_expand,
        trust_eta_shrink=args.trust_eta_shrink,
        trust_eta_expand=args.trust_eta_expand,
        trust_max_reject=args.trust_max_reject,
        trust_subproblem_line_search=args.trust_subproblem_line_search,
        reg=args.reg,
        state_out=args.state_out,
        verbose=not args.quiet,
    )

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

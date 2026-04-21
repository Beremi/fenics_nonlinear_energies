#!/usr/bin/env python3
"""Replay a 3D direct-continuation branch on the JAX/PETSc 3D slope problem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from src.core.cli.threading import configure_jax_cpu_threading
from src.problems.slope_stability_3d.jax_petsc.solve_slope_stability_3d_dof import (
    _build_parser as _build_fixed_parser,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sanitize_lambda_tag(value: float) -> str:
    text = f"{float(value):.6f}"
    return text.replace("-", "m").replace(".", "p")


def _parse_lambda_values(raw: str) -> list[float]:
    out: list[float] = []
    for piece in str(raw).replace(",", " ").split():
        if not piece.strip():
            continue
        out.append(float(piece))
    return out


def _load_lambda_schedule(args) -> list[float]:
    values: list[float] = []
    if str(getattr(args, "lambda_values", "") or "").strip():
        values.extend(_parse_lambda_values(str(args.lambda_values)))
    path_raw = str(getattr(args, "lambda_values_file", "") or "").strip()
    if path_raw:
        path = Path(path_raw)
        text = path.read_text(encoding="utf-8").strip()
        if path.suffix.lower() in {".json", ".jsn"}:
            obj = json.loads(text)
            if isinstance(obj, dict):
                for key in ("lambda_hist", "lambda_values", "values"):
                    if key in obj:
                        values.extend(float(v) for v in obj[key])
                        break
                else:
                    raise ValueError(f"{path} does not contain lambda_hist/lambda_values/values")
            elif isinstance(obj, list):
                values.extend(float(v) for v in obj)
            else:
                raise ValueError(f"Unsupported JSON schedule payload in {path}")
        else:
            values.extend(_parse_lambda_values(text))
    if not values:
        raise ValueError("Provide --lambda-values or --lambda-values-file")
    schedule = [float(v) for v in values]
    if any(not np.isfinite(v) for v in schedule):
        raise ValueError("Lambda schedule contains non-finite values")
    if any(schedule[i + 1] <= schedule[i] for i in range(len(schedule) - 1)):
        raise ValueError("Lambda schedule must be strictly increasing")
    return schedule


def _load_displacement(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    try:
        if "displacement" not in data.files or "coords_ref" not in data.files:
            raise ValueError(f"{path} must contain displacement and coords_ref")
        return (
            np.asarray(data["coords_ref"], dtype=np.float64),
            np.asarray(data["displacement"], dtype=np.float64),
        )
    finally:
        data.close()


def _write_blended_initial_state(
    prev_state: Path | None,
    curr_state: Path,
    *,
    beta: float,
    out_path: Path,
) -> Path:
    coords_ref, curr_disp = _load_displacement(curr_state)
    if prev_state is None:
        blend_disp = np.asarray(curr_disp, dtype=np.float64)
    else:
        prev_coords_ref, prev_disp = _load_displacement(prev_state)
        if prev_coords_ref.shape != coords_ref.shape:
            raise ValueError("Previous and current states do not share the same node layout")
        if float(np.max(np.abs(prev_coords_ref - coords_ref))) > 1.0e-10:
            raise ValueError("Previous and current states do not share the same reference coordinates")
        blend_disp = float(beta) * np.asarray(prev_disp, dtype=np.float64) + (
            1.0 - float(beta)
        ) * np.asarray(curr_disp, dtype=np.float64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        coords_ref=np.asarray(coords_ref, dtype=np.float64),
        displacement=np.asarray(blend_disp, dtype=np.float64),
        coords_final=np.asarray(coords_ref + blend_disp, dtype=np.float64),
    )
    return out_path


def _step_namespace(base_args, *, lam: float, step_dir: Path, initial_state: str) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(base_args))
    ns.lambda_target = float(lam)
    ns.out = str(step_dir / "output.json")
    ns.state_out = str(step_dir / "state.npz")
    ns.progress_out = str(step_dir / "progress.json")
    ns.save_history = bool(getattr(base_args, "save_step_history", True))
    seed_initial_state = str(initial_state or getattr(base_args, "initial_state", "") or "")
    ns.initial_state = seed_initial_state
    if str(initial_state or "").strip():
        ns.elastic_initial_guess = False
    elif str(seed_initial_state).strip():
        ns.elastic_initial_guess = False
    return ns


def _omega_namespace(
    base_args,
    *,
    lam: float,
    step_dir: Path,
    initial_state: str,
) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(base_args))
    ns.lambda_target = float(lam)
    ns.out = str(step_dir / "omega_eps_output.json")
    ns.state_out = ""
    ns.progress_out = ""
    ns.save_history = False
    ns.initial_state = str(initial_state or "")
    ns.elastic_initial_guess = False
    return ns


def _build_parser(profile_defaults):
    parser = _build_fixed_parser(profile_defaults)
    parser.description = "Replay a 3D direct-continuation lambda schedule with JAX + PETSc"
    parser.add_argument("--lambda-values", type=str, default="")
    parser.add_argument("--lambda-values-file", type=str, default="")
    parser.add_argument("--branch-out-dir", type=str, required=True)
    parser.add_argument("--direct-eps", type=float, default=1.0e-2)
    parser.add_argument("--save-step-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial-lambda", type=float, default=None)
    return parser


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--nproc", type=int, default=1)
    pre_args, _ = pre_parser.parse_known_args()
    configure_jax_cpu_threading(pre_args.nproc)

    from src.problems.slope_stability_3d.jax_petsc.solver import PROFILE_DEFAULTS, run

    parser = _build_parser(PROFILE_DEFAULTS)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    rank = int(comm.rank)

    schedule = _load_lambda_schedule(args)
    eps = float(args.direct_eps)
    if eps <= 0.0:
        raise ValueError("--direct-eps must be positive")

    branch_out_dir = Path(str(args.branch_out_dir))
    if rank == 0:
        branch_out_dir.mkdir(parents=True, exist_ok=True)

    branch_records: list[dict[str, object]] = []
    initial_state_seed = str(getattr(args, "initial_state", "") or "").strip()
    initial_lambda_seed = getattr(args, "initial_lambda", None)
    if initial_lambda_seed is not None and not np.isfinite(float(initial_lambda_seed)):
        raise ValueError("--initial-lambda must be finite when provided")

    prev_lambda: float | None = float(initial_lambda_seed) if initial_state_seed else None
    prev_state_path: Path | None = Path(initial_state_seed) if initial_state_seed else None
    solver_success = True
    failure_message = ""

    for step_idx, lam in enumerate(schedule, start=1):
        tag = _sanitize_lambda_tag(lam)
        step_dir = branch_out_dir / f"step_{step_idx:03d}_lambda_{tag}"
        step_args = _step_namespace(
            args,
            lam=float(lam),
            step_dir=step_dir,
            initial_state=str(prev_state_path or ""),
        )
        result = run(step_args)
        step_output_path = Path(step_args.out)
        step_state_path = Path(step_args.state_out)
        if rank == 0:
            _write_json(step_output_path, result)
        comm.Barrier()

        step_record = {
            "step_index": int(step_idx),
            "lambda_value": float(lam),
            "output_json": str(step_output_path),
            "state_npz": str(step_state_path),
            "solver_success": bool(result.get("solver_success", False)),
            "status": str(result.get("status", "")),
            "message": str(result.get("message", "")),
            "nit": int(result.get("nit", 0)),
            "energy": float(result.get("energy", float("nan"))),
            "work": float(result.get("omega", float("nan"))),
            "u_max": float(result.get("u_max", float("nan"))),
            "final_grad_norm": float(result.get("final_grad_norm", float("nan"))),
            "linear_iterations_total": int(result.get("linear_iterations_total", 0)),
            "initial_state": str(prev_state_path or ""),
        }

        if not bool(result.get("solver_success", False)):
            solver_success = False
            failure_message = str(result.get("message", "step solve failed"))
            branch_records.append(step_record)
            break

        d_lambda = float(lam - prev_lambda) if prev_lambda is not None else 1000.0
        beta = min(1.0, eps / max(d_lambda, eps))
        blended_path = step_dir / "omega_eps_initial_state.npz"
        if rank == 0:
            _write_blended_initial_state(
                prev_state_path,
                step_state_path,
                beta=float(beta),
                out_path=blended_path,
            )
        comm.Barrier()
        omega_args = _omega_namespace(
            args,
            lam=float(lam - eps),
            step_dir=step_dir,
            initial_state=str(blended_path),
        )
        omega_result = run(omega_args)
        if rank == 0:
            _write_json(Path(omega_args.out), omega_result)
        comm.Barrier()
        step_record["direct_eps"] = float(eps)
        step_record["direct_beta"] = float(beta)
        step_record["omega_eps_output_json"] = str(Path(omega_args.out))
        step_record["omega_eps_solver_success"] = bool(omega_result.get("solver_success", False))
        step_record["omega_eps_message"] = str(omega_result.get("message", ""))
        step_record["energy_eps"] = float(omega_result.get("energy", float("nan")))
        if bool(omega_result.get("solver_success", False)):
            step_record["direct_omega"] = float(
                (float(omega_result["energy"]) - float(result["energy"])) / float(eps)
            )
        else:
            step_record["direct_omega"] = float("nan")

        branch_records.append(step_record)
        prev_lambda = float(lam)
        prev_state_path = step_state_path

        if rank == 0:
            print(
                "branch step "
                f"{step_idx:03d} | lambda={lam:.6f} | nit={int(result.get('nit', 0))} "
                f"| u_max={float(result.get('u_max', float('nan'))):.6e} "
                f"| work={float(result.get('omega', float('nan'))):.6e}",
                flush=True,
            )

    summary = {
        "mesh_name": str(args.mesh_name),
        "elem_degree": int(args.elem_degree),
        "lambda_schedule": [float(v) for v in schedule],
        "completed_steps": int(len(branch_records)),
        "solver_success": bool(solver_success),
        "failure_message": str(failure_message),
        "steps": branch_records,
    }
    if branch_records:
        summary["lambda_hist"] = [float(step["lambda_value"]) for step in branch_records]
        summary["work_hist"] = [float(step["work"]) for step in branch_records]
        summary["u_max_hist"] = [float(step["u_max"]) for step in branch_records]
        summary["energy_hist"] = [float(step["energy"]) for step in branch_records]
        summary["direct_omega_hist"] = [
            float(step.get("direct_omega", float("nan"))) for step in branch_records
        ]

    if rank == 0:
        _write_json(branch_out_dir / "branch_summary.json", summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

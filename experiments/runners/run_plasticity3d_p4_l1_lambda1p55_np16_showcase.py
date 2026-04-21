#!/usr/bin/env python3
"""Run a preserved Plasticity3D `P4(L1)` showcase at `lambda = 1.55` on 16 MPI ranks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.benchmark.replication import command_text, now_iso, read_json, run_logged_command, write_json

from experiments.runners import run_plasticity3d_backend_mix_compare as mix_tools


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
CASE_RUNNER = REPO_ROOT / "experiments" / "runners" / "run_plasticity3d_backend_mix_case.py"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "tmp" / "source_compare" / "slope_stability_petsc4py"
DEFAULT_RAW_DIR = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "docs_showcase"
    / "plasticity3d_p4_l1_lambda1p55_np16_grad1e2"
)
CONSTRAINT_VARIANT = "glued_bottom"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def _build_command(*, source_root: Path, out_dir: Path) -> list[str]:
    return [
        "mpiexec",
        "-n",
        "16",
        str(PYTHON),
        "-u",
        str(CASE_RUNNER),
        "--assembly-backend",
        "local_constitutiveAD",
        "--solver-backend",
        "local_pmg",
        "--source-root",
        str(source_root),
        "--out-dir",
        str(out_dir),
        "--output-json",
        str(out_dir / "output.json"),
        "--state-out",
        str(out_dir / "state.npz"),
        "--mesh-name",
        "hetero_ssr_L1",
        "--elem-degree",
        "4",
        "--constraint-variant",
        CONSTRAINT_VARIANT,
        "--lambda-target",
        "1.55",
        "--pmg-strategy",
        "same_mesh_p4_p2_p1",
        "--ksp-rtol",
        "1e-1",
        "--ksp-max-it",
        "100",
        "--convergence-mode",
        "gradient_only",
        "--grad-stop-tol",
        "0.01",
        "--stop-tol",
        "0.0",
        "--maxit",
        "200",
        "--line-search",
        "armijo",
        "--armijo-max-ls",
        "40",
    ]


def _write_progress_json(*, output_payload: dict[str, object], out_path: Path) -> None:
    progress = {
        "status": str(output_payload.get("status", "")),
        "message": str(output_payload.get("message", "")),
        "mesh_name": str(output_payload.get("mesh_name", "")),
        "elem_degree": int(output_payload.get("elem_degree", 4)),
        "constraint_variant": str(output_payload.get("constraint_variant", "")),
        "lambda_target": float(output_payload.get("lambda_target", float("nan"))),
        "iterations_completed": int(output_payload.get("nit", 0)),
        "energy": float(output_payload.get("energy", float("nan"))),
        "history": list(output_payload.get("history", [])),
        "newton_regularization": {
            "enabled": False,
            "current_r": 1.0,
            "history": [],
        },
    }
    write_json(out_path, progress)


def _write_showcase_meta(
    *,
    out_dir: Path,
    command: list[str],
    env: dict[str, str],
    output_payload: dict[str, object],
) -> Path:
    meta = {
        "timestamp_utc": now_iso(),
        "nprocs": 16,
        "mesh_name": "hetero_ssr_L1",
        "elem_degree": 4,
        "constraint_variant": CONSTRAINT_VARIANT,
        "same_mesh_case_path": str(output_payload.get("same_mesh_case_path", "")),
        "lambda_target": 1.55,
        "assembly_backend": "local_constitutiveAD",
        "solver_backend": "local_pmg",
        "pmg_strategy": "same_mesh_p4_p2_p1",
        "line_search": "armijo",
        "convergence_mode": "gradient_only",
        "grad_stop_tol": 1.0e-2,
        "maxit": 200,
        "stop_tol": 0.0,
        "ksp_rtol": 1.0e-1,
        "ksp_max_it": 100,
        "command": command_text(command),
        "command_argv": list(command),
        "thread_caps": {
            key: str(env.get(key, ""))
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "result": {
            "status": str(output_payload.get("status", "")),
            "message": str(output_payload.get("message", "")),
            "nit": int(output_payload.get("nit", 0)),
            "final_grad_norm": float(output_payload.get("final_grad_norm", float("nan"))),
            "energy": float(output_payload.get("energy", float("nan"))),
            "omega": float(output_payload.get("omega", float("nan"))),
            "u_max": float(output_payload.get("u_max", float("nan"))),
            "solve_time": float(output_payload.get("solve_time", float("nan"))),
            "total_time": float(output_payload.get("total_time", float("nan"))),
            "linear_iterations_total": int(output_payload.get("linear_iterations_total", 0)),
        },
    }
    path = out_dir / "showcase_meta.json"
    write_json(path, meta)
    return path


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    source_root = Path(args.source_root).resolve()
    output_json = out_dir / "output.json"
    state_npz = out_dir / "state.npz"
    progress_json = out_dir / "progress.json"
    env = mix_tools._mixed_env(source_root)
    command = _build_command(source_root=source_root, out_dir=out_dir)

    run_logged_command(
        command=command,
        cwd=REPO_ROOT,
        leaf_dir=out_dir,
        expected_outputs=[output_json, state_npz],
        env=env,
        resume=bool(args.resume),
        notes="Authoritative 16-rank maintained-local Plasticity3D P4(L1) lambda=1.55 glued-bottom showcase run.",
    )

    payload = read_json(output_json)
    payload["state_out"] = str(state_npz.resolve())
    write_json(output_json, payload)

    if float(payload.get("lambda_target", float("nan"))) != 1.55:
        raise RuntimeError("Showcase output recorded an unexpected lambda_target")
    if str(payload.get("mesh_name", "")) != "hetero_ssr_L1":
        raise RuntimeError("Showcase output recorded an unexpected mesh_name")
    if int(payload.get("elem_degree", 0)) != 4:
        raise RuntimeError("Showcase output recorded an unexpected elem_degree")
    if int(payload.get("ranks", 0)) != 16:
        raise RuntimeError("Showcase output did not record a 16-rank run")
    if int(payload.get("nit", 10**9)) > 200:
        raise RuntimeError("Showcase run exceeded the Newton iteration cap")
    if str(payload.get("constraint_variant", "")) != CONSTRAINT_VARIANT:
        raise RuntimeError("Showcase output recorded an unexpected constraint_variant")

    _write_progress_json(output_payload=payload, out_path=progress_json)
    _write_showcase_meta(
        out_dir=out_dir,
        command=command,
        env=env,
        output_payload=payload,
    )

    print(
        json.dumps(
            {
                "raw_dir": str(out_dir),
                "status": str(payload.get("status", "")),
                "message": str(payload.get("message", "")),
                "nit": int(payload.get("nit", 0)),
                "final_grad_norm": float(payload.get("final_grad_norm", float("nan"))),
                "linear_iterations_total": int(payload.get("linear_iterations_total", 0)),
                "solve_time": float(payload.get("solve_time", float("nan"))),
                "total_time": float(payload.get("total_time", float("nan"))),
                "energy": float(payload.get("energy", float("nan"))),
                "omega": float(payload.get("omega", float("nan"))),
                "u_max": float(payload.get("u_max", float("nan"))),
                "state_out": str(payload.get("state_out", "")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run or assemble the recommended Plasticity3D local-only scaling dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.runners import (
    run_plasticity3d_backend_mix_compare as mix_tools,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "tmp" / "source_compare" / "slope_stability_petsc4py"
DEFAULT_SEED_SUMMARY = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "source_compare"
    / "plasticity3d_l1_2_lambda1_grad1e2_scaling"
    / "comparison_summary.json"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "source_compare"
    / "plasticity3d_l1_2_lambda1_grad1e2_local_pmg_scaling"
)
DEFAULT_RANKS = (1, 2)
SEEDED_RANKS = (4, 8, 16, 32)
SUMMARY_NAME = "comparison_summary.json"
RUNNER_NAME = "plasticity3d_l1_2_lambda1_grad1e2_local_pmg_scaling"

IMPLEMENTATION = {
    "name": "local_constitutiveAD_local_pmg_armijo",
    "display_label": "local_constitutiveAD assembly + local_pmg solver",
    "family": "local",
    "assembly_backend": "local_constitutiveAD",
    "solver_backend": "local_pmg",
    "solver_profile": "local_pmg_armijo_grad1e2",
}

ROW_KEYS = (
    "case_id",
    "implementation",
    "display_label",
    "family",
    "assembly_backend",
    "solver_backend",
    "solver_profile",
    "combo_label",
    "ranks",
    "status",
    "message",
    "solver_success",
    "exit_code",
    "wall_time_s",
    "solve_time_s",
    "nit",
    "linear_iterations_total",
    "final_metric",
    "final_metric_name",
    "energy",
    "omega",
    "u_max",
    "stdout_path",
    "stderr_path",
    "result_json",
    "case_dir",
    "stage_jsonl",
    "source_builder_timings_json",
    "command",
    "history_metric_name",
    "history_iterations",
    "history_metric",
    "initial_guess_enabled",
    "initial_guess_success",
    "initial_guess_ksp_iterations",
    "native_run_info",
    "native_history_json",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _case_id(ranks: int) -> str:
    return (
        f"np{int(ranks)}:"
        f"{IMPLEMENTATION['assembly_backend']}_assembly:"
        f"{IMPLEMENTATION['solver_backend']}_solver"
    )


def _augment_row(base_row: dict[str, object], case_dir: Path) -> dict[str, object]:
    row = dict(base_row)
    stage_jsonl = case_dir / "data" / "stage.jsonl"
    row.update(
        {
            "implementation": str(IMPLEMENTATION["name"]),
            "display_label": str(IMPLEMENTATION["display_label"]),
            "family": str(IMPLEMENTATION["family"]),
            "assembly_backend": str(IMPLEMENTATION["assembly_backend"]),
            "solver_backend": str(IMPLEMENTATION["solver_backend"]),
            "solver_profile": str(IMPLEMENTATION["solver_profile"]),
            "stage_jsonl": mix_tools._repo_rel(stage_jsonl) if stage_jsonl.exists() else "",
            "source_builder_timings_json": "",
            "native_run_info": "",
            "native_history_json": "",
        }
    )
    return row


def _failed_row(
    *,
    ranks: int,
    exit_code: int,
    message: str,
    case_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    result_path: Path,
    command: list[str],
) -> dict[str, object]:
    stage_jsonl = case_dir / "data" / "stage.jsonl"
    return {
        "case_id": _case_id(ranks),
        "implementation": str(IMPLEMENTATION["name"]),
        "display_label": str(IMPLEMENTATION["display_label"]),
        "family": str(IMPLEMENTATION["family"]),
        "assembly_backend": str(IMPLEMENTATION["assembly_backend"]),
        "solver_backend": str(IMPLEMENTATION["solver_backend"]),
        "solver_profile": str(IMPLEMENTATION["solver_profile"]),
        "combo_label": mix_tools._combo_label(
            str(IMPLEMENTATION["assembly_backend"]),
            str(IMPLEMENTATION["solver_backend"]),
        ),
        "ranks": int(ranks),
        "status": "failed",
        "message": str(message),
        "solver_success": False,
        "exit_code": int(exit_code),
        "wall_time_s": float("nan"),
        "solve_time_s": float("nan"),
        "nit": 0,
        "linear_iterations_total": 0,
        "final_metric": float("nan"),
        "final_metric_name": "",
        "energy": float("nan"),
        "omega": float("nan"),
        "u_max": float("nan"),
        "stdout_path": mix_tools._repo_rel(stdout_path),
        "stderr_path": mix_tools._repo_rel(stderr_path),
        "result_json": mix_tools._repo_rel(result_path),
        "case_dir": mix_tools._repo_rel(case_dir),
        "stage_jsonl": mix_tools._repo_rel(stage_jsonl) if stage_jsonl.exists() else "",
        "source_builder_timings_json": "",
        "command": " ".join(command),
        "history_metric_name": "",
        "history_iterations": [],
        "history_metric": [],
        "initial_guess_enabled": False,
        "initial_guess_success": False,
        "initial_guess_ksp_iterations": 0,
        "native_run_info": "",
        "native_history_json": "",
    }


def _build_case_command(
    *,
    source_root: Path,
    case_dir: Path,
    result_path: Path,
    ranks: int,
    grad_stop_tol: float,
    maxit: int,
) -> list[str]:
    return [
        "mpiexec",
        "-n",
        str(int(ranks)),
        str(mix_tools.PYTHON),
        "-u",
        str(mix_tools.CASE_RUNNER),
        "--assembly-backend",
        str(IMPLEMENTATION["assembly_backend"]),
        "--solver-backend",
        str(IMPLEMENTATION["solver_backend"]),
        "--source-root",
        str(source_root),
        "--out-dir",
        str(case_dir),
        "--output-json",
        str(result_path),
        "--mesh-name",
        "hetero_ssr_L1_2",
        "--lambda-target",
        "1.0",
        "--ksp-rtol",
        "1e-1",
        "--ksp-max-it",
        "100",
        "--convergence-mode",
        "gradient_only",
        "--grad-stop-tol",
        str(float(grad_stop_tol)),
        "--stop-tol",
        "0.0",
        "--maxit",
        str(int(maxit)),
        "--line-search",
        "armijo",
        "--armijo-max-ls",
        "40",
    ]


def _seed_rows(seed_summary: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    if not seed_summary.exists():
        return rows
    payload = _read_json(seed_summary)
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("implementation", "")).strip() != str(IMPLEMENTATION["name"]):
            continue
        rows[str(row.get("case_id", ""))] = dict(row)
    return rows


def _write_summary(
    *,
    summary_path: Path,
    rows_by_case: dict[str, dict[str, object]],
    ranks: list[int],
    grad_stop_tol: float,
    maxit: int,
    source_root: Path,
    source_env_mode: str,
) -> None:
    ordered_rows = sorted(
        rows_by_case.values(),
        key=lambda row: int(row.get("ranks", 10**6)),
    )
    payload = {
        "runner": RUNNER_NAME,
        "source_root": mix_tools._repo_rel(source_root),
        "source_env_mode": str(source_env_mode),
        "out_dir": mix_tools._repo_rel(summary_path.parent),
        "ranks": [int(v) for v in ranks],
        "stop_metric_name": "grad_norm",
        "grad_stop_tol": float(grad_stop_tol),
        "maxit": int(maxit),
        "linear_solver_rtol": 1.0e-1,
        "mesh_name": "hetero_ssr_L1_2",
        "lambda_target": 1.0,
        "line_search": "armijo",
        "implementations": [
            {
                "implementation": str(IMPLEMENTATION["name"]),
                "display_label": str(IMPLEMENTATION["display_label"]),
                "family": str(IMPLEMENTATION["family"]),
                "assembly_backend": str(IMPLEMENTATION["assembly_backend"]),
                "solver_backend": str(IMPLEMENTATION["solver_backend"]),
                "solver_profile": str(IMPLEMENTATION["solver_profile"]),
            }
        ],
        "row_keys": list(ROW_KEYS),
        "rows": ordered_rows,
    }
    _write_json(summary_path, payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or assemble the local-only L1_2/lambda=1/grad<1e-2 scaling dataset."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--seed-summary", type=Path, default=DEFAULT_SEED_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ranks", type=int, nargs="+", default=list(DEFAULT_RANKS))
    parser.add_argument("--seed-ranks", type=int, nargs="+", default=list(SEEDED_RANKS))
    parser.add_argument("--grad-stop-tol", type=float, default=1.0e-2)
    parser.add_argument("--maxit", type=int, default=50)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    seed_summary = Path(args.seed_summary).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / SUMMARY_NAME

    source_env_mode = mix_tools._ensure_shared_source_env(source_root)
    env = mix_tools._mixed_env(source_root)

    rows_by_case: dict[str, dict[str, object]] = {}
    if bool(args.resume) and summary_path.exists():
        payload = _read_json(summary_path)
        for row in payload.get("rows", []):
            if isinstance(row, dict):
                rows_by_case[str(row.get("case_id", ""))] = dict(row)

    if not rows_by_case:
        rows_by_case.update(_seed_rows(seed_summary))
    else:
        for case_id, row in _seed_rows(seed_summary).items():
            rows_by_case.setdefault(case_id, row)

    all_ranks = sorted({int(v) for v in list(args.seed_ranks) + list(args.ranks)})

    for ranks in [int(v) for v in args.ranks]:
        case_id = _case_id(ranks)
        case_dir = (
            out_dir
            / "runs"
            / f"np{ranks}"
            / f"solver_{IMPLEMENTATION['solver_backend']}"
            / f"assembly_{IMPLEMENTATION['assembly_backend']}"
        )
        stdout_path = case_dir / "stdout.txt"
        stderr_path = case_dir / "stderr.txt"
        result_path = case_dir / "output.json"

        existing = rows_by_case.get(case_id)
        if (
            bool(args.resume)
            and existing is not None
            and str(existing.get("status", "")).startswith("completed")
            and result_path.exists()
        ):
            continue

        command = _build_case_command(
            source_root=source_root,
            case_dir=case_dir,
            result_path=result_path,
            ranks=ranks,
            grad_stop_tol=float(args.grad_stop_tol),
            maxit=int(args.maxit),
        )
        exit_code, _wall = mix_tools._run_command(
            cmd=command,
            cwd=REPO_ROOT,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            result_path=result_path,
        )

        if exit_code != 0 or not result_path.exists():
            message = (
                mix_tools._tail_text(stderr_path)
                or mix_tools._tail_text(stdout_path)
                or "subprocess failed"
            )
            row = _failed_row(
                ranks=ranks,
                exit_code=int(exit_code),
                message=message,
                case_dir=case_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                result_path=result_path,
                command=command,
            )
        else:
            base_row = mix_tools._normalize_payload(
                case_id=case_id,
                assembly_backend=str(IMPLEMENTATION["assembly_backend"]),
                solver_backend=str(IMPLEMENTATION["solver_backend"]),
                ranks=ranks,
                exit_code=int(exit_code),
                case_dir=case_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                result_path=result_path,
                command=command,
            )
            row = _augment_row(base_row, case_dir)

        rows_by_case[case_id] = row
        _write_summary(
            summary_path=summary_path,
            rows_by_case=rows_by_case,
            ranks=all_ranks,
            grad_stop_tol=float(args.grad_stop_tol),
            maxit=int(args.maxit),
            source_root=source_root,
            source_env_mode=source_env_mode,
        )

    _write_summary(
        summary_path=summary_path,
        rows_by_case=rows_by_case,
        ranks=all_ranks,
        grad_stop_tol=float(args.grad_stop_tol),
        maxit=int(args.maxit),
        source_root=source_root,
        source_env_mode=source_env_mode,
    )
    print(summary_path)


if __name__ == "__main__":
    main()

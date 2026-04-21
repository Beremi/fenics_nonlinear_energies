#!/usr/bin/env python3
"""Run strong scaling for matched Plasticity3D L1_2 gradient-stop PMG variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.runners import (
    run_plasticity3d_backend_mix_compare as mix_tools,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "tmp" / "source_compare" / "slope_stability_petsc4py"
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "source_compare"
    / "plasticity3d_l1_2_lambda1_grad1e2_scaling_all_pmg"
)
DEFAULT_RANKS = (4, 8, 16, 32)
SUMMARY_NAME = "comparison_summary.json"
RUNNER_NAME = "plasticity3d_l1_2_lambda1_grad1e2_scaling_compare_all_pmg"

IMPLEMENTATIONS = (
    {
        "name": "local_constitutiveAD_local_pmg_armijo",
        "display_label": "local_constitutiveAD assembly + local_pmg solver",
        "family": "local",
        "assembly_backend": "local_constitutiveAD",
        "solver_backend": "local_pmg",
        "solver_profile": "local_pmg_armijo_grad1e2",
    },
    {
        "name": "source_local_pmg_armijo",
        "display_label": "source assembly + local_pmg solver",
        "family": "local",
        "assembly_backend": "source",
        "solver_backend": "local_pmg",
        "solver_profile": "local_pmg_armijo_grad1e2",
    },
    {
        "name": "local_constitutiveAD_local_pmg_sourcefixed_armijo",
        "display_label": "local_constitutiveAD assembly + local_pmg_sourcefixed solver",
        "family": "local",
        "assembly_backend": "local_constitutiveAD",
        "solver_backend": "local_pmg_sourcefixed",
        "solver_profile": "local_pmg_sourcefixed_armijo_grad1e2",
    },
    {
        "name": "source_local_pmg_sourcefixed_armijo",
        "display_label": "source assembly + local_pmg_sourcefixed solver",
        "family": "local",
        "assembly_backend": "source",
        "solver_backend": "local_pmg_sourcefixed",
        "solver_profile": "local_pmg_sourcefixed_armijo_grad1e2",
    },
)

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


def _augment_row(
    *,
    impl: dict[str, str],
    base_row: dict[str, object],
    case_dir: Path,
) -> dict[str, object]:
    stage_jsonl = case_dir / "data" / "stage.jsonl"
    source_builder_timings_json = case_dir / "data" / "source_builder_timings.json"
    row = dict(base_row)
    row.update(
        {
            "implementation": str(impl["name"]),
            "display_label": str(impl["display_label"]),
            "family": str(impl["family"]),
            "assembly_backend": str(impl["assembly_backend"]),
            "solver_backend": str(impl["solver_backend"]),
            "solver_profile": str(impl["solver_profile"]),
            "stage_jsonl": (
                mix_tools._repo_rel(stage_jsonl) if stage_jsonl.exists() else ""
            ),
            "source_builder_timings_json": (
                mix_tools._repo_rel(source_builder_timings_json)
                if source_builder_timings_json.exists()
                else ""
            ),
            "native_run_info": "",
            "native_history_json": "",
        }
    )
    return row


def _failed_row(
    *,
    impl: dict[str, str],
    case_id: str,
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
    source_builder_timings_json = case_dir / "data" / "source_builder_timings.json"
    return {
        "case_id": str(case_id),
        "implementation": str(impl["name"]),
        "display_label": str(impl["display_label"]),
        "family": str(impl["family"]),
        "assembly_backend": str(impl["assembly_backend"]),
        "solver_backend": str(impl["solver_backend"]),
        "solver_profile": str(impl["solver_profile"]),
        "combo_label": mix_tools._combo_label(
            str(impl["assembly_backend"]),
            str(impl["solver_backend"]),
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
        "stage_jsonl": (
            mix_tools._repo_rel(stage_jsonl) if stage_jsonl.exists() else ""
        ),
        "source_builder_timings_json": (
            mix_tools._repo_rel(source_builder_timings_json)
            if source_builder_timings_json.exists()
            else ""
        ),
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
    impl_order = {str(item["name"]): idx for idx, item in enumerate(IMPLEMENTATIONS)}
    rows = sorted(
        rows_by_case.values(),
        key=lambda row: (
            list(ranks).index(int(row["ranks"])) if int(row["ranks"]) in ranks else 10**6,
            impl_order.get(str(row["implementation"]), 10**6),
        ),
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
                "implementation": str(item["name"]),
                "display_label": str(item["display_label"]),
                "family": str(item["family"]),
                "assembly_backend": str(item["assembly_backend"]),
                "solver_backend": str(item["solver_backend"]),
                "solver_profile": str(item["solver_profile"]),
            }
            for item in IMPLEMENTATIONS
        ],
        "row_keys": list(ROW_KEYS),
        "rows": rows,
    }
    _write_json(summary_path, payload)


def _build_case_command(
    *,
    source_root: Path,
    case_dir: Path,
    result_path: Path,
    ranks: int,
    assembly_backend: str,
    solver_backend: str,
    grad_stop_tol: float,
    maxit: int,
) -> list[str]:
    return [
        "mpiexec",
        "-n",
        str(ranks),
        str(mix_tools.PYTHON),
        "-u",
        str(mix_tools.CASE_RUNNER),
        "--assembly-backend",
        str(assembly_backend),
        "--solver-backend",
        str(solver_backend),
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
        str(grad_stop_tol),
        "--stop-tol",
        "0.0",
        "--maxit",
        str(maxit),
        "--line-search",
        "armijo",
        "--armijo-max-ls",
        "40",
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run L1_2/lambda=1 gradient-stop scaling for local vs source PMG variants."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ranks", type=int, nargs="+", default=list(DEFAULT_RANKS))
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
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / SUMMARY_NAME

    source_env_mode = mix_tools._ensure_shared_source_env(source_root)
    env = mix_tools._mixed_env(source_root)

    existing_rows: dict[str, dict[str, object]] = {}
    if bool(args.resume) and summary_path.exists():
        payload = _read_json(summary_path)
        for row in payload.get("rows", []):
            existing_rows[str(row.get("case_id", ""))] = dict(row)

    for ranks in [int(v) for v in args.ranks]:
        for impl in IMPLEMENTATIONS:
            assembly_backend = str(impl["assembly_backend"])
            solver_backend = str(impl["solver_backend"])
            case_id = f"np{ranks}:{assembly_backend}_assembly:{solver_backend}_solver"
            case_dir = (
                out_dir
                / "runs"
                / f"np{ranks}"
                / f"solver_{solver_backend}"
                / f"assembly_{assembly_backend}"
            )
            stdout_path = case_dir / "stdout.txt"
            stderr_path = case_dir / "stderr.txt"
            result_path = case_dir / "output.json"

            existing = existing_rows.get(case_id)
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
                ranks=int(ranks),
                assembly_backend=assembly_backend,
                solver_backend=solver_backend,
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
                    impl=impl,
                    case_id=case_id,
                    ranks=int(ranks),
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
                    assembly_backend=assembly_backend,
                    solver_backend=solver_backend,
                    ranks=int(ranks),
                    exit_code=int(exit_code),
                    case_dir=case_dir,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    result_path=result_path,
                    command=command,
                )
                row = _augment_row(impl=impl, base_row=base_row, case_dir=case_dir)

            existing_rows[case_id] = row
            _write_summary(
                summary_path=summary_path,
                rows_by_case=existing_rows,
                ranks=[int(v) for v in args.ranks],
                grad_stop_tol=float(args.grad_stop_tol),
                maxit=int(args.maxit),
                source_root=source_root,
                source_env_mode=source_env_mode,
            )

    print(summary_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a Plasticity3D implementation scaling comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from time import perf_counter

from experiments.runners import (
    run_plasticity3d_backend_mix_compare as mix_tools,
)
from experiments.runners import (
    run_plasticity3d_p4_l1_lambda1p5_source_compare as source_compare_tools,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
CASE_RUNNER = REPO_ROOT / "experiments" / "runners" / "run_plasticity3d_backend_mix_case.py"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "tmp" / "source_compare" / "slope_stability_petsc4py"
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "source_compare"
    / "plasticity3d_impl_scaling"
)
DEFAULT_RANKS = (1, 2, 4, 8, 16, 32)
SUMMARY_NAME = "comparison_summary.json"
RUNNER_NAME = "plasticity3d_impl_scaling_compare"

IMPLEMENTATIONS = (
    {
        "name": "maintained_local_best",
        "display_label": "Maintained local_constitutiveAD + local solver (fast Hypre)",
        "family": "local",
        "assembly_backend": "local_constitutiveAD",
        "solver_backend": "local",
        "solver_profile": "hypre_fast",
        "command_kind": "mix_case",
        "normalize_kind": "mix_case",
    },
    {
        "name": "source_petsc4py",
        "display_label": "Source assembly + source solver (fixed PMG-shell)",
        "family": "source",
        "assembly_backend": "source",
        "solver_backend": "source",
        "solver_profile": "pmg_shell_fixed",
        "command_kind": "source_pmg",
        "normalize_kind": "source_pmg",
    },
    {
        "name": "maintained_local_pmg",
        "display_label": "Maintained local_constitutiveAD + local solver (PMG)",
        "family": "local",
        "assembly_backend": "local_constitutiveAD",
        "solver_backend": "local_pmg",
        "solver_profile": "pmg",
        "command_kind": "local_pmg",
        "normalize_kind": "local_pmg",
    },
    {
        "name": "source_petsc4py_pmg",
        "display_label": "Source assembly + source solver (Hypre legacy)",
        "family": "source",
        "assembly_backend": "source",
        "solver_backend": "source",
        "solver_profile": "hypre_legacy",
        "command_kind": "mix_case",
        "normalize_kind": "mix_case",
    },
)

NORMALIZED_ROW_KEYS = (
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


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tail_text(path: Path) -> str:
    if not path.exists():
        return ""
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    return lines[-1] if lines else ""


def _combo_label(assembly_backend: str, solver_backend: str) -> str:
    return f"{str(assembly_backend)} assembly + {str(solver_backend)} solver"


def _implementation_by_name(name: str) -> dict[str, str]:
    for impl in IMPLEMENTATIONS:
        if str(impl["name"]) == str(name):
            return dict(impl)
    raise KeyError(f"Unknown implementation: {name}")


def _upgrade_row(row: dict[str, object]) -> dict[str, object]:
    impl_name = str(row.get("implementation", "")).strip()
    if not impl_name:
        return dict(row)
    try:
        impl = _implementation_by_name(impl_name)
    except KeyError:
        return dict(row)
    upgraded = dict(row)
    upgraded.setdefault("display_label", str(impl["display_label"]))
    upgraded.setdefault("family", str(impl["family"]))
    upgraded.setdefault("assembly_backend", str(impl["assembly_backend"]))
    upgraded.setdefault("solver_backend", str(impl["solver_backend"]))
    upgraded.setdefault("solver_profile", str(impl["solver_profile"]))
    upgraded.setdefault(
        "combo_label",
        _combo_label(str(impl["assembly_backend"]), str(impl["solver_backend"])),
    )
    upgraded.setdefault("stage_jsonl", "")
    upgraded.setdefault("source_builder_timings_json", "")
    upgraded.setdefault("history_metric_name", "")
    upgraded.setdefault("history_iterations", [])
    upgraded.setdefault("history_metric", [])
    upgraded.setdefault("initial_guess_enabled", False)
    upgraded.setdefault("initial_guess_success", False)
    upgraded.setdefault("initial_guess_ksp_iterations", 0)
    upgraded.setdefault("native_run_info", "")
    upgraded.setdefault("native_history_json", "")
    return upgraded


def _build_mix_case_command(
    *,
    source_root: Path,
    case_dir: Path,
    result_path: Path,
    ranks: int,
    assembly_backend: str,
    solver_backend: str,
    stop_tol: float,
    maxit: int,
    ) -> list[str]:
    return [
        "mpiexec",
        "-n",
        str(ranks),
        str(PYTHON),
        "-u",
        str(CASE_RUNNER),
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
        "--stop-tol",
        str(stop_tol),
        "--maxit",
        str(maxit),
    ]


def _build_local_pmg_command(
    *,
    ranks: int,
    result_path: Path,
    stop_tol: float,
    maxit: int,
) -> list[str]:
    return source_compare_tools._build_local_command(
        ranks=int(ranks),
        result_path=result_path,
        mode="reference",
        fixed_maxit=int(maxit),
        reference_stop_policy="matched_relative_correction",
        reference_stop_tol=float(stop_tol),
        reference_maxit=int(maxit),
    )


def _build_source_pmg_command(
    *,
    source_root: Path,
    source_python: Path,
    helper_path: Path,
    case_dir: Path,
    result_path: Path,
    ranks: int,
    stop_tol: float,
    maxit: int,
) -> list[str]:
    return source_compare_tools._build_source_command(
        source_root=source_root,
        source_python=source_python,
        helper_path=helper_path,
        case_dir=case_dir,
        result_path=result_path,
        ranks=int(ranks),
        mode="reference",
        fixed_maxit=int(maxit),
        reference_stop_policy="matched_relative_correction",
        reference_stop_tol=float(stop_tol),
        reference_maxit=int(maxit),
        source_pc_backend="pmg_shell",
    )


def _build_command(
    *,
    impl: dict[str, str],
    source_root: Path,
    source_python: Path,
    helper_path: Path,
    case_dir: Path,
    result_path: Path,
    ranks: int,
    stop_tol: float,
    maxit: int,
) -> list[str]:
    kind = str(impl["command_kind"])
    if kind == "mix_case":
        return _build_mix_case_command(
            source_root=source_root,
            case_dir=case_dir,
            result_path=result_path,
            ranks=int(ranks),
            assembly_backend=str(impl["assembly_backend"]),
            solver_backend=str(impl["solver_backend"]),
            stop_tol=float(stop_tol),
            maxit=int(maxit),
        )
    if kind == "local_pmg":
        return _build_local_pmg_command(
            ranks=int(ranks),
            result_path=result_path,
            stop_tol=float(stop_tol),
            maxit=int(maxit),
        )
    if kind == "source_pmg":
        return _build_source_pmg_command(
            source_root=source_root,
            source_python=source_python,
            helper_path=helper_path,
            case_dir=case_dir,
            result_path=result_path,
            ranks=int(ranks),
            stop_tol=float(stop_tol),
            maxit=int(maxit),
        )
    raise ValueError(f"Unsupported command kind: {kind}")


def _failed_row(
    *,
    case_id: str,
    impl: dict[str, str],
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
        "combo_label": _combo_label(
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
        "stdout_path": _repo_rel(stdout_path),
        "stderr_path": _repo_rel(stderr_path),
        "result_json": _repo_rel(result_path),
        "case_dir": _repo_rel(case_dir),
        "stage_jsonl": _repo_rel(stage_jsonl),
        "source_builder_timings_json": (
            _repo_rel(source_builder_timings_json)
            if source_builder_timings_json.exists()
            else ""
        ),
        "command": shlex.join(command),
        "history_metric_name": "",
        "history_iterations": [],
        "history_metric": [],
        "initial_guess_enabled": False,
        "initial_guess_success": False,
        "initial_guess_ksp_iterations": 0,
        "native_run_info": "",
        "native_history_json": "",
    }


def _normalize_mix_payload(
    *,
    case_id: str,
    impl: dict[str, str],
    ranks: int,
    exit_code: int,
    case_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    result_path: Path,
    command: list[str],
) -> dict[str, object]:
    payload = _read_json(result_path)
    history = list(payload.get("history", []))
    history_iterations: list[int] = []
    history_metric: list[float] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        raw_it = item.get("it", item.get("iteration"))
        raw_metric = item.get("step_rel", item.get("metric"))
        if raw_it is None or raw_metric is None:
            continue
        history_iterations.append(int(raw_it))
        history_metric.append(float(raw_metric))
    initial_guess = dict(payload.get("initial_guess", {}))
    stage_jsonl = case_dir / "data" / "stage.jsonl"
    source_builder_timings_json = case_dir / "data" / "source_builder_timings.json"
    row = {
        "case_id": str(case_id),
        "implementation": str(impl["name"]),
        "display_label": str(impl["display_label"]),
        "family": str(impl["family"]),
        "assembly_backend": str(impl["assembly_backend"]),
        "solver_backend": str(impl["solver_backend"]),
        "solver_profile": str(impl["solver_profile"]),
        "combo_label": _combo_label(
            str(impl["assembly_backend"]),
            str(impl["solver_backend"]),
        ),
        "ranks": int(ranks),
        "status": str(payload.get("status", "failed")),
        "message": str(payload.get("message", "")),
        "solver_success": bool(payload.get("solver_success", False)),
        "exit_code": int(exit_code),
        "wall_time_s": float(payload.get("total_time", float("nan"))),
        "solve_time_s": float(payload.get("solve_time", float("nan"))),
        "nit": int(payload.get("nit", 0)),
        "linear_iterations_total": int(payload.get("linear_iterations_total", 0)),
        "final_metric": float(payload.get("final_metric", float("nan"))),
        "final_metric_name": str(payload.get("final_metric_name", "")),
        "energy": float(payload.get("energy", float("nan"))),
        "omega": float(payload.get("omega", float("nan"))),
        "u_max": float(payload.get("u_max", float("nan"))),
        "stdout_path": _repo_rel(stdout_path),
        "stderr_path": _repo_rel(stderr_path),
        "result_json": _repo_rel(result_path),
        "case_dir": _repo_rel(case_dir),
        "stage_jsonl": _repo_rel(stage_jsonl),
        "source_builder_timings_json": (
            _repo_rel(source_builder_timings_json)
            if source_builder_timings_json.exists()
            else ""
        ),
        "command": shlex.join(command),
        "history_metric_name": str(
            payload.get("stop_metric_name", payload.get("final_metric_name", ""))
        ),
        "history_iterations": history_iterations,
        "history_metric": history_metric,
        "initial_guess_enabled": bool(initial_guess.get("enabled", False)),
        "initial_guess_success": bool(initial_guess.get("success", False)),
        "initial_guess_ksp_iterations": int(initial_guess.get("ksp_iterations", 0)),
        "native_run_info": "",
        "native_history_json": "",
    }
    return row


def _normalize_external_payload(
    *,
    case_id: str,
    impl: dict[str, str],
    ranks: int,
    exit_code: int,
    case_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    result_path: Path,
    command: list[str],
    maxit: int,
) -> dict[str, object]:
    normalize_kind = str(impl["normalize_kind"])
    if normalize_kind == "local_pmg":
        base = source_compare_tools._normalize_local_payload(
            case_id=str(case_id),
            mode="reference",
            ranks=int(ranks),
            exit_code=int(exit_code),
            fixed_maxit=int(maxit),
            reference_metric_name="relative_correction",
            case_dir=case_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            result_path=result_path,
            command=command,
        )
    elif normalize_kind == "source_pmg":
        base = source_compare_tools._normalize_source_payload(
            case_id=str(case_id),
            mode="reference",
            ranks=int(ranks),
            exit_code=int(exit_code),
            case_dir=case_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            result_path=result_path,
            command=command,
        )
    else:
        raise ValueError(f"Unsupported normalize kind: {normalize_kind}")

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
        "combo_label": _combo_label(
            str(impl["assembly_backend"]),
            str(impl["solver_backend"]),
        ),
        "ranks": int(ranks),
        "status": str(base.get("status", "failed")),
        "message": str(base.get("message", "")),
        "solver_success": bool(base.get("solver_success", False)),
        "exit_code": int(exit_code),
        "wall_time_s": float(base.get("wall_time_s", float("nan"))),
        "solve_time_s": float(base.get("solve_time_s", float("nan"))),
        "nit": int(base.get("nit", 0)),
        "linear_iterations_total": int(base.get("linear_iterations_total", 0)),
        "final_metric": float(base.get("final_metric", float("nan"))),
        "final_metric_name": str(base.get("final_metric_name", "")),
        "energy": float(base.get("energy", float("nan"))),
        "omega": float(base.get("omega", float("nan"))),
        "u_max": float(base.get("u_max", float("nan"))),
        "stdout_path": _repo_rel(stdout_path),
        "stderr_path": _repo_rel(stderr_path),
        "result_json": _repo_rel(result_path),
        "case_dir": _repo_rel(case_dir),
        "stage_jsonl": _repo_rel(stage_jsonl) if stage_jsonl.exists() else "",
        "source_builder_timings_json": (
            _repo_rel(source_builder_timings_json)
            if source_builder_timings_json.exists()
            else ""
        ),
        "command": shlex.join(command),
        "history_metric_name": str(base.get("history_metric_name", "")),
        "history_iterations": list(base.get("history_iterations", [])),
        "history_metric": list(base.get("history_metric", [])),
        "initial_guess_enabled": bool(base.get("initial_guess_enabled", False)),
        "initial_guess_success": bool(base.get("initial_guess_success", False)),
        "initial_guess_ksp_iterations": int(base.get("initial_guess_ksp_iterations", 0)),
        "native_run_info": str(base.get("native_run_info", "")),
        "native_history_json": str(base.get("native_history_json", "")),
    }


def _write_summary(
    *,
    summary_path: Path,
    rows_by_case: dict[str, dict[str, object]],
    ranks: list[int],
    stop_tol: float,
    maxit: int,
    source_root: Path,
    source_env_mode: str,
) -> None:
    impl_order = {str(item["name"]): idx for idx, item in enumerate(IMPLEMENTATIONS)}
    ordered_rows = sorted(
        rows_by_case.values(),
        key=lambda row: (
            list(ranks).index(int(row["ranks"])) if int(row["ranks"]) in ranks else 10**6,
            impl_order.get(str(row["implementation"]), 10**6),
        ),
    )
    payload = {
        "runner": RUNNER_NAME,
        "source_root": _repo_rel(source_root),
        "source_env_mode": str(source_env_mode),
        "out_dir": _repo_rel(summary_path.parent),
        "ranks": [int(v) for v in ranks],
        "stop_metric_name": "relative_correction",
        "stop_tol": float(stop_tol),
        "maxit": int(maxit),
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
        "row_keys": list(NORMALIZED_ROW_KEYS),
        "rows": ordered_rows,
    }
    _write_json(summary_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Plasticity3D scaling comparison with Hypre and PMG variants."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ranks", type=int, nargs="+", default=list(DEFAULT_RANKS))
    parser.add_argument("--stop-tol", type=float, default=2.0e-3)
    parser.add_argument("--maxit", type=int, default=80)
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=tuple(str(item["name"]) for item in IMPLEMENTATIONS),
        default=tuple(str(item["name"]) for item in IMPLEMENTATIONS),
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    summary_path = out_dir / SUMMARY_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    source_env_mode = mix_tools._ensure_shared_source_env(source_root)
    env = mix_tools._mixed_env(source_root)
    source_python, _resolved_source_mode = source_compare_tools.resolve_source_python(source_root)
    helper_path = source_compare_tools.ensure_source_helper(source_root)
    source_compare_tools._ensure_local_assets()

    rows_by_case: dict[str, dict[str, object]] = {}
    if bool(args.resume) and summary_path.exists():
        previous = _read_json(summary_path)
        for row in list(previous.get("rows", [])):
            if not isinstance(row, dict):
                continue
            case_id = str(row.get("case_id", "")).strip()
            if case_id:
                rows_by_case[case_id] = _upgrade_row(dict(row))

    selected_implementations = {str(v) for v in list(args.implementations)}
    for ranks in [int(v) for v in args.ranks]:
        for impl in IMPLEMENTATIONS:
            if str(impl["name"]) not in selected_implementations:
                continue
            implementation = str(impl["name"])
            case_id = f"np{int(ranks)}:{implementation}"
            case_dir = out_dir / "runs" / f"np{int(ranks)}" / implementation
            result_path = case_dir / "output.json"
            stdout_path = case_dir / "stdout.txt"
            stderr_path = case_dir / "stderr.txt"
            command = _build_command(
                impl=impl,
                source_root=source_root,
                source_python=source_python,
                helper_path=helper_path,
                case_dir=case_dir,
                result_path=result_path,
                ranks=int(ranks),
                stop_tol=float(args.stop_tol),
                maxit=int(args.maxit),
            )

            if bool(args.resume) and case_id in rows_by_case and result_path.exists():
                continue

            exit_code, wall_time_s = mix_tools._run_command(
                cmd=command,
                cwd=REPO_ROOT,
                env=env,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                result_path=result_path,
                output_grace_s=20.0,
            )
            if exit_code == 0 and result_path.exists():
                if str(impl["normalize_kind"]) == "mix_case":
                    row = _normalize_mix_payload(
                        case_id=case_id,
                        impl=impl,
                        ranks=int(ranks),
                        exit_code=int(exit_code),
                        case_dir=case_dir,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        result_path=result_path,
                        command=command,
                    )
                else:
                    row = _normalize_external_payload(
                        case_id=case_id,
                        impl=impl,
                        ranks=int(ranks),
                        exit_code=int(exit_code),
                        case_dir=case_dir,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        result_path=result_path,
                        command=command,
                        maxit=int(args.maxit),
                    )
                if not row["wall_time_s"] or not (row["wall_time_s"] > 0.0):
                    row["wall_time_s"] = float(wall_time_s)
            else:
                message = _tail_text(stderr_path) or _tail_text(stdout_path) or "Case failed"
                row = _failed_row(
                    case_id=case_id,
                    impl=impl,
                    ranks=int(ranks),
                    exit_code=int(exit_code),
                    message=message,
                    case_dir=case_dir,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    result_path=result_path,
                    command=command,
                )
            rows_by_case[case_id] = row
            _write_summary(
                summary_path=summary_path,
                rows_by_case=rows_by_case,
                ranks=[int(v) for v in args.ranks],
                stop_tol=float(args.stop_tol),
                maxit=int(args.maxit),
                source_root=source_root,
                source_env_mode=str(source_env_mode),
            )


if __name__ == "__main__":
    main()

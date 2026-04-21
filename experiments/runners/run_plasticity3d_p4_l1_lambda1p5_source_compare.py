#!/usr/bin/env python3
"""Run a source-vs-maintained Plasticity3D comparison for `P4(L1), lambda=1.5`."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import stat
import subprocess
from time import perf_counter


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
LOCAL_SOLVER = (
    REPO_ROOT
    / "src"
    / "problems"
    / "slope_stability_3d"
    / "jax_petsc"
    / "solve_slope_stability_3d_dof.py"
)
DEFAULT_SOURCE_ROOT = REPO_ROOT / "tmp" / "source_compare" / "slope_stability_petsc4py"
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "source_compare"
    / "plasticity3d_p4_l1_lambda1p5"
)
DEFAULT_RANKS = (1, 2, 4, 8, 16, 32)
SUMMARY_NAME = "comparison_summary.json"
RUNNER_NAME = "plasticity3d_p4_l1_lambda1p5_source_compare"

NORMALIZED_ROW_KEYS = (
    "case_id",
    "implementation",
    "mode",
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
    "command",
    "history_metric_name",
    "history_iterations",
    "history_metric",
    "initial_guess_enabled",
    "initial_guess_success",
    "initial_guess_ksp_iterations",
    "native_run_info",
    "native_npz",
    "native_history_json",
    "native_debug_bundle",
    "native_vtu",
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
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def ensure_source_helper(source_root: Path) -> Path:
    helper = source_root / "scripts_local" / "run_fixed_lambda_3d.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    wrapper = """#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

source_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(source_root / "src"))

repo_root = os.environ.get("FNE_REPO_ROOT")
if repo_root:
    sys.path.insert(0, str(Path(repo_root)))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.runners.source_fixed_lambda_3d_impl import main

if __name__ == "__main__":
    main()
"""
    current = helper.read_text(encoding="utf-8") if helper.exists() else None
    if current != wrapper:
        helper.write_text(wrapper, encoding="utf-8")
    helper.chmod(helper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return helper


def _source_env(source_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["BLIS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["FNE_REPO_ROOT"] = str(REPO_ROOT)
    source_path = str((source_root / "src").resolve())
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = source_path if not current else source_path + os.pathsep + current
    return env


def _local_env() -> dict[str, str]:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["BLIS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"
    return env


def _run_probe(*, cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _shared_env_import_probe(source_root: Path) -> subprocess.CompletedProcess[str]:
    return _run_probe(
        cmd=[
            str(PYTHON),
            "-c",
            "from slope_stability.cli.run_case_from_config import main; print('import_ok')",
        ],
        cwd=REPO_ROOT,
        env=_source_env(source_root),
    )


def _shared_env_kernel_probe(source_root: Path) -> subprocess.CompletedProcess[str]:
    return _run_probe(
        cmd=[
            str(PYTHON),
            "-c",
            (
                "import slope_stability._kernels as kernels; "
                "assert hasattr(kernels, 'assemble_overlap_strain_3d'); "
                "print(kernels.__file__)"
            ),
        ],
        cwd=REPO_ROOT,
        env=_source_env(source_root),
    )


def _build_source_extension_inplace(source_root: Path) -> subprocess.CompletedProcess[str]:
    return _run_probe(
        cmd=[str(PYTHON), "setup.py", "build_ext", "--inplace"],
        cwd=source_root,
        env=_source_env(source_root),
    )


def resolve_source_python(source_root: Path) -> tuple[Path, str]:
    smoke = _shared_env_import_probe(source_root)
    if smoke.returncode == 0:
        kernel_probe = _shared_env_kernel_probe(source_root)
        if kernel_probe.returncode == 0:
            return PYTHON, "shared_env"

        build_ext = _build_source_extension_inplace(source_root)
        if build_ext.returncode == 0:
            kernel_probe = _shared_env_kernel_probe(source_root)
            if kernel_probe.returncode == 0:
                return PYTHON, "shared_env_built_ext"

    bootstrap = subprocess.run(
        ["bash", "./bootstrap.sh"],
        cwd=source_root,
        env={**os.environ, "BOOTSTRAP_MODE": "wheel"},
        check=False,
        capture_output=True,
        text=True,
    )
    if bootstrap.returncode != 0:
        raise RuntimeError(
            "Source shared-env import failed and bootstrap fallback also failed.\n"
            f"Smoke stderr: {smoke.stderr}\n"
            f"Bootstrap stderr: {bootstrap.stderr}"
        )
    source_python = source_root / ".venv" / "bin" / "python"
    if not source_python.exists():
        raise FileNotFoundError(f"Expected bootstrapped source Python at {source_python}")
    return source_python, "bootstrap_wheel"


def _run_command(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_fh:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            check=False,
            stdout=stdout_fh,
            stderr=stderr_fh,
            text=True,
        )
    return int(proc.returncode), float(perf_counter() - t0)


def _nan_if_missing(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _is_finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _relative_grad_history(payload: dict) -> tuple[list[int], list[float]]:
    history = list(payload.get("history", []))
    if not history:
        return [], []
    initial = float(history[0].get("grad_norm", float("nan")))
    if not math.isfinite(initial) or abs(initial) <= 1.0e-30:
        return [], []
    return (
        [int(row.get("it", 0)) for row in history],
        [float(row.get("grad_norm", float("nan"))) / initial for row in history],
    )


def _local_step_history(payload: dict) -> tuple[list[int], list[float]]:
    history = list(payload.get("history", []))
    if not history:
        return [], []
    return (
        [int(row.get("it", 0)) for row in history],
        [float(row.get("step_rel", float("nan"))) for row in history],
    )


def _normalize_local_payload(
    *,
    case_id: str,
    mode: str,
    ranks: int,
    exit_code: int,
    fixed_maxit: int,
    reference_metric_name: str = "relative_grad_norm",
    case_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    result_path: Path,
    command: list[str],
) -> dict[str, object]:
    payload = _read_json(result_path)
    if str(mode) == "reference" and str(reference_metric_name) == "relative_correction":
        history_iterations, history_metric = _local_step_history(payload)
        final_metric_name = "relative_correction"
        final_metric = history_metric[-1] if history_metric else float("nan")
    else:
        history_iterations, history_metric = _relative_grad_history(payload)
        final_metric_name = "relative_grad_norm"
        final_metric = (
            history_metric[-1] if history_metric else _nan_if_missing(payload.get("final_grad_norm"))
        )
    nit = int(payload.get("nit", 0))
    initial_guess = dict(payload.get("initial_guess", {}))

    if mode == "fixed_work" and exit_code == 0 and nit >= int(fixed_maxit) and _is_finite(payload.get("energy")):
        status = "completed_fixed_work"
        solver_success = True
        message = f"Reached fixed Newton cap ({int(fixed_maxit)})"
    else:
        status = str(payload.get("status", "failed"))
        solver_success = bool(status == "completed")
        message = str(payload.get("message", ""))

    row = {
        "case_id": str(case_id),
        "implementation": "maintained_local",
        "mode": str(mode),
        "ranks": int(ranks),
        "status": str(status),
        "message": str(message),
        "solver_success": bool(solver_success),
        "exit_code": int(exit_code),
        "wall_time_s": _nan_if_missing(payload.get("total_time")),
        "solve_time_s": _nan_if_missing(payload.get("solve_time")),
        "nit": int(nit),
        "linear_iterations_total": int(payload.get("linear_iterations_total", 0)),
        "final_metric": float(final_metric),
        "final_metric_name": str(final_metric_name),
        "energy": _nan_if_missing(payload.get("energy")),
        "omega": _nan_if_missing(payload.get("omega")),
        "u_max": _nan_if_missing(payload.get("u_max")),
        "stdout_path": _repo_rel(stdout_path),
        "stderr_path": _repo_rel(stderr_path),
        "result_json": _repo_rel(result_path),
        "case_dir": _repo_rel(case_dir),
        "command": shlex.join(command),
        "history_metric_name": str(final_metric_name),
        "history_iterations": history_iterations,
        "history_metric": history_metric,
        "initial_guess_enabled": bool(initial_guess.get("enabled", False)),
        "initial_guess_success": bool(initial_guess.get("success", False)),
        "initial_guess_ksp_iterations": int(initial_guess.get("ksp_iterations", 0)),
        "native_run_info": "",
        "native_npz": "",
        "native_history_json": "",
        "native_debug_bundle": "",
        "native_vtu": "",
    }
    return row


def _normalize_source_payload(
    *,
    case_id: str,
    mode: str,
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
    history_iterations = [int(row.get("iteration", 0)) for row in history]
    history_metric = [float(row.get("metric", float("nan"))) for row in history]
    initial_guess = dict(payload.get("initial_guess", {}))

    row = {
        "case_id": str(case_id),
        "implementation": "source_petsc4py",
        "mode": str(mode),
        "ranks": int(ranks),
        "status": str(payload.get("status", "failed")),
        "message": str(payload.get("message", "")),
        "solver_success": bool(payload.get("solver_success", False)),
        "exit_code": int(exit_code),
        "wall_time_s": _nan_if_missing(payload.get("total_time")),
        "solve_time_s": _nan_if_missing(payload.get("solve_time")),
        "nit": int(payload.get("nit", 0)),
        "linear_iterations_total": int(payload.get("linear_iterations_total", 0)),
        "final_metric": _nan_if_missing(payload.get("final_metric")),
        "final_metric_name": str(payload.get("final_metric_name", "relative_residual")),
        "energy": _nan_if_missing(payload.get("energy")),
        "omega": _nan_if_missing(payload.get("omega")),
        "u_max": _nan_if_missing(payload.get("u_max")),
        "stdout_path": _repo_rel(stdout_path),
        "stderr_path": _repo_rel(stderr_path),
        "result_json": _repo_rel(result_path),
        "case_dir": _repo_rel(case_dir),
        "command": shlex.join(command),
        "history_metric_name": str(payload.get("history_metric_name", "relative_residual")),
        "history_iterations": history_iterations,
        "history_metric": history_metric,
        "initial_guess_enabled": bool(initial_guess.get("enabled", False)),
        "initial_guess_success": bool(initial_guess.get("success", False)),
        "initial_guess_ksp_iterations": int(initial_guess.get("ksp_iterations", 0)),
        "native_run_info": str(payload.get("native_run_info", "")),
        "native_npz": str(payload.get("native_npz", "")),
        "native_history_json": str(payload.get("native_history_json", "")),
        "native_debug_bundle": str(payload.get("native_debug_bundle", "")),
        "native_vtu": str(payload.get("native_vtu", "")),
    }
    return row


def _failed_row(
    *,
    case_id: str,
    implementation: str,
    mode: str,
    ranks: int,
    exit_code: int,
    message: str,
    case_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    result_path: Path,
    command: list[str],
) -> dict[str, object]:
    return {
        "case_id": str(case_id),
        "implementation": str(implementation),
        "mode": str(mode),
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
        "command": shlex.join(command),
        "history_metric_name": "",
        "history_iterations": [],
        "history_metric": [],
        "initial_guess_enabled": False,
        "initial_guess_success": False,
        "initial_guess_ksp_iterations": 0,
        "native_run_info": "",
        "native_npz": "",
        "native_history_json": "",
        "native_debug_bundle": "",
        "native_vtu": "",
    }


def _write_summary(
    *,
    summary_path: Path,
    rows_by_case: dict[str, dict[str, object]],
    ranks: list[int],
    reference_rank: int,
    fixed_maxit: int,
    source_root: Path,
    source_env_mode: str,
    include_fixed_work: bool,
    include_reference: bool,
    reference_stop_policy: str,
    reference_stop_tol: float,
    reference_maxit: int,
    source_pc_backend: str,
) -> None:
    mode_rank_order = {
        f"fixed_work:{rank}": idx for idx, rank in enumerate(ranks)
    }
    for idx, impl in enumerate(("source_petsc4py", "maintained_local")):
        mode_rank_order[f"fixed_work_impl:{impl}"] = idx
    ordered_rows = sorted(
        rows_by_case.values(),
        key=lambda row: (
            0 if str(row["mode"]) == "fixed_work" else 1,
            list(ranks).index(int(row["ranks"])) if int(row["ranks"]) in ranks else 10**6,
            0 if str(row["implementation"]) == "source_petsc4py" else 1,
        ),
    )
    payload = {
        "runner": RUNNER_NAME,
        "source_root": _repo_rel(source_root),
        "source_env_mode": str(source_env_mode),
        "out_dir": _repo_rel(summary_path.parent),
        "ranks": [int(v) for v in ranks],
        "reference_rank": int(reference_rank),
        "fixed_maxit": int(fixed_maxit),
        "include_fixed_work": bool(include_fixed_work),
        "include_reference": bool(include_reference),
        "reference_stop_policy": str(reference_stop_policy),
        "reference_stop_tol": float(reference_stop_tol),
        "reference_maxit": int(reference_maxit),
        "source_pc_backend": str(source_pc_backend),
        "row_keys": list(NORMALIZED_ROW_KEYS),
        "rows": ordered_rows,
    }
    _write_json(summary_path, payload)


def _build_local_command(
    *,
    ranks: int,
    result_path: Path,
    mode: str,
    fixed_maxit: int,
    reference_stop_policy: str = "legacy_relative",
    reference_stop_tol: float = 1.0e-2,
    reference_maxit: int = 100,
) -> list[str]:
    cmd = [
        "mpiexec",
        "-n",
        str(ranks),
        str(PYTHON),
        "-u",
        str(LOCAL_SOLVER),
        "--nproc",
        str(ranks),
        "--mesh_name",
        "hetero_ssr_L1",
        "--elem_degree",
        "4",
        "--lambda-target",
        "1.5",
        "--profile",
        "performance",
        "--pc_type",
        "mg",
        "--ksp_type",
        "fgmres",
        "--ksp_rtol",
        "1e-2",
        "--ksp_max_it",
        "100",
        "--distribution_strategy",
        "overlap_p2p",
        "--problem_build_mode",
        "rank_local",
        "--mg_level_build_mode",
        "rank_local",
        "--mg_transfer_build_mode",
        "owned_rows",
        "--element_reorder_mode",
        "block_xyz",
        "--mg_strategy",
        "same_mesh_p4_p2_p1",
        "--use_near_nullspace",
        "--elastic_initial_guess",
        "--line_search",
        "armijo",
        "--mg_p1_smoother_ksp_type",
        "chebyshev",
        "--mg_p1_smoother_pc_type",
        "jacobi",
        "--mg_p1_smoother_steps",
        "5",
        "--mg_p2_smoother_ksp_type",
        "chebyshev",
        "--mg_p2_smoother_pc_type",
        "jacobi",
        "--mg_p2_smoother_steps",
        "5",
        "--mg_p4_smoother_ksp_type",
        "chebyshev",
        "--mg_p4_smoother_pc_type",
        "jacobi",
        "--mg_p4_smoother_steps",
        "5",
        "--mg_coarse_backend",
        "hypre",
        "--mg_coarse_ksp_type",
        "cg",
        "--mg_coarse_pc_type",
        "hypre",
        "--mg_coarse_hypre_nodal_coarsen",
        "6",
        "--mg_coarse_hypre_vec_interp_variant",
        "3",
        "--mg_coarse_hypre_strong_threshold",
        "0.5",
        "--mg_coarse_hypre_coarsen_type",
        "HMIS",
        "--mg_coarse_hypre_max_iter",
        "2",
        "--mg_coarse_hypre_tol",
        "0.0",
        "--mg_coarse_hypre_relax_type_all",
        "symmetric-SOR/Jacobi",
        "--save_history",
        "--quiet",
        "--out",
        str(result_path),
    ]
    if mode == "fixed_work":
        cmd.extend(
            [
                "--maxit",
                str(fixed_maxit),
                "--tolf",
                "0",
                "--tolg",
                "0",
                "--tolg_rel",
                "0",
                "--tolx_rel",
                "0",
                "--tolx_abs",
                "0",
            ]
        )
    elif str(reference_stop_policy) == "matched_relative_correction":
        cmd.extend(
            [
                "--maxit",
                str(reference_maxit),
                "--tolf",
                "1e100",
                "--tolg",
                "1e100",
                "--tolg_rel",
                "0",
                "--tolx_rel",
                str(reference_stop_tol),
                "--tolx_abs",
                "0",
            ]
        )
    else:
        cmd.extend(
            [
                "--maxit",
                str(reference_maxit),
                "--tolf",
                "1e100",
                "--tolg",
                "0",
                "--tolg_rel",
                str(reference_stop_tol),
                "--tolx_rel",
                "1e100",
                "--tolx_abs",
                "1e100",
            ]
        )
    return cmd


def _build_source_command(
    *,
    source_root: Path,
    source_python: Path,
    helper_path: Path,
    case_dir: Path,
    result_path: Path,
    ranks: int,
    mode: str,
    fixed_maxit: int,
    reference_stop_policy: str = "legacy_relative",
    reference_stop_tol: float = 1.0e-2,
    reference_maxit: int = 100,
    source_pc_backend: str = "pmg_shell",
) -> list[str]:
    mesh_path = source_root / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh"
    cmd = [
        "mpiexec",
        "-n",
        str(ranks),
        str(source_python),
        "-u",
        str(helper_path),
        "--out-dir",
        str(case_dir),
        "--output-json",
        str(result_path),
        "--mesh-path",
        str(mesh_path),
        "--elem-type",
        "P4",
        "--lambda-target",
        "1.5",
        "--node-ordering",
        "block_xyz",
        "--solver-type",
        "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        "--pc-backend",
        str(source_pc_backend),
        "--linear-tolerance",
        "1e-2",
        "--linear-max-iter",
        "100",
        "--threads",
        "1",
        "--elastic-initial-guess",
        "--no-write-debug-bundle",
        "--write-history-json",
        "--no-write-solution-vtu",
        "--no-write-plots",
        "--quiet",
    ]
    if str(source_pc_backend) == "hypre":
        cmd.extend(
            [
                "--pc-hypre-coarsen-type",
                "HMIS",
                "--pc-hypre-interp-type",
                "ext+i",
                "--pc-hypre-boomeramg-max-iter",
                "1",
            ]
        )
    elif str(source_pc_backend) == "pmg":
        cmd.extend(
            [
                "--mg-coarse-ksp-type",
                "cg",
                "--mg-coarse-pc-type",
                "hypre",
                "--petsc-opt",
                "mg_levels_ksp_type=chebyshev",
                "--petsc-opt",
                "mg_levels_pc_type=jacobi",
                "--petsc-opt",
                "mg_levels_ksp_max_it=5",
            ]
        )
    if mode == "fixed_work":
        cmd.extend(
            [
                "--stopping-criterion",
                "relative_residual",
                "--it-newt-max",
                str(fixed_maxit),
                "--fixed-work-mode",
                "--stopping-tol",
                "0.0",
            ]
        )
    else:
        cmd.extend(
            [
                "--stopping-criterion",
                (
                    "relative_correction"
                    if str(reference_stop_policy) == "matched_relative_correction"
                    else "relative_residual"
                ),
                "--it-newt-max",
                str(reference_maxit),
                "--stopping-tol",
                str(reference_stop_tol),
            ]
        )
    return cmd


def _ensure_local_assets() -> None:
    from src.problems.slope_stability_3d.support import ensure_same_mesh_case_hdf5

    ensure_same_mesh_case_hdf5("hetero_ssr_L1", 4)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the source and maintained Plasticity3D P4(L1) lambda=1.5 solvers."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ranks", type=int, nargs="+", default=list(DEFAULT_RANKS))
    parser.add_argument("--reference-rank", type=int, default=16)
    parser.add_argument("--fixed-maxit", type=int, default=20)
    parser.add_argument(
        "--include-fixed-work",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reference-stop-policy",
        choices=("legacy_relative", "matched_relative_correction"),
        default="legacy_relative",
    )
    parser.add_argument("--reference-stop-tol", type=float, default=1.0e-2)
    parser.add_argument("--reference-maxit", type=int, default=100)
    parser.add_argument(
        "--source-pc-backend",
        choices=("hypre", "gamg", "bddc", "pmg", "pmg_shell"),
        default="pmg_shell",
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=("source_petsc4py", "maintained_local"),
        default=("source_petsc4py", "maintained_local"),
    )
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
    helper_path = ensure_source_helper(source_root)
    source_python, source_env_mode = resolve_source_python(source_root)
    _ensure_local_assets()

    existing_rows: dict[str, dict[str, object]] = {}
    if bool(args.resume) and summary_path.exists():
        payload = _read_json(summary_path)
        for row in payload.get("rows", []):
            existing_rows[str(row.get("case_id", ""))] = dict(row)

    selected_implementations = {str(v) for v in list(args.implementations)}
    cases: list[dict[str, object]] = []
    if bool(args.include_fixed_work):
        for ranks in list(args.ranks):
            for implementation in ("source_petsc4py", "maintained_local"):
                if implementation not in selected_implementations:
                    continue
                cases.append(
                    {
                        "case_id": f"fixed_work:{implementation}:np{int(ranks)}",
                        "implementation": implementation,
                        "mode": "fixed_work",
                        "ranks": int(ranks),
                    }
                )
    if bool(args.include_reference):
        for implementation in ("source_petsc4py", "maintained_local"):
            if implementation not in selected_implementations:
                continue
            cases.append(
                {
                    "case_id": f"reference:{implementation}:np{int(args.reference_rank)}",
                    "implementation": implementation,
                    "mode": "reference",
                    "ranks": int(args.reference_rank),
                }
            )

    for case in cases:
        case_id = str(case["case_id"])
        mode = str(case["mode"])
        implementation = str(case["implementation"])
        ranks = int(case["ranks"])
        case_dir = out_dir / "runs" / mode / implementation / f"np{ranks}"
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

        if implementation == "maintained_local":
            command = _build_local_command(
                ranks=ranks,
                result_path=result_path,
                mode=mode,
                fixed_maxit=int(args.fixed_maxit),
                reference_stop_policy=str(args.reference_stop_policy),
                reference_stop_tol=float(args.reference_stop_tol),
                reference_maxit=int(args.reference_maxit),
            )
            env = _local_env()
            run_cwd = REPO_ROOT
        else:
            command = _build_source_command(
                source_root=source_root,
                source_python=source_python,
                helper_path=helper_path,
                case_dir=case_dir,
                result_path=result_path,
                ranks=ranks,
                mode=mode,
                fixed_maxit=int(args.fixed_maxit),
                reference_stop_policy=str(args.reference_stop_policy),
                reference_stop_tol=float(args.reference_stop_tol),
                reference_maxit=int(args.reference_maxit),
                source_pc_backend=str(args.source_pc_backend),
            )
            env = _source_env(source_root)
            run_cwd = REPO_ROOT

        exit_code, _ = _run_command(
            cmd=command,
            cwd=run_cwd,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

        if exit_code != 0 or not result_path.exists():
            message = _tail_text(stderr_path) or _tail_text(stdout_path) or "subprocess failed"
            row = _failed_row(
                case_id=case_id,
                implementation=implementation,
                mode=mode,
                ranks=ranks,
                exit_code=exit_code,
                message=message,
                case_dir=case_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                result_path=result_path,
                command=command,
            )
        elif implementation == "maintained_local":
            row = _normalize_local_payload(
                case_id=case_id,
                mode=mode,
                ranks=ranks,
                exit_code=exit_code,
                fixed_maxit=int(args.fixed_maxit),
                reference_metric_name=(
                    "relative_correction"
                    if str(args.reference_stop_policy) == "matched_relative_correction"
                    else "relative_grad_norm"
                ),
                case_dir=case_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                result_path=result_path,
                command=command,
            )
        else:
            row = _normalize_source_payload(
                case_id=case_id,
                mode=mode,
                ranks=ranks,
                exit_code=exit_code,
                case_dir=case_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                result_path=result_path,
                command=command,
            )

        existing_rows[case_id] = row
        _write_summary(
            summary_path=summary_path,
            rows_by_case=existing_rows,
            ranks=[int(v) for v in args.ranks],
            reference_rank=int(args.reference_rank),
            fixed_maxit=int(args.fixed_maxit),
            source_root=source_root,
            source_env_mode=source_env_mode,
            include_fixed_work=bool(args.include_fixed_work),
            include_reference=bool(args.include_reference),
            reference_stop_policy=str(args.reference_stop_policy),
            reference_stop_tol=float(args.reference_stop_tol),
            reference_maxit=int(args.reference_maxit),
            source_pc_backend=str(args.source_pc_backend),
        )

    _write_summary(
        summary_path=summary_path,
        rows_by_case=existing_rows,
        ranks=[int(v) for v in args.ranks],
        reference_rank=int(args.reference_rank),
        fixed_maxit=int(args.fixed_maxit),
        source_root=source_root,
        source_env_mode=source_env_mode,
        include_fixed_work=bool(args.include_fixed_work),
        include_reference=bool(args.include_reference),
        reference_stop_policy=str(args.reference_stop_policy),
        reference_stop_tol=float(args.reference_stop_tol),
        reference_maxit=int(args.reference_maxit),
        source_pc_backend=str(args.source_pc_backend),
    )
    print(json.dumps({"summary": _repo_rel(summary_path)}, indent=2))


if __name__ == "__main__":
    main()

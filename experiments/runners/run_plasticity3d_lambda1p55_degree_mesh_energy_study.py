#!/usr/bin/env python3
"""Run the maintained Plasticity3D lambda=1.55 degree-vs-resolution energy study."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py

from src.core.benchmark.replication import command_text, now_iso, read_json, run_logged_command, write_json
from src.problems.slope_stability_3d.support.mesh import (
    DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT,
    ensure_same_mesh_case_hdf5,
    same_mesh_case_hdf5_path,
)

from experiments.runners import run_plasticity3d_backend_mix_compare as mix_tools


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
CASE_RUNNER = REPO_ROOT / "experiments" / "runners" / "run_plasticity3d_backend_mix_case.py"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "tmp" / "source_compare" / "slope_stability_petsc4py"
DEFAULT_RAW_ROOT = REPO_ROOT / "artifacts" / "raw_results" / "docs_showcase"
DEFAULT_STUDY_DIR = (
    REPO_ROOT / "artifacts" / "raw_results" / "plasticity3d_lambda1p55_degree_mesh_energy_study"
)
SUMMARY_JSON_NAME = "comparison_summary.json"
REPORT_MD_NAME = "REPORT.md"
CONSTRAINT_VARIANT = DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT


@dataclass(frozen=True)
class StudyCase:
    elem_degree: int
    mesh_name: str
    pmg_strategy: str
    artifact_dir: Path
    reuse_existing: bool

    @property
    def degree_label(self) -> str:
        return f"P{int(self.elem_degree)}"

    @property
    def mesh_alias(self) -> str:
        return str(self.mesh_name).replace("hetero_ssr_", "", 1).upper()

    @property
    def notes(self) -> str:
        action = "Reused" if self.reuse_existing else "Authoritative"
        return (
            f"{action} 32-rank maintained-local Plasticity3D "
            f"{self.degree_label}({self.mesh_alias}) lambda=1.55 degree-energy study run "
            f"with constraint_variant={CONSTRAINT_VARIANT}."
        )


def _artifact_dir(elem_degree: int, mesh_name: str) -> Path:
    mesh_alias_lower = str(mesh_name).replace("hetero_ssr_", "", 1).lower()
    return DEFAULT_RAW_ROOT / f"plasticity3d_p{int(elem_degree)}_{mesh_alias_lower}_lambda1p55_np32_grad1e2"


CASES: tuple[StudyCase, ...] = (
    StudyCase(1, "hetero_ssr_L1", "uniform_refined_p1_chain", _artifact_dir(1, "hetero_ssr_L1"), False),
    StudyCase(1, "hetero_ssr_L1_2", "uniform_refined_p1_chain", _artifact_dir(1, "hetero_ssr_L1_2"), False),
    StudyCase(1, "hetero_ssr_L1_2_3", "uniform_refined_p1_chain", _artifact_dir(1, "hetero_ssr_L1_2_3"), False),
    StudyCase(1, "hetero_ssr_L1_2_3_4", "uniform_refined_p1_chain", _artifact_dir(1, "hetero_ssr_L1_2_3_4"), False),
    StudyCase(2, "hetero_ssr_L1", "same_mesh_p2_p1", _artifact_dir(2, "hetero_ssr_L1"), False),
    StudyCase(2, "hetero_ssr_L1_2", "same_mesh_p2_p1", _artifact_dir(2, "hetero_ssr_L1_2"), False),
    StudyCase(2, "hetero_ssr_L1_2_3", "same_mesh_p2_p1", _artifact_dir(2, "hetero_ssr_L1_2_3"), False),
    StudyCase(4, "hetero_ssr_L1", "same_mesh_p4_p2_p1", _artifact_dir(4, "hetero_ssr_L1"), False),
    StudyCase(4, "hetero_ssr_L1_2", "same_mesh_p4_p2_p1", _artifact_dir(4, "hetero_ssr_L1_2"), False),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _read_free_dofs(mesh_name: str, degree: int) -> int:
    path = ensure_same_mesh_case_hdf5(
        str(mesh_name),
        int(degree),
        constraint_variant=CONSTRAINT_VARIANT,
    )
    with h5py.File(path, "r") as handle:
        return int(handle["freedofs"].shape[0])


def _build_command(*, case: StudyCase, source_root: Path) -> list[str]:
    return [
        "mpiexec",
        "-n",
        "32",
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
        str(case.artifact_dir),
        "--output-json",
        str(case.artifact_dir / "output.json"),
        "--state-out",
        str(case.artifact_dir / "state.npz"),
        "--mesh-name",
        str(case.mesh_name),
        "--elem-degree",
        str(case.elem_degree),
        "--constraint-variant",
        str(CONSTRAINT_VARIANT),
        "--lambda-target",
        "1.55",
        "--pmg-strategy",
        str(case.pmg_strategy),
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
        "elem_degree": int(output_payload.get("elem_degree", 0)),
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
    case: StudyCase,
    command: list[str],
    env: dict[str, str],
    output_payload: dict[str, object],
    reused: bool,
) -> Path:
    meta = {
        "timestamp_utc": now_iso(),
        "reused": bool(reused),
        "nprocs": 32,
        "mesh_name": str(case.mesh_name),
        "elem_degree": int(case.elem_degree),
        "constraint_variant": str(CONSTRAINT_VARIANT),
        "same_mesh_case_path": str(
            same_mesh_case_hdf5_path(
                str(case.mesh_name),
                int(case.elem_degree),
                CONSTRAINT_VARIANT,
            )
        ),
        "lambda_target": 1.55,
        "assembly_backend": "local_constitutiveAD",
        "solver_backend": "local_pmg",
        "pmg_strategy": str(case.pmg_strategy),
        "pmg_realized_levels": int(output_payload.get("pmg_realized_levels", 0) or 0),
        "pmg_pc_backend": str(output_payload.get("pmg_pc_backend", "")),
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
    path = case.artifact_dir / "showcase_meta.json"
    write_json(path, meta)
    return path


def _validate_payload(case: StudyCase, payload: dict[str, object]) -> None:
    if float(payload.get("lambda_target", float("nan"))) != 1.55:
        raise RuntimeError(f"{case.degree_label}({case.mesh_alias}) recorded an unexpected lambda_target")
    if str(payload.get("mesh_name", "")) != str(case.mesh_name):
        raise RuntimeError(f"{case.degree_label}({case.mesh_alias}) recorded an unexpected mesh_name")
    if int(payload.get("elem_degree", 0)) != int(case.elem_degree):
        raise RuntimeError(f"{case.degree_label}({case.mesh_alias}) recorded an unexpected elem_degree")
    if int(payload.get("ranks", 0)) != 32:
        raise RuntimeError(f"{case.degree_label}({case.mesh_alias}) did not record a 32-rank run")
    if int(payload.get("nit", 10**9)) > 200:
        raise RuntimeError(f"{case.degree_label}({case.mesh_alias}) exceeded the Newton cap")
    if str(payload.get("constraint_variant", "")) != str(CONSTRAINT_VARIANT):
        raise RuntimeError(
            f"{case.degree_label}({case.mesh_alias}) recorded an unexpected constraint_variant"
        )


def _summary_row(case: StudyCase, payload: dict[str, object], *, reused: bool, free_dofs_hdf5: int) -> dict[str, object]:
    parallel_setup = dict(payload.get("parallel_setup", {}))
    free_dofs_output = int(parallel_setup.get("owned_free_dofs_sum", 0) or 0)
    return {
        "degree_line": str(case.degree_label),
        "elem_degree": int(case.elem_degree),
        "mesh_name": str(case.mesh_name),
        "mesh_alias": str(case.mesh_alias),
        "constraint_variant": str(payload.get("constraint_variant", CONSTRAINT_VARIANT)),
        "same_mesh_case_path": str(payload.get("same_mesh_case_path", "")),
        "artifact_dir": str(case.artifact_dir.relative_to(REPO_ROOT)),
        "result_json": str((case.artifact_dir / "output.json").relative_to(REPO_ROOT)),
        "state_npz": str((case.artifact_dir / "state.npz").relative_to(REPO_ROOT)),
        "status": str(payload.get("status", "")),
        "message": str(payload.get("message", "")),
        "reused": bool(reused),
        "ranks": int(payload.get("ranks", 0)),
        "nit": int(payload.get("nit", 0)),
        "final_grad_norm": float(payload.get("final_grad_norm", float("nan"))),
        "linear_iterations_total": int(payload.get("linear_iterations_total", 0)),
        "solve_time_s": float(payload.get("solve_time", float("nan"))),
        "total_time_s": float(payload.get("total_time", float("nan"))),
        "energy": float(payload.get("energy", float("nan"))),
        "omega": float(payload.get("omega", float("nan"))),
        "u_max": float(payload.get("u_max", float("nan"))),
        "free_dofs": int(free_dofs_output or free_dofs_hdf5),
        "free_dofs_output": int(free_dofs_output),
        "free_dofs_hdf5": int(free_dofs_hdf5),
        "pmg_strategy": str(payload.get("pmg_strategy", case.pmg_strategy)),
        "pmg_realized_levels": int(payload.get("pmg_realized_levels", 0) or 0),
        "pmg_pc_backend": str(payload.get("pmg_pc_backend", "")),
    }


def _report_text(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Plasticity3D `lambda = 1.55` Degree-vs-Resolution Energy Study",
        "",
        "- maintained stack: `local_constitutiveAD + local_pmg + armijo`",
        f"- constraint variant: `{CONSTRAINT_VARIANT}`",
        "- stop: `grad_norm < 1e-2` or `maxit = 200`",
        "- time metric: end-to-end `total_time`",
        "- time-comparison ranks: `32` for every plotted point",
        "",
        "| Degree | Mesh | Free DOFs | Energy | Total [s] | Newton | Linear | Status | Reused | Artifact |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    ordered = sorted(rows, key=lambda row: (int(str(row["degree_line"]).replace("P", "")), int(row["free_dofs"])))
    for row in ordered:
        lines.append(
            "| {degree_line} | `{mesh_alias}` | `{free_dofs}` | `{energy:.6f}` | `{total_time_s:.3f}` | `{nit}` | `{linear_iterations_total}` | `{status}` | `{reused}` | [{mesh_alias}](/home/michal/repos/fenics_nonlinear_energies/{artifact_dir}) |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def _write_summary(study_dir: Path, rows: list[dict[str, object]]) -> Path:
    ordered = sorted(rows, key=lambda row: (int(str(row["degree_line"]).replace("P", "")), int(row["free_dofs"])))
    payload = {
        "benchmark": "Plasticity3D lambda=1.55 degree-vs-resolution energy study",
        "assembly_backend": "local_constitutiveAD",
        "solver_backend": "local_pmg",
        "constraint_variant": str(CONSTRAINT_VARIANT),
        "line_search": "armijo",
        "lambda_target": 1.55,
        "stop_metric_name": "grad_norm",
        "grad_stop_tol": 1.0e-2,
        "maxit": 200,
        "ksp_rtol": 1.0e-1,
        "ksp_max_it": 100,
        "ranks": 32,
        "time_metric": "total_time_s",
        "rows": ordered,
    }
    summary_path = study_dir / SUMMARY_JSON_NAME
    write_json(summary_path, payload)
    (study_dir / REPORT_MD_NAME).write_text(_report_text(ordered), encoding="utf-8")
    return summary_path


def _run_case(*, case: StudyCase, source_root: Path, resume: bool, free_dofs_hdf5: int) -> dict[str, object]:
    output_json = case.artifact_dir / "output.json"
    state_npz = case.artifact_dir / "state.npz"
    progress_json = case.artifact_dir / "progress.json"
    env = mix_tools._mixed_env(source_root)
    command = _build_command(case=case, source_root=source_root)

    if not case.reuse_existing:
        run_logged_command(
            command=command,
            cwd=REPO_ROOT,
            leaf_dir=case.artifact_dir,
            expected_outputs=[output_json, state_npz],
            env=env,
            resume=bool(resume),
            notes=case.notes,
        )

    payload = read_json(output_json)
    payload["state_out"] = str(state_npz.resolve())
    write_json(output_json, payload)
    _validate_payload(case, payload)
    _write_progress_json(output_payload=payload, out_path=progress_json)
    _write_showcase_meta(
        case=case,
        command=command,
        env=env,
        output_payload=payload,
        reused=bool(case.reuse_existing),
    )
    return _summary_row(case, payload, reused=bool(case.reuse_existing), free_dofs_hdf5=free_dofs_hdf5)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    study_dir = Path(args.study_dir).resolve()
    study_dir.mkdir(parents=True, exist_ok=True)

    case_order: list[tuple[int, int, StudyCase]] = []
    free_dofs_by_case: dict[tuple[int, str], int] = {}
    for case in CASES:
        free_dofs = _read_free_dofs(case.mesh_name, case.elem_degree)
        free_dofs_by_case[(int(case.elem_degree), str(case.mesh_name))] = int(free_dofs)
        case_order.append((int(free_dofs), int(case.elem_degree), case))
    case_order.sort(key=lambda item: (item[0], item[1]))

    rows: list[dict[str, object]] = []
    for _free_dofs, _degree, case in case_order:
        row = _run_case(
            case=case,
            source_root=source_root,
            resume=bool(args.resume),
            free_dofs_hdf5=int(free_dofs_by_case[(int(case.elem_degree), str(case.mesh_name))]),
        )
        rows.append(row)
        _write_summary(study_dir, rows)

    print(
        json.dumps(
            {
                "study_dir": str(study_dir),
                "summary_json": str(study_dir / SUMMARY_JSON_NAME),
                "report_md": str(study_dir / REPORT_MD_NAME),
                "rows_completed": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

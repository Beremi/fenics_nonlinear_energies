from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from importlib.machinery import SourcelessFileLoader

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
ARCTAN_REPORT = "experiments/analysis/generate_plaplace_up_arctan_report.py"
ARCTAN_PAGE = "experiments/analysis/generate_plaplace_up_arctan_problem_page.py"


def _load_arctan_suite():
    module_name = "experiments.runners.run_plaplace_up_arctan_suite"
    pycache_dir = REPO_ROOT / "experiments" / "runners" / "__pycache__"
    for pyc_path in sorted(pycache_dir.glob("run_plaplace_up_arctan_suite.cpython-*.pyc")):
        spec = importlib.util.spec_from_file_location(
            module_name,
            pyc_path,
            loader=SourcelessFileLoader(module_name, str(pyc_path)),
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    raise ModuleNotFoundError(module_name)


arctan_suite = _load_arctan_suite()


def _write_state(path: Path, *, scale: float = 1.0) -> None:
    coords = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    u = scale * np.asarray([0.0, 0.5, 0.5, 1.0], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, coords=coords, triangles=triangles, u=u)


def _write_fake_summary(tmp_path: Path) -> Path:
    state_dir = tmp_path / "states"
    _write_state(state_dir / "state_raw.npz", scale=1.0)
    _write_state(state_dir / "state_certified.npz", scale=0.75)
    _write_state(state_dir / "state_eigen.npz", scale=0.5)

    lambda_cache_dir = tmp_path / "lambda_cache"
    lambda_cache_dir.mkdir(parents=True, exist_ok=True)
    for level, value in zip((4, 5, 6, 7), (64.1, 63.2, 62.9, 62.8)):
        payload = {
            "status": "completed",
            "message": "ok",
            "level": level,
            "p": 3.0,
            "geometry": "square_unit",
            "lambda1": value,
            "lambda_level": level,
            "quotient": value,
            "residual_norm": 1.0e-6 * level,
            "normalization_error": 1.0e-12,
            "outer_iterations": 10 + level,
            "direction_solves": 10 + level,
            "history": [{"outer_it": 1, "quotient": value, "stop_measure": 1.0e-3, "accepted": True, "alpha": 1.0, "halves": 0}],
            "eigenfunction_free": [0.1, 0.2, 0.1, 0.2],
            "eigenfunction_min": 0.0,
            "eigenfunction_max": 1.0,
            "state_out": str(state_dir / "state_eigen.npz"),
        }
        (lambda_cache_dir / f"lambda_p3_l{level}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    for p in (2.0, 3.0):
        for method in ("mpa", "rmpa"):
            for level in (4, 5, 6):
                raw_history = [
                    {"outer_it": 1, "J": -0.6, "stop_measure": 2.0e-1, "dual_residual_norm": 3.0e-1, "gradient_residual_norm": 4.0e-1, "accepted": True},
                    {"outer_it": 2, "J": -0.5, "stop_measure": 1.0e-1, "dual_residual_norm": 2.0e-1, "gradient_residual_norm": 3.0e-1, "accepted": True},
                ]
                certified_history = [
                    {"outer_it": 1, "J": -0.35, "stop_measure": 8.0e-2, "dual_residual_norm": 1.5e-2, "gradient_residual_norm": 2.0e-2, "accepted": True},
                    {"outer_it": 2, "J": -0.33, "stop_measure": 5.0e-2, "dual_residual_norm": 8.0e-3, "gradient_residual_norm": 1.2e-2, "accepted": True},
                ]
                rows.append(
                    {
                        "study": "mesh_refinement",
                        "method": method,
                        "p": p,
                        "level": level,
                        "epsilon": 1.0e-5,
                        "status": "completed",
                        "message": "ok",
                        "lambda1": 2.0 * np.pi**2 if p == 2.0 else (63.0 - 0.1 * level),
                        "lambda_level": level,
                        "raw_status": "maxit" if method == "mpa" else "failed",
                        "raw_message": "diagnostic raw run",
                        "raw_J": (-1.0 if method == "rmpa" else 1.0) * (0.01 * level + 0.001 * p),
                        "raw_residual_norm": 1.0e-2 * level,
                        "raw_gradient_residual_norm": 2.0e-2 * level,
                        "raw_outer_iterations": 5 * level if method == "rmpa" else 7 * level,
                        "raw_accepted_step_count": 4 * level if method == "rmpa" else 6 * level,
                        "raw_reference_error_w1p": 1.0e-3 / level,
                        "raw_solve_time_s": 0.5 * level,
                        "raw_state_path": str(state_dir / "state_raw.npz"),
                        "raw_reported_iterate_source": "best_dual_residual",
                        "raw_start_seed_name": "bubble",
                        "raw_history": raw_history,
                        "certified_status": "completed",
                        "certified_message": "certified polish",
                        "certified_J": -0.5 * (0.01 * level + 0.001 * p),
                        "certified_residual_norm": 1.0e-6 * level,
                        "certified_gradient_residual_norm": 2.0e-6 * level,
                        "certified_outer_iterations": 8 * level if method == "rmpa" else 9 * level,
                        "certified_accepted_step_count": 7 * level if method == "rmpa" else 8 * level,
                        "certified_reference_error_w1p": 5.0e-4 / level,
                        "certified_solve_time_s": 0.75 * level,
                        "certified_state_path": str(state_dir / "state_certified.npz"),
                        "certified_reported_iterate_source": "newton_polish",
                        "certified_start_seed_name": "bubble",
                        "certified_history": certified_history,
                        "status": "completed",
                        "message": "certified polish",
                        "J": -0.5 * (0.01 * level + 0.001 * p),
                        "residual_norm": 1.0e-6 * level,
                        "gradient_residual_norm": 2.0e-6 * level,
                        "outer_iterations": 8 * level if method == "rmpa" else 9 * level,
                        "accepted_step_count": 7 * level if method == "rmpa" else 8 * level,
                        "reference_error_w1p": 5.0e-4 / level,
                        "solve_time_s": 0.75 * level,
                        "state_path": str(state_dir / "state_certified.npz"),
                        "reported_iterate_source": "newton_polish",
                        "start_seed_name": "bubble",
                        "history": certified_history,
                        "result_path": str(tmp_path / "results" / f"{method}_p{int(p)}_l{level}.json"),
                        "configured_maxit": 80 if method == "rmpa" else 120,
                    }
                )
            for epsilon in (1.0e-4, 1.0e-5, 1.0e-6):
                raw_history = [
                    {"outer_it": 1, "J": -0.55, "stop_measure": 1.0e-1, "dual_residual_norm": 1.5e-1, "gradient_residual_norm": 2.0e-1, "accepted": True},
                    {"outer_it": 2, "J": -0.45, "stop_measure": epsilon, "dual_residual_norm": 1.0e-1, "gradient_residual_norm": 1.5e-1, "accepted": True},
                ]
                certified_history = [
                    {"outer_it": 1, "J": -0.32, "stop_measure": 5.0e-2, "dual_residual_norm": epsilon / 2.0, "gradient_residual_norm": epsilon, "accepted": True},
                    {"outer_it": 2, "J": -0.31, "stop_measure": epsilon / 2.0, "dual_residual_norm": epsilon / 4.0, "gradient_residual_norm": epsilon / 2.0, "accepted": True},
                ]
                rows.append(
                    {
                        "study": "tolerance_sweep",
                        "method": method,
                        "p": p,
                        "level": 6,
                        "epsilon": epsilon,
                        "raw_status": "maxit" if method == "mpa" else "failed",
                        "raw_message": "diagnostic raw run",
                        "raw_lambda1": 2.0 * np.pi**2 if p == 2.0 else 62.9,
                        "raw_J": (-1.0 if method == "rmpa" else 1.0) * (0.02 + epsilon),
                        "raw_residual_norm": 2.0e-2,
                        "raw_gradient_residual_norm": 3.0e-2,
                        "raw_outer_iterations": 11 if method == "rmpa" else 13,
                        "raw_accepted_step_count": 10 if method == "rmpa" else 12,
                        "raw_reference_error_w1p": 2.0e-4,
                        "raw_solve_time_s": 1.25,
                        "raw_state_path": str(state_dir / "state_raw.npz"),
                        "raw_reported_iterate_source": "best_dual_residual",
                        "raw_start_seed_name": "bubble",
                        "raw_history": raw_history,
                        "certified_status": "completed",
                        "certified_message": "certified polish",
                        "certified_lambda1": 2.0 * np.pi**2 if p == 2.0 else 62.9,
                        "certified_J": (-0.5 if method == "rmpa" else 0.5) * (0.02 + epsilon),
                        "certified_residual_norm": epsilon / 2.0,
                        "certified_gradient_residual_norm": epsilon,
                        "certified_outer_iterations": 6 if method == "rmpa" else 8,
                        "certified_accepted_step_count": 6 if method == "rmpa" else 7,
                        "certified_reference_error_w1p": 1.0e-4,
                        "certified_solve_time_s": 1.75,
                        "certified_state_path": str(state_dir / "state_certified.npz"),
                        "certified_reported_iterate_source": "newton_polish",
                        "certified_start_seed_name": "bubble",
                        "certified_history": certified_history,
                        "status": "completed",
                        "message": "certified polish",
                        "lambda1": 2.0 * np.pi**2 if p == 2.0 else 62.9,
                        "lambda_level": 6,
                        "J": (-0.5 if method == "rmpa" else 0.5) * (0.02 + epsilon),
                        "residual_norm": epsilon / 2.0,
                        "gradient_residual_norm": epsilon,
                        "outer_iterations": 6 if method == "rmpa" else 8,
                        "accepted_step_count": 6 if method == "rmpa" else 7,
                        "reference_error_w1p": 1.0e-4,
                        "solve_time_s": 1.75,
                        "state_path": str(state_dir / "state_certified.npz"),
                        "reported_iterate_source": "newton_polish",
                        "start_seed_name": "bubble",
                        "history": certified_history,
                        "result_path": str(tmp_path / "results" / f"{method}_p{int(p)}_eps{epsilon}.json"),
                        "configured_maxit": 80 if method == "rmpa" else 120,
                    }
                )

    summary = {
        "suite": "plaplace_up_arctan_full",
        "geometry": "square_unit",
        "reference_level": 7,
        "rows": rows,
        "lambda_cache_dir": str(lambda_cache_dir),
        "generated_case_count": len(rows),
        "status_counts": {"completed": len(rows)},
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _write_fake_petsc_summary(tmp_path: Path) -> Path:
    rows = [
        {
            "study": "mesh_ladder",
            "p": 2.0,
            "level": 8,
            "nprocs": 1,
            "status": "completed",
            "residual_norm": 4.0e-11,
            "setup_time_s": 2.3,
            "solve_time_s": 1.2,
            "total_time_s": 3.6,
            "linear_iterations_total": 36,
            "free_dofs": 65025,
            "lambda_source": "exact",
        },
        {
            "study": "mesh_ladder",
            "p": 2.0,
            "level": 9,
            "nprocs": 1,
            "status": "completed",
            "residual_norm": 3.2e-10,
            "setup_time_s": 7.9,
            "solve_time_s": 2.3,
            "total_time_s": 10.4,
            "linear_iterations_total": 44,
            "free_dofs": 261121,
            "lambda_source": "exact",
        },
        {
            "study": "mesh_ladder",
            "p": 3.0,
            "level": 8,
            "nprocs": 1,
            "status": "completed",
            "residual_norm": 8.6e-10,
            "setup_time_s": 2.3,
            "solve_time_s": 23.9,
            "total_time_s": 26.3,
            "linear_iterations_total": 571,
            "free_dofs": 65025,
            "lambda_source": "frozen_l7_reference",
        },
        {
            "study": "strong_scaling",
            "p": 2.0,
            "level": 9,
            "nprocs": 1,
            "status": "completed",
            "residual_norm": 3.2e-10,
            "setup_time_s": 7.9,
            "solve_time_s": 2.3,
            "total_time_s": 10.4,
            "speedup_total": 1.0,
            "efficiency_total": 1.0,
            "linear_iterations_total": 44,
        },
        {
            "study": "strong_scaling",
            "p": 2.0,
            "level": 9,
            "nprocs": 2,
            "status": "completed",
            "residual_norm": 3.3e-10,
            "setup_time_s": 4.7,
            "solve_time_s": 1.5,
            "total_time_s": 6.2,
            "speedup_total": 1.677,
            "efficiency_total": 0.838,
            "linear_iterations_total": 46,
        },
    ]
    summary = {
        "study": "plaplace_up_arctan_petsc",
        "solver": "jax_petsc",
        "pmg_config": {
            "pc_type": "mg",
            "mg_smoother_ksp_type": "chebyshev",
            "mg_smoother_pc_type": "jacobi",
        },
        "finest_scaling_level": 9,
        "rows": rows,
        "generated_case_count": len(rows),
        "status_counts": {"completed": len(rows)},
    }
    summary_path = tmp_path / "petsc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def test_row_from_result_contract_is_stable() -> None:
    case = arctan_suite.Case("mesh_refinement", "rmpa", 3.0, 6, 1.0e-5)
    row = arctan_suite._row_from_result(
        case,
        {
            "raw": {
                "status": "failed",
                "message": "raw failed",
                "lambda1": 62.8,
                "lambda_level": 6,
                "J": -0.01,
                "residual_norm": 1.0e-4,
                "gradient_residual_norm": 2.0e-4,
                "best_residual_norm": 1.0e-4,
                "best_gradient_residual_norm": 2.0e-4,
                "best_residual_outer_it": 7,
                "reported_iterate_source": "best_dual_residual",
                "outer_iterations": 12,
                "accepted_step_count": 10,
                "reference_error_w1p": 1.0e-3,
                "state_out": "/tmp/raw_state.npz",
                "history": [
                    {"outer_it": 1, "J": -0.1, "stop_measure": 1.0e-2, "dual_residual_norm": 5.0e-4, "gradient_residual_norm": 9.0e-4}
                ],
                "configured_maxit": 240,
                "direction_model": "thesis_dvh_aux_laplace",
            },
            "certified": {
                "status": "completed",
                "message": "ok",
                "lambda1": 62.8,
                "lambda_level": 6,
                "J": -0.02,
                "residual_norm": 1.0e-9,
                "gradient_residual_norm": 2.0e-9,
                "best_residual_norm": 1.0e-9,
                "best_gradient_residual_norm": 2.0e-9,
                "best_residual_outer_it": 3,
                "reported_iterate_source": "final",
                "outer_iterations": 4,
                "accepted_step_count": 4,
                "reference_error_w1p": 1.0e-5,
                "state_out": "/tmp/certified_state.npz",
                "history": [
                    {"outer_it": 1, "J": -0.01, "stop_measure": 1.0e-5, "dual_residual_norm": 5.0e-6, "gradient_residual_norm": 9.0e-6}
                ],
                "configured_maxit": 80,
                "direction_model": "jax_autodiff_stationary_newton",
                "handoff_source": "rmpa:best_dual_residual",
                "certified_newton_iters": 4,
            },
            "solution_track": "certified",
            "start_seed_name": "bubble",
            "solve_time_s": 1.2,
            "result_path": "/tmp/output.json",
            "ray_audit": {"best_kind": "maximum", "stable_interior_extremum": True},
            "certification_message": "",
        },
    )
    assert {"raw_status", "certified_status", "solution_track", "raw_history", "certified_history"} <= set(row)
    assert row["status"] == "completed"
    assert row["raw_status"] == "failed"
    assert row["certified_status"] == "completed"
    assert row["solution_track"] == "certified"
    assert row["history"]


def test_lambda_payload_reuses_completed_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = tmp_path / "lambda_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_path = tmp_path / "state.npz"
    _write_state(state_path)
    payload = {
        "status": "completed",
        "state_out": str(state_path),
        "lambda1": 62.8,
        "lambda_level": 4,
    }
    cache_path = cache_dir / "lambda_p3_l4.json"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(arctan_suite, "DEFAULT_OUT_DIR", tmp_path)

    def _boom(**kwargs):
        raise AssertionError("cache should have been reused")

    monkeypatch.setattr(arctan_suite, "compute_lambda1_cached", _boom)
    reused = arctan_suite._lambda_payload(tmp_path, 4)
    assert reused["lambda1"] == 62.8


def test_build_cases_has_expected_publish_surface() -> None:
    cases = arctan_suite.build_cases()
    assert len(cases) == 24
    assert {case.study for case in cases} == {"mesh_refinement", "tolerance_sweep"}
    assert {case.method for case in cases} == {"mpa", "rmpa"}
    assert {case.p for case in cases} == {2.0, 3.0}


def test_report_and_problem_page_generators_smoke(tmp_path: Path) -> None:
    summary_path = _write_fake_summary(tmp_path)
    petsc_summary_path = _write_fake_petsc_summary(tmp_path)
    report_path = tmp_path / "report" / "README.md"
    doc_path = tmp_path / "docs" / "pLaplace_up_arctan.md"
    asset_dir = tmp_path / "assets"

    subprocess.run(
        [
            str(PYTHON),
            "-u",
            ARCTAN_REPORT,
            "--summary",
            str(summary_path),
            "--petsc-summary",
            str(petsc_summary_path),
            "--out",
            str(report_path),
            "--asset-dir",
            str(asset_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report_text = report_path.read_text(encoding="utf-8")
    assert "# pLaplace_up_arctan Report" in report_text
    assert "## Raw Versus Certified Diagnostics" in report_text
    assert "## p = 2 Validation Study" in report_text
    assert "## p = 3 Eigenvalue Stage" in report_text
    assert "## p = 3 Main Study" in report_text
    assert "## JAX + PETSc Backend" in report_text
    assert "## PETSc Timing And Scaling" in report_text
    assert "nonlinear its" in report_text
    assert "MPA iters" in report_text
    assert "Newton iters" in report_text
    assert "linear its" in report_text
    assert "Chebyshev" in report_text
    assert "Jacobi" in report_text
    assert "## Cross-Method Diagnostics" in report_text
    assert (asset_dir / "p2_solution_panel.png").exists()
    assert (asset_dir / "p3_solution_panel.png").exists()
    assert (asset_dir / "lambda_convergence.png").exists()
    assert (asset_dir / "iteration_counts.png").exists()
    assert (asset_dir / "petsc_mesh_timing.png").exists()
    assert (asset_dir / "petsc_strong_scaling.png").exists()

    subprocess.run(
        [
            str(PYTHON),
            "-u",
            ARCTAN_PAGE,
            "--summary",
            str(summary_path),
            "--petsc-summary",
            str(petsc_summary_path),
            "--out",
            str(doc_path),
            "--asset-dir",
            str(asset_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    doc_text = doc_path.read_text(encoding="utf-8")
    assert "## Mathematical Specification" in doc_text
    assert "### p = 2 Validation Problem" in doc_text
    assert "### p = 3 Main Problem" in doc_text
    assert "## Solvability And Proof Notes" in doc_text
    assert "This page therefore claims existence/solvability" in doc_text
    assert "## p = 2 Validation Study" in doc_text
    assert "## p = 3 Eigenvalue Stage" in doc_text
    assert "## p = 3 Main Study" in doc_text
    assert "## JAX + PETSc Backend" in doc_text
    assert "## PETSc Timing And Scaling" in doc_text
    assert "nonlinear its" in doc_text
    assert "MPA iters" in doc_text
    assert "Newton iters" in doc_text
    assert "linear its" in doc_text
    assert "Chebyshev" in doc_text
    assert "Jacobi" in doc_text
    assert "## Raw Versus Certified Diagnostics" in doc_text
    assert "## Cross-Method Comparison" in doc_text
    assert "MPA" in doc_text
    assert "RMPA" in doc_text
    assert "docs/results/pLaplace_up_arctan.md" in doc_text

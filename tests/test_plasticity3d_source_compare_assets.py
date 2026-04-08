from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_source_compare_asset_generator_writes_report_and_plots(tmp_path: Path) -> None:
    summary_path = tmp_path / "comparison_summary.json"
    out_dir = tmp_path / "assets"

    source_result = tmp_path / "source_output.json"
    source_result.write_text(json.dumps({"message": "Converged"}), encoding="utf-8")
    local_result = tmp_path / "local_output.json"
    local_result.write_text(json.dumps({"message": "Gradient norm converged"}), encoding="utf-8")

    rows = [
        {
            "case_id": "fixed_work:source_petsc4py:np1",
            "implementation": "source_petsc4py",
            "mode": "fixed_work",
            "ranks": 1,
            "status": "completed_fixed_work",
            "message": "Reached fixed Newton cap (20)",
            "solver_success": True,
            "exit_code": 0,
            "wall_time_s": 12.0,
            "solve_time_s": 10.0,
            "nit": 20,
            "linear_iterations_total": 80,
            "final_metric": 2.0e-1,
            "final_metric_name": "relative_residual",
            "energy": -5.0,
            "omega": 2.0,
            "u_max": 0.3,
            "stdout_path": "",
            "stderr_path": "",
            "result_json": str(source_result),
            "case_dir": "",
            "command": "",
            "history_metric_name": "relative_residual",
            "history_iterations": [1, 20],
            "history_metric": [1.0, 2.0e-1],
            "initial_guess_enabled": True,
            "initial_guess_success": True,
            "initial_guess_ksp_iterations": 10,
            "native_run_info": "",
            "native_npz": "",
            "native_history_json": "",
            "native_debug_bundle": "",
            "native_vtu": "",
        },
        {
            "case_id": "fixed_work:maintained_local:np1",
            "implementation": "maintained_local",
            "mode": "fixed_work",
            "ranks": 1,
            "status": "completed_fixed_work",
            "message": "Reached fixed Newton cap (20)",
            "solver_success": True,
            "exit_code": 0,
            "wall_time_s": 15.0,
            "solve_time_s": 13.0,
            "nit": 20,
            "linear_iterations_total": 120,
            "final_metric": 5.0e-2,
            "final_metric_name": "relative_grad_norm",
            "energy": -4.0,
            "omega": 1.9,
            "u_max": 0.28,
            "stdout_path": "",
            "stderr_path": "",
            "result_json": str(local_result),
            "case_dir": "",
            "command": "",
            "history_metric_name": "relative_grad_norm",
            "history_iterations": [1, 20],
            "history_metric": [1.0, 5.0e-2],
            "initial_guess_enabled": True,
            "initial_guess_success": True,
            "initial_guess_ksp_iterations": 12,
            "native_run_info": "",
            "native_npz": "",
            "native_history_json": "",
            "native_debug_bundle": "",
            "native_vtu": "",
        },
        {
            "case_id": "reference:source_petsc4py:np16",
            "implementation": "source_petsc4py",
            "mode": "reference",
            "ranks": 16,
            "status": "completed",
            "message": "Converged",
            "solver_success": True,
            "exit_code": 0,
            "wall_time_s": 8.0,
            "solve_time_s": 7.0,
            "nit": 8,
            "linear_iterations_total": 40,
            "final_metric": 1.0e-2,
            "final_metric_name": "relative_residual",
            "energy": -5.1,
            "omega": 2.1,
            "u_max": 0.31,
            "stdout_path": "",
            "stderr_path": "",
            "result_json": str(source_result),
            "case_dir": "",
            "command": "",
            "history_metric_name": "relative_residual",
            "history_iterations": [1, 8],
            "history_metric": [1.0, 1.0e-2],
            "initial_guess_enabled": True,
            "initial_guess_success": True,
            "initial_guess_ksp_iterations": 10,
            "native_run_info": "",
            "native_npz": "",
            "native_history_json": "",
            "native_debug_bundle": "",
            "native_vtu": "",
        },
        {
            "case_id": "reference:maintained_local:np16",
            "implementation": "maintained_local",
            "mode": "reference",
            "ranks": 16,
            "status": "completed",
            "message": "Converged (energy, step, gradient)",
            "solver_success": True,
            "exit_code": 0,
            "wall_time_s": 9.5,
            "solve_time_s": 8.2,
            "nit": 9,
            "linear_iterations_total": 55,
            "final_metric": 8.0e-3,
            "final_metric_name": "relative_grad_norm",
            "energy": -4.8,
            "omega": 2.0,
            "u_max": 0.29,
            "stdout_path": "",
            "stderr_path": "",
            "result_json": str(local_result),
            "case_dir": "",
            "command": "",
            "history_metric_name": "relative_grad_norm",
            "history_iterations": [1, 9],
            "history_metric": [1.0, 8.0e-3],
            "initial_guess_enabled": True,
            "initial_guess_success": True,
            "initial_guess_ksp_iterations": 12,
            "native_run_info": "",
            "native_npz": "",
            "native_history_json": "",
            "native_debug_bundle": "",
            "native_vtu": "",
        },
    ]

    summary_path.write_text(
        json.dumps(
            {
                "runner": "plasticity3d_p4_l1_lambda1p5_source_compare",
                "source_env_mode": "shared_env",
                "ranks": [1],
                "reference_rank": 16,
                "fixed_maxit": 20,
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "experiments/analysis/generate_plasticity3d_p4_l1_lambda1p5_source_compare_assets.py",
            "--summary-json",
            str(summary_path),
            "--out-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (out_dir / "REPORT.md").exists()
    assert (out_dir / "fixed_work_times.png").exists()
    assert (out_dir / "fixed_work_iterations.png").exists()
    assert (out_dir / "reference_rank_convergence.png").exists()

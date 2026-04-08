from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_backend_mix_asset_generator_writes_report_and_plots(tmp_path: Path) -> None:
    summary_path = tmp_path / "comparison_summary.json"
    out_dir = tmp_path / "assets"
    rows = [
        {
            "case_id": "np8:local_assembly:local_solver",
            "assembly_backend": "local",
            "solver_backend": "local",
            "combo_label": "local assembly + local solver",
            "ranks": 8,
            "status": "completed",
            "message": "Converged",
            "solver_success": True,
            "exit_code": 0,
            "wall_time_s": 410.0,
            "solve_time_s": 390.0,
            "nit": 20,
            "linear_iterations_total": 360,
            "final_metric": 8.0e-4,
            "final_metric_name": "relative_correction",
            "energy": -5.0,
            "omega": 2.1,
            "u_max": 0.30,
            "stdout_path": "",
            "stderr_path": "",
            "result_json": "",
            "case_dir": "",
            "command": "",
            "history_metric_name": "relative_correction",
            "history_iterations": [1, 20],
            "history_metric": [1.0, 8.0e-4],
            "initial_guess_enabled": True,
            "initial_guess_success": True,
            "initial_guess_ksp_iterations": 10,
        },
        {
            "case_id": "np8:source_assembly:local_solver",
            "assembly_backend": "source",
            "solver_backend": "local",
            "combo_label": "source assembly + local solver",
            "ranks": 8,
            "status": "completed",
            "message": "Converged",
            "solver_success": True,
            "exit_code": 0,
            "wall_time_s": 520.0,
            "solve_time_s": 500.0,
            "nit": 18,
            "linear_iterations_total": 420,
            "final_metric": 7.0e-4,
            "final_metric_name": "relative_correction",
            "energy": -5.0,
            "omega": 2.1,
            "u_max": 0.30,
            "stdout_path": "",
            "stderr_path": "",
            "result_json": "",
            "case_dir": "",
            "command": "",
            "history_metric_name": "relative_correction",
            "history_iterations": [1, 18],
            "history_metric": [1.0, 7.0e-4],
            "initial_guess_enabled": True,
            "initial_guess_success": True,
            "initial_guess_ksp_iterations": 12,
        },
        {
            "case_id": "np8:local_assembly:source_solver",
            "assembly_backend": "local",
            "solver_backend": "source",
            "combo_label": "local assembly + source solver",
            "ranks": 8,
            "status": "completed",
            "message": "Converged",
            "solver_success": True,
            "exit_code": 0,
            "wall_time_s": 470.0,
            "solve_time_s": 450.0,
            "nit": 14,
            "linear_iterations_total": 760,
            "final_metric": 6.0e-4,
            "final_metric_name": "relative_correction",
            "energy": -5.0,
            "omega": 2.1,
            "u_max": 0.30,
            "stdout_path": "",
            "stderr_path": "",
            "result_json": "",
            "case_dir": "",
            "command": "",
            "history_metric_name": "relative_correction",
            "history_iterations": [1, 14],
            "history_metric": [1.0, 6.0e-4],
            "initial_guess_enabled": True,
            "initial_guess_success": True,
            "initial_guess_ksp_iterations": 11,
        },
        {
            "case_id": "np8:source_assembly:source_solver",
            "assembly_backend": "source",
            "solver_backend": "source",
            "combo_label": "source assembly + source solver",
            "ranks": 8,
            "status": "completed",
            "message": "Converged",
            "solver_success": True,
            "exit_code": 0,
            "wall_time_s": 640.0,
            "solve_time_s": 620.0,
            "nit": 13,
            "linear_iterations_total": 900,
            "final_metric": 5.0e-4,
            "final_metric_name": "relative_correction",
            "energy": -5.0,
            "omega": 2.1,
            "u_max": 0.30,
            "stdout_path": "",
            "stderr_path": "",
            "result_json": "",
            "case_dir": "",
            "command": "",
            "history_metric_name": "relative_correction",
            "history_iterations": [1, 13],
            "history_metric": [1.0, 5.0e-4],
            "initial_guess_enabled": True,
            "initial_guess_success": True,
            "initial_guess_ksp_iterations": 14,
        },
    ]
    summary_path.write_text(
        json.dumps(
            {
                "runner": "plasticity3d_backend_mix_compare",
                "source_env_mode": "shared_env",
                "ranks": 8,
                "stop_metric_name": "relative_correction",
                "stop_tol": 2.0e-3,
                "maxit": 80,
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "experiments/analysis/generate_plasticity3d_backend_mix_assets.py",
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
    assert (out_dir / "backend_mix_bar_metrics.png").exists()
    assert (out_dir / "backend_mix_solver_pair_times.png").exists()
    assert (out_dir / "backend_mix_solver_pair_convergence.png").exists()

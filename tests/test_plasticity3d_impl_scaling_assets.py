from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_impl_scaling_asset_generator_writes_report_and_plots(tmp_path: Path) -> None:
    summary_path = tmp_path / "comparison_summary.json"
    out_dir = tmp_path / "assets"

    local_case = tmp_path / "local_case"
    source_case = tmp_path / "source_case"
    (local_case / "data").mkdir(parents=True)
    (source_case / "data").mkdir(parents=True)

    local_result = local_case / "output.json"
    local_result.write_text(
        json.dumps(
            {
                "status": "completed",
                "solver_success": True,
                "total_time": 120.0,
                "solve_time": 100.0,
                "nit": 7,
                "linear_iterations_total": 380,
                "final_metric": 1.0e-3,
                "history": [
                    {"it": 1, "step_rel": 1.0, "t_grad": 1.2, "t_ls": 2.4, "t_update": 0.1},
                    {"it": 7, "step_rel": 1.0e-3, "t_grad": 1.1, "t_ls": 2.2, "t_update": 0.1},
                ],
                "linear_history": [
                    {"t_assemble": 10.0, "t_setup": 2.0, "t_solve": 20.0},
                    {"t_assemble": 8.0, "t_setup": 1.5, "t_solve": 18.0},
                ],
                "initial_guess": {"solve_time": 5.0},
                "assembly_callbacks": {
                    "energy": {"total": 0.8},
                    "gradient": {"total": 1.1},
                    "hessian": {
                        "total": 18.0,
                        "hvp_compute": 12.0,
                        "extraction": 4.0,
                        "accumulate": 1.5,
                        "coo_insert": 0.5,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (local_case / "data" / "stage.jsonl").write_text(
        json.dumps({"stage": "backend_ready", "elapsed_s": 15.0}) + "\n",
        encoding="utf-8",
    )

    source_result = source_case / "output.json"
    source_result.write_text(
        json.dumps(
            {
                "status": "completed",
                "solver_success": True,
                "total_time": 180.0,
                "solve_time": 150.0,
                "nit": 14,
                "linear_iterations_total": 547,
                "final_metric": 8.0e-4,
                "history": [
                    {
                        "iteration": 1,
                        "metric": 1.0,
                        "linear_solve_time": 12.0,
                        "linear_preconditioner_time": 8.0,
                        "linear_orthogonalization_time": 1.5,
                        "iteration_wall_time": 18.0,
                    },
                    {
                        "iteration": 14,
                        "metric": 8.0e-4,
                        "linear_solve_time": 11.0,
                        "linear_preconditioner_time": 7.0,
                        "linear_orthogonalization_time": 1.0,
                        "iteration_wall_time": 16.0,
                    },
                ],
                "initial_guess": {"solve_time": 6.0},
            }
        ),
        encoding="utf-8",
    )
    (source_case / "data" / "stage.jsonl").write_text(
        json.dumps({"stage": "backend_ready", "elapsed_s": 22.0}) + "\n",
        encoding="utf-8",
    )
    source_builder = source_case / "data" / "source_builder_timings.json"
    source_builder.write_text(
        json.dumps(
            {
                "local_strain": 3.0,
                "local_constitutive": 4.5,
                "local_constitutive_comm": 0.6,
                "build_tangent_local": 2.4,
                "local_force_assembly": 0.8,
                "local_force_gather": 0.3,
                "build_F": 1.1,
            }
        ),
        encoding="utf-8",
    )

    rows = [
        {
            "case_id": "np8:maintained_local_best",
            "implementation": "maintained_local_best",
            "display_label": "Maintained local_constitutiveAD + local solver (fast Hypre)",
            "family": "local",
            "assembly_backend": "local_constitutiveAD",
            "solver_backend": "local",
            "solver_profile": "hypre_fast",
            "combo_label": "local_constitutiveAD assembly + local solver",
            "ranks": 8,
            "status": "completed",
            "message": "Converged",
            "solver_success": True,
            "exit_code": 0,
            "wall_time_s": 120.0,
            "solve_time_s": 100.0,
            "nit": 7,
            "linear_iterations_total": 380,
            "final_metric": 1.0e-3,
            "final_metric_name": "relative_correction",
            "energy": -5.0,
            "omega": 2.1,
            "u_max": 0.30,
            "stdout_path": "",
            "stderr_path": "",
            "result_json": str(local_result),
            "case_dir": str(local_case),
            "stage_jsonl": str(local_case / "data" / "stage.jsonl"),
            "source_builder_timings_json": "",
            "command": "",
            "history_metric_name": "relative_correction",
            "history_iterations": [1, 7],
            "history_metric": [1.0, 1.0e-3],
            "initial_guess_enabled": True,
            "initial_guess_success": True,
            "initial_guess_ksp_iterations": 12,
            "native_run_info": "",
            "native_history_json": "",
        },
        {
            "case_id": "np8:source_petsc4py",
            "implementation": "source_petsc4py",
            "display_label": "Source assembly + source solver (Hypre)",
            "family": "source",
            "assembly_backend": "source",
            "solver_backend": "source",
            "solver_profile": "hypre",
            "combo_label": "source assembly + source solver",
            "ranks": 8,
            "status": "completed",
            "message": "Converged",
            "solver_success": True,
            "exit_code": 0,
            "wall_time_s": 180.0,
            "solve_time_s": 150.0,
            "nit": 14,
            "linear_iterations_total": 547,
            "final_metric": 8.0e-4,
            "final_metric_name": "relative_correction",
            "energy": float("nan"),
            "omega": 2.1,
            "u_max": 0.30,
            "stdout_path": "",
            "stderr_path": "",
            "result_json": str(source_result),
            "case_dir": str(source_case),
            "stage_jsonl": str(source_case / "data" / "stage.jsonl"),
            "source_builder_timings_json": str(source_builder),
            "command": "",
            "history_metric_name": "relative_correction",
            "history_iterations": [1, 14],
            "history_metric": [1.0, 8.0e-4],
            "initial_guess_enabled": True,
            "initial_guess_success": True,
            "initial_guess_ksp_iterations": 14,
            "native_run_info": "",
            "native_history_json": "",
        },
    ]
    summary_path.write_text(
        json.dumps(
            {
                "runner": "plasticity3d_impl_scaling_compare",
                "source_env_mode": "shared_env",
                "ranks": [8],
                "stop_metric_name": "relative_correction",
                "stop_tol": 2.0e-3,
                "maxit": 80,
                "implementations": [
                    {
                        "implementation": "maintained_local_best",
                        "display_label": "Maintained local_constitutiveAD + local solver (fast Hypre)",
                        "family": "local",
                    },
                    {
                        "implementation": "source_petsc4py",
                        "display_label": "Source assembly + source solver (Hypre)",
                        "family": "source",
                    },
                ],
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "experiments/analysis/generate_plasticity3d_impl_scaling_assets.py",
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
    assert (out_dir / "scaling_breakdown.json").exists()
    assert (out_dir / "overall_scaling.png").exists()
    assert (out_dir / "common_component_scaling.png").exists()
    assert (out_dir / "maintained_local_best_component_scaling.png").exists()
    assert (out_dir / "source_petsc4py_component_scaling.png").exists()

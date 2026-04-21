from __future__ import annotations

import json
from pathlib import Path

from experiments.runners import run_plasticity3d_impl_scaling_compare as runner


def test_normalize_impl_scaling_payload_contract(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True)
    result_path = case_dir / "output.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "message": "Converged",
                "solver_success": True,
                "total_time": 82.0,
                "solve_time": 74.0,
                "nit": 9,
                "linear_iterations_total": 188,
                "final_metric": 9.5e-4,
                "final_metric_name": "relative_correction",
                "energy": -5.5,
                "omega": 2.2,
                "u_max": 0.32,
                "stop_metric_name": "relative_correction",
                "history": [
                    {"iteration": 1, "metric": 1.0},
                    {"iteration": 9, "metric": 9.5e-4},
                ],
                "initial_guess": {
                    "enabled": True,
                    "success": True,
                    "ksp_iterations": 17,
                },
            }
        ),
        encoding="utf-8",
    )
    builder_timings = data_dir / "source_builder_timings.json"
    builder_timings.write_text(json.dumps({"build_tangent_local": 3.5}), encoding="utf-8")

    row = runner._normalize_mix_payload(
        case_id="np8:source_petsc4py",
        impl={
            "name": "source_petsc4py",
            "display_label": "Source assembly + source solver (Hypre)",
            "family": "source",
            "assembly_backend": "source",
            "solver_backend": "source",
            "solver_profile": "hypre",
        },
        ranks=8,
        exit_code=0,
        case_dir=case_dir,
        stdout_path=case_dir / "stdout.txt",
        stderr_path=case_dir / "stderr.txt",
        result_path=result_path,
        command=["python", "case.py"],
    )

    assert set(row) == set(runner.NORMALIZED_ROW_KEYS)
    assert row["implementation"] == "source_petsc4py"
    assert row["display_label"] == "Source assembly + source solver (Hypre)"
    assert row["family"] == "source"
    assert row["solver_profile"] == "hypre"
    assert row["combo_label"] == "source assembly + source solver"
    assert row["history_iterations"] == [1, 9]
    assert row["history_metric"] == [1.0, 9.5e-4]
    assert row["initial_guess_ksp_iterations"] == 17
    assert row["source_builder_timings_json"].endswith("source_builder_timings.json")
    assert row["native_run_info"] == ""

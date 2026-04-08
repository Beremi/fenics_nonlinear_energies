from __future__ import annotations

import json
from pathlib import Path

from experiments.runners import run_plasticity3d_backend_mix_compare as runner


def test_normalize_backend_mix_payload_contract(tmp_path: Path) -> None:
    result_path = tmp_path / "output.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "message": "Converged",
                "solver_success": True,
                "total_time": 45.0,
                "solve_time": 41.0,
                "nit": 12,
                "linear_iterations_total": 220,
                "final_metric": 6.0e-4,
                "final_metric_name": "relative_correction",
                "energy": -5.0,
                "omega": 2.1,
                "u_max": 0.31,
                "history": [
                    {"iteration": 1, "metric": 1.0},
                    {"iteration": 12, "metric": 6.0e-4},
                ],
                "initial_guess": {
                    "enabled": True,
                    "success": True,
                    "ksp_iterations": 13,
                },
            }
        ),
        encoding="utf-8",
    )

    row = runner._normalize_payload(
        case_id="np8:source_assembly:local_solver",
        assembly_backend="source",
        solver_backend="local",
        ranks=8,
        exit_code=0,
        case_dir=tmp_path,
        stdout_path=tmp_path / "stdout.txt",
        stderr_path=tmp_path / "stderr.txt",
        result_path=result_path,
        command=["python", "case.py"],
    )

    assert set(row) == set(runner.NORMALIZED_ROW_KEYS)
    assert row["combo_label"] == "source assembly + local solver"
    assert row["history_iterations"] == [1, 12]
    assert row["history_metric"] == [1.0, 6.0e-4]
    assert row["initial_guess_ksp_iterations"] == 13

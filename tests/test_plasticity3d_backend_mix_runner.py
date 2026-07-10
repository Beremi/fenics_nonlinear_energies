from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.runners import run_plasticity3d_backend_mix_compare as runner
from experiments.runners import run_plasticity3d_backend_mix_case as case_runner


class _SharedBroadcast:
    payload: object = None


class _SequentialFakeComm:
    def __init__(self, rank: int, shared: _SharedBroadcast) -> None:
        self.rank = rank
        self._shared = shared

    def bcast(self, payload: object, root: int) -> object:
        if self.rank == root:
            self._shared.payload = payload
        return self._shared.payload


def test_branch_map_root_validation_error_is_broadcast_to_every_rank() -> None:
    shared = _SharedBroadcast()
    duplicate_map = [
        (np.asarray([7], dtype=np.int64), np.asarray([[0, 1]], dtype=np.int8)),
        (np.asarray([7], dtype=np.int64), np.asarray([[1, 0]], dtype=np.int8)),
    ]

    with pytest.raises(RuntimeError) as root_error:
        case_runner._canonical_endpoint_branch_map_sha256(
            _SequentialFakeComm(0, shared), duplicate_map
        )
    with pytest.raises(RuntimeError) as worker_error:
        case_runner._canonical_endpoint_branch_map_sha256(
            _SequentialFakeComm(1, shared), None
        )

    assert str(worker_error.value) == str(root_error.value)
    assert "duplicate global elements" in str(root_error.value)


def test_normalize_backend_mix_payload_contract(tmp_path: Path) -> None:
    result_path = tmp_path / "output.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "message": "Converged",
                "solver_success": True,
                "quadrature_rule_id": "tetra_24point",
                "quadrature_points": 24,
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
    assert row["quadrature_rule_id"] == "tetra_24point"
    assert row["quadrature_points"] == 24


def test_normalize_backend_mix_payload_accepts_json_null_history_sentinels(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "output.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "message": "linear solve failed",
                "solver_success": False,
                "total_time": None,
                "solve_time": None,
                "final_metric": None,
                "energy": -5.0,
                "omega": None,
                "u_max": None,
                "history": [{"it": 0, "step_rel": None}],
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    row = runner._normalize_payload(
        case_id="np2:local_constitutiveAD:local_solver",
        assembly_backend="local_constitutiveAD",
        solver_backend="local",
        ranks=2,
        exit_code=2,
        case_dir=tmp_path,
        stdout_path=tmp_path / "stdout.txt",
        stderr_path=tmp_path / "stderr.txt",
        result_path=result_path,
        command=["python", "case.py"],
    )

    assert row["history_metric"] == [None]
    assert row["wall_time_s"] is None
    assert row["solve_time_s"] is None
    assert row["final_metric"] is None
    assert row["energy"] == -5.0
    assert row["omega"] is None
    assert row["u_max"] is None

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from src.core.benchmark.run_record import (
    RUN_RECORD_SCHEMA_ID,
    validate_run_record,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
RUNNER = (
    REPO_ROOT
    / "experiments"
    / "runners"
    / "run_plasticity3d_material_point_verification.py"
)


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-standard JSON constant {token}")


def test_exp_mc_material_point_campaign_covers_branches_and_interfaces(
    tmp_path: Path,
) -> None:
    output = tmp_path / "material_point_verification.json"
    report = tmp_path / "pilot_report.md"
    run_record_path = tmp_path / "run_record.json"
    env = os.environ.copy()
    env.update(
        {
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
        }
    )
    completed = subprocess.run(
        [
            str(PYTHON),
            str(RUNNER),
            "--output",
            str(output),
            "--report",
            str(report),
            "--run-record",
            str(run_record_path),
            "--run-kind",
            "pilot",
            "--pilot-dirty-override",
            "--pilot-override-reason",
            "focused EXP-MC-001 test on the active worktree",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0

    payload = json.loads(
        output.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite
    )
    assert payload["schema_name"] == "plasticity3d_material_point_verification"
    assert payload["schema_version"] == 1
    assert payload["experiment_id"] == "EXP-MC-001"
    assert payload["status"] == "passed"
    assert payload["summary"]["cpu_fp64_execution_passed"] is True

    required_branches = {
        "elastic",
        "shear",
        "left_edge",
        "right_edge",
        "apex",
    }
    assert payload["summary"]["branch_interior_counts"] == {
        branch: 1 for branch in ("elastic", "shear", "left_edge", "right_edge", "apex")
    }
    assert {
        record["evaluation"]["branch_diagnostics"]["branch"]
        for record in payload["branch_interiors"]
    } == required_branches
    for record in payload["branch_interiors"]:
        evaluation = record["evaluation"]
        diagnostic = evaluation["branch_diagnostics"]
        directional = record["directional_check"]
        assert evaluation["finite_energy_gradient_hessian"] is True
        assert evaluation["hessian_symmetry_defect"] <= 1.0e-10
        assert evaluation["numpy_selected_energy_relative_error"] <= 1.0e-12
        assert diagnostic["normalized_active_branch_margin"] >= 1.0e-3
        assert diagnostic["minimum_normalized_raw_principal_gap"] > 1.0e-2
        assert diagnostic["minimum_normalized_denominator"] > 0.0
        assert diagnostic["relative_tie_break_scale"] > 0.0
        assert diagnostic["reference_scope"].endswith(
            "not an independent constitutive reference"
        )
        assert directional["branch_stable_for_all_steps"] is True
        assert directional["energy_error_at_gate"] <= 1.0e-7
        assert directional["hvp_error_at_gate"] <= 1.0e-7

    expected_interfaces = {
        "elastic--shear",
        "shear--left_edge",
        "shear--right_edge",
        "left_edge--apex",
        "right_edge--apex",
    }
    assert {sweep["interface"] for sweep in payload["interface_sweeps"]} == expected_interfaces
    assert payload["summary"]["interface_pair_count"] == 15
    for sweep in payload["interface_sweeps"]:
        assert len(sweep["pairs"]) == 3
        assert [
            pair["normalized_offset_fraction_of_anchor_segment"]
            for pair in sweep["pairs"]
        ] == [1.0e-2, 1.0e-4, 1.0e-6]
        for pair in sweep["pairs"]:
            assert pair["left"]["branch_diagnostics"]["branch"] == sweep["left_branch"]
            assert pair["right"]["branch_diagnostics"]["branch"] == sweep["right_branch"]
            assert pair["left"]["finite_energy_gradient_hessian"] is True
            assert pair["right"]["finite_energy_gradient_hessian"] is True
            assert pair["left"]["hessian_symmetry_defect"] <= 1.0e-10
            assert pair["right"]["hessian_symmetry_defect"] <= 1.0e-10
            assert pair["left"]["numpy_selected_energy_relative_error"] <= 1.0e-12
            assert pair["right"]["numpy_selected_energy_relative_error"] <= 1.0e-12
            assert pair["derivative_gate_applied"] is False

    assert payload["summary"]["rotation_check_count"] == 15
    for record in payload["rotation_covariance"]:
        assert record["branch"] == record["rotated_branch"]
        assert record["finite_energy_gradient_hessian"] is True
        assert record["orthogonality_defect"] <= 1.0e-12
        assert abs(record["determinant"] - 1.0) <= 1.0e-12
        assert record["energy_invariance_scaled_error"] <= 1.0e-9
        assert record["stress_covariance_scaled_error"] <= 1.0e-9
        assert record["numpy_selected_energy_relative_error"] <= 1.0e-12
        assert record["hessian_symmetry_defect"] <= 1.0e-10
        assert (
            record["tangent_action_covariance_scaled_error"] <= 1.0e-9
            or record["tangent_action_covariance_absolute_error"] <= 1.0e-9
        )

    repeated = payload["repeated_principal_value_cases"]
    assert len(repeated) == 7
    assert any(
        record["evaluation"]["branch_diagnostics"]["minimum_raw_principal_gap"]
        == 0.0
        for record in repeated
    )
    assert all(
        record["evaluation"]["finite_energy_gradient_hessian"] is True
        and record["evaluation"]["numpy_selected_energy_relative_error"] <= 1.0e-12
        and record["derivative_gate_applied"] is False
        and record["rotation_gate_applied"] is False
        for record in repeated
    )
    assert "no generalized differentiability at a branch switch" in payload[
        "method_scope"
    ]["excluded_claims"]

    report_text = report.read_text(encoding="utf-8")
    assert "# EXP-MC-001 Local Material-Point Pilot" in report_text
    assert "no generalized-differentiability claim" in report_text

    run_record = json.loads(
        run_record_path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite,
    )
    validate_run_record(run_record)
    assert run_record["schema"]["id"] == RUN_RECORD_SCHEMA_ID
    assert run_record["identifiers"]["experiment"] == "EXP-MC-001"
    assert run_record["run_kind"] == "pilot"
    assert run_record["termination"]["status"] == "success"
    assert run_record["accuracy"]["gate_passed"] is True
    assert run_record["counts"]["function_evaluations"] > 0
    assert run_record["timing"]["total_s"] > 0.0
    assert run_record["resources"]["ranks"] == 1
    assert run_record["environment"]["petsc"] == "not-applicable"
    assert run_record["environment"]["mpi"] == "not-applicable"

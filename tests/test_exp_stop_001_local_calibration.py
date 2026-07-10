from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.runners import run_exp_stop_001_local_calibration as campaign
from experiments.runners import run_plasticity3d_backend_mix_case as p3d_runner
from experiments.runners import run_trust_region_case as trust_runner


COMMIT = "a" * 40


def _clean_git() -> dict[str, object]:
    return {"commit": COMMIT, "dirty": False}


def _prepare_args(root: Path, **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "output_root": root,
        "run_kind": "publication",
        "allow_dirty": False,
        "p4_policy": "deferred_cluster",
        "confirm_p4_local_feasible": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_frozen_default_matrix_is_runnable_and_scientifically_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(campaign, "_git_metadata", _clean_git)
    plan = campaign.build_plan(
        tmp_path / "campaign",
        run_kind="publication",
        allow_dirty=False,
        p4_policy="deferred_cluster",
        confirm_p4_local_feasible=False,
    )

    assert plan["row_counts"] == {
        "total": 52,
        "required_local": 40,
        "deferred_cluster_computation": 12,
    }
    counts = Counter(
        (row["family"], row["execution_class"]) for row in plan["rows"]
    )
    assert counts[("ginzburg_landau", "required_local")] == 8
    assert counts[("hyperelasticity_reference_riesz", "required_local")] == 6
    assert counts[("hyperelasticity_nonlinear_stopping", "required_local")] == 8
    assert counts[("plasticity3d_fixed_state_linear", "required_local")] == 10
    assert counts[("plasticity3d_nonlinear_stopping", "required_local")] == 8
    assert counts[("plasticity3d_fixed_state_linear", "deferred_cluster_computation")] == 5
    assert counts[("plasticity3d_nonlinear_stopping", "deferred_cluster_computation")] == 4

    local_rows = [
        row for row in plan["rows"] if row["execution_class"] == "required_local"
    ]
    assert all(Path(row["command"][0]).samefile(Path(campaign.sys.executable)) for row in local_rows)
    for row in local_rows:
        command = row["command"]
        assert isinstance(command, list) and command
        if command[1] == campaign.TRUST_RUNNER_PATH.as_posix():
            trust_runner._build_parser().parse_args(command[2:])
        elif command[1] == campaign.P3D_BACKEND_PATH.as_posix():
            p3d_runner._build_parser().parse_args(command[2:])
        else:
            campaign._build_parser().parse_args(command[2:])
        assert row["expected_outputs"]
        assert row["environment"]["JAX_PLATFORMS"] == "cpu"

    he_setup = next(
        row for row in local_rows if row["row_id"] == "he_l1_riesz_1em08"
    )
    assert he_setup["parameters"]["nonlinear_max_iterations"] == 0
    assert "--maxit" in he_setup["command"]
    assert he_setup["command"][he_setup["command"].index("--maxit") + 1] == "0"

    he_nonlinear = next(
        row for row in local_rows if row["row_id"] == "he_l1_nonlinear_1em02"
    )
    assert "--state-out" in he_nonlinear["command"]
    assert he_nonlinear["parameters"]["load_steps"] == 1

    p3d_nonlinear = next(
        row for row in local_rows if row["row_id"] == "p3d_p1_nonlinear_1em02"
    )
    assert "--convergence-metric" in p3d_nonlinear["command"]
    assert "reference_elastic_energy" in p3d_nonlinear["command"]
    assert "--state-out" in p3d_nonlinear["command"]

    deferred = [
        row
        for row in plan["rows"]
        if row["execution_class"] == "deferred_cluster_computation"
    ]
    assert all(row["censor"]["status"] == "censored" for row in deferred)
    assert all(row["censor"]["timing_admissible"] is False for row in deferred)
    assert {
        row["parameters"].get("element_degree")
        for row in deferred
        if row["family"] == "plasticity3d_nonlinear_stopping"
    } == {4}


def test_clean_preflight_fresh_root_and_p4_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        campaign, "_git_metadata", lambda: {"commit": COMMIT, "dirty": True}
    )
    with pytest.raises(campaign.CampaignError, match="dirty"):
        campaign.build_plan(
            tmp_path / "dirty-publication",
            run_kind="publication",
            allow_dirty=False,
            p4_policy="deferred_cluster",
            confirm_p4_local_feasible=False,
        )

    monkeypatch.setattr(campaign, "_git_metadata", _clean_git)
    with pytest.raises(campaign.CampaignError, match="confirm-p4"):
        campaign.build_plan(
            tmp_path / "p4",
            run_kind="publication",
            allow_dirty=False,
            p4_policy="local",
            confirm_p4_local_feasible=False,
        )
    p4_plan = campaign.build_plan(
        tmp_path / "p4",
        run_kind="publication",
        allow_dirty=False,
        p4_policy="local",
        confirm_p4_local_feasible=True,
    )
    assert p4_plan["row_counts"] == {
        "total": 52,
        "required_local": 45,
        "deferred_cluster_computation": 7,
    }

    root = tmp_path / "fresh"
    path = campaign.prepare_campaign(_prepare_args(root))
    assert path == root / "plan.json"
    assert (root / "receipts").is_dir()
    with pytest.raises(campaign.CampaignError, match="already exists"):
        campaign.prepare_campaign(_prepare_args(root))


def test_incomplete_analysis_retains_cluster_censors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(campaign, "_git_metadata", _clean_git)
    plan_path = campaign.prepare_campaign(_prepare_args(tmp_path / "campaign"))
    analysis = campaign.analyze_plan(plan_path)
    assert analysis["terminal_decision"] == "incomplete_local_execution"
    assert analysis["complete_exp_stop_pass"] is False
    assert analysis["publication_timing_admissible"] is False
    assert analysis["counts"]["required_local"] == 40
    assert analysis["counts"]["missing_local"] == 40
    assert analysis["counts"]["deferred_cluster_computations"] == 12
    assert len(analysis["deferred_cluster_computations"]) == 12
    assert "P4 nonlinear" in analysis["scope_statement"]


def test_failed_frozen_command_is_an_explicit_unclassified_runtime_censor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(campaign, "_git_metadata", _clean_git)
    plan_path = campaign.prepare_campaign(_prepare_args(tmp_path / "campaign"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    row = next(row for row in plan["rows"] if row["execution_class"] == "required_local")
    receipt = {
        "schema_id": campaign.RECEIPT_SCHEMA_ID,
        "schema_version": campaign.RECEIPT_SCHEMA_VERSION,
        "row_id": row["row_id"],
        "plan_sha256": campaign._sha256_file(plan_path),
        "command": row["command"],
        "status": "failed",
        "returncode": 2,
        "timed_out": False,
        "output_hashes": {},
    }
    receipt_path = tmp_path / "campaign" / "receipts" / f"{row['row_id']}.json"
    campaign._atomic_json(receipt_path, receipt)

    status, _payload, errors = campaign._receipt_for_row(
        plan_path=plan_path,
        output_root=tmp_path / "campaign",
        row=row,
    )
    assert status == "runtime_censored"
    assert any("not as convergence evidence" in error for error in errors)


def test_required_policy_grid_blocks_false_local_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(campaign, "_git_metadata", _clean_git)
    plan = campaign.build_plan(
        tmp_path / "campaign",
        run_kind="publication",
        allow_dirty=False,
        p4_policy="local",
        confirm_p4_local_feasible=True,
    )
    local_rows = [
        row for row in plan["rows"] if row["execution_class"] == "required_local"
    ]
    selected: dict[str, dict[str, object]] = {}
    for group in campaign.COMPLETE_REQUIRED_LOCAL_GROUPS:
        reference = next(
            row
            for row in local_rows
            if row["group_id"] == group and row["reference_row"]
        )
        parameter = {
            "ginzburg_landau": "relative_dual_residual_target",
            "hyperelasticity_reference_riesz": "riesz_ksp_rtol",
            "hyperelasticity_nonlinear_stopping": "relative_dual_residual_target",
            "plasticity3d_fixed_state_linear": "ksp_rtol",
            "plasticity3d_nonlinear_stopping": "relative_dual_residual_target",
        }[reference["family"]]
        selected[group] = {
            "status": campaign.ACCEPTED_POLICY_STATUS,
            "row_id": reference["row_id"],
            "parameter": parameter,
            "tolerance": reference["parameters"][parameter],
        }

    passing = campaign._required_local_policy_grid(local_rows, selected)
    assert passing["complete"] is True

    selected["p3d_p4"] = {
        "status": "no_acceptable_policy",
        "row_id": None,
        "tolerance": None,
    }
    failed = campaign._required_local_policy_grid(local_rows, selected)
    assert failed["complete"] is False
    assert failed["rejected_policy_groups"] == ["p3d_p4"]
    assert campaign._local_terminal_decision(
        missing=[],
        invalid=[],
        runtime_censored=[],
        reference_failures=[],
        policy_grid=failed,
    ) == "local_calibration_policy_gate_failed"


def test_p3d_worker_parser_rejects_unmatched_degree_rule() -> None:
    args = campaign._build_parser().parse_args(
        [
            "p3d-fixed-state",
            "--degree",
            "1",
            "--quadrature-rule",
            "tetra_11point",
            "--state-amplitude",
            "0.0002",
            "--ksp-rtol",
            "1e-4",
            "--output",
            "result.json",
            "--state-out",
            "state.npz",
        ]
    )
    # The parser accepts the global rule vocabulary; the worker preflight
    # enforces the frozen degree/rule pairing before it allocates a backend.
    with pytest.raises(campaign.CampaignError, match="requires tetra_1point"):
        campaign._validate_p3d_worker_args(args)

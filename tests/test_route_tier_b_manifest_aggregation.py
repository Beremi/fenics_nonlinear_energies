from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from experiments.analysis import aggregate_route_tier_b_manifests as aggregation
from experiments.analysis import analyze_plasticity3d_route_endpoints as endpoint_analysis


SCRIPT = (
    aggregation.REPO_ROOT
    / "experiments/analysis/aggregate_route_tier_b_manifests.py"
)
SOURCE_COMMIT = "1" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _release(
    root: Path,
    *,
    matrix_sha256: str,
    tiers: set[str],
    phase: str,
) -> dict[str, str]:
    reviewed = root / "reviewed_artifacts" / f"{phase}_review.json"
    _write_json(reviewed, {"status": "reviewed", "phase": phase})
    payload: dict[str, object] = {
        "schema_id": aggregation.RELEASE_SCHEMA_ID,
        "schema_version": 1,
        "status": "approved",
        "decision": "explicit_human_release_after_review",
        "matrix_sha256": matrix_sha256,
        "source_commit": SOURCE_COMMIT,
        "authorizes_experiment": aggregation.EXPERIMENT_ID,
        "authorizes_tiers": sorted(tiers),
        "reviewer": f"{phase}-reviewer",
        "reviewed_artifacts": [
            {
                "path": str(reviewed.relative_to(root)),
                "sha256": _sha256(reviewed),
            }
        ],
    }
    path = root / "release_authorization.json"
    _write_json(path, payload)
    return {
        "schema_id": aggregation.RELEASE_SCHEMA_ID,
        "path": path.name,
        "sha256": _sha256(path),
        "reviewer": f"{phase}-reviewer",
    }


def _environment(root: Path, *, lock_text: str = "shared-lock-v1\n") -> dict[str, object]:
    setup = root / "environment_contract" / "environment_setup.sh"
    lock = root / "environment_contract" / "environment.lock"
    setup.parent.mkdir(parents=True)
    setup.write_text("export TEST_REVIEWED_ENV=1\n", encoding="utf-8")
    lock.write_text(lock_text, encoding="utf-8")
    return {
        "status": "hash_bound",
        "runtime_setup_path": "/cluster/reviewed/environment_setup.sh",
        "setup_sha256": _sha256(setup),
        "runtime_lock_path": "/cluster/reviewed/environment.lock",
        "lock_sha256": _sha256(lock),
        "archived_setup": {
            "path": str(setup.relative_to(root)),
            "sha256": _sha256(setup),
        },
        "archived_lock": {
            "path": str(lock.relative_to(root)),
            "sha256": _sha256(lock),
        },
    }


def _model_freeze(
    root: Path,
    *,
    matrix_sha256: str,
    tier_b_training_manifest: Path,
    scope_case_ids: dict[str, list[str]],
) -> dict[str, str]:
    artifacts = root / "model_freeze_artifacts"
    artifacts.mkdir()
    tier_b_training_root = artifacts / "tier_b_training_campaign"
    shutil.copytree(tier_b_training_manifest.parent, tier_b_training_root)
    archived_tier_b_training = tier_b_training_root / "prepared_manifest.json"

    cost_training_root = artifacts / "cost_model_training_campaign"
    cost_training_root.mkdir()
    with aggregation.MATRIX.open(newline="", encoding="utf-8") as handle:
        matrix_rows = {row["case_id"]: dict(row) for row in csv.DictReader(handle)}
    cost_rows = [matrix_rows[case_id] for case_id in scope_case_ids["cost_model_training"]]
    cost_plan = cost_training_root / "prepared_plan.csv"
    with cost_plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cost_rows[0]))
        writer.writeheader()
        writer.writerows(cost_rows)
    cost_ledger = cost_training_root / "submitted_jobs.jsonl"
    with cost_ledger.open("w", encoding="utf-8") as handle:
        for offset, row in enumerate(cost_rows):
            job_id = str(3000 + offset)
            handle.write(
                json.dumps(
                    {
                        "case_id": row["case_id"],
                        "command": (
                            f"sbatch --job-name {row['case_id']} run_revision_case.sbatch"
                        ),
                        "returncode": 0,
                        "stdout": f"Submitted batch job {job_id}",
                        "stderr": "",
                        "job_id": job_id,
                    }
                )
                + "\n"
            )
    cost_environment = _environment(cost_training_root)
    cost_manifest = cost_training_root / "prepared_manifest.json"
    _write_json(
        cost_manifest,
        {
            "status": "submitted",
            "test_only_commands": False,
            "selected_experiments": [aggregation.EXPERIMENT_ID],
            "selected_tiers": sorted({row["tier"] for row in cost_rows}),
            "include_optional": False,
            "only_optional": False,
            "route_phase": "training",
            "route_phase_case_ids_sha256": aggregation._case_ids_sha256(
                scope_case_ids["cost_model_training"]
            ),
            "case_count": len(cost_rows),
            "matrix_sha256": matrix_sha256,
            "source_commit": SOURCE_COMMIT,
            "source_dirty": False,
            "plan_file": cost_plan.name,
            "plan_sha256": _sha256(cost_plan),
            "environment_contract": cost_environment,
        },
    )
    analysis = artifacts / "001_route_training_analysis.json"
    model = artifacts / "002_frozen_route_model.json"
    contract = json.loads(aggregation.CONTRACT.read_text(encoding="utf-8"))
    feature_order = list(contract["cost_model"]["features_in_order"])
    row_ids = [f"training-row-{index:03d}" for index in range(74)]
    _write_json(
        model,
        {
            "schema_id": "fenics-nonlinear-energies.exp-route-001-frozen-training-model",
            "schema_version": 1,
            "status": "frozen_before_holdout",
            "holdout_rows_seen": 0,
            "matrix_sha256": matrix_sha256,
            "source_commit": SOURCE_COMMIT,
            "contract_sha256": _sha256(aggregation.CONTRACT),
            "training_case_ids_sha256": aggregation._case_ids_sha256(
                scope_case_ids["cost_model_training"]
            ),
            "training_rows": 74,
            "training_row_ids": row_ids,
            "training_row_ids_sha256": aggregation._case_ids_sha256(row_ids),
            "feature_order": feature_order,
            "coefficients": {name: 0.0 for name in feature_order},
            "design_diagnostics": {
                "rows": 74,
                "columns": len(feature_order),
                "rank": len(feature_order),
                "condition_number": 1.0,
            },
        },
    )
    _write_json(
        analysis,
        {
            "schema_id": "fenics-nonlinear-energies.exp-route-001-training-analysis",
            "schema_version": 1,
            "status": "training_fit_admitted",
            "holdout_rows_seen": 0,
            "matrix_sha256": matrix_sha256,
            "source_commit": SOURCE_COMMIT,
            "training_case_ids": scope_case_ids["cost_model_training"],
            "training_case_ids_sha256": aggregation._case_ids_sha256(
                scope_case_ids["cost_model_training"]
            ),
            "training_row_ids": row_ids,
            "training_row_ids_sha256": aggregation._case_ids_sha256(row_ids),
            "frozen_model": {"path": model.name, "sha256": _sha256(model)},
        },
    )
    payload: dict[str, object] = {
        "schema_id": aggregation.MODEL_FREEZE_SCHEMA_ID,
        "schema_version": aggregation.MODEL_FREEZE_SCHEMA_VERSION,
        "status": "frozen_before_holdout",
        "decision": "cost_model_fit_and_tier_b_training_complete_holdouts_unopened",
        "matrix_sha256": matrix_sha256,
        "source_commit": SOURCE_COMMIT,
        "scopes": {
            name: {
                "case_count": aggregation.ROUTE_SCOPE_COUNTS[name],
                "case_ids_sha256": aggregation._case_ids_sha256(case_ids),
            }
            for name, case_ids in scope_case_ids.items()
        },
        "environment_identity": {
            "setup_sha256": cost_environment["setup_sha256"],
            "lock_sha256": cost_environment["lock_sha256"],
        },
        "created_at_utc": "2026-07-10T12:00:00+00:00",
        "reviewer": "holdout-model-reviewer",
        "cost_model_training_manifest": {
            "path": str(cost_manifest.relative_to(root)),
            "sha256": _sha256(cost_manifest),
        },
        "tier_b_training_manifest": {
            "path": str(archived_tier_b_training.relative_to(root)),
            "sha256": _sha256(archived_tier_b_training),
        },
        "training_analysis": {
            "path": str(analysis.relative_to(root)),
            "sha256": _sha256(analysis),
        },
        "frozen_model": {
            "path": str(model.relative_to(root)),
            "sha256": _sha256(model),
        },
    }
    path = root / "route_model_freeze.json"
    _write_json(path, payload)
    return {
        "schema_id": aggregation.MODEL_FREEZE_SCHEMA_ID,
        "schema_version": aggregation.MODEL_FREEZE_SCHEMA_VERSION,
        "path": path.name,
        "sha256": _sha256(path),
        "reviewer": "holdout-model-reviewer",
    }


def _phase_archive(
    archive_root: Path,
    *,
    phase: str,
    matrix_sha256: str,
    rows: dict[str, dict[str, str]],
    scope_case_ids: dict[str, list[str]],
    tier_b_training_manifest: Path | None = None,
    first_job_id: int,
    lock_text: str = "shared-lock-v1\n",
) -> Path:
    root = archive_root / phase
    root.mkdir(parents=True)
    plan = root / "prepared_plan.csv"
    planned = list(rows.values())
    with plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(planned[0]))
        writer.writeheader()
        writer.writerows(planned)
    tiers = {row["tier"] for row in planned}
    ledger = root / "submitted_jobs.jsonl"
    with ledger.open("w", encoding="utf-8") as handle:
        for offset, row in enumerate(planned):
            job_id = str(first_job_id + offset)
            handle.write(
                json.dumps(
                    {
                        "case_id": row["case_id"],
                        "command": (
                            f"sbatch --job-name {row['case_id']} run_revision_case.sbatch"
                        ),
                        "returncode": 0,
                        "stdout": f"Submitted batch job {job_id}",
                        "stderr": "",
                        "job_id": job_id,
                    }
                )
                + "\n"
            )
    route_model_freeze = None
    if phase == "holdout":
        assert tier_b_training_manifest is not None
        route_model_freeze = _model_freeze(
            root,
            matrix_sha256=matrix_sha256,
            tier_b_training_manifest=tier_b_training_manifest,
            scope_case_ids=scope_case_ids,
        )
    manifest: dict[str, object] = {
        "status": "submitted",
        "test_only_commands": False,
        "selected_experiments": [aggregation.EXPERIMENT_ID],
        "selected_tiers": sorted(tiers),
        "include_optional": True,
        "only_optional": True,
        "route_phase": phase,
        "route_phase_case_ids_sha256": aggregation._case_ids_sha256(rows),
        "case_count": len(planned),
        "matrix_sha256": matrix_sha256,
        "source_commit": SOURCE_COMMIT,
        "source_dirty": False,
        "plan_file": plan.name,
        "plan_sha256": _sha256(plan),
        "environment_contract": _environment(root, lock_text=lock_text),
        "release_authorization": _release(
            root,
            matrix_sha256=matrix_sha256,
            tiers=tiers,
            phase=phase,
        ),
        "route_model_freeze": route_model_freeze,
    }
    path = root / "prepared_manifest.json"
    _write_json(path, manifest)
    return path


def _archives(
    tmp_path: Path, *, holdout_lock: str = "shared-lock-v1\n"
) -> tuple[Path, Path, Path]:
    root = tmp_path / "tier_b_archive"
    root.mkdir()
    matrix_sha256, phases, scope_case_ids = aggregation._canonical_rows()
    training = _phase_archive(
        root,
        phase="training",
        matrix_sha256=matrix_sha256,
        rows=phases["training"],
        scope_case_ids=scope_case_ids,
        first_job_id=1000,
    )
    holdout = _phase_archive(
        root,
        phase="holdout",
        matrix_sha256=matrix_sha256,
        rows=phases["holdout"],
        scope_case_ids=scope_case_ids,
        tier_b_training_manifest=training,
        first_job_id=2000,
        lock_text=holdout_lock,
    )
    return root, training, holdout


def test_aggregates_exact_two_phase_tier_b_archives(tmp_path: Path) -> None:
    root, training, holdout = _archives(tmp_path)
    result = aggregation.aggregate(
        training_manifest=training,
        holdout_manifest=holdout,
        archive_root=root,
    )
    assert result["schema_id"] == aggregation.MASTER_SCHEMA_ID
    assert result["status"] == "submitted_phase_archives_complete"
    assert result["case_count"] == 30
    assert result["phase_counts"] == {"training": 20, "holdout": 10}
    assert set(result["case_to_phase"].values()) == {"training", "holdout"}
    assert result["phases"]["training"]["phase_archive_root"] == "training"
    assert result["phases"]["holdout"]["phase_archive_root"] == "holdout"
    assert result["phases"]["training"]["route_model_freeze"] is None
    assert (
        result["phases"]["holdout"]["route_model_freeze"][
            "tier_b_training_manifest_sha256"
        ]
        == result["phases"]["training"]["manifest_sha256"]
    )


def test_endpoint_analyzer_accepts_recomputed_phase_master(tmp_path: Path) -> None:
    root, training, holdout = _archives(tmp_path)
    master = aggregation.aggregate(
        training_manifest=training,
        holdout_manifest=holdout,
        archive_root=root,
    )
    master_path = root / "route_tier_b_campaign_master_manifest.json"
    _write_json(master_path, master)
    validated = endpoint_analysis._validate_manifest(master_path, aggregation.MATRIX)
    assert validated["eligible"] is True
    assert validated["manifest_type"] == "tier_b_phase_master"
    assert set(validated["case_archive_roots"].values()) == {"training", "holdout"}


def test_cli_writes_atomic_master_without_scheduler_code(tmp_path: Path) -> None:
    root, training, holdout = _archives(tmp_path)
    output = root / "route_tier_b_campaign_master_manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--training-manifest",
            str(training),
            "--holdout-manifest",
            str(holdout),
            "--archive-root",
            str(root),
            "--output",
            str(output),
        ],
        cwd=aggregation.REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["case_count"] == 30
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "sacct" not in source


def test_rejects_different_training_and_holdout_environment_locks(tmp_path: Path) -> None:
    root, training, holdout = _archives(tmp_path, holdout_lock="other-lock\n")
    with pytest.raises(aggregation.AggregationError, match="different environment"):
        aggregation.aggregate(
            training_manifest=training,
            holdout_manifest=holdout,
            archive_root=root,
        )


def test_rejects_incomplete_accepted_holdout_ledger(tmp_path: Path) -> None:
    root, training, holdout = _archives(tmp_path)
    ledger = holdout.parent / "submitted_jobs.jsonl"
    ledger.write_text(
        "\n".join(ledger.read_text(encoding="utf-8").splitlines()[:-1]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(aggregation.AggregationError, match="incomplete case coverage"):
        aggregation.aggregate(
            training_manifest=training,
            holdout_manifest=holdout,
            archive_root=root,
        )


def test_rejects_model_freeze_not_bound_to_admitted_training_manifest(
    tmp_path: Path,
) -> None:
    root, training, holdout = _archives(tmp_path)
    holdout_manifest = json.loads(holdout.read_text(encoding="utf-8"))
    freeze_path = holdout.parent / holdout_manifest["route_model_freeze"]["path"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    archived_training = holdout.parent / freeze["tier_b_training_manifest"]["path"]
    archived_payload = json.loads(archived_training.read_text(encoding="utf-8"))
    archived_payload["archival_note"] = "different-byte-copy"
    _write_json(archived_training, archived_payload)
    freeze["tier_b_training_manifest"]["sha256"] = _sha256(archived_training)
    _write_json(freeze_path, freeze)
    holdout_manifest["route_model_freeze"]["sha256"] = _sha256(freeze_path)
    _write_json(holdout, holdout_manifest)
    with pytest.raises(aggregation.AggregationError, match="not bound"):
        aggregation.aggregate(
            training_manifest=training,
            holdout_manifest=holdout,
            archive_root=root,
        )


def test_rejects_ambiguous_v1_model_freeze_receipt(tmp_path: Path) -> None:
    root, training, holdout = _archives(tmp_path)
    holdout_payload = json.loads(holdout.read_text(encoding="utf-8"))
    freeze_path = holdout.parent / holdout_payload["route_model_freeze"]["path"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["schema_version"] = 1
    _write_json(freeze_path, freeze)
    holdout_payload["route_model_freeze"]["schema_version"] = 1
    holdout_payload["route_model_freeze"]["sha256"] = _sha256(freeze_path)
    _write_json(holdout, holdout_payload)
    with pytest.raises(aggregation.AggregationError, match="v2"):
        aggregation.aggregate(
            training_manifest=training,
            holdout_manifest=holdout,
            archive_root=root,
        )


def test_rejects_noncanonical_declared_29_case_holdout_scope(tmp_path: Path) -> None:
    root, training, holdout = _archives(tmp_path)
    holdout_payload = json.loads(holdout.read_text(encoding="utf-8"))
    freeze_path = holdout.parent / holdout_payload["route_model_freeze"]["path"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["scopes"]["cost_model_holdout"]["case_count"] = 28
    _write_json(freeze_path, freeze)
    holdout_payload["route_model_freeze"]["sha256"] = _sha256(freeze_path)
    _write_json(holdout, holdout_payload)
    with pytest.raises(aggregation.AggregationError, match="identity or scope is stale"):
        aggregation.aggregate(
            training_manifest=training,
            holdout_manifest=holdout,
            archive_root=root,
        )


def test_rejects_different_cost_model_and_tier_b_training_environments(
    tmp_path: Path,
) -> None:
    root, training, holdout = _archives(tmp_path)
    holdout_payload = json.loads(holdout.read_text(encoding="utf-8"))
    freeze_path = holdout.parent / holdout_payload["route_model_freeze"]["path"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    cost_manifest_path = holdout.parent / freeze["cost_model_training_manifest"]["path"]
    cost_manifest = json.loads(cost_manifest_path.read_text(encoding="utf-8"))
    lock_record = cost_manifest["environment_contract"]["archived_lock"]
    lock_path = cost_manifest_path.parent / lock_record["path"]
    lock_path.write_text("different-cost-model-lock\n", encoding="utf-8")
    changed_hash = _sha256(lock_path)
    lock_record["sha256"] = changed_hash
    cost_manifest["environment_contract"]["lock_sha256"] = changed_hash
    _write_json(cost_manifest_path, cost_manifest)
    freeze["cost_model_training_manifest"]["sha256"] = _sha256(cost_manifest_path)
    freeze["environment_identity"]["lock_sha256"] = changed_hash
    _write_json(freeze_path, freeze)
    holdout_payload["route_model_freeze"]["sha256"] = _sha256(freeze_path)
    _write_json(holdout, holdout_payload)
    with pytest.raises(aggregation.AggregationError, match="do not share one environment"):
        aggregation.aggregate(
            training_manifest=training,
            holdout_manifest=holdout,
            archive_root=root,
        )


def test_rejects_phase_manifest_outside_explicit_archive_root(tmp_path: Path) -> None:
    root, training, holdout = _archives(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    moved = outside / training.name
    shutil.copy2(training, moved)
    with pytest.raises(aggregation.AggregationError, match="outside the explicit archive root"):
        aggregation.aggregate(
            training_manifest=moved,
            holdout_manifest=holdout,
            archive_root=root,
        )

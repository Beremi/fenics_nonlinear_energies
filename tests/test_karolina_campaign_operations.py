from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.analysis import collect_slurm_accounting as accounting
from experiments.analysis import finalize_karolina_campaign_archive as finalizer
from experiments.analysis import freeze_route_training_model as freezer
from experiments.runners.paper_revision_karolina import prepare_campaign as preparer
from experiments.runners.paper_revision_karolina import resume_partial_submission as resume


MATRIX = preparer.DEFAULT_MATRIX


def _sacct(row: dict[str, str], job_id: str) -> str:
    values = {
        "JobIDRaw": job_id,
        "JobID": job_id,
        "JobName": row["case_id"],
        "Cluster": "karolina",
        "Account": "fta-26-40",
        "Partition": row["partition"],
        "QOS": "3571_6328",
        "State": "COMPLETED",
        "ElapsedRaw": "10",
        "AllocNodes": row["nodes"],
        "AllocCPUS": row["total_ranks"],
        "TotalCPU": "00:00:10",
        "CPUTimeRAW": str(10 * int(row["total_ranks"])),
        "MaxRSS": "1K",
        "MaxVMSize": "2K",
        "ConsumedEnergyRaw": "0",
        "ExitCode": "0:0",
        "Start": "2026-07-10T10:00:00",
        "End": "2026-07-10T10:00:10",
        "NodeList": "cn001",
    }
    return "|".join(accounting.SACCT_FIELDS) + "\n" + "|".join(
        values[field] for field in accounting.SACCT_FIELDS
    ) + "\n"


def test_bulk_offline_accounting_and_copy_back_checksums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(finalizer.campaign, "offline_preflight", lambda *_args, **_kwargs: {})
    row = preparer.read_matrix(MATRIX)[0]
    root = tmp_path / "campaign"
    root.mkdir()
    plan = root / "prepared_plan.csv"
    with plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    manifest = {
        "status": "submitted",
        "plan_file": plan.name,
        "source_commit": "1" * 40,
        "source_dirty": False,
        "test_only_commands": False,
        "environment_contract": {"status": "hash_bound"},
    }
    (root / "prepared_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    (root / "submitted_jobs.jsonl").write_text(
        json.dumps(
            {
                "case_id": row["case_id"],
                "returncode": 0,
                "stdout": "Submitted batch job 123",
                "job_id": "123",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    command = "sbatch fixture"
    (root / "submission_journal.jsonl").write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "event": "intent",
                        "attempt_id": "initial-0001-fixture",
                        "case_id": row["case_id"],
                        "command": command,
                        "recorded_at_utc": "2026-07-10T10:00:00+00:00",
                    }
                ),
                json.dumps(
                    {
                        "event": "result",
                        "attempt_id": "initial-0001-fixture",
                        "case_id": row["case_id"],
                        "command": command,
                        "recorded_at_utc": "2026-07-10T10:00:01+00:00",
                        "returncode": 0,
                        "stdout": "Submitted batch job 123",
                        "stderr": "",
                        "job_id": "123",
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "jobs" / row["case_id"] / "job_123").mkdir(parents=True)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    raw = snapshots / "123.sacct"
    raw.write_text(_sacct(row, "123"), encoding="utf-8")
    index = snapshots / "index.json"
    index.write_text(
        json.dumps(
            {
                "schema_id": finalizer.OFFLINE_INDEX_SCHEMA_ID,
                "schema_version": 1,
                "campaign_manifest_sha256": hashlib.sha256(
                    (root / "prepared_manifest.json").read_bytes()
                ).hexdigest(),
                "records": [
                    {
                        "case_id": row["case_id"],
                        "job_id": "123",
                        "path": raw.name,
                        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = finalizer.finalize(root, offline_index=index)
    digest = result["archive_checksums"]["sha256"]
    assert result["settled_jobs"] == 1
    copied = tmp_path / "copied_back"
    root.rename(copied)
    assert finalizer.verify_archive(
        copied, expected_manifest_sha256=digest
    )["status"] == "verified"
    (copied / "prepared_plan.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(finalizer.FinalizationError, match="missing or changed"):
        finalizer.verify_archive(copied, expected_manifest_sha256=digest)


def test_route_phase_and_hash_bound_model_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preparer, "offline_preflight", lambda *_args, **_kwargs: {})
    source_commit = "1" * 40
    scopes = preparer._route_scopes(MATRIX)
    assert {name: len(rows) for name, rows in scopes.items()} == {
        "cost_model_training": 76,
        "tier_b_training": 20,
        "cost_model_holdout": 29,
        "tier_b_holdout": 10,
    }
    setup = tmp_path / "shared_setup.sh"
    lock = tmp_path / "shared_environment.lock"
    setup.write_text("export TEST_REVIEWED_ENV=1\n", encoding="utf-8")
    lock.write_text("shared-lock-v2\n", encoding="utf-8")

    def write_training_manifest(
        scope_name: str, *, first_job_id: int
    ) -> tuple[Path, dict[str, object]]:
        scope_rows = scopes[scope_name]
        root = tmp_path / scope_name
        root.mkdir()
        plan = root / "prepared_plan.csv"
        with plan.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scope_rows[0]))
            writer.writeheader()
            writer.writerows(scope_rows)
        commands = root / "sbatch_commands.txt"
        commands.write_text("fixture\n", encoding="utf-8")
        source_freeze = root / "reviewed_source_freeze.json"
        source_freeze.write_text("{}\n", encoding="utf-8")
        (root / "submitted_jobs.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "case_id": row["case_id"],
                        "returncode": 0,
                        "job_id": str(first_job_id + index),
                    }
                )
                + "\n"
                for index, row in enumerate(scope_rows)
            ),
            encoding="utf-8",
        )
        (root / "submission_journal.jsonl").write_text("", encoding="utf-8")
        environment = preparer._prepare_environment_contract(
            out_root=root, env_setup=setup, env_lock=lock
        )
        is_tier_b = scope_name == "tier_b_training"
        manifest = root / "prepared_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "status": "submitted",
                    "test_only_commands": False,
                    "route_phase": "training",
                    "selected_experiments": ["EXP-ROUTE-001"],
                    "selected_tiers": sorted({row["tier"] for row in scope_rows}),
                    "include_optional": is_tier_b,
                    "only_optional": is_tier_b,
                    "case_count": len(scope_rows),
                    "route_phase_case_ids_sha256": preparer._case_ids_sha256(
                        row["case_id"] for row in scope_rows
                    ),
                    "matrix_sha256": preparer._sha256(MATRIX),
                    "source_commit": source_commit,
                    "source_dirty": False,
                    "plan_file": plan.name,
                    "plan_sha256": preparer._sha256(plan),
                    "commands_file": commands.name,
                    "queued_source_freeze": {"path": source_freeze.name},
                    "environment_contract": environment,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest, environment

    cost_manifest, environment = write_training_manifest(
        "cost_model_training", first_job_id=1
    )
    tier_b_manifest, tier_b_environment = write_training_manifest(
        "tier_b_training", first_job_id=1001
    )
    assert environment["setup_sha256"] == tier_b_environment["setup_sha256"]
    assert environment["lock_sha256"] == tier_b_environment["lock_sha256"]

    fit_root = tmp_path / "training_fit"
    fit_root.mkdir()
    analysis = fit_root / "analysis.json"
    model = fit_root / "model.json"
    contract = json.loads(preparer.ROUTE_ANALYSIS_CONTRACT.read_text(encoding="utf-8"))
    feature_order = list(contract["cost_model"]["features_in_order"])
    training_row_ids = [f"training-row-{index:03d}" for index in range(74)]
    case_ids = sorted(row["case_id"] for row in scopes["cost_model_training"])
    model.write_text(
        json.dumps(
            {
                "schema_id": freezer.FROZEN_MODEL_SCHEMA_ID,
                "schema_version": 1,
                "status": "frozen_before_holdout",
                "experiment_id": "EXP-ROUTE-001",
                "holdout_rows_seen": 0,
                "matrix_sha256": preparer._sha256(MATRIX),
                "source_commit": source_commit,
                "contract_sha256": preparer._sha256(preparer.ROUTE_ANALYSIS_CONTRACT),
                "training_case_ids_sha256": preparer._case_ids_sha256(case_ids),
                "training_rows": 74,
                "training_row_ids": training_row_ids,
                "training_row_ids_sha256": preparer._case_ids_sha256(training_row_ids),
                "feature_order": feature_order,
                "coefficients": {name: 0.0 for name in feature_order},
                "design_diagnostics": {
                    "rows": 74,
                    "columns": len(feature_order),
                    "rank": len(feature_order),
                    "condition_number": 1.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    analysis.write_text(
        json.dumps(
            {
                "schema_id": freezer.TRAINING_ANALYSIS_SCHEMA_ID,
                "schema_version": 1,
                "status": "training_fit_admitted",
                "experiment_id": "EXP-ROUTE-001",
                "holdout_rows_seen": 0,
                "matrix_sha256": preparer._sha256(MATRIX),
                "source_commit": source_commit,
                "contract": {
                    "sha256": preparer._sha256(preparer.ROUTE_ANALYSIS_CONTRACT)
                },
                "training_case_count": len(case_ids),
                "training_case_ids": case_ids,
                "training_case_ids_sha256": preparer._case_ids_sha256(case_ids),
                "training_row_count": 74,
                "training_row_ids": training_row_ids,
                "training_row_ids_sha256": preparer._case_ids_sha256(training_row_ids),
                "karolina_training_campaign": {
                    "route_phase": "training",
                    "case_ids": case_ids,
                },
                "frozen_model": {
                    "path": model.name,
                    "sha256": preparer._sha256(model),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = tmp_path / "freeze.json"
    payload = {
        "schema_id": preparer.MODEL_FREEZE_SCHEMA_ID,
        "schema_version": preparer.MODEL_FREEZE_SCHEMA_VERSION,
        "status": "frozen_before_holdout",
        "decision": "cost_model_fit_and_tier_b_training_complete_holdouts_unopened",
        "matrix_sha256": preparer._sha256(MATRIX),
        "source_commit": source_commit,
        "scopes": preparer._scope_receipt(scopes),
        "environment_identity": {
            "setup_sha256": environment["setup_sha256"],
            "lock_sha256": environment["lock_sha256"],
        },
        "created_at_utc": "2026-07-10T12:00:00+00:00",
        "reviewer": "test-reviewer",
        "cost_model_training_manifest": {
            "path": str(cost_manifest.relative_to(tmp_path)),
            "sha256": preparer._sha256(cost_manifest),
        },
        "tier_b_training_manifest": {
            "path": str(tier_b_manifest.relative_to(tmp_path)),
            "sha256": preparer._sha256(tier_b_manifest),
        },
        "training_analysis": {
            "path": str(analysis.relative_to(tmp_path)),
            "sha256": preparer._sha256(analysis),
        },
        "frozen_model": {
            "path": str(model.relative_to(tmp_path)),
            "sha256": preparer._sha256(model),
        },
    }
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    validated = preparer._validate_model_freeze_receipt(
        receipt, matrix=MATRIX, source_commit=source_commit
    )
    archive = tmp_path / "holdout"
    archive.mkdir()
    metadata = preparer._archive_model_freeze_receipt(validated, out_root=archive)
    assert metadata["schema_version"] == preparer.MODEL_FREEZE_SCHEMA_VERSION
    assert (archive / metadata["path"]).is_file()
    preparer._validate_model_freeze_receipt(
        archive / metadata["path"], matrix=MATRIX, source_commit=source_commit
    )
    holdout_rows = scopes["cost_model_holdout"]
    holdout_plan = archive / "prepared_plan.csv"
    with holdout_plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(holdout_rows[0]))
        writer.writeheader()
        writer.writerows(holdout_rows)
    holdout_environment = preparer._prepare_environment_contract(
        out_root=archive, env_setup=setup, env_lock=lock
    )
    holdout_manifest = {
        "route_model_freeze": metadata,
        "source_commit": source_commit,
        "matrix_sha256": preparer._sha256(MATRIX),
        "route_phase": "holdout",
        "include_optional": False,
        "only_optional": False,
        "case_count": len(holdout_rows),
        "route_phase_case_ids_sha256": preparer._case_ids_sha256(
            row["case_id"] for row in holdout_rows
        ),
        "plan_file": holdout_plan.name,
        "environment_contract": holdout_environment,
    }
    preparer._validate_archived_model_freeze(
        archive, holdout_manifest, matrix=MATRIX
    )
    incomplete_holdout = dict(holdout_manifest)
    incomplete_holdout["case_count"] = 28
    with pytest.raises(RuntimeError, match="canonical cost_model_holdout"):
        preparer._validate_archived_model_freeze(
            archive, incomplete_holdout, matrix=MATRIX
        )
    changed_matrix_holdout = dict(holdout_manifest)
    changed_matrix_holdout["matrix_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="frozen matrix identity"):
        preparer._validate_archived_model_freeze(
            archive, changed_matrix_holdout, matrix=MATRIX
        )
    payload["scopes"]["cost_model_holdout"]["case_ids_sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="scope"):
        preparer._validate_model_freeze_receipt(
            receipt, matrix=MATRIX, source_commit=source_commit
        )


def test_resume_journal_blocks_ambiguous_pending_intent(tmp_path: Path) -> None:
    journal = tmp_path / "submission_journal.jsonl"
    journal.write_text(
        json.dumps(
            {
                "event": "intent",
                "attempt_id": "initial-1-case",
                "case_id": "case",
                "command": "sbatch case",
                "recorded_at_utc": "2026-07-10T10:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    accepted, pending = resume._journal_state(tmp_path)
    assert accepted == {}
    assert pending == {"initial-1-case"}


def test_resume_lock_rejects_concurrent_submitter(tmp_path: Path) -> None:
    with resume._exclusive_resume_lock(tmp_path):
        with pytest.raises(RuntimeError, match="holds the campaign lock"):
            with resume._exclusive_resume_lock(tmp_path):
                pass


def test_environment_contract_is_archived_and_hash_bound(tmp_path: Path) -> None:
    setup = tmp_path / "setup.sh"
    lock = tmp_path / "environment.lock"
    setup.write_text("export TEST_ENV=1\n", encoding="utf-8")
    lock.write_text("lock-v1\n", encoding="utf-8")
    root = tmp_path / "archive"
    root.mkdir()
    contract = preparer._prepare_environment_contract(
        out_root=root, env_setup=setup, env_lock=lock
    )
    assert contract["status"] == "hash_bound"
    assert contract["setup_sha256"] == hashlib.sha256(setup.read_bytes()).hexdigest()
    assert (root / contract["archived_setup"]["path"]).is_file()

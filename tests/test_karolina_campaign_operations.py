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
    rows = preparer.read_matrix(MATRIX)
    training = preparer.select_rows(
        rows,
        experiments={"EXP-ROUTE-001"},
        include_optional=False,
        tiers={"fixed_state_screen", "factorized_quadrature", "factorized_microbenchmark"},
        route_phase="training",
    )
    holdout = preparer.select_rows(
        rows,
        experiments={"EXP-ROUTE-001"},
        include_optional=False,
        tiers={"fixed_state_screen", "factorized_quadrature", "factorized_microbenchmark"},
        route_phase="holdout",
    )
    assert len(training) == 76
    assert len(holdout) == 29
    assert all(int(row["total_ranks"]) in {1, 8} for row in training)
    assert all(int(row["total_ranks"]) == 32 for row in holdout)

    training_root = tmp_path / "training"
    training_root.mkdir()
    plan = training_root / "prepared_plan.csv"
    with plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(training[0]))
        writer.writeheader()
        writer.writerows(training)
    source_commit = "1" * 40
    commands = training_root / "sbatch_commands.txt"
    source_freeze = training_root / "reviewed_source_freeze.json"
    commands.write_text("fixture\n", encoding="utf-8")
    source_freeze.write_text("{}\n", encoding="utf-8")
    (training_root / "submitted_jobs.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "case_id": row["case_id"],
                    "returncode": 0,
                    "job_id": str(index + 1),
                }
            )
            + "\n"
            for index, row in enumerate(training)
        ),
        encoding="utf-8",
    )
    (training_root / "submission_journal.jsonl").write_text("", encoding="utf-8")
    training_manifest = training_root / "prepared_manifest.json"
    training_manifest.write_text(
        json.dumps(
            {
                "status": "submitted",
                "route_phase": "training",
                "selected_experiments": ["EXP-ROUTE-001"],
                "matrix_sha256": preparer._sha256(MATRIX),
                "source_commit": source_commit,
                    "source_dirty": False,
                    "plan_file": plan.name,
                    "commands_file": commands.name,
                    "queued_source_freeze": {"path": source_freeze.name},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    analysis = training_root / "analysis.json"
    model = training_root / "model.json"
    contract = json.loads(preparer.ROUTE_ANALYSIS_CONTRACT.read_text(encoding="utf-8"))
    feature_order = list(contract["cost_model"]["features_in_order"])
    training_row_ids = [f"training-row-{index:03d}" for index in range(74)]
    case_ids = sorted(row["case_id"] for row in training)
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
    receipt = training_root / "freeze.json"
    payload = {
        "schema_id": preparer.MODEL_FREEZE_SCHEMA_ID,
        "schema_version": 1,
        "status": "frozen_before_holdout",
        "decision": "training_fit_complete_holdout_unopened",
        "matrix_sha256": preparer._sha256(MATRIX),
        "source_commit": source_commit,
        "training_case_ids_sha256": preparer._case_ids_sha256(
            row["case_id"] for row in training
        ),
        "created_at_utc": "2026-07-10T12:00:00+00:00",
        "reviewer": "test-reviewer",
        "training_manifest": {
            "path": training_manifest.name,
            "sha256": preparer._sha256(training_manifest),
        },
        "training_analysis": {
            "path": analysis.name,
            "sha256": preparer._sha256(analysis),
        },
        "frozen_model": {"path": model.name, "sha256": preparer._sha256(model)},
    }
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    validated = preparer._validate_model_freeze_receipt(
        receipt, matrix=MATRIX, source_commit=source_commit
    )
    archive = tmp_path / "holdout"
    archive.mkdir()
    metadata = preparer._archive_model_freeze_receipt(validated, out_root=archive)
    assert (archive / metadata["path"]).is_file()
    preparer._validate_model_freeze_receipt(
        archive / metadata["path"], matrix=MATRIX, source_commit=source_commit
    )
    payload["training_case_ids_sha256"] = "0" * 64
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

#!/usr/bin/env python3
"""Fail-closed resume of a journaled partial Karolina submission.

The default is a read-only report.  ``--execute`` is required to invoke
``sbatch`` and reuses only commands whose case IDs have no accepted job ID.
Any intent without a matching result requires manual scheduler reconciliation
and blocks automatic resume.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.runners.paper_revision_karolina import prepare_campaign as campaign


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path} contains a non-object record")
            rows.append(value)
    return rows


def _journal_state(root: Path) -> tuple[dict[str, str], set[str]]:
    intents: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    for event in _jsonl(root / "submission_journal.jsonl"):
        attempt_id = str(event.get("attempt_id", ""))
        kind = event.get("event")
        if not attempt_id or kind not in {"intent", "result"}:
            raise RuntimeError("submission journal contains an invalid event")
        try:
            recorded = datetime.fromisoformat(
                str(event.get("recorded_at_utc", "")).replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise RuntimeError("submission journal contains an invalid timestamp") from exc
        if (
            recorded.tzinfo is None
            or recorded.utcoffset() is None
            or recorded.utcoffset() != timezone.utc.utcoffset(recorded)
        ):
            raise RuntimeError("submission journal timestamps must be UTC")
        target = intents if kind == "intent" else results
        if attempt_id in target:
            raise RuntimeError("submission journal contains a duplicate event")
        target[attempt_id] = event
    pending = set(intents).difference(results)
    accepted_from_journal: dict[str, str] = {}
    for attempt_id, result in results.items():
        intent = intents.get(attempt_id)
        if intent is None or result.get("case_id") != intent.get("case_id"):
            raise RuntimeError("submission journal result lacks its matching intent")
        intent_time = datetime.fromisoformat(
            str(intent["recorded_at_utc"]).replace("Z", "+00:00")
        )
        result_time = datetime.fromisoformat(
            str(result["recorded_at_utc"]).replace("Z", "+00:00")
        )
        if result_time < intent_time:
            raise RuntimeError("submission journal result predates its intent")
        if int(result.get("returncode", 1)) == 0:
            case_id = str(result.get("case_id", ""))
            job_id = str(result.get("job_id", ""))
            if case_id in accepted_from_journal or not job_id.isdigit():
                raise RuntimeError("submission journal has duplicate or invalid acceptance")
            accepted_from_journal[case_id] = job_id
    return accepted_from_journal, pending


def _accepted_ledger(root: Path) -> dict[str, str]:
    accepted: dict[str, str] = {}
    for record in _jsonl(root / "submitted_jobs.jsonl"):
        case_id = str(record.get("case_id", ""))
        job_id = str(record.get("job_id", ""))
        if (
            not case_id
            or case_id in accepted
            or int(record.get("returncode", 1)) != 0
            or not job_id.isdigit()
        ):
            raise RuntimeError("accepted-job ledger has a duplicate or invalid record")
        accepted[case_id] = job_id
    return accepted


@contextmanager
def _exclusive_resume_lock(root: Path):
    lock_path = root / "submission_resume.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another submission resume process holds the campaign lock") from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _validate_archived_decision_artifacts(
    root: Path, *, record_name: str, record: dict[str, Any]
) -> None:
    receipt_path = (root / str(record.get("path", ""))).resolve()
    try:
        receipt_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"resume {record_name} path escapes the campaign archive") from exc
    if not receipt_path.is_file() or campaign._sha256(receipt_path) != record.get(
        "sha256"
    ):
        raise RuntimeError(f"resume {record_name} is missing or changed")
    payload = _json(receipt_path)
    artifact_keys = (
        ("reviewed_artifacts",)
        if record_name == "release_authorization"
        else (
            "cost_model_training_manifest",
            "tier_b_training_manifest",
            "training_analysis",
            "frozen_model",
        )
    )
    if record_name == "release_authorization":
        artifacts = payload.get("reviewed_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise RuntimeError("resume release authorization has no reviewed artifacts")
    else:
        if (
            payload.get("schema_id") != campaign.MODEL_FREEZE_SCHEMA_ID
            or payload.get("schema_version") != campaign.MODEL_FREEZE_SCHEMA_VERSION
            or record.get("schema_version") != campaign.MODEL_FREEZE_SCHEMA_VERSION
        ):
            raise RuntimeError("resume route_model_freeze is not the supported v2 receipt")
        artifacts = [payload.get(key) for key in artifact_keys]
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
            raise RuntimeError(f"resume {record_name} contains malformed artifact metadata")
        artifact_path = (root / str(artifact["path"])).resolve()
        try:
            artifact_path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"resume {record_name} artifact escapes the archive") from exc
        if not artifact_path.is_file() or campaign._sha256(artifact_path) != artifact[
            "sha256"
        ]:
            raise RuntimeError(f"resume {record_name} reviewed artifact is missing or changed")


def _resume(root: Path, *, execute: bool = False) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "prepared_manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("status") not in {
        "partial_submission",
        "submission_failed",
        "submission_reconciliation_required",
    }:
        raise RuntimeError("resume requires a failed or partial real-submission manifest")
    if manifest.get("test_only_commands") is not False:
        raise RuntimeError("test-only admission records cannot be resumed as real jobs")
    campaign.offline_preflight(root, matrix=campaign.DEFAULT_MATRIX)
    git = campaign._git_metadata()
    if git.get("dirty") is not False or git.get("commit") != manifest.get("source_commit"):
        raise RuntimeError("resume requires the same clean source commit")
    contract = dict(manifest.get("environment_contract") or {})
    for path_key, hash_key in (
        ("runtime_setup_path", "setup_sha256"),
        ("runtime_lock_path", "lock_sha256"),
    ):
        path = Path(str(contract.get(path_key, "")))
        if not path.is_file() or campaign._sha256(path) != contract.get(hash_key):
            raise RuntimeError("resume environment setup/lock is missing or changed")
    for record_name in ("release_authorization", "route_model_freeze"):
        record = manifest.get(record_name)
        if record is None:
            continue
        if not isinstance(record, dict):
            raise RuntimeError(f"resume {record_name} metadata is malformed")
        _validate_archived_decision_artifacts(
            root, record_name=record_name, record=record
        )

    journal_accepted, pending = _journal_state(root)
    if pending:
        raise RuntimeError(
            "automatic resume blocked: an sbatch intent has no result; reconcile job IDs manually"
        )
    ledger_accepted = _accepted_ledger(root)
    if ledger_accepted != journal_accepted:
        raise RuntimeError("accepted-job ledger disagrees with the fsynced submission journal")
    with (root / str(manifest["plan_file"])).open(newline="", encoding="utf-8") as handle:
        plan = [dict(row) for row in csv.DictReader(handle)]
    command_lines = (root / str(manifest["commands_file"])).read_text(
        encoding="utf-8"
    ).splitlines()
    if len(plan) != len(command_lines):
        raise RuntimeError("resume plan and command counts differ")
    commands = {
        row["case_id"]: shlex.split(line)
        for row, line in zip(plan, command_lines, strict=True)
    }
    planned_ids = [row["case_id"] for row in plan]
    if not set(ledger_accepted).issubset(planned_ids):
        raise RuntimeError("accepted-job ledger contains an unplanned case")
    unsent = [case_id for case_id in planned_ids if case_id not in ledger_accepted]
    if not execute:
        return {
            "status": "resume_preflight_passed",
            "accepted_case_count": len(ledger_accepted),
            "unsent_case_count": len(unsent),
            "unsent_case_ids": unsent,
            "scheduler_invoked": False,
        }
    campaign._require_revalidation(test_only=False)
    accepted = len(ledger_accepted)
    existing_intents = sum(
        event.get("event") == "intent"
        for event in _jsonl(root / "submission_journal.jsonl")
    )
    for index, case_id in enumerate(unsent, start=1):
        command = commands[case_id]
        command_text = shlex.join(command)
        sequence = existing_intents + index
        attempt_id = f"resume-{sequence:08d}-{case_id}"
        campaign._append_jsonl(
            root / "submission_journal.jsonl",
            {
                "event": "intent",
                "attempt_id": attempt_id,
                "sequence": sequence,
                "case_id": case_id,
                "command": command_text,
                "recorded_at_utc": campaign._utc_now(),
            },
        )
        try:
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            result: dict[str, object] = {
                "event": "result",
                "attempt_id": attempt_id,
                "sequence": sequence,
                "case_id": case_id,
                "command": command_text,
                "recorded_at_utc": campaign._utc_now(),
                "returncode": int(completed.returncode),
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
            if int(completed.returncode) == 0:
                result["job_id"] = campaign._submitted_job_id(completed.stdout)
            campaign._append_jsonl(root / "submission_journal.jsonl", result)
        except BaseException as exc:
            manifest["status"] = "submission_reconciliation_required"
            manifest["submission_error"] = f"{type(exc).__name__}: {exc}"
            campaign._atomic_write_json(manifest_path, manifest)
            raise
        if int(completed.returncode) != 0:
            manifest["status"] = "partial_submission" if accepted else "submission_failed"
            manifest["submission_error"] = f"sbatch failed for {case_id}"
            campaign._atomic_write_json(manifest_path, manifest)
            raise RuntimeError(f"sbatch failed for {case_id}")
        accepted_record = {
            key: value
            for key, value in result.items()
            if key not in {"event", "attempt_id"}
        }
        campaign._append_jsonl(root / "submitted_jobs.jsonl", accepted_record)
        accepted += 1
        manifest["submission_progress"] = {
            "attempted": len(plan),
            "accepted": accepted,
            "total": len(plan),
            "last_case_id": case_id,
        }
        campaign._atomic_write_json(manifest_path, manifest)
    manifest["status"] = "submitted"
    manifest.pop("submission_error", None)
    campaign._atomic_write_json(manifest_path, manifest)
    return {
        "status": "submitted",
        "accepted_case_count": accepted,
        "resumed_case_count": len(unsent),
        "scheduler_invoked": bool(unsent),
    }


def resume(root: Path, *, execute: bool = False) -> dict[str, Any]:
    root = root.resolve()
    if not execute:
        return _resume(root, execute=False)
    with _exclusive_resume_lock(root):
        return _resume(root, execute=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        result = resume(args.campaign_root, execute=bool(args.execute))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

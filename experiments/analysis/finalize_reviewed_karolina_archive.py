#!/usr/bin/env python3
"""Settle and checksum a reviewed Karolina campaign from offline snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analysis.collect_slurm_accounting import collect_accounting, parse_sacct
from experiments.analysis.finalize_karolina_campaign_archive import (
    OFFLINE_INDEX_SCHEMA_ID,
    OFFLINE_INDEX_SCHEMA_VERSION,
    verify_archive,
    write_archive_checksums,
)
from experiments.runners import karolina_reviewed_campaign as contract


def _jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise contract.CampaignContractError(f"{path}:{number} is not an object")
        result.append(value)
    return result


def submitted_jobs(root: Path, plan: dict[str, Any]) -> dict[str, str]:
    records = _jsonl(root / "submitted_jobs.jsonl")
    expected = {case["case_id"] for case in plan["cases"]}
    jobs: dict[str, str] = {}
    for record in records:
        case_id = str(record.get("case_id", ""))
        job_id = str(record.get("job_id", ""))
        if case_id not in expected or case_id in jobs or not job_id.isdigit() or int(job_id) < 1:
            raise contract.CampaignContractError("submission ledger has a stale or duplicate identity")
        if int(record.get("returncode", 1)) != 0 or job_id in jobs.values():
            raise contract.CampaignContractError("submission ledger has a failed or reused job")
        jobs[case_id] = job_id
    if set(jobs) != expected:
        raise contract.CampaignContractError("submission ledger does not cover the exact plan")
    journal = _jsonl(root / "submission_journal.jsonl")
    pairs: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in expected}
    for record in journal:
        case_id = str(record.get("case_id", ""))
        event = str(record.get("event", ""))
        if case_id not in pairs or event not in {"intent", "result"}:
            raise contract.CampaignContractError("submission journal is malformed")
        pairs[case_id].append(record)
    if any(
        len(records) != 2
        or {record.get("event") for record in records} != {"intent", "result"}
        or len({record.get("attempt_id") for record in records}) != 1
        for records in pairs.values()
    ):
        raise contract.CampaignContractError("submission journal has an unresolved intent")
    return jobs


def offline_sources(index_path: Path, *, root: Path, jobs: dict[str, str]) -> dict[str, Path]:
    index_path = Path(index_path).resolve()
    index = contract.read_object(index_path)
    if (
        index.get("schema_id") != OFFLINE_INDEX_SCHEMA_ID
        or index.get("schema_version") != OFFLINE_INDEX_SCHEMA_VERSION
        or index.get("campaign_manifest_sha256")
        != contract.sha256_file(root / "prepared_manifest.json")
    ):
        raise contract.CampaignContractError("offline accounting index identity is stale")
    records = index.get("records")
    if not isinstance(records, list):
        raise contract.CampaignContractError("offline accounting records must be a list")
    sources: dict[str, Path] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"case_id", "job_id", "path", "sha256"}:
            raise contract.CampaignContractError("offline accounting record shape is invalid")
        case_id = str(record["case_id"])
        if case_id not in jobs or case_id in sources or str(record["job_id"]) != jobs[case_id]:
            raise contract.CampaignContractError("offline accounting case/job identity is stale")
        relative = Path(str(record["path"]))
        if relative.is_absolute():
            raise contract.CampaignContractError("offline accounting path must be index-relative")
        source = (index_path.parent / relative).resolve()
        try:
            source.relative_to(index_path.parent)
        except ValueError as exc:
            raise contract.CampaignContractError("offline accounting path escapes its archive") from exc
        if not source.is_file() or source.is_symlink() or contract.sha256_file(source) != record["sha256"]:
            raise contract.CampaignContractError("offline accounting snapshot is missing or stale")
        sources[case_id] = source
    if set(sources) != set(jobs):
        raise contract.CampaignContractError("offline accounting index does not cover every job")
    return sources


def _validate_job(
    *, job_root: Path, case: dict[str, Any], job_id: str, accounting: dict[str, Any],
    source_commit: str, plan_sha256: str, source_freeze_sha256: str,
) -> None:
    metadata = contract.read_object(job_root / "job_metadata.json")
    execution = contract.read_object(job_root / "execution.json")
    for retained in ("environment.json", "stdout.log", "stderr.log"):
        path = job_root / retained
        if not path.is_file() or path.is_symlink():
            raise contract.CampaignContractError(f"compute-node evidence is missing: {retained}")
    resources = metadata.get("resources")
    expected_resources = {
        "account": contract.ACCOUNT,
        "qos": contract.QOS,
        "partition": case["partition"],
        "nodes": case["nodes"],
        "total_ranks": case["total_ranks"],
    }
    if (
        metadata.get("case_id") != case["case_id"]
        or metadata.get("job_id") != job_id
        or metadata.get("source_commit") != source_commit
        or metadata.get("plan_sha256") != plan_sha256
        or metadata.get("source_freeze_sha256") != source_freeze_sha256
        or not isinstance(resources, dict)
        or any(resources.get(key) != value for key, value in expected_resources.items())
        or execution.get("case_id") != case["case_id"]
        or execution.get("job_id") != job_id
        or int(execution.get("returncode", 1)) != 0
    ):
        raise contract.CampaignContractError("compute-node job identity or exit status is invalid")
    output_hashes = execution.get("output_hashes")
    if not isinstance(output_hashes, dict) or set(output_hashes) != set(case["expected_outputs"]):
        raise contract.CampaignContractError("execution output inventory differs from the plan")
    for raw, expected in output_hashes.items():
        output = (job_root / raw).resolve()
        try:
            output.relative_to(job_root)
        except ValueError as exc:
            raise contract.CampaignContractError("execution output escapes the job archive") from exc
        if not output.is_file() or output.is_symlink() or contract.sha256_file(output) != expected:
            raise contract.CampaignContractError("execution output is missing or stale")
    source = dict(accounting.get("source") or {})
    raw = source.get("raw_parsable2")
    if not isinstance(raw, str) or not raw:
        raise contract.CampaignContractError("settled accounting lacks raw parsable2 evidence")
    reparsed = parse_sacct(raw, job_id=job_id)
    for key in ("job_id", "allocation", "rows", "derived"):
        if accounting.get(key) != reparsed[key]:
            raise contract.CampaignContractError("settled accounting differs from raw evidence")
    allocation = accounting["allocation"]
    exact = {
        "job_id_raw": job_id,
        "cluster": "karolina",
        "account": contract.ACCOUNT,
        "qos": contract.QOS,
        "partition": case["partition"],
        "state": "COMPLETED",
        "exit_code": "0:0",
        "alloc_nodes": int(case["nodes"]),
        "alloc_cpus": int(case["total_ranks"]),
    }
    for key, expected in exact.items():
        actual = allocation.get(key)
        if key == "cluster":
            actual = str(actual).lower()
        if actual != expected:
            raise contract.CampaignContractError(f"accounting {key} differs from the plan")


def finalize(root: Path, *, offline_index: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest, plan = contract.load_plan(root)
    if manifest.get("status") != "submitted" or manifest.get("scheduler_contact") is not True:
        raise contract.CampaignContractError("archive settlement requires a submitted campaign")
    jobs = submitted_jobs(root, plan)
    sources = offline_sources(offline_index, root=root, jobs=jobs)
    cases = {case["case_id"]: case for case in plan["cases"]}
    plan_sha256 = str(manifest["plan"]["sha256"])
    source_freeze_sha256 = str(manifest["source_freeze"]["sha256"])
    for case_id, job_id in jobs.items():
        job_root = root / "jobs" / case_id / f"job_{job_id}"
        if not job_root.is_dir() or job_root.is_symlink():
            raise contract.CampaignContractError(f"job archive is missing for {case_id}")
        raw_path = job_root / "sacct_raw.parsable2"
        shutil.copy2(sources[case_id], raw_path)
        accounting = collect_accounting(job_id=job_id, sacct_file=raw_path)
        accounting["source"]["path"] = raw_path.name
        _validate_job(
            job_root=job_root,
            case=cases[case_id],
            job_id=job_id,
            accounting=accounting,
            source_commit=str(plan["source_commit"]),
            plan_sha256=plan_sha256,
            source_freeze_sha256=source_freeze_sha256,
        )
        contract.atomic_json(job_root / "sacct_final.json", accounting)
    checksum = write_archive_checksums(root)
    return {
        "status": "settled_and_checksums_written",
        "source_mode": "offline_index",
        "settled_jobs": len(jobs),
        "archive_checksums": checksum,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--campaign-root", type=Path, required=True)
    result.add_argument("--offline-index", type=Path)
    result.add_argument("--verify-only", action="store_true")
    result.add_argument("--expected-checksum-manifest-sha256")
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        if args.verify_only:
            if args.offline_index is not None or not args.expected_checksum_manifest_sha256:
                raise contract.CampaignContractError("verify-only requires exactly the checksum digest")
            result = verify_archive(
                args.campaign_root,
                expected_manifest_sha256=args.expected_checksum_manifest_sha256,
            )
        else:
            if args.offline_index is None:
                raise contract.CampaignContractError("offline settlement requires --offline-index")
            result = finalize(args.campaign_root, offline_index=args.offline_index)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

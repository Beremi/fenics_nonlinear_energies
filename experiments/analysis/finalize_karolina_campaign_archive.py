#!/usr/bin/env python3
"""Bulk-settle and checksum one submitted Karolina campaign archive.

The default data source is an explicit JSON index of already captured
``sacct --parsable2`` files.  Live accounting is available only with
``--query-live``.  This program never submits work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analysis.collect_slurm_accounting import collect_accounting, parse_sacct
from experiments.runners.paper_revision_karolina import prepare_campaign as campaign
from experiments.runners.paper_revision_karolina import resume_partial_submission as resumer
from src.core.benchmark.run_record import atomic_write_json


SCHEMA_ID = "fenics-nonlinear-energies.karolina-archive-checksums"
SCHEMA_VERSION = 1
OFFLINE_INDEX_SCHEMA_ID = "fenics-nonlinear-energies.offline-accounting-index"
OFFLINE_INDEX_SCHEMA_VERSION = 1
CHECKSUM_NAME = "campaign_archive_checksums.json"
_SUBMITTED = re.compile(r"Submitted batch job ([1-9][0-9]*)")


class FinalizationError(ValueError):
    """Campaign settlement or archive integrity failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise FinalizationError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise FinalizationError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _submitted_jobs(root: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    manifest = _read_object(root / "prepared_manifest.json")
    if manifest.get("status") != "submitted":
        raise FinalizationError("bulk settlement requires a completed submitted manifest")
    plan_path = root / str(manifest.get("plan_file", ""))
    with plan_path.open(newline="", encoding="utf-8") as handle:
        plan = [dict(row) for row in csv.DictReader(handle)]
    expected = {row["case_id"]: row for row in plan}
    ledger = _read_jsonl(root / "submitted_jobs.jsonl")
    jobs: dict[str, str] = {}
    for record in ledger:
        case_id = str(record.get("case_id", ""))
        if case_id not in expected or case_id in jobs:
            raise FinalizationError("submission ledger has an unknown or duplicate case ID")
        if int(record.get("returncode", 1)) != 0:
            raise FinalizationError("submission ledger contains a failed scheduler response")
        job_id = str(record.get("job_id", ""))
        if not job_id:
            match = _SUBMITTED.fullmatch(str(record.get("stdout", "")).strip())
            job_id = "" if match is None else match.group(1)
        if not job_id.isdigit() or int(job_id) <= 0:
            raise FinalizationError(f"submission ledger lacks a job ID for {case_id}")
        if job_id in jobs.values():
            raise FinalizationError("submission ledger reuses a Slurm job ID")
        jobs[case_id] = job_id
    if set(jobs) != set(expected):
        raise FinalizationError("submission ledger does not cover the prepared plan")
    return jobs, expected


def _validate_submission_journal(root: Path, jobs: dict[str, str]) -> None:
    journal_jobs, pending = resumer._journal_state(root)
    if pending:
        raise FinalizationError(
            "submission journal contains an unresolved scheduler intent"
        )
    ledger_jobs = resumer._accepted_ledger(root)
    if journal_jobs != ledger_jobs or ledger_jobs != jobs:
        raise FinalizationError(
            "submission journal, accepted ledger, and prepared plan disagree"
        )


def _offline_sources(
    index_path: Path, *, root: Path, jobs: dict[str, str]
) -> dict[str, Path]:
    index_path = index_path.resolve()
    payload = _read_object(index_path)
    if (
        payload.get("schema_id") != OFFLINE_INDEX_SCHEMA_ID
        or payload.get("schema_version") != OFFLINE_INDEX_SCHEMA_VERSION
        or payload.get("campaign_manifest_sha256")
        != _sha256(root / "prepared_manifest.json")
    ):
        raise FinalizationError("offline accounting index identity or manifest hash is stale")
    records = payload.get("records")
    if not isinstance(records, list):
        raise FinalizationError("offline accounting index records must be a list")
    sources: dict[str, Path] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "case_id",
            "job_id",
            "path",
            "sha256",
        }:
            raise FinalizationError("offline accounting record has an invalid shape")
        case_id = str(record["case_id"])
        if case_id not in jobs or case_id in sources or str(record["job_id"]) != jobs[case_id]:
            raise FinalizationError("offline accounting record has a stale case/job identity")
        relative = Path(str(record["path"]))
        if relative.is_absolute():
            raise FinalizationError("offline accounting paths must be index-relative")
        source = (index_path.parent / relative).resolve()
        try:
            source.relative_to(index_path.parent.resolve())
        except ValueError as exc:
            raise FinalizationError("offline accounting path escapes its snapshot archive") from exc
        if not source.is_file() or record["sha256"] != _sha256(source):
            raise FinalizationError("offline accounting snapshot is missing or has a stale hash")
        sources[case_id] = source
    if set(sources) != set(jobs):
        raise FinalizationError("offline accounting index does not cover every submitted job")
    return sources


def _validate_accounting(
    payload: dict[str, Any], *, job_id: str, row: dict[str, str]
) -> None:
    source = dict(payload.get("source") or {})
    raw = source.get("raw_parsable2")
    if not isinstance(raw, str) or not raw:
        raise FinalizationError("settled accounting lacks raw parsable2 evidence")
    reparsed = parse_sacct(raw, job_id=job_id)
    for key in ("job_id", "allocation", "rows", "derived"):
        if payload.get(key) != reparsed[key]:
            raise FinalizationError(f"settled accounting {key} disagrees with raw evidence")
    allocation = dict(payload["allocation"])
    exact: dict[str, Any] = {
        "job_id_raw": job_id,
        "account": "fta-26-40",
        "qos": "3571_6328",
        "partition": row["partition"],
        "state": "COMPLETED",
        "exit_code": "0:0",
        "alloc_nodes": int(row["nodes"]),
        "alloc_cpus": int(row["total_ranks"]),
    }
    for key, expected in exact.items():
        if allocation.get(key) != expected:
            raise FinalizationError(f"settled accounting {key} differs from the matrix")
    if str(allocation.get("cluster", "")).lower() != "karolina":
        raise FinalizationError("settled accounting is not from Karolina")


def _archive_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise FinalizationError(f"campaign archive contains a forbidden symlink: {path}")
        if path.is_file() and path != root / CHECKSUM_NAME:
            files.append(path)
    return files


def write_archive_checksums(root: Path) -> dict[str, Any]:
    root = root.resolve()
    records = [
        {
            "path": str(path.relative_to(root)),
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
        }
        for path in _archive_files(root)
    ]
    payload = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "prepared_manifest_sha256": _sha256(root / "prepared_manifest.json"),
        "file_count": len(records),
        "files": records,
    }
    destination = root / CHECKSUM_NAME
    atomic_write_json(destination, payload)
    verify_archive(root, expected_manifest_sha256=_sha256(destination))
    return {
        "path": destination.name,
        "sha256": _sha256(destination),
        "file_count": len(records),
    }


def verify_archive(root: Path, *, expected_manifest_sha256: str) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / CHECKSUM_NAME
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise FinalizationError("archive checksum manifest digest differs after copy-back")
    payload = _read_object(manifest_path)
    if payload.get("schema_id") != SCHEMA_ID or payload.get("schema_version") != SCHEMA_VERSION:
        raise FinalizationError("archive checksum manifest schema is invalid")
    prepared_manifest = root / "prepared_manifest.json"
    if (
        not prepared_manifest.is_file()
        or payload.get("prepared_manifest_sha256") != _sha256(prepared_manifest)
    ):
        raise FinalizationError("archive checksum manifest has a stale campaign identity")
    records = payload.get("files")
    if not isinstance(records, list) or int(payload.get("file_count", -1)) != len(records):
        raise FinalizationError("archive checksum manifest file count is invalid")
    expected_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes"}:
            raise FinalizationError("archive checksum entry has an invalid shape")
        relative = Path(str(record["path"]))
        if relative.is_absolute() or str(relative) in expected_paths:
            raise FinalizationError("archive checksum entry path is absolute or duplicated")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise FinalizationError("archive checksum entry escapes the archive") from exc
        if (
            not path.is_file()
            or path.is_symlink()
            or _sha256(path) != record["sha256"]
            or int(path.stat().st_size) != int(record["bytes"])
        ):
            raise FinalizationError(f"archive file is missing or changed: {relative}")
        expected_paths.add(str(relative))
    actual_paths = {str(path.relative_to(root)) for path in _archive_files(root)}
    if actual_paths != expected_paths:
        raise FinalizationError("archive has missing or unindexed files after copy-back")
    return {
        "status": "verified",
        "checksum_manifest_sha256": expected_manifest_sha256,
        "file_count": len(records),
    }


def finalize(
    root: Path,
    *,
    offline_index: Path | None = None,
    query_live: bool = False,
    sacct_executable: str = "sacct",
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    if (offline_index is None) == (not query_live):
        raise FinalizationError("select exactly one of offline_index or query_live")
    try:
        campaign.offline_preflight(root, matrix=campaign.DEFAULT_MATRIX)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"campaign offline preflight failed: {exc}") from exc
    jobs, rows = _submitted_jobs(root)
    manifest = _read_object(root / "prepared_manifest.json")
    if (
        manifest.get("test_only_commands") is not False
        or manifest.get("source_dirty") is not False
        or dict(manifest.get("environment_contract") or {}).get("status")
        != "hash_bound"
    ):
        raise FinalizationError(
            "archive settlement requires real jobs from a clean, hash-bound environment"
        )
    _validate_submission_journal(root, jobs)
    sources = (
        _offline_sources(offline_index, root=root, jobs=jobs)
        if offline_index is not None
        else {}
    )
    for case_id, job_id in jobs.items():
        batch = root / "jobs" / case_id / f"job_{job_id}"
        if not batch.is_dir():
            raise FinalizationError(f"compute-node evidence directory is missing for {case_id}")
        if offline_index is not None:
            copied_raw = batch / "sacct_raw.parsable2"
            shutil.copy2(sources[case_id], copied_raw)
            payload = collect_accounting(job_id=job_id, sacct_file=copied_raw)
            payload["source"]["path"] = copied_raw.name
        else:
            kwargs: dict[str, Any] = {}
            if runner is not None:
                kwargs["runner"] = runner
            payload = collect_accounting(
                job_id=job_id,
                query_live=True,
                executable=sacct_executable,
                **kwargs,
            )
        _validate_accounting(payload, job_id=job_id, row=rows[case_id])
        atomic_write_json(batch / "sacct_final.json", payload)
    checksum = write_archive_checksums(root)
    return {
        "status": "settled_and_checksums_written",
        "source_mode": "offline_index" if offline_index is not None else "explicit_live_query",
        "settled_jobs": len(jobs),
        "archive_checksums": checksum,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline-index", type=Path)
    mode.add_argument("--query-live", action="store_true")
    parser.add_argument("--sacct-executable", default="sacct")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--expected-checksum-manifest-sha256")
    parser.add_argument("--receipt", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.verify_only:
            if args.offline_index is not None or args.query_live:
                raise FinalizationError("verify-only mode does not accept an accounting source")
            expected = str(args.expected_checksum_manifest_sha256 or "")
            if len(expected) != 64:
                raise FinalizationError("verify-only mode requires the pre-copy checksum digest")
            result = verify_archive(args.campaign_root, expected_manifest_sha256=expected)
        else:
            result = finalize(
                args.campaign_root,
                offline_index=args.offline_index,
                query_live=bool(args.query_live),
                sacct_executable=str(args.sacct_executable),
            )
        rendered = json.dumps(result, indent=2) + "\n"
        if args.receipt is not None:
            receipt = Path(args.receipt).resolve()
            root = Path(args.campaign_root).resolve()
            if receipt == root or root in receipt.parents:
                raise FinalizationError(
                    "detached archive receipt must be outside the campaign root"
                )
            receipt.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(receipt, result)
        print(rendered, end="")
    except (FinalizationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

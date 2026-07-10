#!/usr/bin/env python3
"""Build a deterministic offline index for captured ``sacct`` snapshots.

The utility never invokes Slurm.  It requires one exact ``<job-id>.sacct``
file per accepted submission, reparses every file, checks its terminal Karolina
allocation against the frozen campaign row, rejects every missing/additional/
symlinked snapshot, and emits the index consumed by the archive finalizers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analysis.collect_slurm_accounting import parse_sacct
from experiments.analysis import finalize_karolina_campaign_archive as legacy_finalizer
from experiments.analysis import finalize_reviewed_karolina_archive as reviewed_finalizer
from experiments.runners import karolina_reviewed_campaign as reviewed


SCHEMA_ID = legacy_finalizer.OFFLINE_INDEX_SCHEMA_ID
SCHEMA_VERSION = legacy_finalizer.OFFLINE_INDEX_SCHEMA_VERSION


class IndexGenerationError(ValueError):
    """Snapshot coverage, identity, or allocation evidence is invalid."""


def _campaign_scope(
    root: Path,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    manifest = reviewed.read_object(root / "prepared_manifest.json")
    if manifest.get("schema_id") == reviewed.MANIFEST_SCHEMA_ID:
        validated_manifest, plan = reviewed.load_plan(root)
        if (
            validated_manifest.get("status") != "submitted"
            or validated_manifest.get("scheduler_contact") is not True
        ):
            raise IndexGenerationError(
                "offline accounting indexing requires a fully submitted campaign"
            )
        jobs = reviewed_finalizer.submitted_jobs(root, plan)
        rows = {str(case["case_id"]): dict(case) for case in plan["cases"]}
        return jobs, rows

    try:
        legacy_finalizer.campaign.offline_preflight(
            root, matrix=legacy_finalizer.campaign.DEFAULT_MATRIX
        )
        jobs, rows = legacy_finalizer._submitted_jobs(root)
        legacy_finalizer._validate_submission_journal(root, jobs)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise IndexGenerationError(
            f"legacy Karolina campaign preflight failed: {exc}"
        ) from exc
    return jobs, {case_id: dict(row) for case_id, row in rows.items()}


def _expected_resources(row: dict[str, Any]) -> dict[str, Any]:
    try:
        return {
            "job_name": str(row["case_id"]),
            "cluster": "karolina",
            "account": reviewed.ACCOUNT,
            "partition": str(row["partition"]),
            "qos": reviewed.QOS,
            "state": "COMPLETED",
            "exit_code": "0:0",
            "alloc_nodes": int(row["nodes"]),
            "alloc_cpus": int(row["total_ranks"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise IndexGenerationError("campaign row has an invalid resource contract") from exc


def _validate_snapshot(
    path: Path, *, case_id: str, job_id: str, row: dict[str, Any]
) -> None:
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = parse_sacct(raw, job_id=job_id)
    except (OSError, UnicodeError, ValueError) as exc:
        raise IndexGenerationError(
            f"raw accounting snapshot is invalid for {case_id}/{job_id}: {exc}"
        ) from exc
    allocation = dict(parsed["allocation"])
    expected = _expected_resources(row)
    for key, value in expected.items():
        actual = allocation.get(key)
        if key == "cluster":
            actual = str(actual).lower()
        if actual != value:
            raise IndexGenerationError(
                f"raw accounting {key} differs from the campaign for {case_id}"
            )


def _paths(
    *, campaign_root: Path, snapshot_root: Path, output: Path, verify: bool
) -> tuple[Path, Path, Path]:
    campaign_root = Path(campaign_root).resolve()
    snapshot_root = Path(snapshot_root).resolve()
    output = Path(output).resolve()
    if not campaign_root.is_dir() or campaign_root.is_symlink():
        raise IndexGenerationError("campaign root is missing or symlinked")
    if not snapshot_root.is_dir() or snapshot_root.is_symlink():
        raise IndexGenerationError("snapshot root is missing or symlinked")
    if snapshot_root == campaign_root or campaign_root in snapshot_root.parents:
        raise IndexGenerationError(
            "snapshot root must be detached from the campaign archive"
        )
    if output.parent != snapshot_root or output.name.endswith(".sacct"):
        raise IndexGenerationError(
            "offline index output must be a non-.sacct file directly in snapshot root"
        )
    if output.is_symlink():
        raise IndexGenerationError("offline index output may not be a symlink")
    if verify and not output.is_file():
        raise IndexGenerationError("verify mode requires an existing regular index")
    if not verify and output.exists():
        raise IndexGenerationError("generation refuses to overwrite an existing index")
    return campaign_root, snapshot_root, output


def build_payload(
    *, campaign_root: Path, snapshot_root: Path, output: Path, verify: bool = False
) -> dict[str, Any]:
    campaign_root, snapshot_root, output = _paths(
        campaign_root=campaign_root,
        snapshot_root=snapshot_root,
        output=output,
        verify=verify,
    )
    jobs, rows = _campaign_scope(campaign_root)
    expected_names = {f"{job_id}.sacct" for job_id in jobs.values()}
    allowed_names = set(expected_names)
    if verify:
        allowed_names.add(output.name)
    actual_names: set[str] = set()
    for member in snapshot_root.iterdir():
        if member.is_symlink() or not member.is_file():
            raise IndexGenerationError(
                f"snapshot archive contains a symlink or non-file member: {member.name}"
            )
        actual_names.add(member.name)
    if actual_names != allowed_names:
        missing = sorted(expected_names - actual_names)
        additional = sorted(actual_names - allowed_names)
        raise IndexGenerationError(
            "snapshot coverage is not exact; "
            f"missing={missing}, additional={additional}"
        )
    records: list[dict[str, str]] = []
    for case_id in sorted(jobs):
        job_id = jobs[case_id]
        snapshot = snapshot_root / f"{job_id}.sacct"
        _validate_snapshot(snapshot, case_id=case_id, job_id=job_id, row=rows[case_id])
        records.append(
            {
                "case_id": case_id,
                "job_id": job_id,
                "path": snapshot.name,
                "sha256": reviewed.sha256_file(snapshot),
            }
        )
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "campaign_manifest_sha256": reviewed.sha256_file(
            campaign_root / "prepared_manifest.json"
        ),
        "records": records,
    }


def generate(
    *, campaign_root: Path, snapshot_root: Path, output: Path
) -> dict[str, Any]:
    payload = build_payload(
        campaign_root=campaign_root,
        snapshot_root=snapshot_root,
        output=output,
        verify=False,
    )
    reviewed.atomic_json(output, payload)
    return payload


def verify(
    *, campaign_root: Path, snapshot_root: Path, output: Path
) -> dict[str, Any]:
    expected = build_payload(
        campaign_root=campaign_root,
        snapshot_root=snapshot_root,
        output=output,
        verify=True,
    )
    actual = reviewed.read_object(output)
    if actual != expected:
        raise IndexGenerationError(
            "existing offline index differs from the deterministic reconstruction"
        )
    canonical = json.dumps(actual, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if Path(output).read_text(encoding="utf-8") != canonical:
        raise IndexGenerationError("existing offline index is not canonically serialized")
    return {
        "status": "verified",
        "record_count": len(expected["records"]),
        "index_sha256": reviewed.sha256_file(output),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    for name in ("generate", "verify"):
        child = subparsers.add_parser(name)
        child.add_argument("--campaign-root", type=Path, required=True)
        child.add_argument("--snapshot-root", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        if args.command == "generate":
            result = generate(
                campaign_root=args.campaign_root,
                snapshot_root=args.snapshot_root,
                output=args.output,
            )
        else:
            result = verify(
                campaign_root=args.campaign_root,
                snapshot_root=args.snapshot_root,
                output=args.output,
            )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()


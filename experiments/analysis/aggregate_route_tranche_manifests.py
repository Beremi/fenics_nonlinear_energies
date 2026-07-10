#!/usr/bin/env python3
"""Create one immutable index over separately released EXP-ROUTE tranches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.benchmark.run_record import atomic_write_json


CONTRACT = REPO_ROOT / "paper/protocols/EXP-ROUTE-001-analysis-contract.json"
MATRIX = REPO_ROOT / "experiments/runners/paper_revision_karolina/campaign_matrix.csv"
EXPECTED_TIER_COUNTS = {
    "fixed_state_screen": 78,
    "factorized_quadrature": 18,
    "factorized_microbenchmark": 9,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON token {token!r}")
            ),
        )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _archive_relative(path: Path, *, archive_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(archive_root))
    except ValueError as exc:
        raise ValueError(f"{resolved} escapes the master-manifest archive root") from exc


def _canonical_case_ids(matrix_sha256: str) -> dict[str, set[str]]:
    if _sha256(MATRIX) != matrix_sha256:
        raise ValueError("canonical route matrix hash disagrees with the analysis contract")
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_tier = {
        tier: {
            str(row["case_id"])
            for row in rows
            if row.get("experiment_id") == "EXP-ROUTE-001"
            and row.get("tier") == tier
            and row.get("optional") == "0"
        }
        for tier in EXPECTED_TIER_COUNTS
    }
    for tier, expected_count in EXPECTED_TIER_COUNTS.items():
        if len(by_tier[tier]) != expected_count:
            raise ValueError(f"canonical matrix tier {tier} has the wrong case count")
    return by_tier


def aggregate(
    manifest_paths: list[Path], *, archive_root: Path
) -> dict[str, Any]:
    archive_root = archive_root.resolve()
    contract = _read_object(CONTRACT)
    matrix_sha256 = str(
        contract["publication_model_input_gates"]["karolina_matrix_sha256"]
    )
    canonical_by_tier = _canonical_case_ids(matrix_sha256)
    tiers_seen: set[str] = set()
    commits: set[str] = set()
    cases_seen: set[str] = set()
    tranches: list[dict[str, Any]] = []
    for raw_path in manifest_paths:
        path = raw_path.resolve()
        manifest = _read_object(path)
        if manifest.get("status") != "submitted":
            raise ValueError(f"{path} does not record completed real submission")
        if manifest.get("matrix_sha256") != matrix_sha256:
            raise ValueError(f"{path} carries a stale matrix hash")
        if set(manifest.get("selected_experiments") or []) != {"EXP-ROUTE-001"}:
            raise ValueError(f"{path} is not an isolated EXP-ROUTE tranche")
        selected_tiers = {str(value) for value in manifest.get("selected_tiers") or []}
        if not selected_tiers or not selected_tiers.issubset(EXPECTED_TIER_COUNTS):
            raise ValueError(f"{path} has unknown or missing route tiers")
        if tiers_seen & selected_tiers:
            raise ValueError("route tranche tiers overlap")
        expected_count = sum(EXPECTED_TIER_COUNTS[tier] for tier in selected_tiers)
        if int(manifest.get("case_count", -1)) != expected_count:
            raise ValueError(f"{path} case count disagrees with its selected tiers")
        if manifest.get("test_only_commands") is not False:
            raise ValueError(f"{path} is a test-only tranche")
        commit = str(manifest.get("source_commit", ""))
        if (
            manifest.get("source_dirty") is not False
            or len(commit) != 40
            or any(char not in "0123456789abcdef" for char in commit.lower())
        ):
            raise ValueError(f"{path} lacks clean source provenance")
        release = dict(manifest.get("release_authorization") or {})
        release_path = Path(str(release.get("path", "")))
        if release_path.is_absolute():
            raise ValueError(f"{path} release authorization path is not relocatable")
        release_path = path.parent / release_path
        release_path = release_path.resolve()
        try:
            release_path.relative_to(path.parent.resolve())
        except ValueError as exc:
            raise ValueError(f"{path} release authorization escapes its tranche") from exc
        if (
            release.get("schema_id")
            != "fenics-nonlinear-energies.human-release-authorization"
            or not release_path.is_file()
            or release.get("sha256") != _sha256(release_path)
        ):
            raise ValueError(f"{path} lacks its hash-bound release authorization")
        release_record = _read_object(release_path)
        if (
            release_record.get("schema_id")
            != "fenics-nonlinear-energies.human-release-authorization"
            or int(release_record.get("schema_version", -1)) != 1
            or release_record.get("status") != "approved"
            or release_record.get("decision")
            != "explicit_human_release_after_review"
            or release_record.get("matrix_sha256") != matrix_sha256
            or release_record.get("source_commit") != commit
            or release_record.get("authorizes_experiment") != "EXP-ROUTE-001"
            or not selected_tiers.issubset(
                {str(value) for value in release_record.get("authorizes_tiers") or []}
            )
            or not str(release_record.get("reviewer", "")).strip()
        ):
            raise ValueError(f"{release_path} does not authorize this exact tranche")
        reviewed = release_record.get("reviewed_artifacts")
        if not isinstance(reviewed, list) or not reviewed:
            raise ValueError(f"{release_path} has no archived reviewed artifacts")
        for index, artifact in enumerate(reviewed):
            if not isinstance(artifact, dict):
                raise ValueError("reviewed artifact entry is not an object")
            artifact_path = Path(str(artifact.get("path", "")))
            if artifact_path.is_absolute():
                raise ValueError("reviewed artifact path is not relocatable")
            if not artifact_path.parts or artifact_path.parts[0] != "reviewed_artifacts":
                raise ValueError("reviewed artifact was not copied into the tranche archive")
            artifact_path = (release_path.parent / artifact_path).resolve()
            try:
                artifact_path.relative_to(release_path.parent.resolve())
            except ValueError as exc:
                raise ValueError("reviewed artifact path escapes its tranche") from exc
            if (
                not artifact_path.is_file()
                or artifact.get("sha256") != _sha256(artifact_path)
            ):
                raise ValueError(
                    f"{release_path} reviewed artifact {index} is missing or stale"
                )
        ledger_path = path.parent / "submitted_jobs.jsonl"
        if not ledger_path.is_file():
            raise ValueError(f"{path} lacks submitted_jobs.jsonl")
        ledger_rows: list[dict[str, Any]] = []
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("submission ledger contains a non-object")
                ledger_rows.append(row)
        case_ids = [str(row.get("case_id", "")) for row in ledger_rows]
        expected_case_ids = set().union(
            *(canonical_by_tier[tier] for tier in selected_tiers)
        )
        if (
            len(case_ids) != expected_count
            or len(set(case_ids)) != expected_count
            or set(case_ids) != expected_case_ids
            or any(int(row.get("returncode", 1)) != 0 for row in ledger_rows)
            or any(
                not str(row.get("command", "")).startswith("sbatch ")
                or re.fullmatch(
                    r"Submitted batch job [1-9][0-9]*",
                    str(row.get("stdout", "")).strip(),
                )
                is None
                for row in ledger_rows
            )
        ):
            raise ValueError(f"{ledger_path} does not prove every submitted row")
        if cases_seen & set(case_ids):
            raise ValueError("submission ledgers contain duplicate case IDs")
        cases_seen.update(case_ids)
        tiers_seen.update(selected_tiers)
        commits.add(commit)
        tranches.append(
            {
                "manifest_path": _archive_relative(path, archive_root=archive_root),
                "manifest_sha256": _sha256(path),
                "submitted_jobs_path": _archive_relative(
                    ledger_path, archive_root=archive_root
                ),
                "submitted_jobs_sha256": _sha256(ledger_path),
                "release_authorization_path": _archive_relative(
                    release_path, archive_root=archive_root
                ),
                "release_authorization_sha256": _sha256(release_path),
                "selected_tiers": sorted(selected_tiers),
                "case_count": expected_count,
            }
        )
    if tiers_seen != set(EXPECTED_TIER_COUNTS):
        raise ValueError(f"route tranche union is incomplete: {sorted(tiers_seen)}")
    if len(commits) != 1:
        raise ValueError("route tranches do not share one source commit")
    return {
        "schema_id": "fenics-nonlinear-energies.exp-route-001-campaign-master",
        "schema_version": 1,
        "status": "submitted_tranches_complete",
        "experiment_id": "EXP-ROUTE-001",
        "matrix_sha256": matrix_sha256,
        "source_commit": next(iter(commits)),
        "source_dirty": False,
        "selected_tiers": sorted(tiers_seen),
        "case_count": len(cases_seen),
        "case_ids": sorted(cases_seen),
        "contract_path": str(CONTRACT.relative_to(REPO_ROOT)),
        "contract_sha256": _sha256(CONTRACT),
        "tranches": sorted(tranches, key=lambda row: row["selected_tiers"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tranche-manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    result = aggregate(
        list(args.tranche_manifest), archive_root=output.parent
    )
    atomic_write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()

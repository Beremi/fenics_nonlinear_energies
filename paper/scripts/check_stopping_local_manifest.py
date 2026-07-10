#!/usr/bin/env python3
"""Deep-check and byte-regenerate the local EXP-STOP-001 status artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

from stopping_local_evidence import (
    ANALYSIS_NAME,
    MANIFEST_NAME,
    PLAN_NAME,
    TABLE_NAME,
    audit_campaign,
    read_strict_json,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "paper/tables/generated" / MANIFEST_NAME
GENERATOR = REPO_ROOT / "paper/scripts/generate_stopping_local_status.py"
MANIFEST_SCHEMA_ID = "fenics-nonlinear-energies.exp-stop-001-local-table-manifest"


def _safe_repo_path(raw: object, *, repo_root: Path, label: str) -> Path:
    if (
        not isinstance(raw, str)
        or not raw
        or Path(raw).is_absolute()
        or ".." in Path(raw).parts
        or Path(raw).as_posix() != raw
    ):
        raise ValueError(f"{label} must be a canonical repository-relative path")
    current = repo_root.absolute()
    for part in Path(raw).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} contains a forbidden symlink")
    path = (repo_root / raw).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside the repository") from exc
    return path


def validate_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    repo_root: Path = REPO_ROOT,
    require_canonical: bool = True,
) -> list[str]:
    errors: list[str] = []
    repo_root = repo_root.resolve()
    manifest_path = manifest_path.resolve()
    if require_canonical and manifest_path != (
        repo_root / "paper/tables/generated" / MANIFEST_NAME
    ).resolve():
        errors.append("manifest is not the canonical local-stopping manifest")
    try:
        payload = read_strict_json(manifest_path)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    expected = {
        "schema_id": MANIFEST_SCHEMA_ID,
        "schema_version": 1,
        "status": "admitted_local_calibration_cluster_deferred",
        "publication_evidence": True,
        "experiment_id": "EXP-STOP-001",
        "complete_exp_stop_pass": False,
        "timing_claim_admissible": False,
        "population_robustness_claim_admissible": False,
        "allow_unreferenced_tables": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            errors.append(f"manifest field {key} must equal {value!r}")
    try:
        evidence_root = _safe_repo_path(
            payload.get("evidence_root"), repo_root=repo_root, label="evidence_root"
        )
        evidence_root.relative_to((repo_root / "artifacts/reproduction").resolve())
    except ValueError as exc:
        errors.append(str(exc))
        evidence_root = repo_root / "__invalid_evidence__"
    for key, name in (("source_plan", PLAN_NAME), ("source_analysis", ANALYSIS_NAME)):
        row = payload.get(key)
        if not isinstance(row, dict):
            errors.append(f"{key} must be an object")
            continue
        try:
            path = _safe_repo_path(
                row.get("path"), repo_root=repo_root, label=f"{key}.path"
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if path != (evidence_root / name).resolve():
            errors.append(f"{key} path is noncanonical")
        elif not path.is_file() or row.get("sha256") != sha256_file(path):
            errors.append(f"{key} is missing or has a stale hash")
    tools = payload.get("tools")
    generator_path = GENERATOR
    if not isinstance(tools, dict) or set(tools) != {"validator", "generator", "checker"}:
        errors.append("tool inventory must bind validator, generator, and checker")
    else:
        for key, row in sorted(tools.items()):
            if not isinstance(row, dict):
                errors.append(f"tool {key} record must be an object")
                continue
            try:
                path = _safe_repo_path(
                    row.get("path"), repo_root=repo_root, label=f"tool {key}"
                )
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if not path.is_file() or row.get("sha256") != sha256_file(path):
                errors.append(f"tool {key} is missing or has a stale hash")
            elif key == "generator":
                generator_path = path
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {TABLE_NAME}:
        errors.append("output inventory must bind exactly the local stopping status table")
    else:
        table_path = manifest_path.parent / TABLE_NAME
        if not table_path.is_file() or outputs.get(TABLE_NAME) != sha256_file(table_path):
            errors.append("local stopping status table is missing or has a stale hash")
    fresh: dict[str, object] | None = None
    if not errors:
        try:
            fresh = audit_campaign(evidence_root, repo_root=repo_root)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append(f"independent campaign revalidation failed: {exc}")
        else:
            if payload.get("audit") != fresh:
                errors.append("stored admission audit differs from fresh revalidation")
            if payload.get("source_commit") != fresh.get("source_commit"):
                errors.append("manifest source commit differs from fresh revalidation")
            adjudication = fresh.get("scientific_adjudication")
            if not isinstance(adjudication, dict):
                errors.append("fresh scientific adjudication is missing")
            elif (
                adjudication.get("complete_exp_stop_pass") is not False
                or adjudication.get("timing_claim_admissible") is not False
                or adjudication.get("population_robustness_claim_admissible") is not False
            ):
                errors.append("fresh audit exceeds the local bounded claim scope")
    if not errors and fresh is not None:
        with tempfile.TemporaryDirectory(prefix="stopping-local-check-") as temporary:
            regenerated = Path(temporary)
            process = subprocess.run(
                [
                    sys.executable,
                    str(generator_path),
                    "--evidence-root",
                    str(evidence_root),
                    "--out-dir",
                    str(regenerated),
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            if process.returncode != 0:
                detail = process.stderr.strip() or process.stdout.strip()
                errors.append(f"independent artifact regeneration failed: {detail}")
            else:
                for name in (TABLE_NAME, MANIFEST_NAME):
                    canonical = manifest_path.parent / name
                    candidate = regenerated / name
                    if not candidate.is_file() or candidate.read_bytes() != canonical.read_bytes():
                        errors.append(f"regenerated {name} differs byte-for-byte")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    errors = validate_manifest(args.manifest)
    if errors:
        print("Local stopping evidence is not publication-admissible:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Local stopping evidence OK: bounded status regenerated byte-for-byte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

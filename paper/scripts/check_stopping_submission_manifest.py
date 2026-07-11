#!/usr/bin/env python3
"""Validate and byte-regenerate the narrowed-scope stopping presentation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Mapping

from generate_stopping_submission_status import (
    DEFAULT_OUT_DIR,
    DEFAULT_SOURCE_MANIFEST,
    MANIFEST_NAME,
    REPO_ROOT,
    SCHEMA_ID,
    TABLE_NAME,
    read_strict_json,
    safe_repo_path,
    sha256_file,
)


DEFAULT_MANIFEST = DEFAULT_OUT_DIR / MANIFEST_NAME
GENERATOR = REPO_ROOT / "paper/scripts/generate_stopping_submission_status.py"


def validate_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    require_canonical: bool = True,
    regenerate: bool = True,
) -> list[str]:
    findings: list[str] = []
    manifest_path = manifest_path.resolve()
    if require_canonical and manifest_path != DEFAULT_MANIFEST.resolve():
        findings.append("presentation manifest is not in the canonical location")
    try:
        payload = read_strict_json(manifest_path)
    except (OSError, ValueError) as exc:
        return [str(exc)]

    expected = {
        "schema_id": SCHEMA_ID,
        "schema_version": 1,
        "status": "admitted_reported_local_subset",
        "publication_evidence": True,
        "experiment_id": "EXP-STOP-001",
        "claim_scope": "deterministic_same_discretization_local_subset",
        "reported_local_subset_complete": True,
        "complete_exp_stop_pass": False,
        "timing_claim_admissible": False,
        "population_robustness_claim_admissible": False,
        "allow_unreferenced_tables": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            findings.append(f"field {key} must equal {value!r}")

    counts = payload.get("presentation_counts")
    expected_counts = {
        "executions": 45,
        "admitted_records": 43,
        "accepted_comparisons": 28,
        "rejected_comparisons": 15,
        "endpoint_censored_comparisons": 2,
        "reference_self_comparisons": 11,
        "accepted_nonreference_candidates": 17,
    }
    if counts != expected_counts:
        findings.append("presentation counts differ from the admitted 45-row subset")

    for key, expected_path in (
        ("source_manifest", DEFAULT_SOURCE_MANIFEST.resolve()),
        (
            "source_analysis",
            None,
        ),
    ):
        binding = payload.get(key)
        if not isinstance(binding, Mapping):
            findings.append(f"{key} must be an object")
            continue
        try:
            path = safe_repo_path(binding.get("path"), label=f"{key}.path")
        except ValueError as exc:
            findings.append(str(exc))
            continue
        if expected_path is not None and path != expected_path:
            findings.append(f"{key} path is noncanonical")
        if not path.is_file() or binding.get("sha256") != sha256_file(path):
            findings.append(f"{key} is missing or has a stale hash")

    tools = payload.get("tools")
    if not isinstance(tools, Mapping) or set(tools) != {"generator", "checker"}:
        findings.append("tools must bind generator and checker")
    else:
        for name, binding in sorted(tools.items()):
            if not isinstance(binding, Mapping):
                findings.append(f"tool {name} must be an object")
                continue
            try:
                path = safe_repo_path(binding.get("path"), label=f"tool {name}")
            except ValueError as exc:
                findings.append(str(exc))
                continue
            if not path.is_file() or binding.get("sha256") != sha256_file(path):
                findings.append(f"tool {name} is missing or has a stale hash")

    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {TABLE_NAME}:
        findings.append(f"outputs must bind exactly {TABLE_NAME}")
    else:
        table = manifest_path.parent / TABLE_NAME
        if not table.is_file() or outputs[TABLE_NAME] != sha256_file(table):
            findings.append("presentation table is missing or has a stale hash")

    if regenerate and not findings:
        source = payload["source_manifest"]
        assert isinstance(source, Mapping)
        with tempfile.TemporaryDirectory(
            prefix="stopping-submission-check-"
        ) as temporary:
            process = subprocess.run(
                [
                    sys.executable,
                    str(GENERATOR),
                    "--source-manifest",
                    str(safe_repo_path(source["path"], label="source_manifest.path")),
                    "--out-dir",
                    temporary,
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            if process.returncode != 0:
                findings.append(
                    "presentation regeneration failed: "
                    + (process.stderr.strip() or process.stdout.strip())
                )
            else:
                for name in (TABLE_NAME, MANIFEST_NAME):
                    candidate = Path(temporary) / name
                    canonical = manifest_path.parent / name
                    if (
                        not candidate.is_file()
                        or candidate.read_bytes() != canonical.read_bytes()
                    ):
                        findings.append(f"regenerated {name} differs byte-for-byte")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args(argv)
    findings = validate_manifest(
        args.manifest,
        require_canonical=args.manifest.resolve() == DEFAULT_MANIFEST.resolve(),
        regenerate=not args.structure_only,
    )
    if findings:
        print("Stopping submission presentation is invalid:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("Stopping submission presentation OK: regenerated byte-for-byte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

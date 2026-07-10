#!/usr/bin/env python3
"""Fail unless the revised manuscript tables use admitted clean evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from admit_revision_publication_evidence import (
    EVIDENCE_SPECS,
    validate_publication_source_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "paper/tables/generated/revision_evidence_manifest.json"
EXPECTED_GENERATOR = REPO_ROOT / "paper/scripts/generate_revision_evidence_tables.py"
EXPECTED_INPUTS = {spec.key: spec.relative_path for spec in EVIDENCE_SPECS}
EXPECTED_OUTPUTS = frozenset(
    {
        "revision_verification_summary.tex",
        "revision_derivative_checks.tex",
        "revision_quadrature_sensitivity.tex",
        "revision_evidence_status.tex",
    }
)
MANUSCRIPT_INPUT_BINDINGS = {
    "revision_verification_summary.tex": (
        Path("paper/sections/validation.tex"),
        r"\input{tables/generated/revision_verification_summary.tex}",
    ),
    "revision_derivative_checks.tex": (
        Path("paper/sections/validation.tex"),
        r"\input{tables/generated/revision_derivative_checks.tex}",
    ),
    "revision_quadrature_sensitivity.tex": (
        Path("paper/sections/results.tex"),
        r"\input{tables/generated/revision_quadrature_sensitivity.tex}",
    ),
    "revision_evidence_status.tex": (
        Path("paper/sections/results.tex"),
        r"\input{tables/generated/revision_evidence_status.tex}",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_repo_relative_path(value: Any, *, label: str, errors: list[str]) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a non-empty repository-relative path")
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        errors.append(f"{label} must not be absolute, non-canonical, or contain '..'")
        return None
    return path


def _contained(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _git_is_ancestor(repo_root: Path, older: str, newer: str) -> bool:
    if not older or not newer:
        return False
    return subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", older, newer],
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0


def _git_metadata(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, not bool(status.strip())


def validate_revision_evidence_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    repo_root: Path = REPO_ROOT,
    require_clean_worktree: bool = True,
) -> list[str]:
    repo_root = repo_root.resolve()
    manifest_path = manifest_path.resolve()
    errors: list[str] = []
    if not manifest_path.is_file():
        return [f"missing manifest: {manifest_path}"]
    try:
        payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot parse manifest: {exc}"]

    canonical_manifest = repo_root / DEFAULT_MANIFEST.relative_to(REPO_ROOT)
    if manifest_path != canonical_manifest.resolve():
        errors.append(
            "revision evidence manifest must be the canonical manuscript table manifest at "
            "paper/tables/generated/revision_evidence_manifest.json"
        )

    if int(payload.get("schema_version", 0)) != 2:
        errors.append("schema_version must be 2")
    if payload.get("evidence_class") != "publication":
        errors.append("evidence_class must be publication")
    if payload.get("publication_evidence") is not True:
        errors.append("publication_evidence must be true")
    if payload.get("status") != "clean_publication_tables":
        errors.append("status must be clean_publication_tables")

    generator_relative = _safe_repo_relative_path(
        payload.get("generator"), label="generator", errors=errors
    )
    generator_path = repo_root / generator_relative if generator_relative is not None else repo_root
    expected_generator = (
        repo_root / EXPECTED_GENERATOR.relative_to(REPO_ROOT)
        if repo_root != REPO_ROOT
        else EXPECTED_GENERATOR
    )
    if generator_path.resolve() != expected_generator.resolve():
        errors.append("generator path does not identify generate_revision_evidence_tables.py")
    elif not generator_path.is_file():
        errors.append("recorded generator is missing")
    elif str(payload.get("generator_sha256", "")) != _sha256(generator_path):
        errors.append("generator SHA-256 mismatch")

    git_row = payload.get("git")
    if not isinstance(git_row, dict):
        errors.append("git metadata is missing")
        git_row = {}
    try:
        current_commit, current_clean = _git_metadata(repo_root)
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"cannot inspect Git state: {exc}")
        current_commit, current_clean = "", False
    table_generation_commit = str(git_row.get("commit", ""))
    if not _git_is_ancestor(repo_root, table_generation_commit, current_commit):
        errors.append("table-generation commit is not an ancestor of current HEAD")
    if git_row.get("worktree_clean") is not True:
        errors.append("table manifest was not generated from a clean worktree")
    if require_clean_worktree and not current_clean:
        errors.append("current Git worktree is not clean")

    evidence_root_relative = _safe_repo_relative_path(
        payload.get("evidence_root"), label="evidence_root", errors=errors
    )
    evidence_root = (
        repo_root / evidence_root_relative
        if evidence_root_relative is not None
        else repo_root / "__invalid_evidence_root__"
    )
    if evidence_root_relative is not None and not _contained(evidence_root, repo_root):
        errors.append("evidence_root resolves outside the repository")

    source_manifest_payload: dict[str, Any] | None = None
    source = payload.get("source_evidence_manifest")
    if not isinstance(source, dict):
        errors.append("source_evidence_manifest is missing")
    else:
        source_relative = _safe_repo_relative_path(
            source.get("path"), label="source_evidence_manifest.path", errors=errors
        )
        source_path = repo_root / source_relative if source_relative is not None else repo_root
        if source_relative is None:
            errors.append(
                "source evidence admission failed deep revalidation: source manifest path is unsafe"
            )
        if source_relative is not None and not _contained(source_path, evidence_root):
            errors.append("source evidence manifest must resolve inside evidence_root")
        if source_relative is None or not source_path.is_file():
            errors.append("source evidence manifest is missing")
        elif str(source.get("sha256", "")) != _sha256(source_path):
            errors.append("source evidence manifest SHA-256 mismatch")
        else:
            try:
                source_manifest_payload = validate_publication_source_manifest(
                    source_path,
                    evidence_root=evidence_root,
                    repo_root=repo_root,
                )
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                errors.append(f"source evidence admission failed deep revalidation: {exc}")

    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        errors.append("input hash map is missing")
    else:
        if set(inputs) != set(EXPECTED_INPUTS):
            errors.append("input hash map must contain exactly the 14 configured input keys")
        for key, relative in EXPECTED_INPUTS.items():
            row = inputs.get(key)
            if not isinstance(row, dict):
                errors.append(f"input {key} record is not an object")
                continue
            recorded = _safe_repo_relative_path(
                row.get("path"), label=f"input {key}.path", errors=errors
            )
            expected_path = (evidence_root / relative).resolve()
            path = repo_root / recorded if recorded is not None else repo_root
            if recorded is not None and path.resolve() != expected_path:
                errors.append(f"input {key} path does not equal evidence_root/{relative.as_posix()}")
            if recorded is not None and not _contained(path, evidence_root):
                errors.append(f"input {key} resolves outside evidence_root")
            if row.get("path_within_evidence_root") != relative.as_posix():
                errors.append(f"input {key} path_within_evidence_root is not canonical")
            if recorded is None or not path.is_file():
                errors.append(f"input {key} is missing")
            elif str(row.get("sha256", "")) != _sha256(path):
                errors.append(f"input {key} SHA-256 mismatch")
            if source_manifest_payload is not None:
                source_row = source_manifest_payload.get("inputs", {}).get(key, {})
                if not isinstance(source_row, dict) or row.get("sha256") != source_row.get("sha256"):
                    errors.append(f"input {key} hash differs from the admitted source manifest")

    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        errors.append("output hash map is missing")
    else:
        if set(outputs) != EXPECTED_OUTPUTS:
            errors.append("output hash map must contain exactly the four manuscript revision tables")
        canonical_output_dir = repo_root / "paper/tables/generated"
        for name in sorted(EXPECTED_OUTPUTS):
            expected_hash = outputs.get(name)
            safe_name = _safe_repo_relative_path(
                name, label=f"output {name}", errors=errors
            )
            if safe_name is None or len(safe_name.parts) != 1:
                errors.append(f"output {name} must be one canonical basename")
                continue
            path = manifest_path.parent / safe_name
            if path.resolve() != (canonical_output_dir / name).resolve():
                errors.append(f"output {name} is not the canonical manuscript table file")
            if not path.is_file():
                errors.append(f"output {name} is missing")
            elif str(expected_hash) != _sha256(path):
                errors.append(f"output {name} SHA-256 mismatch")

    for output_name, (tex_relative, input_literal) in MANUSCRIPT_INPUT_BINDINGS.items():
        tex_path = repo_root / tex_relative
        if not tex_path.is_file():
            errors.append(f"manuscript source is missing: {tex_relative.as_posix()}")
            continue
        occurrences = tex_path.read_text(encoding="utf-8").count(input_literal)
        if occurrences != 1:
            errors.append(
                f"{tex_relative.as_posix()} must contain exactly one canonical input for {output_name}"
            )

    # A valid release must be reproducible from the admitted sources.  Run the
    # hash-bound generator in a temporary directory and compare the actual
    # manuscript table bytes, not merely arbitrary paths named in a manifest.
    if not errors and source_manifest_payload is not None:
        source_path = repo_root / str(source["path"])
        with tempfile.TemporaryDirectory(prefix="revision-evidence-check-") as temporary:
            regenerated_dir = Path(temporary)
            process = subprocess.run(
                [
                    sys.executable,
                    str(expected_generator),
                    "--out-dir",
                    str(regenerated_dir),
                    "--evidence-root",
                    str(evidence_root),
                    "--evidence-class",
                    "publication",
                    "--evidence-manifest",
                    str(source_path),
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            if process.returncode != 0:
                detail = process.stderr.strip() or process.stdout.strip()
                errors.append(f"independent table regeneration failed: {detail}")
            else:
                for name in sorted(EXPECTED_OUTPUTS):
                    canonical = repo_root / "paper/tables/generated" / name
                    regenerated = regenerated_dir / name
                    if not regenerated.is_file() or regenerated.read_bytes() != canonical.read_bytes():
                        errors.append(f"regenerated output {name} differs byte-for-byte")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--expect-diagnostic",
        action="store_true",
        help="succeed only when the manifest is correctly rejected",
    )
    args = parser.parse_args(argv)
    errors = validate_revision_evidence_manifest(
        args.manifest,
        repo_root=args.repo_root,
    )
    if errors:
        print("Revision evidence manifest is not submission-admissible:")
        for error in errors:
            print(f"- {error}")
        return 0 if args.expect_diagnostic else 1
    if args.expect_diagnostic:
        print("Expected diagnostic evidence, but the manifest is publication-admissible.")
        return 1
    print("Revision evidence manifest OK: clean publication inputs and outputs verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

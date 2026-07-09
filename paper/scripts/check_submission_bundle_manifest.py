#!/usr/bin/env python3
"""Verify SHA-256 records in the local paper submission-bundle manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import REPO_ROOT


DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "reproduction" / "paper_submission_2026_07_08" / "manifest.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
EXTERNAL_PREFIXES = ("external_reference/",)
BUNDLE_REFRESH_PATHS = (
    "paper/main.tex",
    "paper/sections/",
    "paper/figures/generated/",
    "paper/tables/generated/",
    "paper/scripts/build_submission_bundle.py",
    "paper/scripts/generate_paper_figures.py",
    "paper/scripts/generate_paper_tables.py",
)
SOURCE_PATH_ALIASES = (
    (
        "artifacts/reports/supplemental_solver_evidence/",
        "artifacts/reports/paper_reviewer_gap_experiments/",
    ),
    (
        "artifacts/raw_results/supplemental_solver_evidence/",
        "artifacts/raw_results/paper_reviewer_gap_experiments/",
    ),
)


@dataclass(frozen=True)
class BundleCheckResult:
    bundle_files: int
    local_sources: int
    external_sources: int
    git_commit: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_repo_path(repo_root: Path, raw_path: Any, label: str, findings: list[str]) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        findings.append(f"{label}: missing nonempty string path")
        return None
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        findings.append(f"{label}: path is not repository-relative and safe: {raw_path!r}")
        return None
    candidate = (repo_root / path).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError:
        findings.append(f"{label}: path resolves outside repository: {raw_path!r}")
        return None
    return candidate


def _check_hash(path: Path, expected: Any, label: str, findings: list[str]) -> bool:
    if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
        findings.append(f"{label}: missing or malformed SHA-256 value")
        return False
    if not path.exists():
        findings.append(f"{label}: file is missing: {path}")
        return False
    actual = _sha256(path)
    if actual != expected:
        findings.append(f"{label}: SHA-256 mismatch: expected {expected}, got {actual}")
        return False
    return True


def _source_path_for_hash(repo_root: Path, raw_path: str, label: str, findings: list[str]) -> Path | None:
    path = _safe_repo_path(repo_root, raw_path, label, findings)
    if path is None or path.exists():
        return path
    for public_prefix, local_prefix in SOURCE_PATH_ALIASES:
        if not raw_path.startswith(public_prefix):
            continue
        alias = local_prefix + raw_path[len(public_prefix) :]
        alias_path = _safe_repo_path(repo_root, alias, f"{label} local alias", findings)
        if alias_path is not None and alias_path.exists():
            return alias_path
    return path


def _check_source_record(
    repo_root: Path,
    record: dict[str, Any],
    label: str,
    findings: list[str],
) -> tuple[int, int]:
    source_path = record.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        findings.append(f"{label}: missing nonempty source_path")
        return 0, 0
    if source_path.startswith(EXTERNAL_PREFIXES):
        expected = record.get("source_sha256")
        if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
            findings.append(f"{label}: external source has missing or malformed source_sha256")
        return 0, 1
    path = _source_path_for_hash(repo_root, source_path, f"{label} source_path", findings)
    if path is None:
        return 0, 0
    ok = _check_hash(path, record.get("source_sha256"), f"{label} source", findings)
    return (1 if ok else 0), 0


def _git(
    repo_root: Path,
    args: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _changed_bundle_refresh_paths(repo_root: Path, git_commit: str) -> list[str]:
    result = _git(
        repo_root,
        ["diff", "--name-only", f"{git_commit}..HEAD", "--", *BUNDLE_REFRESH_PATHS],
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _check_git_commit_freshness(
    repo_root: Path,
    manifest: dict[str, Any],
    findings: list[str],
) -> str | None:
    raw_commit = manifest.get("git_commit")
    if not isinstance(raw_commit, str) or GIT_COMMIT_RE.fullmatch(raw_commit) is None:
        findings.append("git_commit: missing or malformed 40-character Git commit SHA")
        return None

    commit = raw_commit.lower()
    try:
        _git(repo_root, ["cat-file", "-e", f"{commit}^{{commit}}"])
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        stderr = getattr(exc, "stderr", "") or str(exc)
        findings.append(f"git_commit: commit is not present in this repository: {commit} ({stderr.strip()})")
        return commit

    try:
        _git(repo_root, ["merge-base", "--is-ancestor", commit, "HEAD"])
    except subprocess.CalledProcessError:
        findings.append(f"git_commit: commit is not an ancestor of HEAD: {commit}")
        return commit

    changed_paths = _changed_bundle_refresh_paths(repo_root, commit)
    if changed_paths:
        preview = ", ".join(changed_paths[:8])
        suffix = "" if len(changed_paths) <= 8 else f", ... ({len(changed_paths)} paths)"
        findings.append(
            "git_commit: bundle-refresh paths changed after the recorded commit "
            f"{commit}: {preview}{suffix}. Run `make -C paper submission-bundle` "
            "after committing manuscript/generated paper changes."
        )
    return commit


def check_manifest(
    manifest_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    check_git_commit: bool = True,
) -> BundleCheckResult:
    repo_root = repo_root.resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"missing submission-bundle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SystemExit("submission-bundle manifest must be a JSON object")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list):
        raise SystemExit("submission-bundle manifest field `source_files` must be a list")

    findings: list[str] = []
    git_commit = _check_git_commit_freshness(repo_root, manifest, findings) if check_git_commit else None
    bundle_files = 0
    local_sources = 0
    external_sources = 0
    for index, entry in enumerate(source_files, start=1):
        label = f"source_files[{index}]"
        if not isinstance(entry, dict):
            findings.append(f"{label}: entry must be an object")
            continue
        bundle_path = _safe_repo_path(repo_root, entry.get("bundle_path"), f"{label} bundle_path", findings)
        if bundle_path is not None and _check_hash(
            bundle_path, entry.get("bundle_sha256"), f"{label} bundle", findings
        ):
            bundle_files += 1
        source_count, external_count = _check_source_record(repo_root, entry, label, findings)
        local_sources += source_count
        external_sources += external_count
        dependencies = entry.get("source_dependencies", [])
        if dependencies is None:
            dependencies = []
        if not isinstance(dependencies, list):
            findings.append(f"{label}: source_dependencies must be a list when present")
            continue
        for dep_index, dependency in enumerate(dependencies, start=1):
            dep_label = f"{label}.source_dependencies[{dep_index}]"
            if not isinstance(dependency, dict):
                findings.append(f"{dep_label}: dependency must be an object")
                continue
            source_count, external_count = _check_source_record(repo_root, dependency, dep_label, findings)
            local_sources += source_count
            external_sources += external_count

    if findings:
        raise SystemExit("Submission-bundle manifest check failed:\n" + "\n".join(findings))
    return BundleCheckResult(
        bundle_files=bundle_files,
        local_sources=local_sources,
        external_sources=external_sources,
        git_commit=git_commit,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--skip-git-commit-check",
        action="store_true",
        help="verify hashes only; used for isolated fixture tests or non-git archive inspection",
    )
    args = parser.parse_args(argv)
    result = check_manifest(args.manifest, check_git_commit=not args.skip_git_commit_check)
    print(
        "Submission bundle manifest OK: "
        f"{result.bundle_files} bundle files, "
        f"{result.local_sources} local source hashes, "
        f"{result.external_sources} external source hashes recorded"
        + (f"; git_commit {result.git_commit} is fresh for bundle-refresh paths." if result.git_commit else ".")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

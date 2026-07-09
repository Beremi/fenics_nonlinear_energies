#!/usr/bin/env python3
"""Report final paper-release blockers that local build checks cannot resolve."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from common import PAPER_ROOT, REPO_ROOT


DEFAULT_BUNDLE_MANIFEST = REPO_ROOT / "artifacts" / "reproduction" / "paper_submission_2026_07_08" / "manifest.json"
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")


@dataclass(frozen=True)
class ReleaseBlocker:
    code: str
    message: str
    evidence: str
    required_action: str


def _main_text(repo_root: Path) -> str:
    path = repo_root / "paper" / "main.tex"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _availability_text(main_tex: str) -> str:
    match = re.search(
        r"\\section\*\{Code and Data Availability\}(.*?)(?:\\bibliographystyle|\\bibliography|\\end\{document\})",
        main_tex,
        flags=re.DOTALL,
    )
    if match is None:
        return ""
    return match.group(1)


def _has_root_license(repo_root: Path) -> bool:
    names = {path.name.lower() for path in repo_root.iterdir() if path.is_file()}
    return any(name == "license" or name.startswith("license.") or name.startswith("copying") for name in names)


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _target_template_blocker(main_tex: str) -> ReleaseBlocker | None:
    if re.search(r"\\documentclass(?:\[[^\]]*\])?\{article\}", main_tex):
        return ReleaseBlocker(
            code="target-template",
            message="The manuscript still uses the generic LaTeX article class.",
            evidence="paper/main.tex front matter",
            required_action=(
                "Choose the target venue, apply its template, and fill the required author, "
                "funding, acknowledgement, competing-interest, and availability fields."
            ),
        )
    return None


def _license_blocker(repo_root: Path) -> ReleaseBlocker | None:
    if _has_root_license(repo_root):
        return None
    return ReleaseBlocker(
        code="repository-license",
        message="No root LICENSE or COPYING file is present.",
        evidence="repository root",
        required_action="Add the chosen repository license before creating the durable software/artifact release.",
    )


def _doi_blocker(main_tex: str) -> ReleaseBlocker | None:
    availability = _availability_text(main_tex)
    if DOI_RE.search(availability) and "No separate archival DOI is cited" not in availability:
        return None
    return ReleaseBlocker(
        code="archival-doi",
        message="The availability statement does not cite a separate archival DOI.",
        evidence="paper/main.tex Code and Data Availability section",
        required_action=(
            "Archive the final source/artifact snapshot, mint or record its DOI, and cite that durable version "
            "in the availability statement."
        ),
    )


def _bundle_release_blocker(manifest_path: Path, repo_root: Path) -> ReleaseBlocker | None:
    evidence = _display_path(manifest_path, repo_root)
    if not manifest_path.is_file():
        return ReleaseBlocker(
            code="submission-bundle",
            message="The local submission-bundle manifest is missing.",
            evidence=evidence,
            required_action="Build the local provenance bundle before release packaging.",
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    limitations = manifest.get("known_limitations", [])
    limitation_text = "\n".join(str(item) for item in limitations if isinstance(item, str))
    if "permanent archive DOI remain outside this bundle" in limitation_text:
        return ReleaseBlocker(
            code="durable-archive",
            message="The bundle manifest still says the permanent archive DOI is outside the bundle.",
            evidence=evidence,
            required_action=(
                "Include the bundle in the final licensed archive/release, rerun provenance validation from that "
                "released snapshot, and update the manifest or manuscript availability statement accordingly."
            ),
        )
    return None


def find_release_blockers(
    *,
    repo_root: Path = REPO_ROOT,
    bundle_manifest: Path | None = None,
) -> list[ReleaseBlocker]:
    repo_root = repo_root.resolve()
    bundle_manifest = bundle_manifest or repo_root / DEFAULT_BUNDLE_MANIFEST.relative_to(REPO_ROOT)
    main_tex = _main_text(repo_root)
    candidates = (
        _target_template_blocker(main_tex),
        _license_blocker(repo_root),
        _doi_blocker(main_tex),
        _bundle_release_blocker(bundle_manifest, repo_root),
    )
    return [blocker for blocker in candidates if blocker is not None]


def _print_blockers(blockers: list[ReleaseBlocker]) -> None:
    if not blockers:
        print("Release blocker audit OK: no final-release blockers detected.")
        return
    print("Release blocker audit found unresolved final-submission blockers:")
    for blocker in blockers:
        print(f"- {blocker.code}: {blocker.message}")
        print(f"  Evidence: {blocker.evidence}")
        print(f"  Required action: {blocker.required_action}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--bundle-manifest", type=Path, default=None)
    parser.add_argument(
        "--expect-blockers",
        action="store_true",
        help="exit successfully only when blockers are currently present",
    )
    args = parser.parse_args(argv)
    blockers = find_release_blockers(repo_root=args.repo_root, bundle_manifest=args.bundle_manifest)
    _print_blockers(blockers)
    if args.expect_blockers:
        if blockers:
            return 0
        print("Expected release blockers, but none were detected.")
        return 1
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common import FIGURES_ROOT, PAPER_ROOT, REPO_ROOT, TABLES_ROOT, ensure_paper_dirs


INCLUDE_RE = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")
INPUT_IF_EXISTS_RE = re.compile(r"\\InputIfFileExists\s*\{([^{}]+)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\s*\[[^\]]*\])*\s*\{([^{}]+)\}")
PROVENANCE_BANNED_SNIPPETS = (
    "/home/",
    "\\/home\\/",
    "/workdir/",
    "\\/workdir\\/",
    ".venv",
    "tmp/source_compare",
    "tmp_work",
    "local_env",
    "Locked Cases",
    "fairness-gated",
    "reviewer_",
    "local_vs_source",
    "source_continuation_compare",
    "sourcefixed",
    "source-operator",
    "source operator",
    "NaN",
)
ARCHIVE_NEUTRAL_BLOCKED_PREFIXES = (
    "artifacts/raw_results/",
    "artifacts/reports/",
)
SUBMISSION_BUNDLE_MANIFEST = PAPER_ROOT.parent / "artifacts" / "reproduction" / "paper_submission_2026_07_08" / "manifest.json"
TEXT_SCAN_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".tex",
    ".txt",
    ".yml",
    ".yaml",
}


def _strip_tex_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        escaped = False
        kept: list[str] = []
        for char in line:
            if char == "%" and not escaped:
                break
            kept.append(char)
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
        lines.append("".join(kept))
    return "\n".join(lines)


def _tex_path(raw: str) -> str:
    path = raw.strip()
    if not Path(path).suffix:
        path += ".tex"
    return path


def _resolve_tex_path(raw: str, current_dir: Path) -> Path:
    path = Path(_tex_path(raw))
    candidates = [PAPER_ROOT / path]
    if not path.is_absolute():
        candidates.append(current_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _figure_name(raw: str) -> str:
    path = Path(raw.strip())
    if not path.suffix:
        path = path.with_suffix(".pdf")
    return path.name


def _table_name(raw: str) -> str | None:
    path = Path(_tex_path(raw))
    parts = path.parts
    if len(parts) >= 3 and parts[-3:-1] == ("tables", "generated"):
        return path.name
    return None


def _collect_tex_assets(path: Path, *, seen: set[Path] | None = None) -> tuple[set[str], set[str], set[Path]]:
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        return set(), set(), seen
    seen.add(path)
    if not path.exists():
        return set(), set(), seen

    text = _strip_tex_comments(path.read_text(encoding="utf-8"))
    figures = {_figure_name(match.group(1)) for match in GRAPHICS_RE.finditer(text)}
    tables: set[str] = set()
    input_names = [match.group(1) for match in INCLUDE_RE.finditer(text)]
    input_names.extend(match.group(1) for match in INPUT_IF_EXISTS_RE.finditer(text))
    for raw in input_names:
        table_name = _table_name(raw)
        if table_name is not None:
            tables.add(table_name)
            continue
        child_figures, child_tables, seen = _collect_tex_assets(_resolve_tex_path(raw, path.parent), seen=seen)
        figures.update(child_figures)
        tables.update(child_tables)
    return figures, tables, seen


def _manifest_assets(figures_dir: Path) -> set[str]:
    manifest_path = figures_dir / "manifest.json"
    if not manifest_path.exists():
        return set()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets: set[str] = set()
    for key in ("generated_assets", "copied_assets"):
        for name in manifest.get(key, []):
            if isinstance(name, str):
                assets.add(Path(name).name)
    return assets


def _provenance_targets(
    tex_path: Path,
    seen_tex: set[Path],
    required_tables: set[str],
    figures_dir: Path,
    tables_dir: Path,
) -> list[Path]:
    targets = {tex_path.resolve()}
    targets.update(path for path in seen_tex if path.exists())
    targets.update((tables_dir / name).resolve() for name in required_tables)
    manifest_path = (figures_dir / "manifest.json").resolve()
    if manifest_path.exists():
        targets.add(manifest_path)
    return sorted(targets)


def _find_banned_snippets(path: Path, snippets: tuple[str, ...]) -> list[str]:
    findings: list[str] = []
    text = path.read_text(encoding="utf-8")
    for snippet in snippets:
        if snippet not in text:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if snippet in line:
                rel_path = path.relative_to(REPO_ROOT)
                findings.append(f"{rel_path}:{line_number}: contains banned paper provenance snippet {snippet!r}")
                break
    return findings


def _validate_provenance_text(paths: list[Path]) -> None:
    findings: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        findings.extend(_find_banned_snippets(path, PROVENANCE_BANNED_SNIPPETS))
    if findings:
        raise SystemExit("Paper provenance scan failed:\n" + "\n".join(findings))


def _iter_manifest_inputs(manifest: dict[str, object]) -> list[tuple[str, object]]:
    inputs = manifest.get("generated_asset_inputs", {})
    if not isinstance(inputs, dict):
        raise SystemExit("Figure manifest field `generated_asset_inputs` must be an object.")
    entries: list[tuple[str, object]] = []
    for asset, values in sorted(inputs.items()):
        if not isinstance(asset, str):
            raise SystemExit("Figure manifest asset names must be strings.")
        if not isinstance(values, list):
            raise SystemExit(f"Figure manifest inputs for {asset!r} must be a list.")
        for value in values:
            entries.append((asset, value))
    return entries


def _archive_neutral_findings_for_input(asset: str, entry: object) -> tuple[list[str], list[Path]]:
    findings: list[str] = []
    paths_to_scan: list[Path] = []
    if isinstance(entry, str):
        if entry.startswith("/"):
            findings.append(f"{asset}: legacy manifest input is absolute: {entry}")
        if entry.startswith(ARCHIVE_NEUTRAL_BLOCKED_PREFIXES):
            findings.append(f"{asset}: legacy manifest input is not in a submission bundle: {entry}")
        for snippet in PROVENANCE_BANNED_SNIPPETS:
            if snippet in entry:
                findings.append(f"{asset}: legacy manifest input contains {snippet!r}: {entry}")
                break
        candidate = (REPO_ROOT / entry).resolve() if not entry.startswith("/") else Path(entry)
        if candidate.exists():
            paths_to_scan.append(candidate)
        return findings, paths_to_scan
    if not isinstance(entry, dict):
        findings.append(f"{asset}: manifest input must be a string or object, got {type(entry).__name__}")
        return findings, paths_to_scan
    kind = entry.get("kind")
    if kind == "external_reference":
        identifier = str(entry.get("identifier", ""))
        findings.append(
            f"{asset}: external reference {identifier!r} is not archive-neutral; replace it with a bundle-relative artifact."
        )
        return findings, paths_to_scan
    if kind != "repository_path":
        findings.append(f"{asset}: unknown manifest input kind {kind!r}")
        return findings, paths_to_scan
    raw_path = entry.get("path")
    if not isinstance(raw_path, str):
        findings.append(f"{asset}: repository_path input is missing a string `path` field.")
        return findings, paths_to_scan
    if raw_path.startswith("/") or ".." in Path(raw_path).parts:
        findings.append(f"{asset}: repository_path input is not a safe repo-relative path: {raw_path}")
        return findings, paths_to_scan
    if raw_path.startswith(ARCHIVE_NEUTRAL_BLOCKED_PREFIXES):
        findings.append(f"{asset}: repository_path input is not in a submission bundle: {raw_path}")
    for snippet in PROVENANCE_BANNED_SNIPPETS:
        if snippet in raw_path:
            findings.append(f"{asset}: repository_path input contains {snippet!r}: {raw_path}")
            break
    candidate = (REPO_ROOT / raw_path).resolve()
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError:
        findings.append(f"{asset}: repository_path resolves outside the repository: {raw_path}")
        return findings, paths_to_scan
    if not candidate.exists():
        findings.append(f"{asset}: repository_path input is missing: {raw_path}")
        return findings, paths_to_scan
    paths_to_scan.append(candidate)
    return findings, paths_to_scan


def _validate_archive_neutral_manifest(figures_dir: Path) -> None:
    manifest_path = figures_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Figure manifest missing for archive-neutral check: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    findings: list[str] = []
    paths_to_scan: set[Path] = set()
    for asset, entry in _iter_manifest_inputs(manifest):
        entry_findings, entry_paths = _archive_neutral_findings_for_input(asset, entry)
        findings.extend(entry_findings)
        paths_to_scan.update(path for path in entry_paths if path.suffix.lower() in TEXT_SCAN_SUFFIXES)
    reproducibility_note = PAPER_ROOT / "build" / "reproducibility_note.md"
    if reproducibility_note.exists():
        paths_to_scan.add(reproducibility_note)
    if SUBMISSION_BUNDLE_MANIFEST.exists():
        paths_to_scan.add(SUBMISSION_BUNDLE_MANIFEST)
    for path in sorted(paths_to_scan):
        findings.extend(_find_banned_snippets(path, PROVENANCE_BANNED_SNIPPETS))
    if findings:
        raise SystemExit("Archive-neutral paper provenance check failed:\n" + "\n".join(findings))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that the paper asset generation produced the expected files.")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_ROOT)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_ROOT)
    parser.add_argument("--tex", type=Path, default=PAPER_ROOT / "main.tex")
    parser.add_argument(
        "--archive-neutral",
        action="store_true",
        help="also require manifest inputs to come from archive-neutral bundle paths and recursively scan text inputs",
    )
    args = parser.parse_args()
    ensure_paper_dirs()
    required_figures, required_tables, seen_tex = _collect_tex_assets(args.tex)
    missing: list[str] = []
    for name in sorted(required_figures):
        path = args.figures_dir / name
        if not path.exists():
            missing.append(str(path))
    for name in sorted(required_tables):
        path = args.tables_dir / name
        if not path.exists():
            missing.append(str(path))
    manifest_assets = _manifest_assets(args.figures_dir)
    untracked_figures = sorted(required_figures - manifest_assets)
    if missing:
        raise SystemExit("Missing paper assets:\n" + "\n".join(missing))
    if untracked_figures:
        raise SystemExit("TeX-included figures missing from figure manifest:\n" + "\n".join(untracked_figures))
    provenance_targets = _provenance_targets(args.tex, seen_tex, required_tables, args.figures_dir, args.tables_dir)
    _validate_provenance_text(provenance_targets)
    if args.archive_neutral:
        _validate_archive_neutral_manifest(args.figures_dir)
    print(
        f"Paper assets validated ({len(required_figures)} figures, {len(required_tables)} tables); "
        f"provenance scan passed ({len(provenance_targets)} files)."
    )


if __name__ == "__main__":
    main()

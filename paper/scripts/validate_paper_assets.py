#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common import (
    FIGURES_ROOT,
    PAPER_BUNDLE_MANIFEST,
    PAPER_ROOT,
    REPO_ROOT,
    TABLES_ROOT,
    ensure_paper_dirs,
    sha256_file,
    add_paper_bundle_root_argument,
)


INCLUDE_RE = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")
INPUT_IF_EXISTS_RE = re.compile(r"\\InputIfFileExists\s*\{([^{}]+)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\s*\[[^\]]*\])*\s*\{([^{}]+)\}")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DISTRIBUTED_COLORED_MANIFEST = "distributed_colored_manifest.json"
GLOBALIZATION_LOCAL_MANIFEST = "globalization_local_manifest.json"
STOPPING_LOCAL_MANIFEST = "stopping_local_manifest.json"
STOPPING_SUBMISSION_MANIFEST = "stopping_submission_manifest.json"
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
SUBMISSION_BUNDLE_MANIFEST = PAPER_BUNDLE_MANIFEST
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


def _validate_no_unexpected_generated_figures(figures_dir: Path, manifest_assets: set[str]) -> None:
    expected = set(manifest_assets)
    expected.update(Path(name).with_suffix(".png").name for name in manifest_assets if Path(name).suffix == ".pdf")
    actual = {
        path.name
        for path in figures_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".png"}
    }
    unexpected = sorted(actual - expected)
    if unexpected:
        raise SystemExit(
            "Unexpected generated figure files not listed in figure manifest:\n"
            + "\n".join(unexpected)
        )


def _manifest_asset_sources(figures_dir: Path) -> dict[str, object]:
    manifest_path = figures_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest.get("generated_asset_sources", {})
    if not isinstance(sources, dict):
        raise SystemExit("Figure manifest field `generated_asset_sources` must be an object.")
    return {str(Path(name).name): value for name, value in sources.items()}


def _validate_manifest_sources(required_figures: set[str], figures_dir: Path) -> None:
    sources = _manifest_asset_sources(figures_dir)
    missing = sorted(required_figures - set(sources))
    if missing:
        raise SystemExit("TeX-included figures missing source provenance:\n" + "\n".join(missing))
    findings: list[str] = []
    allowed_status = {"archive_neutral", "needs_final_archive"}
    for name in sorted(required_figures):
        source = sources[name]
        if not isinstance(source, dict):
            findings.append(f"{name}: generated_asset_sources entry must be an object.")
            continue
        generator = source.get("generator")
        if not isinstance(generator, dict):
            findings.append(f"{name}: source provenance is missing a generator object.")
        status = source.get("archive_status")
        if status not in allowed_status:
            findings.append(
                f"{name}: archive_status must be one of {sorted(allowed_status)}, got {status!r}."
            )
        data_inputs = source.get("data_inputs", [])
        if data_inputs is not None and not isinstance(data_inputs, list):
            findings.append(f"{name}: data_inputs must be a list when present.")
            continue
        for entry in data_inputs or []:
            findings.extend(_validate_manifest_input_reference(name, entry))
    if findings:
        raise SystemExit("Figure source provenance is malformed:\n" + "\n".join(findings))


def _table_manifest(tables_dir: Path) -> dict[str, object]:
    manifest_path = tables_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SystemExit("Table manifest must be a JSON object.")
    return manifest


def _manifest_tables(tables_dir: Path) -> set[str]:
    manifest = _table_manifest(tables_dir)
    tables: set[str] = set()
    for name in manifest.get("generated_tables", []):
        if isinstance(name, str):
            tables.add(Path(name).name)
    return tables


def _revision_table_manifest(tables_dir: Path) -> dict[str, object]:
    manifest_path = tables_dir / "revision_evidence_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SystemExit("Revision evidence table manifest must be a JSON object.")
    return manifest


def _revision_manifest_tables(tables_dir: Path) -> set[str]:
    outputs = _revision_table_manifest(tables_dir).get("outputs", {})
    if not isinstance(outputs, dict):
        raise SystemExit("Revision evidence manifest field `outputs` must be an object.")
    return {Path(str(name)).name for name in outputs}


def _distributed_colored_manifest(tables_dir: Path) -> dict[str, object]:
    manifest_path = tables_dir / DISTRIBUTED_COLORED_MANIFEST
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SystemExit("Distributed-colored table manifest must be a JSON object.")
    return manifest


def _distributed_colored_manifest_tables(tables_dir: Path) -> set[str]:
    outputs = _distributed_colored_manifest(tables_dir).get("outputs", {})
    if not isinstance(outputs, dict):
        raise SystemExit("Distributed-colored manifest field `outputs` must be an object.")
    return {Path(str(name)).name for name in outputs}


def _validate_distributed_colored_manifest(tables_dir: Path) -> None:
    manifest = _distributed_colored_manifest(tables_dir)
    if not manifest:
        return
    findings: list[str] = []
    exact = {
        "schema_id": "fenics-nonlinear-energies.distributed-colored-table-manifest",
        "schema_version": 1,
        "status": "admitted_correctness_only",
        "publication_evidence": True,
        "experiment_id": "EXP-DIST-001",
        "timing_claim_admissible": False,
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            findings.append(f"field {key} must equal {expected!r}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"distributed_colored_verification.tex"}:
        findings.append("outputs must bind exactly distributed_colored_verification.tex")
    else:
        for name, expected_hash in outputs.items():
            path = tables_dir / name
            if (
                not isinstance(expected_hash, str)
                or SHA256_RE.fullmatch(expected_hash) is None
                or not path.is_file()
                or sha256_file(path) != expected_hash
            ):
                findings.append(f"{name}: output is missing or its SHA-256 hash is stale")
    tools = manifest.get("tools")
    if not isinstance(tools, dict) or set(tools) != {"validator", "generator", "checker"}:
        findings.append("tools must bind validator, generator, and checker")
    else:
        for name, entry in sorted(tools.items()):
            if not isinstance(entry, dict):
                findings.append(f"tool {name}: record must be an object")
                continue
            raw = entry.get("path")
            expected_hash = entry.get("sha256")
            if (
                not isinstance(raw, str)
                or Path(raw).is_absolute()
                or ".." in Path(raw).parts
            ):
                findings.append(f"tool {name}: path is not safe and repository-relative")
                continue
            path = (REPO_ROOT / raw).resolve()
            try:
                path.relative_to(REPO_ROOT)
            except ValueError:
                findings.append(f"tool {name}: path resolves outside the repository")
            else:
                if not path.is_file() or expected_hash != sha256_file(path):
                    findings.append(f"tool {name}: file is missing or its hash is stale")
    source = manifest.get("source_campaign_manifest")
    if not isinstance(source, dict):
        findings.append("source_campaign_manifest must be an object")
    else:
        raw = source.get("path")
        expected_hash = source.get("sha256")
        if not isinstance(raw, str) or Path(raw).is_absolute() or ".." in Path(raw).parts:
            findings.append("source campaign path is not safe and repository-relative")
        else:
            path = (REPO_ROOT / raw).resolve()
            reproduction = (REPO_ROOT / "artifacts/reproduction").resolve()
            try:
                path.relative_to(reproduction)
            except ValueError:
                findings.append("source campaign manifest is outside artifacts/reproduction")
            else:
                if not path.is_file() or expected_hash != sha256_file(path):
                    findings.append("source campaign manifest is missing or its hash is stale")
    if findings:
        raise SystemExit(
            "Distributed-colored table provenance is malformed:\n" + "\n".join(findings)
        )


def _globalization_local_manifest(tables_dir: Path) -> dict[str, object]:
    manifest_path = tables_dir / GLOBALIZATION_LOCAL_MANIFEST
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SystemExit("Local-globalization table manifest must be a JSON object.")
    return manifest


def _globalization_local_manifest_tables(tables_dir: Path) -> set[str]:
    outputs = _globalization_local_manifest(tables_dir).get("outputs", {})
    if not isinstance(outputs, dict):
        raise SystemExit("Local-globalization manifest field `outputs` must be an object.")
    return {Path(str(name)).name for name in outputs}


def _validate_globalization_local_manifest(tables_dir: Path) -> None:
    manifest = _globalization_local_manifest(tables_dir)
    if not manifest:
        return
    findings: list[str] = []
    exact = {
        "schema_id": "fenics-nonlinear-energies.exp-glob-001-local-table-manifest",
        "schema_version": 1,
        "status": "admitted_bounded_local_outcomes",
        "publication_evidence": True,
        "experiment_id": "EXP-GLOB-001",
        "timing_claim_admissible": False,
        "population_robustness_claim_admissible": False,
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            findings.append(f"field {key} must equal {expected!r}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"globalization_local_status.tex"}:
        findings.append("outputs must bind exactly globalization_local_status.tex")
    else:
        for name, expected_hash in outputs.items():
            path = tables_dir / name
            if (
                not isinstance(expected_hash, str)
                or SHA256_RE.fullmatch(expected_hash) is None
                or not path.is_file()
                or sha256_file(path) != expected_hash
            ):
                findings.append(f"{name}: output is missing or its SHA-256 hash is stale")
    tools = manifest.get("tools")
    if not isinstance(tools, dict) or set(tools) != {"validator", "generator", "checker"}:
        findings.append("tools must bind validator, generator, and checker")
    else:
        for name, entry in sorted(tools.items()):
            if not isinstance(entry, dict):
                findings.append(f"tool {name}: record must be an object")
                continue
            raw = entry.get("path")
            expected_hash = entry.get("sha256")
            if (
                not isinstance(raw, str)
                or Path(raw).is_absolute()
                or ".." in Path(raw).parts
            ):
                findings.append(f"tool {name}: path is not safe and repository-relative")
                continue
            path = (REPO_ROOT / raw).resolve()
            try:
                path.relative_to(REPO_ROOT)
            except ValueError:
                findings.append(f"tool {name}: path resolves outside the repository")
            else:
                if not path.is_file() or expected_hash != sha256_file(path):
                    findings.append(f"tool {name}: file is missing or its hash is stale")
    source = manifest.get("source_campaign_manifest")
    if not isinstance(source, dict):
        findings.append("source_campaign_manifest must be an object")
    else:
        raw = source.get("path")
        expected_hash = source.get("sha256")
        if not isinstance(raw, str) or Path(raw).is_absolute() or ".." in Path(raw).parts:
            findings.append("source campaign path is not safe and repository-relative")
        else:
            path = (REPO_ROOT / raw).resolve()
            reproduction = (REPO_ROOT / "artifacts/reproduction").resolve()
            try:
                path.relative_to(reproduction)
            except ValueError:
                findings.append("source campaign manifest is outside artifacts/reproduction")
            else:
                if not path.is_file() or expected_hash != sha256_file(path):
                    findings.append("source campaign manifest is missing or its hash is stale")
    if findings:
        raise SystemExit(
            "Local-globalization table provenance is malformed:\n" + "\n".join(findings)
        )


def _stopping_local_manifest(tables_dir: Path) -> dict[str, object]:
    manifest_path = tables_dir / STOPPING_LOCAL_MANIFEST
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SystemExit("Local-stopping table manifest must be a JSON object.")
    return manifest


def _stopping_local_manifest_tables(tables_dir: Path) -> set[str]:
    outputs = _stopping_local_manifest(tables_dir).get("outputs", {})
    if not isinstance(outputs, dict):
        raise SystemExit("Local-stopping manifest field `outputs` must be an object.")
    return {Path(str(name)).name for name in outputs}


def _validate_stopping_local_manifest(tables_dir: Path) -> None:
    manifest = _stopping_local_manifest(tables_dir)
    if not manifest:
        return
    findings: list[str] = []
    exact = {
        "schema_id": "fenics-nonlinear-energies.exp-stop-001-local-table-manifest",
        "schema_version": 1,
        "status": "admitted_local_calibration_cluster_deferred",
        "publication_evidence": True,
        "experiment_id": "EXP-STOP-001",
        "complete_exp_stop_pass": False,
        "timing_claim_admissible": False,
        "population_robustness_claim_admissible": False,
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            findings.append(f"field {key} must equal {expected!r}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"stopping_local_status.tex"}:
        findings.append("outputs must bind exactly stopping_local_status.tex")
    else:
        for name, expected_hash in outputs.items():
            path = tables_dir / name
            if (
                not isinstance(expected_hash, str)
                or SHA256_RE.fullmatch(expected_hash) is None
                or not path.is_file()
                or sha256_file(path) != expected_hash
            ):
                findings.append(f"{name}: output is missing or its SHA-256 hash is stale")
    tools = manifest.get("tools")
    if not isinstance(tools, dict) or set(tools) != {"validator", "generator", "checker"}:
        findings.append("tools must bind validator, generator, and checker")
    else:
        for name, entry in sorted(tools.items()):
            if not isinstance(entry, dict):
                findings.append(f"tool {name}: record must be an object")
                continue
            raw = entry.get("path")
            expected_hash = entry.get("sha256")
            if (
                not isinstance(raw, str)
                or Path(raw).is_absolute()
                or ".." in Path(raw).parts
            ):
                findings.append(f"tool {name}: path is not safe and repository-relative")
                continue
            path = (REPO_ROOT / raw).resolve()
            try:
                path.relative_to(REPO_ROOT)
            except ValueError:
                findings.append(f"tool {name}: path resolves outside the repository")
            else:
                if not path.is_file() or expected_hash != sha256_file(path):
                    findings.append(f"tool {name}: file is missing or its hash is stale")
    for source_key, expected_name in (
        ("source_plan", "plan.json"),
        ("source_analysis", "analysis.json"),
    ):
        source = manifest.get(source_key)
        if not isinstance(source, dict):
            findings.append(f"{source_key} must be an object")
            continue
        raw = source.get("path")
        expected_hash = source.get("sha256")
        if not isinstance(raw, str) or Path(raw).is_absolute() or ".." in Path(raw).parts:
            findings.append(f"{source_key} path is not safe and repository-relative")
            continue
        path = (REPO_ROOT / raw).resolve()
        reproduction = (REPO_ROOT / "artifacts/reproduction").resolve()
        try:
            path.relative_to(reproduction)
        except ValueError:
            findings.append(f"{source_key} is outside artifacts/reproduction")
        else:
            if path.name != expected_name:
                findings.append(f"{source_key} does not identify {expected_name}")
            elif not path.is_file() or expected_hash != sha256_file(path):
                findings.append(f"{source_key} is missing or its hash is stale")
    if findings:
        raise SystemExit(
            "Local-stopping table provenance is malformed:\n" + "\n".join(findings)
        )


def _stopping_submission_manifest(tables_dir: Path) -> dict[str, object]:
    manifest_path = tables_dir / STOPPING_SUBMISSION_MANIFEST
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SystemExit("Stopping-submission table manifest must be a JSON object.")
    return manifest


def _stopping_submission_manifest_tables(tables_dir: Path) -> set[str]:
    outputs = _stopping_submission_manifest(tables_dir).get("outputs", {})
    if not isinstance(outputs, dict):
        raise SystemExit(
            "Stopping-submission manifest field outputs must be an object."
        )
    return {Path(str(name)).name for name in outputs}


def _validate_stopping_submission_manifest(tables_dir: Path) -> None:
    manifest = _stopping_submission_manifest(tables_dir)
    if not manifest:
        return
    findings: list[str] = []
    exact = {
        "schema_id": (
            "fenics-nonlinear-energies.exp-stop-001-submission-table-manifest"
        ),
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
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            findings.append(f"field {key} must equal {expected!r}")
    expected_counts = {
        "executions": 45,
        "admitted_records": 43,
        "accepted_comparisons": 28,
        "rejected_comparisons": 15,
        "endpoint_censored_comparisons": 2,
        "reference_self_comparisons": 11,
        "accepted_nonreference_candidates": 17,
    }
    if manifest.get("presentation_counts") != expected_counts:
        findings.append("presentation_counts differ from the admitted 45-row subset")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {
        "stopping_submission_status.tex"
    }:
        findings.append("outputs must bind exactly stopping_submission_status.tex")
    else:
        for name, expected_hash in outputs.items():
            path = tables_dir / name
            if (
                not isinstance(expected_hash, str)
                or SHA256_RE.fullmatch(expected_hash) is None
                or not path.is_file()
                or sha256_file(path) != expected_hash
            ):
                findings.append(f"{name}: output is missing or its SHA-256 hash is stale")

    tools = manifest.get("tools")
    if not isinstance(tools, dict) or set(tools) != {"generator", "checker"}:
        findings.append("tools must bind generator and checker")
    else:
        for name, entry in sorted(tools.items()):
            if not isinstance(entry, dict):
                findings.append(f"tool {name}: record must be an object")
                continue
            raw = entry.get("path")
            expected_hash = entry.get("sha256")
            if (
                not isinstance(raw, str)
                or Path(raw).is_absolute()
                or ".." in Path(raw).parts
            ):
                findings.append(f"tool {name}: path is not safe and repository-relative")
                continue
            path = (REPO_ROOT / raw).resolve()
            try:
                path.relative_to(REPO_ROOT)
            except ValueError:
                findings.append(f"tool {name}: path resolves outside the repository")
            else:
                if not path.is_file() or expected_hash != sha256_file(path):
                    findings.append(f"tool {name}: file is missing or its hash is stale")

    source_manifest = manifest.get("source_manifest")
    if not isinstance(source_manifest, dict):
        findings.append("source_manifest must be an object")
    else:
        raw = source_manifest.get("path")
        expected_hash = source_manifest.get("sha256")
        path = (
            (REPO_ROOT / raw).resolve()
            if isinstance(raw, str)
            and not Path(raw).is_absolute()
            and ".." not in Path(raw).parts
            else None
        )
        canonical = (tables_dir / STOPPING_LOCAL_MANIFEST).resolve()
        if path != canonical:
            findings.append("source_manifest must identify stopping_local_manifest.json")
        elif not path.is_file() or expected_hash != sha256_file(path):
            findings.append("source_manifest is missing or has a stale hash")

    source_analysis = manifest.get("source_analysis")
    if not isinstance(source_analysis, dict):
        findings.append("source_analysis must be an object")
    else:
        raw = source_analysis.get("path")
        expected_hash = source_analysis.get("sha256")
        if (
            not isinstance(raw, str)
            or Path(raw).is_absolute()
            or ".." in Path(raw).parts
        ):
            findings.append("source_analysis path is not safe and repository-relative")
        else:
            path = (REPO_ROOT / raw).resolve()
            reproduction = (REPO_ROOT / "artifacts/reproduction").resolve()
            try:
                path.relative_to(reproduction)
            except ValueError:
                findings.append("source_analysis is outside artifacts/reproduction")
            else:
                if (
                    path.name != "analysis.json"
                    or not path.is_file()
                    or expected_hash != sha256_file(path)
                ):
                    findings.append("source_analysis is missing or has a stale hash")
    if findings:
        raise SystemExit(
            "Stopping-submission table provenance is malformed:\n"
            + "\n".join(findings)
        )


def _validate_no_unexpected_generated_tables(
    tables_dir: Path,
    *,
    required_tables: set[str],
    manifest_tables: set[str],
    allow_unreferenced_tables: bool,
) -> None:
    actual = {path.name for path in tables_dir.iterdir() if path.is_file() and path.suffix == ".tex"}
    unexpected_files = sorted(actual - manifest_tables)
    unused_manifest_tables = (
        []
        if allow_unreferenced_tables
        else sorted(manifest_tables - required_tables)
    )
    findings: list[str] = []
    if unexpected_files:
        findings.append(
            "Generated table files not listed in table manifest:\n" + "\n".join(unexpected_files)
        )
    if unused_manifest_tables:
        findings.append(
            "Generated tables listed in manifest but not included by the manuscript:\n"
            + "\n".join(unused_manifest_tables)
        )
    if findings:
        raise SystemExit("\n".join(findings))


def _validate_revision_table_manifest(
    required_tables: set[str], tables_dir: Path
) -> None:
    if not required_tables:
        return
    manifest = _revision_table_manifest(tables_dir)
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        raise SystemExit("Revision evidence manifest field `outputs` must be an object.")
    findings: list[str] = []
    for name in sorted(required_tables):
        expected = outputs.get(name)
        path = tables_dir / name
        if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
            findings.append(f"{name}: revision output is missing a valid SHA-256 hash")
        elif not path.is_file() or sha256_file(path) != expected:
            findings.append(f"{name}: revision output hash is stale")

    generator = manifest.get("generator")
    generator_sha = manifest.get("generator_sha256")
    if not isinstance(generator, str) or Path(generator).is_absolute() or ".." in Path(generator).parts:
        findings.append("revision table generator path is not safe and repository-relative")
    else:
        generator_path = (REPO_ROOT / generator).resolve()
        try:
            generator_path.relative_to(REPO_ROOT)
        except ValueError:
            findings.append("revision table generator resolves outside the repository")
        else:
            if not generator_path.is_file():
                findings.append("revision table generator is missing")
            elif not isinstance(generator_sha, str) or sha256_file(generator_path) != generator_sha:
                findings.append("revision table generator hash is stale")

    inputs = manifest.get("inputs", {})
    if not isinstance(inputs, dict) or len(inputs) != 14:
        findings.append("revision table manifest must bind exactly 14 inputs")
    else:
        for key, entry in sorted(inputs.items()):
            if not isinstance(entry, dict):
                findings.append(f"revision input {key}: entry must be an object")
                continue
            raw_path = entry.get("path")
            expected = entry.get("sha256")
            if not isinstance(raw_path, str) or Path(raw_path).is_absolute() or ".." in Path(raw_path).parts:
                findings.append(f"revision input {key}: path is not safe and repository-relative")
                continue
            path = (REPO_ROOT / raw_path).resolve()
            try:
                path.relative_to(REPO_ROOT)
            except ValueError:
                findings.append(f"revision input {key}: path resolves outside the repository")
                continue
            if not path.is_file():
                findings.append(f"revision input {key}: file is missing")
            elif not isinstance(expected, str) or sha256_file(path) != expected:
                findings.append(f"revision input {key}: SHA-256 hash is stale")
    if findings:
        raise SystemExit("Revision evidence table provenance is malformed:\n" + "\n".join(findings))


def _table_manifest_sources(tables_dir: Path) -> dict[str, object]:
    manifest = _table_manifest(tables_dir)
    sources = manifest.get("generated_table_sources", {})
    if not isinstance(sources, dict):
        raise SystemExit("Table manifest field `generated_table_sources` must be an object.")
    return {str(Path(name).name): value for name, value in sources.items()}


def _validate_manifest_input_reference(label: str, entry: object) -> list[str]:
    findings: list[str] = []
    if not isinstance(entry, dict):
        return [f"{label}: data input must be an object, got {type(entry).__name__}."]
    kind = entry.get("kind")
    if kind != "repository_path":
        return [f"{label}: data input kind must be 'repository_path', got {kind!r}."]
    raw_path = entry.get("path")
    if not isinstance(raw_path, str):
        return [f"{label}: repository_path input is missing a string `path` field."]
    if raw_path.startswith("/") or ".." in Path(raw_path).parts:
        return [f"{label}: repository_path input is not a safe repo-relative path: {raw_path}"]
    candidate = (REPO_ROOT / raw_path).resolve()
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError:
        return [f"{label}: repository_path resolves outside the repository: {raw_path}"]
    if not candidate.exists():
        findings.append(f"{label}: repository_path input is missing: {raw_path}")
        return findings
    expected_hash = entry.get("sha256")
    if not isinstance(expected_hash, str) or SHA256_RE.fullmatch(expected_hash) is None:
        findings.append(f"{label}: repository_path input is missing a valid SHA-256 hash: {raw_path}")
    elif sha256_file(candidate) != expected_hash:
        findings.append(f"{label}: repository_path input hash is stale: {raw_path}")
    return findings


def _validate_table_manifest_sources(required_tables: set[str], tables_dir: Path) -> None:
    manifest_tables = _manifest_tables(tables_dir)
    missing_tables = sorted(required_tables - manifest_tables)
    if missing_tables:
        raise SystemExit("TeX-included generated tables missing from table manifest:\n" + "\n".join(missing_tables))
    sources = _table_manifest_sources(tables_dir)
    missing_sources = sorted(required_tables - set(sources))
    if missing_sources:
        raise SystemExit("TeX-included generated tables missing source provenance:\n" + "\n".join(missing_sources))
    findings: list[str] = []
    allowed_status = {"archive_neutral", "needs_final_archive"}
    for name in sorted(required_tables):
        source = sources[name]
        if not isinstance(source, dict):
            findings.append(f"{name}: generated_table_sources entry must be an object.")
            continue
        generator = source.get("generator")
        if not isinstance(generator, dict):
            findings.append(f"{name}: source provenance is missing a generator object.")
        status = source.get("archive_status")
        if status not in allowed_status:
            findings.append(
                f"{name}: archive_status must be one of {sorted(allowed_status)}, got {status!r}."
            )
        data_inputs = source.get("data_inputs", [])
        if not isinstance(data_inputs, list):
            findings.append(f"{name}: data_inputs must be a list.")
            continue
        for entry in data_inputs:
            findings.extend(_validate_manifest_input_reference(name, entry))
    if findings:
        raise SystemExit("Table source provenance is malformed:\n" + "\n".join(findings))


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


def _iter_table_manifest_inputs(manifest: dict[str, object]) -> list[tuple[str, object]]:
    inputs = manifest.get("generated_table_inputs", {})
    if not isinstance(inputs, dict):
        raise SystemExit("Table manifest field `generated_table_inputs` must be an object.")
    entries: list[tuple[str, object]] = []
    for asset, values in sorted(inputs.items()):
        if not isinstance(asset, str):
            raise SystemExit("Table manifest asset names must be strings.")
        if not isinstance(values, list):
            raise SystemExit(f"Table manifest inputs for {asset!r} must be a list.")
        for value in values:
            entries.append((f"table:{asset}", value))
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


def _validate_archive_neutral_table_manifest(tables_dir: Path) -> None:
    manifest_path = tables_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Table manifest missing for archive-neutral check: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    findings: list[str] = []
    paths_to_scan: set[Path] = set()
    for asset, entry in _iter_table_manifest_inputs(manifest):
        entry_findings, entry_paths = _archive_neutral_findings_for_input(asset, entry)
        findings.extend(entry_findings)
        paths_to_scan.update(path for path in entry_paths if path.suffix.lower() in TEXT_SCAN_SUFFIXES)
    for path in sorted(paths_to_scan):
        findings.extend(_find_banned_snippets(path, PROVENANCE_BANNED_SNIPPETS))
    if findings:
        raise SystemExit("Archive-neutral table provenance check failed:\n" + "\n".join(findings))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that the paper asset generation produced the expected files.")
    add_paper_bundle_root_argument(parser)
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
    _validate_no_unexpected_generated_figures(args.figures_dir, manifest_assets)
    _validate_manifest_sources(required_figures, args.figures_dir)
    base_table_manifest = _table_manifest(args.tables_dir)
    base_manifest_tables = _manifest_tables(args.tables_dir)
    revision_manifest_tables = _revision_manifest_tables(args.tables_dir)
    distributed_manifest = _distributed_colored_manifest(args.tables_dir)
    distributed_manifest_tables = _distributed_colored_manifest_tables(args.tables_dir)
    globalization_manifest = _globalization_local_manifest(args.tables_dir)
    globalization_manifest_tables = _globalization_local_manifest_tables(args.tables_dir)
    stopping_manifest = _stopping_local_manifest(args.tables_dir)
    stopping_manifest_tables = _stopping_local_manifest_tables(args.tables_dir)
    stopping_submission_manifest = _stopping_submission_manifest(args.tables_dir)
    stopping_submission_manifest_tables = _stopping_submission_manifest_tables(
        args.tables_dir
    )
    all_manifest_tables = (
        base_manifest_tables
        | revision_manifest_tables
        | distributed_manifest_tables
        | globalization_manifest_tables
        | stopping_manifest_tables
        | stopping_submission_manifest_tables
    )
    _validate_no_unexpected_generated_tables(
        args.tables_dir,
        required_tables=required_tables,
        manifest_tables=all_manifest_tables,
        allow_unreferenced_tables=bool(
            base_table_manifest.get("allow_unreferenced_tables") is True
            or distributed_manifest.get("allow_unreferenced_tables") is True
            or globalization_manifest.get("allow_unreferenced_tables") is True
            or stopping_manifest.get("allow_unreferenced_tables") is True
            or stopping_submission_manifest.get("allow_unreferenced_tables") is True
        ),
    )
    missing_table_manifests = sorted(required_tables - all_manifest_tables)
    if missing_table_manifests:
        raise SystemExit(
            "TeX-included generated tables missing from all table manifests:\n"
            + "\n".join(missing_table_manifests)
        )
    _validate_table_manifest_sources(
        required_tables & base_manifest_tables, args.tables_dir
    )
    _validate_revision_table_manifest(
        required_tables & revision_manifest_tables, args.tables_dir
    )
    _validate_distributed_colored_manifest(args.tables_dir)
    _validate_globalization_local_manifest(args.tables_dir)
    _validate_stopping_local_manifest(args.tables_dir)
    _validate_stopping_submission_manifest(args.tables_dir)
    provenance_targets = _provenance_targets(args.tex, seen_tex, required_tables, args.figures_dir, args.tables_dir)
    _validate_provenance_text(provenance_targets)
    if args.archive_neutral:
        _validate_archive_neutral_manifest(args.figures_dir)
        _validate_archive_neutral_table_manifest(args.tables_dir)
    print(
        f"Paper assets validated ({len(required_figures)} figures, {len(required_tables)} tables); "
        f"provenance scan passed ({len(provenance_targets)} files)."
    )


if __name__ == "__main__":
    main()

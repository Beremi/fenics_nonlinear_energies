#!/usr/bin/env python3
"""Execute, finalize, and verify clean revision-publication source evidence.

This module is the producer-side counterpart of
``paper/scripts/admit_revision_publication_evidence.py``.  It deliberately
does not turn an arbitrary directory of results into publication evidence.
Publication sources must first be produced by the ``execute`` subcommand from
one immutable, clean Git commit.  ``execute`` uses argv directly (never a
shell), captures the effective environment and logs, and writes a hash-bound
receipt.  ``finalize`` then:

* accepts only successful receipts from that experiment commit;
* permits a clean release HEAD only when the experiment commit is its ancestor;
* rehashes every producer, configuration, input, raw output, and run record;
* rejects pilot/dirty declarations and tampering;
* preserves each raw JSON source under ``_publication_staging``;
* writes a decorated, table-facing copy plus the companion manifests required
  by the independent admission boundary; and
* validates and copies strict publication run records for EXP-MC-001 and
  EXP-DIST-001 without altering them.

The staging directory is part of the evidence archive.  Do not delete it after
finalization: its hashes are the audit trail showing exactly which raw producer
payload was decorated.  No scheduler command is issued by this program; on an
HPC system, invoke ``execute`` *inside* an already released allocation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.benchmark.run_record import (  # noqa: E402
    RUN_RECORD_SCHEMA_ID,
    RUN_RECORD_SCHEMA_VERSION,
    ExperimentPreflightError,
    RunRecordValidationError,
    atomic_write_json,
    check_experiment_preflight,
    validate_run_record,
)


PLAN_SCHEMA_ID = "fenics-nonlinear-energies.revision-publication-execution-plan"
PLAN_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_ID = "fenics-nonlinear-energies.revision-publication-execution-receipt"
RECEIPT_SCHEMA_VERSION = 1
SOURCE_PROVENANCE_SCHEMA_ID = (
    "fenics-nonlinear-energies.revision-publication-source-provenance"
)
SOURCE_PROVENANCE_SCHEMA_VERSION = 1
COMPANION_SCHEMA_ID = "fenics-nonlinear-energies.revision-publication-companion"
COMPANION_SCHEMA_VERSION = 1
FINALIZATION_SCHEMA_ID = "fenics-nonlinear-energies.revision-publication-finalization"
FINALIZATION_SCHEMA_VERSION = 1
FINALIZATION_STATUS = "finalized_clean_publication_sources"

STAGING_DIRECTORY = "_publication_staging"
RECEIPT_DIRECTORY = "_publication_receipts"
LOG_DIRECTORY = "_publication_logs"
FINALIZATION_MANIFEST = "publication/finalization_manifest.json"
FINALIZER_PATH = Path("experiments/analysis/finalize_revision_publication_campaign.py")

HEX40_RE = re.compile(r"[0-9a-f]{40}")
HEX64_RE = re.compile(r"[0-9a-f]{64}")
SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}")

ENVIRONMENT_VARIABLES = (
    "JAX_PLATFORMS",
    "JAX_ENABLE_X64",
    "XLA_FLAGS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "PETSC_ARCH",
    "PETSC_DIR",
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NTASKS",
    "SLURM_CPUS_PER_TASK",
)
PACKAGE_NAMES = (
    "numpy",
    "scipy",
    "jax",
    "jaxlib",
    "mpi4py",
    "petsc4py",
    "dolfinx",
    "h5py",
)


@dataclass(frozen=True, slots=True)
class SourceSpec:
    key: str
    relative_path: Path
    producer_path: Path
    companion_manifest: Path
    experiment_id: str
    native_version_field: str | None = None
    native_version: int | None = None
    native_schema_id: str | None = None
    native_schema_name: str | None = None
    run_records: tuple[Path, ...] = ()


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        "plaplace",
        Path("EXP-VAL-001/plaplace_manufactured.json"),
        Path("experiments/runners/run_manufactured_plaplace_verification.py"),
        Path("EXP-VAL-001/pilot_manifest.json"),
        "EXP-VAL-001-PLAPLACE-MANUFACTURED",
        "schema_version",
        1,
    ),
    SourceSpec(
        "ginzburg_landau",
        Path("EXP-VAL-001/ginzburg_landau_manufactured.json"),
        Path("experiments/runners/run_manufactured_ginzburg_landau_verification.py"),
        Path("EXP-VAL-001/pilot_manifest.json"),
        "EXP-VAL-001-GINZBURG-LANDAU-MANUFACTURED",
        "schema_version",
        1,
    ),
    SourceSpec(
        "hyperelastic_patch",
        Path("EXP-VAL-001/hyperelastic_affine_patch.json"),
        Path("experiments/runners/run_hyperelastic_affine_patch_verification.py"),
        Path("EXP-VAL-001/pilot_manifest.json"),
        "EXP-VAL-001-HYPERELASTIC-AFFINE-PATCH",
        "schema_version",
        1,
    ),
    SourceSpec(
        "hyperelastic_nonaffine",
        Path("EXP-VAL-001/hyperelastic_nonaffine_quadrature_refinement_v2/result.json"),
        Path("experiments/runners/run_manufactured_hyperelastic_verification.py"),
        Path("EXP-VAL-001/pilot_manifest.json"),
        "EXP-VAL-001-HYPERELASTIC-NONAFFINE-MANUFACTURED",
        "schema_version",
        2,
    ),
    SourceSpec(
        "smooth_derivatives",
        Path("EXP-DERIV-001/smooth_fixed_element_v1.json"),
        Path("experiments/runners/run_smooth_element_derivative_verification.py"),
        Path("EXP-DERIV-001/pilot_manifest.json"),
        "EXP-DERIV-001-SMOOTH-FIXED-ELEMENT",
    ),
    SourceSpec(
        "p1_derivatives",
        Path("EXP-DERIV-001/p1_l1_fixed_element_v2.json"),
        Path("experiments/runners/run_paper_derivative_verification.py"),
        Path("EXP-DERIV-001/pilot_manifest.json"),
        "EXP-DERIV-001-P3D-FIXED-ELEMENT",
    ),
    SourceSpec(
        "p2_derivatives",
        Path("EXP-DERIV-001/p2_l1_fixed_element_v2.json"),
        Path("experiments/runners/run_paper_derivative_verification.py"),
        Path("EXP-DERIV-001/pilot_manifest.json"),
        "EXP-DERIV-001-P3D-FIXED-ELEMENT",
    ),
    SourceSpec(
        "p4_derivatives",
        Path("EXP-DERIV-001/p4_l1_fixed_element_v2.json"),
        Path("experiments/runners/run_paper_derivative_verification.py"),
        Path("EXP-DERIV-001/pilot_manifest.json"),
        "EXP-DERIV-001-P3D-FIXED-ELEMENT",
    ),
    SourceSpec(
        "material_point",
        Path("EXP-MC-001/material_point_verification.json"),
        Path("experiments/runners/run_plasticity3d_material_point_verification.py"),
        Path("EXP-MC-001/pilot_manifest.json"),
        "EXP-MC-001",
        "schema_version",
        1,
        native_schema_name="plasticity3d_material_point_verification",
        run_records=(Path("EXP-MC-001/run_record.json"),),
    ),
    SourceSpec(
        "distribution",
        Path("EXP-DIST-001/distribution_equivalence.json"),
        Path("experiments/runners/run_hyperelasticity_distribution_equivalence.py"),
        Path("EXP-DIST-001/pilot_manifest.json"),
        "EXP-DIST-001",
        native_schema_id="fenics-nonlinear-energies.exp-dist-he-comparison",
        native_version=1,
        run_records=(
            Path("EXP-DIST-001/run_record_np1.json"),
            Path("EXP-DIST-001/run_record_np2.json"),
            Path("EXP-DIST-001/run_record_np4.json"),
        ),
    ),
    SourceSpec(
        "p1_quadrature",
        Path("EXP-DISC-001/p1_l1_fixed_state_quadrature_v2.json"),
        Path("experiments/runners/run_plasticity3d_fixed_state_quadrature.py"),
        Path("EXP-DISC-001/pilot_manifest.json"),
        "EXP-DISC-001-P3D-FIXED-STATE-QUADRATURE",
    ),
    SourceSpec(
        "p2_quadrature",
        Path("EXP-DISC-001/p2_l1_fixed_state_quadrature_v2.json"),
        Path("experiments/runners/run_plasticity3d_fixed_state_quadrature.py"),
        Path("EXP-DISC-001/pilot_manifest.json"),
        "EXP-DISC-001-P3D-FIXED-STATE-QUADRATURE",
    ),
    SourceSpec(
        "p4_quadrature",
        Path("EXP-DISC-001/p4_l1_fixed_state_quadrature_v2.json"),
        Path("experiments/runners/run_plasticity3d_fixed_state_quadrature.py"),
        Path("EXP-DISC-001/pilot_manifest.json"),
        "EXP-DISC-001-P3D-FIXED-STATE-QUADRATURE",
    ),
    SourceSpec(
        "route_analysis",
        Path("EXP-ROUTE-001/analysis_contract_v1/analysis.json"),
        Path("experiments/analysis/analyze_plasticity3d_route_cost_model.py"),
        Path("EXP-ROUTE-001/analysis_contract_v1/manifest.json"),
        "EXP-ROUTE-001",
        "analysis_schema_version",
        1,
    ),
)
SOURCE_BY_KEY = {spec.key: spec for spec in SOURCE_SPECS}


class FinalizationError(RuntimeError):
    """Raised when evidence cannot cross the publication finalization boundary."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizationError(f"{path} must contain a top-level JSON object")
    if not _finite_tree(value):
        raise FinalizationError(f"{path} contains a non-finite scientific value")
    return value


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON token {token}")


def _finite_tree(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _finite_tree(child) for key, child in value.items())
    if isinstance(value, list):
        return all(_finite_tree(child) for child in value)
    return False


def _canonical_relative(raw: str | Path, *, label: str) -> Path:
    text = str(raw)
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise FinalizationError(f"{label} must be a canonical relative path without '.' or '..': {text!r}")
    normalized = Path(path.as_posix())
    if normalized.as_posix() != text.replace(os.sep, "/"):
        raise FinalizationError(f"{label} is not canonical: {text!r}")
    return normalized


def _confined(root: Path, relative: str | Path, *, label: str, require_exists: bool = False) -> Path:
    rel = _canonical_relative(relative, label=label)
    root = root.resolve()
    candidate = root / rel
    resolved = candidate.resolve(strict=require_exists)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FinalizationError(f"{label} escapes {root}: {rel.as_posix()}") from exc
    if require_exists and not resolved.exists():
        raise FinalizationError(f"{label} does not exist: {rel.as_posix()}")
    if candidate.is_symlink() or (require_exists and resolved != candidate.absolute()):
        raise FinalizationError(f"{label} may not be a symlink or traverse one: {rel.as_posix()}")
    return candidate


def _git(repo_root: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown Git error"
        raise FinalizationError(f"Git command failed: {detail}")
    return completed.stdout


def _git_head(repo_root: Path) -> str:
    commit = _git(repo_root, "rev-parse", "--verify", "HEAD").strip().lower()
    if not HEX40_RE.fullmatch(commit):
        raise FinalizationError(f"Git HEAD is not a full SHA-1 commit: {commit!r}")
    return commit


def _git_clean(repo_root: Path) -> bool:
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    return not bool(status.strip())


def _require_clean_head(repo_root: Path) -> str:
    head = _git_head(repo_root)
    if not _git_clean(repo_root):
        raise FinalizationError("publication execution/finalization requires a clean worktree")
    return head


def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise FinalizationError(completed.stderr.strip() or "cannot evaluate Git ancestry")
    return completed.returncode == 0


def _committed_file_sha256(repo_root: Path, commit: str, relative: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{commit}:{relative.as_posix()}"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise FinalizationError(
            f"{relative.as_posix()} is not an immutable tracked file at experiment commit {commit}"
        )
    return hashlib.sha256(completed.stdout).hexdigest()


def _environment_snapshot(overrides: Mapping[str, str]) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    variables = {
        name: os.environ.get(name)
        for name in ENVIRONMENT_VARIABLES
        if os.environ.get(name) is not None
    }
    variables.update({str(key): str(value) for key, value in overrides.items()})
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "packages": packages,
        "selected_variables": dict(sorted(variables.items())),
    }


def _expand_argv(
    argv: Sequence[str], *, repo_root: Path, evidence_root: Path, staging_root: Path
) -> list[str]:
    replacements = {
        "{repo_root}": str(repo_root.resolve()),
        "{evidence_root}": str(evidence_root.resolve()),
        "{staging_root}": str(staging_root.resolve()),
        "{python}": str(Path(sys.executable).resolve()),
    }
    expanded: list[str] = []
    for raw in argv:
        if not isinstance(raw, str) or not raw:
            raise FinalizationError("every command argv item must be a non-empty string")
        value = raw
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        if "{" in value or "}" in value:
            raise FinalizationError(f"unsupported or unmatched argv template token in {raw!r}")
        expanded.append(value)
    return expanded


def _plan_command_map(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    commands = plan.get("commands")
    if not isinstance(commands, list) or not commands:
        raise FinalizationError("execution plan commands must be a non-empty array")
    result: dict[str, dict[str, Any]] = {}
    seen_keys: set[str] = set()
    for index, raw in enumerate(commands):
        if not isinstance(raw, Mapping):
            raise FinalizationError(f"plan command {index} must be an object")
        command = dict(raw)
        command_id = command.get("id")
        if not isinstance(command_id, str) or not SAFE_ID_RE.fullmatch(command_id):
            raise FinalizationError(f"plan command {index} has an invalid id")
        if command_id in result:
            raise FinalizationError(f"duplicate plan command id {command_id!r}")
        source_keys = command.get("source_keys")
        if not isinstance(source_keys, list):
            raise FinalizationError(f"plan command {command_id!r} must declare source_keys")
        if any(not isinstance(key, str) or key not in SOURCE_BY_KEY for key in source_keys):
            raise FinalizationError(f"plan command {command_id!r} has an unknown source key")
        duplicates = seen_keys.intersection(source_keys)
        if duplicates:
            raise FinalizationError(f"source keys assigned more than once: {sorted(duplicates)}")
        seen_keys.update(source_keys)
        producer = _canonical_relative(command.get("producer", ""), label="command producer")
        if source_keys:
            producers = {SOURCE_BY_KEY[key].producer_path for key in source_keys}
            if len(producers) != 1:
                raise FinalizationError(f"plan command {command_id!r} mixes source producers")
            if producer != next(iter(producers)):
                raise FinalizationError(
                    f"plan command {command_id!r} producer must be {next(iter(producers)).as_posix()}"
                )
        elif command.get("role") != "preparation":
            raise FinalizationError(
                f"source-free command {command_id!r} must declare role='preparation'"
            )
        argv = command.get("argv")
        if not isinstance(argv, list) or not argv:
            raise FinalizationError(f"plan command {command_id!r} argv must be non-empty")
        environment = command.get("environment", {})
        if not isinstance(environment, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items()
        ):
            raise FinalizationError(f"plan command {command_id!r} environment must map strings to strings")
        for field in ("configuration_files", "input_files", "expected_artifacts"):
            values = command.get(field, [])
            if not isinstance(values, list):
                raise FinalizationError(f"plan command {command_id!r} {field} must be an array")
        if source_keys and not command.get("configuration_files"):
            raise FinalizationError(
                f"source command {command_id!r} must bind at least one tracked protocol/configuration file"
            )
        if not source_keys and not command.get("expected_artifacts"):
            raise FinalizationError(
                f"preparation command {command_id!r} must declare expected_artifacts"
            )
        if "route_analysis" in source_keys:
            endpoint_path = command.get("route_endpoint_analysis")
            _canonical_relative(endpoint_path or "", label="route_endpoint_analysis")
            declared_staging_inputs = {
                str(item.get("path"))
                for item in command.get("input_files", [])
                if isinstance(item, Mapping) and item.get("scope") == "staging"
            }
            if endpoint_path not in declared_staging_inputs:
                raise FinalizationError(
                    "route_endpoint_analysis must also be a hash-bound staging input"
                )
        result[command_id] = command
    plan_kind = plan.get("plan_kind", "source_campaign")
    if plan_kind not in {"source_campaign", "dependency_preparation"}:
        raise FinalizationError("execution plan plan_kind must be source_campaign or dependency_preparation")
    expected = set(SOURCE_BY_KEY)
    if plan_kind == "source_campaign" and seen_keys != expected:
        raise FinalizationError(
            "execution plan must assign exactly the 14 configured sources; "
            f"missing={sorted(expected - seen_keys)}, extra={sorted(seen_keys - expected)}"
        )
    if plan_kind == "dependency_preparation" and seen_keys:
        raise FinalizationError("dependency_preparation commands may not write table-facing sources")
    return result


def load_plan(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    plan = _read_json(path)
    if plan.get("schema_id") != PLAN_SCHEMA_ID or plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise FinalizationError(f"execution plan must use {PLAN_SCHEMA_ID} v{PLAN_SCHEMA_VERSION}")
    campaign = plan.get("campaign_id")
    if not isinstance(campaign, str) or not SAFE_ID_RE.fullmatch(campaign):
        raise FinalizationError("execution plan campaign_id is invalid")
    commit = str(plan.get("experiment_commit", "")).lower()
    if not HEX40_RE.fullmatch(commit):
        raise FinalizationError("execution plan experiment_commit must be a full 40-digit commit")
    commands = _plan_command_map(plan)
    return plan, commands


def _input_hashes(
    command: Mapping[str, Any],
    *,
    repo_root: Path,
    evidence_root: Path,
    staging_root: Path,
    experiment_commit: str,
) -> tuple[dict[str, str], dict[str, str]]:
    configuration: dict[str, str] = {}
    for raw in command.get("configuration_files", []):
        relative = _canonical_relative(raw, label="configuration file")
        path = _confined(repo_root, relative, label="configuration file", require_exists=True)
        actual = sha256_file(path)
        committed = _committed_file_sha256(repo_root, experiment_commit, relative)
        if actual != committed:
            raise FinalizationError(
                f"configuration file differs from experiment commit: {relative.as_posix()}"
            )
        configuration[relative.as_posix()] = actual

    inputs: dict[str, str] = {}
    for index, raw in enumerate(command.get("input_files", [])):
        if not isinstance(raw, Mapping):
            raise FinalizationError(f"input_files[{index}] must be an object")
        scope = raw.get("scope")
        relative = _canonical_relative(raw.get("path", ""), label=f"input_files[{index}].path")
        if scope == "repo":
            path = _confined(repo_root, relative, label="repository input", require_exists=True)
            actual = sha256_file(path)
            committed = _committed_file_sha256(repo_root, experiment_commit, relative)
            if actual != committed:
                raise FinalizationError(
                    f"repository input differs from experiment commit: {relative.as_posix()}"
                )
        elif scope == "staging":
            path = _confined(staging_root, relative, label="staging input", require_exists=True)
            actual = sha256_file(path)
            attestation = raw.get("attestation")
            if attestation is not None:
                if not isinstance(attestation, Mapping):
                    raise FinalizationError(
                        f"input_files[{index}].attestation must be an object"
                    )
                attestation_relative = _canonical_relative(
                    attestation.get("path", ""),
                    label=f"input_files[{index}].attestation.path",
                )
                attestation_path = _confined(
                    evidence_root,
                    attestation_relative,
                    label="staging input attestation",
                    require_exists=True,
                )
                attested = _read_json(attestation_path)
                if (
                    attested.get("schema_id") != RECEIPT_SCHEMA_ID
                    or attested.get("schema_version") != RECEIPT_SCHEMA_VERSION
                    or attested.get("status") != "completed"
                    or attested.get("experiment_commit") != experiment_commit
                ):
                    raise FinalizationError(
                        f"staging input attestation is not a completed managed receipt: {attestation_relative.as_posix()}"
                    )
                pre = attested.get("preflight")
                post = attested.get("postflight")
                raw_hashes = attested.get("raw_output_hashes")
                if (
                    not isinstance(pre, Mapping)
                    or not isinstance(post, Mapping)
                    or pre.get("git_commit") != experiment_commit
                    or pre.get("git_clean") is not True
                    or pre.get("git_status_porcelain") != []
                    or pre.get("pilot_override") is not False
                    or post.get("git_commit") != experiment_commit
                    or post.get("git_clean") is not True
                    or not isinstance(raw_hashes, Mapping)
                    or raw_hashes.get(
                        (Path(STAGING_DIRECTORY) / relative).as_posix()
                    )
                    != actual
                ):
                    raise FinalizationError(
                        f"staging input attestation does not bind clean output {relative.as_posix()}"
                    )
                inputs[attestation_relative.as_posix()] = sha256_file(
                    attestation_path
                )
        else:
            raise FinalizationError(f"input_files[{index}].scope must be 'repo' or 'staging'")
        expected = raw.get("sha256")
        if expected is not None and (
            not isinstance(expected, str)
            or not HEX64_RE.fullmatch(expected.lower())
            or expected.lower() != actual
        ):
            raise FinalizationError(f"input hash mismatch for {scope}:{relative.as_posix()}")
        hash_key = (
            relative.as_posix()
            if scope == "repo"
            else (Path(STAGING_DIRECTORY) / relative).as_posix()
        )
        inputs[hash_key] = actual
    return dict(sorted(configuration.items())), dict(sorted(inputs.items()))


def _staging_input_has_attestation(command: Mapping[str, Any], relative: str) -> bool:
    for item in command.get("input_files", []):
        if (
            isinstance(item, Mapping)
            and item.get("scope") == "staging"
            and item.get("path") == relative
            and isinstance(item.get("attestation"), Mapping)
        ):
            return True
    return False


def _required_raw_paths(command: Mapping[str, Any]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for key in command["source_keys"]:
        spec = SOURCE_BY_KEY[key]
        paths.append(spec.relative_path)
        paths.extend(spec.run_records)
    for raw in command.get("expected_artifacts", []):
        paths.append(_canonical_relative(raw, label="expected artifact"))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return tuple(unique)


def execute_plan_command(
    *,
    plan_path: Path,
    command_id: str,
    evidence_root: Path,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Execute one planned producer and persist an immutable hash receipt."""
    repo_root = repo_root.resolve()
    evidence_root = evidence_root.resolve()
    plan_path = plan_path.resolve()
    plan, commands = load_plan(plan_path)
    if command_id not in commands:
        raise FinalizationError(f"unknown command id {command_id!r}")
    command = commands[command_id]
    commit = str(plan["experiment_commit"]).lower()
    # Use the shared preflight contract so producer execution and solver run
    # records agree on the meaning of a publication run.
    preflight = check_experiment_preflight(repo_root, run_kind="publication")
    if preflight.git_commit.lower() != commit:
        raise FinalizationError(
            f"execute must run at experiment commit {commit}, not {preflight.git_commit.lower()}"
        )

    staging_root = evidence_root / STAGING_DIRECTORY
    receipt_root = evidence_root / RECEIPT_DIRECTORY
    log_root = evidence_root / LOG_DIRECTORY
    receipt_path = receipt_root / f"{command_id}.json"
    stdout_path = log_root / f"{command_id}.stdout.log"
    stderr_path = log_root / f"{command_id}.stderr.log"
    marker = receipt_root / f".{command_id}.in-progress"
    for path, label in (
        (receipt_path, "receipt"),
        (stdout_path, "stdout log"),
        (stderr_path, "stderr log"),
        (marker, "in-progress marker"),
    ):
        if path.exists():
            raise FinalizationError(f"refusing to overwrite existing {label}: {path}")
    required = _required_raw_paths(command)
    for relative in required:
        target = _confined(staging_root, relative, label="raw output")
        if target.exists() or target.is_symlink():
            raise FinalizationError(f"refusing to reuse pre-existing raw output {relative.as_posix()}")

    producer_relative = _canonical_relative(command["producer"], label="producer")
    producer = _confined(repo_root, producer_relative, label="producer", require_exists=True)
    producer_hash = sha256_file(producer)
    if producer_hash != _committed_file_sha256(repo_root, commit, producer_relative):
        raise FinalizationError("producer does not match the immutable experiment commit")
    configuration_hashes, input_hashes = _input_hashes(
        command,
        repo_root=repo_root,
        evidence_root=evidence_root,
        staging_root=staging_root,
        experiment_commit=commit,
    )
    receipt_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    for relative in required:
        _confined(staging_root, relative, label="raw output").parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{os.getpid()}\n", encoding="utf-8")

    environment_overrides = {str(k): str(v) for k, v in command.get("environment", {}).items()}
    effective_environment = os.environ.copy()
    effective_environment.update(environment_overrides)
    argv_template = [str(item) for item in command["argv"]]
    argv = _expand_argv(
        argv_template,
        repo_root=repo_root,
        evidence_root=evidence_root,
        staging_root=staging_root,
    )
    started = _utc_now()
    return_code: int | None = None
    execution_error: str | None = None
    try:
        with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
            completed = subprocess.run(
                argv,
                cwd=repo_root,
                env=effective_environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
        return_code = int(completed.returncode)
    except OSError as exc:
        execution_error = f"{type(exc).__name__}: {exc}"
    finished = _utc_now()

    output_hashes: dict[str, str] = {}
    missing: list[str] = []
    for relative in required:
        path = _confined(staging_root, relative, label="raw output")
        if path.is_file() and not path.is_symlink():
            output_hashes[(Path(STAGING_DIRECTORY) / relative).as_posix()] = sha256_file(path)
        else:
            missing.append(relative.as_posix())
    post_commit = _git_head(repo_root)
    post_clean = _git_clean(repo_root)
    completed_ok = (
        return_code == 0
        and execution_error is None
        and not missing
        and post_commit == commit
        and post_clean
    )
    receipt: dict[str, Any] = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "completed" if completed_ok else "failed",
        "campaign_id": plan["campaign_id"],
        "command_id": command_id,
        "source_keys": list(command["source_keys"]),
        "experiment_commit": commit,
        "preflight": {
            "git_commit": preflight.git_commit.lower(),
            "git_clean": preflight.git_clean,
            "git_status_porcelain": list(preflight.git_status_porcelain),
            "pilot_override": preflight.pilot_override,
            "checked_at_utc": preflight.checked_at_utc,
        },
        "postflight": {
            "git_commit": post_commit,
            "git_clean": post_clean,
        },
        "command": {
            "argv_template": argv_template,
            "argv": argv,
            "working_directory": ".",
            "return_code": return_code,
            "execution_error": execution_error,
            "started_at_utc": started,
            "finished_at_utc": finished,
        },
        "environment": _environment_snapshot(environment_overrides),
        "plan": {
            "path": plan_path.as_posix(),
            "sha256": sha256_file(plan_path),
        },
        "producer": {"path": producer_relative.as_posix(), "sha256": producer_hash},
        "configuration_hashes": configuration_hashes,
        "input_hashes": input_hashes,
        "raw_output_hashes": dict(sorted(output_hashes.items())),
        "logs": {
            stdout_path.relative_to(evidence_root).as_posix(): sha256_file(stdout_path),
            stderr_path.relative_to(evidence_root).as_posix(): sha256_file(stderr_path),
        },
        "missing_outputs": missing,
    }
    receipt["receipt_fingerprint_sha256"] = _json_sha256(receipt)
    atomic_write_json(receipt_path, receipt)
    marker.unlink(missing_ok=True)
    if not completed_ok:
        raise FinalizationError(
            f"command {command_id!r} failed publication execution; inspect {receipt_path}"
        )
    return receipt_path


def _dirty_or_pilot_errors(payload: Mapping[str, Any], *, experiment_commit: str) -> list[str]:
    errors: list[str] = []
    if str(payload.get("run_kind", "")).lower() in {"pilot", "diagnostic"}:
        errors.append(f"raw payload run_kind={payload.get('run_kind')!r}")
    if payload.get("pilot_override") is True:
        errors.append("raw payload declares a pilot override")

    declarations: list[tuple[str, Mapping[str, Any]]] = []
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        declarations.append(("provenance", provenance))
        if isinstance(provenance.get("git"), Mapping):
            declarations.append(("provenance.git", provenance["git"]))
    preflight = payload.get("preflight")
    if isinstance(preflight, Mapping):
        declarations.append(("preflight", preflight))
    for label, declaration in declarations:
        dirty = declaration.get("dirty", declaration.get("git_dirty"))
        clean = declaration.get("git_clean", declaration.get("worktree_clean"))
        override = declaration.get("pilot_override")
        commit = declaration.get(
            "commit", declaration.get("git_commit", declaration.get("experiment_commit"))
        )
        if dirty is True or clean is False:
            errors.append(f"raw payload {label} declares a dirty worktree")
        if override is True:
            errors.append(f"raw payload {label} declares a pilot override")
        if commit is not None and str(commit).lower() != experiment_commit:
            errors.append(
                f"raw payload {label} commit {str(commit).lower()} differs from {experiment_commit}"
            )
    return errors


def validate_raw_source_payload(
    spec: SourceSpec,
    payload: Mapping[str, Any],
    *,
    experiment_commit: str,
) -> None:
    """Reject wrong-family, dirty, pilot, or already-finalized raw payloads."""
    experiment_value = payload.get("experiment", payload.get("experiment_id"))
    if experiment_value != spec.experiment_id:
        raise FinalizationError(
            f"raw {spec.key} experiment id must be {spec.experiment_id!r}, got {experiment_value!r}"
        )
    source_schema = payload.get("source_schema")
    if source_schema is not None or "publication_provenance" in payload:
        raise FinalizationError(f"raw {spec.key} is already decorated; post-hoc re-finalization is forbidden")
    if spec.native_version_field is not None and payload.get(spec.native_version_field) != spec.native_version:
        raise FinalizationError(
            f"raw {spec.key} must have {spec.native_version_field}={spec.native_version}"
        )
    if spec.native_schema_id is not None:
        schema = payload.get("schema")
        if not isinstance(schema, Mapping) or schema.get("id") != spec.native_schema_id:
            raise FinalizationError(f"raw {spec.key} has the wrong native schema id")
        if schema.get("version") != spec.native_version:
            raise FinalizationError(f"raw {spec.key} has the wrong native schema version")
    if spec.native_schema_name is not None and payload.get("schema_name") != spec.native_schema_name:
        raise FinalizationError(f"raw {spec.key} has the wrong native schema name")
    blockers = _dirty_or_pilot_errors(payload, experiment_commit=experiment_commit)
    if blockers:
        raise FinalizationError(f"raw {spec.key} is dirty/pilot evidence: " + "; ".join(blockers))


def _verify_receipt(
    receipt_path: Path,
    *,
    command: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_path: Path,
    repo_root: Path,
    evidence_root: Path,
    release_commit: str,
) -> dict[str, Any]:
    receipt = _read_json(receipt_path)
    if receipt.get("schema_id") != RECEIPT_SCHEMA_ID or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise FinalizationError(f"invalid receipt schema: {receipt_path}")
    fingerprint = receipt.get("receipt_fingerprint_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_fingerprint_sha256", None)
    if fingerprint != _json_sha256(unsigned):
        raise FinalizationError(f"receipt fingerprint mismatch: {receipt_path}")
    commit = str(plan["experiment_commit"]).lower()
    if receipt.get("status") != "completed" or receipt.get("experiment_commit") != commit:
        raise FinalizationError(f"receipt is not a completed execution at {commit}: {receipt_path}")
    if receipt.get("campaign_id") != plan["campaign_id"] or receipt.get("command_id") != command["id"]:
        raise FinalizationError(f"receipt identity differs from execution plan: {receipt_path}")
    if receipt.get("source_keys") != command["source_keys"]:
        raise FinalizationError(f"receipt source keys differ from execution plan: {receipt_path}")
    preflight = receipt.get("preflight")
    postflight = receipt.get("postflight")
    if not isinstance(preflight, Mapping) or not isinstance(postflight, Mapping):
        raise FinalizationError(f"receipt lacks pre/postflight provenance: {receipt_path}")
    if (
        preflight.get("git_commit") != commit
        or preflight.get("git_clean") is not True
        or preflight.get("git_status_porcelain") != []
        or preflight.get("pilot_override") is not False
        or postflight.get("git_commit") != commit
        or postflight.get("git_clean") is not True
    ):
        raise FinalizationError(f"receipt does not prove one clean immutable experiment commit: {receipt_path}")
    if receipt.get("plan", {}).get("sha256") != sha256_file(plan_path):
        raise FinalizationError(f"receipt plan hash is stale: {receipt_path}")
    producer_relative = _canonical_relative(receipt.get("producer", {}).get("path", ""), label="receipt producer")
    expected_producer = _canonical_relative(command["producer"], label="planned producer")
    if producer_relative != expected_producer:
        raise FinalizationError(f"receipt producer identity mismatch: {receipt_path}")
    producer = _confined(repo_root, producer_relative, label="receipt producer", require_exists=True)
    producer_hash = sha256_file(producer)
    if receipt.get("producer", {}).get("sha256") != producer_hash:
        raise FinalizationError(f"producer was modified after execution: {producer_relative.as_posix()}")
    committed_hash = _committed_file_sha256(repo_root, commit, producer_relative)
    if committed_hash != producer_hash:
        raise FinalizationError(f"producer differs from experiment commit: {producer_relative.as_posix()}")

    configuration_hashes, input_hashes = _input_hashes(
        command,
        repo_root=repo_root,
        evidence_root=evidence_root,
        staging_root=evidence_root / STAGING_DIRECTORY,
        experiment_commit=commit,
    )
    if receipt.get("configuration_hashes") != configuration_hashes:
        raise FinalizationError(f"configuration hash set changed after execution: {receipt_path}")
    if receipt.get("input_hashes") != input_hashes:
        raise FinalizationError(f"input hash set changed after execution: {receipt_path}")

    raw_hashes = receipt.get("raw_output_hashes")
    if not isinstance(raw_hashes, Mapping):
        raise FinalizationError(f"receipt has no raw output hashes: {receipt_path}")
    required = {
        (Path(STAGING_DIRECTORY) / path).as_posix()
        for path in _required_raw_paths(command)
    }
    if set(raw_hashes) != required:
        raise FinalizationError(f"receipt raw output set differs from plan: {receipt_path}")
    for relative, expected in raw_hashes.items():
        path = _confined(evidence_root, relative, label="receipt raw output", require_exists=True)
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise FinalizationError(f"raw output was tampered after execution: {relative}")
    command_block = receipt.get("command")
    if not isinstance(command_block, Mapping) or command_block.get("return_code") != 0:
        raise FinalizationError(f"receipt command did not terminate successfully: {receipt_path}")
    if command_block.get("argv_template") != command["argv"]:
        raise FinalizationError(f"receipt command argv differs from plan: {receipt_path}")
    if not isinstance(receipt.get("environment"), Mapping) or not receipt["environment"]:
        raise FinalizationError(f"receipt environment is missing: {receipt_path}")
    return receipt


def _record_commit(record: Mapping[str, Any]) -> str:
    provenance = record.get("provenance")
    return str(provenance.get("git_commit", "")).lower() if isinstance(provenance, Mapping) else ""


def _validate_run_record_identity(
    spec: SourceSpec, relative: Path, record: Mapping[str, Any]
) -> None:
    identifiers = record.get("identifiers")
    resources = record.get("resources")
    solver = record.get("solver")
    if not isinstance(identifiers, Mapping) or identifiers.get("experiment") != spec.experiment_id:
        raise FinalizationError(
            f"run record {relative.as_posix()} belongs to experiment "
            f"{identifiers.get('experiment') if isinstance(identifiers, Mapping) else None!r}, "
            f"not {spec.experiment_id!r}"
        )
    if not isinstance(resources, Mapping):
        raise FinalizationError(f"run record {relative.as_posix()} lacks resources")
    if spec.key == "material_point":
        if (
            identifiers.get("case") != "dimensionless-five-branch-material-point-matrix"
            or identifiers.get("method") != "jax-scalar-autodiff"
            or identifiers.get("route") != "production-mohr-coulomb-scalar"
            or resources.get("ranks") != 1
        ):
            raise FinalizationError(
                f"run record {relative.as_posix()} does not identify the frozen EXP-MC-001 case"
            )
    elif spec.key == "distribution":
        match = re.fullmatch(r"run_record_np(1|2|4)\.json", relative.name)
        expected_ranks = int(match.group(1)) if match else -1
        expected_parameters = {
            "problem": "hyperelasticity",
            "mesh_source": "procedural",
            "problem_build_mode": "rank_local",
            "distribution_strategy": "overlap_p2p",
            "assembly_backend": "coo_local",
            "local_hessian_mode": "element",
            "element_reorder_mode": "block_xyz",
            "element_degree": 1,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "factor_solver_type": "mumps",
            "use_near_nullspace": False,
            "mesh_level": 1,
            "canonical_twist_angle_rad": 0.15,
            "repetitions": 3,
            "ksp_rtol": 1.0e-12,
            "linear_residual_tolerance": 1.0e-10,
            "residual_scale_floor": 1.0,
        }
        if (
            identifiers.get("case") != f"hyperelasticity-p1-l1-np{expected_ranks}"
            or identifiers.get("method") != "fixed-state-distributed-equivalence"
            or identifiers.get("route") != "rank-local-procedural-p2p-local-coo"
            or resources.get("ranks") != expected_ranks
            or not isinstance(solver, Mapping)
            or solver.get("parameters") != expected_parameters
        ):
            raise FinalizationError(
                f"run record {relative.as_posix()} does not identify its frozen EXP-DIST-001 rank case"
            )


def _decorate_payload(
    spec: SourceSpec,
    raw_payload: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    evidence_root: Path,
    experiment_commit: str,
) -> dict[str, Any]:
    # JSON round-trip gives a deep copy while asserting strict JSON types.
    payload = json.loads(json.dumps(raw_payload, allow_nan=False))
    raw_relative = Path(STAGING_DIRECTORY) / spec.relative_path
    receipt_relative = receipt_path.relative_to(evidence_root)
    payload["source_schema"] = {
        "id": f"fenics-nonlinear-energies.revision-source.{spec.key}",
        "version": 1,
    }
    payload["publication_evidence"] = True
    payload["run_kind"] = "publication"
    payload["experiment_commit"] = experiment_commit
    payload["publication_provenance"] = {
        "schema_id": SOURCE_PROVENANCE_SCHEMA_ID,
        "schema_version": SOURCE_PROVENANCE_SCHEMA_VERSION,
        "experiment_commit": experiment_commit,
        "run_kind": "publication",
        "git_clean": True,
        "git": {"commit": experiment_commit, "worktree_clean": True},
        "command_argv": list(receipt["command"]["argv_template"]),
        "environment": receipt["environment"],
        "producer": dict(receipt["producer"]),
        "configuration_hashes": dict(receipt["configuration_hashes"]),
        "input_hashes": dict(receipt["input_hashes"]),
        "input_policy": (
            "hash_bound_file_inputs"
            if receipt["input_hashes"]
            else "no_external_file_inputs"
        ),
        "raw_output": {
            "path": raw_relative.as_posix(),
            "sha256": receipt["raw_output_hashes"][raw_relative.as_posix()],
        },
        "execution_receipt": {
            "path": receipt_relative.as_posix(),
            "sha256": sha256_file(receipt_path),
        },
    }
    return payload


def _route_contract(repo_root: Path, payload: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    raw_path = payload.get("contract_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise FinalizationError("route analysis lacks contract_path")
    path = Path(raw_path)
    if path.is_absolute():
        try:
            relative = path.resolve().relative_to(repo_root.resolve())
        except ValueError as exc:
            raise FinalizationError("route analysis contract_path escapes the repository") from exc
    else:
        relative = _canonical_relative(path, label="route analysis contract")
    contract_path = _confined(
        repo_root, relative, label="route analysis contract", require_exists=True
    )
    expected = payload.get("contract_sha256")
    if not isinstance(expected, str) or sha256_file(contract_path) != expected.lower():
        raise FinalizationError("route analysis contract hash is missing or stale")
    return contract_path, _read_json(contract_path)


def _route_empirical_map_gate(payload: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    """Independently reject invented runtime censors before table admission.

    The independent admission boundary performs the complete 102-slot audit.
    This producer-side check intentionally overlaps the most dangerous part:
    every non-admitted row must be one of the six prespecified P4 colored-SFD
    non-attempts, with the exact contract reason.  Nothing may be silently
    dropped, imputed, or reclassified by the finalizer.
    """
    rows = payload.get("empirical_map")
    if not isinstance(rows, list) or len(rows) != 102:
        raise FinalizationError(
            f"route analysis must expose exactly 102 contract slots, got {len(rows) if isinstance(rows, list) else 'non-array'}"
        )
    rules = contract.get("structural_censors")
    if not isinstance(rules, list) or len(rules) != 1 or not isinstance(rules[0], Mapping):
        raise FinalizationError("route contract must contain exactly one frozen structural-censor rule")
    rule = rules[0]
    exact_reason = "prespecified_not_attempted_memory_risk_no_threshold_claim"
    if rule.get("reason") != exact_reason:
        raise FinalizationError("route structural-censor reason differs from the frozen contract")
    censored: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise FinalizationError(f"route empirical row {index} is not an object")
        if row.get("status") == "admitted":
            if row.get("publication_model_eligible") is not True:
                raise FinalizationError(
                    f"active route row {index} is not eligible under the frozen paired design"
                )
            continue
        if row.get("status") != "censored" or row.get("reason") != exact_reason:
            raise FinalizationError(
                f"route empirical row {index} is neither admitted nor the frozen structural censor"
            )
        if (
            row.get("hardware_id") != rule.get("hardware_id")
            or row.get("configuration_id") != rule.get("configuration_id")
            or row.get("route") != rule.get("route")
        ):
            raise FinalizationError(f"route empirical row {index} is an uncontracted censor")
        if (
            row.get("warm_median_s") is not None
            or row.get("timing_s") is not None
            or row.get("admitted_wall_time_median_s") is not None
        ):
            raise FinalizationError(f"structural censor row {index} exposes or imputes a timing")
        censored.append(row)
    if len(censored) != 6:
        raise FinalizationError(f"route analysis must contain exactly six structural censors, got {len(censored)}")
    if len(rows) - len(censored) != 96:
        raise FinalizationError("route analysis must admit all 96 active slots")


def _route_terminal_decision_gate(
    payload: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    positive = "predictive_selector_admissible"
    negative = "finite_empirical_map_only"
    if contract.get("terminal_policy") != {
        "selector_claim_requires_all_model_gates": True,
        "selector_admitted": positive,
        "otherwise": negative,
        "never_impute_censored_or_missing_timings": True,
    }:
        raise FinalizationError("route terminal policy differs from the frozen contract")
    terminal = payload.get("terminal_decision")
    if terminal not in {positive, negative}:
        raise FinalizationError("route analysis has an uncontracted terminal decision")
    model = payload.get("cost_model")
    if not isinstance(model, Mapping):
        raise FinalizationError("route analysis lacks a cost-model decision object")
    rows = payload.get("empirical_map")
    if not isinstance(rows, list):
        raise FinalizationError("route analysis lacks its finite empirical map")
    training_rows = sum(
        isinstance(row, Mapping)
        and row.get("status") == "admitted"
        and row.get("publication_model_eligible") is True
        and row.get("split") == "training"
        for row in rows
    )
    holdout_rows = sum(
        isinstance(row, Mapping)
        and row.get("status") == "admitted"
        and row.get("publication_model_eligible") is True
        and row.get("split") == "holdout"
        for row in rows
    )
    if (
        training_rows != 74
        or holdout_rows != 22
        or model.get("training_rows") != training_rows
        or model.get("holdout_rows") != holdout_rows
    ):
        raise FinalizationError(
            "route model decision must bind all 74 training and 22 holdout rows"
        )
    if model.get("feature_order") != contract["cost_model"]["features_in_order"]:
        raise FinalizationError("route model feature order differs from the frozen contract")
    factor = payload.get("factorized_microbenchmark_gate")
    factor_policy = contract["factorized_calibration_policy"]
    if not isinstance(factor, Mapping) or (
        factor_policy.get("required_for_selector_claim") is not False
        or factor.get("calibration_integrated") is not False
        or factor.get("selector_use") != factor_policy.get("current_status")
        or factor.get("selector_blockers") != []
        or factor.get("required_ranks") != [1, 8, 32]
        or factor.get("independent_blocks_per_rank")
        != factor_policy.get("independent_blocks_per_rank")
    ):
        raise FinalizationError("factorized diagnostic violates its descriptive-only contract")
    factor_passed = factor.get("passed")
    factor_failures = factor.get("failures")
    calibration = factor.get("calibration_model")
    calibration_passed = (
        isinstance(calibration, Mapping) and calibration.get("status") == "passed"
    )
    if not isinstance(factor_passed, bool) or not isinstance(
        factor_failures, list
    ) or any(not isinstance(value, str) or not value for value in factor_failures):
        raise FinalizationError("factorized diagnostic outcome is malformed")
    if factor_passed is True and (factor_failures != [] or not calibration_passed):
        raise FinalizationError("passed factorized diagnostic is internally inconsistent")
    if factor_passed is False and (not factor_failures or calibration_passed):
        raise FinalizationError("failed factorized diagnostic is internally inconsistent")
    if terminal == positive:
        if (
            model.get("status") != "selection_rule_passed"
            or model.get("selector_claim_admissible") is not True
        ):
            raise FinalizationError("positive route terminal lacks a passed selector")
        return
    allowed_keys = {
        "status",
        "selector_claim_admissible",
        "feature_order",
        "training_rows",
        "holdout_rows",
        "preflight_failures",
        "failed_gates",
    }
    if set(model) != allowed_keys or model.get("selector_claim_admissible") is not False:
        raise FinalizationError(
            "finite-map-only terminal contains predictive or uncontracted model fields"
        )
    gate_names = {
        "median_absolute_percentage_error",
        "p90_absolute_percentage_error",
        "minimum_resolved_holdout_groups",
        "resolved_ordering_accuracy",
        "distinct_observed_holdout_winners",
    }
    preflight = model.get("preflight_failures")
    failed = model.get("failed_gates")
    if model.get("status") == "fit_gate_failed":
        valid = (
            preflight == []
            and isinstance(failed, list)
            and bool(failed)
            and len(failed) == len(set(failed))
            and set(failed) <= gate_names
        )
    elif model.get("status") == "not_fit_invalid_design":
        valid = (
            preflight == ["rank_deficient_or_ill_conditioned_design"]
            and failed == []
        )
    else:
        valid = False
    if not valid:
        raise FinalizationError("finite-map-only terminal lacks a coherent frozen gate failure")
    allowed_top_level = {
        "analysis_schema_version",
        "experiment_id",
        "contract_path",
        "contract_sha256",
        "sources",
        "terminal_decision",
        "empirical_map",
        "cost_model",
        "endpoint_analysis",
        "factorized_microbenchmark_gate",
        "invalid_records",
        "provenance",
    }
    required_top_level = allowed_top_level - {"endpoint_analysis"}
    if not required_top_level <= set(payload) or set(payload) - allowed_top_level:
        raise FinalizationError(
            "finite-map-only terminal contains uncontracted publication fields"
        )
    prohibited_tokens = (
        "coefficient",
        "prediction",
        "predicted",
        "ordering",
        "ranking",
        "winner",
        "crossover",
        "selector_table",
        "recommend",
        "choice",
    )

    def leaks_predictive_field(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(
                any(token in str(key).lower() for token in prohibited_tokens)
                or str(key).lower()
                in {"best_route", "selected_route", "recommended_route", "route_choice"}
                or leaks_predictive_field(nested)
                for key, nested in value.items()
            )
        if isinstance(value, list):
            return any(leaks_predictive_field(nested) for nested in value)
        return False

    if any(
        leaks_predictive_field(row)
        for row in rows
        if isinstance(row, Mapping)
    ):
        raise FinalizationError("finite-map-only empirical map leaks predictive fields")


def _route_endpoint_summary(
    *,
    command: Mapping[str, Any],
    evidence_root: Path,
) -> tuple[dict[str, Any], dict[str, str], Path, bytes]:
    relative = _canonical_relative(
        command.get("route_endpoint_analysis", ""), label="route endpoint analysis"
    )
    path = _confined(
        evidence_root / STAGING_DIRECTORY,
        relative,
        label="route endpoint analysis",
        require_exists=True,
    )
    payload = _read_json(path)
    schema = payload.get("schema")
    if (
        not isinstance(schema, Mapping)
        or schema.get("id") != "fenics-nonlinear-energies.exp-route-001.tier-b-endpoints"
        or schema.get("version") != 1
        or payload.get("experiment_id") != "EXP-ROUTE-001"
    ):
        raise FinalizationError("Tier-B endpoint analysis has the wrong schema or experiment")
    terminal = payload.get("terminal_decision")
    allowed_terminal = {
        "tier_b_descriptive_timing_only",
        "tier_b_comparative_ranking_admissible",
    }
    blocks = payload.get("blocks")
    admitted_rows = (
        sum(
            1
            for block in blocks
            if isinstance(block, Mapping) and block.get("status") == "timing_admitted"
        )
        if isinstance(blocks, list)
        else 0
    )
    comparative = payload.get("comparative_ranking_admissible")
    expected_terminal = (
        "tier_b_comparative_ranking_admissible"
        if comparative is True
        else "tier_b_descriptive_timing_only"
    )
    publication_admissible = bool(
        payload.get("endpoint_correct_timing_admissible") is True
        and terminal in allowed_terminal
        and isinstance(comparative, bool)
        and terminal == expected_terminal
        and admitted_rows == 30
        and len(blocks) == 30
        and payload.get("matrix_policy_violations") == []
        and payload.get("coverage_and_campaign_failure_reasons") == []
    )
    if not publication_admissible:
        raise FinalizationError(
            "Tier-B endpoint analysis is not publication-admissible under its frozen 30-block contract"
        )
    structural = payload.get("structural_censors")
    if not isinstance(structural, list) or len(structural) != 2:
        raise FinalizationError("Tier-B endpoint analysis must expose two rank-specific P4 colored-SFD non-attempts")
    if any(
        not isinstance(row, Mapping)
        or row.get("status") != "censored"
        or row.get("reason") != "prespecified_not_attempted_memory_risk_no_threshold_claim"
        or row.get("route") != "colored_sfd"
        or row.get("timing_exposed") is not False
        or row.get("admitted_collective_max_wall_time_s") is not None
        for row in structural
    ):
        raise FinalizationError("Tier-B endpoint analysis contains an uncontracted structural censor")
    archive_path = (Path(STAGING_DIRECTORY) / relative).as_posix()
    publication_relative = Path(
        "EXP-ROUTE-001/analysis_contract_v1/tier_b_endpoint_analysis.json"
    )
    publication_payload = json.loads(json.dumps(payload, allow_nan=False))
    publication_payload.update(
        {
            "publication_admissible": True,
            "required_rows": 30,
            "admitted_rows": admitted_rows,
            "raw_analysis": {"path": archive_path, "sha256": sha256_file(path)},
        }
    )
    publication_bytes = _json_bytes(publication_payload)
    summary = {
        "path": publication_relative.as_posix(),
        "sha256": hashlib.sha256(publication_bytes).hexdigest(),
        "schema_version": 1,
        "terminal_decision": terminal,
        "comparative_ranking_admissible": comparative,
        "publication_admissible": True,
        "required_rows": 30,
        "admitted_rows": admitted_rows,
    }
    return (
        summary,
        {archive_path: sha256_file(path)},
        publication_relative,
        publication_bytes,
    )


def _route_input_evidence(
    *,
    payload: Mapping[str, Any],
    evidence_root: Path,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    provenance = payload.get("provenance")
    entries = provenance.get("input_files") if isinstance(provenance, Mapping) else None
    if not isinstance(entries, list) or not entries:
        raise FinalizationError("route analysis has no exact input-file evidence inventory")
    hashes: dict[str, str] = {}
    roles: set[str] = set()
    relocated_entries: list[dict[str, Any]] = []
    staging_root = (evidence_root / STAGING_DIRECTORY).resolve()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise FinalizationError(f"route provenance input_files[{index}] is not an object")
        raw_path = entry.get("path")
        digest = entry.get("sha256")
        role = entry.get("role")
        if not isinstance(raw_path, str) or not isinstance(digest, str) or not isinstance(role, str):
            raise FinalizationError(f"route provenance input_files[{index}] is incomplete")
        path = Path(raw_path)
        if not path.is_absolute():
            raise FinalizationError("route analyzer input inventory must identify exact generated files")
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(staging_root)
        except ValueError as exc:
            raise FinalizationError(
                f"route analyzer input escapes the managed staging archive: {raw_path}"
            ) from exc
        confined = _confined(staging_root, relative, label="route analyzer input", require_exists=True)
        if not confined.is_file() or confined.is_symlink() or sha256_file(confined) != digest.lower():
            raise FinalizationError(f"route analyzer input hash is stale: {raw_path}")
        archive_path = (Path(STAGING_DIRECTORY) / relative).as_posix()
        hashes[archive_path] = digest.lower()
        roles.add(role)
        relocated = dict(entry)
        relocated["path"] = archive_path
        relocated["sha256"] = digest.lower()
        relocated_entries.append(relocated)
    required_roles = {
        "route_campaign_master",
        "route_tranche_manifest",
        "route_submission_ledger",
        "route_release_authorization",
        "reviewed_release_artifact",
    }
    missing = required_roles - roles
    if missing:
        raise FinalizationError(
            "route analyzer input inventory lacks required reviewed campaign evidence roles: "
            + ", ".join(sorted(missing))
        )
    return dict(sorted(hashes.items())), relocated_entries


def _companion_experiment_id(path: Path) -> str:
    return path.parts[0] if path.parts else ""


def _write_many_atomically(files: Mapping[Path, bytes]) -> None:
    if not files:
        return
    common_root = Path(os.path.commonpath([str(path.parent) for path in files])).resolve()
    common_root.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".revision-publication-finalize-", dir=common_root)
    )
    staged: list[tuple[Path, Path]] = []
    try:
        for index, (destination, data) in enumerate(files.items()):
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = temporary_root / f"{index:04d}.tmp"
            temporary.write_bytes(data)
            staged.append((temporary, destination))
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def finalize_campaign(
    *,
    plan_path: Path,
    evidence_root: Path,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Finalize all 14 raw sources after a complete managed campaign."""
    repo_root = repo_root.resolve()
    evidence_root = evidence_root.resolve()
    plan_path = plan_path.resolve()
    plan, commands = load_plan(plan_path)
    if plan.get("plan_kind", "source_campaign") != "source_campaign":
        raise FinalizationError("only a source_campaign plan can be finalized")
    experiment_commit = str(plan["experiment_commit"]).lower()
    release_commit = _require_clean_head(repo_root)
    if not _is_ancestor(repo_root, experiment_commit, release_commit):
        raise FinalizationError(
            f"experiment commit {experiment_commit} is not an ancestor of release HEAD {release_commit}"
        )
    # Ensure the referenced object is a commit even when experiment == release.
    _git(repo_root, "cat-file", "-e", f"{experiment_commit}^{{commit}}")

    finalizer = _confined(repo_root, FINALIZER_PATH, label="finalizer", require_exists=True)
    finalizer_hash = sha256_file(finalizer)
    if finalizer_hash != _committed_file_sha256(repo_root, experiment_commit, FINALIZER_PATH):
        raise FinalizationError(
            "the campaign finalizer changed after the experiment commit; rerun the evidence campaign"
        )

    receipts: dict[str, dict[str, Any]] = {}
    receipt_paths: dict[str, Path] = {}
    for command_id, command in commands.items():
        path = evidence_root / RECEIPT_DIRECTORY / f"{command_id}.json"
        if not path.is_file():
            raise FinalizationError(f"missing managed execution receipt: {path}")
        receipts[command_id] = _verify_receipt(
            path,
            command=command,
            plan=plan,
            plan_path=plan_path,
            repo_root=repo_root,
            evidence_root=evidence_root,
            release_commit=release_commit,
        )
        receipt_paths[command_id] = path

    # A staging input may not be an unexplained imported pilot artifact.  It
    # must be an exact output of another successful receipt in this campaign.
    produced: dict[str, str] = {}
    for receipt in receipts.values():
        for archived_path, digest in receipt["raw_output_hashes"].items():
            archived = Path(archived_path)
            if archived.parts and archived.parts[0] == STAGING_DIRECTORY:
                produced[Path(*archived.parts[1:]).as_posix()] = digest
    for command_id, receipt in receipts.items():
        for item in commands[command_id].get("input_files", []):
            if not isinstance(item, Mapping) or item.get("scope") != "staging":
                continue
            relative = str(item["path"])
            digest = receipt["input_hashes"].get(
                (Path(STAGING_DIRECTORY) / relative).as_posix()
            )
            if produced.get(relative) != digest and not _staging_input_has_attestation(
                commands[command_id], relative
            ):
                raise FinalizationError(
                    f"{command_id} staging input {relative} lacks a matching managed producer receipt"
                )

    source_to_command = {
        key: command_id
        for command_id, command in commands.items()
        for key in command["source_keys"]
    }
    decorated: dict[str, dict[str, Any]] = {}
    raw_payloads: dict[str, dict[str, Any]] = {}
    record_bytes: dict[Path, bytes] = {}
    route_archive_input_hashes: dict[str, str] = {}
    route_endpoint_output: tuple[Path, bytes] | None = None
    for spec in SOURCE_SPECS:
        command_id = source_to_command[spec.key]
        receipt = receipts[command_id]
        raw_path = _confined(
            evidence_root / STAGING_DIRECTORY,
            spec.relative_path,
            label=f"raw {spec.key}",
            require_exists=True,
        )
        raw_payload = _read_json(raw_path)
        validate_raw_source_payload(spec, raw_payload, experiment_commit=experiment_commit)
        raw_payloads[spec.key] = raw_payload
        decorated[spec.key] = _decorate_payload(
            spec,
            raw_payload,
            receipt=receipt,
            receipt_path=receipt_paths[command_id],
            evidence_root=evidence_root,
            experiment_commit=experiment_commit,
        )
        if spec.key == "route_analysis":
            if raw_payload.get("invalid_records") != []:
                raise FinalizationError("route analysis retains invalid input records")
            _contract_path, contract = _route_contract(repo_root, raw_payload)
            _route_empirical_map_gate(raw_payload, contract)
            _route_terminal_decision_gate(raw_payload, contract)
            endpoint_summary, endpoint_hashes, endpoint_relative, endpoint_bytes = _route_endpoint_summary(
                command=commands[command_id], evidence_root=evidence_root
            )
            route_endpoint_output = (endpoint_relative, endpoint_bytes)
            route_archive_input_hashes, relocated_route_inputs = _route_input_evidence(
                payload=raw_payload, evidence_root=evidence_root
            )
            route_archive_input_hashes.update(endpoint_hashes)
            decorated[spec.key]["endpoint_analysis"] = endpoint_summary
            decorated_provenance = decorated[spec.key].get("provenance")
            if not isinstance(decorated_provenance, dict):
                raise FinalizationError("decorated route analysis lacks provenance")
            decorated_provenance["input_files"] = relocated_route_inputs
        for relative in spec.run_records:
            raw_record_path = _confined(
                evidence_root / STAGING_DIRECTORY,
                relative,
                label="raw publication run record",
                require_exists=True,
            )
            record = _read_json(raw_record_path)
            try:
                validate_run_record(record, require_publication_ready=True)
            except RunRecordValidationError as exc:
                raise FinalizationError(
                    f"strict publication run record rejected at {relative.as_posix()}: {exc}"
                ) from exc
            _validate_run_record_identity(spec, relative, record)
            if _record_commit(record) != experiment_commit:
                raise FinalizationError(
                    f"run record {relative.as_posix()} commit differs from experiment commit"
                )
            record_bytes[relative] = raw_record_path.read_bytes()

    destinations = [spec.relative_path for spec in SOURCE_SPECS]
    destinations.extend(record_bytes)
    destinations.extend({spec.companion_manifest for spec in SOURCE_SPECS})
    if route_endpoint_output is not None:
        destinations.append(route_endpoint_output[0])
    destinations.append(Path(FINALIZATION_MANIFEST))
    for relative in destinations:
        path = _confined(evidence_root, relative, label="final publication output")
        if path.exists() or path.is_symlink():
            raise FinalizationError(f"refusing to overwrite publication output {relative.as_posix()}")

    serialized_sources = {key: _json_bytes(value) for key, value in decorated.items()}
    source_hashes = {
        key: hashlib.sha256(serialized_sources[key]).hexdigest() for key in serialized_sources
    }
    serialized_records = {path: data for path, data in record_bytes.items()}
    record_hashes = {path: hashlib.sha256(data).hexdigest() for path, data in serialized_records.items()}

    companion_groups: dict[Path, list[SourceSpec]] = {}
    for spec in SOURCE_SPECS:
        companion_groups.setdefault(spec.companion_manifest, []).append(spec)
    companion_payloads: dict[Path, dict[str, Any]] = {}
    for companion_path, specs in companion_groups.items():
        command_ids = sorted({source_to_command[spec.key] for spec in specs})
        code_hashes: dict[str, str] = {FINALIZER_PATH.as_posix(): finalizer_hash}
        configuration_hashes: dict[str, str] = {}
        input_hashes: dict[str, str] = {}
        input_policies: dict[str, str] = {}
        raw_output_hashes: dict[str, str] = {}
        receipt_hashes: dict[str, str] = {}
        commands_map: dict[str, list[str]] = {}
        environments: dict[str, Any] = {}
        for command_id in command_ids:
            receipt = receipts[command_id]
            code_hashes[receipt["producer"]["path"]] = receipt["producer"]["sha256"]
            configuration_hashes.update(receipt["configuration_hashes"])
            input_hashes.update(receipt["input_hashes"])
            input_policies[command_id] = (
                "hash_bound_file_inputs"
                if receipt["input_hashes"]
                else "no_external_file_inputs"
            )
            commands_map[command_id] = list(receipt["command"]["argv_template"])
            environments[command_id] = receipt["environment"]
            relative_receipt = receipt_paths[command_id].relative_to(evidence_root).as_posix()
            receipt_hashes[relative_receipt] = sha256_file(receipt_paths[command_id])
        if any(spec.key == "route_analysis" for spec in specs):
            input_hashes.update(route_archive_input_hashes)
        output_hashes: dict[str, str] = {}
        for spec in specs:
            raw_relative = (Path(STAGING_DIRECTORY) / spec.relative_path).as_posix()
            raw_output_hashes[raw_relative] = receipts[source_to_command[spec.key]][
                "raw_output_hashes"
            ][raw_relative]
            output_hashes[spec.relative_path.as_posix()] = source_hashes[spec.key]
            for record_path in spec.run_records:
                raw_record_relative = (Path(STAGING_DIRECTORY) / record_path).as_posix()
                raw_output_hashes[raw_record_relative] = receipts[source_to_command[spec.key]][
                    "raw_output_hashes"
                ][raw_record_relative]
                output_hashes[record_path.as_posix()] = record_hashes[record_path]
        if any(spec.key == "route_analysis" for spec in specs):
            if route_endpoint_output is None:
                raise FinalizationError("route endpoint publication copy is missing")
            output_hashes[route_endpoint_output[0].as_posix()] = hashlib.sha256(
                route_endpoint_output[1]
            ).hexdigest()
        companion_payloads[companion_path] = {
            "schema_id": COMPANION_SCHEMA_ID,
            "schema_version": COMPANION_SCHEMA_VERSION,
            "experiment_id": _companion_experiment_id(companion_path),
            "campaign_id": plan["campaign_id"],
            "run_kind": "publication",
            "publication_evidence": True,
            "experiment_commit": experiment_commit,
            "git": {"commit": experiment_commit, "worktree_clean": True},
            "release_head": {
                "commit": release_commit,
                "worktree_clean": True,
                "experiment_commit_is_ancestor": True,
            },
            "source_keys": [spec.key for spec in specs],
            "commands": commands_map,
            "environment": environments,
            "code_hashes": dict(sorted(code_hashes.items())),
            "configuration_hashes": dict(sorted(configuration_hashes.items())),
            "input_hashes": dict(sorted(input_hashes.items())),
            "input_policies": dict(sorted(input_policies.items())),
            "raw_output_hashes": dict(sorted(raw_output_hashes.items())),
            "output_hashes": dict(sorted(output_hashes.items())),
            "execution_receipts": dict(sorted(receipt_hashes.items())),
            "artifacts": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(route_archive_input_hashes.items())
            ]
            if any(spec.key == "route_analysis" for spec in specs)
            else [],
        }

    serialized_companions = {
        path: _json_bytes(payload) for path, payload in companion_payloads.items()
    }
    output_hash_manifest = {
        spec.relative_path.as_posix(): source_hashes[spec.key] for spec in SOURCE_SPECS
    }
    output_hash_manifest.update({path.as_posix(): digest for path, digest in record_hashes.items()})
    output_hash_manifest.update(
        {path.as_posix(): hashlib.sha256(data).hexdigest() for path, data in serialized_companions.items()}
    )
    if route_endpoint_output is not None:
        output_hash_manifest[route_endpoint_output[0].as_posix()] = hashlib.sha256(
            route_endpoint_output[1]
        ).hexdigest()
    finalization_payload: dict[str, Any] = {
        "schema_id": FINALIZATION_SCHEMA_ID,
        "schema_version": FINALIZATION_SCHEMA_VERSION,
        "status": FINALIZATION_STATUS,
        "publication_evidence": True,
        "campaign_id": plan["campaign_id"],
        "created_at_utc": _utc_now(),
        "experiment_commit": experiment_commit,
        "release_commit": release_commit,
        "experiment_commit_is_ancestor": True,
        "worktree_clean": True,
        "plan": {
            "path": plan_path.as_posix(),
            "sha256": sha256_file(plan_path),
        },
        "finalizer": {"path": FINALIZER_PATH.as_posix(), "sha256": finalizer_hash},
        "raw_source_hashes": {
            (Path(STAGING_DIRECTORY) / spec.relative_path).as_posix(): receipts[
                source_to_command[spec.key]
            ]["raw_output_hashes"][(Path(STAGING_DIRECTORY) / spec.relative_path).as_posix()]
            for spec in SOURCE_SPECS
        },
        "execution_receipts": {
            path.relative_to(evidence_root).as_posix(): sha256_file(path)
            for path in receipt_paths.values()
        },
        "output_hashes": dict(sorted(output_hash_manifest.items())),
    }
    finalization_payload["finalization_fingerprint_sha256"] = _json_sha256(finalization_payload)

    writes: dict[Path, bytes] = {}
    for spec in SOURCE_SPECS:
        writes[evidence_root / spec.relative_path] = serialized_sources[spec.key]
    for relative, data in serialized_records.items():
        writes[evidence_root / relative] = data
    for relative, data in serialized_companions.items():
        writes[evidence_root / relative] = data
    if route_endpoint_output is not None:
        writes[evidence_root / route_endpoint_output[0]] = route_endpoint_output[1]
    manifest_path = evidence_root / FINALIZATION_MANIFEST
    writes[manifest_path] = _json_bytes(finalization_payload)
    _write_many_atomically(writes)
    verify_finalized_campaign(manifest_path=manifest_path, evidence_root=evidence_root, repo_root=repo_root)
    return manifest_path


def verify_finalized_campaign(
    *,
    manifest_path: Path,
    evidence_root: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Rehash the finalization archive and reject any source/output tampering."""
    repo_root = repo_root.resolve()
    evidence_root = evidence_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema_id") != FINALIZATION_SCHEMA_ID or manifest.get("schema_version") != FINALIZATION_SCHEMA_VERSION:
        raise FinalizationError("invalid finalization manifest schema")
    if manifest.get("status") != FINALIZATION_STATUS or manifest.get("publication_evidence") is not True:
        raise FinalizationError("finalization manifest is not publication-ready")
    fingerprint = manifest.get("finalization_fingerprint_sha256")
    unsigned = dict(manifest)
    unsigned.pop("finalization_fingerprint_sha256", None)
    if fingerprint != _json_sha256(unsigned):
        raise FinalizationError("finalization manifest fingerprint mismatch")
    experiment_commit = str(manifest.get("experiment_commit", "")).lower()
    release_commit = _require_clean_head(repo_root)
    if not HEX40_RE.fullmatch(experiment_commit) or not _is_ancestor(
        repo_root, experiment_commit, release_commit
    ):
        raise FinalizationError("recorded experiment commit is not an ancestor of clean release HEAD")
    finalizer_path = _confined(
        repo_root,
        manifest.get("finalizer", {}).get("path", ""),
        label="finalizer identity",
        require_exists=True,
    )
    if sha256_file(finalizer_path) != manifest.get("finalizer", {}).get("sha256"):
        raise FinalizationError("finalizer hash is stale")
    hashes = manifest.get("output_hashes")
    if not isinstance(hashes, Mapping):
        raise FinalizationError("finalization manifest output_hashes must be an object")
    for raw, expected in hashes.items():
        path = _confined(evidence_root, raw, label="finalized output", require_exists=True)
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise FinalizationError(f"finalized output hash mismatch: {raw}")
    raw_hashes = manifest.get("raw_source_hashes")
    if not isinstance(raw_hashes, Mapping):
        raise FinalizationError("finalization manifest raw_source_hashes must be an object")
    for raw, expected in raw_hashes.items():
        path = _confined(evidence_root, raw, label="raw source", require_exists=True)
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise FinalizationError(f"raw source hash mismatch: {raw}")
    receipts = manifest.get("execution_receipts")
    if not isinstance(receipts, Mapping):
        raise FinalizationError("finalization manifest execution_receipts must be an object")
    for raw, expected in receipts.items():
        path = _confined(evidence_root, raw, label="execution receipt", require_exists=True)
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise FinalizationError(f"execution receipt hash mismatch: {raw}")
    return manifest


def _template_input(
    scope: str,
    path: str,
    *,
    attestation: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {"scope": scope, "path": path}
    if attestation is not None:
        value["attestation"] = {"path": attestation}
    return value


def build_execution_plan_template(*, experiment_commit: str) -> dict[str, Any]:
    """Return the canonical 14-source plan for a future clean campaign.

    Staging state files and route archives are intentionally named rather than
    silently borrowed from the current pilot directory.  Their attestation
    receipts must be produced at the same commit before dependent commands can
    execute.  This makes the three clean Plasticity3D states and the reviewed
    route archives visible prerequisites instead of hidden post-hoc inputs.
    """
    commit = experiment_commit.lower()
    if not HEX40_RE.fullmatch(commit):
        raise FinalizationError("template experiment_commit must be a full 40-digit commit")
    cpu_env = {
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
    }

    def command(
        command_id: str,
        source_key: str,
        argv: Sequence[str],
        *,
        protocol: str,
        inputs: Sequence[Mapping[str, Any]] = (),
        environment: Mapping[str, str] | None = None,
        expected_artifacts: Sequence[str] = (),
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = SOURCE_BY_KEY[source_key]
        value: dict[str, Any] = {
            "id": command_id,
            "source_keys": [source_key],
            "producer": spec.producer_path.as_posix(),
            "argv": list(argv),
            "environment": dict(environment or cpu_env),
            "configuration_files": [protocol],
            "input_files": [dict(item) for item in inputs],
            "expected_artifacts": list(expected_artifacts),
        }
        if extra:
            value.update(extra)
        return value

    def preparation_command(
        command_id: str,
        argv: Sequence[str],
        *,
        producer: str,
        protocol: str,
        expected_artifacts: Sequence[str],
        environment: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": command_id,
            "source_keys": [],
            "role": "preparation",
            "producer": producer,
            "argv": list(argv),
            "environment": dict(environment or cpu_env),
            "configuration_files": [protocol],
            "input_files": [],
            "expected_artifacts": list(expected_artifacts),
        }

    commands: list[dict[str, Any]] = [
        command(
            "val_plaplace",
            "plaplace",
            [
                "{python}",
                "experiments/runners/run_manufactured_plaplace_verification.py",
                "--subdivisions",
                "8",
                "16",
                "32",
                "64",
                "--output",
                "{staging_root}/EXP-VAL-001/plaplace_manufactured.json",
            ],
            protocol="paper/protocols/EXP-VAL-001.md",
        ),
        command(
            "val_ginzburg_landau",
            "ginzburg_landau",
            [
                "{python}",
                "experiments/runners/run_manufactured_ginzburg_landau_verification.py",
                "--subdivisions",
                "8",
                "16",
                "32",
                "64",
                "--output",
                "{staging_root}/EXP-VAL-001/ginzburg_landau_manufactured.json",
            ],
            protocol="paper/protocols/EXP-VAL-001.md",
        ),
        command(
            "val_hyperelastic_patch",
            "hyperelastic_patch",
            [
                "{python}",
                "experiments/runners/run_hyperelastic_affine_patch_verification.py",
                "--output",
                "{staging_root}/EXP-VAL-001/hyperelastic_affine_patch.json",
            ],
            protocol="paper/protocols/EXP-VAL-001.md",
        ),
        command(
            "val_hyperelastic_nonaffine",
            "hyperelastic_nonaffine",
            [
                "{python}",
                "experiments/runners/run_manufactured_hyperelastic_verification.py",
                "--subdivisions",
                "4",
                "8",
                "16",
                "24",
                "--output",
                "{staging_root}/EXP-VAL-001/hyperelastic_nonaffine_quadrature_refinement_v2/result.json",
            ],
            protocol="paper/protocols/EXP-VAL-001.md",
        ),
        command(
            "deriv_smooth",
            "smooth_derivatives",
            [
                "{python}",
                "experiments/runners/run_smooth_element_derivative_verification.py",
                "--output",
                "{staging_root}/EXP-DERIV-001/smooth_fixed_element_v1.json",
            ],
            protocol="paper/protocols/EXP-DERIV-001.md",
        ),
    ]
    for degree, key in ((1, "p1_derivatives"), (2, "p2_derivatives"), (4, "p4_derivatives")):
        mesh = (
            "data/meshes/SlopeStability3D/hetero_ssr/"
            f"hetero_ssr_L1_p{degree}_same_mesh_glued_bottom.h5"
        )
        commands.append(
            command(
                f"deriv_p{degree}",
                key,
                [
                    "{python}",
                    "experiments/runners/run_paper_derivative_verification.py",
                    "--degree",
                    str(degree),
                    "--states",
                    "5",
                    "--assembled-route-equivalence",
                    "--output",
                    f"{{staging_root}}/EXP-DERIV-001/p{degree}_l1_fixed_element_v2.json",
                ],
                protocol="paper/protocols/EXP-DERIV-001.md",
                inputs=[_template_input("repo", mesh)],
            )
        )
    commands.extend(
        [
            command(
                "mc_material_point",
                "material_point",
                [
                    "{python}",
                    "experiments/runners/run_plasticity3d_material_point_verification.py",
                    "--run-kind",
                    "publication",
                    "--output",
                    "{staging_root}/EXP-MC-001/material_point_verification.json",
                    "--report",
                    "{staging_root}/EXP-MC-001/report.md",
                    "--run-record",
                    "{staging_root}/EXP-MC-001/run_record.json",
                ],
                protocol="paper/protocols/EXP-MC-001.md",
                expected_artifacts=["EXP-MC-001/report.md"],
            ),
            command(
                "dist_hyperelastic",
                "distribution",
                [
                    "{python}",
                    "experiments/runners/run_hyperelasticity_distribution_equivalence.py",
                    "--run-kind",
                    "publication",
                    "--output-dir",
                    "{staging_root}/EXP-DIST-001",
                ],
                protocol="paper/protocols/EXP-DIST-001.md",
            ),
        ]
    )
    for degree, key in ((1, "p1_quadrature"), (2, "p2_quadrature"), (4, "p4_quadrature")):
        state = f"EXP-DISC-001/clean_inputs/p{degree}_l1_state.npz"
        state_manifest = f"EXP-DISC-001/clean_inputs/p{degree}_l1_state_manifest.json"
        attestation = f"{RECEIPT_DIRECTORY}/prepare_p{degree}_l1_state.json"
        commands.append(
            preparation_command(
                f"prepare_p{degree}_l1_state",
                [
                    "{python}",
                    "experiments/runners/prepare_plasticity3d_fixed_state.py",
                    "--degree",
                    str(degree),
                    "--mesh-name",
                    "hetero_ssr_L1",
                    "--constraint-variant",
                    "glued_bottom",
                    "--lambda-target",
                    "1.55",
                    "--state-label",
                    "mixed",
                    "--amplitude",
                    "0.02",
                    "--run-kind",
                    "publication",
                    "--output",
                    f"{{staging_root}}/{state}",
                    "--manifest",
                    f"{{staging_root}}/{state_manifest}",
                ],
                producer="experiments/runners/prepare_plasticity3d_fixed_state.py",
                protocol="paper/protocols/EXP-DISC-001.md",
                expected_artifacts=[state, state_manifest],
            )
        )
        commands.append(
            command(
                f"disc_p{degree}",
                key,
                [
                    "{python}",
                    "experiments/runners/run_plasticity3d_fixed_state_quadrature.py",
                    "--state",
                    f"{{staging_root}}/{state}",
                    "--quadrature-rules",
                    "tetra_1point,tetra_11point,tetra_24point,tetra_duffy_125point",
                    "--action-output-dir",
                    f"{{staging_root}}/EXP-DISC-001/actions/p{degree}_l1",
                    "--output",
                    f"{{staging_root}}/EXP-DISC-001/p{degree}_l1_fixed_state_quadrature_v2.json",
                ],
                protocol="paper/protocols/EXP-DISC-001.md",
                inputs=[_template_input("staging", state, attestation=attestation)],
            )
        )
    endpoint = "EXP-ROUTE-001/reviewed_inputs/tier_b_endpoint_analysis.json"
    endpoint_receipt = f"{RECEIPT_DIRECTORY}/prepare_tier_b_endpoint_analysis.json"
    master = "EXP-ROUTE-001/source_archives/karolina/route_campaign_master_manifest.json"
    master_receipt = f"{RECEIPT_DIRECTORY}/prepare_route_campaign_master.json"
    workstation_manifest = "EXP-ROUTE-001/source_archives/workstation/workstation_manifest.json"
    workstation_receipt = f"{RECEIPT_DIRECTORY}/prepare_workstation_archive.json"
    commands.append(
        command(
            "route_cost_analysis",
            "route_analysis",
            [
                "{python}",
                "experiments/analysis/analyze_plasticity3d_route_cost_model.py",
                "--contract",
                "{repo_root}/paper/protocols/EXP-ROUTE-001-analysis-contract.json",
                "--source",
                "workstation_local={staging_root}/EXP-ROUTE-001/source_archives/workstation",
                "--source",
                "karolina_cpu={staging_root}/EXP-ROUTE-001/source_archives/karolina",
                "--output-dir",
                "{staging_root}/EXP-ROUTE-001/analysis_contract_v1",
            ],
            protocol="paper/protocols/EXP-ROUTE-001-analysis-contract.json",
            inputs=[
                _template_input("staging", endpoint, attestation=endpoint_receipt),
                _template_input("staging", master, attestation=master_receipt),
                _template_input(
                    "staging", workstation_manifest, attestation=workstation_receipt
                ),
            ],
            expected_artifacts=[
                "EXP-ROUTE-001/analysis_contract_v1/empirical_route_map.csv",
                "EXP-ROUTE-001/analysis_contract_v1/report.md",
                "EXP-ROUTE-001/analysis_contract_v1/manifest.json",
            ],
            extra={"route_endpoint_analysis": endpoint},
        )
    )
    return {
        "schema_id": PLAN_SCHEMA_ID,
        "schema_version": PLAN_SCHEMA_VERSION,
        "campaign_id": "paper_revision_publication_clean_v1",
        "plan_kind": "source_campaign",
        "experiment_commit": commit,
        "template_status": "includes_clean_state_producers_and_requires_route_dependency_receipts",
        "source_count": 14,
        "commands": commands,
        "dependency_contract": {
            "clean_state_receipts": [
                f"{RECEIPT_DIRECTORY}/prepare_p1_l1_state.json",
                f"{RECEIPT_DIRECTORY}/prepare_p2_l1_state.json",
                f"{RECEIPT_DIRECTORY}/prepare_p4_l1_state.json",
            ],
            "route_receipts": [
                endpoint_receipt,
                master_receipt,
                workstation_receipt,
            ],
            "rule": (
                "Each dependency receipt must be generated by this managed-execution schema "
                "at experiment_commit and bind the staged file by exact SHA-256."
            ),
        },
    }


def write_execution_plan_template(
    *, output: Path, repo_root: Path = REPO_ROOT
) -> Path:
    repo_root = repo_root.resolve()
    commit = _require_clean_head(repo_root)
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FinalizationError(f"refusing to overwrite execution plan {output}")
    plan = build_execution_plan_template(experiment_commit=commit)
    # Validate before persistence; generated templates must never omit one of
    # the 14 admission sources.
    _plan_command_map(plan)
    atomic_write_json(output, plan)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    execute = subparsers.add_parser("execute", help="run one plan command and write its receipt")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--command-id", required=True)
    execute.add_argument("--evidence-root", type=Path, required=True)
    finalize = subparsers.add_parser("finalize", help="finalize all 14 managed source outputs")
    finalize.add_argument("--plan", type=Path, required=True)
    finalize.add_argument("--evidence-root", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="rehash an existing finalized campaign")
    verify.add_argument("--evidence-root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path)
    validate = subparsers.add_parser("validate-plan", help="validate plan shape without executing")
    validate.add_argument("--plan", type=Path, required=True)
    initialize = subparsers.add_parser(
        "init-plan", help="write the canonical 14-source plan from clean HEAD"
    )
    initialize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.mode == "execute":
            path = execute_plan_command(
                plan_path=args.plan,
                command_id=args.command_id,
                evidence_root=args.evidence_root,
            )
            print(path)
        elif args.mode == "finalize":
            path = finalize_campaign(plan_path=args.plan, evidence_root=args.evidence_root)
            print(path)
        elif args.mode == "verify":
            evidence_root = args.evidence_root.resolve()
            manifest = args.manifest or evidence_root / FINALIZATION_MANIFEST
            verify_finalized_campaign(
                manifest_path=manifest,
                evidence_root=evidence_root,
            )
            print(manifest)
        elif args.mode == "init-plan":
            print(write_execution_plan_template(output=args.output))
        else:
            plan, commands = load_plan(args.plan.resolve())
            print(
                f"valid plan {plan['campaign_id']} at {plan['experiment_commit']} "
                f"with {len(commands)} commands and {len(SOURCE_SPECS)} sources"
            )
    except (
        ExperimentPreflightError,
        FinalizationError,
        RunRecordValidationError,
        OSError,
        ValueError,
    ) as exc:
        print(f"publication campaign finalization refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

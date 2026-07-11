#!/usr/bin/env python3
"""Aggregate the two submitted EXP-ROUTE-001 Tier-B phase archives.

This command is deliberately scheduler-free.  It validates immutable archive
evidence already written by the Karolina campaign preparer and emits one
relocatable master manifest; it never invokes or queries Slurm.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.benchmark.run_record import atomic_write_json
from experiments.runners.paper_revision_karolina.tier_b_stopping import (
    POLICY_PATH as TIER_B_STOPPING_POLICY,
    sha256_file as stopping_sha256,
    validate_stop_adjudication,
)


MATRIX = REPO_ROOT / "experiments/runners/paper_revision_karolina/campaign_matrix.csv"
CONTRACT = REPO_ROOT / "paper/protocols/EXP-ROUTE-001-analysis-contract.json"
EXPERIMENT_ID = "EXP-ROUTE-001"
TIER_B_TIERS = frozenset({"full_solve_confirmation", "low_order_confirmation"})
PHASES = ("training", "holdout")
EXPECTED_COUNTS = {"training": 20, "holdout": 10}
RELEASE_SCHEMA_ID = "fenics-nonlinear-energies.human-release-authorization"
MODEL_FREEZE_SCHEMA_ID = "fenics-nonlinear-energies.route-model-freeze"
MODEL_FREEZE_SCHEMA_VERSION = 2
MASTER_SCHEMA_ID = "fenics-nonlinear-energies.exp-route-001-tier-b-campaign-master"
ROUTE_SCOPE_COUNTS = {
    "cost_model_training": 76,
    "tier_b_training": 20,
    "cost_model_holdout": 29,
    "tier_b_holdout": 10,
}
_SUBMITTED = re.compile(r"Submitted batch job ([1-9][0-9]*)")


class AggregationError(ValueError):
    """A phase archive failed Tier-B provenance admission."""


def _reject_nonfinite(token: str) -> None:
    raise AggregationError(f"nonfinite JSON token {token!r} is forbidden")


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=_reject_nonfinite)
    if not isinstance(value, dict):
        raise AggregationError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line, parse_constant=_reject_nonfinite)
        if not isinstance(value, dict):
            raise AggregationError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _case_ids_sha256(case_ids: Iterable[str]) -> str:
    canonical = json.dumps(sorted(case_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _within(base: Path, raw: object, *, label: str) -> Path:
    relative = Path(str(raw or ""))
    if not str(relative) or relative.is_absolute():
        raise AggregationError(f"{label} must be a nonempty archive-relative path")
    candidate = base / relative
    if candidate.is_symlink():
        raise AggregationError(f"{label} must not be a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise AggregationError(f"{label} escapes its phase archive") from exc
    return resolved


def _archive_relative(path: Path, *, archive_root: Path, label: str) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(archive_root.resolve()))
    except ValueError as exc:
        raise AggregationError(f"{label} is outside the explicit archive root") from exc


def _canonical_rows() -> tuple[
    str,
    dict[str, dict[str, dict[str, str]]],
    dict[str, list[str]],
]:
    contract = _read_object(CONTRACT)
    matrix_sha256 = str(
        contract["publication_model_input_gates"]["karolina_matrix_sha256"]
    )
    if _sha256(MATRIX) != matrix_sha256:
        raise AggregationError("canonical matrix hash disagrees with the analysis contract")
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    by_phase: dict[str, dict[str, dict[str, str]]] = {phase: {} for phase in PHASES}
    for row in rows:
        if (
            row.get("experiment_id") != EXPERIMENT_ID
            or row.get("optional") != "1"
            or row.get("tier") not in TIER_B_TIERS
        ):
            continue
        phase = "holdout" if int(row["total_ranks"]) == 32 else "training"
        case_id = str(row["case_id"])
        if case_id in by_phase[phase]:
            raise AggregationError(f"canonical matrix repeats Tier-B case {case_id}")
        by_phase[phase][case_id] = row
    for phase, expected in EXPECTED_COUNTS.items():
        if len(by_phase[phase]) != expected:
            raise AggregationError(
                f"canonical matrix has {len(by_phase[phase])} {phase} rows, expected {expected}"
            )
    if any(int(row["total_ranks"]) not in {1, 8} for row in by_phase["training"].values()):
        raise AggregationError("canonical Tier-B training rows are not confined to ranks 1/8")
    if any(int(row["total_ranks"]) != 32 for row in by_phase["holdout"].values()):
        raise AggregationError("canonical Tier-B holdout rows are not rank 32")
    route_rows = [row for row in rows if row.get("experiment_id") == EXPERIMENT_ID]
    scopes = {
        "cost_model_training": sorted(
            str(row["case_id"])
            for row in route_rows
            if row.get("optional") == "0" and int(row["total_ranks"]) in {1, 8}
        ),
        "tier_b_training": sorted(by_phase["training"]),
        "cost_model_holdout": sorted(
            str(row["case_id"])
            for row in route_rows
            if row.get("optional") == "0" and int(row["total_ranks"]) == 32
        ),
        "tier_b_holdout": sorted(by_phase["holdout"]),
    }
    seen: set[str] = set()
    for name, expected in ROUTE_SCOPE_COUNTS.items():
        if len(scopes[name]) != expected or len(set(scopes[name])) != expected:
            raise AggregationError(
                f"canonical {name} scope has {len(scopes[name])} cases, expected {expected}"
            )
        if seen.intersection(scopes[name]):
            raise AggregationError("canonical route scopes overlap")
        seen.update(scopes[name])
    if len(seen) != len(route_rows):
        raise AggregationError("canonical route scopes do not partition EXP-ROUTE-001")
    return matrix_sha256, by_phase, scopes


def _validate_environment(
    root: Path, manifest: dict[str, Any], *, phase: str
) -> dict[str, str]:
    contract = manifest.get("environment_contract")
    if not isinstance(contract, dict) or contract.get("status") != "hash_bound":
        raise AggregationError(f"{phase} manifest lacks a hash-bound environment contract")
    result: dict[str, str] = {}
    for role, record_key, hash_key in (
        ("setup", "archived_setup", "setup_sha256"),
        ("lock", "archived_lock", "lock_sha256"),
    ):
        digest = contract.get(hash_key)
        record = contract.get(record_key)
        if not _is_lower_hex(digest, 64) or not isinstance(record, dict):
            raise AggregationError(f"{phase} environment {role} record is malformed")
        artifact = _within(
            root,
            record.get("path"),
            label=f"{phase} environment {role}",
        )
        if (
            not artifact.is_file()
            or record.get("sha256") != digest
            or _sha256(artifact) != digest
        ):
            raise AggregationError(f"{phase} environment {role} is missing or stale")
        result[f"{role}_path"] = str(artifact)
        result[f"{role}_sha256"] = str(digest)
    return result


def _validate_stopping_gate(
    root: Path, manifest: dict[str, Any], *, phase: str
) -> dict[str, Any]:
    record = manifest.get("tier_b_stopping_gate")
    if not isinstance(record, dict):
        raise AggregationError(f"{phase} manifest lacks its Tier-B STOP gate")
    if record.get("policy") != {
        "path": str(TIER_B_STOPPING_POLICY.relative_to(REPO_ROOT)),
        "sha256": stopping_sha256(TIER_B_STOPPING_POLICY),
    }:
        raise AggregationError(f"{phase} manifest has a stale Tier-B stopping policy")
    if (
        record.get("status")
        != "validated_and_archived_before_scheduler_contact"
        or record.get("submission_admissible") is not True
    ):
        raise AggregationError(f"{phase} manifest was not admitted by final STOP evidence")
    declared = record.get("adjudication")
    if not isinstance(declared, dict):
        raise AggregationError(f"{phase} STOP adjudication metadata is malformed")
    path = _within(root, declared.get("path"), label=f"{phase} STOP adjudication")
    if not path.is_file():
        raise AggregationError(f"{phase} STOP adjudication is missing")
    validated = validate_stop_adjudication(path)
    expected = dict(validated)
    expected["path"] = str(declared.get("path"))
    if declared != expected:
        raise AggregationError(f"{phase} STOP adjudication binding is stale")
    result = dict(validated)
    result["absolute_path"] = str(path)
    result.pop("path", None)
    return result


def _validate_release(
    root: Path,
    manifest: dict[str, Any],
    *,
    phase: str,
    matrix_sha256: str,
    source_commit: str,
    selected_tiers: set[str],
) -> dict[str, str]:
    record = manifest.get("release_authorization")
    if not isinstance(record, dict) or record.get("schema_id") != RELEASE_SCHEMA_ID:
        raise AggregationError(f"{phase} manifest lacks its release authorization")
    path = _within(root, record.get("path"), label=f"{phase} release authorization")
    if (
        not path.is_file()
        or record.get("sha256") != _sha256(path)
        or not str(record.get("reviewer", "")).strip()
    ):
        raise AggregationError(f"{phase} release authorization is missing or stale")
    payload = _read_object(path)
    if (
        payload.get("schema_id") != RELEASE_SCHEMA_ID
        or payload.get("schema_version") != 1
        or payload.get("status") != "approved"
        or payload.get("decision") != "explicit_human_release_after_review"
        or payload.get("matrix_sha256") != matrix_sha256
        or payload.get("source_commit") != source_commit
        or payload.get("authorizes_experiment") != EXPERIMENT_ID
        or set(payload.get("authorizes_tiers") or []) != selected_tiers
        or not str(payload.get("reviewer", "")).strip()
        or payload.get("reviewer") != record.get("reviewer")
    ):
        raise AggregationError(f"{phase} release authorization has stale scope or identity")
    reviewed = payload.get("reviewed_artifacts")
    if not isinstance(reviewed, list) or not reviewed:
        raise AggregationError(f"{phase} release authorization has no reviewed artifacts")
    seen: set[Path] = set()
    for index, artifact_record in enumerate(reviewed):
        if not isinstance(artifact_record, dict):
            raise AggregationError(f"{phase} reviewed artifact {index} is malformed")
        artifact = _within(
            root,
            artifact_record.get("path"),
            label=f"{phase} reviewed artifact {index}",
        )
        if (
            artifact in seen
            or not artifact.is_file()
            or artifact_record.get("sha256") != _sha256(artifact)
        ):
            raise AggregationError(f"{phase} reviewed artifact {index} is missing or stale")
        seen.add(artifact)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "reviewer": str(payload["reviewer"]),
    }


def _validate_plan(
    root: Path,
    manifest: dict[str, Any],
    *,
    phase: str,
    expected_rows: dict[str, dict[str, str]],
) -> tuple[Path, list[str]]:
    path = _within(root, manifest.get("plan_file"), label=f"{phase} prepared plan")
    if not path.is_file() or manifest.get("plan_sha256") != _sha256(path):
        raise AggregationError(f"{phase} prepared plan is missing or stale")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    case_ids = [str(row.get("case_id", "")) for row in rows]
    if (
        len(case_ids) != len(expected_rows)
        or len(set(case_ids)) != len(case_ids)
        or set(case_ids) != set(expected_rows)
        or any(expected_rows[case_id] != row for case_id, row in zip(case_ids, rows, strict=True))
    ):
        raise AggregationError(f"{phase} prepared plan differs from the canonical matrix")
    return path, case_ids


def _validate_ledger(
    root: Path, *, phase: str, expected_case_ids: set[str]
) -> tuple[Path, list[str]]:
    path = root / "submitted_jobs.jsonl"
    if not path.is_file() or path.is_symlink():
        raise AggregationError(f"{phase} accepted-job ledger is missing")
    rows = _read_jsonl(path)
    case_ids = [str(row.get("case_id", "")) for row in rows]
    job_ids: list[str] = []
    if (
        len(rows) != len(expected_case_ids)
        or len(set(case_ids)) != len(case_ids)
        or set(case_ids) != expected_case_ids
    ):
        raise AggregationError(f"{phase} accepted-job ledger has incomplete case coverage")
    for row in rows:
        case_id = str(row.get("case_id", ""))
        job_id = str(row.get("job_id", ""))
        match = _SUBMITTED.fullmatch(str(row.get("stdout", "")).strip())
        try:
            command = shlex.split(str(row.get("command", "")))
        except ValueError as exc:
            raise AggregationError(f"{phase} ledger command for {case_id} is malformed") from exc
        if (
            int(row.get("returncode", 1)) != 0
            or not job_id.isdigit()
            or int(job_id) <= 0
            or match is None
            or match.group(1) != job_id
            or not command
            or command[0] != "sbatch"
            or command.count("--job-name") != 1
        ):
            raise AggregationError(f"{phase} ledger does not prove acceptance for {case_id}")
        job_name_index = command.index("--job-name")
        if job_name_index + 1 >= len(command) or command[job_name_index + 1] != case_id:
            raise AggregationError(f"{phase} ledger command names the wrong case for {case_id}")
        job_ids.append(job_id)
    if len(set(job_ids)) != len(job_ids):
        raise AggregationError(f"{phase} accepted-job ledger reuses a Slurm job ID")
    return path, job_ids


def _validate_freeze_training_manifest(
    path: Path,
    *,
    scope_name: str,
    expected_case_ids: list[str],
    matrix_sha256: str,
    source_commit: str,
) -> dict[str, Any]:
    manifest = _read_object(path)
    root = path.parent
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        matrix_by_case = {row["case_id"]: dict(row) for row in csv.DictReader(handle)}
    expected_rows = {case_id: matrix_by_case[case_id] for case_id in expected_case_ids}
    expected_tiers = {row["tier"] for row in expected_rows.values()}
    is_tier_b = scope_name == "tier_b_training"
    if (
        manifest.get("status") != "submitted"
        or manifest.get("test_only_commands") is not False
        or manifest.get("selected_experiments") != [EXPERIMENT_ID]
        or set(manifest.get("selected_tiers") or []) != expected_tiers
        or manifest.get("include_optional") is not is_tier_b
        or manifest.get("only_optional") is not is_tier_b
        or manifest.get("route_phase") != "training"
        or int(manifest.get("case_count", -1)) != len(expected_case_ids)
        or manifest.get("matrix_sha256") != matrix_sha256
        or manifest.get("source_commit") != source_commit
        or manifest.get("source_dirty") is not False
        or manifest.get("route_phase_case_ids_sha256")
        != _case_ids_sha256(expected_case_ids)
    ):
        raise AggregationError(f"model-freeze {scope_name} manifest has stale scope")
    plan_path, case_ids = _validate_plan(
        root,
        manifest,
        phase=scope_name,
        expected_rows=expected_rows,
    )
    ledger_path, job_ids = _validate_ledger(
        root,
        phase=scope_name,
        expected_case_ids=set(expected_case_ids),
    )
    environment = _validate_environment(root, manifest, phase=scope_name)
    return {
        "manifest_sha256": _sha256(path),
        "plan_path": str(plan_path),
        "ledger_path": str(ledger_path),
        "case_ids": sorted(case_ids),
        "job_ids": job_ids,
        "environment": environment,
    }


def _validate_model_freeze(
    root: Path,
    manifest: dict[str, Any],
    *,
    matrix_sha256: str,
    source_commit: str,
    scope_case_ids: dict[str, list[str]],
) -> dict[str, Any]:
    record = manifest.get("route_model_freeze")
    if (
        not isinstance(record, dict)
        or record.get("schema_id") != MODEL_FREEZE_SCHEMA_ID
        or record.get("schema_version") != MODEL_FREEZE_SCHEMA_VERSION
    ):
        raise AggregationError("holdout manifest lacks its v2 route-model freeze receipt")
    path = _within(root, record.get("path"), label="holdout route-model freeze")
    if (
        not path.is_file()
        or record.get("sha256") != _sha256(path)
        or not str(record.get("reviewer", "")).strip()
    ):
        raise AggregationError("holdout route-model freeze receipt is missing or stale")
    payload = _read_object(path)
    required = {
        "schema_id",
        "schema_version",
        "status",
        "decision",
        "matrix_sha256",
        "source_commit",
        "scopes",
        "environment_identity",
        "created_at_utc",
        "reviewer",
        "cost_model_training_manifest",
        "tier_b_training_manifest",
        "training_analysis",
        "frozen_model",
    }
    expected_scopes = {
        name: {
            "case_count": ROUTE_SCOPE_COUNTS[name],
            "case_ids_sha256": _case_ids_sha256(case_ids),
        }
        for name, case_ids in scope_case_ids.items()
    }
    if (
        set(payload) != required
        or payload.get("schema_id") != MODEL_FREEZE_SCHEMA_ID
        or payload.get("schema_version") != MODEL_FREEZE_SCHEMA_VERSION
        or payload.get("status") != "frozen_before_holdout"
        or payload.get("decision")
        != "cost_model_fit_and_tier_b_training_complete_holdouts_unopened"
        or payload.get("matrix_sha256") != matrix_sha256
        or payload.get("source_commit") != source_commit
        or payload.get("scopes") != expected_scopes
        or not str(payload.get("reviewer", "")).strip()
        or payload.get("reviewer") != record.get("reviewer")
    ):
        raise AggregationError("holdout route-model freeze identity or scope is stale")
    try:
        created = datetime.fromisoformat(
            str(payload.get("created_at_utc", "")).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise AggregationError("holdout route-model freeze timestamp is invalid") from exc
    if (
        created.tzinfo is None
        or created.utcoffset() is None
        or created.utcoffset() != timezone.utc.utcoffset(created)
    ):
        raise AggregationError("holdout route-model freeze timestamp is not UTC")
    artifact_hashes: dict[str, str] = {}
    artifacts: dict[str, Path] = {}
    for key in (
        "cost_model_training_manifest",
        "tier_b_training_manifest",
        "training_analysis",
        "frozen_model",
    ):
        artifact_record = payload.get(key)
        if not isinstance(artifact_record, dict) or set(artifact_record) != {"path", "sha256"}:
            raise AggregationError(f"holdout route-model freeze {key} is malformed")
        artifact = _within(root, artifact_record.get("path"), label=f"model-freeze {key}")
        if not artifact.is_file() or artifact_record.get("sha256") != _sha256(artifact):
            raise AggregationError(f"holdout route-model freeze {key} is missing or stale")
        _read_object(artifact)
        artifacts[key] = artifact
        artifact_hashes[f"{key}_sha256"] = _sha256(artifact)

    training_manifests = {
        scope_name: _validate_freeze_training_manifest(
            artifacts[f"{scope_name}_manifest"],
            scope_name=scope_name,
            expected_case_ids=scope_case_ids[scope_name],
            matrix_sha256=matrix_sha256,
            source_commit=source_commit,
        )
        for scope_name in ("cost_model_training", "tier_b_training")
    }
    environments = [
        record_["environment"] for record_ in training_manifests.values()
    ]
    receipt_environment = payload.get("environment_identity")
    expected_environment = {
        "setup_sha256": environments[0]["setup_sha256"],
        "lock_sha256": environments[0]["lock_sha256"],
    }
    if (
        environments[0]["setup_sha256"] != environments[1]["setup_sha256"]
        or environments[0]["lock_sha256"] != environments[1]["lock_sha256"]
        or receipt_environment != expected_environment
    ):
        raise AggregationError(
            "model-freeze training manifests do not share one environment identity"
        )

    training_model_case_ids = scope_case_ids["cost_model_training"]
    analysis = _read_object(artifacts["training_analysis"])
    model = _read_object(artifacts["frozen_model"])
    contract = _read_object(CONTRACT)
    row_ids = analysis.get("training_row_ids")
    analysis_cases = analysis.get("training_case_ids")
    analysis_model = analysis.get("frozen_model")
    if (
        analysis.get("schema_id")
        != "fenics-nonlinear-energies.exp-route-001-training-analysis"
        or analysis.get("schema_version") != 1
        or analysis.get("status") != "training_fit_admitted"
        or analysis.get("holdout_rows_seen") != 0
        or analysis.get("matrix_sha256") != matrix_sha256
        or analysis.get("source_commit") != source_commit
        or not isinstance(analysis_cases, list)
        or sorted(analysis_cases) != training_model_case_ids
        or analysis.get("training_case_ids_sha256")
        != _case_ids_sha256(training_model_case_ids)
        or not isinstance(row_ids, list)
        or len(row_ids) != 74
        or len(set(row_ids)) != 74
        or analysis.get("training_row_ids_sha256") != _case_ids_sha256(row_ids)
        or not isinstance(analysis_model, dict)
        or analysis_model.get("sha256") != _sha256(artifacts["frozen_model"])
    ):
        raise AggregationError(
            "holdout model-freeze training analysis is incomplete or contaminated"
        )
    feature_order = list(contract["cost_model"]["features_in_order"])
    coefficients = model.get("coefficients")
    design = model.get("design_diagnostics")
    if (
        model.get("schema_id")
        != "fenics-nonlinear-energies.exp-route-001-frozen-training-model"
        or model.get("schema_version") != 1
        or model.get("status") != "frozen_before_holdout"
        or model.get("holdout_rows_seen") != 0
        or model.get("matrix_sha256") != matrix_sha256
        or model.get("source_commit") != source_commit
        or model.get("contract_sha256") != _sha256(CONTRACT)
        or model.get("training_case_ids_sha256")
        != _case_ids_sha256(training_model_case_ids)
        or model.get("training_rows") != 74
        or model.get("training_row_ids") != row_ids
        or model.get("training_row_ids_sha256") != _case_ids_sha256(row_ids)
        or model.get("feature_order") != feature_order
        or not isinstance(coefficients, dict)
        or list(coefficients) != feature_order
        or any(not math.isfinite(float(coefficients[name])) for name in feature_order)
        or not isinstance(design, dict)
        or design.get("rows") != 74
        or design.get("columns") != len(feature_order)
        or design.get("rank") != len(feature_order)
        or not math.isfinite(float(design.get("condition_number", float("nan"))))
        or float(design["condition_number"])
        > float(contract["cost_model"]["maximum_design_condition_number"])
    ):
        raise AggregationError("holdout frozen route model violates its training design")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "reviewer": str(payload["reviewer"]),
        "environment_identity": expected_environment,
        **artifact_hashes,
    }


def _validate_phase(
    manifest_path: Path,
    *,
    phase: str,
    archive_root: Path,
    matrix_sha256: str,
    expected_rows: dict[str, dict[str, str]],
    scope_case_ids: dict[str, list[str]],
) -> dict[str, Any]:
    path = manifest_path.resolve()
    root = path.parent
    _archive_relative(root, archive_root=archive_root, label=f"{phase} archive root")
    if not path.is_file() or path.is_symlink():
        raise AggregationError(f"{phase} prepared manifest is missing")
    manifest = _read_object(path)
    selected_tiers = {str(value) for value in manifest.get("selected_tiers") or []}
    expected_tiers = (
        set(TIER_B_TIERS) if phase == "training" else {"full_solve_confirmation"}
    )
    source_commit = str(manifest.get("source_commit", ""))
    if (
        manifest.get("status") != "submitted"
        or manifest.get("test_only_commands") is not False
        or set(manifest.get("selected_experiments") or []) != {EXPERIMENT_ID}
        or manifest.get("include_optional") is not True
        or manifest.get("only_optional") is not True
        or manifest.get("route_phase") != phase
        or selected_tiers != expected_tiers
        or int(manifest.get("case_count", -1)) != EXPECTED_COUNTS[phase]
        or manifest.get("matrix_sha256") != matrix_sha256
        or manifest.get("source_dirty") is not False
        or not _is_lower_hex(source_commit, 40)
        or manifest.get("route_phase_case_ids_sha256")
        != _case_ids_sha256(expected_rows)
    ):
        raise AggregationError(f"{phase} manifest has stale scope or source provenance")
    plan_path, case_ids = _validate_plan(
        root,
        manifest,
        phase=phase,
        expected_rows=expected_rows,
    )
    ledger_path, job_ids = _validate_ledger(
        root,
        phase=phase,
        expected_case_ids=set(expected_rows),
    )
    environment = _validate_environment(root, manifest, phase=phase)
    stopping_gate = _validate_stopping_gate(root, manifest, phase=phase)
    release = _validate_release(
        root,
        manifest,
        phase=phase,
        matrix_sha256=matrix_sha256,
        source_commit=source_commit,
        selected_tiers=selected_tiers,
    )
    if phase == "holdout":
        model_freeze: dict[str, str] | None = _validate_model_freeze(
            root,
            manifest,
            matrix_sha256=matrix_sha256,
            source_commit=source_commit,
            scope_case_ids=scope_case_ids,
        )
    else:
        if manifest.get("route_model_freeze") is not None:
            raise AggregationError(
                "training manifest must not carry a holdout model-freeze receipt"
            )
        model_freeze = None
    return {
        "phase": phase,
        "root": root,
        "manifest": path,
        "manifest_sha256": _sha256(path),
        "source_commit": source_commit,
        "selected_tiers": sorted(selected_tiers),
        "case_ids": sorted(case_ids),
        "plan": plan_path,
        "plan_sha256": _sha256(plan_path),
        "ledger": ledger_path,
        "ledger_sha256": _sha256(ledger_path),
        "job_ids": job_ids,
        "environment": environment,
        "stopping_gate": stopping_gate,
        "release": release,
        "model_freeze": model_freeze,
    }


def aggregate(
    *,
    training_manifest: Path,
    holdout_manifest: Path,
    archive_root: Path,
) -> dict[str, Any]:
    """Validate the two phase archives without any scheduler interaction."""

    archive_root = archive_root.resolve()
    if not archive_root.is_dir():
        raise AggregationError("explicit archive root does not exist")
    matrix_sha256, canonical, scope_case_ids = _canonical_rows()
    phases = {
        "training": _validate_phase(
            training_manifest,
            phase="training",
            archive_root=archive_root,
            matrix_sha256=matrix_sha256,
            expected_rows=canonical["training"],
            scope_case_ids=scope_case_ids,
        ),
        "holdout": _validate_phase(
            holdout_manifest,
            phase="holdout",
            archive_root=archive_root,
            matrix_sha256=matrix_sha256,
            expected_rows=canonical["holdout"],
            scope_case_ids=scope_case_ids,
        ),
    }
    roots = [Path(phases[phase]["root"]) for phase in PHASES]
    if roots[0] == roots[1] or any(
        left in right.parents for left, right in ((roots[0], roots[1]), (roots[1], roots[0]))
    ):
        raise AggregationError("training and holdout must be distinct non-nested phase archives")
    commits = {str(phases[phase]["source_commit"]) for phase in PHASES}
    if len(commits) != 1:
        raise AggregationError("training and holdout do not share one clean source commit")
    setup_hashes = {
        str(phases[phase]["environment"]["setup_sha256"]) for phase in PHASES
    }
    lock_hashes = {
        str(phases[phase]["environment"]["lock_sha256"]) for phase in PHASES
    }
    if len(setup_hashes) != 1 or len(lock_hashes) != 1:
        raise AggregationError("training and holdout use different environment contracts")
    stopping_identities = []
    for phase in PHASES:
        stopping = dict(phases[phase]["stopping_gate"])
        stopping.pop("absolute_path", None)
        stopping_identities.append(stopping)
    if stopping_identities[0] != stopping_identities[1]:
        raise AggregationError("training and holdout bind different STOP adjudications")
    all_job_ids = [job_id for phase in PHASES for job_id in phases[phase]["job_ids"]]
    if len(set(all_job_ids)) != len(all_job_ids):
        raise AggregationError("training and holdout ledgers reuse a Slurm job ID")
    freeze = phases["holdout"]["model_freeze"]
    assert isinstance(freeze, dict)
    if (
        freeze["tier_b_training_manifest_sha256"]
        != phases["training"]["manifest_sha256"]
    ):
        raise AggregationError(
            "holdout model-freeze receipt is not bound to the admitted Tier-B training manifest"
        )
    frozen_environment = freeze["environment_identity"]
    if (
        frozen_environment["setup_sha256"] != next(iter(setup_hashes))
        or frozen_environment["lock_sha256"] != next(iter(lock_hashes))
    ):
        raise AggregationError(
            "cost-model, Tier-B training, and Tier-B holdout use different environments"
        )

    case_to_phase = {
        case_id: phase
        for phase in PHASES
        for case_id in sorted(canonical[phase])
    }
    phase_records: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        record = phases[phase]
        environment = dict(record["environment"])
        release = dict(record["release"])
        model_freeze = record["model_freeze"]
        stopping = dict(record["stopping_gate"])
        phase_records[phase] = {
            "phase_archive_root": _archive_relative(
                Path(record["root"]),
                archive_root=archive_root,
                label=f"{phase} archive root",
            ),
            "manifest_path": _archive_relative(
                Path(record["manifest"]),
                archive_root=archive_root,
                label=f"{phase} manifest",
            ),
            "manifest_sha256": record["manifest_sha256"],
            "plan_path": _archive_relative(
                Path(record["plan"]), archive_root=archive_root, label=f"{phase} plan"
            ),
            "plan_sha256": record["plan_sha256"],
            "submitted_jobs_path": _archive_relative(
                Path(record["ledger"]),
                archive_root=archive_root,
                label=f"{phase} ledger",
            ),
            "submitted_jobs_sha256": record["ledger_sha256"],
            "environment_setup_path": _archive_relative(
                Path(environment["setup_path"]),
                archive_root=archive_root,
                label=f"{phase} environment setup",
            ),
            "environment_setup_sha256": environment["setup_sha256"],
            "environment_lock_path": _archive_relative(
                Path(environment["lock_path"]),
                archive_root=archive_root,
                label=f"{phase} environment lock",
            ),
            "environment_lock_sha256": environment["lock_sha256"],
            "release_authorization_path": _archive_relative(
                Path(release["path"]),
                archive_root=archive_root,
                label=f"{phase} release authorization",
            ),
            "release_authorization_sha256": release["sha256"],
            "release_reviewer": release["reviewer"],
            "stopping_adjudication_path": _archive_relative(
                Path(stopping["absolute_path"]),
                archive_root=archive_root,
                label=f"{phase} STOP adjudication",
            ),
            "stopping_adjudication_sha256": stopping["sha256"],
            "route_model_freeze": (
                None
                if model_freeze is None
                else {
                    "path": _archive_relative(
                        Path(model_freeze["path"]),
                        archive_root=archive_root,
                        label="holdout route-model freeze",
                    ),
                    "sha256": model_freeze["sha256"],
                    "reviewer": model_freeze["reviewer"],
                    "cost_model_training_manifest_sha256": model_freeze[
                        "cost_model_training_manifest_sha256"
                    ],
                    "tier_b_training_manifest_sha256": model_freeze[
                        "tier_b_training_manifest_sha256"
                    ],
                    "training_analysis_sha256": model_freeze[
                        "training_analysis_sha256"
                    ],
                    "frozen_model_sha256": model_freeze["frozen_model_sha256"],
                }
            ),
            "selected_tiers": record["selected_tiers"],
            "case_count": len(record["case_ids"]),
            "case_ids": record["case_ids"],
        }
    return {
        "schema_id": MASTER_SCHEMA_ID,
        "schema_version": 2,
        "status": "submitted_phase_archives_complete",
        "experiment_id": EXPERIMENT_ID,
        "archive_root": ".",
        "matrix_path": str(MATRIX.relative_to(REPO_ROOT)),
        "matrix_sha256": matrix_sha256,
        "source_commit": next(iter(commits)),
        "source_dirty": False,
        "selected_tiers": sorted(TIER_B_TIERS),
        "case_count": sum(EXPECTED_COUNTS.values()),
        "phase_counts": dict(EXPECTED_COUNTS),
        "environment_contract": {
            "status": "same_hash_bound_contract",
            "setup_sha256": next(iter(setup_hashes)),
            "lock_sha256": next(iter(lock_hashes)),
        },
        "tier_b_stopping_gate": {
            "status": "same_adjudication_bound_before_both_phase_submissions",
            "policy": {
                "path": str(TIER_B_STOPPING_POLICY.relative_to(REPO_ROOT)),
                "sha256": stopping_sha256(TIER_B_STOPPING_POLICY),
            },
            "adjudication": stopping_identities[0],
            "phase_paths": {
                phase: phase_records[phase]["stopping_adjudication_path"]
                for phase in PHASES
            },
        },
        "case_to_phase": dict(sorted(case_to_phase.items())),
        "phases": phase_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        archive_root = args.archive_root.resolve()
        output = args.output.resolve()
        _archive_relative(output, archive_root=archive_root, label="master manifest output")
        if output in {
            args.training_manifest.resolve(),
            args.holdout_manifest.resolve(),
        }:
            raise AggregationError("master manifest output must not replace a phase manifest")
        result = aggregate(
            training_manifest=args.training_manifest,
            holdout_manifest=args.holdout_manifest,
            archive_root=archive_root,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output, result)
    except (AggregationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze the EXP-ROUTE-001 training fit before rank-32 is exposed.

This utility is intentionally scheduler-free.  It consumes one completed
workstation campaign and one *training-only* Karolina archive, validates the
archived plan and accepted submission ledger, reconstructs only the explicitly
planned rank-1/rank-8 records, and writes the immutable OLS fit products.  It
does not enumerate or read rank-32 result paths and contains no scheduler query
or submission path.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analysis import analyze_plasticity3d_route_cost_model as route_analysis
from experiments.analysis import finalize_karolina_campaign_archive as archive_finalizer
from experiments.runners.paper_revision_karolina import prepare_campaign as campaign_preparer
from src.core.benchmark.run_record import atomic_write_json


DEFAULT_CONTRACT = REPO_ROOT / "paper/protocols/EXP-ROUTE-001-analysis-contract.json"
TRAINING_ANALYSIS_NAME = "training_analysis.json"
FROZEN_MODEL_NAME = "frozen_model.json"
TRAINING_ANALYSIS_SCHEMA_ID = (
    "fenics-nonlinear-energies.exp-route-001-training-analysis"
)
FROZEN_MODEL_SCHEMA_ID = (
    "fenics-nonlinear-energies.exp-route-001-frozen-training-model"
)
SCHEMA_VERSION = 1
EXPECTED_KAROLINA_CASES = 76
EXPECTED_TRAINING_ROWS = 74
EXPECTED_STRUCTURAL_CENSORS = 4
TRAINING_TIERS = frozenset(
    {"fixed_state_screen", "factorized_quadrature", "factorized_microbenchmark"}
)
_SUBMITTED = re.compile(r"Submitted batch job ([1-9][0-9]*)")


class TrainingFreezeError(ValueError):
    """The pre-holdout training evidence does not meet the frozen contract."""


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=lambda token: (_ for _ in ()).throw(
                TrainingFreezeError(f"nonfinite JSON token {token!r} is forbidden")
            ),
        )
    if not isinstance(value, dict):
        raise TrainingFreezeError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(values: Iterable[str]) -> str:
    canonical = json.dumps(
        sorted(str(value) for value in values), separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _archive_member(root: Path, raw: object, *, name: str) -> Path:
    relative = Path(str(raw or ""))
    if relative.is_absolute() or not relative.parts:
        raise TrainingFreezeError(f"{name} must be a nonempty archive-relative path")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise TrainingFreezeError(f"{name} escapes its campaign archive") from exc
    if not resolved.is_file():
        raise TrainingFreezeError(f"{name} is missing: {resolved}")
    return resolved


def _artifact(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise TrainingFreezeError(f"{role} escapes its evidence root") from exc
    if not resolved.is_file():
        raise TrainingFreezeError(f"required {role} is missing: {resolved}")
    return {
        "role": role,
        "path": str(relative),
        "sha256": _sha256(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def _read_plan(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise TrainingFreezeError("Karolina training plan is empty")
    return rows


def _validate_training_plan_rows(
    rows: list[dict[str, str]], *, matrix_sha256: str
) -> list[str]:
    """Validate scope before any result file is opened."""

    if _sha256(route_analysis.REVIEWED_MATRIX) != matrix_sha256:
        raise TrainingFreezeError("reviewed matrix hash differs from the route contract")
    with route_analysis.REVIEWED_MATRIX.open(newline="", encoding="utf-8") as handle:
        reviewed = {row["case_id"]: dict(row) for row in csv.DictReader(handle)}
    expected_ids = campaign_preparer._model_training_case_ids(
        route_analysis.REVIEWED_MATRIX
    )
    case_ids = [str(row.get("case_id", "")) for row in rows]
    if (
        len(rows) != EXPECTED_KAROLINA_CASES
        or len(set(case_ids)) != EXPECTED_KAROLINA_CASES
        or sorted(case_ids) != expected_ids
    ):
        raise TrainingFreezeError(
            "Karolina training plan must contain the exact 76 frozen training cases"
        )
    if any(
        row.get("experiment_id") != "EXP-ROUTE-001"
        or row.get("optional") != "0"
        or row.get("tier") not in TRAINING_TIERS
        or int(row.get("total_ranks", 0)) not in {1, 8}
        or reviewed.get(row["case_id"]) != row
        for row in rows
    ):
        raise TrainingFreezeError(
            "Karolina training plan contains a holdout, optional, or changed matrix row"
        )
    if {row["tier"] for row in rows} != TRAINING_TIERS:
        raise TrainingFreezeError("Karolina training plan does not cover all training tiers")
    return sorted(case_ids)


def _read_accepted_ledger(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TrainingFreezeError(
                f"accepted ledger row {line_number} is not a JSON object"
            )
        rows.append(value)
    return rows


def _validate_release(
    root: Path,
    manifest: dict[str, Any],
    *,
    source_commit: str,
    matrix_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = manifest.get("release_authorization")
    if not isinstance(metadata, dict):
        raise TrainingFreezeError("submitted training archive lacks release authorization")
    release_path = _archive_member(
        root, metadata.get("path"), name="release_authorization.path"
    )
    if (
        metadata.get("schema_id")
        != "fenics-nonlinear-energies.human-release-authorization"
        or metadata.get("sha256") != _sha256(release_path)
    ):
        raise TrainingFreezeError("release authorization metadata is missing or stale")
    release = _read_object(release_path)
    try:
        campaign_preparer._validate_release_authorization_shape(release)
    except RuntimeError as exc:
        raise TrainingFreezeError(str(exc)) from exc
    if (
        release.get("matrix_sha256") != matrix_sha256
        or release.get("source_commit") != source_commit
        or release.get("authorizes_experiment") != "EXP-ROUTE-001"
        or not TRAINING_TIERS.issubset(set(release.get("authorizes_tiers") or []))
        or release.get("reviewer") != metadata.get("reviewer")
    ):
        raise TrainingFreezeError("release authorization does not bind this training tranche")
    reviewed_evidence: list[dict[str, Any]] = []
    for index, record in enumerate(release["reviewed_artifacts"]):
        artifact = _archive_member(
            root,
            record.get("path"),
            name=f"release reviewed artifact {index}",
        )
        if record.get("sha256") != _sha256(artifact):
            raise TrainingFreezeError(f"release reviewed artifact {index} is stale")
        reviewed_evidence.append(
            _artifact(root, artifact, role=f"release_reviewed_artifact_{index:03d}")
        )
    return _artifact(root, release_path, role="release_authorization"), {
        "reviewer": str(release["reviewer"]),
        "reviewed_artifacts": reviewed_evidence,
    }


def _validate_karolina_archive(
    root: Path, *, contract: dict[str, Any]
) -> dict[str, Any]:
    root = root.resolve()
    try:
        preflight = campaign_preparer.offline_preflight(
            root, matrix=route_analysis.REVIEWED_MATRIX
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise TrainingFreezeError(f"Karolina offline preflight failed: {exc}") from exc
    if preflight.get("mode") != "offline_no_scheduler_access":
        raise TrainingFreezeError("Karolina validation did not use scheduler-free preflight")

    manifest_path = root / "prepared_manifest.json"
    manifest = _read_object(manifest_path)
    matrix_sha256 = str(
        contract["publication_model_input_gates"]["karolina_matrix_sha256"]
    )
    source_commit = str(manifest.get("source_commit", ""))
    if (
        manifest.get("status") != "submitted"
        or manifest.get("route_phase") != "training"
        or manifest.get("selected_experiments") != ["EXP-ROUTE-001"]
        or set(manifest.get("selected_tiers") or []) != TRAINING_TIERS
        or manifest.get("test_only_commands") is not False
        or manifest.get("source_dirty") is not False
        or manifest.get("matrix_sha256") != matrix_sha256
        or not re.fullmatch(r"[0-9a-f]{40}", source_commit)
        or int(manifest.get("case_count", -1)) != EXPECTED_KAROLINA_CASES
    ):
        raise TrainingFreezeError(
            "Karolina archive is not one clean, real, submitted training tranche"
        )

    plan_path = _archive_member(root, manifest.get("plan_file"), name="plan_file")
    rows = _read_plan(plan_path)
    case_ids = _validate_training_plan_rows(rows, matrix_sha256=matrix_sha256)
    if manifest.get("route_phase_case_ids_sha256") != _ids_sha256(case_ids):
        raise TrainingFreezeError("training case-ID hash differs from the exact plan")

    try:
        jobs, planned = archive_finalizer._submitted_jobs(root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise TrainingFreezeError(f"accepted submission ledger is invalid: {exc}") from exc
    if set(jobs) != set(case_ids) or planned != {row["case_id"]: row for row in rows}:
        raise TrainingFreezeError("accepted ledger differs from the exact training plan")
    try:
        archive_finalizer._validate_submission_journal(root, jobs)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise TrainingFreezeError(f"submission journal is invalid: {exc}") from exc
    ledger_path = root / "submitted_jobs.jsonl"
    ledger = _read_accepted_ledger(ledger_path)
    if len(ledger) != EXPECTED_KAROLINA_CASES:
        raise TrainingFreezeError("accepted ledger does not contain exactly 76 rows")
    for record in ledger:
        case_id = str(record.get("case_id", ""))
        match = _SUBMITTED.fullmatch(str(record.get("stdout", "")).strip())
        if (
            case_id not in jobs
            or int(record.get("returncode", 1)) != 0
            or not str(record.get("command", "")).startswith("sbatch ")
            or match is None
            or match.group(1) != jobs[case_id]
        ):
            raise TrainingFreezeError("accepted ledger contains an unaccepted job row")

    jobs_root = root / "jobs"
    if not jobs_root.is_dir():
        raise TrainingFreezeError("completed Karolina archive lacks its jobs directory")
    archived_case_ids = {path.name for path in jobs_root.iterdir() if path.is_dir()}
    if archived_case_ids != set(case_ids):
        raise TrainingFreezeError(
            "Karolina jobs directory differs from the exact training case scope"
        )
    for case_id, job_id in jobs.items():
        case_root = jobs_root / case_id
        expected_job = case_root / f"job_{job_id}"
        job_dirs = {path.resolve() for path in case_root.iterdir() if path.is_dir()}
        if job_dirs != {expected_job.resolve()}:
            raise TrainingFreezeError(
                f"Karolina case {case_id} does not contain exactly its accepted job"
            )

    release_artifact, release_details = _validate_release(
        root,
        manifest,
        source_commit=source_commit,
        matrix_sha256=matrix_sha256,
    )
    environment = dict(manifest.get("environment_contract") or {})
    if environment.get("status") != "hash_bound":
        raise TrainingFreezeError("real training archive lacks a hash-bound environment")
    setup = _archive_member(
        root,
        dict(environment.get("archived_setup") or {}).get("path"),
        name="environment archived_setup",
    )
    lock = _archive_member(
        root,
        dict(environment.get("archived_lock") or {}).get("path"),
        name="environment archived_lock",
    )
    if (
        _sha256(setup) != environment.get("setup_sha256")
        or _sha256(lock) != environment.get("lock_sha256")
    ):
        raise TrainingFreezeError("archived environment contract hash is stale")

    evidence = [
        _artifact(root, manifest_path, role="karolina_submitted_manifest"),
        _artifact(root, plan_path, role="karolina_training_plan"),
        _artifact(root, ledger_path, role="karolina_accepted_submission_ledger"),
        _artifact(
            root,
            _archive_member(root, manifest.get("commands_file"), name="commands_file"),
            role="karolina_sbatch_argument_archive",
        ),
        _artifact(
            root,
            _archive_member(
                root,
                dict(manifest.get("queued_source_freeze") or {}).get("path"),
                name="queued_source_freeze.path",
            ),
            role="karolina_queued_source_freeze",
        ),
        _artifact(root, setup, role="karolina_environment_setup"),
        _artifact(root, lock, role="karolina_environment_lock"),
        release_artifact,
    ]
    checksum_path = root / archive_finalizer.CHECKSUM_NAME
    if not checksum_path.is_file():
        raise TrainingFreezeError(
            "Karolina training archive must be settled and checksum-sealed before fitting"
        )
    try:
        archive_finalizer.verify_archive(
            root,
            expected_manifest_sha256=_sha256(checksum_path),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise TrainingFreezeError(f"Karolina training archive checksum failed: {exc}") from exc
    evidence.append(
        _artifact(root, checksum_path, role="karolina_archive_checksum_index")
    )
    return {
        "root": root,
        "manifest": manifest,
        "rows": rows,
        "jobs": jobs,
        "case_ids": case_ids,
        "source_commit": source_commit,
        "matrix_sha256": matrix_sha256,
        "preflight": preflight,
        "evidence": evidence,
        "release": release_details,
    }


def _validate_workstation(root: Path, *, contract: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    gate = route_analysis._source_provenance_gate(
        "workstation_local", root, contract
    )
    if gate.get("eligible") is not True:
        raise TrainingFreezeError(str(gate.get("reason", "workstation gate failed")))
    manifest_path = root / "workstation_manifest.json"
    manifest = _read_object(manifest_path)
    plan_path = _archive_member(root, manifest.get("plan_path"), name="workstation plan")
    plan = _read_object(plan_path)
    case_ids = [str(value) for value in plan.get("case_ids") or []]
    if (
        len(case_ids) != 12
        or len(set(case_ids)) != 12
        or case_ids != list(manifest.get("case_ids") or [])
        or plan.get("source_matrix_sha256")
        != contract["publication_model_input_gates"]["karolina_matrix_sha256"]
    ):
        raise TrainingFreezeError("workstation campaign does not match its frozen 12-case plan")
    with route_analysis.REVIEWED_MATRIX.open(newline="", encoding="utf-8") as handle:
        by_case = {row["case_id"]: dict(row) for row in csv.DictReader(handle)}
    rows = [by_case[case_id] for case_id in case_ids]
    if any(
        row["runner"] != "p3d_fixed_state_block"
        or row["tier"] != "fixed_state_screen"
        or int(row["total_ranks"]) != 1
        or int(row["element_degree"]) not in {1, 2}
        for row in rows
    ):
        raise TrainingFreezeError("workstation plan contains an out-of-scope case")
    run_id = str(manifest.get("run_id", ""))
    cases_root = root / "cases"
    if not run_id or not cases_root.is_dir():
        raise TrainingFreezeError("workstation campaign lacks run identity or cases")
    if {path.name for path in cases_root.iterdir() if path.is_dir()} != set(case_ids):
        raise TrainingFreezeError("workstation cases differ from the frozen plan")
    for case_id in case_ids:
        case_root = cases_root / case_id
        expected = case_root / f"job_{run_id}"
        actual = {path.resolve() for path in case_root.iterdir() if path.is_dir()}
        if actual != {expected.resolve()}:
            raise TrainingFreezeError(
                f"workstation case {case_id} does not contain exactly the frozen run"
            )
    environment_path = _archive_member(
        root, manifest.get("environment_path"), name="workstation environment"
    )
    return {
        "root": root,
        "manifest": manifest,
        "rows": rows,
        "jobs": {case_id: run_id for case_id in case_ids},
        "case_ids": case_ids,
        "source_commit": str(gate["source_commit"]),
        "provenance": gate,
        "evidence": [
            _artifact(root, manifest_path, role="workstation_manifest"),
            _artifact(root, plan_path, role="workstation_plan"),
            _artifact(root, environment_path, role="workstation_environment"),
        ],
    }


def _scan_fixed_rows(
    context: dict[str, Any],
    *,
    hardware_id: str,
    contract: dict[str, Any],
) -> tuple[
    dict[tuple[str, str, str, int, str], dict[str, Any]],
    list[dict[str, Any]],
]:
    root = Path(context["root"])
    base = "cases" if hardware_id == "workstation_local" else "jobs"
    provenance = (
        dict(context["provenance"])
        if hardware_id == "workstation_local"
        else {
            "eligible": True,
            "reason": "completed_real_training_tranche",
            "manifest_path": str(root / "prepared_manifest.json"),
            "manifest_sha256": _sha256(root / "prepared_manifest.json"),
            "source_commit": str(context["source_commit"]),
            "hardware_id": "karolina_cpu",
        }
    )
    candidates: dict[
        tuple[str, str, str, int, str], list[dict[str, Any]]
    ] = {}
    record_index: list[dict[str, Any]] = []
    fixed_rows = [
        row for row in context["rows"] if row["runner"] == "p3d_fixed_state_block"
    ]
    for row in fixed_rows:
        case_id = row["case_id"]
        job_id = str(context["jobs"][case_id])
        job_dir = root / base / case_id / f"job_{job_id}"
        route_order = [value for value in row["route_order"].split("|") if value]
        block_path = job_dir / "measure_01" / "block_result.json"
        block = _read_object(block_path)
        if (
            block.get("status") != "admitted_correctness_block"
            or block.get("comparison_id") != row["comparison_id"]
            or int(block.get("block_repetition", -1))
            != int(row["block_repetition"])
        ):
            raise TrainingFreezeError(f"fixed block {case_id} did not pass admission")
        for position, route in enumerate(route_order):
            path = job_dir / "measure_01" / route / "output.json"
            payload = _read_object(path)
            try:
                slot = route_analysis._record_slot(
                    payload, hardware_id=hardware_id, contract=contract
                )
                record = route_analysis._validate_record(
                    path, payload, contract=contract
                )
                route_analysis._bind_fixed_record_to_reviewed_matrix(
                    path, payload, contract=contract
                )
                if hardware_id == "workstation_local":
                    evidence_files = route_analysis._workstation_fixed_evidence_files(
                        root, path, payload
                    )
                else:
                    evidence_files = route_analysis._cluster_fixed_evidence_files(
                        root, path, payload
                    )
            except (KeyError, OSError, TypeError, ValueError) as exc:
                raise TrainingFreezeError(f"invalid fixed record {path}: {exc}") from exc
            design = dict(payload.get("comparison_design") or {})
            if (
                design.get("route_order_policy") != "seeded_balanced_cyclic_v1"
                or int(design.get("route_order_position", -1)) != position
                or design.get("comparison_id") != row["comparison_id"]
            ):
                raise TrainingFreezeError(f"fixed record {path} changes paired design")
            record["source_provenance"] = provenance
            record["matrix_row"] = row
            record["block_result_path"] = str(block_path)
            record["evidence_files"] = evidence_files
            candidates.setdefault(slot, []).append(record)
            record_index.append(
                {
                    "case_id": case_id,
                    "job_id": job_id,
                    "hardware_id": hardware_id,
                    "configuration_id": slot[1],
                    "state_id": slot[2],
                    "rank_count": slot[3],
                    "route": slot[4],
                    "path": str(path.relative_to(root)),
                    "sha256": _sha256(path),
                }
            )
    observed: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    for slot, records in candidates.items():
        try:
            observed[slot] = route_analysis._aggregate_block_records(
                records, contract=contract
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingFreezeError(
                f"paired records for {slot} do not aggregate: {exc}"
            ) from exc
    return observed, sorted(
        record_index,
        key=lambda row: (
            row["hardware_id"],
            row["configuration_id"],
            row["state_id"],
            row["rank_count"],
            row["route"],
            row["case_id"],
        ),
    )


def _validate_training_factor_jobs(
    context: dict[str, Any], *, contract: dict[str, Any]
) -> list[dict[str, Any]]:
    root = Path(context["root"])
    rows = [
        row for row in context["rows"] if row["runner"] == "route_factor_microbench"
    ]
    if len(rows) != 6:
        raise TrainingFreezeError("training archive must contain six factor jobs")
    evidence: list[dict[str, Any]] = []
    blocks_by_rank: dict[int, set[int]] = {}
    expected_factors = {
        "element_dofs",
        "quadrature_points",
        "constitutive_dimension",
        "color_count",
        "nonzeros_per_row",
        "message_bytes",
        "imbalance_ratio",
    }
    for row in rows:
        case_id = row["case_id"]
        job_id = str(context["jobs"][case_id])
        path = root / "jobs" / case_id / f"job_{job_id}" / "measure_01" / "output.json"
        payload = _read_object(path)
        try:
            route_analysis._bind_factor_record_to_reviewed_matrix(
                path, payload, contract=contract
            )
            route_analysis._validate_factor_payload_design_and_timings(payload)
            batch_evidence = route_analysis._cluster_factor_evidence_files(
                root, path, payload
            )
        except (KeyError, IndexError, OSError, TypeError, ValueError) as exc:
            raise TrainingFreezeError(f"invalid factor record {path}: {exc}") from exc
        ranks = int(payload["results"][0]["mpi_ranks"])
        block = int(payload.get("block_repetition", 0))
        runtime = dict(payload.get("numerical_runtime") or {})
        git = dict(payload.get("git") or {})
        job_metadata = dict(payload.get("job_metadata") or {})
        if (
            ranks not in {1, 8}
            or payload.get("status") != "completed"
            or payload.get("timing_reduction") != "mpi_collective_max"
            or set(payload.get("factors") or []) != expected_factors
            or not str(payload.get("command", "")).strip()
            or not str(runtime.get("numpy", ""))
            or not isinstance(runtime.get("cpu_affinity"), list)
            or git.get("dirty") is not False
            or git.get("commit") != context["source_commit"]
            or str(job_metadata.get("slurm_job_id", "")) != job_id
            or block not in {1, 2, 3}
        ):
            raise TrainingFreezeError(f"factor record {path} lacks frozen provenance")
        blocks_by_rank.setdefault(ranks, set()).add(block)
        evidence.append(
            {
                "case_id": case_id,
                "job_id": job_id,
                "rank_count": ranks,
                "block_repetition": block,
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "batch_evidence_sha256": _ids_sha256(
                    entry["sha256"] for entry in batch_evidence
                ),
            }
        )
    if blocks_by_rank != {1: {1, 2, 3}, 8: {1, 2, 3}}:
        raise TrainingFreezeError("factor jobs do not cover three blocks at ranks 1 and 8")
    return sorted(evidence, key=lambda row: (row["rank_count"], row["block_repetition"]))


def _training_contract(contract: dict[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(contract)
    frozen["expected_scope"]["karolina_cpu"]["ranks"] = [1, 8]
    frozen["hardware"]["karolina_cpu"]["holdout_ranks"] = []
    return frozen


def _row_id(row: dict[str, Any]) -> str:
    return (
        f"{row['hardware_id']}/{row['configuration_id']}/{row['state_id']}"
        f"/np{int(row['rank_count'])}/{row['route']}"
    )


def _fit_training_rows(
    rows: list[dict[str, Any]], *, contract: dict[str, Any]
) -> dict[str, Any]:
    feature_order = list(contract["cost_model"]["features_in_order"])
    if len(rows) != EXPECTED_TRAINING_ROWS:
        raise TrainingFreezeError(
            f"training model requires exactly {EXPECTED_TRAINING_ROWS} admitted rows"
        )
    row_ids = [_row_id(row) for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise TrainingFreezeError("training row identities are not unique")
    if any(
        row.get("status") != "admitted"
        or row.get("split") != "training"
        or row.get("publication_model_eligible") is not True
        or row.get("model_covariates") is None
        for row in rows
    ):
        raise TrainingFreezeError("training scope contains an unadmitted model row")
    model = contract["cost_model"]
    groups = {
        (
            str(row["hardware_id"]),
            str(row["configuration_id"]),
            str(row["state_id"]),
            int(row["rank_count"]),
        )
        for row in rows
    }
    reasons: list[str] = []
    if contract["publication_model_input_gates"].get("design_released_for_fitting") is not True:
        reasons.append("cost_model_design_not_released")
    if len(groups) < int(model["minimum_training_groups"]):
        reasons.append("insufficient_training_groups")
    for hardware_id in model["required_training_hardware"]:
        if hardware_id not in {row["hardware_id"] for row in rows}:
            reasons.append(f"missing_training_hardware_{hardware_id}")
    for route in contract["route_order"]:
        if sum(row["route"] == route for row in rows) < int(
            model["minimum_training_rows_per_route"]
        ):
            reasons.append(f"insufficient_training_rows_{route}")
    commits = {str(row.get("source_commit", "")) for row in rows}
    if len(commits) != 1 or "" in commits:
        reasons.append("training_rows_do_not_share_one_source_commit")
    for group in groups:
        grouped = [
            row
            for row in rows
            if (
                row["hardware_id"],
                row["configuration_id"],
                row["state_id"],
                int(row["rank_count"]),
            )
            == group
        ]
        repetitions = {
            tuple(int(value) for value in row.get("paired_block_repetitions") or [])
            for row in grouped
        }
        if (
            len(repetitions) != 1
            or len(next(iter(repetitions), ())) < 3
            or any(
                len(row.get("paired_block_medians_s") or [])
                != len(next(iter(repetitions), ()))
                for row in grouped
            )
        ):
            reasons.append("training_paired_block_evidence_missing_or_unaligned")
            break
    if reasons:
        raise TrainingFreezeError(
            "training preflight failed: " + ", ".join(sorted(set(reasons)))
        )

    x = np.vstack(
        [
            route_analysis._feature_vector(row, feature_order, factorized_gate={})
            for row in rows
        ]
    )
    y = np.log(
        np.asarray(
            [row["admitted_wall_time_median_s"] for row in rows],
            dtype=np.float64,
        )
    )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise TrainingFreezeError("training design or response contains nonfinite values")
    rank = int(np.linalg.matrix_rank(x))
    condition = float(np.linalg.cond(x))
    maximum_condition = float(model["maximum_design_condition_number"])
    if (
        rank != x.shape[1]
        or not math.isfinite(condition)
        or condition > maximum_condition
    ):
        raise TrainingFreezeError(
            "training design is rank deficient or exceeds its condition-number gate"
        )
    coefficients, residuals, fitted_rank, singular_values = np.linalg.lstsq(
        x, y, rcond=None
    )
    fitted = x @ coefficients
    if not np.all(np.isfinite(coefficients)) or int(fitted_rank) != rank:
        raise TrainingFreezeError("OLS fit produced invalid coefficients")
    return {
        "feature_order": feature_order,
        "coefficients": {
            name: float(value)
            for name, value in zip(feature_order, coefficients, strict=True)
        },
        "training_row_ids": row_ids,
        "training_row_ids_sha256": _ids_sha256(row_ids),
        "design_diagnostics": {
            "rows": int(x.shape[0]),
            "columns": int(x.shape[1]),
            "rank": rank,
            "condition_number": condition,
            "maximum_condition_number": maximum_condition,
            "singular_values": [float(value) for value in singular_values],
            "residual_sum_of_squares": float(np.sum((y - fitted) ** 2)),
            "log_response_rmse": float(np.sqrt(np.mean((y - fitted) ** 2))),
            "lstsq_residuals": [float(value) for value in residuals],
        },
    }


def collect_training_evidence(
    workstation_root: Path,
    karolina_training_root: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = _read_object(contract_path)
    if contract.get("experiment_id") != "EXP-ROUTE-001":
        raise TrainingFreezeError("analysis contract is not EXP-ROUTE-001")
    workstation = _validate_workstation(workstation_root, contract=contract)
    karolina = _validate_karolina_archive(karolina_training_root, contract=contract)
    if workstation["source_commit"] != karolina["source_commit"]:
        raise TrainingFreezeError("workstation and Karolina evidence use different commits")
    if workstation["manifest"].get("matrix_sha256") != karolina["matrix_sha256"]:
        raise TrainingFreezeError("workstation and Karolina evidence use different matrices")

    scoped_contract = _training_contract(contract)
    workstation_observed, workstation_records = _scan_fixed_rows(
        workstation,
        hardware_id="workstation_local",
        contract=scoped_contract,
    )
    karolina_observed, karolina_records = _scan_fixed_rows(
        karolina,
        hardware_id="karolina_cpu",
        contract=scoped_contract,
    )
    factor_records = _validate_training_factor_jobs(karolina, contract=scoped_contract)
    observed = {**workstation_observed, **karolina_observed}
    if len(observed) != len(workstation_observed) + len(karolina_observed):
        raise TrainingFreezeError("workstation and Karolina training slots overlap")
    empirical = route_analysis.build_empirical_map(
        contract=scoped_contract,
        hardware_ids=("workstation_local", "karolina_cpu"),
        observed=observed,
        runtime_censors={},
    )
    admitted = [
        row
        for row in empirical
        if row["status"] == "admitted"
        and row["publication_model_eligible"] is True
        and row["split"] == "training"
    ]
    structural = [row for row in empirical if row["status"] == "censored"]
    failures = [
        row
        for row in empirical
        if row["status"] not in {"admitted", "censored"}
        or (row["status"] == "censored" and not row["reason"].startswith("prespecified_"))
    ]
    if (
        len(admitted) != EXPECTED_TRAINING_ROWS
        or len(structural) != EXPECTED_STRUCTURAL_CENSORS
        or failures
        or any(row["split"] != "training" for row in admitted)
    ):
        raise TrainingFreezeError(
            "complete equivalence-admitted training scope was not reconstructed"
        )
    admitted.sort(key=_row_id)
    fit = _fit_training_rows(admitted, contract=contract)
    source_records = workstation_records + karolina_records
    by_slot: dict[str, list[dict[str, Any]]] = {}
    for record in source_records:
        identity = (
            f"{record['hardware_id']}/{record['configuration_id']}/{record['state_id']}"
            f"/np{record['rank_count']}/{record['route']}"
        )
        by_slot.setdefault(identity, []).append(
            {
                "case_id": record["case_id"],
                "job_id": record["job_id"],
                "path": record["path"],
                "sha256": record["sha256"],
            }
        )
    row_evidence = [
        {
            "row_id": _row_id(row),
            "hardware_id": row["hardware_id"],
            "configuration_id": row["configuration_id"],
            "state_id": row["state_id"],
            "rank_count": int(row["rank_count"]),
            "route": row["route"],
            "state_sha256": row["state_sha256"],
            "action_sha256": row["action_sha256"],
            "source_commit": row["source_commit"],
            "source_records": sorted(
                by_slot[_row_id(row)], key=lambda value: value["case_id"]
            ),
        }
        for row in admitted
    ]
    return {
        "contract": contract,
        "contract_path": contract_path,
        "contract_sha256": _sha256(contract_path),
        "matrix_sha256": karolina["matrix_sha256"],
        "source_commit": karolina["source_commit"],
        "training_case_ids": karolina["case_ids"],
        "training_case_ids_sha256": _ids_sha256(karolina["case_ids"]),
        "training_rows": admitted,
        "row_evidence": row_evidence,
        "fit": fit,
        "workstation": workstation,
        "karolina": karolina,
        "factor_records": factor_records,
    }


def write_training_products(context: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / FROZEN_MODEL_NAME
    analysis_path = output_dir / TRAINING_ANALYSIS_NAME
    fit = dict(context["fit"])
    row_ids = list(fit["training_row_ids"])
    case_ids = list(context["training_case_ids"])
    contract = dict(context["contract"])
    frozen_model = {
        "schema_id": FROZEN_MODEL_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_before_holdout",
        "experiment_id": "EXP-ROUTE-001",
        "fit": "ordinary_least_squares",
        "response": contract["cost_model"]["response"],
        "intercept_added": False,
        "feature_order": list(fit["feature_order"]),
        "coefficients": dict(fit["coefficients"]),
        "design_diagnostics": dict(fit["design_diagnostics"]),
        "training_rows": len(row_ids),
        "holdout_rows_seen": 0,
        "training_row_ids": row_ids,
        "training_row_ids_sha256": fit["training_row_ids_sha256"],
        "training_case_ids_sha256": context["training_case_ids_sha256"],
        "matrix_sha256": context["matrix_sha256"],
        "source_commit": context["source_commit"],
        "contract_sha256": context["contract_sha256"],
    }
    atomic_write_json(model_path, frozen_model)
    frozen_model_sha256 = _sha256(model_path)

    workstation = context["workstation"]
    karolina = context["karolina"]
    training_analysis = {
        "schema_id": TRAINING_ANALYSIS_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "training_fit_admitted",
        "experiment_id": "EXP-ROUTE-001",
        "holdout_rows_seen": 0,
        "holdout_access_policy": "rank_32_result_paths_not_enumerated_or_read",
        "matrix_sha256": context["matrix_sha256"],
        "source_commit": context["source_commit"],
        "contract": {
            "path": str(Path(context["contract_path"]).relative_to(REPO_ROOT)),
            "sha256": context["contract_sha256"],
        },
        "training_case_count": len(case_ids),
        "training_case_ids": case_ids,
        "training_case_ids_sha256": context["training_case_ids_sha256"],
        "training_row_count": len(row_ids),
        "training_row_ids": row_ids,
        "training_row_ids_sha256": fit["training_row_ids_sha256"],
        "equivalence_admitted_model_rows": context["row_evidence"],
        "completed_non_model_factor_jobs": context["factor_records"],
        "design_diagnostics": fit["design_diagnostics"],
        "workstation_campaign": {
            "case_ids": list(workstation["case_ids"]),
            "evidence": list(workstation["evidence"]),
        },
        "karolina_training_campaign": {
            "route_phase": "training",
            "case_ids": case_ids,
            "accepted_job_ids": dict(sorted(karolina["jobs"].items())),
            "offline_preflight": dict(karolina["preflight"]),
            "release": dict(karolina["release"]),
            "evidence": list(karolina["evidence"]),
        },
        "frozen_model": {
            "path": FROZEN_MODEL_NAME,
            "sha256": frozen_model_sha256,
        },
    }
    atomic_write_json(analysis_path, training_analysis)
    return {"training_analysis": analysis_path, "frozen_model": model_path}


def freeze(
    workstation_root: Path,
    karolina_training_root: Path,
    output_dir: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, Path]:
    context = collect_training_evidence(
        workstation_root,
        karolina_training_root,
        contract_path=contract_path,
    )
    return write_training_products(context, output_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workstation-root", type=Path, required=True)
    parser.add_argument("--karolina-training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        outputs = freeze(
            args.workstation_root,
            args.karolina_training_root,
            args.output_dir,
            contract_path=args.contract,
        )
    except (OSError, TrainingFreezeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(
        json.dumps(
            {name: str(path) for name, path in outputs.items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()

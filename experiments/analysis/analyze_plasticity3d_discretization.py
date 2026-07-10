#!/usr/bin/env python3
"""Fail-closed adjudication of the six EXP-DISC-001 Karolina rows.

The campaign is deliberately staged across one submitted root per protocol
stage.  This analyzer admits evidence only when all five submitted manifests,
all six matrix rows, the solver/Riesz records, the saved states, and both
independent quadrature evaluations agree with the reviewed policy.  It then
compares the two solved quadrature endpoints on each mesh under the common
125-point evaluator and keeps mesh, quadrature, and tolerance effects
separate.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.runners.paper_revision_karolina.execute_case import (
    validate_p3d_solve_output,
)
from experiments.analysis.collect_slurm_accounting import (
    SCHEMA_ID as SLURM_ACCOUNTING_SCHEMA_ID,
    SCHEMA_VERSION as SLURM_ACCOUNTING_SCHEMA_VERSION,
    parse_sacct,
)
from src.core.benchmark.run_record import atomic_write_json


SCHEMA_ID = "fenics-nonlinear-energies.exp-disc-001.adjudication"
SCHEMA_VERSION = 2
EXPERIMENT_ID = "EXP-DISC-001"
KAROLINA_CLUSTER = "karolina"
KAROLINA_ACCOUNT = "fta-26-40"
KAROLINA_QOS = "3571_6328"
PROTOCOL = REPO_ROOT / "paper/protocols/EXP-DISC-001.md"
REVIEWED_DISC_ROWS_SHA256 = (
    "60a7cda2549c9e5515b0c2fd1de766402cf36dbd50c4bc5e59502dab8e850e5e"
)
REVIEWED_PROTOCOL_SHA256 = (
    "6759caaf6c9eb1c917f2be4b182d50f49fd52b725c0f73bbe111a518711f78bd"
)
RULES = ("tetra_24point", "tetra_duffy_125point")
BRANCHES = ("elastic", "shear", "left_edge", "right_edge", "apex")
CASE_ORDER = (
    "disc_p4l1_q24_smoke_np64",
    "disc_p4l1_q24_np64",
    "disc_p4l1_q125_np64",
    "disc_p4l2_q24_np128",
    "disc_p4l2_q125_np128",
    "disc_p4l1_q24_tight_np64",
)
EXPECTED_TIERS = ("smoke", "quadrature", "mesh", "mesh_quadrature", "tolerance")
SCALAR_OBSERVABLES = (
    "internal_energy",
    "external_work",
    "total_potential_energy",
    "u_max",
)
VECTOR_KINDS = ("residual", "hessian_action", "branch_map")
FROZEN_GATES: dict[str, float] = {
    "common_scalar_relative_max": 1.0e-3,
    "branch_margin": 1.0e-8,
    "own_rule_residual_match_rtol": 1.0e-9,
    "own_rule_residual_match_atol": 1.0e-10,
    "common_riesz_cross_symmetry_rtol": 1.0e-10,
    "tolerance_to_discretization_effect_ratio_max": 0.25,
}


class AdmissionError(ValueError):
    """A required scientific or provenance gate failed."""


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"nonfinite JSON token {token!r} is forbidden")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_nonfinite)
    if not isinstance(payload, dict):
        raise AdmissionError(f"{path} must contain a JSON object")
    return payload


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_nonfinite)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise AdmissionError(f"{path} must contain a list of JSON objects")
    return [dict(row) for row in payload]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence_file(root: Path, path: Path, role: str) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise AdmissionError(f"{role} escapes the submitted campaign root") from exc
    if not resolved.is_file():
        raise AdmissionError(f"required {role} is missing: {resolved}")
    return {
        "role": role,
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        if "=" not in line:
            raise AdmissionError(f"{path}:{line_number} is not key=value metadata")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise AdmissionError(
                f"{path}:{line_number} has an empty or duplicate metadata key"
            )
        values[key] = value
    return values


def _iso_timestamp(value: object, name: str) -> datetime:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AdmissionError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AdmissionError(f"{name} must include a UTC offset")
    return parsed


def _validate_accounting(
    path: Path,
    *,
    row: dict[str, str],
    job_id: str,
) -> dict[str, Any]:
    payload = _read_json(path)
    if (
        payload.get("schema_id") != SLURM_ACCOUNTING_SCHEMA_ID
        or payload.get("schema_version") != SLURM_ACCOUNTING_SCHEMA_VERSION
    ):
        raise AdmissionError("sacct_final.json has the wrong accounting schema")
    if str(payload.get("job_id", "")) != job_id:
        raise AdmissionError("sacct_final.json has a stale job identity")
    _iso_timestamp(payload.get("collected_at_utc"), "accounting collection time")
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("mode") not in {
        "offline_file",
        "explicit_live_query",
    }:
        raise AdmissionError("sacct_final.json lacks a recognized source mode")
    raw = source.get("raw_parsable2")
    if not isinstance(raw, str) or not raw:
        raise AdmissionError("sacct_final.json does not retain raw parsable2 evidence")
    raw_bytes = raw.encode("utf-8")
    if source.get("sha256") != hashlib.sha256(raw_bytes).hexdigest():
        raise AdmissionError("sacct_final.json raw accounting hash is stale")
    if _integer(source.get("byte_count"), "accounting raw byte count") != len(raw_bytes):
        raise AdmissionError("sacct_final.json raw accounting byte count is stale")
    try:
        reparsed = parse_sacct(raw, job_id=job_id)
    except ValueError as exc:
        raise AdmissionError(f"raw sacct evidence is invalid: {exc}") from exc
    for key in ("job_id", "allocation", "rows", "derived"):
        if payload.get(key) != reparsed[key]:
            raise AdmissionError(f"sacct_final.json {key} disagrees with its raw evidence")

    allocation = dict(payload["allocation"])
    exact = {
        "job_id_raw": job_id,
        "account": KAROLINA_ACCOUNT,
        "qos": KAROLINA_QOS,
        "partition": row["partition"],
        "state": "COMPLETED",
        "exit_code": "0:0",
        "alloc_nodes": int(row["nodes"]),
        "alloc_cpus": int(row["total_ranks"]),
    }
    for key, expected in exact.items():
        if allocation.get(key) != expected:
            raise AdmissionError(
                f"accounting allocation {key} differs from the frozen campaign"
            )
    if str(allocation.get("cluster", "")).lower() != KAROLINA_CLUSTER:
        raise AdmissionError("accounting allocation is not from Karolina")
    elapsed = _integer(allocation.get("elapsed_raw_s"), "accounting elapsed seconds", minimum=1)
    derived = dict(payload["derived"])
    if _integer(
        derived.get("allocated_node_seconds"), "accounting allocated node seconds"
    ) != elapsed * int(row["nodes"]):
        raise AdmissionError("accounting allocated node seconds are inconsistent")
    if _integer(
        derived.get("allocated_cpu_seconds"), "accounting allocated CPU seconds"
    ) != elapsed * int(row["total_ranks"]):
        raise AdmissionError("accounting allocated CPU seconds are inconsistent")
    return {
        "job_id": job_id,
        "cluster": KAROLINA_CLUSTER,
        "account": KAROLINA_ACCOUNT,
        "qos": KAROLINA_QOS,
        "partition": row["partition"],
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed_raw_s": elapsed,
        "allocated_node_seconds": int(derived["allocated_node_seconds"]),
        "allocated_cpu_seconds": int(derived["allocated_cpu_seconds"]),
        "maximum_step_rss_bytes": derived.get("maximum_step_rss_bytes"),
        "maximum_step_vm_size_bytes": derived.get("maximum_step_vm_size_bytes"),
    }


def _validate_job_evidence(
    *,
    row: dict[str, str],
    root: Path,
    manifest: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    batch_job = root / "jobs" / row["case_id"] / f"job_{job_id}"
    slurm_root = root / "slurm"
    paths = {
        "job_metadata": batch_job / "job_metadata.env",
        "environment": batch_job / "environment.txt",
        "execute_log": batch_job / "execute.log",
        "accounting": batch_job / "sacct_final.json",
        "slurm_stdout": slurm_root / f"{row['case_id']}-{job_id}.out",
        "slurm_stderr": slurm_root / f"{row['case_id']}-{job_id}.err",
    }
    artifacts = {
        name: _evidence_file(root, path, name.replace("_", " "))
        for name, path in paths.items()
    }
    metadata = _parse_env(paths["job_metadata"])
    exact_metadata = {
        "case_id": row["case_id"],
        "job_id": job_id,
        "account": KAROLINA_ACCOUNT,
        "nodes": row["nodes"],
        "ntasks": row["total_ranks"],
        "cpus_per_task": "1",
        "matrix_sha256": manifest["matrix_sha256"],
        "git_commit": manifest["source_commit"],
        "git_dirty": "false",
        "allocation_revalidated": "YES",
        "account_qos_revalidated": "YES",
    }
    for key, expected in exact_metadata.items():
        if metadata.get(key) != str(expected):
            raise AdmissionError(f"job metadata {key} differs from the frozen campaign")
    started = _iso_timestamp(metadata.get("started_at"), "job start time")
    finished = _iso_timestamp(metadata.get("finished_at"), "job finish time")
    if finished < started:
        raise AdmissionError("job finish time precedes its start time")
    try:
        datetime.fromisoformat(str(metadata.get("allocation_valid_until", "")))
    except ValueError as exc:
        raise AdmissionError(
            "job metadata lacks a valid revalidated allocation end date"
        ) from exc
    if metadata.get("accounting_status") != "pending_post_job_collection":
        raise AdmissionError("job metadata does not record post-job accounting handoff")
    if artifacts["environment"]["bytes"] <= 0:
        raise AdmissionError("captured job environment is empty")
    accounting = _validate_accounting(paths["accounting"], row=row, job_id=job_id)
    return {
        "artifacts": artifacts,
        "metadata": {
            "source_commit": manifest["source_commit"],
            "source_dirty": False,
            "matrix_sha256": manifest["matrix_sha256"],
            "started_at": metadata["started_at"],
            "finished_at": metadata["finished_at"],
            "allocation_revalidated": True,
            "account_qos_revalidated": True,
            "allocation_valid_until": metadata["allocation_valid_until"],
        },
        "accounting": accounting,
    }


def _content_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower())


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise AdmissionError(f"{name} must be finite") from exc
    if not math.isfinite(converted):
        raise AdmissionError(f"{name} must be finite")
    if nonnegative and converted < 0.0:
        raise AdmissionError(f"{name} must be nonnegative")
    return converted


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise AdmissionError(f"{name} must be an integer") from exc
    if minimum is not None and converted < minimum:
        raise AdmissionError(f"{name} must be at least {minimum}")
    return converted


def _exact_float(actual: object, expected: float, name: str) -> None:
    value = _finite(actual, name)
    if not math.isclose(value, expected, rel_tol=1.0e-14, abs_tol=0.0):
        raise AdmissionError(f"{name} changed from frozen value {expected!r}")


def _matrix_rows(path: Path) -> list[dict[str, str]]:
    if _sha256(PROTOCOL) != REVIEWED_PROTOCOL_SHA256:
        raise AdmissionError("EXP-DISC-001 protocol changed after analyzer review")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if row.get("experiment_id") == EXPERIMENT_ID
        ]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != REVIEWED_DISC_ROWS_SHA256:
        raise AdmissionError("EXP-DISC-001 matrix rows changed after analyzer review")
    if tuple(row.get("case_id") for row in rows) != CASE_ORDER:
        raise AdmissionError("EXP-DISC-001 row identity or order changed")
    expected = {
        CASE_ORDER[0]: ("smoke", "hetero_ssr_L1", "tetra_24point", 64, 1, 1.0e-2, 100, 2.0e-3, 1.0e-4),
        CASE_ORDER[1]: ("quadrature", "hetero_ssr_L1", "tetra_24point", 64, 80, 1.0e-2, 100, 2.0e-3, 1.0e-4),
        CASE_ORDER[2]: ("quadrature", "hetero_ssr_L1", "tetra_duffy_125point", 64, 80, 1.0e-2, 100, 2.0e-3, 1.0e-4),
        CASE_ORDER[3]: ("mesh", "hetero_ssr_L1_2", "tetra_24point", 128, 80, 1.0e-2, 100, 2.0e-3, 1.0e-4),
        CASE_ORDER[4]: ("mesh_quadrature", "hetero_ssr_L1_2", "tetra_duffy_125point", 128, 80, 1.0e-2, 100, 2.0e-3, 1.0e-4),
        CASE_ORDER[5]: ("tolerance", "hetero_ssr_L1", "tetra_24point", 64, 120, 1.0e-3, 200, 2.0e-4, 1.0e-5),
    }
    for row in rows:
        tier, mesh, rule, ranks, maxit, ksp_rtol, ksp_cap, stop, grad = expected[row["case_id"]]
        exact = {
            "tier": tier,
            "mesh_name": mesh,
            "quadrature_rule": rule,
            "total_ranks": str(ranks),
            "element_degree": "4",
            "runner": "p3d_solve",
            "route": "constitutive_ad",
            "assembly_backend": "local_constitutiveAD",
            "solver_backend": "local_pmg",
            "pmg_strategy": "same_mesh_p4_p2_p1",
            "maxit": str(maxit),
            "ksp_max_it": str(ksp_cap),
            "convergence_metric": "reference_elastic_energy",
            "optional": "0",
            "repetitions": "1",
            "warmups": "0",
        }
        for key, value in exact.items():
            if row.get(key) != value:
                raise AdmissionError(f"{row['case_id']} matrix {key} changed")
        for key, value in (("ksp_rtol", ksp_rtol), ("stop_tol", stop), ("grad_stop_tol", grad)):
            _exact_float(row.get(key), value, f"{row['case_id']} matrix {key}")
    return rows


def _manifest(root: Path, matrix_path: Path) -> dict[str, Any]:
    path = root / "prepared_manifest.json"
    payload = _read_json(path)
    if payload.get("status") != "submitted":
        raise AdmissionError(f"{path} does not record a real submission")
    if payload.get("matrix_sha256") != _sha256(matrix_path):
        raise AdmissionError(f"{path} has a stale matrix hash")
    try:
        expected_matrix = str(matrix_path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError as exc:
        raise AdmissionError("analysis matrix is outside the reviewed repository") from exc
    matrix_record = Path(str(payload.get("matrix", "")))
    if matrix_record.is_absolute() or str(matrix_record) != expected_matrix:
        raise AdmissionError(
            f"{path} matrix path is not the reviewed repository-relative path"
        )
    if payload.get("selected_experiments") != [EXPERIMENT_ID]:
        raise AdmissionError(f"{path} does not isolate EXP-DISC-001")
    if (
        payload.get("cluster") != "Karolina CPU"
        or payload.get("account") != KAROLINA_ACCOUNT
        or str(payload.get("qos")) != KAROLINA_QOS
    ):
        raise AdmissionError(f"{path} has stale Karolina account/QOS provenance")
    tiers = payload.get("selected_tiers")
    if not isinstance(tiers, list) or len(tiers) != 1 or tiers[0] not in EXPECTED_TIERS:
        raise AdmissionError(f"{path} must contain exactly one frozen protocol tier")
    if payload.get("source_dirty") is not False:
        raise AdmissionError(f"{path} was prepared from a dirty worktree")
    commit = str(payload.get("source_commit", ""))
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit.lower()):
        raise AdmissionError(f"{path} lacks a valid source commit")
    if payload.get("test_only_commands") is not False:
        raise AdmissionError(f"{path} records test-only commands")
    if payload.get("include_optional") is not False or payload.get("only_optional") is not False:
        raise AdmissionError(f"{path} changed the required-only scope")
    if payload.get("out_root") != ".":
        raise AdmissionError(
            f"{path} out_root is not the relocatable archive root '.'"
        )
    for path_key, hash_key in (
        ("plan_file", "plan_sha256"),
        ("commands_file", "commands_sha256"),
    ):
        relative = Path(str(payload.get(path_key, "")))
        if relative.is_absolute() or not str(relative):
            raise AdmissionError(f"{path} {path_key} must be archive-relative")
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root.resolve())
        except ValueError as exc:
            raise AdmissionError(
                f"{path} {path_key} escapes the copied archive"
            ) from exc
        if not artifact.is_file() or payload.get(hash_key) != _sha256(artifact):
            raise AdmissionError(f"{path} {path_key} is missing or has a stale hash")
    freeze = payload.get("queued_source_freeze")
    if not isinstance(freeze, dict):
        raise AdmissionError(f"{path} lacks a queued source-freeze record")
    freeze_relative = Path(str(freeze.get("path", "")))
    if freeze_relative.is_absolute() or not str(freeze_relative):
        raise AdmissionError(f"{path} source-freeze path must be archive-relative")
    freeze_path = (root / freeze_relative).resolve()
    try:
        freeze_path.relative_to(root.resolve())
    except ValueError as exc:
        raise AdmissionError(
            f"{path} source-freeze path escapes the copied archive"
        ) from exc
    if not freeze_path.is_file() or freeze.get("sha256") != _sha256(freeze_path):
        raise AdmissionError(f"{path} source freeze is missing or has a stale hash")
    stage = payload.get("disc_release_stage")
    tier = str(tiers[0])
    expected_position = EXPECTED_TIERS.index(tier) + 1
    expected_count = 2 if tier == "quadrature" else 1
    expected_stage = {
        "unit": "protocol_stage",
        "stage": tier,
        "position": expected_position,
        "stage_count": len(EXPECTED_TIERS),
        "case_count": expected_count,
        "prerequisite_stage": (
            None if expected_position == 1 else EXPECTED_TIERS[expected_position - 2]
        ),
        "later_stage_release_requires_separate_human_authorization": True,
    }
    if not isinstance(stage, dict) or stage != expected_stage:
        raise AdmissionError(
            f"{path} has an invalid sequential DISC release-stage record"
        )
    return {**payload, "path": str(path), "sha256": _sha256(path), "tier": tiers[0]}


def _load_manifests(roots: Iterable[Path], matrix_path: Path) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for root in roots:
        resolved = Path(root).resolve()
        item = _manifest(resolved, matrix_path)
        tier = str(item["tier"])
        if tier in manifests:
            raise AdmissionError(f"duplicate submitted manifest for tier {tier}")
        item["campaign_root"] = str(resolved)
        manifests[tier] = item
    if set(manifests) != set(EXPECTED_TIERS):
        missing = sorted(set(EXPECTED_TIERS).difference(manifests))
        raise AdmissionError(f"submitted tier coverage is incomplete: {missing}")
    commits = {str(item["source_commit"]) for item in manifests.values()}
    if len(commits) != 1:
        raise AdmissionError("submitted tiers do not use one clean source commit")
    expected_counts = {"smoke": 1, "quadrature": 2, "mesh": 1, "mesh_quadrature": 1, "tolerance": 1}
    for tier, expected in expected_counts.items():
        if _integer(manifests[tier].get("case_count"), f"{tier} case_count") != expected:
            raise AdmissionError(f"{tier} manifest case count changed")
    return manifests


def _matrix_equal(expected: dict[str, str], actual: dict[str, Any]) -> bool:
    return all(str(actual.get(key, "")) == str(value) for key, value in expected.items())


def _within(base: Path, raw: object) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = base / path
    resolved = path.resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise AdmissionError(f"artifact escapes its measure directory: {resolved}") from exc
    return resolved


def _load_state(path: Path, payload: dict[str, Any], row: dict[str, str]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "coords_ref",
            "coords_final",
            "displacement",
            "tetrahedra",
            "free_displacement_reordered",
            "reference_elastic_action",
            "reference_elastic_state_quadratic",
        )
        for name in required:
            if name not in archive:
                raise AdmissionError(f"state archive lacks {name}")
        scalar = lambda name: np.asarray(archive[name]).item()
        coords = np.asarray(archive["coords_ref"], dtype=np.float64)
        final = np.asarray(archive["coords_final"], dtype=np.float64)
        displacement = np.asarray(archive["displacement"], dtype=np.float64)
        tetrahedra = np.asarray(archive["tetrahedra"])
        free = np.asarray(archive["free_displacement_reordered"], dtype=np.float64).reshape(-1)
        action = np.asarray(archive["reference_elastic_action"], dtype=np.float64).reshape(-1)
        quadratic = _finite(scalar("reference_elastic_state_quadratic"), "state Riesz quadratic", nonnegative=True)
        if coords.ndim != 2 or coords.shape[1] != 3 or final.shape != coords.shape or displacement.shape != coords.shape:
            raise AdmissionError("state coordinate arrays are malformed")
        if not np.array_equal(displacement, final - coords):
            raise AdmissionError("state displacement is not exactly final-reference coordinates")
        if free.size == 0 or action.shape != free.shape:
            raise AdmissionError("state Riesz arrays are malformed")
        if not all(np.all(np.isfinite(value)) for value in (coords, final, displacement, free, action)):
            raise AdmissionError("state contains nonfinite values")
        if not math.isclose(float(np.dot(free, action)), quadratic, rel_tol=1.0e-12, abs_tol=1.0e-14):
            raise AdmissionError("state Riesz quadratic is internally inconsistent")
        expected_metadata = {
            "mesh_name": row["mesh_name"],
            "element_degree": int(row["element_degree"]),
            "quadrature_rule_id": row["quadrature_rule"],
            "constraint_variant": "glued_bottom",
            "assembly_backend": "local_constitutiveAD",
            "mpi_ranks": int(row["total_ranks"]),
        }
        for key, expected in expected_metadata.items():
            if scalar(key) != expected:
                raise AdmissionError(f"state {key} disagrees with the matrix")
        _exact_float(scalar("lambda_target"), 1.55, "state load factor")
        state_energy = _finite(scalar("energy"), "state energy")
    root_energy = _finite(payload.get("energy"), "root energy")
    if not math.isclose(state_energy, root_energy, rel_tol=1.0e-12, abs_tol=1.0e-10):
        raise AdmissionError("state and root energies disagree")
    if _within(path.parent, payload.get("state_out")) != path.resolve():
        raise AdmissionError("root output points to a stale state archive")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "coords_ref": coords,
        "displacement": displacement,
        "tetrahedra": tetrahedra,
        "free_displacement": free,
        "reference_action": action,
        "quadratic": quadratic,
    }


def _load_vector_artifact(
    run_dir: Path,
    evaluation: dict[str, Any],
    kind: str,
    *,
    expected_size: int,
) -> np.ndarray:
    key = f"{kind}_artifact"
    artifact = evaluation.get(key)
    if not isinstance(artifact, dict):
        raise AdmissionError(f"{evaluation.get('quadrature_rule_id')} lacks {key}")
    path = _within(run_dir, artifact.get("path"))
    if not path.is_file() or artifact.get("sha256") != _sha256(path):
        raise AdmissionError(f"{key} file is missing or has a stale hash")
    values = np.load(path, allow_pickle=False)
    expected_dtype = np.int8 if kind == "branch_map" else np.float64
    if values.dtype != np.dtype(expected_dtype) or values.shape != (expected_size,):
        raise AdmissionError(f"{key} has the wrong dtype or shape")
    if kind != "branch_map" and not np.all(np.isfinite(values)):
        raise AdmissionError(f"{key} contains nonfinite values")
    content_hash = _content_sha256(values)
    if artifact.get("content_sha256") != content_hash:
        raise AdmissionError(f"{key} content hash is stale")
    declared_hash = evaluation.get(f"{kind}_content_sha256")
    if declared_hash != content_hash:
        raise AdmissionError(f"{kind} summary content hash is stale")
    return np.asarray(values)


def _validate_evaluation(run_dir: Path, raw: object, *, rule: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise AdmissionError(f"{rule} evaluation is not an object")
    row = dict(raw)
    if row.get("quadrature_rule_id") != rule:
        raise AdmissionError("quadrature evaluation order or identity changed")
    expected_points = 24 if rule == RULES[0] else 125
    if _integer(row.get("quadrature_points_per_element"), f"{rule} point count") != expected_points:
        raise AdmissionError(f"{rule} has the wrong number of points per element")
    dofs = _integer(row.get("degrees_of_freedom"), f"{rule} DOFs", minimum=1)
    free_dofs = _integer(row.get("free_degrees_of_freedom"), f"{rule} free DOFs", minimum=1)
    elements = _integer(row.get("elements"), f"{rule} elements", minimum=1)
    samples = _integer(row.get("branch_sample_points"), f"{rule} branch samples", minimum=1)
    if samples != elements * expected_points:
        raise AdmissionError(f"{rule} branch sample count is inconsistent")
    scalars = {}
    for name in (*SCALAR_OBSERVABLES, "full_residual_l2_norm", "full_residual_linf_norm", "free_residual_l2_norm", "free_residual_linf_norm", "full_hessian_action_l2_norm", "full_hessian_action_linf_norm", "free_hessian_action_l2_norm", "free_hessian_action_linf_norm", "minimum_normalized_active_branch_margin", "minimum_raw_principal_value_gap", "minimum_normalized_principal_value_gap", "minimum_normalized_constitutive_denominator"):
        scalars[name] = _finite(row.get(name), f"{rule} {name}", nonnegative=name not in {"internal_energy", "external_work", "total_potential_energy", "minimum_raw_principal_value_gap"})
    counts_raw = row.get("branch_point_counts")
    if not isinstance(counts_raw, dict) or set(counts_raw) != set(BRANCHES):
        raise AdmissionError(f"{rule} branch counts have the wrong labels")
    counts = {name: _integer(counts_raw[name], f"{rule} {name} count", minimum=0) for name in BRANCHES}
    if sum(counts.values()) != samples:
        raise AdmissionError(f"{rule} branch counts do not sum to the sample count")
    near = _integer(row.get("quadrature_points_at_or_below_margin_gate"), f"{rule} near-switch count", minimum=0)
    _exact_float(row.get("branch_margin_gate"), FROZEN_GATES["branch_margin"], f"{rule} branch margin gate")
    residual = _load_vector_artifact(run_dir, row, "residual", expected_size=dofs)
    action = _load_vector_artifact(run_dir, row, "hessian_action", expected_size=dofs)
    branch_map = _load_vector_artifact(run_dir, row, "branch_map", expected_size=samples)
    if np.any(branch_map < 0) or np.any(branch_map >= len(BRANCHES)):
        raise AdmissionError(f"{rule} branch map contains an invalid branch code")
    map_counts = {name: int(np.count_nonzero(branch_map == index)) for index, name in enumerate(BRANCHES)}
    if map_counts != counts:
        raise AdmissionError(f"{rule} branch map and branch counts disagree")
    if not math.isclose(float(np.linalg.norm(residual)), scalars["full_residual_l2_norm"], rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise AdmissionError(f"{rule} residual artifact norm disagrees with its summary")
    if not math.isclose(float(np.linalg.norm(action)), scalars["full_hessian_action_l2_norm"], rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise AdmissionError(f"{rule} action artifact norm disagrees with its summary")
    if rule == RULES[1] and row.get("quadrature_weights_are_strictly_positive") is not True:
        raise AdmissionError("the 125-point reference does not report strictly positive weights")
    return {
        "rule": rule,
        "dofs": dofs,
        "free_dofs": free_dofs,
        "elements": elements,
        "samples": samples,
        "scalars": scalars,
        "branch_counts": counts,
        "near_switch_count": near,
        "residual": residual,
        "action": action,
        "branch_map": branch_map,
    }


def _validate_reference(
    path: Path,
    run_dir: Path,
    state_path: Path,
    row: dict[str, str],
) -> dict[str, Any]:
    payload = _read_json(path)
    exact = {
        "experiment_id": "EXP-DISC-001-P3D-FIXED-STATE-QUADRATURE",
        "status": "completed",
        "mesh_name": row["mesh_name"],
        "element_degree": int(row["element_degree"]),
        "constraint_variant": "glued_bottom",
        "solve_quadrature_rule_id": row["quadrature_rule"],
        "reference_rule_id": RULES[1],
        "common_free_dof_set": True,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise AdmissionError(f"quadrature reference {key} disagrees with policy")
    if _within(run_dir, payload.get("state_path")) != state_path.resolve():
        raise AdmissionError("quadrature reference points to a stale state")
    _exact_float(payload.get("lambda_target"), 1.55, "quadrature-reference load factor")
    if not _is_sha256(payload.get("common_direction_content_sha256")):
        raise AdmissionError("quadrature reference lacks a common direction hash")
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 2:
        raise AdmissionError("quadrature reference must contain exactly 24/125 evaluations")
    checked = {
        rule: _validate_evaluation(run_dir, raw, rule=rule)
        for rule, raw in zip(RULES, evaluations, strict=True)
    }
    first, second = (checked[rule] for rule in RULES)
    for key in ("dofs", "free_dofs", "elements"):
        if first[key] != second[key]:
            raise AdmissionError(f"24/125 evaluations disagree in {key}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "common_direction_content_sha256": payload["common_direction_content_sha256"],
        "evaluations": checked,
    }


def _validate_root_output(
    payload: dict[str, Any],
    row: dict[str, str],
    *,
    source_commit: str,
    job_id: str,
    smoke: bool,
) -> dict[str, Any]:
    exact = {
        "assembly_backend": "local_constitutiveAD",
        "solver_backend": "local_pmg",
        "mesh_name": row["mesh_name"],
        "elem_degree": 4,
        "quadrature_rule_id": row["quadrature_rule"],
        "constraint_variant": "glued_bottom",
        "pmg_strategy": "same_mesh_p4_p2_p1",
        "ranks": int(row["total_ranks"]),
        "maxit": int(row["maxit"]),
        "ksp_max_it": int(row["ksp_max_it"]),
        "line_search": "armijo",
        "use_trust_region": True,
        "trust_subproblem_line_search": True,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise AdmissionError(f"root output {key} disagrees with the matrix")
    git = payload.get("git")
    if not isinstance(git, dict) or git.get("dirty") is not False or git.get("commit") != source_commit:
        raise AdmissionError("root output does not match the clean submitted commit")
    metadata = payload.get("job_metadata")
    if not isinstance(metadata, dict) or str(metadata.get("slurm_job_id", "")) != job_id:
        raise AdmissionError("root output has stale Slurm job identity")
    _exact_float(payload.get("lambda_target"), 1.55, "root load factor")
    for key in ("ksp_rtol", "stop_tol", "grad_stop_tol"):
        _exact_float(payload.get(key), float(row[key]), f"root {key}")
    _exact_float(payload.get("linesearch_tol"), 1.0e-3, "root line-search tolerance")
    expected_points = 24 if row["quadrature_rule"] == RULES[0] else 125
    if _integer(payload.get("quadrature_points"), "root quadrature points") != expected_points:
        raise AdmissionError("root quadrature point count changed")
    validation = validate_p3d_solve_output(payload, row)
    if smoke:
        return validation
    if payload.get("status") != "completed" or payload.get("solver_success") is not True:
        raise AdmissionError("non-smoke endpoint did not complete successfully")
    convergence = dict(payload.get("nonlinear_convergence") or {})
    dual = _finite(dict(convergence.get("absolute_dual_residual") or {}).get("value"), "terminal dual residual", nonnegative=True)
    correction = _finite(dict(convergence.get("relative_correction") or {}).get("value"), "terminal relative correction", nonnegative=True)
    if dual >= float(row["grad_stop_tol"]):
        raise AdmissionError("terminal dual residual did not pass the row tolerance")
    if correction >= float(row["stop_tol"]):
        raise AdmissionError("terminal relative correction did not pass the row tolerance")
    return validation


def _analyze_case(
    row: dict[str, str],
    root: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    jobs_root = root / "cases" / row["case_id"]
    jobs = sorted(path for path in jobs_root.glob("job_*") if path.is_dir()) if jobs_root.is_dir() else []
    if len(jobs) != 1:
        raise AdmissionError(f"{row['case_id']} requires exactly one job directory")
    job = jobs[0]
    job_id = job.name.removeprefix("job_")
    if not job_id:
        raise AdmissionError(f"{row['case_id']} job ID is empty")
    if not _matrix_equal(row, _read_json(job / "matrix_row.json")):
        raise AdmissionError(f"{row['case_id']} executed matrix row is stale")
    job_evidence = _validate_job_evidence(
        row=row,
        root=root,
        manifest=manifest,
        job_id=job_id,
    )
    records = _read_json_list(job / "run_records.json")
    if len(records) != 1:
        raise AdmissionError(f"{row['case_id']} must contain exactly one measure record")
    record = records[0]
    if record.get("kind") != "measure" or _integer(record.get("index"), "record index") != 1:
        raise AdmissionError(f"{row['case_id']} run record is not the frozen measure")
    if _integer(record.get("returncode"), "wrapper returncode") != 0:
        raise AdmissionError(f"{row['case_id']} wrapper or mandatory postprocessor failed")
    if dict(record.get("scientific_validation") or {}).get("status") != "passed":
        raise AdmissionError(f"{row['case_id']} executor scientific validation did not pass")
    reference_record = record.get("quadrature_reference")
    if not isinstance(reference_record, dict) or _integer(reference_record.get("returncode"), "quadrature-reference returncode") != 0:
        raise AdmissionError(f"{row['case_id']} mandatory quadrature evaluator failed")
    if dict(record.get("quadrature_reference_validation") or {}).get("status") != "passed":
        raise AdmissionError(f"{row['case_id']} quadrature output validation did not pass")
    run_dir = job / "measure_01"
    output_path = run_dir / "output.json"
    state_path = run_dir / "state.npz"
    reference_path = run_dir / "quadrature_reference.json"
    for path in (output_path, state_path, reference_path):
        if not path.is_file():
            raise AdmissionError(f"{row['case_id']} is missing {path.name}")
    payload = _read_json(output_path)
    root_validation = _validate_root_output(
        payload,
        row,
        source_commit=str(manifest["source_commit"]),
        job_id=job_id,
        smoke=row["tier"] == "smoke",
    )
    state = _load_state(state_path, payload, row)
    reference = _validate_reference(reference_path, run_dir, state_path, row)
    own = reference["evaluations"][row["quadrature_rule"]]
    root_gradient = _finite(payload.get("final_grad_norm"), "root final gradient", nonnegative=True)
    own_residual = float(own["scalars"]["free_residual_l2_norm"])
    if not math.isclose(
        root_gradient,
        own_residual,
        rel_tol=FROZEN_GATES["own_rule_residual_match_rtol"],
        abs_tol=FROZEN_GATES["own_rule_residual_match_atol"],
    ):
        raise AdmissionError("independent own-rule free residual disagrees with the root gradient")
    own_gate = bool(row["tier"] == "smoke" or own_residual < float(row["grad_stop_tol"]))
    if not own_gate:
        raise AdmissionError("independent own-rule free residual failed the row stopping gate")
    public = {
        "case_id": row["case_id"],
        "tier": row["tier"],
        "mesh_name": row["mesh_name"],
        "solve_quadrature_rule": row["quadrature_rule"],
        "rank_count": int(row["total_ranks"]),
        "status": "smoke_plumbing_passed" if row["tier"] == "smoke" else "endpoint_admitted",
        "job_id": job_id,
        "job_path": str(job),
        "job_evidence": job_evidence,
        "output_sha256": _sha256(output_path),
        "state_sha256": state["sha256"],
        "quadrature_reference_sha256": reference["sha256"],
        "own_rule_free_residual_l2": own_residual,
        "own_rule_residual_target": float(row["grad_stop_tol"]),
        "own_rule_residual_gate_passed": own_gate,
        "riesz_validation": root_validation,
        "common_evaluations": {
            rule: {
                "scalars": reference["evaluations"][rule]["scalars"],
                "branch_counts": reference["evaluations"][rule]["branch_counts"],
                "near_switch_count": reference["evaluations"][rule]["near_switch_count"],
            }
            for rule in RULES
        },
    }
    private = {"row": row, "payload": payload, "state": state, "reference": reference}
    return public, private


def _vector_effect(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape != right.shape:
        raise AdmissionError("common-evaluator vectors have different shapes")
    delta = np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
    absolute = float(np.linalg.norm(delta))
    relative = absolute / max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(np.float64).tiny,
    )
    return {
        "absolute_l2_difference": absolute,
        "relative_l2_difference": relative,
        "absolute_linf_difference": float(np.linalg.norm(delta, ord=np.inf)),
    }


def _scalar_effects(
    left: dict[str, Any], right: dict[str, Any], *, labels: tuple[str, str]
) -> dict[str, dict[str, float]]:
    energy_scale = max(
        *(
            abs(float(left["scalars"][name]))
            for name in ("internal_energy", "external_work", "total_potential_energy")
        ),
        *(
            abs(float(right["scalars"][name]))
            for name in ("internal_energy", "external_work", "total_potential_energy")
        ),
        np.finfo(np.float64).tiny,
    )
    result: dict[str, dict[str, float]] = {}
    for name in SCALAR_OBSERVABLES:
        left_value = float(left["scalars"][name])
        right_value = float(right["scalars"][name])
        scale = (
            energy_scale
            if name == "total_potential_energy"
            else max(abs(left_value), abs(right_value), np.finfo(np.float64).tiny)
        )
        result[name] = {
            labels[0]: left_value,
            labels[1]: right_value,
            "absolute_difference": abs(left_value - right_value),
            "relative_difference": abs(left_value - right_value) / scale,
            "relative_scale": scale,
        }
    return result


def _same_mesh_comparison(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    ls = left["state"]
    rs = right["state"]
    for key in ("coords_ref", "tetrahedra"):
        if not np.array_equal(ls[key], rs[key]):
            raise AdmissionError(f"{label} states disagree in {key}")
    lu = np.asarray(ls["free_displacement"], dtype=np.float64)
    ru = np.asarray(rs["free_displacement"], dtype=np.float64)
    la = np.asarray(ls["reference_action"], dtype=np.float64)
    ra = np.asarray(rs["reference_action"], dtype=np.float64)
    if lu.shape != ru.shape or la.shape != ra.shape or lu.shape != la.shape:
        raise AdmissionError(f"{label} Riesz arrays have different shapes")
    cross_left = float(np.dot(lu, ra))
    cross_right = float(np.dot(ru, la))
    cross_scale = max(abs(cross_left), abs(cross_right), np.finfo(np.float64).tiny)
    cross_symmetry = abs(cross_left - cross_right) / cross_scale
    if cross_symmetry > FROZEN_GATES["common_riesz_cross_symmetry_rtol"]:
        raise AdmissionError(f"{label} 24/125 reference-elastic actions are not cross-symmetric")
    difference = ru - lu
    action_difference = ra - la
    distance_squared = float(np.dot(difference, action_difference))
    scale = max(float(ls["quadratic"]), float(rs["quadratic"]), np.finfo(np.float64).tiny)
    if distance_squared < -1.0e-12 * scale:
        raise AdmissionError(f"{label} common reference-elastic distance is negative")
    distance = math.sqrt(max(0.0, distance_squared))
    relative_distance = distance / max(
        math.sqrt(max(0.0, float(ls["quadratic"]))),
        math.sqrt(max(0.0, float(rs["quadratic"]))),
        np.finfo(np.float64).tiny,
    )
    le = left["reference"]["evaluations"][RULES[1]]
    re = right["reference"]["evaluations"][RULES[1]]
    scalar_effects = _scalar_effects(le, re, labels=("left", "right"))
    max_scalar = max(value["relative_difference"] for value in scalar_effects.values())
    residual = _vector_effect(le["residual"], re["residual"])
    action = _vector_effect(le["action"], re["action"])
    left_map = np.asarray(le["branch_map"], dtype=np.int8)
    right_map = np.asarray(re["branch_map"], dtype=np.int8)
    if left_map.shape != right_map.shape:
        raise AdmissionError(f"{label} common 125-point branch maps have different shapes")
    changed = int(np.count_nonzero(left_map != right_map))
    return {
        "label": label,
        "left_case_id": left["row"]["case_id"],
        "right_case_id": right["row"]["case_id"],
        "evaluation_rule": RULES[1],
        "state_relative_reference_elastic_riesz": relative_distance,
        "state_reference_elastic_riesz_distance": distance,
        "reference_elastic_cross_symmetry_relative_defect": cross_symmetry,
        "state_max_absolute_displacement_difference": float(
            np.max(np.abs(np.asarray(rs["displacement"]) - np.asarray(ls["displacement"])))
        ),
        "scalar_effects": scalar_effects,
        "maximum_principal_scalar_relative_difference": max_scalar,
        "common_scalar_gate_passed": bool(max_scalar < FROZEN_GATES["common_scalar_relative_max"]),
        "residual_vector_effect": residual,
        "hessian_action_vector_effect": action,
        "branch_map_changed_samples": changed,
        "branch_map_changed_fraction": float(changed / max(1, left_map.size)),
        "near_switch_samples": int(le["near_switch_count"] + re["near_switch_count"]),
        "riesz_interpretation": (
            "24- and 125-point rules both exactly integrate the elementwise degree-six "
            "reference-elastic P4 form; saved actions additionally pass the frozen "
            "cross-symmetry check"
        ),
    }


def _mesh_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    le = left["reference"]["evaluations"][RULES[1]]
    re = right["reference"]["evaluations"][RULES[1]]
    scalar_effects = _scalar_effects(le, re, labels=("coarse", "fine"))
    max_scalar = max(value["relative_difference"] for value in scalar_effects.values())
    left_fractions = np.asarray([le["branch_counts"][name] / le["samples"] for name in BRANCHES])
    right_fractions = np.asarray([re["branch_counts"][name] / re["samples"] for name in BRANCHES])
    return {
        "label": "mesh_enriched_endpoint_comparison",
        "coarse_case_id": left["row"]["case_id"],
        "fine_case_id": right["row"]["case_id"],
        "evaluation_rule": RULES[1],
        "scalar_effects": scalar_effects,
        "maximum_principal_scalar_relative_difference": max_scalar,
        "common_scalar_gate_passed": bool(max_scalar < FROZEN_GATES["common_scalar_relative_max"]),
        "branch_fraction_l1_difference": float(np.sum(np.abs(left_fractions - right_fractions))),
        "pointwise_branch_map_compared": False,
        "state_error_quoted": False,
        "state_error_reason": "different meshes require a declared common-space projection",
    }


def analyze(
    matrix_path: Path,
    campaign_roots: Iterable[Path],
) -> dict[str, Any]:
    matrix_path = Path(matrix_path).resolve()
    roots = [Path(path).resolve() for path in campaign_roots]
    rows = _matrix_rows(matrix_path)
    manifests = _load_manifests(roots, matrix_path)
    public_cases: list[dict[str, Any]] = []
    records: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    for row in rows:
        manifest = manifests[row["tier"]]
        root = Path(str(manifest["campaign_root"]))
        try:
            public, private = _analyze_case(row, root, manifest)
            public_cases.append(public)
            records[row["case_id"]] = private
        except (AdmissionError, OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append({"case_id": row["case_id"], "reason": str(exc)})
            public_cases.append(
                {
                    "case_id": row["case_id"],
                    "tier": row["tier"],
                    "mesh_name": row["mesh_name"],
                    "solve_quadrature_rule": row["quadrature_rule"],
                    "status": "invalid",
                    "reason": str(exc),
                }
            )
    comparisons: dict[str, Any] = {}
    terminal_decision = "INVALID"
    decision_reasons: list[str] = []
    if not failures:
        l1_quadrature = _same_mesh_comparison(
            records[CASE_ORDER[1]], records[CASE_ORDER[2]], label="l1_quadrature_endpoints"
        )
        l2_quadrature = _same_mesh_comparison(
            records[CASE_ORDER[3]], records[CASE_ORDER[4]], label="l2_quadrature_endpoints"
        )
        tolerance = _same_mesh_comparison(
            records[CASE_ORDER[1]], records[CASE_ORDER[5]], label="l1_production_vs_tight_tolerance"
        )
        mesh = _mesh_comparison(records[CASE_ORDER[2]], records[CASE_ORDER[4]])
        discretization_effect = max(
            l1_quadrature["maximum_principal_scalar_relative_difference"],
            l2_quadrature["maximum_principal_scalar_relative_difference"],
            mesh["maximum_principal_scalar_relative_difference"],
        )
        tolerance_effect = tolerance["maximum_principal_scalar_relative_difference"]
        tolerance_ratio = tolerance_effect / max(discretization_effect, np.finfo(np.float64).tiny)
        tolerance["reference_discretization_scalar_effect"] = discretization_effect
        tolerance["effect_ratio_to_discretization"] = tolerance_ratio
        tolerance["materially_smaller_gate"] = FROZEN_GATES[
            "tolerance_to_discretization_effect_ratio_max"
        ]
        tolerance["materially_smaller"] = bool(
            tolerance_effect == 0.0
            if discretization_effect == 0.0
            else tolerance_ratio
            < FROZEN_GATES["tolerance_to_discretization_effect_ratio_max"]
        )
        comparisons = {
            "l1_quadrature": l1_quadrature,
            "l2_quadrature": l2_quadrature,
            "mesh": mesh,
            "tolerance": tolerance,
        }
        same_mesh = (l1_quadrature, l2_quadrature)
        if any(item["branch_map_changed_samples"] > 0 for item in same_mesh):
            decision_reasons.append("common_125_point_branch_maps_changed")
        if any(item["near_switch_samples"] > 0 for item in (*same_mesh, tolerance)):
            decision_reasons.append("common_evaluator_contains_near_switch_samples")
        if not all(item["common_scalar_gate_passed"] for item in same_mesh):
            decision_reasons.append("quadrature_common_scalar_gate_failed")
        if not mesh["common_scalar_gate_passed"]:
            decision_reasons.append("consecutive_enriched_mesh_scalar_gate_failed")
        if not tolerance["materially_smaller"]:
            decision_reasons.append("algebraic_tolerance_effect_not_materially_smaller")
        terminal_decision = "VERIFIED_POLICY" if not decision_reasons else "ENDPOINT_SENSITIVITY"
    else:
        decision_reasons.extend(f"{item['case_id']}: {item['reason']}" for item in failures)
    return {
        "schema": {"id": SCHEMA_ID, "version": SCHEMA_VERSION},
        "experiment_id": EXPERIMENT_ID,
        "matrix_path": str(matrix_path),
        "matrix_sha256": _sha256(matrix_path),
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": _sha256(PROTOCOL),
        "analysis_script_path": str(Path(__file__).resolve()),
        "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        "frozen_gates": FROZEN_GATES,
        "manifests": {tier: manifests[tier] for tier in EXPECTED_TIERS},
        "source_commit": next(iter({item["source_commit"] for item in manifests.values()})),
        "all_six_rows_admitted": not failures,
        "publication_evidence_valid": not failures,
        "discretization_policy_verified": terminal_decision == "VERIFIED_POLICY",
        "terminal_decision": terminal_decision,
        "decision_reasons": decision_reasons,
        "case_failures": failures,
        "cases": public_cases,
        "comparisons": comparisons,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        action="append",
        required=True,
        help="Repeat for the five separately submitted protocol-tier roots.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--require-valid-evidence",
        action="store_true",
        help="Return 2 after writing the map unless all six rows are admitted.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = analyze(Path(args.matrix), list(args.campaign_root))
    atomic_write_json(Path(args.output_json), result, nonfinite_as_null=False)
    print(
        json.dumps(
            {
                "terminal_decision": result["terminal_decision"],
                "all_six_rows_admitted": result["all_six_rows_admitted"],
                "output_json": str(Path(args.output_json).resolve()),
            },
            indent=2,
            allow_nan=False,
        )
    )
    if args.require_valid_evidence and not result["publication_evidence_valid"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

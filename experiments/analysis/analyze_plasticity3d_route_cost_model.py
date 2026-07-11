#!/usr/bin/env python3
"""Admit, map, and optionally model EXP-ROUTE-001 fixed-state records.

The program is deliberately fail-closed.  It reads the machine-readable
analysis contract, reconstructs every saved state and tangent action, and
admits timing only after route equivalence has passed for the complete active
route set at that comparison point.  Missing and censored routes remain
visible.  Model fitting is skipped until the frozen train/holdout design has
enough admissible rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analysis import aggregate_route_tranche_manifests
from experiments.runners.paper_revision_karolina.tier_b_stopping import (
    POLICY_PATH as TIER_B_STOPPING_POLICY_PATH,
    sha256_file as stopping_sha256_file,
    validate_stop_adjudication,
)
from src.core.benchmark.run_record import atomic_write_json


DEFAULT_CONTRACT = (
    REPO_ROOT / "paper/protocols/EXP-ROUTE-001-analysis-contract.json"
)
REVIEWED_MATRIX = (
    REPO_ROOT / "experiments/runners/paper_revision_karolina/campaign_matrix.csv"
)
WORKSTATION_ENVIRONMENT_CONTRACT = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "True",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
}


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"nonfinite JSON token {token!r} is forbidden")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=_reject_nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _reviewed_route_rows(contract: dict[str, Any]) -> dict[str, dict[str, str]]:
    if _sha256_file(REVIEWED_MATRIX) != str(
        contract["publication_model_input_gates"]["karolina_matrix_sha256"]
    ):
        raise ValueError("local reviewed Karolina matrix hash disagrees with the contract")
    with REVIEWED_MATRIX.open(newline="", encoding="utf-8") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if row.get("experiment_id") == contract["experiment_id"]
        ]
    return {row["case_id"]: row for row in rows}


def _matrix_rows_equal(
    planned: dict[str, str], observed: dict[str, Any]
) -> bool:
    return set(observed) == set(planned) and all(
        str(observed.get(key, "")) == str(value) for key, value in planned.items()
    )


def _bind_fixed_record_to_reviewed_matrix(
    path: Path,
    payload: dict[str, Any],
    *,
    contract: dict[str, Any],
) -> dict[str, str]:
    job_dir = path.parents[2]
    case_id = job_dir.parent.name
    reviewed = _reviewed_route_rows(contract).get(case_id)
    if reviewed is None:
        raise ValueError("fixed-state output case is absent from the reviewed matrix")
    matrix_row_path = job_dir / "matrix_row.json"
    observed = _read_json(matrix_row_path)
    if not _matrix_rows_equal(reviewed, observed):
        raise ValueError("executed fixed-state matrix row differs from the reviewed row")
    if reviewed["runner"] != "p3d_fixed_state_block":
        raise ValueError("fixed-state output is bound to the wrong reviewed runner")
    route = str(payload.get("route", ""))
    order = reviewed["route_order"].split("|")
    if route not in order:
        raise ValueError("fixed-state output route is absent from its reviewed block")
    exact_payload = {
        "experiment_id": reviewed["experiment_id"],
        "tier": reviewed["tier"],
        "mesh_name": reviewed["mesh_name"],
        "element_degree": int(reviewed["element_degree"]),
        "quadrature_rule_id": reviewed["quadrature_rule"],
        "state_label": reviewed["state_label"],
        "state_amplitude": float(reviewed["state_amplitude"]),
        "mpi_ranks": int(reviewed["total_ranks"]),
        "probe_count": int(reviewed["probe_count"]),
        "route": route,
        "constraint_variant": "glued_bottom",
        "lambda_target": 1.55,
        "warmup_repetitions": int(reviewed["warmups"]),
        "measured_repetitions": int(reviewed["repetitions"]),
    }
    for key, expected in exact_payload.items():
        if payload.get(key) != expected:
            raise ValueError(f"fixed-state output {key} differs from its reviewed row")
    design = dict(payload.get("comparison_design") or {})
    exact = {
        "comparison_id": reviewed["comparison_id"],
        "block_repetition": int(reviewed["block_repetition"]),
        "route_order_position": order.index(route),
        "route_order_policy": reviewed["route_order_policy"],
        "timing_reduction": reviewed["timing_reduction"],
    }
    for key, expected in exact.items():
        if design.get(key) != expected:
            raise ValueError(f"fixed-state output {key} differs from its reviewed row")
    return reviewed


def _bind_factor_record_to_reviewed_matrix(
    path: Path,
    payload: dict[str, Any],
    *,
    contract: dict[str, Any],
) -> dict[str, str]:
    job_dir = path.parents[1]
    case_id = job_dir.parent.name
    reviewed = _reviewed_route_rows(contract).get(case_id)
    if reviewed is None:
        raise ValueError("factor output case is absent from the reviewed matrix")
    observed = _read_json(job_dir / "matrix_row.json")
    if not _matrix_rows_equal(reviewed, observed):
        raise ValueError("executed factor matrix row differs from the reviewed row")
    if reviewed["runner"] != "route_factor_microbench":
        raise ValueError("factor output is bound to the wrong reviewed runner")
    exact = {
        "experiment_id": reviewed["experiment_id"],
        "tier": reviewed["tier"],
        "block_repetition": int(reviewed["block_repetition"]),
        "repetitions": int(reviewed["repetitions"]),
        "timing_reduction": reviewed["timing_reduction"],
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise ValueError(f"factor output {key} differs from its reviewed row")
    ranks = int(payload["results"][0]["mpi_ranks"])
    if ranks != int(reviewed["total_ranks"]):
        raise ValueError("factor output rank count differs from its reviewed row")
    return reviewed


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence_entry(path: Path, role: str) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"required evidence file is missing: {path}")
    return {"role": role, "path": str(path), "sha256": _sha256_file(path)}


def _parse_env_record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"required job metadata is missing: {path}")
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def _cluster_batch_evidence(
    *,
    campaign_root: Path,
    case_id: str,
    job_id: str,
    expected_commit: str,
) -> list[dict[str, str]]:
    batch_job = campaign_root / "jobs" / case_id / f"job_{job_id}"
    metadata_path = batch_job / "job_metadata.env"
    environment_path = batch_job / "environment.txt"
    execute_log = batch_job / "execute.log"
    accounting_path = batch_job / "sacct_final.json"
    stdout_path = campaign_root / "slurm" / f"{case_id}-{job_id}.out"
    stderr_path = campaign_root / "slurm" / f"{case_id}-{job_id}.err"
    metadata = _parse_env_record(metadata_path)
    exact = {
        "case_id": case_id,
        "job_id": job_id,
        "account": "fta-26-40",
        "qos": "3571_6328",
        "cluster": "karolina",
        "git_commit": expected_commit,
        "git_dirty": "false",
        "allocation_revalidated": "YES",
        "account_qos_revalidated": "YES",
    }
    for key, expected in exact.items():
        if str(metadata.get(key, "")).lower() != str(expected).lower():
            raise ValueError(f"batch job metadata {key} differs from reviewed provenance")
    if not metadata.get("allocation_valid_until"):
        raise ValueError("batch job metadata lacks allocation validity")
    accounting = _read_json(accounting_path)
    if (
        accounting.get("schema_id")
        != "fenics-nonlinear-energies.slurm-accounting-snapshot"
        or int(accounting.get("schema_version", -1)) != 1
        or str(accounting.get("job_id", "")) != job_id
    ):
        raise ValueError("Slurm accounting identity or schema is invalid")
    source = dict(accounting.get("source") or {})
    raw = source.get("raw_parsable2")
    if not isinstance(raw, str) or not raw:
        raise ValueError("Slurm accounting lacks raw parsable2 evidence")
    raw_bytes = raw.encode("utf-8")
    if (
        source.get("sha256") != hashlib.sha256(raw_bytes).hexdigest()
        or int(source.get("byte_count", -1)) != len(raw_bytes)
    ):
        raise ValueError("Slurm accounting raw evidence hash is invalid")
    allocation = dict(accounting.get("allocation") or {})
    if (
        str(allocation.get("job_id_raw", "")) != job_id
        or str(allocation.get("cluster", "")).lower() != "karolina"
        or allocation.get("account") != "fta-26-40"
        or allocation.get("qos") != "3571_6328"
        or allocation.get("state") != "COMPLETED"
        or allocation.get("exit_code") != "0:0"
    ):
        raise ValueError("Slurm accounting does not prove a successful Karolina job")
    return [
        _evidence_entry(metadata_path, "batch_job_metadata"),
        _evidence_entry(environment_path, "batch_environment"),
        _evidence_entry(execute_log, "batch_execute_log"),
        _evidence_entry(accounting_path, "settled_slurm_accounting"),
        _evidence_entry(stdout_path, "slurm_stdout"),
        _evidence_entry(stderr_path, "slurm_stderr"),
    ]


def _containing_campaign_root(path: Path, outer_root: Path) -> Path:
    outer = outer_root.resolve()
    current = path.resolve().parent
    while True:
        if (current / "prepared_manifest.json").is_file():
            return current
        if current == outer:
            break
        if outer not in current.parents:
            break
        current = current.parent
    raise ValueError(f"evidence path is not inside a prepared tranche: {path}")


def _cluster_fixed_evidence_files(
    root: Path,
    output_path: Path,
    payload: dict[str, Any],
) -> list[dict[str, str]]:
    route_dir = output_path.parent
    measure_dir = route_dir.parent
    job_dir = measure_dir.parent
    case_id = job_dir.parent.name
    job_id = job_dir.name.removeprefix("job_")
    campaign_root = _containing_campaign_root(output_path, root)
    paths = [
        (output_path, "fixed_route_output"),
        (
            _resolve_record_artifact(
                output_path, payload.get("action_out"), "action_out"
            ),
            "fixed_route_action",
        ),
        (route_dir / "command.txt", "fixed_route_command"),
        (measure_dir / "block_result.json", "paired_block_result"),
        (job_dir / "matrix_row.json", "executed_matrix_row"),
        (job_dir / "run_records.json", "wrapper_run_records"),
    ]
    direct = str(payload.get("direct_matrix_out", "")).strip()
    if direct:
        paths.append(
            (
                _resolve_record_artifact(
                    output_path, direct, "direct_matrix_out"
                ),
                "direct_csr_matrix",
            )
        )
    entries: list[dict[str, str]] = []
    root_resolved = root.resolve()
    for path, role in paths:
        resolved = path.resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"declared evidence escapes the campaign root: {resolved}") from exc
        entries.append(_evidence_entry(resolved, role))
    entries.extend(
        _cluster_batch_evidence(
            campaign_root=campaign_root,
            case_id=case_id,
            job_id=job_id,
            expected_commit=str(dict(payload.get("git") or {}).get("commit", "")),
        )
    )
    return entries


def _cluster_factor_evidence_files(
    root: Path,
    output_path: Path,
    payload: dict[str, Any],
) -> list[dict[str, str]]:
    measure_dir = output_path.parent
    job_dir = measure_dir.parent
    case_id = job_dir.parent.name
    job_id = job_dir.name.removeprefix("job_")
    campaign_root = _containing_campaign_root(output_path, root)
    paths = (
        (output_path, "factor_output"),
        (measure_dir / "command.txt", "factor_command"),
        (job_dir / "matrix_row.json", "factor_matrix_row"),
        (job_dir / "run_records.json", "factor_run_records"),
    )
    root_resolved = root.resolve()
    entries: list[dict[str, str]] = []
    for path, role in paths:
        resolved = path.resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"factor evidence escapes campaign root: {resolved}") from exc
        entries.append(_evidence_entry(resolved, role))
    entries.extend(
        _cluster_batch_evidence(
            campaign_root=campaign_root,
            case_id=case_id,
            job_id=job_id,
            expected_commit=str(dict(payload.get("git") or {}).get("commit", "")),
        )
    )
    return entries


def _workstation_fixed_evidence_files(
    root: Path,
    output_path: Path,
    payload: dict[str, Any],
) -> list[dict[str, str]]:
    route_dir = output_path.parent
    measure_dir = route_dir.parent
    job_dir = measure_dir.parent
    paths = [
        (output_path, "workstation_route_output"),
        (
            _resolve_record_artifact(
                output_path, payload.get("action_out"), "action_out"
            ),
            "workstation_route_action",
        ),
        (route_dir / "command.txt", "workstation_route_command"),
        (measure_dir / "block_result.json", "workstation_block_result"),
        (job_dir / "matrix_row.json", "workstation_matrix_row"),
        (job_dir / "run_records.json", "workstation_run_records"),
        (root / "workstation_manifest.json", "workstation_manifest"),
        (root / "workstation_plan.json", "workstation_plan"),
        (root / "environment.json", "workstation_environment"),
    ]
    direct = str(payload.get("direct_matrix_out", "")).strip()
    if direct:
        paths.append(
            (
                _resolve_record_artifact(
                    output_path, direct, "direct_matrix_out"
                ),
                "workstation_direct_csr_matrix",
            )
        )
    root_resolved = root.resolve()
    entries: list[dict[str, str]] = []
    for path, role in paths:
        resolved = path.resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"workstation evidence escapes campaign root: {resolved}") from exc
        entries.append(_evidence_entry(resolved, role))
    return entries


def _git_metadata() -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--short")),
    }


def _safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--source must have the form HARDWARE_ID=PATH")
    hardware_id, raw_path = value.split("=", 1)
    hardware_id = hardware_id.strip()
    if not hardware_id:
        raise ValueError("--source hardware identifier cannot be empty")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"source directory does not exist: {path}")
    return hardware_id, path


def _configuration_maps(
    contract: dict[str, Any],
) -> tuple[dict[tuple[str, int, str], str], dict[str, dict[str, Any]]]:
    by_signature: dict[tuple[str, int, str], str] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for row in contract["configurations"]:
        config_id = str(row["configuration_id"])
        signature = (
            str(row["mesh_name"]),
            int(row["element_degree"]),
            str(row["quadrature_rule_id"]),
        )
        if config_id in by_id or signature in by_signature:
            raise ValueError("configuration IDs and signatures must be unique")
        by_id[config_id] = dict(row)
        by_signature[signature] = config_id
    return by_signature, by_id


def _state_maps(
    contract: dict[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    label_to_id: dict[str, str] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for row in contract["states"]:
        state_id = str(row["state_id"])
        by_id[state_id] = dict(row)
        for label in row["accepted_labels"]:
            label = str(label)
            if label in label_to_id:
                raise ValueError(f"state label {label!r} is ambiguous")
            label_to_id[label] = state_id
    return label_to_id, by_id


def _record_slot(
    payload: dict[str, Any],
    *,
    hardware_id: str,
    contract: dict[str, Any],
) -> tuple[str, str, str, int, str]:
    config_signatures, _ = _configuration_maps(contract)
    state_labels, states = _state_maps(contract)
    signature = (
        str(payload.get("mesh_name", "")),
        int(payload.get("element_degree", -1)),
        str(payload.get("quadrature_rule_id", "")),
    )
    if signature not in config_signatures:
        raise ValueError(f"record has out-of-contract configuration {signature}")
    state_label = str(payload.get("state_label", ""))
    if state_label not in state_labels:
        raise ValueError(f"record has out-of-contract state label {state_label!r}")
    state_id = state_labels[state_label]
    amplitude = _safe_float(payload.get("state_amplitude"))
    expected_amplitude = float(states[state_id]["amplitude"])
    if amplitude is None or amplitude != expected_amplitude:
        raise ValueError(
            f"state amplitude {amplitude!r} does not equal frozen value "
            f"{expected_amplitude}"
        )
    route = str(payload.get("route", ""))
    if route not in contract["route_order"]:
        raise ValueError(f"record has unrecognized route {route!r}")
    ranks = int(payload.get("mpi_ranks", 0))
    return hardware_id, config_signatures[signature], state_id, ranks, route


def _matrix_slot(
    row: dict[str, Any],
    *,
    hardware_id: str,
    contract: dict[str, Any],
) -> tuple[str, str, str, int, str]:
    payload = {
        "mesh_name": row.get("mesh_name"),
        "element_degree": row.get("element_degree"),
        "quadrature_rule_id": row.get("quadrature_rule"),
        "state_label": row.get("state_label"),
        "state_amplitude": row.get("state_amplitude"),
        "route": row.get("route"),
        "mpi_ranks": row.get("total_ranks"),
    }
    return _record_slot(payload, hardware_id=hardware_id, contract=contract)


def _resolve_record_artifact(
    output_path: Path, raw: object, field: str
) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"{field} is missing")
    declared = Path(text)
    if declared.is_absolute():
        raise ValueError(f"{field} must be relative to the output record")
    record_dir = output_path.resolve().parent
    resolved = (record_dir / declared).resolve()
    try:
        resolved.relative_to(record_dir)
    except ValueError as exc:
        raise ValueError(f"{field} escapes the output record directory") from exc
    if not resolved.is_file():
        raise ValueError(f"{field} does not identify an archived file")
    return resolved


def _locate_action(path: Path, payload: dict[str, Any]) -> Path:
    return _resolve_record_artifact(path, payload.get("action_out"), "action_out")


def _load_record_arrays(
    path: Path, payload: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Path]:
    action_path = _locate_action(path, payload)
    with np.load(action_path, allow_pickle=False) as archive:
        if "state" not in archive or "tangent_action" not in archive:
            raise ValueError(f"{action_path} lacks state/tangent_action arrays")
        state = np.asarray(archive["state"], dtype=np.float64)
        action = np.asarray(archive["tangent_action"], dtype=np.float64)
        actions = np.asarray(
            archive["tangent_actions"] if "tangent_actions" in archive else action[None, :],
            dtype=np.float64,
        )
        gradient = (
            np.asarray(archive["gradient"], dtype=np.float64)
            if "gradient" in archive
            else None
        )
        if "route" in archive:
            stored_route = str(np.asarray(archive["route"]).item())
            if stored_route != str(payload["route"]):
                raise ValueError(
                    f"NPZ route {stored_route!r} disagrees with JSON route "
                    f"{payload['route']!r}"
                )
        if "state_label" in archive:
            stored_label = str(np.asarray(archive["state_label"]).item())
            if stored_label != str(payload["state_label"]):
                raise ValueError("NPZ and JSON state labels disagree")
    if (
        state.ndim != 1
        or action.ndim != 1
        or actions.ndim != 2
        or state.size == 0
        or action.size == 0
    ):
        raise ValueError("saved state and tangent action must be nonempty vectors")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(actions)):
        raise ValueError("saved state/action contains nonfinite values")
    if gradient is not None and (gradient.ndim != 1 or not np.all(np.isfinite(gradient))):
        raise ValueError("saved gradient/residual is invalid")
    if _sha256_array(state) != str(payload.get("state_sha256", "")):
        raise ValueError("saved state SHA-256 disagrees with JSON")
    if _sha256_array(action) != str(payload.get("action_sha256", "")):
        raise ValueError("saved action SHA-256 disagrees with JSON")
    expected_hashes = payload.get("action_sha256_by_probe")
    if expected_hashes is not None:
        actual_hashes = [_sha256_array(row) for row in actions]
        if list(expected_hashes) != actual_hashes:
            raise ValueError("saved multi-probe action hashes disagree with JSON")
    if gradient is not None and _sha256_array(gradient) != str(
        payload.get("gradient_sha256", "")
    ):
        raise ValueError("saved gradient SHA-256 disagrees with JSON")
    return state, actions, gradient, action_path


def _validate_record(
    path: Path,
    payload: dict[str, Any],
    *,
    contract: dict[str, Any],
) -> dict[str, Any]:
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("fixed-state route record must use schema version 1")
    if payload.get("experiment_id") != contract["experiment_id"]:
        raise ValueError("wrong experiment_id")
    if payload.get("tier") not in {"fixed_state_screen", "factorized_quadrature"}:
        raise ValueError("wrong experiment tier")
    if payload.get("status") != "completed":
        raise ValueError(f"record status is {payload.get('status')!r}, not completed")
    gates = contract["equivalence_gates"]
    timings = np.asarray(payload.get("wall_times_s", []), dtype=np.float64)
    if timings.ndim != 1 or timings.size < int(gates["minimum_warm_repetitions"]):
        raise ValueError("record has too few measured warm repetitions")
    if not np.all(np.isfinite(timings)) or np.any(timings <= 0.0):
        raise ValueError("warm timings must all be finite and strictly positive")
    if payload.get("wall_time_reduction") != "mpi_collective_max":
        raise ValueError("record timing is not an MPI collective maximum")
    raw_rank_timings = np.asarray(payload.get("wall_times_by_rank_s", []), dtype=np.float64)
    ranks = int(payload.get("mpi_ranks", 0))
    if raw_rank_timings.shape != (timings.size, ranks):
        raise ValueError("record lacks one raw timing per repetition and MPI rank")
    if not np.all(np.isfinite(raw_rank_timings)) or np.any(raw_rank_timings <= 0.0):
        raise ValueError("raw rank timing samples are invalid")
    if not np.allclose(
        timings,
        np.max(raw_rank_timings, axis=1),
        rtol=1.0e-12,
        atol=1.0e-15,
    ):
        raise ValueError("collective-max timings disagree with raw rank samples")
    reported_median = _safe_float(payload.get("wall_time_median_s"))
    median = float(np.median(timings))
    if reported_median is None or not math.isclose(
        median, reported_median, rel_tol=1.0e-12, abs_tol=1.0e-15
    ):
        raise ValueError("reported timing median does not match raw repetitions")
    branch = dict(payload.get("branch_diagnostics") or {})
    margin = _safe_float(branch.get("normalized_boundary_margin_min"))
    near_fraction = _safe_float(branch.get("near_boundary_fraction"))
    if margin is None or margin < float(gates["minimum_normalized_branch_margin"]):
        raise ValueError("branch-boundary margin fails the frozen gate")
    if near_fraction is None or near_fraction > float(
        gates["maximum_near_boundary_fraction"]
    ):
        raise ValueError("near-boundary fraction fails the frozen gate")
    counts = branch.get("counts")
    if not isinstance(counts, dict) or not counts:
        raise ValueError("branch counts are missing")
    state, actions, gradient, action_path = _load_record_arrays(path, payload)
    return {
        "path": str(path),
        "payload": payload,
        "state": state,
        "action": actions[0],
        "actions": actions,
        "gradient": gradient,
        "action_path": str(action_path),
        "timings": timings,
        "median": median,
        "record_evidence": [
            _evidence_entry(path.resolve(), "fixed_route_output"),
            _evidence_entry(action_path.resolve(), "fixed_route_action"),
        ],
    }


def _source_provenance_gate(
    hardware_id: str, root: Path, contract: dict[str, Any]
) -> dict[str, object]:
    gates = contract["publication_model_input_gates"]
    if hardware_id == "workstation_local":
        manifest_path = root / "workstation_manifest.json"
        if not manifest_path.is_file():
            return {
                "eligible": False,
                "reason": "workstation_manifest_missing",
                "manifest_path": str(manifest_path),
            }
        try:
            manifest = _read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {
                "eligible": False,
                "reason": f"workstation_manifest_invalid: {exc}",
                "manifest_path": str(manifest_path),
            }
        if (
            manifest.get("schema_id")
            != "fenics-nonlinear-energies.exp-route-001-workstation-manifest"
            or int(manifest.get("schema_version", -1)) != 1
            or manifest.get("status") != "completed"
            or manifest.get("hardware_id") != "workstation_local"
            or int(manifest.get("case_count", 0)) != 12
            or int(manifest.get("route_process_executions", 0)) != 36
            or manifest.get("matrix_sha256") != gates["karolina_matrix_sha256"]
            or manifest.get("plan_sha256") != gates["workstation_plan_sha256"]
        ):
            return {
                "eligible": False,
                "reason": "workstation_manifest_scope_or_hash_mismatch",
                "manifest_path": str(manifest_path),
            }
        plan_path = root / str(manifest.get("plan_path", ""))
        environment_path = root / str(manifest.get("environment_path", ""))
        if (
            not plan_path.is_file()
            or _sha256_file(plan_path) != manifest.get("plan_sha256")
            or not environment_path.is_file()
            or _sha256_file(environment_path) != manifest.get("environment_sha256")
        ):
            return {
                "eligible": False,
                "reason": "workstation_manifest_archived_evidence_mismatch",
                "manifest_path": str(manifest_path),
            }
        try:
            environment = _read_json(environment_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {
                "eligible": False,
                "reason": f"workstation_environment_invalid: {exc}",
                "manifest_path": str(manifest_path),
            }
        if environment.get("controlled_environment") != {
            "status": "passed",
            "values": WORKSTATION_ENVIRONMENT_CONTRACT,
        }:
            return {
                "eligible": False,
                "reason": "workstation_environment_contract_mismatch",
                "manifest_path": str(manifest_path),
            }
        source_commit = str(manifest.get("source_commit", ""))
        if (
            manifest.get("source_dirty") is not False
            or len(source_commit) != 40
            or any(char not in "0123456789abcdef" for char in source_commit.lower())
        ):
            return {
                "eligible": False,
                "reason": "workstation_manifest_lacks_clean_source_commit",
                "manifest_path": str(manifest_path),
            }
        return {
            "eligible": True,
            "reason": "completed_frozen_workstation_campaign",
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "source_commit": source_commit,
            "hardware_id": "workstation_local",
            "run_id": str(manifest.get("run_id", "")),
        }
    if hardware_id != "karolina_cpu":
        return {"eligible": False, "reason": "unknown_hardware_provenance_policy"}
    manifest_path = root / "route_campaign_master_manifest.json"
    if not manifest_path.is_file():
        return {
            "eligible": False,
            "reason": "karolina_route_campaign_master_manifest_missing",
            "manifest_path": str(manifest_path),
        }
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "eligible": False,
            "reason": f"karolina_campaign_master_manifest_invalid: {exc}",
            "manifest_path": str(manifest_path),
        }
    if manifest.get("matrix_sha256") != gates["karolina_matrix_sha256"]:
        return {
            "eligible": False,
            "reason": "karolina_matrix_hash_mismatch",
            "manifest_path": str(manifest_path),
        }
    if (
        manifest.get("schema_id")
        != "fenics-nonlinear-energies.exp-route-001-campaign-master"
        or int(manifest.get("schema_version", -1)) != 1
        or manifest.get("status") != "submitted_tranches_complete"
    ):
        return {
            "eligible": False,
            "reason": "karolina_campaign_master_is_not_complete",
            "manifest_path": str(manifest_path),
        }
    if manifest.get("experiment_id") != contract["experiment_id"]:
        return {
            "eligible": False,
            "reason": "karolina_campaign_master_has_wrong_experiment",
            "manifest_path": str(manifest_path),
        }
    required_tiers = {
        "fixed_state_screen",
        "factorized_quadrature",
        "factorized_microbenchmark",
    }
    if set(manifest.get("selected_tiers") or []) != required_tiers:
        return {
            "eligible": False,
            "reason": "karolina_manifest_lacks_required_route_tiers",
            "manifest_path": str(manifest_path),
        }
    if int(manifest.get("case_count", 0)) != 105:
        return {
            "eligible": False,
            "reason": "karolina_manifest_cannot_cover_required_route_rows",
            "manifest_path": str(manifest_path),
        }
    tranches = manifest.get("tranches")
    if not isinstance(tranches, list) or not tranches:
        return {
            "eligible": False,
            "reason": "karolina_campaign_master_lacks_tranche_indices",
            "manifest_path": str(manifest_path),
        }
    tranche_manifest_paths: list[Path] = []
    for entry in tranches:
        if not isinstance(entry, dict):
            return {
                "eligible": False,
                "reason": "karolina_campaign_master_has_invalid_tranche_entry",
                "manifest_path": str(manifest_path),
            }
        for path_key, hash_key in (
            ("manifest_path", "manifest_sha256"),
            ("submitted_jobs_path", "submitted_jobs_sha256"),
            ("release_authorization_path", "release_authorization_sha256"),
        ):
            evidence_path = Path(str(entry.get(path_key, "")))
            if evidence_path.is_absolute():
                return {
                    "eligible": False,
                    "reason": "karolina_campaign_master_path_not_relocatable",
                    "manifest_path": str(manifest_path),
                }
            evidence_path = (manifest_path.parent / evidence_path).resolve()
            try:
                evidence_path.relative_to(manifest_path.parent.resolve())
            except ValueError:
                return {
                    "eligible": False,
                    "reason": "karolina_campaign_master_path_escape",
                    "manifest_path": str(manifest_path),
                }
            if not evidence_path.is_file() or entry.get(hash_key) != _sha256_file(
                evidence_path
            ):
                return {
                    "eligible": False,
                    "reason": "karolina_campaign_master_tranche_hash_mismatch",
                    "manifest_path": str(manifest_path),
                }
            if path_key == "manifest_path":
                tranche_manifest_paths.append(evidence_path)
    try:
        semantic_master = aggregate_route_tranche_manifests.aggregate(
            tranche_manifest_paths,
            archive_root=manifest_path.parent,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "eligible": False,
            "reason": f"karolina_campaign_master_semantic_validation_failed: {exc}",
            "manifest_path": str(manifest_path),
        }
    semantic_keys = (
        "schema_id",
        "schema_version",
        "status",
        "experiment_id",
        "matrix_sha256",
        "source_commit",
        "source_dirty",
        "selected_tiers",
        "case_count",
        "case_ids",
        "contract_path",
        "contract_sha256",
        "tranches",
    )
    if any(manifest.get(key) != semantic_master.get(key) for key in semantic_keys):
        return {
            "eligible": False,
            "reason": "karolina_campaign_master_semantic_content_mismatch",
            "manifest_path": str(manifest_path),
        }
    source_commit = str(manifest.get("source_commit", ""))
    if (
        manifest.get("source_dirty") is not False
        or len(source_commit) != 40
        or any(char not in "0123456789abcdef" for char in source_commit.lower())
    ):
        return {
            "eligible": False,
            "reason": "karolina_campaign_master_lacks_clean_source_commit",
            "manifest_path": str(manifest_path),
        }
    return {
        "eligible": True,
        "reason": "reviewed_submitted_tranche_master_manifest",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "source_commit": source_commit,
    }


def _stopping_semantic_binding_matches(
    observed: object, expected: dict[str, Any]
) -> bool:
    """Compare a STOP binding while allowing its archive path to relocate."""
    expected_keys = set(expected)
    semantic_keys = expected_keys - {"path"}
    return bool(
        isinstance(observed, dict)
        and set(observed) == expected_keys
        and isinstance(observed.get("path"), str)
        and bool(str(observed.get("path", "")).strip())
        and {key: observed.get(key) for key in semantic_keys}
        == {key: expected[key] for key in semantic_keys}
    )


def _endpoint_analysis_gate(
    raw_path: Path | None,
    stopping_adjudication_path: Path | None,
    *,
    sources: list[tuple[str, Path]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    if raw_path is None:
        return {
            "publication_admissible": False,
            "reason": "tier_b_endpoint_analysis_missing",
        }
    if stopping_adjudication_path is None:
        return {
            "publication_admissible": False,
            "reason": "tier_b_stopping_adjudication_missing",
        }
    karolina_roots = [root.resolve() for hardware, root in sources if hardware == "karolina_cpu"]
    if len(karolina_roots) != 1:
        return {
            "publication_admissible": False,
            "reason": "tier_b_endpoint_analysis_requires_one_karolina_source",
        }
    source_root = karolina_roots[0]
    path = raw_path.resolve()
    stopping_path = stopping_adjudication_path.resolve()
    try:
        relative_path = path.relative_to(source_root)
    except ValueError:
        return {
            "publication_admissible": False,
            "reason": "tier_b_endpoint_analysis_escapes_karolina_source",
        }
    try:
        stopping_relative_path = stopping_path.relative_to(source_root)
    except ValueError:
        return {
            "publication_admissible": False,
            "reason": "tier_b_stopping_adjudication_escapes_karolina_source",
        }
    try:
        payload = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "publication_admissible": False,
            "reason": f"tier_b_endpoint_analysis_invalid: {exc}",
        }
    try:
        validated_stopping = validate_stop_adjudication(stopping_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "publication_admissible": False,
            "reason": f"tier_b_stopping_adjudication_invalid: {exc}",
            "path": str(relative_path),
            "sha256": _sha256_file(path),
        }
    gates = contract["publication_model_input_gates"]
    expected_policy = {
        "path": str(TIER_B_STOPPING_POLICY_PATH.relative_to(REPO_ROOT)),
        "sha256": stopping_sha256_file(TIER_B_STOPPING_POLICY_PATH),
    }
    if (
        gates.get("tier_b_stopping_policy_path") != expected_policy["path"]
        or gates.get("tier_b_stopping_policy_sha256") != expected_policy["sha256"]
        or gates.get("tier_b_stopping_adjudication_required_before_submission")
        is not True
        or gates.get("tier_b_stopping_adjudication_schema_version")
        != validated_stopping["schema_version"]
        or gates.get("tier_b_stopping_p4_reference_row_id")
        != validated_stopping["p4_reference_row_id"]
        or gates.get("tier_b_stopping_p4_reference_status")
        != validated_stopping["p4_reference_status"]
        or payload.get("stopping_policy") != expected_policy
        or payload.get("stopping_binding_matches_manifest") is not True
    ):
        return {
            "publication_admissible": False,
            "reason": "tier_b_endpoint_stopping_policy_or_contract_binding_failed",
            "path": str(relative_path),
            "sha256": _sha256_file(path),
        }
    if not _stopping_semantic_binding_matches(
        payload.get("stopping_adjudication"), validated_stopping
    ):
        return {
            "publication_admissible": False,
            "reason": "tier_b_endpoint_nested_stopping_binding_failed",
            "path": str(relative_path),
            "sha256": _sha256_file(path),
        }
    expected_rows = int(gates["endpoint_required_rows"])
    schema = dict(payload.get("schema") or {})
    blocks = payload.get("blocks")
    censors = payload.get("structural_censors")
    allowed_terminals = {
        str(value) for value in gates["endpoint_allowed_terminal_decisions"]
    }
    comparative = payload.get("comparative_ranking_admissible")
    expected_terminal = (
        "tier_b_comparative_ranking_admissible"
        if comparative is True
        else "tier_b_descriptive_timing_only"
    )
    expected_censor_reason = str(contract["structural_censors"][0]["reason"])
    semantic_passed = bool(
        schema.get("id") == "fenics-nonlinear-energies.exp-route-001.tier-b-endpoints"
        and int(schema.get("version", -1))
        == int(gates["endpoint_analysis_schema_version"])
        and payload.get("experiment_id") == contract["experiment_id"]
        and payload.get("matrix_sha256") == gates["karolina_matrix_sha256"]
        and payload.get("analysis_contract_sha256") == _sha256_file(DEFAULT_CONTRACT)
        and payload.get("terminal_decision") in allowed_terminals
        and isinstance(comparative, bool)
        and payload.get("terminal_decision") == expected_terminal
        and payload.get("endpoint_correct_timing_admissible") is True
        and payload.get("descriptive_timing_available") is True
        and payload.get("publication_admissible") is True
        and int(payload.get("required_rows", -1)) == expected_rows
        and int(payload.get("admitted_rows", -1)) == expected_rows
        and isinstance(blocks, list)
        and len(blocks) == expected_rows
        and all(
            isinstance(block, dict)
            and block.get("status") == "timing_admitted"
            and all(
                dict(block.get("routes") or {}).get(route, {}).get("status")
                == "timing_admitted"
                for route in ("element_ad", "constitutive_ad")
            )
            for block in blocks or []
        )
        and isinstance(censors, list)
        and len(censors) == 2
        and all(
            isinstance(row, dict)
            and row.get("route") == "colored_sfd"
            and row.get("status") == "censored"
            and row.get("reason") == expected_censor_reason
            and row.get("timing_exposed") is False
            for row in censors or []
        )
    )
    source_gate = _source_provenance_gate("karolina_cpu", source_root, contract)
    endpoint_manifest = dict(payload.get("manifest") or {})
    if (
        not semantic_passed
        or source_gate.get("eligible") is not True
        or endpoint_manifest.get("source_commit") != source_gate.get("source_commit")
        or not _stopping_semantic_binding_matches(
            endpoint_manifest.get("stopping_adjudication"), validated_stopping
        )
    ):
        return {
            "publication_admissible": False,
            "reason": "tier_b_endpoint_analysis_semantic_or_source_gate_failed",
            "path": str(relative_path),
            "sha256": _sha256_file(path),
        }
    return {
        "publication_admissible": True,
        "reason": "hash_bound_tier_b_endpoint_analysis_admitted",
        "path": str(relative_path),
        "sha256": _sha256_file(path),
        "schema_version": int(gates["endpoint_analysis_schema_version"]),
        "terminal_decision": str(payload["terminal_decision"]),
        "required_rows": expected_rows,
        "admitted_rows": expected_rows,
        "comparative_ranking_admissible": comparative,
        "stopping_policy": expected_policy,
        "stopping_adjudication": {
            **validated_stopping,
            "path": str(stopping_relative_path),
        },
        "stopping_binding_matches_manifest": True,
        "_source_path": str(path),
        "_stopping_adjudication_source_path": str(stopping_path),
    }


def _archive_endpoint_gate(
    endpoint_gate: dict[str, Any], output_dir: Path
) -> tuple[dict[str, Any], Path | None, Path | None]:
    """Copy admitted endpoint and STOP evidence into one analysis archive."""
    private_keys = {"_source_path", "_stopping_adjudication_source_path"}
    public = {key: value for key, value in endpoint_gate.items() if key not in private_keys}
    if endpoint_gate.get("publication_admissible") is not True:
        return public, None, None
    output_dir = output_dir.resolve()
    archived_endpoint = output_dir / "endpoint_analysis.json"
    archived_stopping = output_dir / "stopping_adjudication.json"
    source_endpoint = Path(str(endpoint_gate["_source_path"])).resolve()
    source_stopping = Path(
        str(endpoint_gate["_stopping_adjudication_source_path"])
    ).resolve()
    if source_endpoint != archived_endpoint:
        shutil.copy2(source_endpoint, archived_endpoint)
    if source_stopping != archived_stopping:
        shutil.copy2(source_stopping, archived_stopping)
    if _sha256_file(archived_endpoint) != endpoint_gate["sha256"]:
        raise ValueError("Tier-B endpoint analysis changed during archival")
    stopping_binding = dict(endpoint_gate["stopping_adjudication"])
    if _sha256_file(archived_stopping) != stopping_binding["sha256"]:
        raise ValueError("Tier-B STOP adjudication changed during archival")
    public["source_archive_path"] = public["path"]
    public["path"] = archived_endpoint.name
    stopping_binding["source_archive_path"] = stopping_binding["path"]
    stopping_binding["path"] = archived_stopping.name
    public["stopping_adjudication"] = stopping_binding
    return public, archived_endpoint, archived_stopping


def _scan_source(
    hardware_id: str,
    root: Path,
    *,
    contract: dict[str, Any],
    source_provenance: dict[str, object] | None = None,
) -> tuple[
    dict[tuple[str, str, str, int, str], dict[str, Any]],
    dict[tuple[str, str, str, int, str], str],
    list[dict[str, str]],
]:
    observed: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    candidates: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = {}
    censors: dict[tuple[str, str, str, int, str], str] = {}
    invalid: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("experiment_id") != contract["experiment_id"]:
            continue
        if payload.get("tier") not in {"fixed_state_screen", "factorized_quadrature"}:
            continue
        try:
            slot = _record_slot(
                payload, hardware_id=hardware_id, contract=contract
            )
            record = _validate_record(path, payload, contract=contract)
            record["source_provenance"] = dict(source_provenance or {})
            if hardware_id == "karolina_cpu":
                record["matrix_row"] = _bind_fixed_record_to_reviewed_matrix(
                    path,
                    payload,
                    contract=contract,
                )
                record["evidence_files"] = _cluster_fixed_evidence_files(
                    root,
                    path,
                    payload,
                )
            elif hardware_id == "workstation_local" and dict(
                source_provenance or {}
            ).get("eligible") is True:
                record["matrix_row"] = _bind_fixed_record_to_reviewed_matrix(
                    path,
                    payload,
                    contract=contract,
                )
                record["evidence_files"] = _workstation_fixed_evidence_files(
                    root,
                    path,
                    payload,
                )
            design = dict(payload.get("comparison_design") or {})
            if design.get("route_order_policy") == "seeded_balanced_cyclic_v1":
                block_path = path.parent.parent / "block_result.json"
                block_result = _read_json(block_path)
                if block_result.get("status") != "admitted_correctness_block":
                    raise ValueError("paired block_result did not pass correctness admission")
                if block_result.get("comparison_id") != design.get("comparison_id"):
                    raise ValueError("paired block_result comparison identity mismatch")
                record["block_result_path"] = str(block_path)
            candidates.setdefault(slot, []).append(record)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            invalid.append({"path": str(path), "reason": str(exc)})

    for slot, records in candidates.items():
        try:
            observed[slot] = _aggregate_block_records(records, contract=contract)
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append(
                {
                    "path": ";".join(record["path"] for record in records),
                    "reason": str(exc),
                }
            )

    # A started matrix row with a nonzero runner result is a censored outcome,
    # not an absent observation.  This preserves failed routes without
    # fabricating a timing.
    for matrix_path in sorted(root.rglob("matrix_row.json")):
        try:
            row = _read_json(matrix_path)
            if hardware_id == "karolina_cpu" or (
                hardware_id == "workstation_local"
                and dict(source_provenance or {}).get("eligible") is True
            ):
                case_id = matrix_path.parent.parent.name
                planned = _reviewed_route_rows(contract).get(case_id)
                if planned is None or not _matrix_rows_equal(planned, row):
                    raise ValueError(
                        "started job matrix_row.json differs from the reviewed matrix"
                    )
            if row.get("experiment_id") != contract["experiment_id"]:
                continue
            if row.get("tier") not in {"fixed_state_screen", "factorized_quadrature"}:
                continue
            if row.get("runner") == "p3d_fixed_state_block":
                matrix_routes = [
                    value.strip()
                    for value in str(row.get("route_order", "")).split("|")
                    if value.strip()
                ]
            else:
                matrix_routes = [str(row.get("route", ""))]
            slots = []
            for route in matrix_routes:
                route_row = dict(row)
                route_row["route"] = route
                slots.append(
                    _matrix_slot(
                        route_row, hardware_id=hardware_id, contract=contract
                    )
                )
            if all(slot in observed for slot in slots):
                continue
            run_records_path = matrix_path.parent / "run_records.json"
            if not run_records_path.is_file():
                continue
            with run_records_path.open(encoding="utf-8") as handle:
                run_records = json.load(handle, parse_constant=_reject_nonfinite)
            if not isinstance(run_records, list) or not run_records:
                continue
            route_records: dict[str, list[dict[str, Any]]] = {}
            block_records: list[dict[str, Any]] = []
            for raw_record in run_records:
                record = dict(raw_record)
                route_name = str(record.get("route", ""))
                if route_name in matrix_routes:
                    route_records.setdefault(route_name, []).append(record)
                elif route_name == "block_validation":
                    block_records.append(record)
            for slot, route in zip(slots, matrix_routes, strict=True):
                if slot in observed:
                    continue
                attempts = route_records.get(route, [])
                failures = [
                    record
                    for record in attempts
                    if int(record.get("returncode", 1)) != 0
                ]
                if failures:
                    failure = failures[-1]
                    returncode = int(failure.get("returncode", 1))
                    censors[slot] = (
                        "runner_timeout"
                        if bool(failure.get("timed_out", False))
                        else f"runner_nonzero_exit_{returncode}"
                    )
                elif block_records:
                    code = int(block_records[-1].get("returncode", 86))
                    censors[slot] = f"paired_block_validation_failed_{code}"
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            invalid.append({"path": str(matrix_path), "reason": str(exc)})
    return observed, censors, invalid


def _aggregate_block_records(
    records: list[dict[str, Any]], *, contract: dict[str, Any]
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate an empty route record list")
    if len(records) == 1:
        record = dict(records[0])
        record["paired_block_design_passed"] = False
        record["independent_block_count"] = 1
        record["block_medians"] = [float(record["median"])]
        design = dict(record["payload"].get("comparison_design") or {})
        record["block_repetitions"] = [int(design.get("block_repetition", 0))]
        record["block_route_positions"] = [
            int(design.get("route_order_position", -1))
        ]
        return record
    records = sorted(
        records,
        key=lambda record: int(
            dict(record["payload"].get("comparison_design") or {}).get(
                "block_repetition", -1
            )
        ),
    )
    designs = [dict(record["payload"].get("comparison_design") or {}) for record in records]
    if any(design.get("route_order_policy") != "seeded_balanced_cyclic_v1" for design in designs):
        raise ValueError("repeated route rows lack the balanced paired-block policy")
    if any(design.get("timing_reduction") != "mpi_collective_max" for design in designs):
        raise ValueError("repeated route rows lack collective-max timing")
    comparison_ids = {str(design.get("comparison_id", "")) for design in designs}
    repetitions = [int(design.get("block_repetition", -1)) for design in designs]
    if len(comparison_ids) != 1 or "" in comparison_ids:
        raise ValueError("paired route rows change comparison_id")
    if len(set(repetitions)) != len(repetitions) or min(repetitions) < 1:
        raise ValueError("paired route rows repeat or omit block identifiers")
    positions = [int(design.get("route_order_position", -1)) for design in designs]
    position_count = max(positions) + 1 if positions else 0
    position_histogram = [positions.count(index) for index in range(position_count)]
    if (
        min(positions) < 0
        or position_count not in {2, 3}
        or len(set(position_histogram)) != 1
    ):
        raise ValueError("route is not balanced across all route-order positions")
    reference = records[0]
    commits = {
        str(dict(record["payload"].get("git") or {}).get("commit", ""))
        for record in records
    }
    if len(commits) != 1:
        raise ValueError("independent comparison blocks use different source commits")
    for record in records[1:]:
        if not np.array_equal(reference["state"], record["state"]):
            raise ValueError("state changed across independent comparison blocks")
        if reference["actions"].shape != record["actions"].shape:
            raise ValueError("probe count changed across independent blocks")
        for index in range(reference["actions"].shape[0]):
            relative, _maximum = _relative_action_error(
                reference["actions"][index], record["actions"][index]
            )
            if relative > float(contract["equivalence_gates"]["action_relative_l2_max"]):
                raise ValueError("same-route tangent action changed across blocks")
        if reference["gradient"] is None or record["gradient"] is None:
            raise ValueError("paired publication blocks require saved gradients")
        gradient_error, _maximum = _relative_action_error(
            reference["gradient"], record["gradient"]
        )
        if gradient_error > 1.0e-12:
            raise ValueError("same-route gradient changed across blocks")
    aggregated = dict(reference)
    block_medians = [float(record["median"]) for record in records]
    aggregated["median"] = float(np.median(np.asarray(block_medians)))
    aggregated["block_medians"] = block_medians
    aggregated["block_repetitions"] = repetitions
    aggregated["block_route_positions"] = positions
    aggregated["independent_block_count"] = len(records)
    aggregated["paired_block_design_passed"] = True
    aggregated["paths"] = [record["path"] for record in records]
    aggregated["source_provenance"] = dict(reference.get("source_provenance") or {})
    combined_evidence: list[dict[str, str]] = []
    for record in records:
        combined_evidence.extend(record.get("record_evidence") or [])
        combined_evidence.extend(record.get("evidence_files") or [])
    aggregated["record_evidence"] = list(
        {
            (entry["role"], entry["path"], entry["sha256"]): entry
            for entry in combined_evidence
        }.values()
    )
    return aggregated


def _publication_model_eligibility(record: dict[str, Any]) -> tuple[bool, str]:
    source = dict(record.get("source_provenance") or {})
    if source and source.get("eligible") is not True:
        return False, str(source.get("reason", "source_provenance_failed"))
    git = dict(record["payload"].get("git") or {})
    commit = str(git.get("commit", ""))
    if git.get("dirty") is not False:
        return False, "record_git_worktree_not_clean"
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit.lower()):
        return False, "record_git_commit_missing_or_invalid"
    manifest_commit = str(source.get("source_commit", ""))
    if manifest_commit and commit != manifest_commit:
        return False, "record_commit_differs_from_prepared_manifest"
    if manifest_commit:
        job_dir = Path(str(record.get("path", ""))).parents[2]
        expected_job_id = job_dir.name.removeprefix("job_")
        metadata = dict(record["payload"].get("job_metadata") or {})
        if source.get("hardware_id") == "workstation_local":
            actual_job_id = str(metadata.get("workstation_run_id", ""))
            if actual_job_id != str(source.get("run_id", "")):
                return False, "record_workstation_run_identity_differs_from_manifest"
        else:
            actual_job_id = str(metadata.get("slurm_job_id", ""))
        if not expected_job_id or actual_job_id != expected_job_id:
            return False, "record_job_identity_missing_or_stale"
    if record.get("paired_block_design_passed") is not True:
        return False, "paired_balanced_comparison_blocks_missing"
    if int(record.get("independent_block_count", 0)) < 3:
        return False, "insufficient_independent_comparison_blocks"
    if int(record["payload"].get("probe_count", 0)) < 4 or record.get("gradient") is None:
        return False, "multiple_probe_or_gradient_admission_missing"
    return True, "clean_committed_record"


def _split(hardware_id: str, ranks: int, contract: dict[str, Any]) -> str:
    if hardware_id == "workstation_local":
        return "training"
    if hardware_id == "karolina_cpu":
        spec = contract["hardware"][hardware_id]
        if ranks in [int(value) for value in spec["training_ranks"]]:
            return "training"
        if ranks in [int(value) for value in spec["holdout_ranks"]]:
            return "holdout"
    return "out_of_contract"


def _is_structural_censor(
    slot: tuple[str, str, str, int, str], contract: dict[str, Any]
) -> str | None:
    hardware_id, config_id, _state_id, _ranks, route = slot
    for rule in contract["structural_censors"]:
        if (
            hardware_id == rule["hardware_id"]
            and config_id == rule["configuration_id"]
            and route == rule["route"]
        ):
            return str(rule["reason"])
    return None


def _expected_slots(
    hardware_ids: Iterable[str], contract: dict[str, Any]
) -> set[tuple[str, str, str, int, str]]:
    slots: set[tuple[str, str, str, int, str]] = set()
    _by_signature, configurations = _configuration_maps(contract)
    for hardware_id in sorted(set(hardware_ids)):
        scope = contract["expected_scope"].get(hardware_id)
        if scope is None:
            continue
        config_ids = list(scope["configuration_ids"]) + list(
            scope.get("factor_configuration_ids", [])
        )
        for config_id in config_ids:
            configured_ranks = dict(configurations[config_id].get("hardware_ranks") or {}).get(
                hardware_id, scope["ranks"]
            )
            for state in contract["states"]:
                for ranks in configured_ranks:
                    for route in contract["route_order"]:
                        slots.add(
                            (
                                hardware_id,
                                str(config_id),
                                str(state["state_id"]),
                                int(ranks),
                                str(route),
                            )
                        )
    return slots


def _relative_action_error(reference: np.ndarray, action: np.ndarray) -> tuple[float, float]:
    if reference.shape != action.shape:
        return float("inf"), float("inf")
    difference = np.asarray(action - reference, dtype=np.float64)
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return float(np.linalg.norm(difference) / denominator), float(np.max(np.abs(difference)))


def _group_key(slot: tuple[str, str, str, int, str]) -> tuple[str, str, str, int]:
    return slot[:4]


def _structural_rank_summary(payload: dict[str, Any]) -> list[tuple[int, ...]]:
    rows = payload.get("rank_summaries")
    if not isinstance(rows, list) or not rows:
        raise ValueError("rank_summaries are missing")
    keys = (
        "rank",
        "owned_dofs",
        "local_elements",
        "owned_elements",
        "overlap_dofs",
        "owned_matrix_nonzeros",
    )
    normalized = [tuple(int(row[key]) for key in keys) for row in rows]
    normalized.sort(key=lambda row: row[0])
    if [row[0] for row in normalized] != list(range(len(normalized))):
        raise ValueError("rank_summaries do not contain one ordered row per rank")
    return normalized


def _route_work_proxy(
    route: str,
    rank_rows: list[dict[str, Any]],
    *,
    element_dofs: int,
    constitutive_dimension: int,
    quadrature_points: int,
) -> int:
    """Return the collective-max-aligned structural work count for one route.

    The response modeled below is an MPI collective maximum.  The proxy must
    therefore describe the busiest rank, including overlap elements, rather
    than the global number of uniquely owned elements.  These are structural
    operation-shape counts, not claims about exact floating-point operation
    counts or compiler scheduling.
    """

    local_elements = [int(row["local_elements"]) for row in rank_rows]
    if not local_elements or min(local_elements) <= 0:
        raise ValueError("rank_summaries contain a nonpositive local element count")
    if min(element_dofs, constitutive_dimension, quadrature_points) <= 0:
        raise ValueError("route-work dimensions must be positive")

    if route == "element_ad":
        # One dense m_e-by-m_e element Hessian contribution per quadrature
        # point and overlap element.
        return (
            max(local_elements)
            * quadrature_points
            * element_dofs**2
        )
    if route == "colored_sfd":
        # Every local color triggers one exact AD-HVP.  Each HVP propagates an
        # m_e-vector contribution through every overlap element and quadrature
        # point on that rank.
        color_counts = [int(row["local_color_count"]) for row in rank_rows]
        if min(color_counts) <= 0:
            raise ValueError("colored recovery has a nonpositive local color count")
        return (
            max(
                count * colors
                for count, colors in zip(local_elements, color_counts, strict=True)
            )
            * quadrature_points
            * element_dofs
        )
    if route == "constitutive_ad":
        # s^2 counts the dense constitutive tangent.  Forming C B and then
        # B^T(C B) contributes structural counts s^2 m_e and s m_e^2.
        constitutive_shape = (
            constitutive_dimension**2
            + constitutive_dimension**2 * element_dofs
            + constitutive_dimension * element_dofs**2
        )
        return max(local_elements) * quadrature_points * constitutive_shape
    raise ValueError(f"unsupported derivative route {route!r}")


def _model_covariates(record: dict[str, Any]) -> dict[str, Any] | None:
    payload = record["payload"]
    cov = dict(payload.get("model_covariates") or {})
    rank_rows = payload.get("rank_summaries")
    if not isinstance(rank_rows, list) or not rank_rows:
        return None
    try:
        element_dofs = int(cov["element_dofs"])
        constitutive_dimension = int(cov["constitutive_dimension"])
        quadrature_points = int(cov["quadrature_points_per_element"])
        owned_elements = int(cov["total_owned_elements"])
        rank_count = int(cov["rank_count"])
        global_free_dofs = int(cov.get("global_free_dofs", rank_count))
        maximum_nnz = max(int(row["owned_matrix_nonzeros"]) for row in rank_rows)
        maximum_overlap = max(int(row["overlap_dofs"]) for row in rank_rows)
        maximum_owned = max(
            int(row.get("owned_dofs", max(1, math.ceil(global_free_dofs / rank_count))))
            for row in rank_rows
        )
        peak_rss = max(int(row["peak_rss_bytes"]) for row in rank_rows)
        tracked = max(int(row["tracked_allocation_bytes"]) for row in rank_rows)
        owned_elements_by_rank = [int(row["owned_elements"]) for row in rank_rows]
        local_elements_by_rank = [int(row["local_elements"]) for row in rank_rows]
        overlap_by_rank = [int(row["overlap_dofs"]) for row in rank_rows]
        route = str(payload["route"])
        if route == "colored_sfd":
            colors = cov.get("maximum_local_color_count")
            if colors is None:
                colors = max(int(row["local_color_count"]) for row in rank_rows)
            colors = int(colors)
        else:
            colors = 0
        plastic_fraction = float(payload["branch_diagnostics"]["plastic_fraction"])
    except (KeyError, TypeError, ValueError):
        return None
    values = (
        element_dofs,
        constitutive_dimension,
        quadrature_points,
        owned_elements,
        rank_count,
        maximum_nnz,
        maximum_overlap,
        maximum_owned,
    )
    if min(values) <= 0 or not math.isfinite(plastic_fraction):
        return None
    if route == "colored_sfd" and colors <= 0:
        return None
    try:
        work = _route_work_proxy(
            route,
            rank_rows,
            element_dofs=element_dofs,
            constitutive_dimension=constitutive_dimension,
            quadrature_points=quadrature_points,
        )
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "route_work_proxy": float(work),
        "owned_matrix_nonzeros": float(maximum_nnz),
        "maximum_rank_overlap_dofs": float(maximum_overlap),
        "maximum_rank_owned_dofs": float(maximum_owned),
        "rank_count": float(rank_count),
        "plastic_fraction": float(plastic_fraction),
        "element_dofs": float(element_dofs),
        "quadrature_points_per_element": float(quadrature_points),
        "constitutive_dimension": float(constitutive_dimension),
        "maximum_local_color_count": float(colors),
        "total_owned_elements": float(owned_elements),
        "maximum_rank_local_elements": float(max(local_elements_by_rank)),
        "peak_rank_rss_bytes": float(peak_rss),
        "tracked_allocation_bytes": float(tracked),
        "owned_element_imbalance": float(
            max(owned_elements_by_rank) / max(np.mean(owned_elements_by_rank), 1.0)
        ),
        "overlap_dof_imbalance": float(
            max(overlap_by_rank) / max(np.mean(overlap_by_rank), 1.0)
        ),
        "route": route,
    }


def build_empirical_map(
    *,
    contract: dict[str, Any],
    hardware_ids: Iterable[str],
    observed: dict[tuple[str, str, str, int, str], dict[str, Any]],
    runtime_censors: dict[tuple[str, str, str, int, str], str],
) -> list[dict[str, Any]]:
    expected = _expected_slots(hardware_ids, contract)
    rows: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    for slot in sorted(expected | set(observed) | set(runtime_censors)):
        hardware_id, config_id, state_id, ranks, route = slot
        structural_reason = _is_structural_censor(slot, contract)
        status = "missing"
        reason = "no_record"
        if structural_reason is not None:
            status = "censored"
            reason = structural_reason
        elif slot in runtime_censors:
            status = "censored"
            reason = runtime_censors[slot]
        elif slot in observed:
            status = "pending_equivalence"
            reason = ""
        rows[slot] = {
            "hardware_id": hardware_id,
            "configuration_id": config_id,
            "state_id": state_id,
            "rank_count": ranks,
            "route": route,
            "split": _split(hardware_id, ranks, contract),
            "status": status,
            "reason": reason,
            "state_sha256": "",
            "action_sha256": "",
            "action_relative_l2_error": None,
            "action_relative_l2_errors": None,
            "action_max_absolute_error": None,
            "gradient_residual_relative_error": None,
            "admitted_wall_time_median_s": None,
            "paired_block_medians_s": None,
            "paired_block_repetitions": None,
            "paired_block_route_positions": None,
            "model_covariates": None,
            "publication_model_eligible": False,
            "model_exclusion_reason": reason,
            "record_path": "",
            "source_commit": "",
        }

    gates = contract["equivalence_gates"]
    group_keys = sorted({_group_key(slot) for slot in rows})
    for group in group_keys:
        group_slots = [slot for slot in rows if _group_key(slot) == group]
        active_slots = [
            slot for slot in group_slots if _is_structural_censor(slot, contract) is None
        ]
        active_records = [slot for slot in active_slots if slot in observed]
        group_failure = ""
        if len(active_records) != len(active_slots):
            group_failure = "incomplete_active_route_set"
        elif not active_records:
            group_failure = "no_active_route_records"
        else:
            reference_slot = next(
                (
                    slot
                    for slot in active_records
                    if slot[-1] == contract["reference_route"]
                ),
                None,
            )
            if reference_slot is None:
                group_failure = "reference_route_missing"
            else:
                reference = observed[reference_slot]
                reference_state = reference["state"]
                reference_action = reference["action"]
                reference_actions = reference["actions"]
                reference_gradient = reference["gradient"]
                reference_counts = reference["payload"]["branch_diagnostics"]["counts"]
                reference_rank_structure = _structural_rank_summary(
                    reference["payload"]
                )
                reference_cov = reference["payload"].get("model_covariates") or {}
                invariant_keys = (
                    "element_dofs",
                    "constitutive_dimension",
                    "quadrature_points_per_element",
                    "total_owned_elements",
                    "global_free_dofs",
                    "rank_count",
                )
                for slot in active_records:
                    record = observed[slot]
                    row = rows[slot]
                    payload = record["payload"]
                    row["record_path"] = record["path"]
                    row["state_sha256"] = str(payload["state_sha256"])
                    row["action_sha256"] = str(payload["action_sha256"])
                    row["source_commit"] = str(
                        dict(payload.get("git") or {}).get("commit", "")
                    )
                    relative, maximum = _relative_action_error(
                        reference_action, record["action"]
                    )
                    if reference_actions.shape != record["actions"].shape:
                        group_failure = "probe_count_mismatch"
                        action_errors = [float("inf")]
                    else:
                        action_errors = [
                            _relative_action_error(
                                reference_actions[index], record["actions"][index]
                            )[0]
                            for index in range(reference_actions.shape[0])
                        ]
                    row["action_relative_l2_error"] = relative
                    row["action_relative_l2_errors"] = action_errors
                    row["action_max_absolute_error"] = maximum
                    gradient_error = None
                    if reference_gradient is not None or record["gradient"] is not None:
                        if reference_gradient is None or record["gradient"] is None:
                            group_failure = "gradient_residual_missing"
                        else:
                            gradient_error = _relative_action_error(
                                reference_gradient, record["gradient"]
                            )[0]
                            if gradient_error > 1.0e-12:
                                group_failure = "gradient_residual_mismatch"
                    row["gradient_residual_relative_error"] = gradient_error
                    if not np.array_equal(reference_state, record["state"]):
                        group_failure = "state_array_mismatch"
                    elif payload["state_sha256"] != reference["payload"]["state_sha256"]:
                        group_failure = "state_sha256_mismatch"
                    elif max(action_errors) > float(gates["action_relative_l2_max"]):
                        group_failure = "tangent_action_mismatch"
                    elif payload["branch_diagnostics"]["counts"] != reference_counts:
                        group_failure = "branch_count_mismatch"
                    elif _structural_rank_summary(payload) != reference_rank_structure:
                        group_failure = "route_invariant_rank_summary_mismatch"
                    cov = payload.get("model_covariates") or {}
                    if reference_cov and cov and any(
                        cov.get(key) != reference_cov.get(key) for key in invariant_keys
                    ):
                        group_failure = "route_invariant_covariate_mismatch"

        for slot in active_records:
            row = rows[slot]
            record = observed[slot]
            if group_failure:
                row["status"] = "equivalence_failed"
                row["reason"] = group_failure
                continue
            row["status"] = "admitted"
            row["reason"] = "all_equivalence_gates_passed"
            row["admitted_wall_time_median_s"] = float(record["median"])
            row["paired_block_medians_s"] = [
                float(value) for value in record.get("block_medians", [])
            ]
            row["paired_block_repetitions"] = [
                int(value) for value in record.get("block_repetitions", [])
            ]
            row["paired_block_route_positions"] = [
                int(value) for value in record.get("block_route_positions", [])
            ]
            row["model_covariates"] = _model_covariates(record)
            eligible, exclusion = _publication_model_eligibility(record)
            if row["model_covariates"] is None:
                eligible = False
                exclusion = "required_model_covariates_missing"
            row["publication_model_eligible"] = bool(eligible)
            row["model_exclusion_reason"] = "" if eligible else exclusion
        if group_failure:
            for slot in active_slots:
                if rows[slot]["status"] == "missing":
                    rows[slot]["reason"] = group_failure
    return [rows[slot] for slot in sorted(rows)]


def _feature_vector(
    row: dict[str, Any],
    feature_order: list[str],
    *,
    factorized_gate: dict[str, Any],
) -> np.ndarray:
    cov = row["model_covariates"]
    route = str(row["route"])
    values = {
        "route_is_element_ad": 1.0 if route == "element_ad" else 0.0,
        "route_is_colored_sfd": 1.0 if route == "colored_sfd" else 0.0,
        "route_is_constitutive_ad": 1.0 if route == "constitutive_ad" else 0.0,
        "karolina_route_is_element_ad": (
            1.0
            if row["hardware_id"] == "karolina_cpu" and route == "element_ad"
            else 0.0
        ),
        "karolina_route_is_colored_sfd": (
            1.0
            if row["hardware_id"] == "karolina_cpu" and route == "colored_sfd"
            else 0.0
        ),
        "karolina_route_is_constitutive_ad": (
            1.0
            if row["hardware_id"] == "karolina_cpu" and route == "constitutive_ad"
            else 0.0
        ),
        "log1p_route_work_proxy": math.log1p(float(cov["route_work_proxy"])),
        "log1p_owned_matrix_nonzeros": math.log1p(
            float(cov["owned_matrix_nonzeros"])
        ),
        "log1p_maximum_rank_overlap_dofs": math.log1p(
            float(cov["maximum_rank_overlap_dofs"])
        ),
        "log_rank_count": math.log(float(cov["rank_count"])),
        "plastic_fraction": float(cov["plastic_fraction"]),
        "owned_element_imbalance": float(cov["owned_element_imbalance"]),
        "overlap_dof_imbalance": float(cov["overlap_dof_imbalance"]),
    }
    return np.asarray([values[name] for name in feature_order], dtype=np.float64)


def _model_preflight(
    training: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    contract: dict[str, Any],
    factorized_gate: dict[str, Any] | None = None,
    endpoint_gate: dict[str, Any] | None = None,
) -> list[str]:
    model = contract["cost_model"]
    reasons: list[str] = []
    if contract["publication_model_input_gates"].get(
        "design_released_for_fitting"
    ) is not True:
        reasons.append("cost_model_design_not_released")
    if (
        contract["publication_model_input_gates"].get(
            "endpoint_analysis_required_for_selector"
        )
        is True
        and (not endpoint_gate or endpoint_gate.get("publication_admissible") is not True)
    ):
        reasons.append("tier_b_endpoint_analysis_gate_not_passed")
    calibration_policy = dict(contract.get("factorized_calibration_policy") or {})
    if (
        calibration_policy.get("required_for_selector_claim") is True
        and (not factorized_gate or factorized_gate.get("calibration_integrated") is not True)
    ):
        reasons.append(
            str(
                calibration_policy.get(
                    "fail_closed_reason",
                    "factorized_microbenchmark_calibration_not_integrated",
                )
            )
        )
    if len(training) < int(model["minimum_training_rows"]):
        reasons.append("insufficient_training_rows")
    if len(holdout) < int(model["minimum_holdout_rows"]):
        reasons.append("insufficient_holdout_rows")
    for route in contract["route_order"]:
        if sum(row["route"] == route for row in training) < int(
            model["minimum_training_rows_per_route"]
        ):
            reasons.append(f"insufficient_training_rows_{route}")
        if sum(row["route"] == route for row in holdout) < int(
            model["minimum_holdout_rows_per_route"]
        ):
            reasons.append(f"insufficient_holdout_rows_{route}")
    train_groups = {
        (
            row["hardware_id"],
            row["configuration_id"],
            row["state_id"],
            row["rank_count"],
        )
        for row in training
    }
    holdout_groups = {
        (
            row["hardware_id"],
            row["configuration_id"],
            row["state_id"],
            row["rank_count"],
        )
        for row in holdout
    }
    if len(train_groups) < int(model["minimum_training_groups"]):
        reasons.append("insufficient_training_groups")
    if len(holdout_groups) < int(model["minimum_holdout_groups"]):
        reasons.append("insufficient_holdout_groups")
    training_hardware = {row["hardware_id"] for row in training}
    for hardware_id in model["required_training_hardware"]:
        if hardware_id not in training_hardware:
            reasons.append(f"missing_training_hardware_{hardware_id}")
    model_commits = {str(row.get("source_commit", "")) for row in training + holdout}
    if len(model_commits) != 1 or "" in model_commits:
        reasons.append("model_rows_do_not_share_one_source_commit")
    for rows, label in ((training, "training"), (holdout, "holdout")):
        grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
        for row in rows:
            key = (
                str(row["hardware_id"]),
                str(row["configuration_id"]),
                str(row["state_id"]),
                int(row["rank_count"]),
            )
            grouped.setdefault(key, []).append(row)
        for entries in grouped.values():
            repetitions = [
                tuple(int(value) for value in row.get("paired_block_repetitions") or [])
                for row in entries
            ]
            medians = [
                tuple(float(value) for value in row.get("paired_block_medians_s") or [])
                for row in entries
            ]
            if (
                not repetitions
                or len(set(repetitions)) != 1
                or len(repetitions[0]) < 3
                or len(set(repetitions[0])) != len(repetitions[0])
                or any(len(values) != len(repetitions[0]) for values in medians)
                or any(
                    not math.isfinite(value) or value <= 0.0
                    for values in medians
                    for value in values
                )
            ):
                reasons.append(f"{label}_paired_block_evidence_missing_or_unaligned")
                break
    return sorted(set(reasons))


def _paired_block_bootstrap_values(
    rows: list[dict[str, Any]],
    *,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap route medians with one shared block draw per comparison group."""

    values = np.empty((resamples, len(rows)), dtype=np.float64)
    grouped: dict[tuple[str, str, str, int], list[int]] = {}
    for index, row in enumerate(rows):
        key = (
            str(row["hardware_id"]),
            str(row["configuration_id"]),
            str(row["state_id"]),
            int(row["rank_count"]),
        )
        grouped.setdefault(key, []).append(index)
    for indices in grouped.values():
        repetitions = tuple(
            int(value) for value in rows[indices[0]]["paired_block_repetitions"]
        )
        block_count = len(repetitions)
        draws = rng.integers(0, block_count, size=(resamples, block_count))
        for index in indices:
            row_values = np.asarray(
                rows[index]["paired_block_medians_s"], dtype=np.float64
            )
            if (
                tuple(int(value) for value in rows[index]["paired_block_repetitions"])
                != repetitions
                or row_values.shape != (block_count,)
            ):
                raise ValueError("paired allocation blocks are not aligned by repetition")
            values[:, index] = np.median(row_values[draws], axis=1)
    return values


def fit_cost_model(
    empirical_rows: list[dict[str, Any]],
    contract: dict[str, Any],
    *,
    factorized_gate: dict[str, Any] | None = None,
    endpoint_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feature_order = list(contract["cost_model"]["features_in_order"])
    eligible = [
        row
        for row in empirical_rows
        if row["status"] == "admitted"
        and row["model_covariates"] is not None
        and row["publication_model_eligible"] is True
        and row["split"] in {"training", "holdout"}
    ]
    training = [row for row in eligible if row["split"] == "training"]
    holdout = [row for row in eligible if row["split"] == "holdout"]
    reasons = _model_preflight(
        training,
        holdout,
        contract=contract,
        factorized_gate=factorized_gate,
        endpoint_gate=endpoint_gate,
    )
    preflight_status = (
        "not_fit_design_not_released"
        if "cost_model_design_not_released" in reasons
        else ("not_fit_insufficient_data" if reasons else "preflight_passed")
    )
    base: dict[str, Any] = {
        "status": preflight_status,
        "selector_claim_admissible": False,
        "feature_order": feature_order,
        "training_rows": len(training),
        "holdout_rows": len(holdout),
        "preflight_failures": reasons,
    }
    if reasons:
        return base
    factorized_gate = dict(factorized_gate or {})
    x_train = np.vstack(
        [
            _feature_vector(row, feature_order, factorized_gate=factorized_gate)
            for row in training
        ]
    )
    y_train = np.log(
        np.asarray(
            [row["admitted_wall_time_median_s"] for row in training],
            dtype=np.float64,
        )
    )
    rank = int(np.linalg.matrix_rank(x_train))
    condition = float(np.linalg.cond(x_train))
    base["design_rank"] = rank
    base["design_columns"] = int(x_train.shape[1])
    base["design_condition_number"] = condition
    if rank != x_train.shape[1] or condition > float(
        contract["cost_model"]["maximum_design_condition_number"]
    ):
        base["status"] = "not_fit_invalid_design"
        base["preflight_failures"] = ["rank_deficient_or_ill_conditioned_design"]
        return base
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
        x_train, y_train, rcond=None
    )
    x_holdout = np.vstack(
        [
            _feature_vector(row, feature_order, factorized_gate=factorized_gate)
            for row in holdout
        ]
    )
    predictions = np.exp(x_holdout @ coefficients)
    observed = np.asarray(
        [row["admitted_wall_time_median_s"] for row in holdout], dtype=np.float64
    )
    ape = np.abs(predictions - observed) / observed
    median_ape = float(np.median(ape))
    p90_ape = float(np.quantile(ape, 0.9))

    model = contract["cost_model"]
    bootstrap_resamples = int(model["paired_block_bootstrap_resamples"])
    bootstrap_seed = int(model["paired_block_bootstrap_seed"])
    bootstrap_confidence = float(model["paired_block_bootstrap_confidence_level"])
    if bootstrap_resamples < 100 or not 0.0 < bootstrap_confidence < 1.0:
        raise ValueError("paired-block bootstrap policy is invalid")
    bootstrap_alpha = 0.5 * (1.0 - bootstrap_confidence)
    training_bootstrap = _paired_block_bootstrap_values(
        training,
        resamples=bootstrap_resamples,
        rng=np.random.Generator(np.random.PCG64(bootstrap_seed)),
    )
    holdout_bootstrap = _paired_block_bootstrap_values(
        holdout,
        resamples=bootstrap_resamples,
        rng=np.random.Generator(np.random.PCG64(bootstrap_seed + 1)),
    )
    coefficient_bootstrap = np.log(training_bootstrap) @ np.linalg.pinv(x_train).T
    prediction_bootstrap = np.exp(coefficient_bootstrap @ x_holdout.T)
    if not (
        np.all(np.isfinite(coefficient_bootstrap))
        and np.all(np.isfinite(prediction_bootstrap))
    ):
        base["status"] = "fit_gate_failed"
        base["preflight_failures"] = ["paired_block_bootstrap_nonfinite"]
        return base
    coefficient_intervals = {
        name: [
            float(np.quantile(coefficient_bootstrap[:, index], bootstrap_alpha)),
            float(
                np.quantile(
                    coefficient_bootstrap[:, index], 1.0 - bootstrap_alpha
                )
            ),
        ]
        for index, name in enumerate(feature_order)
    }

    grouped: dict[tuple[str, str, str, int], list[int]] = {}
    for index, row in enumerate(holdout):
        key = (
            row["hardware_id"],
            row["configuration_id"],
            row["state_id"],
            row["rank_count"],
        )
        grouped.setdefault(key, []).append(index)
    tie_ratio = float(contract["cost_model"]["practical_ordering_tie_ratio"])
    resolved = 0
    resolved_correct = 0
    observed_winners: set[str] = set()
    ordering_rows: list[dict[str, Any]] = []
    for key, indices in sorted(grouped.items()):
        if len(indices) < 2:
            continue
        observed_index = min(indices, key=lambda index: float(observed[index]))
        predicted_index = min(indices, key=lambda index: float(predictions[index]))
        observed_order = sorted(indices, key=lambda index: float(observed[index]))
        best_time = float(observed[observed_order[0]])
        second_time = float(observed[observed_order[1]])
        observed_winner = str(holdout[observed_index]["route"])
        predicted_winner = str(holdout[predicted_index]["route"])
        observed_ratio_intervals: dict[str, list[float]] = {}
        predicted_ratio_intervals: dict[str, list[float]] = {}
        observed_lower_bounds: list[float] = []
        predicted_lower_bounds: list[float] = []
        for competitor in indices:
            if competitor != observed_index:
                ratios = (
                    holdout_bootstrap[:, competitor]
                    / holdout_bootstrap[:, observed_index]
                )
                interval = [
                    float(np.quantile(ratios, bootstrap_alpha)),
                    float(np.quantile(ratios, 1.0 - bootstrap_alpha)),
                ]
                observed_ratio_intervals[str(holdout[competitor]["route"])] = interval
                observed_lower_bounds.append(interval[0])
            if competitor != predicted_index:
                ratios = (
                    prediction_bootstrap[:, competitor]
                    / prediction_bootstrap[:, predicted_index]
                )
                interval = [
                    float(np.quantile(ratios, bootstrap_alpha)),
                    float(np.quantile(ratios, 1.0 - bootstrap_alpha)),
                ]
                predicted_ratio_intervals[str(holdout[competitor]["route"])] = interval
                predicted_lower_bounds.append(interval[0])
        observed_resolved = bool(
            observed_lower_bounds and min(observed_lower_bounds) > tie_ratio
        )
        predicted_resolved = bool(
            predicted_lower_bounds and min(predicted_lower_bounds) > tie_ratio
        )
        is_resolved = bool(observed_resolved and predicted_resolved)
        correct = predicted_winner == observed_winner
        if is_resolved:
            resolved += 1
            resolved_correct += int(correct)
            observed_winners.add(observed_winner)
        ordering_rows.append(
            {
                "group": list(key),
                "observed_winner": observed_winner,
                "predicted_winner": predicted_winner,
                "observed_second_to_first_ratio": second_time / best_time,
                "observed_competitor_over_winner_intervals": observed_ratio_intervals,
                "predicted_competitor_over_winner_intervals": predicted_ratio_intervals,
                "observed_winner_interval_clears_tie_band": observed_resolved,
                "predicted_winner_interval_clears_tie_band": predicted_resolved,
                "uncertainty_resolved": is_resolved,
                "resolved": is_resolved,
                "correct": correct if is_resolved else None,
            }
        )
    accuracy = float(resolved_correct / resolved) if resolved else None
    gate_results = {
        "median_absolute_percentage_error": median_ape
        <= float(model["median_absolute_percentage_error_max"]),
        "p90_absolute_percentage_error": p90_ape
        <= float(model["p90_absolute_percentage_error_max"]),
        "minimum_resolved_holdout_groups": resolved
        >= int(model["minimum_resolved_holdout_groups"]),
        "resolved_ordering_accuracy": accuracy is not None
        and accuracy >= float(model["resolved_ordering_accuracy_min"]),
        "distinct_observed_holdout_winners": len(observed_winners)
        >= int(model["minimum_distinct_observed_holdout_winners"]),
    }
    selector_pass = all(gate_results.values())
    base.update(
        {
            "status": "selection_rule_passed" if selector_pass else "fit_gate_failed",
            "selector_claim_admissible": selector_pass,
            "coefficients": {
                name: float(value)
                for name, value in zip(feature_order, coefficients, strict=True)
            },
            "coefficient_bootstrap_confidence_intervals": coefficient_intervals,
            "uncertainty_method": {
                "name": "paired_allocation_block_nonparametric_bootstrap",
                "resamples": bootstrap_resamples,
                "seed": bootstrap_seed,
                "confidence_level": bootstrap_confidence,
                "joint_resampling": "same_block_indices_for_all_routes_within_each_comparison_group",
            },
            "holdout_median_absolute_percentage_error": median_ape,
            "holdout_p90_absolute_percentage_error": p90_ape,
            "resolved_holdout_groups": resolved,
            "resolved_ordering_accuracy": accuracy,
            "distinct_observed_holdout_winners": sorted(observed_winners),
            "gate_results": gate_results,
            "holdout_ordering": ordering_rows,
        }
    )
    return base


PREDICTIVE_SELECTOR_TERMINAL = "predictive_selector_admissible"
FINITE_EMPIRICAL_MAP_TERMINAL = "finite_empirical_map_only"


def _publication_safe_cost_model(model: dict[str, Any]) -> dict[str, Any]:
    """Remove predictive fit products when the prespecified selector gates fail.

    A failed fit remains useful as the reason that the terminal result is the
    finite empirical map only.  Coefficients, bootstrap coefficient intervals,
    predictions, and observed/predicted ordering rows are not publication
    products on that branch and therefore never enter ``analysis.json``.
    """
    if model.get("selector_claim_admissible") is True:
        return model
    gate_results = model.get("gate_results")
    failed_gates = (
        sorted(
            str(name)
            for name, passed in gate_results.items()
            if passed is not True
        )
        if isinstance(gate_results, dict)
        else []
    )
    return {
        "status": str(model.get("status", "")),
        "selector_claim_admissible": False,
        "feature_order": list(model.get("feature_order") or []),
        "training_rows": int(model.get("training_rows", 0)),
        "holdout_rows": int(model.get("holdout_rows", 0)),
        "preflight_failures": [
            str(value) for value in list(model.get("preflight_failures") or [])
        ],
        "failed_gates": failed_gates,
    }


def _factorized_diagnostic_integrity_errors(
    factor: Any, contract: dict[str, Any]
) -> list[str]:
    """Check diagnostic consistency without turning its outcome into a gate."""
    if not isinstance(factor, dict):
        return ["factorized diagnostic is missing"]
    errors: list[str] = []
    policy = dict(contract.get("factorized_calibration_policy") or {})
    if policy.get("required_for_selector_claim") is not False:
        errors.append("factorized diagnostic was changed into a selector gate")
    if factor.get("calibration_integrated") is not False:
        errors.append("factorized diagnostic was integrated into the selector")
    if factor.get("selector_use") != policy.get("current_status"):
        errors.append("factorized diagnostic selector-use label differs from contract")
    if factor.get("selector_blockers") != []:
        errors.append("factorized diagnostic improperly declares selector blockers")
    if factor.get("required_ranks") != [1, 8, 32]:
        errors.append("factorized diagnostic required ranks differ from contract")
    if factor.get("independent_blocks_per_rank") != int(
        policy.get("independent_blocks_per_rank", 0)
    ):
        errors.append("factorized diagnostic block count differs from contract")
    passed = factor.get("passed")
    failures = factor.get("failures")
    if not isinstance(passed, bool):
        errors.append("factorized diagnostic passed flag is not Boolean")
    if not isinstance(failures, list) or any(
        not isinstance(value, str) or not value for value in failures
    ):
        errors.append("factorized diagnostic failures are malformed")
        failures = []
    calibration = factor.get("calibration_model")
    calibration_passed = (
        isinstance(calibration, dict) and calibration.get("status") == "passed"
    )
    if passed is True and (failures != [] or not calibration_passed):
        errors.append("passed factorized diagnostic is internally inconsistent")
    if passed is False and (not failures or calibration_passed):
        errors.append("failed factorized diagnostic is internally inconsistent")
    return errors


def _complete_finite_map_is_publication_ready(
    rows: Any, contract: dict[str, Any]
) -> bool:
    if not isinstance(rows, list):
        return False
    expected = _expected_slots(
        ("workstation_local", "karolina_cpu"), contract
    )
    by_slot: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            return False
        try:
            slot = (
                str(row["hardware_id"]),
                str(row["configuration_id"]),
                str(row["state_id"]),
                int(row["rank_count"]),
                str(row["route"]),
            )
        except (KeyError, TypeError, ValueError):
            return False
        if slot in by_slot:
            return False
        by_slot[slot] = row
    if set(by_slot) != expected:
        return False
    for slot, row in by_slot.items():
        censor_reason = _is_structural_censor(slot, contract)
        if censor_reason is not None:
            if (
                row.get("status") != "censored"
                or row.get("reason") != censor_reason
                or row.get("admitted_wall_time_median_s") is not None
            ):
                return False
            continue
        timing = row.get("admitted_wall_time_median_s")
        if (
            row.get("status") != "admitted"
            or isinstance(timing, bool)
            or not isinstance(timing, (int, float))
            or not math.isfinite(float(timing))
            or float(timing) <= 0.0
        ):
            return False
    return True


def _publication_evidence_is_admissible(
    *,
    clean_committed_analysis: bool,
    terminal_decision: str,
    empirical_rows: Any,
    cost_model: Any,
    endpoint_gate: Any,
    factorized_gate: Any,
    invalid_records: Any,
    contract: dict[str, Any],
) -> bool:
    """Outcome-independent release gate for either prespecified terminal."""
    if not isinstance(empirical_rows, list) or not isinstance(cost_model, dict):
        return False
    training_rows = sum(
        isinstance(row, dict)
        and row.get("status") == "admitted"
        and row.get("publication_model_eligible") is True
        and row.get("split") == "training"
        for row in empirical_rows
    )
    holdout_rows = sum(
        isinstance(row, dict)
        and row.get("status") == "admitted"
        and row.get("publication_model_eligible") is True
        and row.get("split") == "holdout"
        for row in empirical_rows
    )
    common_model_ready = bool(
        training_rows == 74
        and holdout_rows == 22
        and cost_model.get("training_rows") == training_rows
        and cost_model.get("holdout_rows") == holdout_rows
        and cost_model.get("feature_order") == contract["cost_model"]["features_in_order"]
    )
    if terminal_decision == PREDICTIVE_SELECTOR_TERMINAL:
        terminal_model_ready = bool(
            cost_model.get("status") == "selection_rule_passed"
            and cost_model.get("selector_claim_admissible") is True
        )
    elif terminal_decision == FINITE_EMPIRICAL_MAP_TERMINAL:
        allowed_keys = {
            "status",
            "selector_claim_admissible",
            "feature_order",
            "training_rows",
            "holdout_rows",
            "preflight_failures",
            "failed_gates",
        }
        frozen_gates = {
            "median_absolute_percentage_error",
            "p90_absolute_percentage_error",
            "minimum_resolved_holdout_groups",
            "resolved_ordering_accuracy",
            "distinct_observed_holdout_winners",
        }
        preflight = cost_model.get("preflight_failures")
        failed = cost_model.get("failed_gates")
        if cost_model.get("status") == "fit_gate_failed":
            negative_reason_ready = bool(
                preflight == []
                and isinstance(failed, list)
                and failed
                and len(failed) == len(set(failed))
                and set(failed) <= frozen_gates
            )
        elif cost_model.get("status") == "not_fit_invalid_design":
            negative_reason_ready = bool(
                preflight == ["rank_deficient_or_ill_conditioned_design"]
                and failed == []
            )
        else:
            negative_reason_ready = False
        terminal_model_ready = bool(
            set(cost_model) == allowed_keys
            and cost_model.get("selector_claim_admissible") is False
            and negative_reason_ready
        )
    else:
        terminal_model_ready = False
    return bool(
        clean_committed_analysis
        and common_model_ready
        and terminal_model_ready
        and _complete_finite_map_is_publication_ready(empirical_rows, contract)
        and isinstance(endpoint_gate, dict)
        and endpoint_gate.get("publication_admissible") is True
        and invalid_records == []
        and _factorized_diagnostic_integrity_errors(factorized_gate, contract) == []
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "hardware_id",
        "configuration_id",
        "state_id",
        "rank_count",
        "route",
        "split",
        "status",
        "reason",
        "state_sha256",
        "action_sha256",
        "action_relative_l2_error",
        "action_relative_l2_errors",
        "action_max_absolute_error",
        "gradient_residual_relative_error",
        "admitted_wall_time_median_s",
        "paired_block_medians_s",
        "paired_block_repetitions",
        "paired_block_route_positions",
        "publication_model_eligible",
        "model_exclusion_reason",
        "record_path",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    temporary.replace(path)


def _write_report(
    path: Path,
    *,
    empirical_rows: list[dict[str, Any]],
    model: dict[str, Any],
    factorized_gate: dict[str, Any],
    invalid_records: list[dict[str, str]],
) -> None:
    counts: dict[str, int] = {}
    for row in empirical_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    lines = [
        "# EXP-ROUTE-001 finite empirical route map",
        "",
        "Timing appears in the map only after the exact-state and numerical "
        "tangent-action equivalence gates pass for the complete active route set. "
        "Missing, censored, and invalid rows are never imputed.",
        "",
        "## Admission summary",
        "",
        "| Status | Rows |",
        "| --- | ---: |",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"| `{status}` | {count} |")
    lines.extend(
        [
            "",
            "## Predictive model decision",
            "",
            f"- Status: `{model['status']}`.",
            f"- Selector claim admissible: `{str(bool(model['selector_claim_admissible'])).lower()}`.",
            f"- Publication-model-eligible training rows: {model['training_rows']}.",
            f"- Publication-model-eligible holdout rows: {model['holdout_rows']}.",
        ]
    )
    failures = model.get("preflight_failures") or []
    if failures:
        lines.append(f"- Blocking gates: {', '.join(f'`{value}`' for value in failures)}.")
    failed_gates = model.get("failed_gates") or []
    if failed_gates:
        lines.append(
            "- Failed frozen validation gates: "
            + ", ".join(f"`{value}`" for value in failed_gates)
            + "."
        )
    factor_failures = list(factorized_gate.get("failures") or [])
    lines.extend(
        [
            "",
            "## Synthetic factor diagnostic",
            "",
            "- This mechanism diagnostic is descriptive and is not a selector gate.",
            f"- Diagnostic passed: `{str(factorized_gate.get('passed') is True).lower()}`.",
            f"- Recorded diagnostic failures: {len(factor_failures)}.",
        ]
    )
    if invalid_records:
        lines.extend(
            [
                "",
                "## Invalid input records",
                "",
                "Invalid inputs are excluded before equivalence or timing admission.",
                "",
            ]
        )
        for row in invalid_records:
            lines.append(f"- `{row['path']}`: {row['reason']}")
    lines.extend(
        [
            "",
            "The machine-readable decisions are in `analysis.json`; the complete "
            "finite map is in `empirical_route_map.csv`.",
            "",
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


_FACTOR_NAMES = (
    "element_dofs",
    "quadrature_points",
    "constitutive_dimension",
    "color_count",
    "nonzeros_per_row",
    "message_bytes",
    "imbalance_ratio",
)
_FACTOR_BASELINE = {
    "element_dofs": 30.0,
    "quadrature_points": 11.0,
    "constitutive_dimension": 6.0,
    "color_count": 64.0,
    "nonzeros_per_row": 48.0,
    "message_bytes": 65536.0,
    "imbalance_ratio": 1.0,
}
_FACTOR_STAGES = ("contraction", "color_hvp", "insertion", "communication")
_FACTOR_FEATURES = ("intercept", "log_rank_count", *[f"log_{name}_ratio" for name in _FACTOR_NAMES])


def _expected_factor_design() -> list[dict[str, int | str]]:
    baseline = {name: int(value) for name, value in _FACTOR_BASELINE.items()}
    rows: list[dict[str, int | str]] = [{"case_id": "baseline", **baseline}]
    levels = {
        "element_dofs": (12, 105),
        "quadrature_points": (1, 24, 125),
        "constitutive_dimension": (3, 9),
        "color_count": (32, 128),
        "nonzeros_per_row": (20, 100),
        "message_bytes": (8192, 1048576),
        "imbalance_ratio": (2, 4),
    }
    for factor, values in levels.items():
        for value in values:
            row = dict(baseline)
            row[factor] = int(value)
            rows.append({"case_id": f"{factor}_{value}", **row})
    return rows


def _validate_factor_payload_design_and_timings(payload: dict[str, Any]) -> None:
    if payload.get("one_factor_at_a_time") is not True:
        raise ValueError("factor payload does not declare one-factor-at-a-time design")
    if payload.get("factor_order_policy") != "deterministic_rotated_v1":
        raise ValueError("factor payload changed its deterministic order policy")
    expected = _expected_factor_design()
    offset = int(payload.get("block_repetition", -1)) % len(expected)
    expected = expected[offset:] + expected[:offset]
    results = list(payload.get("results") or [])
    if len(results) != len(expected):
        raise ValueError("factorized calibration must retain the exact 16-case design")
    for index, (result, planned) in enumerate(zip(results, expected, strict=True)):
        for key, value in planned.items():
            if result.get(key) != value:
                raise ValueError(f"factor result {index} changes reviewed field {key}")
        ranks = int(result.get("mpi_ranks", 0))
        if ranks < 1:
            raise ValueError("factor result has an invalid MPI rank count")
        local_batches = np.asarray(result.get("local_batch_by_rank", []), dtype=np.int64)
        insertion_rows = np.asarray(
            result.get("insertion_rows_by_rank", []), dtype=np.int64
        )
        if (
            local_batches.shape != (ranks,)
            or insertion_rows.shape != (ranks,)
            or np.any(local_batches < 0)
            or np.any(insertion_rows < 0)
            or int(np.sum(local_batches)) != 8 * ranks
            or int(np.sum(insertion_rows)) != 512 * ranks
        ):
            raise ValueError("factor imbalance design does not conserve total work")
        target_imbalance = int(planned["imbalance_ratio"])
        applicable = not (ranks == 1 and target_imbalance > 1)
        if result.get("imbalance_factor_applicable") is not applicable:
            raise ValueError("factor imbalance applicability is mislabeled")
        expected_realized = float(target_imbalance if applicable else 1.0)
        for field in (
            "realized_batch_max_over_mean",
            "realized_insertion_max_over_mean",
        ):
            if not math.isclose(
                float(result.get(field, float("nan"))),
                expected_realized,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"factor {field} differs from the realized design")
        allocation_raw = np.asarray(
            result.get("allocation_times_by_rank_s", []), dtype=np.float64
        )
        allocation_max = float(result.get("allocation_collective_max_s", float("nan")))
        if (
            allocation_raw.shape != (ranks,)
            or not np.all(np.isfinite(allocation_raw))
            or np.any(allocation_raw <= 0.0)
            or not math.isclose(
                allocation_max,
                float(np.max(allocation_raw)),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        ):
            raise ValueError("factor allocation MPI_MAX lacks raw-rank proof")
        maxima = dict(result.get("stage_collective_max_times_s") or {})
        raw_stages = dict(result.get("stage_times_by_rank_s") or {})
        if set(maxima) != set(_FACTOR_STAGES) or set(raw_stages) != set(_FACTOR_STAGES):
            raise ValueError("factor result changed the frozen stage set")
        for stage in _FACTOR_STAGES:
            values = np.asarray(maxima[stage], dtype=np.float64)
            raw = np.asarray(raw_stages[stage], dtype=np.float64)
            if (
                values.size < 5
                or raw.shape != (values.size, ranks)
                or not np.all(np.isfinite(values))
                or not np.all(np.isfinite(raw))
                or np.any(values <= 0.0)
                or np.any(raw <= 0.0)
                or not np.allclose(
                    values,
                    np.max(raw, axis=1),
                    rtol=1.0e-12,
                    atol=1.0e-15,
                )
            ):
                raise ValueError(f"factor stage {stage} lacks raw-rank MPI_MAX proof")


def _factor_calibration_vector(row: dict[str, Any], *, ranks: int) -> np.ndarray:
    values = [1.0, math.log(float(ranks))]
    for name in _FACTOR_NAMES:
        value = float(row[name])
        baseline = float(_FACTOR_BASELINE[name])
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"factor {name} must be finite and positive")
        values.append(math.log(value / baseline))
    return np.asarray(values, dtype=np.float64)


def _fit_factorized_calibration(
    payload_by_rank: dict[int, dict[str, Any] | list[dict[str, Any]]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    policy = dict(contract.get("factorized_calibration_policy") or {})
    training_ranks = [int(value) for value in policy.get("training_ranks", [])]
    validation_rank = int(policy.get("validation_rank", 0))
    stages = tuple(str(value) for value in policy.get("stage_names", []))
    if training_ranks != [1, 8] or validation_rank != 32 or stages != _FACTOR_STAGES:
        raise ValueError("factorized calibration policy differs from the reviewed split")

    x_rows: list[np.ndarray] = []
    y_by_stage: dict[str, list[float]] = {stage: [] for stage in stages}
    for ranks in training_ranks:
        payloads = payload_by_rank[ranks]
        if isinstance(payloads, dict):
            payloads = [payloads]
        for payload in payloads:
            for row in payload["results"]:
                if ranks == 1 and int(row["imbalance_ratio"]) > 1:
                    continue
                x_rows.append(_factor_calibration_vector(row, ranks=ranks))
                for stage in stages:
                    samples = np.asarray(
                        row["stage_collective_max_times_s"][stage], dtype=np.float64
                    )
                    warm = samples[1:]
                    y_by_stage[stage].append(float(np.log(np.median(warm))))
    x_train = np.vstack(x_rows)
    rank = int(np.linalg.matrix_rank(x_train))
    condition = float(np.linalg.cond(x_train))
    if rank != x_train.shape[1] or not math.isfinite(condition) or condition > 1.0e10:
        raise ValueError("factorized calibration design is rank deficient or ill conditioned")

    coefficients: dict[str, list[float]] = {}
    validation_errors: dict[str, dict[str, float]] = {}
    all_errors: list[float] = []
    validation_payloads = payload_by_rank[validation_rank]
    if isinstance(validation_payloads, dict):
        validation_payloads = [validation_payloads]
    validation_rows = [
        row for payload in validation_payloads for row in payload["results"]
    ]
    x_validation = np.vstack(
        [
            _factor_calibration_vector(row, ranks=validation_rank)
            for row in validation_rows
        ]
    )
    for stage in stages:
        y_train = np.asarray(y_by_stage[stage], dtype=np.float64)
        coefficient, _residuals, _rank, _singular = np.linalg.lstsq(
            x_train, y_train, rcond=None
        )
        coefficients[stage] = [float(value) for value in coefficient]
        predicted = np.exp(x_validation @ coefficient)
        observed = np.asarray(
            [
                np.median(
                    np.asarray(row["stage_collective_max_times_s"][stage], dtype=np.float64)[1:]
                )
                for row in validation_rows
            ],
            dtype=np.float64,
        )
        errors = np.abs(predicted - observed) / observed
        all_errors.extend(float(value) for value in errors)
        validation_errors[stage] = {
            "median_absolute_percentage_error": float(np.median(errors)),
            "p90_absolute_percentage_error": float(np.quantile(errors, 0.9)),
        }
    median_ape = float(np.median(np.asarray(all_errors, dtype=np.float64)))
    p90_ape = float(np.quantile(np.asarray(all_errors, dtype=np.float64), 0.9))
    gates = {
        "median_absolute_percentage_error": median_ape
        <= float(policy["median_absolute_percentage_error_max"]),
        "p90_absolute_percentage_error": p90_ape
        <= float(policy["p90_absolute_percentage_error_max"]),
    }
    return {
        "status": "passed" if all(gates.values()) else "validation_gate_failed",
        "training_ranks": training_ranks,
        "validation_rank": validation_rank,
        "independent_blocks_by_rank": {
            str(ranks): len(payloads) if isinstance(payloads, list) else 1
            for ranks, payloads in payload_by_rank.items()
        },
        "feature_order": list(_FACTOR_FEATURES),
        "design_rank": rank,
        "design_columns": int(x_train.shape[1]),
        "design_condition_number": condition,
        "coefficients_by_stage": coefficients,
        "validation_by_stage": validation_errors,
        "validation_median_absolute_percentage_error": median_ape,
        "validation_p90_absolute_percentage_error": p90_ape,
        "gate_results": gates,
    }


def _calibrated_shared_stage_seconds(
    covariates: dict[str, float], factorized_gate: dict[str, Any]
) -> float:
    calibration = dict(factorized_gate.get("calibration_model") or {})
    if calibration.get("status") != "passed":
        raise ValueError("factorized calibration model has not passed")
    owned_dofs = max(float(covariates["maximum_rank_owned_dofs"]), 1.0)
    overlap = max(float(covariates["maximum_rank_overlap_dofs"]), owned_dofs)
    factor_row = {
        "element_dofs": float(covariates["element_dofs"]),
        "quadrature_points": float(covariates["quadrature_points_per_element"]),
        "constitutive_dimension": float(covariates["constitutive_dimension"]),
        "color_count": max(float(covariates["maximum_local_color_count"]), 1.0),
        "nonzeros_per_row": max(
            float(covariates["owned_matrix_nonzeros"]) / owned_dofs, 1.0
        ),
        "message_bytes": max(8.0 * (overlap - owned_dofs), 8.0),
        "imbalance_ratio": max(
            float(covariates["owned_element_imbalance"]),
            float(covariates["overlap_dof_imbalance"]),
            1.0,
        ),
    }
    vector = _factor_calibration_vector(
        factor_row, ranks=int(round(float(covariates["rank_count"])))
    )
    coefficients = dict(calibration["coefficients_by_stage"])
    predictions = {
        stage: float(np.exp(vector @ np.asarray(coefficients[stage], dtype=np.float64)))
        for stage in _FACTOR_STAGES
    }
    route = str(covariates["route"])
    if route == "colored_sfd":
        selected = ("color_hvp", "insertion", "communication")
    else:
        selected = ("contraction", "insertion", "communication")
    total = float(sum(predictions[stage] for stage in selected))
    if not (math.isfinite(total) and total > 0.0):
        raise ValueError("calibrated shared-stage prediction is invalid")
    return total


def _factorized_microbenchmark_gate(
    sources: list[tuple[str, Path]], contract: dict[str, Any]
) -> dict[str, Any]:
    required_ranks = {1, 8, 32}
    policy = dict(contract.get("factorized_calibration_policy") or {})
    blocks_per_rank = int(policy.get("independent_blocks_per_rank", 0))
    admitted: dict[int, list[dict[str, Any]]] = {}
    payload_by_rank: dict[int, list[dict[str, Any]]] = {}
    failures: list[str] = []
    expected_factors = {
        "element_dofs",
        "quadrature_points",
        "constitutive_dimension",
        "color_count",
        "nonzeros_per_row",
        "message_bytes",
        "imbalance_ratio",
    }
    for hardware_id, root in sources:
        if hardware_id != "karolina_cpu":
            continue
        source_gate = _source_provenance_gate(hardware_id, root, contract)
        for path in sorted(root.rglob("*.json")):
            try:
                payload = _read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if payload.get("experiment_id") != "EXP-ROUTE-001" or payload.get(
                "tier"
            ) != "factorized_microbenchmark":
                continue
            try:
                _bind_factor_record_to_reviewed_matrix(
                    path,
                    payload,
                    contract=contract,
                )
                ranks = int(payload["results"][0]["mpi_ranks"])
                if ranks not in required_ranks:
                    raise ValueError("factor rank point is out of contract")
                if payload.get("status") != "completed":
                    raise ValueError("factorized calibration did not complete")
                if payload.get("timing_reduction") != "mpi_collective_max":
                    raise ValueError("factorized calibration is not collective-max timed")
                if set(payload.get("factors", [])) != expected_factors:
                    raise ValueError("factorized calibration changed its factor set")
                _validate_factor_payload_design_and_timings(payload)
                results = list(payload.get("results", []))
                if not str(payload.get("command", "")).strip():
                    raise ValueError("factorized calibration lacks the exact command")
                runtime = dict(payload.get("numerical_runtime") or {})
                if not str(runtime.get("numpy", "")) or not isinstance(
                    runtime.get("cpu_affinity"), list
                ):
                    raise ValueError("factorized calibration lacks runtime/affinity metadata")
                git = dict(payload.get("git") or {})
                if git.get("dirty") is not False or len(str(git.get("commit", ""))) != 40:
                    raise ValueError("factorized calibration lacks clean commit provenance")
                if source_gate.get("eligible") is not True:
                    raise ValueError(str(source_gate.get("reason")))
                if git.get("commit") != source_gate.get("source_commit"):
                    raise ValueError("factorized calibration commit differs from manifest")
                expected_job_id = path.parents[1].name.removeprefix("job_")
                actual_job_id = str(
                    dict(payload.get("job_metadata") or {}).get("slurm_job_id", "")
                )
                if not expected_job_id or actual_job_id != expected_job_id:
                    raise ValueError("factorized calibration Slurm job identity is stale")
                evidence_files = _cluster_factor_evidence_files(root, path, payload)
                block = int(payload.get("block_repetition", 0))
                if block not in range(1, blocks_per_rank + 1):
                    raise ValueError("factor block repetition is out of contract")
                if any(item["block_repetition"] == block for item in admitted.get(ranks, [])):
                    raise ValueError("factor rank/block point is duplicated")
                admitted.setdefault(ranks, []).append({
                    "path": str(path),
                    "case_count": len(results),
                    "block_repetition": block,
                    "evidence_files": evidence_files,
                })
                payload_by_rank.setdefault(ranks, []).append(payload)
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                failures.append(f"{path}: {exc}")
    missing = sorted(required_ranks - set(admitted))
    if missing:
        failures.append(f"missing factorized rank points: {missing}")
    for ranks in sorted(required_ranks & set(admitted)):
        blocks = sorted(item["block_repetition"] for item in admitted[ranks])
        expected_blocks = list(range(1, blocks_per_rank + 1))
        if blocks != expected_blocks:
            failures.append(
                f"factorized rank {ranks} blocks {blocks} != {expected_blocks}"
            )
    calibration_model: dict[str, Any] | None = None
    if not failures and set(payload_by_rank) == required_ranks:
        try:
            for ranks in payload_by_rank:
                payload_by_rank[ranks].sort(
                    key=lambda payload: int(payload.get("block_repetition", 0))
                )
            calibration_model = _fit_factorized_calibration(payload_by_rank, contract)
            if calibration_model.get("status") != "passed":
                failures.append("factorized calibration holdout gates failed")
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            failures.append(f"factorized calibration fit failed: {exc}")
    diagnostic_passed = not failures and calibration_model is not None
    return {
        "passed": diagnostic_passed,
        "calibration_integrated": False,
        "selector_use": "descriptive_replicated_synthetic_non_route_faithful_proxy",
        "selector_blockers": [],
        "independent_blocks_per_rank": blocks_per_rank,
        "required_ranks": sorted(required_ranks),
        "admitted": admitted,
        "calibration_model": calibration_model,
        "failures": failures,
    }


def _collect_analysis_input_evidence(
    *,
    sources: list[tuple[str, Path]],
    observed: dict[tuple[str, str, str, int, str], dict[str, Any]],
    invalid_records: list[dict[str, str]],
    factorized_gate: dict[str, Any],
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for _hardware, root in sources:
        master = root / "route_campaign_master_manifest.json"
        if master.is_file():
            entries.append(_evidence_entry(master.resolve(), "route_campaign_master"))
            try:
                master_payload = _read_json(master)
                for tranche in master_payload.get("tranches") or []:
                    if not isinstance(tranche, dict):
                        continue
                    for key, role in (
                        ("manifest_path", "route_tranche_manifest"),
                        ("submitted_jobs_path", "route_submission_ledger"),
                        ("release_authorization_path", "route_release_authorization"),
                    ):
                        relative = Path(str(tranche.get(key, "")))
                        if relative.is_absolute():
                            continue
                        evidence = (master.parent / relative).resolve()
                        if evidence.is_file():
                            entries.append(_evidence_entry(evidence, role))
                            if key == "release_authorization_path":
                                release = _read_json(evidence)
                                for artifact in release.get("reviewed_artifacts") or []:
                                    if not isinstance(artifact, dict):
                                        continue
                                    artifact_relative = Path(
                                        str(artifact.get("path", ""))
                                    )
                                    if artifact_relative.is_absolute():
                                        continue
                                    artifact_path = (
                                        evidence.parent / artifact_relative
                                    ).resolve()
                                    if artifact_path.is_file():
                                        entries.append(
                                            _evidence_entry(
                                                artifact_path,
                                                "reviewed_release_artifact",
                                            )
                                        )
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        for manifest in root.rglob("prepared_manifest.json"):
            entries.append(_evidence_entry(manifest.resolve(), "prepared_manifest"))
    for record in observed.values():
        entries.extend(record.get("record_evidence") or [])
        entries.extend(record.get("evidence_files") or [])
    for record in invalid_records:
        for raw_path in str(record.get("path", "")).split(";"):
            path = Path(raw_path)
            if path.is_file():
                entries.append(_evidence_entry(path.resolve(), "invalid_input_record"))
    for admitted_blocks in dict(factorized_gate.get("admitted") or {}).values():
        for admitted in list(admitted_blocks):
            entries.extend(dict(admitted).get("evidence_files") or [])
    unique = {
        (entry["role"], entry["path"], entry["sha256"]): entry for entry in entries
    }
    return sorted(unique.values(), key=lambda entry: (entry["path"], entry["role"]))


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    contract_path = Path(args.contract).resolve()
    contract = _read_json(contract_path)
    if contract.get("experiment_id") != "EXP-ROUTE-001":
        raise ValueError("analysis contract has the wrong experiment_id")
    terminal_policy = contract.get("terminal_policy")
    if terminal_policy != {
        "selector_claim_requires_all_model_gates": True,
        "selector_admitted": PREDICTIVE_SELECTOR_TERMINAL,
        "otherwise": FINITE_EMPIRICAL_MAP_TERMINAL,
        "never_impute_censored_or_missing_timings": True,
    }:
        raise ValueError("analysis contract has the wrong two-branch terminal policy")
    sources = [_parse_source(value) for value in args.source]
    endpoint_argument = getattr(args, "endpoint_analysis", None)
    endpoint_input = None if endpoint_argument is None else Path(endpoint_argument).resolve()
    stopping_argument = getattr(args, "stopping_adjudication", None)
    stopping_input = (
        None if stopping_argument is None else Path(stopping_argument).resolve()
    )
    normalized_command = shlex.join(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--contract",
            str(contract_path),
            *[
                value
                for hardware, root in sources
                for value in ("--source", f"{hardware}={root.resolve()}")
            ],
            *(
                []
                if endpoint_input is None
                else ["--endpoint-analysis", str(endpoint_input)]
            ),
            *(
                []
                if stopping_input is None
                else ["--stopping-adjudication", str(stopping_input)]
            ),
            "--output-dir",
            str(Path(args.output_dir).resolve()),
        ]
    )
    unknown = sorted(
        {hardware for hardware, _root in sources} - set(contract["hardware"])
    )
    if unknown:
        raise ValueError(f"unknown hardware identifiers: {unknown}")
    observed: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    runtime_censors: dict[tuple[str, str, str, int, str], str] = {}
    invalid_records: list[dict[str, str]] = []
    for hardware_id, root in sources:
        source_provenance = _source_provenance_gate(hardware_id, root, contract)
        new_observed, new_censors, new_invalid = _scan_source(
            hardware_id,
            root,
            contract=contract,
            source_provenance=source_provenance,
        )
        overlap = set(observed) & set(new_observed)
        if overlap:
            raise ValueError(f"duplicate slots across source roots: {sorted(overlap)}")
        observed.update(new_observed)
        runtime_censors.update(new_censors)
        invalid_records.extend(new_invalid)
    empirical_rows = build_empirical_map(
        contract=contract,
        hardware_ids=[hardware for hardware, _root in sources],
        observed=observed,
        runtime_censors=runtime_censors,
    )
    factorized_gate = _factorized_microbenchmark_gate(sources, contract)
    endpoint_gate = _endpoint_analysis_gate(
        endpoint_input,
        stopping_input,
        sources=sources,
        contract=contract,
    )
    fitted_model = fit_cost_model(
        empirical_rows,
        contract,
        factorized_gate=factorized_gate,
        endpoint_gate=endpoint_gate,
    )
    terminal_decision = (
        str(terminal_policy["selector_admitted"])
        if fitted_model["selector_claim_admissible"]
        else str(terminal_policy["otherwise"])
    )
    model = _publication_safe_cost_model(fitted_model)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint_public, archived_endpoint, archived_stopping = _archive_endpoint_gate(
        endpoint_gate, output_dir
    )
    _write_csv(output_dir / "empirical_route_map.csv", empirical_rows)
    result = {
        "analysis_schema_version": 1,
        "experiment_id": "EXP-ROUTE-001",
        "contract_path": str(contract_path),
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "sources": [
            {
                "hardware_id": hardware,
                "root": str(root),
                "publication_provenance_gate": _source_provenance_gate(
                    hardware, root, contract
                ),
            }
            for hardware, root in sources
        ],
        "terminal_decision": terminal_decision,
        "empirical_map": empirical_rows,
        "cost_model": model,
        "endpoint_analysis": endpoint_public,
        "factorized_microbenchmark_gate": factorized_gate,
        "invalid_records": invalid_records,
        "provenance": {
            "git": _git_metadata(),
            "command": normalized_command,
            "normalized_exact_command": normalized_command,
            "environment": {
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "numpy_version": np.__version__,
                "thread_environment": {
                    name: os.environ.get(name, "")
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
            },
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": _sha256_file(Path(__file__).resolve()),
            "input_files": [
                *_collect_analysis_input_evidence(
                    sources=sources,
                    observed=observed,
                    invalid_records=invalid_records,
                    factorized_gate=factorized_gate,
                ),
                *(
                    []
                    if archived_endpoint is None
                    else [
                        _evidence_entry(
                            archived_endpoint, "tier_b_endpoint_analysis"
                        ),
                        _evidence_entry(
                            archived_stopping, "tier_b_stopping_adjudication"
                        ),
                    ]
                ),
            ],
        },
    }
    analysis_path = output_dir / "analysis.json"
    map_path = output_dir / "empirical_route_map.csv"
    report_path = output_dir / "report.md"
    atomic_write_json(analysis_path, result)
    _write_report(
        report_path,
        empirical_rows=empirical_rows,
        model=model,
        factorized_gate=factorized_gate,
        invalid_records=invalid_records,
    )
    analysis_git = dict(result["provenance"]["git"])
    source_gates = [dict(source["publication_provenance_gate"]) for source in result["sources"]]
    source_commits = {str(gate.get("source_commit", "")) for gate in source_gates}
    clean_committed_analysis = bool(
        analysis_git.get("dirty") is False
        and len(str(analysis_git.get("commit", ""))) == 40
        and all(gate.get("eligible") is True for gate in source_gates)
        and source_commits == {str(analysis_git.get("commit", ""))}
    )
    publication_evidence = _publication_evidence_is_admissible(
        clean_committed_analysis=clean_committed_analysis,
        terminal_decision=result["terminal_decision"],
        empirical_rows=empirical_rows,
        cost_model=model,
        endpoint_gate=endpoint_public,
        factorized_gate=factorized_gate,
        invalid_records=invalid_records,
        contract=contract,
    )
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "manifest_version": 1,
            "experiment_id": "EXP-ROUTE-001",
            "status": (
                "publication_evidence"
                if publication_evidence
                else (
                    "clean_committed_analysis_not_released"
                    if clean_committed_analysis
                    else "diagnostic_not_publication_evidence"
                )
            ),
            "publication_evidence": publication_evidence,
            "run_kind": "publication" if publication_evidence else "diagnostic",
            "terminal_decision": result["terminal_decision"],
            "contract_path": result["contract_path"],
            "contract_sha256": result["contract_sha256"],
            "normalized_exact_command": normalized_command,
            "command": normalized_command,
            "environment": result["provenance"]["environment"],
            "code_hashes": {
                str(Path(__file__).resolve().relative_to(REPO_ROOT)): _sha256_file(
                    Path(__file__).resolve()
                ),
                str(
                    Path(validate_stop_adjudication.__code__.co_filename)
                    .resolve()
                    .relative_to(REPO_ROOT)
                ): _sha256_file(
                    Path(validate_stop_adjudication.__code__.co_filename).resolve()
                ),
            },
            "output_hashes": {
                analysis_path.name: _sha256_file(analysis_path),
                map_path.name: _sha256_file(map_path),
                report_path.name: _sha256_file(report_path),
                **(
                    {}
                    if archived_endpoint is None
                    else {
                        archived_endpoint.name: _sha256_file(archived_endpoint),
                        archived_stopping.name: _sha256_file(archived_stopping),
                    }
                ),
            },
            "input_hashes": (
                {}
                if archived_endpoint is None
                else {
                    archived_endpoint.name: _sha256_file(archived_endpoint),
                    archived_stopping.name: _sha256_file(archived_stopping),
                    str(TIER_B_STOPPING_POLICY_PATH.relative_to(REPO_ROOT)): (
                        stopping_sha256_file(TIER_B_STOPPING_POLICY_PATH)
                    ),
                }
            ),
            "endpoint_analysis": endpoint_public,
            "provenance": result["provenance"],
        },
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Repeat HARDWARE_ID=PATH for each fixed-state result root.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint-analysis",
        type=Path,
        help="Hash-bound Tier-B endpoint analysis archived under the Karolina source.",
    )
    parser.add_argument(
        "--stopping-adjudication",
        type=Path,
        required=True,
        help=(
            "Final checksum-bound EXP-STOP-001 adjudication archived under the "
            "same Karolina source as the Tier-B endpoint analysis."
        ),
    )
    return parser


def main() -> None:
    try:
        result = analyze(_parser().parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps({
        "terminal_decision": result["terminal_decision"],
        "model_status": result["cost_model"]["status"],
        "rows": len(result["empirical_map"]),
    }, indent=2))


if __name__ == "__main__":
    main()

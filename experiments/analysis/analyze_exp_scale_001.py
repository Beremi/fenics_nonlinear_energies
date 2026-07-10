#!/usr/bin/env python3
"""Fail-closed analysis for the two separate EXP-SCALE-001 series.

The required Hyperelasticity series is the default.  The optional
Plasticity3D series requires an explicit ``--series optional_p3d`` invocation
and is emitted as a different result block; the program never pools them.
No scheduler command is invoked by this analyzer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.benchmark.run_record import atomic_write_json


SCHEMA_ID = "fenics-nonlinear-energies.exp-scale-001-analysis"
SCHEMA_VERSION = 1
DEFAULT_CONTRACT = REPO_ROOT / "paper/protocols/EXP-SCALE-001-analysis-contract.json"
DEFAULT_MATRIX = (
    REPO_ROOT / "experiments/runners/paper_revision_karolina/campaign_matrix.csv"
)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SUBMITTED_JOB = re.compile(r"(?:Submitted\s+batch\s+job\s+)?([0-9]+(?:_[0-9]+)?)")


class AdmissionError(ValueError):
    """A frozen design, provenance, accuracy, or timing gate failed."""


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"nonfinite JSON token {token!r} is forbidden")


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=_reject_nonfinite)
    if not isinstance(value, dict):
        raise AdmissionError(f"{path} must contain a JSON object")
    return dict(value)


def _read_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=_reject_nonfinite)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise AdmissionError(f"{path} must contain a list of JSON objects")
    return [dict(row) for row in value]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AdmissionError(f"{name} must be finite") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise AdmissionError(f"{name} must be {qualifier}")
    return result


def _exact(actual: object, expected: object, name: str) -> None:
    if isinstance(expected, bool):
        if type(actual) is not bool or actual is not expected:
            raise AdmissionError(f"{name} differs from the frozen value {expected!r}")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        value = _finite(actual, name)
        if not math.isclose(value, float(expected), rel_tol=1.0e-14, abs_tol=0.0):
            raise AdmissionError(f"{name} differs from the frozen value {expected!r}")
    elif str(actual) != str(expected):
        raise AdmissionError(f"{name} differs from the frozen value {expected!r}")


def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        if "=" not in line:
            raise AdmissionError(f"{path}:{number} is not key=value metadata")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise AdmissionError(f"{path}:{number} has an empty or duplicate key")
        values[key] = value
    return values


def _matrix_rows(
    matrix: Path, *, tier: str, contract_series: dict[str, Any]
) -> list[dict[str, str]]:
    with matrix.open(newline="", encoding="utf-8") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if row.get("experiment_id") == "EXP-SCALE-001" and row.get("tier") == tier
        ]
    points = {
        (int(point["nodes"]), int(point["ranks"]), str(point["partition"]))
        for point in contract_series["rank_node_points"]
    }
    observed = {
        (int(row["nodes"]), int(row["total_ranks"]), row["partition"]) for row in rows
    }
    if len(rows) != len(points) or observed != points:
        raise AdmissionError(
            f"{tier} matrix points differ from the frozen rank/node design"
        )
    if len({row["case_id"] for row in rows}) != len(rows):
        raise AdmissionError(f"{tier} contains duplicate case IDs")
    for row in rows:
        exact_matrix = {
            "experiment_id": "EXP-SCALE-001",
            "tier": tier,
            "runner": contract_series["runner"],
            "optional": "1" if contract_series["optional"] else "0",
            "ranks_per_node": "128",
            "repetitions": str(contract_series["repetitions_per_point"]),
            "warmups": "0",
        }
        exact_matrix.update(contract_series["matrix_scientific_fields"])
        for key, expected in exact_matrix.items():
            _exact(row.get(key), expected, f"matrix {row['case_id']} {key}")
        if int(row["total_ranks"]) != int(row["nodes"]) * int(row["ranks_per_node"]):
            raise AdmissionError(f"matrix {row['case_id']} has inconsistent resources")
    return sorted(rows, key=lambda row: int(row["nodes"]))


def _submission_jobs(campaign_root: Path, expected_cases: set[str]) -> dict[str, str]:
    ledger = campaign_root / "submitted_jobs.jsonl"
    if not ledger.is_file():
        raise AdmissionError("submitted campaign lacks submitted_jobs.jsonl")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_nonfinite)
        except json.JSONDecodeError as exc:
            raise AdmissionError(f"submission ledger line {number} is invalid") from exc
        if not isinstance(row, dict):
            raise AdmissionError(f"submission ledger line {number} is not an object")
        records.append(dict(row))
    if {str(row.get("case_id", "")) for row in records} != expected_cases:
        raise AdmissionError("submission ledger case set differs from the scale series")
    jobs: dict[str, str] = {}
    for row in records:
        case_id = str(row["case_id"])
        if int(row.get("returncode", 1)) != 0:
            raise AdmissionError(f"submission failed for {case_id}")
        match = _SUBMITTED_JOB.fullmatch(str(row.get("stdout", "")).strip())
        if match is None:
            raise AdmissionError(f"submission ledger lacks an exact job ID for {case_id}")
        jobs[case_id] = match.group(1)
    return jobs


def _validate_campaign(
    campaign_root: Path,
    *,
    matrix: Path,
    tier: str,
    rows: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, str]]:
    manifest = _read_object(campaign_root / "prepared_manifest.json")
    if manifest.get("status") != "submitted" or manifest.get("test_only_commands") is not False:
        raise AdmissionError("analysis requires a completed real-submission manifest")
    if set(manifest.get("selected_experiments") or []) != {"EXP-SCALE-001"}:
        raise AdmissionError("campaign is not isolated to EXP-SCALE-001")
    if set(manifest.get("selected_tiers") or []) != {tier}:
        raise AdmissionError(
            "required HE and optional P3D must be submitted and analyzed as separate tranches"
        )
    if int(manifest.get("case_count", -1)) != len(rows):
        raise AdmissionError("submitted manifest case count is inconsistent")
    if manifest.get("cluster") != "Karolina CPU":
        raise AdmissionError("submitted manifest does not identify Karolina CPU")
    if manifest.get("account") != "fta-26-40" or str(manifest.get("qos")) != "3571_6328":
        raise AdmissionError("submitted manifest account/QOS differs from the frozen contract")
    if str(manifest.get("matrix_sha256", "")) != _sha256(matrix):
        raise AdmissionError("submitted manifest matrix hash differs from the local matrix")
    commit = str(manifest.get("source_commit", ""))
    if manifest.get("source_dirty") is not False or _COMMIT.fullmatch(commit) is None:
        raise AdmissionError("submitted campaign does not prove a clean source commit")
    jobs = _submission_jobs(campaign_root, {row["case_id"] for row in rows})
    return manifest, jobs


def _validate_accounting(
    path: Path,
    *,
    job_id: str,
    row: dict[str, str],
    gates: dict[str, Any],
) -> dict[str, Any]:
    payload = _read_object(path)
    _exact(payload.get("schema_id"), gates["slurm_accounting_schema_id"], "accounting schema")
    _exact(
        payload.get("schema_version"),
        gates["slurm_accounting_schema_version"],
        "accounting schema version",
    )
    _exact(payload.get("job_id"), job_id, "accounting job ID")
    source = dict(payload.get("source") or {})
    if source.get("mode") not in {"offline_file", "explicit_live_query"}:
        raise AdmissionError("accounting source mode is absent or unknown")
    raw_accounting = source.get("raw_parsable2")
    if not isinstance(raw_accounting, str) or not raw_accounting:
        raise AdmissionError("accounting source does not retain the raw parsable2 evidence")
    raw_bytes = raw_accounting.encode("utf-8")
    if str(source.get("sha256", "")) != hashlib.sha256(raw_bytes).hexdigest():
        raise AdmissionError("accounting raw evidence does not match its SHA-256 digest")
    _exact(source.get("byte_count"), len(raw_bytes), "accounting raw byte count")
    allocation = dict(payload.get("allocation") or {})
    _exact(allocation.get("job_id_raw"), job_id, "allocation job ID")
    _exact(str(allocation.get("cluster", "")).lower(), gates["slurm_cluster"], "cluster")
    _exact(allocation.get("account"), gates["slurm_account"], "account")
    _exact(allocation.get("qos"), gates["slurm_qos"], "QOS")
    _exact(allocation.get("partition"), row["partition"], "partition")
    _exact(allocation.get("state"), gates["completed_state"], "Slurm state")
    _exact(allocation.get("exit_code"), gates["exit_code"], "Slurm exit code")
    _exact(allocation.get("alloc_nodes"), int(row["nodes"]), "allocated nodes")
    _exact(allocation.get("alloc_cpus"), int(row["total_ranks"]), "allocated CPUs")
    elapsed = _finite(allocation.get("elapsed_raw_s"), "allocation elapsed", positive=True)
    derived = dict(payload.get("derived") or {})
    _exact(
        derived.get("allocated_node_seconds"),
        int(round(elapsed)) * int(row["nodes"]),
        "allocated node seconds",
    )
    _exact(
        derived.get("allocated_cpu_seconds"),
        int(round(elapsed)) * int(row["total_ranks"]),
        "allocated CPU seconds",
    )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "elapsed_raw_s": elapsed,
        "allocated_node_seconds": int(derived["allocated_node_seconds"]),
        "maximum_step_rss_bytes": derived.get("maximum_step_rss_bytes"),
        "maximum_step_vm_size_bytes": derived.get("maximum_step_vm_size_bytes"),
    }


def _validate_timing(
    payload: dict[str, Any],
    *,
    ranks: int,
    gates: dict[str, Any],
    required_phases: list[str],
) -> tuple[dict[str, float], dict[str, list[float]]]:
    publication_timing = payload.get("publication_timing")
    if isinstance(publication_timing, dict):
        timing = dict(publication_timing)
        _exact(timing.get("schema_id"), gates["timing_schema_id"], "timing schema")
        _exact(timing.get("schema_version"), gates["timing_schema_version"], "timing version")
        _exact(timing.get("reduction"), gates["timing_reduction"], "timing reduction")
        _exact(timing.get("rank_count"), ranks, "timing rank count")
        if timing.get("measured_region_excludes_reporting_collective") is not True:
            raise AdmissionError("timing does not exclude its reporting collective")
        phases = dict(timing.get("phases") or {})
    elif required_phases == ["total"]:
        # The optional P3D runner predates the multi-phase envelope but already
        # retains the exact collective and every rank-local total.  This narrow
        # adapter is not available to the required HE series.
        _exact(
            payload.get("total_time_reduction"),
            gates["timing_reduction"],
            "P3D total timing reduction",
        )
        phases = {
            "total": {
                "collective_max_s": payload.get("total_time"),
                "per_rank_s": payload.get("total_time_by_rank_s"),
            }
        }
    else:
        raise AdmissionError("required multi-phase publication timing is missing")
    maxima: dict[str, float] = {}
    vectors: dict[str, list[float]] = {}
    for phase in required_phases:
        evidence = dict(phases.get(phase) or {})
        raw = evidence.get("per_rank_s")
        if not isinstance(raw, list) or len(raw) != ranks:
            raise AdmissionError(f"{phase} timing lacks one value per rank")
        values = [_finite(value, f"{phase} rank timing", positive=True) for value in raw]
        declared = _finite(
            evidence.get("collective_max_s"), f"{phase} collective maximum", positive=True
        )
        recomputed = max(values)
        if not math.isclose(declared, recomputed, rel_tol=1.0e-12, abs_tol=0.0):
            raise AdmissionError(f"{phase} collective maximum is not proved by rank values")
        maxima[phase] = declared
        vectors[phase] = values
    if {"setup", "first_step", "solve", "total"}.issubset(vectors):
        for rank, (setup, first_step, solve, total) in enumerate(
            zip(
                vectors["setup"],
                vectors["first_step"],
                vectors["solve"],
                vectors["total"],
                strict=True,
            )
        ):
            tolerance = 1.0e-12 * max(total, 1.0)
            if first_step > solve + tolerance:
                raise AdmissionError(
                    f"rank {rank} first-step time exceeds its enclosing solve time"
                )
            if setup + solve > total + tolerance:
                raise AdmissionError(
                    f"rank {rank} setup plus solve time exceeds its total time"
                )
    return maxima, vectors


def _state_payload(path: Path, *, kind: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"coords_ref", "displacement", "tetrahedra", "energy"}
        if not required.issubset(archive.files):
            raise AdmissionError(f"{path} lacks required state arrays {sorted(required)}")
        coords = np.asarray(archive["coords_ref"], dtype=np.float64)
        displacement = np.asarray(archive["displacement"], dtype=np.float64)
        cells = np.asarray(archive["tetrahedra"], dtype=np.int64)
        energy = _finite(np.asarray(archive["energy"]).reshape(()), "state energy")
    if coords.ndim != 2 or coords.shape[1] != 3 or displacement.shape != coords.shape:
        raise AdmissionError(f"{path} has inconsistent three-dimensional state arrays")
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise AdmissionError(f"{path} has invalid tetrahedron connectivity")
    if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(displacement)):
        raise AdmissionError(f"{path} state contains nonfinite values")
    point_norms = np.linalg.norm(displacement, axis=1)
    return {
        "kind": kind,
        "coords": coords,
        "cells": cells,
        "state": displacement,
        "energy": energy,
        "observables": {
            "displacement_l2": float(np.linalg.norm(displacement)),
            "maximum_displacement": float(np.max(point_norms, initial=0.0)),
        },
        "path": str(path),
        "sha256": _sha256(path),
    }


def _result_payload(wrapper: dict[str, Any], *, series_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if series_name == "required_he":
        case = wrapper.get("case")
        result = wrapper.get("result")
        if not isinstance(case, dict) or not isinstance(result, dict):
            raise AdmissionError("HE output must contain case and result objects")
        return dict(case), dict(result)
    return {}, wrapper


def _validate_accuracy(
    result: dict[str, Any], *, series_name: str, gates: dict[str, Any]
) -> tuple[float, int, dict[str, Any]]:
    if series_name == "required_he":
        steps = result.get("steps")
        if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], dict):
            raise AdmissionError("HE first-step output must contain exactly one step")
        step = dict(steps[0])
        if step.get("success") is not True:
            raise AdmissionError("HE first step did not report solver success")
        if gates["primary_attempt_only"] and step.get("attempt") != "primary":
            raise AdmissionError("a repaired HE attempt is not fixed-policy timing evidence")
        convergence = dict(step.get("convergence") or {})
        if gates["terminal_accuracy_gate_required"] and convergence.get(
            "dual_residual_gate_pass"
        ) is not True:
            raise AdmissionError("HE endpoint did not pass its terminal residual gate")
        return (
            _finite(step.get("energy"), "HE terminal energy"),
            int(step.get("nit", -1)),
            {
                "dual_residual_norm": _finite(
                    convergence.get("dual_residual_norm"), "HE dual residual"
                ),
                "relative_correction": _finite(
                    convergence.get("relative_correction"), "HE relative correction"
                ),
            },
        )
    if result.get("solver_success") is not True and result.get("status") != "completed":
        raise AdmissionError("P3D solve did not report success")
    convergence = dict(result.get("nonlinear_convergence") or {})
    if gates["terminal_accuracy_gate_required"] and dict(
        convergence.get("residual_gate") or {}
    ).get("passed") is not True:
        raise AdmissionError("P3D endpoint did not pass its terminal residual gate")
    return (
        _finite(result.get("energy"), "P3D terminal energy"),
        int(result.get("nit", -1)),
        {"residual_gate_passed": True},
    )


def _compare_endpoint(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    reference_energy: float,
    candidate_energy: float,
    reference_iterations: int,
    candidate_iterations: int,
    gates: dict[str, Any],
) -> dict[str, Any]:
    if not np.array_equal(reference["coords"], candidate["coords"]):
        raise AdmissionError("endpoint reference coordinates differ")
    if not np.array_equal(reference["cells"], candidate["cells"]):
        raise AdmissionError("endpoint topology differs")
    left = np.asarray(reference["state"], dtype=np.float64)
    right = np.asarray(candidate["state"], dtype=np.float64)
    delta = right - left
    relative = float(
        np.linalg.norm(delta)
        / max(np.linalg.norm(left), np.linalg.norm(right), np.finfo(float).tiny)
    )
    maximum = float(np.max(np.abs(delta), initial=0.0))
    if relative > float(gates["state_relative_l2_max"]):
        raise AdmissionError("endpoint state relative-L2 gate failed")
    if maximum > float(gates["state_max_absolute_max"]):
        raise AdmissionError("endpoint state maximum-absolute gate failed")
    energy_absolute = abs(candidate_energy - reference_energy)
    energy_relative = energy_absolute / max(
        abs(candidate_energy), abs(reference_energy), np.finfo(float).tiny
    )
    if energy_absolute > float(gates["energy_absolute_max"]) + float(
        gates["energy_relative_max"]
    ) * max(abs(candidate_energy), abs(reference_energy)):
        raise AdmissionError("endpoint energy gate failed")
    observable_defects: dict[str, dict[str, float]] = {}
    for key, left_value in reference["observables"].items():
        right_value = float(candidate["observables"][key])
        absolute = abs(float(left_value) - right_value)
        relative_observable = absolute / max(
            abs(float(left_value)), abs(right_value), np.finfo(float).tiny
        )
        if absolute > float(gates["observable_absolute_max"]) + float(
            gates["observable_relative_max"]
        ) * max(abs(float(left_value)), abs(right_value)):
            raise AdmissionError(f"endpoint observable {key} gate failed")
        observable_defects[key] = {"absolute": absolute, "relative": relative_observable}
    if gates["nonlinear_iterations_exact"] and candidate_iterations != reference_iterations:
        raise AdmissionError("endpoint nonlinear iteration count differs from one-node reference")
    return {
        "state_relative_l2": relative,
        "state_max_absolute": maximum,
        "energy_absolute": energy_absolute,
        "energy_relative": energy_relative,
        "observable_defects": observable_defects,
        "iterations_equal": candidate_iterations == reference_iterations,
    }


def _bootstrap_interval(values: np.ndarray, *, seed: int, resamples: int, level: float) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, values.size), replace=True)
    estimates = np.median(draws, axis=1)
    alpha = (1.0 - level) / 2.0
    return [float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))]


def _scaling_statistics(
    values_by_nodes: dict[int, list[float]], *, uncertainty: dict[str, Any]
) -> list[dict[str, Any]]:
    seed = int(uncertainty["bootstrap_seed"])
    resamples = int(uncertainty["bootstrap_resamples"])
    level = float(uncertainty["confidence_level"])
    baseline = np.asarray(values_by_nodes[1], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for nodes, values_raw in sorted(values_by_nodes.items()):
        values = np.asarray(values_raw, dtype=np.float64)
        median = float(np.median(values))
        rng = np.random.default_rng(seed + 1009 * nodes)
        base_draw = rng.choice(baseline, size=(resamples, baseline.size), replace=True)
        point_draw = rng.choice(values, size=(resamples, values.size), replace=True)
        speedup_draw = np.median(base_draw, axis=1) / np.median(point_draw, axis=1)
        efficiency_draw = speedup_draw / float(nodes)
        alpha = (1.0 - level) / 2.0
        speedup = float(np.median(baseline) / median)
        rows.append(
            {
                "nodes": nodes,
                "repetitions": int(values.size),
                "collective_max_seconds": values.tolist(),
                "median_seconds": median,
                "median_confidence_interval": _bootstrap_interval(
                    values,
                    seed=seed + nodes,
                    resamples=resamples,
                    level=level,
                ),
                "speedup": speedup,
                "speedup_confidence_interval": [
                    float(np.quantile(speedup_draw, alpha)),
                    float(np.quantile(speedup_draw, 1.0 - alpha)),
                ],
                "efficiency": speedup / float(nodes),
                "efficiency_confidence_interval": [
                    float(np.quantile(efficiency_draw, alpha)),
                    float(np.quantile(efficiency_draw, 1.0 - alpha)),
                ],
                "efficiency_basis": "nodes_relative_to_one_node",
            }
        )
    return rows


def analyze(
    *,
    campaign_root: Path,
    matrix: Path = DEFAULT_MATRIX,
    contract_path: Path = DEFAULT_CONTRACT,
    series_name: str = "required_he",
) -> dict[str, Any]:
    campaign_root = Path(campaign_root).resolve()
    matrix = Path(matrix).resolve()
    contract_path = Path(contract_path).resolve()
    contract = _read_object(contract_path)
    _exact(contract.get("schema_version"), 1, "analysis contract version")
    if series_name not in {"required_he", "optional_p3d"}:
        raise AdmissionError("series must be required_he or optional_p3d")
    series = dict(contract["series"][series_name])
    gates = dict(contract["admission_gates"])
    tier = str(series["tier"])
    rows = _matrix_rows(matrix, tier=tier, contract_series=series)
    manifest, job_ids = _validate_campaign(
        campaign_root, matrix=matrix, tier=tier, rows=rows
    )
    common_commit = str(manifest["source_commit"])

    violations: list[dict[str, str]] = []
    repetitions: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    reference_energy = 0.0
    reference_iterations = -1
    policy_identity: str | None = None

    for row in rows:
        case_id = row["case_id"]
        job_id = job_ids[case_id]
        case_job = campaign_root / "cases" / case_id / f"job_{job_id}"
        batch_job = campaign_root / "jobs" / case_id / f"job_{job_id}"
        try:
            observed_row = _read_object(case_job / "matrix_row.json")
            if observed_row != row:
                raise AdmissionError("executed matrix row differs from the reviewed row")
            metadata = _parse_env(batch_job / "job_metadata.env")
            exact_metadata = {
                "case_id": case_id,
                "job_id": job_id,
                "account": gates["slurm_account"],
                "qos": gates["slurm_qos"],
                "partition": row["partition"],
                "cluster": gates["slurm_cluster"],
                "nodes": row["nodes"],
                "ntasks": row["total_ranks"],
                "matrix_sha256": _sha256(matrix),
                "git_commit": common_commit,
                "git_dirty": "false",
                "allocation_revalidated": "YES",
                "account_qos_revalidated": "YES",
            }
            for key, expected in exact_metadata.items():
                _exact(metadata.get(key), expected, f"{case_id} job metadata {key}")
            if not str(metadata.get("allocation_valid_until", "")).strip():
                raise AdmissionError("job metadata lacks the revalidated allocation end date")
            environment = batch_job / "environment.txt"
            if not environment.is_file() or environment.stat().st_size == 0:
                raise AdmissionError("job environment capture is missing or empty")
            accounting = _validate_accounting(
                batch_job / "sacct_final.json",
                job_id=job_id,
                row=row,
                gates=gates,
            )
            stdout = campaign_root / "slurm" / f"{case_id}-{job_id}.out"
            stderr = campaign_root / "slurm" / f"{case_id}-{job_id}.err"
            if not stdout.is_file() or not stderr.is_file():
                raise AdmissionError("Slurm stdout/stderr evidence is missing")
            run_records = _read_list(case_job / "run_records.json")
            expected_repetitions = int(series["repetitions_per_point"])
            if len(run_records) != expected_repetitions:
                raise AdmissionError("case lacks five independent process records")
            if {(record.get("kind"), int(record.get("index", -1))) for record in run_records} != {
                ("measure", index) for index in range(1, expected_repetitions + 1)
            }:
                raise AdmissionError("process repetition identities are incomplete or duplicated")
            commands: set[str] = set()
            for process in run_records:
                index = int(process["index"])
                if int(process.get("returncode", 1)) != int(
                    gates["successful_process_returncode"]
                ) or process.get("timed_out") is not False:
                    raise AdmissionError(f"measure {index} process failed or timed out")
                run_dir = case_job / f"measure_{index:02d}"
                command_path = run_dir / "command.txt"
                command = command_path.read_text(encoding="utf-8").strip()
                if command != str(process.get("command", "")):
                    raise AdmissionError(f"measure {index} command record differs")
                if command in commands:
                    raise AdmissionError("process command identities are not distinct")
                commands.add(command)
                wrapper = _read_object(run_dir / "output.json")
                case, result = _result_payload(wrapper, series_name=series_name)
                for key, expected in dict(series.get("output_case_fields") or {}).items():
                    _exact(case.get(key), expected, f"measure {index} case {key}")
                for key, expected in dict(series.get("output_result_fields") or {}).items():
                    _exact(result.get(key), expected, f"measure {index} result {key}")
                if series_name == "required_he":
                    _exact(
                        dict(result.get("metadata") or {}).get("nprocs"),
                        int(row["total_ranks"]),
                        "HE result rank count",
                    )
                else:
                    _exact(result.get("ranks"), int(row["total_ranks"]), "P3D result rank count")
                    result_git = dict(result.get("git") or {})
                    _exact(result_git.get("commit"), common_commit, "P3D result commit")
                    if result_git.get("dirty") is not False:
                        raise AdmissionError("P3D result was produced by a dirty worktree")
                    result_job = dict(result.get("job_metadata") or {})
                    _exact(result_job.get("slurm_job_id"), job_id, "P3D result job ID")
                    _exact(
                        str(result_job.get("slurm_cluster_name", "")).lower(),
                        gates["slurm_cluster"],
                        "P3D result cluster",
                    )
                policy_fields = {
                    "case": {
                        key: case.get(key)
                        for key in dict(series.get("output_case_fields") or {})
                    },
                    "result": {
                        key: result.get(key)
                        for key in dict(series.get("output_result_fields") or {})
                    },
                }
                current_policy = hashlib.sha256(
                    json.dumps(
                        policy_fields,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                if policy_identity is None:
                    policy_identity = current_policy
                elif policy_identity != current_policy:
                    raise AdmissionError("scientific policy differs between repetitions or points")
                maxima, vectors = _validate_timing(
                    result,
                    ranks=int(row["total_ranks"]),
                    gates=gates,
                    required_phases=[str(value) for value in series["required_timing_phases"]],
                )
                energy, iterations, accuracy = _validate_accuracy(
                    result, series_name=series_name, gates=gates
                )
                state = _state_payload(
                    run_dir / "state.npz", kind=str(series["state_kind"])
                )
                if not math.isclose(
                    energy, float(state["energy"]), rel_tol=1.0e-12, abs_tol=1.0e-12
                ):
                    raise AdmissionError("output energy disagrees with the exported state")
                if reference is None:
                    if int(row["nodes"]) != 1 or index != 1:
                        raise AdmissionError("endpoint reference is not one-node repetition one")
                    reference = state
                    reference_energy = energy
                    reference_iterations = iterations
                endpoint = _compare_endpoint(
                    reference,
                    state,
                    reference_energy=reference_energy,
                    candidate_energy=energy,
                    reference_iterations=reference_iterations,
                    candidate_iterations=iterations,
                    gates=gates,
                )
                repetitions.append(
                    {
                        "case_id": case_id,
                        "job_id": job_id,
                        "nodes": int(row["nodes"]),
                        "ranks": int(row["total_ranks"]),
                        "repetition": index,
                        "timing_collective_max_s": maxima,
                        "timing_rank_vectors_sha256": {
                            phase: hashlib.sha256(
                                np.asarray(values, dtype=np.float64).tobytes()
                            ).hexdigest()
                            for phase, values in vectors.items()
                        },
                        "energy": energy,
                        "nonlinear_iterations": iterations,
                        "accuracy": accuracy,
                        "endpoint_equivalence": endpoint,
                        "output_sha256": _sha256(run_dir / "output.json"),
                        "state_sha256": state["sha256"],
                    }
                )
            for record in repetitions:
                if record["case_id"] == case_id:
                    record["accounting"] = accounting
                    record["environment_sha256"] = _sha256(environment)
                    record["slurm_stdout_sha256"] = _sha256(stdout)
                    record["slurm_stderr_sha256"] = _sha256(stderr)
        except (AdmissionError, FileNotFoundError, OSError, KeyError, TypeError, ValueError) as exc:
            violations.append({"case_id": case_id, "reason": str(exc)})

    expected_records = len(rows) * int(series["repetitions_per_point"])
    complete = not violations and len(repetitions) == expected_records
    response_phase = str(series["timing_response_phase"])
    statistics: list[dict[str, Any]] = []
    if complete:
        values_by_nodes: dict[int, list[float]] = {}
        for record in repetitions:
            values_by_nodes.setdefault(int(record["nodes"]), []).append(
                float(record["timing_collective_max_s"][response_phase])
            )
        if any(
            len(values) != int(series["repetitions_per_point"])
            for values in values_by_nodes.values()
        ):
            raise AdmissionError("admitted timing does not contain five values per point")
        statistics = _scaling_statistics(
            values_by_nodes, uncertainty=dict(contract["uncertainty"])
        )

    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "EXP-SCALE-001",
        "series": series_name,
        "tier": tier,
        "status": (
            "admitted_fixed_policy_viability"
            if complete
            else "invalid_no_timing_claim"
        ),
        "timing_claim_released": complete,
        "must_not_merge_with": (
            "optional_p3d" if series_name == "required_he" else "required_he"
        ),
        "source_commit": common_commit,
        "source_dirty": False,
        "matrix_path": str(matrix),
        "matrix_sha256": _sha256(matrix),
        "contract_path": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "campaign_manifest_path": str(campaign_root / "prepared_manifest.json"),
        "campaign_manifest_sha256": _sha256(campaign_root / "prepared_manifest.json"),
        "expected_repetitions": expected_records,
        "admitted_repetitions": len(repetitions) if complete else 0,
        "violations": violations,
        "response_phase": response_phase,
        "repetitions": repetitions,
        "scaling_statistics": statistics,
        "uncertainty": dict(contract["uncertainty"]),
        "scalable_adjective_released": False,
    }


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# EXP-SCALE-001: {result['series']}",
        "",
        f"- Status: `{result['status']}`",
        f"- Timing claim released: `{str(result['timing_claim_released']).lower()}`",
        f"- Source commit: `{result['source_commit']}`",
        f"- Admitted repetitions: {result['admitted_repetitions']}/{result['expected_repetitions']}",
        f"- Response phase: `{result['response_phase']}`",
        f"- Separate from: `{result['must_not_merge_with']}`",
        "",
    ]
    if result["violations"]:
        lines.extend(["## Violations", ""])
        lines.extend(
            f"- `{row['case_id']}`: {row['reason']}" for row in result["violations"]
        )
        lines.append("")
    if result["scaling_statistics"]:
        lines.extend(
            [
                "## Admitted scaling",
                "",
                "| Nodes | Reps | Median (s) | 95% interval | Speedup | Efficiency |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["scaling_statistics"]:
            interval = row["median_confidence_interval"]
            lines.append(
                f"| {row['nodes']} | {row['repetitions']} | {row['median_seconds']:.6g} "
                f"| [{interval[0]:.6g}, {interval[1]:.6g}] | {row['speedup']:.4g} "
                f"| {row['efficiency']:.4g} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--series", choices=("required_he", "optional_p3d"), default="required_he"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = analyze(
        campaign_root=args.campaign_root,
        matrix=args.matrix,
        contract_path=args.contract,
        series_name=args.series,
    )
    destination = Path(args.output).resolve()
    atomic_write_json(destination, result)
    if args.report is not None:
        _write_report(Path(args.report).resolve(), result)
    print(destination)
    if not result["timing_claim_released"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

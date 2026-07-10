#!/usr/bin/env python3
"""Validate and prepare the publication-revision Karolina Slurm matrix.

This program is non-submitting unless ``--execute`` is supplied.  Execution is
still guarded by explicit, current allocation revalidation environment values.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import shlex
import subprocess
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MATRIX = Path(__file__).with_name("campaign_matrix.csv")
ROUTE_ANALYSIS_CONTRACT = REPO_ROOT / "paper/protocols/EXP-ROUTE-001-analysis-contract.json"
SBATCH_RUNNER = Path(__file__).with_name("run_revision_case.sbatch")
STATIC_CAMPAIGN_MANIFEST = Path(__file__).with_name("campaign_manifest.yaml")
RELEASE_AUTHORIZATION_SCHEMA = (
    REPO_ROOT / "paper/protocols/human-release-authorization-v1.schema.json"
)
RELEASE_AUTHORIZATION_EXAMPLE = (
    REPO_ROOT / "paper/protocols/human-release-authorization-v1.example.json"
)
SOURCE_FREEZE_SCHEMA_ID = "fenics-nonlinear-energies.queued-source-freeze"
SOURCE_FREEZE_SCHEMA_VERSION = 1
ACCOUNT = "fta-26-40"
QOS = "3571_6328"
ALLOCATION_RECORD_END = "2026-07-11"
MAX_RANKS_PER_NODE = 128
DEFAULT_NODE_HOUR_GUARD = 100.0
PROTOCOLS = {
    "EXP-ROUTE-001": "paper/protocols/EXP-ROUTE-001.md",
    "EXP-DISC-001": "paper/protocols/EXP-DISC-001.md",
    "EXP-SCALE-001": "paper/protocols/EXP-SCALE-001.md",
}
REVIEWED_SOURCES = {
    "analysis_contract": ROUTE_ANALYSIS_CONTRACT,
    "route_protocol": REPO_ROOT / "paper/protocols/EXP-ROUTE-001.md",
    "discretization_protocol": REPO_ROOT / "paper/protocols/EXP-DISC-001.md",
    "scaling_protocol": REPO_ROOT / "paper/protocols/EXP-SCALE-001.md",
    "scaling_analysis_contract": REPO_ROOT
    / "paper/protocols/EXP-SCALE-001-analysis-contract.json",
    "cost_model_analyzer": REPO_ROOT
    / "experiments/analysis/analyze_plasticity3d_route_cost_model.py",
    "endpoint_analyzer": REPO_ROOT
    / "experiments/analysis/analyze_plasticity3d_route_endpoints.py",
    "route_tranche_aggregator": REPO_ROOT
    / "experiments/analysis/aggregate_route_tranche_manifests.py",
    "discretization_analyzer": REPO_ROOT
    / "experiments/analysis/analyze_plasticity3d_discretization.py",
    "scaling_analyzer": REPO_ROOT / "experiments/analysis/analyze_exp_scale_001.py",
    "slurm_accounting_collector": REPO_ROOT
    / "experiments/analysis/collect_slurm_accounting.py",
    "fixed_state_runner": REPO_ROOT
    / "experiments/runners/run_plasticity3d_fixed_state_route_screen.py",
    "factor_runner": REPO_ROOT
    / "experiments/runners/run_route_factor_microbenchmarks.py",
    "backend_mix_runner": REPO_ROOT
    / "experiments/runners/run_plasticity3d_backend_mix_case.py",
    "quadrature_runner": REPO_ROOT
    / "experiments/runners/run_plasticity3d_fixed_state_quadrature.py",
    "executor": Path(__file__).with_name("execute_case.py"),
    "preparer": Path(__file__),
    "offline_preflight": Path(__file__).with_name("preflight_prepared_campaign.py"),
    "batch_runner": SBATCH_RUNNER,
    "submitter": Path(__file__).with_name("submit_prepared_campaigns.sh"),
    "state_export": REPO_ROOT / "src/core/benchmark/state_export.py",
    "fixed_state_support": REPO_ROOT
    / "src/problems/slope_stability_3d/support/fixed_state.py",
    "quadrature_support": REPO_ROOT
    / "src/problems/slope_stability_3d/support/mesh.py",
    "release_authorization_schema": RELEASE_AUTHORIZATION_SCHEMA,
    "release_authorization_example": RELEASE_AUTHORIZATION_EXAMPLE,
}

TIER_B_TIERS = frozenset({"full_solve_confirmation", "low_order_confirmation"})
HE_SCALING_TIER = "fixed_policy_he_l5"
P3D_SCALING_TIER = "optional_fixed_policy_p3d"
DISC_RELEASE_STAGES = (
    "smoke",
    "quadrature",
    "mesh",
    "mesh_quadrature",
    "tolerance",
)
DISC_STAGE_CASE_COUNTS = {
    "smoke": 1,
    "quadrature": 2,
    "mesh": 1,
    "mesh_quadrature": 1,
    "tolerance": 1,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError as exc:
        raise RuntimeError(f"reviewed source is outside the repository: {resolved}") from exc


def _source_freeze_payload(
    *,
    matrix: Path,
    source_commit: str,
    reviewed_source_hashes: dict[str, str],
) -> dict[str, object]:
    if not _is_lower_hex(source_commit, 40):
        raise RuntimeError("source freeze requires a lowercase 40-character Git commit")
    if set(reviewed_source_hashes) != set(REVIEWED_SOURCES):
        raise RuntimeError("source freeze does not cover the complete reviewed source set")
    reviewed: dict[str, dict[str, str]] = {}
    for key, source in REVIEWED_SOURCES.items():
        digest = reviewed_source_hashes[key]
        if not _is_lower_hex(digest, 64):
            raise RuntimeError(f"reviewed source {key} has an invalid SHA-256 value")
        reviewed[key] = {
            "path": _repo_relative(source),
            "sha256": digest,
        }
    return {
        "schema_id": SOURCE_FREEZE_SCHEMA_ID,
        "schema_version": SOURCE_FREEZE_SCHEMA_VERSION,
        "source_commit": source_commit,
        "matrix": {
            "path": _repo_relative(matrix),
            "sha256": _sha256(matrix),
        },
        "reviewed_sources": reviewed,
    }


def _validate_source_freeze_payload(
    payload: dict[str, object],
    *,
    matrix: Path,
    source_commit: str,
) -> None:
    expected_keys = {
        "schema_id",
        "schema_version",
        "source_commit",
        "matrix",
        "reviewed_sources",
    }
    if set(payload) != expected_keys:
        raise RuntimeError("queued source freeze has an unexpected top-level shape")
    if (
        payload.get("schema_id") != SOURCE_FREEZE_SCHEMA_ID
        or payload.get("schema_version") != SOURCE_FREEZE_SCHEMA_VERSION
        or payload.get("source_commit") != source_commit
    ):
        raise RuntimeError("queued source freeze identity or source commit changed")
    matrix_record = payload.get("matrix")
    if not isinstance(matrix_record, dict) or set(matrix_record) != {"path", "sha256"}:
        raise RuntimeError("queued source freeze has an invalid matrix record")
    if (
        matrix_record.get("path") != _repo_relative(matrix)
        or matrix_record.get("sha256") != _sha256(matrix)
    ):
        raise RuntimeError("queued source freeze matrix path or hash changed")
    reviewed = payload.get("reviewed_sources")
    if not isinstance(reviewed, dict) or set(reviewed) != set(REVIEWED_SOURCES):
        raise RuntimeError("queued source freeze reviewed-source set changed")
    for key, source in REVIEWED_SOURCES.items():
        record = reviewed.get(key)
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise RuntimeError(f"queued source freeze record {key} is malformed")
        if record.get("path") != _repo_relative(source):
            raise RuntimeError(f"queued source freeze path for {key} changed")
        if record.get("sha256") != _sha256(source):
            raise RuntimeError(f"queued source freeze hash for {key} is stale")


def _write_source_freeze(
    *,
    out_root: Path,
    matrix: Path,
    source_commit: str,
    reviewed_source_hashes: dict[str, str],
) -> dict[str, str]:
    payload = _source_freeze_payload(
        matrix=matrix,
        source_commit=source_commit,
        reviewed_source_hashes=reviewed_source_hashes,
    )
    _validate_source_freeze_payload(
        payload,
        matrix=matrix,
        source_commit=source_commit,
    )
    path = out_root / "reviewed_source_freeze.json"
    _atomic_write_json(path, payload)
    return {
        "schema_id": SOURCE_FREEZE_SCHEMA_ID,
        "path": path.name,
        "sha256": _sha256(path),
    }


def _reviewed_source_hashes(path: Path = STATIC_CAMPAIGN_MANIFEST) -> dict[str, str]:
    hashes: dict[str, str] = {}
    in_section = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line == "source_sha256:":
            in_section = True
            continue
        if not in_section:
            continue
        if raw_line and not raw_line.startswith("  "):
            break
        stripped = raw_line.strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = (part.strip() for part in stripped.split(":", 1))
        hashes[key] = value.strip('"\'')
    return hashes


def _validate_reviewed_sources() -> dict[str, str]:
    recorded = _reviewed_source_hashes()
    missing = sorted(set(REVIEWED_SOURCES) - set(recorded))
    extra = sorted(set(recorded) - set(REVIEWED_SOURCES))
    if missing or extra:
        raise RuntimeError(
            "static campaign source-hash map differs from the executable source set; "
            f"missing={missing}, extra={extra}"
        )
    mismatches = []
    for key, source in REVIEWED_SOURCES.items():
        if not source.is_file():
            mismatches.append(f"{key}:missing:{source}")
            continue
        actual = _sha256(source)
        if actual != recorded[key]:
            mismatches.append(f"{key}:{recorded[key]}!={actual}")
    if mismatches:
        raise RuntimeError(
            "campaign sources differ from campaign_manifest.yaml; review and refresh "
            "the static hashes before preparing any Slurm command: "
            + "; ".join(mismatches)
        )
    return recorded


def _git_metadata() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _time_seconds(value: str) -> int:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"time_limit must be HH:MM:SS, got {value!r}")
    hours, minutes, seconds = (int(part) for part in parts)
    if min(hours, minutes, seconds) < 0 or minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid time_limit {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def read_matrix(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("campaign matrix is empty")
    seen: set[str] = set()
    for row in rows:
        case_id = str(row["case_id"])
        if not case_id or case_id in seen:
            raise ValueError(f"case_id must be nonempty and unique: {case_id!r}")
        seen.add(case_id)
        nodes = int(row["nodes"])
        ranks_per_node = int(row["ranks_per_node"])
        total_ranks = int(row["total_ranks"])
        if nodes < 1 or not 1 <= ranks_per_node <= MAX_RANKS_PER_NODE:
            raise ValueError(f"invalid resource shape for {case_id}")
        if total_ranks != nodes * ranks_per_node:
            raise ValueError(f"total_ranks mismatch for {case_id}")
        expected_partition = "qcpu_exp" if nodes <= 2 else "qcpu"
        if row["partition"] != expected_partition:
            raise ValueError(
                f"{case_id} must use {expected_partition} for {nodes} Karolina node(s)"
            )
        seconds = _time_seconds(row["time_limit"])
        expected_hours = nodes * seconds / 3600.0
        if abs(float(row["estimated_node_hours"]) - expected_hours) > 1.0e-9:
            raise ValueError(f"estimated_node_hours mismatch for {case_id}")
        if int(row["repetitions"]) < 1 or int(row["warmups"]) < 0:
            raise ValueError(f"invalid repetition counts for {case_id}")
        if row["experiment_id"] not in PROTOCOLS:
            raise ValueError(f"unrecognized experiment_id for {case_id}")
        expected_metric = {
            "p3d_solve": "reference_elastic_energy",
            "p3d_solve_block": "reference_elastic_energy",
            "p3d_fixed_state": "not_applicable",
            "p3d_fixed_state_block": "not_applicable",
            "he_first_step": "problem_specific_existing",
            "route_factor_microbench": "not_applicable",
        }.get(row["runner"])
        if expected_metric is None:
            raise ValueError(f"unrecognized runner for {case_id}: {row['runner']!r}")
        if row.get("convergence_metric") != expected_metric:
            raise ValueError(
                f"{case_id} must use convergence_metric={expected_metric}; "
                f"got {row.get('convergence_metric')!r}"
            )
        if (
            row["experiment_id"] == "EXP-ROUTE-001"
            and row["tier"] in {"full_solve_confirmation", "low_order_confirmation"}
            and float(row["ksp_rtol"]) > 1.0e-8
        ):
            raise ValueError(
                f"{case_id} uses obsolete loose route-comparison KSP tolerance; "
                "Tier-B requires ksp_rtol <= 1e-8"
            )
    _validate_route_design(rows)
    return rows


def _validate_route_design(rows: list[dict[str, str]]) -> None:
    route_rows = [row for row in rows if row["experiment_id"] == "EXP-ROUTE-001"]
    screens = [row for row in route_rows if row["tier"] == "fixed_state_screen"]
    quadrature = [row for row in route_rows if row["tier"] == "factorized_quadrature"]
    factor_micro = [
        row for row in route_rows if row["tier"] == "factorized_microbenchmark"
    ]
    confirmations = [
        row
        for row in route_rows
        if row["tier"] in {"full_solve_confirmation", "low_order_confirmation"}
    ]
    configurations = {
        ("hetero_ssr_L1", 1, "tetra_1point"): 3,
        ("hetero_ssr_L1_2", 1, "tetra_1point"): 3,
        ("hetero_ssr_L1", 2, "tetra_11point"): 3,
        ("hetero_ssr_L1", 4, "tetra_24point"): 4,
    }
    expected: set[tuple[str, int, str, str, int, int]] = set()
    for (mesh, degree, rule), blocks in configurations.items():
        for state in ("elastic", "mixed"):
            for ranks in (1, 8, 32):
                for block in range(1, blocks + 1):
                    expected.add((mesh, degree, rule, state, ranks, block))
    actual = {
        (
            row["mesh_name"],
            int(row["element_degree"]),
            row["quadrature_rule"],
            row["state_label"],
            int(row["total_ranks"]),
            int(row["block_repetition"]),
        )
        for row in screens
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"EXP-ROUTE fixed-state design mismatch; missing={missing}, extra={extra}"
        )
    if any(
        (int(row["warmups"]), int(row["repetitions"])) != (1, 5)
        for row in screens
    ):
        raise ValueError("every fixed-state route row requires one warmup and five measures")
    for row in screens + quadrature:
        if row["runner"] != "p3d_fixed_state_block":
            raise ValueError("route screening must use paired fixed-state block runner")
        if row["route_order_policy"] != "seeded_balanced_cyclic_v1":
            raise ValueError("route screening must retain balanced cyclic order")
        if row["timing_reduction"] != "mpi_collective_max" or int(row["probe_count"]) < 4:
            raise ValueError("route screening requires collective-max timing and >=4 probes")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in screens + quadrature + confirmations:
        grouped.setdefault(row["comparison_id"], []).append(row)
    for comparison_id, group in grouped.items():
        orders = [row["route_order"].split("|") for row in group]
        routes = set(orders[0])
        if any(set(order) != routes for order in orders):
            raise ValueError(f"{comparison_id} changes its active route set")
        position_counts = {
            (route, position): sum(order[position] == route for order in orders)
            for route in routes
            for position in range(len(routes))
        }
        counts = list(position_counts.values())
        if max(counts) - min(counts) > 0:
            raise ValueError(f"{comparison_id} route positions are not exactly balanced")
        policy = str(group[0]["route_order_policy"])
        canonical = (
            ["element_ad", "colored_sfd", "constitutive_ad"]
            if len(routes) == 3
            else ["element_ad", "constitutive_ad"]
        )
        seeded = list(canonical)
        seed = int(hashlib.sha256(comparison_id.encode("utf-8")).hexdigest()[:16], 16)
        random.Random(seed).shuffle(seeded)
        for row in group:
            repetition = int(row["block_repetition"])
            if policy == "seeded_balanced_cyclic_v1":
                offset = (repetition - 1) % len(seeded)
                expected_order = seeded[offset:] + seeded[:offset]
            elif policy == "seeded_balanced_alternating_v1":
                expected_order = seeded if repetition % 2 == 1 else list(reversed(seeded))
            else:
                raise ValueError(f"{comparison_id} has an unknown route-order policy")
            if row["route_order"].split("|") != expected_order:
                raise ValueError(
                    f"{comparison_id} route order is not the frozen SHA-256-seeded order"
                )

    expected_quadrature = {
        (rule, state, block)
        for rule in ("tetra_1point", "tetra_24point", "tetra_duffy_125point")
        for state in ("elastic", "mixed")
        for block in (1, 2, 3)
    }
    actual_quadrature = {
        (row["quadrature_rule"], row["state_label"], int(row["block_repetition"]))
        for row in quadrature
    }
    if actual_quadrature != expected_quadrature:
        raise ValueError("factorized quadrature design changed")
    expected_factor_micro = {
        (ranks, block, "route_factor_microbench")
        for ranks in (1, 8, 32)
        for block in (1, 2, 3)
    }
    actual_factor_micro = {
        (int(row["total_ranks"]), int(row["block_repetition"]), row["runner"])
        for row in factor_micro
    }
    if actual_factor_micro != expected_factor_micro:
        raise ValueError("factorized component microbenchmark rank design changed")
    if len(confirmations) != 30 or any(row["optional"] != "1" for row in confirmations):
        raise ValueError("paired full-solve confirmations require 30 optional blocks")
    confirmation_shape = {
        (row["tier"], int(row["element_degree"]), int(row["total_ranks"]))
        for row in confirmations
    }
    if confirmation_shape != {
        ("full_solve_confirmation", 4, 8),
        ("full_solve_confirmation", 4, 32),
        ("low_order_confirmation", 1, 8),
    }:
        raise ValueError("full-solve high/low-order confirmation design changed")
    if any(
        row["runner"] != "p3d_solve_block"
        or row["route_order_policy"] != "seeded_balanced_alternating_v1"
        or row["timing_reduction"] != "mpi_collective_max"
        for row in confirmations
    ):
        raise ValueError("full-solve confirmation pairing policy changed")


def select_rows(
    rows: Iterable[dict[str, str]],
    *,
    experiments: set[str],
    include_optional: bool,
    only_optional: bool = False,
    tiers: set[str] | None = None,
) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        if experiments and row["experiment_id"] not in experiments:
            continue
        if tiers and row["tier"] not in tiers:
            continue
        if only_optional and row["optional"] != "1":
            continue
        if row["optional"] == "1" and not (include_optional or only_optional):
            continue
        selected.append(row)
    if not selected:
        raise ValueError("no campaign rows selected")
    return selected


def _validate_optional_tranche_scope(
    selected: list[dict[str, str]], *, only_optional: bool
) -> None:
    """Reject partial or mixed optional tranches even during preparation."""

    optional = [row for row in selected if row["optional"] == "1"]
    if not optional:
        return
    experiments = {row["experiment_id"] for row in optional}
    if len(experiments) != 1 or any(row["optional"] != "1" for row in selected):
        raise RuntimeError(
            "optional rows must be prepared as one isolated experiment tranche"
        )
    if not only_optional:
        raise RuntimeError(
            "optional rows require --only-optional so they cannot be mixed with required rows"
        )
    experiment = next(iter(experiments))
    tiers = {row["tier"] for row in selected}
    if experiment == "EXP-ROUTE-001":
        if tiers != TIER_B_TIERS or len(selected) != 30:
            raise RuntimeError(
                "optional EXP-ROUTE-001 must be the exact 30-row Tier-B scope: "
                "full_solve_confirmation plus low_order_confirmation"
            )
    elif experiment == "EXP-SCALE-001":
        if tiers != {P3D_SCALING_TIER} or len(selected) != 3:
            raise RuntimeError(
                "optional EXP-SCALE-001 must be the exact three-row Plasticity3D tranche"
            )
    else:
        raise RuntimeError(f"optional rows are not defined for {experiment}")


def _validate_real_submission_scope(
    args: argparse.Namespace, *, selected: list[dict[str, str]]
) -> None:
    """Enforce executable tranche boundaries before any scheduler command."""

    experiments = {row["experiment_id"] for row in selected}
    tiers = {row["tier"] for row in selected}
    if len(experiments) != 1 or not args.experiment:
        raise RuntimeError(
            "real submission requires exactly one explicit --experiment tranche"
        )
    if not args.tier:
        raise RuntimeError(
            "real submission requires explicit --tier selection; the all-required default "
            "is preparation-only"
        )
    experiment = next(iter(experiments))
    only_optional = bool(getattr(args, "only_optional", False))
    include_optional = bool(getattr(args, "include_optional", False))

    if experiment == "EXP-DISC-001":
        if len(tiers) != 1:
            raise RuntimeError("EXP-DISC-001 must be released one protocol stage at a time")
        tier = next(iter(tiers))
        expected_count = DISC_STAGE_CASE_COUNTS.get(tier)
        if expected_count is None or len(selected) != expected_count:
            raise RuntimeError("EXP-DISC-001 release stage has incomplete case coverage")
        if only_optional or include_optional:
            raise RuntimeError("EXP-DISC-001 has no optional release scope")
    elif experiment == "EXP-SCALE-001":
        if tiers == {HE_SCALING_TIER}:
            if only_optional or include_optional or any(
                row["optional"] != "0" for row in selected
            ):
                raise RuntimeError(
                    "Hyperelasticity scaling must be a required-only real submission"
                )
        elif tiers == {P3D_SCALING_TIER}:
            _validate_optional_tranche_scope(selected, only_optional=only_optional)
        else:
            raise RuntimeError(
                "EXP-SCALE-001 real submission must select exactly one of the "
                "Hyperelasticity or optional Plasticity3D tiers"
            )
    elif experiment == "EXP-ROUTE-001" and tiers.intersection(TIER_B_TIERS):
        _validate_optional_tranche_scope(selected, only_optional=only_optional)
    elif any(row["optional"] == "1" for row in selected):
        _validate_optional_tranche_scope(selected, only_optional=only_optional)


def _validate_release_authorization_shape(gate: object) -> dict[str, object]:
    required = {
        "schema_id",
        "schema_version",
        "status",
        "decision",
        "matrix_sha256",
        "source_commit",
        "authorizes_experiment",
        "authorizes_tiers",
        "reviewer",
        "reviewed_artifacts",
    }
    if not isinstance(gate, dict) or set(gate) != required:
        raise RuntimeError(
            "release record must contain exactly the fields in the maintained "
            "human-release-authorization v1 schema"
        )
    if (
        gate.get("schema_id")
        != "fenics-nonlinear-energies.human-release-authorization"
        or type(gate.get("schema_version")) is not int
        or gate.get("schema_version") != 1
        or gate.get("status") != "approved"
        or gate.get("decision") != "explicit_human_release_after_review"
    ):
        raise RuntimeError(
            "release record must use the frozen human-release-authorization v1 schema"
        )
    if not _is_lower_hex(gate.get("matrix_sha256"), 64):
        raise RuntimeError("release record matrix_sha256 must be lowercase SHA-256")
    if not _is_lower_hex(gate.get("source_commit"), 40):
        raise RuntimeError("release record source_commit must be a lowercase Git commit")
    if gate.get("authorizes_experiment") not in PROTOCOLS:
        raise RuntimeError("release record authorizes an unknown experiment")
    tiers = gate.get("authorizes_tiers")
    if (
        not isinstance(tiers, list)
        or not tiers
        or not all(
            isinstance(value, str) and value and value == value.strip()
            for value in tiers
        )
        or len(set(tiers)) != len(tiers)
    ):
        raise RuntimeError("release record must authorize unique nonempty tiers")
    if not isinstance(gate.get("reviewer"), str) or not str(gate["reviewer"]).strip():
        raise RuntimeError("release record must identify the human reviewer")
    reviewed = gate.get("reviewed_artifacts")
    if not isinstance(reviewed, list) or not reviewed:
        raise RuntimeError("release record must enumerate reviewed analysis artifacts")
    reviewed_paths: set[str] = set()
    for index, artifact in enumerate(reviewed):
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
            raise RuntimeError(
                f"reviewed_artifacts entry {index} must contain exactly path and sha256"
            )
        if not isinstance(artifact.get("path"), str) or not str(artifact["path"]).strip():
            raise RuntimeError(f"reviewed artifact {index} has an empty path")
        artifact_path = str(artifact["path"])
        if artifact_path != artifact_path.strip() or artifact_path in reviewed_paths:
            raise RuntimeError(f"reviewed artifact {index} has a padded or duplicate path")
        reviewed_paths.add(artifact_path)
        if not _is_lower_hex(artifact.get("sha256"), 64):
            raise RuntimeError(f"reviewed artifact {index} has an invalid SHA-256 value")
    return gate


def _require_staged_real_submission(
    args: argparse.Namespace,
    *,
    selected: list[dict[str, str]],
    matrix: Path,
    git: dict[str, object],
) -> dict[str, str] | None:
    """Prevent downstream experiments from being co-submitted before admission."""

    if bool(args.test_only):
        return None
    _validate_real_submission_scope(args, selected=selected)
    experiments = {row["experiment_id"] for row in selected}
    tiers = {row["tier"] for row in selected}
    experiment = next(iter(experiments))
    gated = (
        experiment in {"EXP-ROUTE-001", "EXP-SCALE-001"}
        or (experiment == "EXP-DISC-001" and tiers != {"smoke"})
    )
    if not gated:
        return None
    gate_path = None if args.admission_gate is None else Path(args.admission_gate).resolve()
    if gate_path is None or not gate_path.is_file():
        raise RuntimeError(
            "this downstream tranche requires --admission-gate from the preceding "
            "correctness/discretization adjudication"
        )
    with gate_path.open(encoding="utf-8") as handle:
        gate = json.load(handle)
    gate = _validate_release_authorization_shape(gate)
    if gate.get("matrix_sha256") != _sha256(matrix):
        raise RuntimeError("admission gate matrix hash is stale")
    if gate.get("source_commit") != git["commit"] or git["dirty"] is not False:
        raise RuntimeError("admission gate does not match the clean source commit")
    if gate.get("authorizes_experiment") != experiment:
        raise RuntimeError("admission gate authorizes a different experiment")
    authorized = {str(value) for value in gate["authorizes_tiers"]}
    known_tiers = {
        row["tier"]
        for row in read_matrix(matrix)
        if row["experiment_id"] == experiment
    }
    if not authorized.issubset(known_tiers):
        raise RuntimeError("admission gate contains a tier unknown to its experiment")
    if not tiers.issubset(authorized):
        raise RuntimeError("admission gate does not authorize every selected tier")
    reviewed = gate["reviewed_artifacts"]
    for index, artifact in enumerate(reviewed):
        assert isinstance(artifact, dict)
        artifact_path = Path(str(artifact.get("path", "")))
        if not artifact_path.is_absolute():
            artifact_path = gate_path.parent / artifact_path
        artifact_path = artifact_path.resolve()
        if not artifact_path.is_file() or artifact.get("sha256") != _sha256(artifact_path):
            raise RuntimeError(f"reviewed artifact {index} is missing or has a stale hash")
    return {
        "schema_id": str(gate["schema_id"]),
        "path": str(gate_path),
        "sha256": _sha256(gate_path),
        "reviewer": str(gate["reviewer"]),
    }


def _archive_release_authorization(
    release_authorization: dict[str, str], *, out_root: Path
) -> dict[str, str]:
    """Archive a relocatable release record and every artifact it reviewed."""

    source_path = Path(release_authorization["path"]).resolve()
    with source_path.open(encoding="utf-8") as handle:
        gate = json.load(handle)
    if not isinstance(gate, dict):
        raise RuntimeError("release authorization must contain a JSON object")
    reviewed = gate.get("reviewed_artifacts")
    if not isinstance(reviewed, list) or not reviewed:
        raise RuntimeError("release authorization has no reviewed artifacts to archive")

    archive_dir = out_root / "reviewed_artifacts"
    archive_dir.mkdir(parents=True, exist_ok=False)
    archived_reviewed: list[dict[str, str]] = []
    for index, artifact in enumerate(reviewed):
        if not isinstance(artifact, dict):
            raise RuntimeError("reviewed_artifacts entries must be JSON objects")
        source = Path(str(artifact.get("path", "")))
        if not source.is_absolute():
            source = source_path.parent / source
        source = source.resolve()
        destination = archive_dir / f"{index:03d}_{source.name}"
        shutil.copy2(source, destination)
        digest = _sha256(destination)
        if digest != artifact.get("sha256"):
            raise RuntimeError(f"reviewed artifact {index} changed during archival")
        archived_reviewed.append(
            {
                "path": str(destination.relative_to(out_root)),
                "sha256": digest,
            }
        )

    archived_gate = dict(gate)
    archived_gate["reviewed_artifacts"] = archived_reviewed
    archived_release = out_root / "release_authorization.json"
    _atomic_write_json(archived_release, archived_gate)
    return {
        "schema_id": str(archived_gate["schema_id"]),
        "path": archived_release.name,
        "sha256": _sha256(archived_release),
        "reviewer": str(archived_gate["reviewer"]),
    }


def sbatch_command(
    row: dict[str, str],
    *,
    matrix: Path,
    out_root: Path,
    test_only: bool,
    expected_source_commit: str,
    expected_matrix_sha256: str,
    source_freeze: dict[str, str],
) -> list[str]:
    command = [
        "sbatch",
        "--job-name",
        str(row["case_id"]),
        "--account",
        ACCOUNT,
        "--qos",
        QOS,
        "--partition",
        str(row["partition"]),
        "--nodes",
        str(row["nodes"]),
        "--ntasks",
        str(row["total_ranks"]),
        "--ntasks-per-node",
        str(row["ranks_per_node"]),
        "--cpus-per-task",
        "1",
        "--distribution",
        "block:block",
        "--time",
        str(row["time_limit"]),
        "--chdir",
        str(REPO_ROOT),
        "--output",
        str(out_root / "slurm" / "%x-%j.out"),
        "--error",
        str(out_root / "slurm" / "%x-%j.err"),
    ]
    if test_only:
        command.append("--test-only")
    command.extend(
        [
            str(SBATCH_RUNNER),
            str(matrix),
            str(row["case_id"]),
            str(out_root),
            expected_source_commit,
            expected_matrix_sha256,
            str(out_root / source_freeze["path"]),
            source_freeze["sha256"],
        ]
    )
    return command


def _require_revalidation(*, test_only: bool) -> None:
    if os.environ.get("ALLOCATION_REVALIDATED") != "YES":
        raise RuntimeError(
            "submission disabled: set ALLOCATION_REVALIDATED=YES only after checking "
            "the current Karolina allocation in SCS/sacctmgr"
        )
    if os.environ.get("ACCOUNT_QOS_REVALIDATED") != "YES":
        raise RuntimeError(
            "submission disabled: revalidate account fta-26-40 and QoS 3571_6328, "
            "then set ACCOUNT_QOS_REVALIDATED=YES"
        )
    valid_until_raw = os.environ.get("ALLOCATION_VALID_UNTIL", "")
    try:
        valid_until = date.fromisoformat(valid_until_raw)
    except ValueError as exc:
        raise RuntimeError(
            "submission disabled: ALLOCATION_VALID_UNTIL must be a revalidated YYYY-MM-DD date"
        ) from exc
    if valid_until <= date.today():
        raise RuntimeError(
            f"submission disabled: revalidated allocation end {valid_until} is not after today"
        )
    if not test_only and os.environ.get("SUBMIT_CONFIRMED") != "YES":
        raise RuntimeError(
            "real submission disabled: set SUBMIT_CONFIRMED=YES only after reviewing the generated plan"
        )
    if not test_only:
        status = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status:
            raise RuntimeError("real publication jobs require a clean committed worktree")


def _archive_member(root: Path, raw: object, *, name: str) -> Path:
    path = Path(str(raw or ""))
    if path.is_absolute() or not str(path):
        raise RuntimeError(f"prepared manifest {name} must be archive-relative")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"prepared manifest {name} escapes the campaign archive") from exc
    return resolved


def _command_option(tokens: list[str], option: str) -> str:
    if tokens.count(option) != 1:
        raise RuntimeError(f"prepared sbatch command must contain exactly one {option}")
    index = tokens.index(option)
    if index + 1 >= len(tokens):
        raise RuntimeError(f"prepared sbatch command has no value for {option}")
    return tokens[index + 1]


def offline_preflight(out_root: Path, *, matrix: Path = DEFAULT_MATRIX) -> dict[str, object]:
    """Validate a prepared archive without invoking or querying Slurm."""

    root = Path(out_root).resolve()
    matrix = Path(matrix).resolve()
    manifest_path = root / "prepared_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise RuntimeError("prepared_manifest.json must contain a JSON object")
    if manifest.get("status") not in {
        "prepared_not_submitted",
        "testing_admission",
        "test_only_completed",
        "submitting",
        "submitted",
    }:
        raise RuntimeError("prepared manifest has no recognized preparation/submission status")
    if manifest.get("matrix") != _repo_relative(matrix):
        raise RuntimeError("prepared manifest matrix path is not repository-relative")
    if manifest.get("matrix_sha256") != _sha256(matrix):
        raise RuntimeError("prepared manifest matrix hash is stale")
    if manifest.get("out_root") != ".":
        raise RuntimeError("prepared manifest out_root must be the relocatable archive root '.'")

    plan_path = _archive_member(root, manifest.get("plan_file"), name="plan_file")
    commands_path = _archive_member(
        root, manifest.get("commands_file"), name="commands_file"
    )
    freeze_record = manifest.get("queued_source_freeze")
    if not isinstance(freeze_record, dict):
        raise RuntimeError("prepared manifest lacks a queued source freeze")
    freeze_path = _archive_member(
        root, freeze_record.get("path"), name="queued_source_freeze.path"
    )
    for path, expected, name in (
        (plan_path, manifest.get("plan_sha256"), "prepared plan"),
        (commands_path, manifest.get("commands_sha256"), "prepared commands"),
        (freeze_path, freeze_record.get("sha256"), "queued source freeze"),
    ):
        if not path.is_file() or expected != _sha256(path):
            raise RuntimeError(f"{name} is missing or has a stale hash")

    with freeze_path.open(encoding="utf-8") as handle:
        freeze = json.load(handle)
    if not isinstance(freeze, dict):
        raise RuntimeError("queued source freeze must contain a JSON object")
    _validate_source_freeze_payload(
        freeze,
        matrix=matrix,
        source_commit=str(manifest.get("source_commit", "")),
    )

    with plan_path.open(newline="", encoding="utf-8") as handle:
        plan = [dict(row) for row in csv.DictReader(handle)]
    if not plan or len(plan) != int(manifest.get("case_count", -1)):
        raise RuntimeError("prepared plan count disagrees with its manifest")
    if len({row["case_id"] for row in plan}) != len(plan):
        raise RuntimeError("prepared plan contains duplicate case IDs")
    matrix_by_case = {row["case_id"]: row for row in read_matrix(matrix)}
    for row in plan:
        if matrix_by_case.get(row["case_id"]) != row:
            raise RuntimeError(f"prepared row {row['case_id']} differs from the matrix")
    if sorted({row["experiment_id"] for row in plan}) != manifest.get(
        "selected_experiments"
    ):
        raise RuntimeError("prepared experiment scope disagrees with its manifest")
    if sorted({row["tier"] for row in plan}) != manifest.get("selected_tiers"):
        raise RuntimeError("prepared tier scope disagrees with its manifest")
    expected_hours = sum(float(row["estimated_node_hours"]) for row in plan)
    if not math.isclose(
        expected_hours,
        float(manifest.get("estimated_node_hours", -1.0)),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError("prepared node-hour total disagrees with its plan")
    if any(row["optional"] == "1" for row in plan):
        _validate_optional_tranche_scope(
            plan, only_optional=bool(manifest.get("only_optional"))
        )

    command_lines = commands_path.read_text(encoding="utf-8").splitlines()
    if len(command_lines) != len(plan):
        raise RuntimeError("prepared command count disagrees with its plan")
    common_out_root: str | None = None
    common_freeze_path: str | None = None
    for row, line in zip(plan, command_lines, strict=True):
        tokens = shlex.split(line)
        if not tokens or tokens[0] != "sbatch":
            raise RuntimeError(f"{row['case_id']} command is not an sbatch argument vector")
        for forbidden in ("--exclusive", "--mem", "--mem-per-cpu"):
            if forbidden in tokens:
                raise RuntimeError(f"{row['case_id']} command contains forbidden {forbidden}")
        exact_options = {
            "--job-name": row["case_id"],
            "--account": ACCOUNT,
            "--qos": QOS,
            "--partition": row["partition"],
            "--nodes": row["nodes"],
            "--ntasks": row["total_ranks"],
            "--ntasks-per-node": row["ranks_per_node"],
            "--cpus-per-task": "1",
            "--distribution": "block:block",
            "--time": row["time_limit"],
        }
        for option, expected in exact_options.items():
            if _command_option(tokens, option) != expected:
                raise RuntimeError(f"{row['case_id']} command {option} changed")
        if ("--test-only" in tokens) is not bool(manifest.get("test_only_commands")):
            raise RuntimeError(f"{row['case_id']} command test-only status changed")
        try:
            batch_index = tokens.index(str(SBATCH_RUNNER))
        except ValueError as exc:
            raise RuntimeError(f"{row['case_id']} command uses an unreviewed batch runner") from exc
        arguments = tokens[batch_index + 1 :]
        if len(arguments) != 7:
            raise RuntimeError(f"{row['case_id']} batch argument count changed")
        if (
            Path(arguments[0]).resolve() != matrix
            or arguments[1] != row["case_id"]
            or arguments[3] != manifest.get("source_commit")
            or arguments[4] != manifest.get("matrix_sha256")
            or arguments[6] != freeze_record.get("sha256")
        ):
            raise RuntimeError(f"{row['case_id']} batch provenance arguments changed")
        common_out_root = arguments[2] if common_out_root is None else common_out_root
        common_freeze_path = arguments[5] if common_freeze_path is None else common_freeze_path
        if arguments[2] != common_out_root or arguments[5] != common_freeze_path:
            raise RuntimeError("prepared commands do not share one output root/source freeze")
        if Path(arguments[5]).name != freeze_path.name:
            raise RuntimeError(f"{row['case_id']} batch source-freeze name changed")
    return {
        "status": "passed",
        "mode": "offline_no_scheduler_access",
        "case_count": len(plan),
        "estimated_node_hours": expected_hours,
        "matrix_sha256": _sha256(matrix),
        "plan_sha256": _sha256(plan_path),
        "commands_sha256": _sha256(commands_path),
        "source_freeze_sha256": _sha256(freeze_path),
    }


def prepare(args: argparse.Namespace) -> dict[str, object]:
    matrix = Path(args.matrix).resolve()
    reviewed_source_hashes = _validate_reviewed_sources()
    with ROUTE_ANALYSIS_CONTRACT.open(encoding="utf-8") as handle:
        contract = json.load(handle)
    if _sha256(matrix) != str(
        contract["publication_model_input_gates"]["karolina_matrix_sha256"]
    ):
        raise RuntimeError("campaign matrix hash disagrees with the frozen route contract")
    rows = read_matrix(matrix)
    selected = select_rows(
        rows,
        experiments=set(args.experiment),
        include_optional=bool(args.include_optional),
        only_optional=bool(args.only_optional),
        tiers=set(args.tier),
    )
    if any(row["optional"] == "1" for row in selected):
        _validate_optional_tranche_scope(
            selected, only_optional=bool(args.only_optional)
        )
    total_node_hours = sum(float(row["estimated_node_hours"]) for row in selected)
    if total_node_hours > float(args.max_node_hours):
        raise RuntimeError(
            f"selected campaign requires {total_node_hours:.2f} node-hours, above guard "
            f"{float(args.max_node_hours):.2f}"
        )

    out_root = Path(args.out_root).resolve()
    out_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_root.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(
            "campaign output root already exists; use a fresh unique campaign ID so "
            "prepared, test-only, and submitted provenance can never be mixed"
        ) from exc
    (out_root / "slurm").mkdir(exist_ok=True)
    (out_root / "cases").mkdir(exist_ok=True)
    git = _git_metadata()
    matrix_sha256 = _sha256(matrix)
    source_freeze = _write_source_freeze(
        out_root=out_root,
        matrix=matrix,
        source_commit=str(git["commit"]),
        reviewed_source_hashes=reviewed_source_hashes,
    )
    commands = [
        sbatch_command(
            row,
            matrix=matrix,
            out_root=out_root,
            test_only=bool(args.test_only),
            expected_source_commit=str(git["commit"]),
            expected_matrix_sha256=matrix_sha256,
            source_freeze=source_freeze,
        )
        for row in selected
    ]

    plan_path = out_root / "prepared_plan.csv"
    with plan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    commands_path = out_root / "sbatch_commands.txt"
    commands_path.write_text(
        "".join(shlex.join(command) + "\n" for command in commands),
        encoding="utf-8",
    )
    manifest = {
        "manifest_version": 1,
        "status": "prepared_not_submitted",
        "cluster": "Karolina CPU",
        "account": ACCOUNT,
        "qos": QOS,
        "known_allocation_record_ends": ALLOCATION_RECORD_END,
        "allocation_revalidation_required": True,
        "matrix": _repo_relative(matrix),
        "matrix_sha256": matrix_sha256,
        "static_campaign_manifest": _repo_relative(STATIC_CAMPAIGN_MANIFEST),
        "static_campaign_manifest_sha256": _sha256(STATIC_CAMPAIGN_MANIFEST),
        "reviewed_source_sha256": reviewed_source_hashes,
        "queued_source_freeze": source_freeze,
        "selected_experiments": sorted({row["experiment_id"] for row in selected}),
        "selected_tiers": sorted({row["tier"] for row in selected}),
        "protocol_cards": {
            key: value
            for key, value in PROTOCOLS.items()
            if key in {row["experiment_id"] for row in selected}
        },
        "include_optional": bool(args.include_optional or args.only_optional),
        "only_optional": bool(args.only_optional),
        "case_count": len(selected),
        "estimated_node_hours": float(total_node_hours),
        "node_hour_guard": float(args.max_node_hours),
        "out_root": ".",
        "test_only_commands": bool(args.test_only),
        "source_commit": git["commit"],
        "source_dirty": git["dirty"],
        "commands_file": commands_path.name,
        "commands_sha256": _sha256(commands_path),
        "plan_file": plan_path.name,
        "plan_sha256": _sha256(plan_path),
    }
    if set(manifest["selected_experiments"]) == {"EXP-DISC-001"} and len(
        manifest["selected_tiers"]
    ) == 1:
        stage = str(manifest["selected_tiers"][0])
        if stage in DISC_RELEASE_STAGES:
            index = DISC_RELEASE_STAGES.index(stage)
            manifest["disc_release_stage"] = {
                "unit": "protocol_stage",
                "stage": stage,
                "position": index + 1,
                "stage_count": len(DISC_RELEASE_STAGES),
                "case_count": DISC_STAGE_CASE_COUNTS[stage],
                "prerequisite_stage": None if index == 0 else DISC_RELEASE_STAGES[index - 1],
                "later_stage_release_requires_separate_human_authorization": True,
            }
    manifest_path = out_root / "prepared_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    preflight = offline_preflight(out_root, matrix=matrix)
    manifest["offline_preflight"] = preflight
    _atomic_write_json(manifest_path, manifest)

    if args.execute:
        _require_revalidation(test_only=bool(args.test_only))
        release_authorization = _require_staged_real_submission(
            args,
            selected=selected,
            matrix=matrix,
            git=git,
        )
        if release_authorization is not None:
            release_authorization = _archive_release_authorization(
                release_authorization, out_root=out_root
            )
        manifest["release_authorization"] = release_authorization
        manifest["status"] = "testing_admission" if args.test_only else "submitting"
        manifest["submission_progress"] = {
            "attempted": 0,
            "accepted": 0,
            "total": len(selected),
            "last_case_id": None,
        }
        _atomic_write_json(manifest_path, manifest)
        submitted_path = out_root / (
            "test_only_results.jsonl" if args.test_only else "submitted_jobs.jsonl"
        )
        accepted = 0
        try:
            with submitted_path.open("x", encoding="utf-8") as handle:
                for attempted, (row, command) in enumerate(
                    zip(selected, commands, strict=True), start=1
                ):
                    completed = subprocess.run(
                        command, check=False, capture_output=True, text=True
                    )
                    record = {
                        "case_id": row["case_id"],
                        "command": shlex.join(command),
                        "returncode": int(completed.returncode),
                        "stdout": completed.stdout.strip(),
                        "stderr": completed.stderr.strip(),
                    }
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    if completed.returncode == 0:
                        accepted += 1
                    manifest["submission_progress"] = {
                        "attempted": attempted,
                        "accepted": accepted,
                        "total": len(selected),
                        "last_case_id": row["case_id"],
                    }
                    _atomic_write_json(manifest_path, manifest)
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f"sbatch {'test-only ' if args.test_only else ''}failed for "
                            f"{row['case_id']}"
                        )
        except BaseException as exc:
            if args.test_only:
                manifest["status"] = "test_only_failed"
            else:
                manifest["status"] = (
                    "partial_submission" if accepted else "submission_failed"
                )
            manifest["submission_error"] = f"{type(exc).__name__}: {exc}"
            _atomic_write_json(manifest_path, manifest)
            raise
        manifest["status"] = "test_only_completed" if args.test_only else "submitted"
        _atomic_write_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        action="append",
        choices=tuple(PROTOCOLS),
        default=[],
        help="Repeat to select experiments; omission selects all required rows",
    )
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument(
        "--tier",
        action="append",
        default=[],
        help="Repeat to select explicit protocol tiers; required for real submission.",
    )
    parser.add_argument(
        "--only-optional",
        action="store_true",
        help="Select only optional rows (usually with one --experiment tranche).",
    )
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--admission-gate",
        type=Path,
        help="Passed JSON gate required before downstream DISC/SCALE/Tier-B tranches.",
    )
    parser.add_argument("--max-node-hours", type=float, default=DEFAULT_NODE_HOUR_GUARD)
    return parser


def main() -> None:
    try:
        manifest = prepare(_parser().parse_args())
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate and prepare the publication-revision Karolina Slurm matrix.

This program is non-submitting unless ``--execute`` is supplied.  Execution is
still guarded by explicit, current allocation revalidation environment values.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
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
MODEL_FREEZE_SCHEMA = REPO_ROOT / "paper/protocols/route-model-freeze-v2.schema.json"
MODEL_FREEZE_EXAMPLE = REPO_ROOT / "paper/protocols/route-model-freeze-v2.example.json"
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
    "route_tier_b_aggregator": REPO_ROOT
    / "experiments/analysis/aggregate_route_tier_b_manifests.py",
    "route_training_freezer": REPO_ROOT
    / "experiments/analysis/freeze_route_training_model.py",
    "discretization_analyzer": REPO_ROOT
    / "experiments/analysis/analyze_plasticity3d_discretization.py",
    "scaling_analyzer": REPO_ROOT / "experiments/analysis/analyze_exp_scale_001.py",
    "slurm_accounting_collector": REPO_ROOT
    / "experiments/analysis/collect_slurm_accounting.py",
    "offline_accounting_index_generator": REPO_ROOT
    / "experiments/analysis/generate_offline_accounting_index.py",
    "campaign_archive_finalizer": REPO_ROOT
    / "experiments/analysis/finalize_karolina_campaign_archive.py",
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
    "partial_submission_resumer": Path(__file__).with_name(
        "resume_partial_submission.py"
    ),
    "batch_runner": SBATCH_RUNNER,
    "submitter": Path(__file__).with_name("submit_prepared_campaigns.sh"),
    "state_export": REPO_ROOT / "src/core/benchmark/state_export.py",
    "fixed_state_support": REPO_ROOT
    / "src/problems/slope_stability_3d/support/fixed_state.py",
    "quadrature_support": REPO_ROOT
    / "src/problems/slope_stability_3d/support/mesh.py",
    "release_authorization_schema": RELEASE_AUTHORIZATION_SCHEMA,
    "release_authorization_example": RELEASE_AUTHORIZATION_EXAMPLE,
    "model_freeze_schema": MODEL_FREEZE_SCHEMA,
    "model_freeze_example": MODEL_FREEZE_EXAMPLE,
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
ROUTE_PHASES = frozenset({"training", "holdout"})
MODEL_FREEZE_SCHEMA_ID = "fenics-nonlinear-energies.route-model-freeze"
MODEL_FREEZE_SCHEMA_VERSION = 2
ROUTE_SCOPE_COUNTS = {
    "cost_model_training": 76,
    "tier_b_training": 20,
    "cost_model_holdout": 29,
    "tier_b_holdout": 10,
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


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _submitted_job_id(stdout: str) -> str:
    prefix = "Submitted batch job "
    text = str(stdout).strip()
    if not text.startswith(prefix) or not text.removeprefix(prefix).isdigit():
        raise RuntimeError("successful sbatch response lacks an unambiguous numeric job ID")
    return text.removeprefix(prefix)


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
    route_phase: str | None = None,
) -> list[dict[str, str]]:
    if route_phase is not None and route_phase not in ROUTE_PHASES:
        raise ValueError(f"unknown EXP-ROUTE phase {route_phase!r}")
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
        if route_phase is not None and row["experiment_id"] == "EXP-ROUTE-001":
            ranks = int(row["total_ranks"])
            if route_phase == "training" and ranks == 32:
                continue
            if route_phase == "holdout" and ranks != 32:
                continue
        selected.append(row)
    if not selected:
        raise ValueError("no campaign rows selected")
    return selected


def _validate_optional_tranche_scope(
    selected: list[dict[str, str]], *, only_optional: bool, route_phase: str | None = None
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
        expected_tiers = (
            {"full_solve_confirmation"}
            if route_phase == "holdout"
            else set(TIER_B_TIERS)
        )
        expected_count = {None: 30, "training": 20, "holdout": 10}.get(route_phase)
        if tiers != expected_tiers or len(selected) != expected_count:
            raise RuntimeError(
                "optional EXP-ROUTE-001 must be the exact prespecified Tier-B "
                f"{route_phase or 'combined'} phase scope"
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
    route_phase = getattr(args, "route_phase", None)

    if experiment == "EXP-ROUTE-001":
        if route_phase not in ROUTE_PHASES:
            raise RuntimeError(
                "real EXP-ROUTE-001 submission requires --route-phase training or holdout"
            )
        if route_phase == "training" and any(
            int(row["total_ranks"]) == 32 for row in selected
        ):
            raise RuntimeError("training tranche contains a rank-32 holdout row")
        if route_phase == "holdout" and any(
            int(row["total_ranks"]) != 32 for row in selected
        ):
            raise RuntimeError("holdout tranche contains a rank-1/8 training row")
    elif route_phase is not None:
        raise RuntimeError("--route-phase is defined only for EXP-ROUTE-001")

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
        _validate_optional_tranche_scope(
            selected,
            only_optional=only_optional,
            route_phase=getattr(args, "route_phase", None),
        )
    elif any(row["optional"] == "1" for row in selected):
        _validate_optional_tranche_scope(
            selected,
            only_optional=only_optional,
            route_phase=getattr(args, "route_phase", None),
        )


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


def _case_ids_sha256(case_ids: Iterable[str]) -> str:
    canonical = json.dumps(sorted(case_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _route_scopes(matrix: Path) -> dict[str, list[dict[str, str]]]:
    """Return the four non-overlapping route scopes frozen by receipt v2."""

    rows = [
        row
        for row in read_matrix(matrix)
        if row["experiment_id"] == "EXP-ROUTE-001"
    ]
    scopes = {
        "cost_model_training": [
            row
            for row in rows
            if row["optional"] == "0" and int(row["total_ranks"]) in {1, 8}
        ],
        "tier_b_training": [
            row
            for row in rows
            if row["optional"] == "1" and int(row["total_ranks"]) in {1, 8}
        ],
        "cost_model_holdout": [
            row
            for row in rows
            if row["optional"] == "0" and int(row["total_ranks"]) == 32
        ],
        "tier_b_holdout": [
            row
            for row in rows
            if row["optional"] == "1" and int(row["total_ranks"]) == 32
        ],
    }
    seen: set[str] = set()
    for name, expected_count in ROUTE_SCOPE_COUNTS.items():
        case_ids = [row["case_id"] for row in scopes[name]]
        if len(case_ids) != expected_count or len(set(case_ids)) != expected_count:
            raise RuntimeError(
                f"canonical {name} scope must contain exactly {expected_count} cases"
            )
        if seen.intersection(case_ids):
            raise RuntimeError("canonical route scopes overlap")
        seen.update(case_ids)
    if len(seen) != len(rows):
        raise RuntimeError("canonical route scopes do not partition EXP-ROUTE-001")
    return scopes


def _model_training_case_ids(matrix: Path) -> list[str]:
    return sorted(row["case_id"] for row in _route_scopes(matrix)["cost_model_training"])


def _scope_receipt(scopes: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    return {
        name: {
            "case_count": ROUTE_SCOPE_COUNTS[name],
            "case_ids_sha256": _case_ids_sha256(row["case_id"] for row in rows),
        }
        for name, rows in scopes.items()
    }


def _resolve_receipt_artifacts(
    receipt_path: Path, receipt: dict[str, object]
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for key in (
        "cost_model_training_manifest",
        "tier_b_training_manifest",
        "training_analysis",
        "frozen_model",
    ):
        artifact = receipt.get(key)
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
            raise RuntimeError(f"route model-freeze {key} artifact is malformed")
        artifact_path = Path(str(artifact["path"]))
        if not artifact_path.is_absolute():
            artifact_path = receipt_path.parent / artifact_path
        if (
            artifact_path.is_symlink()
            or not artifact_path.resolve().is_file()
            or artifact.get("sha256") != _sha256(artifact_path.resolve())
        ):
            raise RuntimeError(f"route model-freeze {key} artifact is missing or stale")
        resolved[key] = artifact_path.resolve()
    return resolved


def _validate_freeze_training_manifest(
    manifest_path: Path,
    *,
    matrix: Path,
    source_commit: str,
    scope_name: str,
    expected_rows: list[dict[str, str]],
) -> dict[str, str]:
    """Validate one completed training tranche without scheduler access."""

    root = manifest_path.parent.resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{scope_name} manifest is not a JSON object")
    expected_ids = sorted(row["case_id"] for row in expected_rows)
    expected_tiers = sorted({row["tier"] for row in expected_rows})
    is_tier_b = scope_name == "tier_b_training"
    if (
        manifest.get("status") != "submitted"
        or manifest.get("test_only_commands") is not False
        or manifest.get("route_phase") != "training"
        or manifest.get("selected_experiments") != ["EXP-ROUTE-001"]
        or manifest.get("selected_tiers") != expected_tiers
        or manifest.get("include_optional") is not is_tier_b
        or manifest.get("only_optional") is not is_tier_b
        or int(manifest.get("case_count", -1)) != len(expected_ids)
        or manifest.get("matrix_sha256") != _sha256(matrix)
        or manifest.get("source_commit") != source_commit
        or manifest.get("source_dirty") is not False
        or manifest.get("route_phase_case_ids_sha256")
        != _case_ids_sha256(expected_ids)
    ):
        raise RuntimeError(f"{scope_name} manifest has stale scope or source identity")
    plan_path = _archive_member(root, manifest.get("plan_file"), name=f"{scope_name}.plan")
    if manifest.get("plan_sha256") != _sha256(plan_path):
        raise RuntimeError(f"{scope_name} plan hash is stale")
    with plan_path.open(newline="", encoding="utf-8") as handle:
        plan = [dict(row) for row in csv.DictReader(handle)]
    canonical = {row["case_id"]: row for row in expected_rows}
    if (
        len(plan) != len(expected_ids)
        or len({row.get("case_id") for row in plan}) != len(expected_ids)
        or any(canonical.get(str(row.get("case_id", ""))) != row for row in plan)
    ):
        raise RuntimeError(f"{scope_name} plan is not the exact canonical scope")

    environment = manifest.get("environment_contract")
    if not isinstance(environment, dict) or environment.get("status") != "hash_bound":
        raise RuntimeError(f"{scope_name} manifest lacks a hash-bound environment")
    identity: dict[str, str] = {}
    for record_key, hash_key in (
        ("archived_setup", "setup_sha256"),
        ("archived_lock", "lock_sha256"),
    ):
        record = environment.get(record_key)
        digest = environment.get(hash_key)
        if not isinstance(record, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise RuntimeError(f"{scope_name} environment identity is malformed")
        artifact = _archive_member(
            root, record.get("path"), name=f"{scope_name}.{record_key}"
        )
        if record.get("sha256") != digest or _sha256(artifact) != digest:
            raise RuntimeError(f"{scope_name} environment identity is stale")
        identity[hash_key] = str(digest)

    try:
        preflight = offline_preflight(root, matrix=matrix)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{scope_name} archive preflight failed: {exc}") from exc
    if preflight and preflight.get("mode") not in {None, "offline_no_scheduler_access"}:
        raise RuntimeError(f"{scope_name} archive preflight was not scheduler-free")

    ledger_path = _archive_member(
        root, "submitted_jobs.jsonl", name=f"{scope_name}.submitted_jobs"
    )
    ledger: list[dict[str, object]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{scope_name} submission ledger is malformed")
            ledger.append(value)
    ledger_ids = [str(record.get("case_id", "")) for record in ledger]
    job_ids = [str(record.get("job_id", "")) for record in ledger]
    if (
        sorted(ledger_ids) != expected_ids
        or len(set(ledger_ids)) != len(expected_ids)
        or len(set(job_ids)) != len(expected_ids)
        or any(
            int(record.get("returncode", 1)) != 0
            or not str(record.get("job_id", "")).isdigit()
            or int(str(record.get("job_id", "0"))) <= 0
            for record in ledger
        )
    ):
        raise RuntimeError(f"{scope_name} ledger does not prove complete submission")
    return identity


def _validate_model_freeze_receipt(
    path: Path, *, matrix: Path, source_commit: str
) -> dict[str, object]:
    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        receipt = json.load(handle)
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
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise RuntimeError("route model-freeze receipt has an unexpected shape")
    canonical_scopes = _route_scopes(matrix)
    expected_scopes = _scope_receipt(canonical_scopes)
    expected_training = sorted(
        row["case_id"] for row in canonical_scopes["cost_model_training"]
    )
    if (
        receipt.get("schema_id") != MODEL_FREEZE_SCHEMA_ID
        or receipt.get("schema_version") != MODEL_FREEZE_SCHEMA_VERSION
        or receipt.get("status") != "frozen_before_holdout"
        or receipt.get("decision")
        != "cost_model_fit_and_tier_b_training_complete_holdouts_unopened"
        or receipt.get("matrix_sha256") != _sha256(matrix)
        or receipt.get("source_commit") != source_commit
        or receipt.get("scopes") != expected_scopes
        or not str(receipt.get("reviewer", "")).strip()
    ):
        raise RuntimeError("route model-freeze receipt identity, scope, or commit is stale")
    try:
        created = datetime.fromisoformat(
            str(receipt.get("created_at_utc", "")).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise RuntimeError("route model-freeze timestamp is invalid") from exc
    if (
        created.tzinfo is None
        or created.utcoffset() is None
        or created.utcoffset() != timezone.utc.utcoffset(created)
    ):
        raise RuntimeError("route model-freeze timestamp must be UTC")

    resolved_artifacts = _resolve_receipt_artifacts(path, receipt)
    identities = {
        scope_name: _validate_freeze_training_manifest(
            resolved_artifacts[f"{scope_name}_manifest"],
            matrix=matrix,
            source_commit=source_commit,
            scope_name=scope_name,
            expected_rows=canonical_scopes[scope_name],
        )
        for scope_name in ("cost_model_training", "tier_b_training")
    }
    identity = receipt.get("environment_identity")
    if (
        not isinstance(identity, dict)
        or set(identity) != {"setup_sha256", "lock_sha256"}
        or identities["cost_model_training"] != identities["tier_b_training"]
        or identity != identities["cost_model_training"]
    ):
        raise RuntimeError(
            "cost-model and Tier-B training manifests do not share the receipt environment"
        )

    with resolved_artifacts["training_analysis"].open(encoding="utf-8") as handle:
        training_analysis = json.load(handle)
    with resolved_artifacts["frozen_model"].open(encoding="utf-8") as handle:
        frozen_model = json.load(handle)
    with ROUTE_ANALYSIS_CONTRACT.open(encoding="utf-8") as handle:
        route_contract = json.load(handle)
    if not isinstance(training_analysis, dict) or not isinstance(frozen_model, dict):
        raise RuntimeError("route training analysis and frozen model must be JSON objects")
    analysis_case_ids = training_analysis.get("training_case_ids")
    training_row_ids = training_analysis.get("training_row_ids")
    model_row_ids = frozen_model.get("training_row_ids")
    feature_order = list(route_contract["cost_model"]["features_in_order"])
    coefficients = frozen_model.get("coefficients")
    design = frozen_model.get("design_diagnostics")
    analysis_model = training_analysis.get("frozen_model")
    analysis_contract = training_analysis.get("contract")
    karolina_campaign = training_analysis.get("karolina_training_campaign")
    karolina_case_ids = (
        karolina_campaign.get("case_ids")
        if isinstance(karolina_campaign, dict)
        else None
    )
    if (
        training_analysis.get("schema_id")
        != "fenics-nonlinear-energies.exp-route-001-training-analysis"
        or training_analysis.get("schema_version") != 1
        or training_analysis.get("status") != "training_fit_admitted"
        or training_analysis.get("experiment_id") != "EXP-ROUTE-001"
        or training_analysis.get("holdout_rows_seen") != 0
        or training_analysis.get("matrix_sha256") != _sha256(matrix)
        or training_analysis.get("source_commit") != source_commit
        or training_analysis.get("training_case_count") != len(expected_training)
        or not isinstance(analysis_case_ids, list)
        or not all(isinstance(value, str) and value for value in analysis_case_ids)
        or sorted(analysis_case_ids) != expected_training
        or training_analysis.get("training_case_ids_sha256")
        != _case_ids_sha256(expected_training)
        or not isinstance(training_row_ids, list)
        or len(training_row_ids) != 74
        or not all(isinstance(value, str) and value for value in training_row_ids)
        or len(set(training_row_ids)) != 74
        or training_analysis.get("training_row_count") != 74
        or training_analysis.get("training_row_ids_sha256")
        != _case_ids_sha256(str(value) for value in training_row_ids)
        or not isinstance(analysis_contract, dict)
        or analysis_contract.get("sha256") != _sha256(ROUTE_ANALYSIS_CONTRACT)
        or not isinstance(analysis_model, dict)
        or analysis_model.get("sha256") != _sha256(resolved_artifacts["frozen_model"])
        or not isinstance(karolina_campaign, dict)
        or karolina_campaign.get("route_phase") != "training"
        or not isinstance(karolina_case_ids, list)
        or not all(isinstance(value, str) and value for value in karolina_case_ids)
        or sorted(karolina_case_ids) != expected_training
    ):
        raise RuntimeError("route training analysis is incomplete, stale, or holdout-contaminated")
    declared_model_path = Path(str(analysis_model.get("path", "")))
    if not declared_model_path.is_absolute():
        declared_model_path = (
            resolved_artifacts["training_analysis"].parent / declared_model_path
        )
    if declared_model_path.resolve() != resolved_artifacts["frozen_model"]:
        raise RuntimeError("route training analysis names a different frozen model")
    if (
        frozen_model.get("schema_id")
        != "fenics-nonlinear-energies.exp-route-001-frozen-training-model"
        or frozen_model.get("schema_version") != 1
        or frozen_model.get("status") != "frozen_before_holdout"
        or frozen_model.get("experiment_id") != "EXP-ROUTE-001"
        or frozen_model.get("holdout_rows_seen") != 0
        or frozen_model.get("matrix_sha256") != _sha256(matrix)
        or frozen_model.get("source_commit") != source_commit
        or frozen_model.get("contract_sha256") != _sha256(ROUTE_ANALYSIS_CONTRACT)
        or frozen_model.get("training_case_ids_sha256")
        != _case_ids_sha256(expected_training)
        or frozen_model.get("training_rows") != 74
        or model_row_ids != training_row_ids
        or frozen_model.get("training_row_ids_sha256")
        != _case_ids_sha256(str(value) for value in training_row_ids)
        or frozen_model.get("feature_order") != feature_order
        or not isinstance(coefficients, dict)
        or list(coefficients) != feature_order
        or any(not math.isfinite(float(coefficients[name])) for name in feature_order)
        or not isinstance(design, dict)
        or design.get("rows") != 74
        or design.get("columns") != len(feature_order)
        or design.get("rank") != len(feature_order)
        or not math.isfinite(float(design.get("condition_number", float("nan"))))
        or float(design["condition_number"])
        > float(route_contract["cost_model"]["maximum_design_condition_number"])
    ):
        raise RuntimeError("frozen route training model violates its prespecified design")

    return {
        "receipt": receipt,
        "receipt_path": path,
        "artifacts": resolved_artifacts,
    }


def _archive_model_freeze_receipt(
    validated: dict[str, object], *, out_root: Path
) -> dict[str, str]:
    receipt = dict(validated["receipt"])
    artifacts = dict(validated["artifacts"])
    archive = out_root / "model_freeze_artifacts"
    archive.mkdir(parents=True, exist_ok=False)

    for manifest_key in (
        "cost_model_training_manifest",
        "tier_b_training_manifest",
    ):
        training_manifest_source = Path(artifacts[manifest_key])
        training_root = training_manifest_source.parent.resolve()
        with training_manifest_source.open(encoding="utf-8") as handle:
            training_manifest = json.load(handle)
        if not isinstance(training_manifest, dict):
            raise RuntimeError(f"route {manifest_key} is not a JSON object")
        scope_name = manifest_key.removesuffix("_manifest")
        snapshot_root = archive / f"{scope_name}_campaign"
        snapshot_root.mkdir()
        relative_sources: set[Path] = {
            Path(str(training_manifest["plan_file"])),
            Path(str(training_manifest["commands_file"])),
            Path(str(dict(training_manifest["queued_source_freeze"])["path"])),
            Path("submitted_jobs.jsonl"),
            Path("submission_journal.jsonl"),
        }
        environment = dict(training_manifest.get("environment_contract") or {})
        for key in ("archived_setup", "archived_lock"):
            record = environment.get(key)
            if isinstance(record, dict):
                relative_sources.add(Path(str(record.get("path", ""))))
        release = training_manifest.get("release_authorization")
        if isinstance(release, dict):
            release_relative = Path(str(release.get("path", "")))
            relative_sources.add(release_relative)
            release_path = (training_root / release_relative).resolve()
            with release_path.open(encoding="utf-8") as handle:
                release_payload = json.load(handle)
            for record in list(dict(release_payload).get("reviewed_artifacts") or []):
                if isinstance(record, dict):
                    relative_sources.add(Path(str(record.get("path", ""))))
        for relative in sorted(relative_sources, key=str):
            if relative.is_absolute() or not str(relative):
                raise RuntimeError("route training provenance path is not archive-relative")
            source = (training_root / relative).resolve()
            try:
                source.relative_to(training_root)
            except ValueError as exc:
                raise RuntimeError("route training provenance path escapes its archive") from exc
            if not source.is_file():
                raise RuntimeError(f"route training provenance file is missing: {relative}")
            destination = snapshot_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if _sha256(destination) != _sha256(source):
                raise RuntimeError(
                    f"route training provenance changed during archival: {relative}"
                )
        training_manifest_destination = snapshot_root / "prepared_manifest.json"
        shutil.copy2(training_manifest_source, training_manifest_destination)
        receipt[manifest_key] = {
            "path": str(training_manifest_destination.relative_to(out_root)),
            "sha256": _sha256(training_manifest_destination),
        }

    fit_archive = archive / "training_fit"
    fit_archive.mkdir()
    for key in ("training_analysis", "frozen_model"):
        source = Path(artifacts[key])
        destination = fit_archive / source.name
        if destination.exists():
            raise RuntimeError("route training analysis and model filenames collide")
        shutil.copy2(source, destination)
        if _sha256(destination) != receipt[key]["sha256"]:
            raise RuntimeError(f"route model-freeze {key} changed during archival")
        receipt[key] = {
            "path": str(destination.relative_to(out_root)),
            "sha256": _sha256(destination),
        }
    destination = out_root / "route_model_freeze.json"
    _atomic_write_json(destination, receipt)
    return {
        "schema_id": MODEL_FREEZE_SCHEMA_ID,
        "schema_version": MODEL_FREEZE_SCHEMA_VERSION,
        "path": destination.name,
        "sha256": _sha256(destination),
        "reviewer": str(receipt["reviewer"]),
    }


def _prepare_environment_contract(
    *, out_root: Path, env_setup: Path | None, env_lock: Path | None
) -> dict[str, object]:
    if (env_setup is None) != (env_lock is None):
        raise RuntimeError("--env-setup and --env-lock must be supplied together")
    if env_setup is None:
        return {
            "status": "unbound_preparation_only",
            "runtime_setup_path": "UNBOUND",
            "setup_sha256": "0" * 64,
            "runtime_lock_path": "UNBOUND",
            "lock_sha256": "0" * 64,
            "archived_setup": None,
            "archived_lock": None,
        }
    setup = Path(env_setup).resolve()
    lock = Path(env_lock).resolve()
    if not setup.is_file() or not lock.is_file():
        raise RuntimeError("reviewed environment setup or lock file is missing")
    archive = out_root / "environment_contract"
    archive.mkdir(parents=True, exist_ok=False)
    setup_copy = archive / "environment_setup.sh"
    lock_copy = archive / "environment.lock"
    shutil.copy2(setup, setup_copy)
    shutil.copy2(lock, lock_copy)
    setup_sha = _sha256(setup)
    lock_sha = _sha256(lock)
    if _sha256(setup_copy) != setup_sha or _sha256(lock_copy) != lock_sha:
        raise RuntimeError("environment contract changed during archival")
    return {
        "status": "hash_bound",
        "runtime_setup_path": str(setup),
        "setup_sha256": setup_sha,
        "runtime_lock_path": str(lock),
        "lock_sha256": lock_sha,
        "archived_setup": {
            "path": str(setup_copy.relative_to(out_root)),
            "sha256": setup_sha,
        },
        "archived_lock": {
            "path": str(lock_copy.relative_to(out_root)),
            "sha256": lock_sha,
        },
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
    environment_contract: dict[str, object],
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
            str(environment_contract["runtime_setup_path"]),
            str(environment_contract["setup_sha256"]),
            str(environment_contract["runtime_lock_path"]),
            str(environment_contract["lock_sha256"]),
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
            "real submission disabled: set SUBMIT_CONFIRMED=YES only after "
            "reviewing the generated plan"
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


def _validate_archived_release_authorization(
    root: Path,
    manifest: dict[str, object],
    plan: list[dict[str, str]],
) -> None:
    record = manifest.get("release_authorization")
    if not isinstance(record, dict):
        raise RuntimeError("real downstream campaign lacks its release authorization")
    path = _archive_member(root, record.get("path"), name="release_authorization.path")
    if not path.is_file() or record.get("sha256") != _sha256(path):
        raise RuntimeError("release authorization is missing or has a stale hash")
    with path.open(encoding="utf-8") as handle:
        payload = _validate_release_authorization_shape(json.load(handle))
    experiments = {row["experiment_id"] for row in plan}
    tiers = {row["tier"] for row in plan}
    if (
        payload.get("matrix_sha256") != manifest.get("matrix_sha256")
        or payload.get("source_commit") != manifest.get("source_commit")
        or {payload.get("authorizes_experiment")} != experiments
        or not tiers.issubset(set(payload.get("authorizes_tiers") or []))
        or record.get("reviewer") != payload.get("reviewer")
    ):
        raise RuntimeError("release authorization scope or source identity is stale")
    for index, artifact in enumerate(payload["reviewed_artifacts"]):
        artifact_path = _archive_member(
            root,
            artifact.get("path"),
            name=f"release_authorization.reviewed_artifacts[{index}]",
        )
        if not artifact_path.is_file() or artifact.get("sha256") != _sha256(
            artifact_path
        ):
            raise RuntimeError(f"release authorization artifact {index} is missing or stale")


def _validate_archived_model_freeze(
    root: Path, manifest: dict[str, object], *, matrix: Path
) -> None:
    record = manifest.get("route_model_freeze")
    if (
        not isinstance(record, dict)
        or record.get("schema_id") != MODEL_FREEZE_SCHEMA_ID
        or record.get("schema_version") != MODEL_FREEZE_SCHEMA_VERSION
    ):
        raise RuntimeError("rank-32 holdout lacks its route model-freeze receipt")
    path = _archive_member(root, record.get("path"), name="route_model_freeze.path")
    if not path.is_file() or record.get("sha256") != _sha256(path):
        raise RuntimeError("route model-freeze receipt is missing or has a stale hash")
    validated = _validate_model_freeze_receipt(
        path,
        matrix=matrix,
        source_commit=str(manifest.get("source_commit", "")),
    )
    receipt = dict(validated["receipt"])
    if record.get("reviewer") != receipt.get("reviewer"):
        raise RuntimeError("route model-freeze reviewer identity changed")
    if (
        manifest.get("matrix_sha256") != _sha256(matrix)
        or receipt.get("matrix_sha256") != manifest.get("matrix_sha256")
    ):
        raise RuntimeError("rank-32 holdout does not share the frozen matrix identity")

    is_tier_b = manifest.get("only_optional") is True
    if manifest.get("include_optional") is not is_tier_b:
        raise RuntimeError("rank-32 holdout mixes required and Tier-B scope")
    scope_name = "tier_b_holdout" if is_tier_b else "cost_model_holdout"
    expected_rows = _route_scopes(matrix)[scope_name]
    expected_ids = sorted(row["case_id"] for row in expected_rows)
    plan_path = _archive_member(root, manifest.get("plan_file"), name="holdout.plan")
    with plan_path.open(newline="", encoding="utf-8") as handle:
        plan = [dict(row) for row in csv.DictReader(handle)]
    canonical = {row["case_id"]: row for row in expected_rows}
    declared_scope = dict(receipt.get("scopes") or {}).get(scope_name)
    if (
        manifest.get("route_phase") != "holdout"
        or int(manifest.get("case_count", -1)) != len(expected_ids)
        or manifest.get("route_phase_case_ids_sha256")
        != _case_ids_sha256(expected_ids)
        or len(plan) != len(expected_ids)
        or len({row.get("case_id") for row in plan}) != len(expected_ids)
        or any(canonical.get(str(row.get("case_id", ""))) != row for row in plan)
        or declared_scope
        != {
            "case_count": len(expected_ids),
            "case_ids_sha256": _case_ids_sha256(expected_ids),
        }
    ):
        raise RuntimeError(f"rank-32 holdout is not the canonical {scope_name} scope")

    environment = manifest.get("environment_contract")
    receipt_environment = receipt.get("environment_identity")
    if (
        not isinstance(environment, dict)
        or environment.get("status") != "hash_bound"
        or not isinstance(receipt_environment, dict)
        or {
            "setup_sha256": environment.get("setup_sha256"),
            "lock_sha256": environment.get("lock_sha256"),
        }
        != receipt_environment
    ):
        raise RuntimeError(
            "rank-32 holdout does not share the frozen training environment identity"
        )


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
        "partial_submission",
        "submission_failed",
        "submission_reconciliation_required",
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
    environment_contract = manifest.get("environment_contract")
    if not isinstance(environment_contract, dict) or environment_contract.get(
        "status"
    ) not in {"unbound_preparation_only", "hash_bound"}:
        raise RuntimeError("prepared manifest lacks a recognized environment contract")
    if environment_contract["status"] == "hash_bound":
        for key, hash_key in (
            ("archived_setup", "setup_sha256"),
            ("archived_lock", "lock_sha256"),
        ):
            record = environment_contract.get(key)
            if not isinstance(record, dict):
                raise RuntimeError(f"environment contract lacks {key}")
            artifact = _archive_member(
                root, record.get("path"), name=f"environment_contract.{key}"
            )
            if (
                not artifact.is_file()
                or record.get("sha256") != _sha256(artifact)
                or record.get("sha256") != environment_contract.get(hash_key)
            ):
                raise RuntimeError(f"environment contract {key} is missing or stale")

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
            plan,
            only_optional=bool(manifest.get("only_optional")),
            route_phase=manifest.get("route_phase"),
        )
    route_phase = manifest.get("route_phase")
    route_rows = [row for row in plan if row["experiment_id"] == "EXP-ROUTE-001"]
    if route_phase is not None:
        if len(route_rows) != len(plan) or route_phase not in ROUTE_PHASES:
            raise RuntimeError("prepared route phase is attached to a non-route scope")
        if route_phase == "training" and any(
            int(row["total_ranks"]) == 32 for row in route_rows
        ):
            raise RuntimeError("prepared training phase contains rank-32 rows")
        if route_phase == "holdout" and any(
            int(row["total_ranks"]) != 32 for row in route_rows
        ):
            raise RuntimeError("prepared holdout phase contains training rows")
        if manifest.get("route_phase_case_ids_sha256") != _case_ids_sha256(
            row["case_id"] for row in route_rows
        ):
            raise RuntimeError("prepared route phase case-ID hash is stale")

    real_submission_record = (
        manifest.get("test_only_commands") is False
        and manifest.get("status")
        in {
            "submitting",
            "submitted",
            "partial_submission",
            "submission_failed",
            "submission_reconciliation_required",
        }
    )
    if real_submission_record:
        experiments = {row["experiment_id"] for row in plan}
        tiers = {row["tier"] for row in plan}
        gated = bool(
            experiments.intersection({"EXP-ROUTE-001", "EXP-SCALE-001"})
            or (experiments == {"EXP-DISC-001"} and tiers != {"smoke"})
        )
        if gated:
            _validate_archived_release_authorization(root, manifest, plan)
        if experiments == {"EXP-ROUTE-001"} and route_phase == "holdout":
            _validate_archived_model_freeze(root, manifest, matrix=matrix)

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
        if len(arguments) != 11:
            raise RuntimeError(f"{row['case_id']} batch argument count changed")
        if (
            Path(arguments[0]).resolve() != matrix
            or arguments[1] != row["case_id"]
            or arguments[3] != manifest.get("source_commit")
            or arguments[4] != manifest.get("matrix_sha256")
            or arguments[6] != freeze_record.get("sha256")
            or arguments[7] != environment_contract.get("runtime_setup_path")
            or arguments[8] != environment_contract.get("setup_sha256")
            or arguments[9] != environment_contract.get("runtime_lock_path")
            or arguments[10] != environment_contract.get("lock_sha256")
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
        route_phase=getattr(args, "route_phase", None),
    )
    if any(row["optional"] == "1" for row in selected):
        _validate_optional_tranche_scope(
            selected,
            only_optional=bool(args.only_optional),
            route_phase=getattr(args, "route_phase", None),
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
    environment_contract = _prepare_environment_contract(
        out_root=out_root,
        env_setup=getattr(args, "env_setup", None),
        env_lock=getattr(args, "env_lock", None),
    )
    if args.execute and environment_contract["status"] != "hash_bound":
        raise RuntimeError(
            "scheduler admission or submission requires reviewed --env-setup and --env-lock"
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
            environment_contract=environment_contract,
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
        "environment_contract": environment_contract,
        "selected_experiments": sorted({row["experiment_id"] for row in selected}),
        "selected_tiers": sorted({row["tier"] for row in selected}),
        "protocol_cards": {
            key: value
            for key, value in PROTOCOLS.items()
            if key in {row["experiment_id"] for row in selected}
        },
        "include_optional": bool(args.include_optional or args.only_optional),
        "only_optional": bool(args.only_optional),
        "route_phase": getattr(args, "route_phase", None),
        "route_phase_case_ids_sha256": (
            _case_ids_sha256(row["case_id"] for row in selected)
            if set(row["experiment_id"] for row in selected) == {"EXP-ROUTE-001"}
            and getattr(args, "route_phase", None) is not None
            else None
        ),
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
        model_freeze = None
        if (
            not args.test_only
            and set(row["experiment_id"] for row in selected) == {"EXP-ROUTE-001"}
            and getattr(args, "route_phase", None) == "holdout"
        ):
            if getattr(args, "model_freeze_receipt", None) is None:
                raise RuntimeError(
                    "rank-32 holdout submission requires --model-freeze-receipt"
                )
            model_freeze = _archive_model_freeze_receipt(
                _validate_model_freeze_receipt(
                    Path(args.model_freeze_receipt),
                    matrix=matrix,
                    source_commit=str(git["commit"]),
                ),
                out_root=out_root,
            )
        manifest["release_authorization"] = release_authorization
        manifest["route_model_freeze"] = model_freeze
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
        journal_path = out_root / "submission_journal.jsonl"
        accepted = 0
        pending_intent = False
        try:
            submitted_path.open("x", encoding="utf-8").close()
            if not args.test_only:
                journal_path.open("x", encoding="utf-8").close()
            for attempted, (row, command) in enumerate(
                zip(selected, commands, strict=True), start=1
            ):
                command_text = shlex.join(command)
                attempt_id = f"initial-{attempted:04d}-{row['case_id']}"
                if not args.test_only:
                    _append_jsonl(
                        journal_path,
                        {
                            "event": "intent",
                            "attempt_id": attempt_id,
                            "case_id": row["case_id"],
                            "command": command_text,
                            "recorded_at_utc": _utc_now(),
                        },
                    )
                    pending_intent = True
                completed = subprocess.run(
                    command, check=False, capture_output=True, text=True
                )
                record: dict[str, object] = {
                    "case_id": row["case_id"],
                    "command": command_text,
                    "returncode": int(completed.returncode),
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                }
                if args.test_only:
                    _append_jsonl(submitted_path, record)
                else:
                    journal_result = {
                        "event": "result",
                        "attempt_id": attempt_id,
                        "recorded_at_utc": _utc_now(),
                        **record,
                    }
                    if int(completed.returncode) == 0:
                        journal_result["job_id"] = _submitted_job_id(completed.stdout)
                    _append_jsonl(journal_path, journal_result)
                    pending_intent = False
                    if int(completed.returncode) == 0:
                        record["job_id"] = journal_result["job_id"]
                        _append_jsonl(submitted_path, record)
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
                manifest["status"] = "partial_submission" if accepted else "submission_failed"
                if pending_intent:
                    manifest["status"] = "submission_reconciliation_required"
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
        "--route-phase",
        choices=tuple(sorted(ROUTE_PHASES)),
        help="Separate EXP-ROUTE rank-1/8 training from rank-32 holdout rows.",
    )
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
        "--env-setup",
        type=Path,
        help="Reviewed compute environment setup script; required for scheduler use.",
    )
    parser.add_argument(
        "--env-lock",
        type=Path,
        help="Reviewed environment lock/manifest; required with --env-setup.",
    )
    parser.add_argument(
        "--admission-gate",
        type=Path,
        help="Passed JSON gate required before downstream DISC/SCALE/Tier-B tranches.",
    )
    parser.add_argument(
        "--model-freeze-receipt",
        type=Path,
        help="Hash-bound frozen training fit required before real rank-32 holdout submission.",
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

#!/usr/bin/env python3
"""Fail-closed Tier-B endpoint admission for EXP-ROUTE-001.

This analyzer consumes the paired Plasticity3D full-solve blocks prepared for
Karolina.  A timing is deliberately absent from its output until every frozen
matrix-policy, Riesz-stopping, endpoint-equivalence, repetition-coverage, and
route-order-balance gate passes.  Missing, censored, and invalid blocks remain
visible and are never imputed.

Input layout for one matrix row::

    CAMPAIGN_ROOT/cases/CASE_ID/job_JOBID/
      matrix_row.json
      run_records.json
      measure_01/
        block_result.json
        element_ad/output.json
        element_ad/state.npz
        constitutive_ad/output.json
        constitutive_ad/state.npz

``block_result.json`` has schema version 1, identifies the comparison and
block repetition, records the two-route execution order, declares
``timing_reduction=mpi_collective_max``, and stores each route's positive
``collective_max_wall_time_s``.  The route output and state paths may be
declared relative to the measure directory; otherwise the paths above are
used.
"""

from __future__ import annotations

import argparse
import csv
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

from src.core.benchmark.run_record import atomic_write_json
from experiments.analysis.analyze_plasticity3d_route_cost_model import (
    _cluster_batch_evidence,
    _git_metadata,
)


SCHEMA_ID = "fenics-nonlinear-energies.exp-route-001.tier-b-endpoints"
SCHEMA_VERSION = 1
EXPERIMENT_ID = "EXP-ROUTE-001"
ROUTE_ANALYSIS_CONTRACT = REPO_ROOT / "paper/protocols/EXP-ROUTE-001-analysis-contract.json"
with ROUTE_ANALYSIS_CONTRACT.open(encoding="utf-8") as _contract_handle:
    REVIEWED_MATRIX_SHA256 = str(
        json.load(_contract_handle)["publication_model_input_gates"][
            "karolina_matrix_sha256"
        ]
    )
ROUTES = ("element_ad", "constitutive_ad")
TIERS = ("full_solve_confirmation", "low_order_confirmation")
ROUTE_BACKENDS = {
    "element_ad": "local",
    "constitutive_ad": "local_constitutiveAD",
}

# These gates are part of the analysis implementation, rather than CLI
# parameters, so they cannot be relaxed after inspecting cluster results.
FROZEN_GATES: dict[str, Any] = {
    "expected_block_repetitions": 10,
    "expected_first_route_count": {"element_ad": 5, "constitutive_ad": 5},
    "state_relative_riesz_max": 1.0e-7,
    "state_max_absolute_max": 1.0e-8,
    "energy_relative_max": 1.0e-8,
    "energy_absolute_max": 1.0e-8,
    "work_relative_max": 1.0e-8,
    "work_absolute_max": 1.0e-8,
    "u_max_relative_max": 1.0e-7,
    "u_max_absolute_max": 1.0e-8,
    "dual_residual_relative_difference_max": 0.25,
    "relative_correction_relative_difference_max": 0.25,
    "nonlinear_iterations_exact": True,
    "total_krylov_iterations_exact": True,
    "per_solve_krylov_iterations_exact": True,
    "solver_ksp_rtol": 1.0e-8,
    "solver_ksp_max_it": 500,
    "nonlinear_max_it": 80,
    "relative_correction_target": 2.0e-3,
    "absolute_dual_residual_target": 1.0e-4,
    "riesz_ksp_type": "gmres",
    "riesz_pc_type": "hypre",
    "riesz_ksp_rtol": 1.0e-10,
    "riesz_ksp_atol": 1.0e-14,
    "riesz_ksp_max_it": 1000,
    "riesz_true_residual_rtol": 1.0e-8,
    "riesz_spd_factor_solver_type": "mumps",
    "riesz_symmetry_relative_tolerance": 1.0e-12,
    "minimum_normalized_branch_margin": 1.0e-8,
    "maximum_near_branch_fraction": 0.0,
    "bootstrap_seed": 20260710,
    "bootstrap_resamples": 10000,
    "bootstrap_confidence_level": 0.95,
    "practical_ranking_tie_ratio": 1.1,
    "minimum_order_stratum_blocks": 4,
}

EXPECTED_SCOPES = {
    "full_solve_confirmation": {
        "mesh_name": "hetero_ssr_L1",
        "element_degree": 4,
        "quadrature_rule": "tetra_24point",
        "ranks": (8, 32),
        "pmg_strategy": "same_mesh_p4_p2_p1",
    },
    "low_order_confirmation": {
        "mesh_name": "hetero_ssr_L1",
        "element_degree": 1,
        "quadrature_rule": "tetra_1point",
        "ranks": (8,),
        # The exact low-order strategy is frozen by the matrix and checked for
        # equality between the route outputs.  It is not inferred here.
        "pmg_strategy": "uniform_refined_p1_chain",
    },
}


class AdmissionError(ValueError):
    """Scientific or provenance evidence failed a frozen admission gate."""


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"nonfinite JSON token {token!r} is forbidden")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=_reject_nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=_reject_nonfinite)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path} must contain a list of JSON objects")
    return [dict(row) for row in value]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
    if str(value).strip() not in {str(converted), f"{converted}.0"} and not isinstance(
        value, int
    ):
        raise AdmissionError(f"{name} must be an integer")
    if minimum is not None and converted < minimum:
        raise AdmissionError(f"{name} must be at least {minimum}")
    return converted


def _safe_int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _exact_float(actual: object, expected: float, name: str) -> None:
    value = _finite(actual, name)
    if not math.isclose(value, expected, rel_tol=1.0e-14, abs_tol=0.0):
        raise AdmissionError(f"{name} changed from frozen value {expected!r}")


def _relative_difference(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(
        abs(float(left)), abs(float(right)), np.finfo(np.float64).tiny
    )


def _combined_scalar_gate(
    left: float, right: float, *, relative: float, absolute: float
) -> tuple[float, float, bool]:
    difference = abs(left - right)
    rel = _relative_difference(left, right)
    return difference, rel, bool(difference <= absolute + relative * max(abs(left), abs(right)))


def _normalize_route_order(value: object) -> tuple[str, str]:
    if isinstance(value, list):
        order = tuple(str(item) for item in value)
    else:
        text = str(value).strip()
        aliases = {
            "element_ad_first": ROUTES,
            "element_ad_then_constitutive_ad": ROUTES,
            "constitutive_ad_first": tuple(reversed(ROUTES)),
            "constitutive_ad_then_element_ad": tuple(reversed(ROUTES)),
        }
        if text in aliases:
            order = aliases[text]
        else:
            separator = "|" if "|" in text else ","
            order = tuple(part.strip() for part in text.split(separator) if part.strip())
    if len(order) != 2 or set(order) != set(ROUTES):
        raise AdmissionError("route_order must be one permutation of the two frozen routes")
    return order  # type: ignore[return-value]


def _load_matrix(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    selected = [
        dict(row)
        for row in all_rows
        if row.get("experiment_id") == EXPERIMENT_ID and row.get("tier") in TIERS
    ]
    violations: list[dict[str, str]] = []
    if not selected:
        violations.append(
            {"case_id": "", "reason": "no_paired_tier_b_matrix_rows"}
        )
    seen_cases: set[str] = set()
    seen_blocks: set[tuple[str, int]] = set()
    for row in selected:
        case_id = str(row.get("case_id", ""))
        try:
            _validate_matrix_row(row)
            comparison_id = str(row["comparison_id"])
            repetition = int(row["block_repetition"])
            if case_id in seen_cases:
                raise AdmissionError("duplicate case_id")
            if (comparison_id, repetition) in seen_blocks:
                raise AdmissionError("duplicate comparison_id/block_repetition")
            seen_cases.add(case_id)
            seen_blocks.add((comparison_id, repetition))
        except (AdmissionError, KeyError, TypeError, ValueError) as exc:
            violations.append({"case_id": case_id, "reason": str(exc)})
    return selected, violations


def _validate_matrix_row(row: dict[str, str]) -> None:
    tier = str(row.get("tier", ""))
    if tier not in TIERS:
        raise AdmissionError("matrix row has an unsupported endpoint tier")
    scope = EXPECTED_SCOPES[tier]
    exact = {
        "experiment_id": EXPERIMENT_ID,
        "runner": "p3d_solve_block",
        "route": "paired_element_constitutive",
        "assembly_backend": "paired_block",
        "mesh_name": scope["mesh_name"],
        "quadrature_rule": scope["quadrature_rule"],
        "solver_backend": "local_pmg",
        "maxit": str(FROZEN_GATES["nonlinear_max_it"]),
        "ksp_max_it": str(FROZEN_GATES["solver_ksp_max_it"]),
        "convergence_metric": "reference_elastic_energy",
        "state_label": "solver_initial_state",
        "optional": "1",
        "nodes": "1",
        "partition": "qcpu_exp",
        "route_order_policy": "seeded_balanced_alternating_v1",
        "timing_reduction": "mpi_collective_max",
        "probe_count": "0",
    }
    for key, expected in exact.items():
        if str(row.get(key, "")) != str(expected):
            raise AdmissionError(
                f"matrix {key}={row.get(key)!r}, expected frozen value {expected!r}"
            )
    if not str(row.get("case_id", "")):
        raise AdmissionError("case_id is required")
    if not str(row.get("comparison_id", "")):
        raise AdmissionError("comparison_id is required")
    repetition = _integer(row.get("block_repetition"), "block_repetition", minimum=1)
    if repetition > int(FROZEN_GATES["expected_block_repetitions"]):
        raise AdmissionError("block_repetition is outside the frozen 1..10 design")
    _normalize_route_order(row.get("route_order"))
    if _integer(row.get("element_degree"), "element_degree") != int(
        scope["element_degree"]
    ):
        raise AdmissionError("matrix element_degree changed from the frozen scope")
    ranks = _integer(row.get("total_ranks"), "total_ranks", minimum=1)
    if ranks not in tuple(int(item) for item in scope["ranks"]):
        raise AdmissionError("matrix rank count changed from the frozen scope")
    if _integer(row.get("ranks_per_node"), "ranks_per_node", minimum=1) != ranks:
        raise AdmissionError("paired Tier-B blocks must fit on one Karolina CPU node")
    if _integer(row.get("repetitions"), "repetitions", minimum=1) != 1:
        raise AdmissionError("each comparison-block row must run one measured repetition")
    if _integer(row.get("warmups"), "warmups", minimum=0) != 0:
        raise AdmissionError("independent comparison blocks must not embed warmup repetitions")
    _exact_float(row.get("state_amplitude"), 0.0, "matrix state_amplitude")
    _exact_float(
        row.get("ksp_rtol"), FROZEN_GATES["solver_ksp_rtol"], "matrix ksp_rtol"
    )
    _exact_float(
        row.get("stop_tol"),
        FROZEN_GATES["relative_correction_target"],
        "matrix stop_tol",
    )
    _exact_float(
        row.get("grad_stop_tol"),
        FROZEN_GATES["absolute_dual_residual_target"],
        "matrix grad_stop_tol",
    )
    if row.get("pmg_strategy") != scope["pmg_strategy"]:
        raise AdmissionError("matrix PMG strategy changed from the frozen tier policy")


def _validate_manifest(manifest_path: Path, matrix_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(manifest_path),
        "eligible": False,
        "reason": "manifest_missing",
    }
    if not manifest_path.is_file():
        return result
    try:
        payload = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result["reason"] = f"manifest_invalid: {exc}"
        return result
    result["sha256"] = _sha256_file(manifest_path)
    source_commit = str(payload.get("source_commit", ""))
    if len(source_commit) == 40 and all(
        char in "0123456789abcdef" for char in source_commit.lower()
    ):
        result["source_commit"] = source_commit
    if _sha256_file(matrix_path) != REVIEWED_MATRIX_SHA256:
        result["reason"] = "analysis_matrix_is_not_the_frozen_reviewed_matrix"
        return result
    if payload.get("status") != "submitted":
        result["reason"] = "manifest_does_not_record_real_submission"
        return result
    if payload.get("matrix_sha256") != _sha256_file(matrix_path):
        result["reason"] = "manifest_matrix_sha256_mismatch"
        return result
    if EXPERIMENT_ID not in list(payload.get("selected_experiments") or []):
        result["reason"] = "manifest_does_not_select_exp_route_001"
        return result
    if payload.get("include_optional") is not True:
        result["reason"] = "manifest_did_not_enable_optional_tier_b_blocks"
        return result
    if payload.get("only_optional") is not True:
        result["reason"] = "manifest_did_not_isolate_optional_route_tranche"
        return result
    if set(payload.get("selected_tiers") or []) != {
        "full_solve_confirmation",
        "low_order_confirmation",
    }:
        result["reason"] = "manifest_does_not_select_exact_tier_b_scope"
        return result
    if payload.get("test_only_commands") is not False:
        result["reason"] = "manifest_records_test_only_or_unknown_commands"
        return result
    if int(payload.get("case_count", 0)) != 30:
        result["reason"] = "manifest_case_count_cannot_cover_tier_b_blocks"
        return result
    if len(source_commit) != 40 or any(
        char not in "0123456789abcdef" for char in source_commit.lower()
    ):
        result["reason"] = "manifest_source_commit_missing_or_invalid"
        return result
    if payload.get("source_dirty") is not False:
        result["reason"] = "manifest_source_worktree_not_clean"
        return result
    release = payload.get("release_authorization")
    if not isinstance(release, dict):
        result["reason"] = "manifest_release_authorization_missing"
        return result
    if (
        release.get("schema_id")
        != "fenics-nonlinear-energies.human-release-authorization"
    ):
        result["reason"] = "manifest_release_authorization_schema_mismatch"
        return result
    release_relative = Path(str(release.get("path", "")))
    if release_relative.is_absolute():
        result["reason"] = "manifest_release_authorization_not_relocatable"
        return result
    try:
        release_path = _path_within(
            manifest_path.parent,
            release_relative,
            manifest_path.parent / "release_authorization.json",
        )
    except AdmissionError:
        result["reason"] = "manifest_release_authorization_path_escape"
        return result
    if (
        not release_path.is_file()
        or release.get("sha256") != _sha256_file(release_path)
        or not str(release.get("reviewer", "")).strip()
    ):
        result["reason"] = "manifest_release_authorization_hash_mismatch"
        return result
    try:
        release_record = _read_json(release_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result["reason"] = f"release_authorization_invalid: {exc}"
        return result
    if (
        release_record.get("schema_id")
        != "fenics-nonlinear-energies.human-release-authorization"
        or int(release_record.get("schema_version", -1)) != 1
        or release_record.get("status") != "approved"
        or release_record.get("decision")
        != "explicit_human_release_after_review"
        or release_record.get("matrix_sha256") != _sha256_file(matrix_path)
        or release_record.get("source_commit") != source_commit
        or release_record.get("authorizes_experiment") != EXPERIMENT_ID
        or not set(payload.get("selected_tiers") or []).issubset(
            {
                str(value)
                for value in release_record.get("authorizes_tiers") or []
            }
        )
        or not str(release_record.get("reviewer", "")).strip()
    ):
        result["reason"] = "release_authorization_scope_or_commit_mismatch"
        return result
    reviewed = release_record.get("reviewed_artifacts")
    if not isinstance(reviewed, list) or not reviewed:
        result["reason"] = "release_authorization_reviewed_artifacts_missing"
        return result
    for artifact in reviewed:
        if not isinstance(artifact, dict):
            result["reason"] = "release_authorization_reviewed_artifact_invalid"
            return result
        artifact_relative = Path(str(artifact.get("path", "")))
        if artifact_relative.is_absolute():
            result["reason"] = "release_authorization_artifact_not_relocatable"
            return result
        try:
            artifact_path = _path_within(
                release_path.parent,
                artifact_relative,
                release_path.parent / "missing-reviewed-artifact",
            )
        except AdmissionError:
            result["reason"] = "release_authorization_artifact_path_escape"
            return result
        if (
            not artifact_path.is_file()
            or artifact.get("sha256") != _sha256_file(artifact_path)
        ):
            result["reason"] = "release_authorization_artifact_hash_mismatch"
            return result
    result["release_authorization"] = {
        "path": str(release_path),
        "sha256": _sha256_file(release_path),
        "reviewer": str(release_record["reviewer"]),
    }
    result.update({"eligible": True, "reason": "submitted_reviewed_matrix"})
    result["source_commit"] = source_commit
    return result


def _path_within(base: Path, raw: object, default: Path) -> Path:
    text = str(raw or "").strip()
    if text:
        path = Path(text)
        if path.is_absolute():
            raise AdmissionError("declared artifact path must be relative to its archive")
        path = base / path
    else:
        path = default
    resolved = path.resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise AdmissionError(f"declared artifact escapes comparison block: {resolved}") from exc
    return resolved


def _validate_riesz_evidence(payload: dict[str, Any], row: dict[str, str]) -> dict[str, Any]:
    if payload.get("convergence_metric_requested") != "reference_elastic_energy":
        raise AdmissionError("output did not request reference_elastic_energy")
    if payload.get("convergence_metric") != "reference_elastic_energy":
        raise AdmissionError("output did not use reference_elastic_energy")
    convergence = dict(payload.get("nonlinear_convergence") or {})
    configuration = dict(convergence.get("configuration") or {})
    if configuration.get("selection") != "reference_elastic_energy":
        raise AdmissionError("coefficient or unknown nonlinear stopping was used")
    if configuration.get("correction_normalization") != "metric_current_state":
        raise AdmissionError("correction did not use metric-current-state normalization")
    if configuration.get("state_scale_source") != "initial_nonlinear_iterate_primal_norm":
        raise AdmissionError("Riesz state scale did not come from the initial nonlinear iterate")
    state_scale = _finite(configuration.get("state_scale"), "Riesz state scale")
    if state_scale <= 0.0:
        raise AdmissionError("Riesz state scale must be positive")

    requested = dict(payload.get("riesz_solver_requested") or {})
    requested_exact = {
        "ksp_type": FROZEN_GATES["riesz_ksp_type"],
        "pc_type": FROZEN_GATES["riesz_pc_type"],
        "max_it": FROZEN_GATES["riesz_ksp_max_it"],
        "spd_factor_solver_type": FROZEN_GATES["riesz_spd_factor_solver_type"],
    }
    for key, expected in requested_exact.items():
        if requested.get(key) != expected:
            raise AdmissionError(f"requested Riesz {key} changed from {expected!r}")
    for key, expected in {
        "rtol": FROZEN_GATES["riesz_ksp_rtol"],
        "atol": FROZEN_GATES["riesz_ksp_atol"],
        "true_residual_rtol": FROZEN_GATES["riesz_true_residual_rtol"],
        "symmetry_relative_tolerance": FROZEN_GATES[
            "riesz_symmetry_relative_tolerance"
        ],
    }.items():
        _exact_float(requested.get(key), expected, f"requested Riesz {key}")

    metric = dict(convergence.get("metric") or {})
    if metric.get("name") != "plasticity3d_reference_elastic_energy":
        raise AdmissionError("unexpected Riesz metric identity")
    if metric.get("riesz_operator") != "petsc_matrix":
        raise AdmissionError("Riesz metric is not a PETSc matrix operator")
    if metric.get("set_from_petsc_options") is not False:
        raise AdmissionError("Riesz solver allowed PETSc option overrides")
    if str(metric.get("petsc_options_prefix", "")):
        raise AdmissionError("Riesz solver options prefix must remain empty")
    expected_metric = {
        "ksp_type": FROZEN_GATES["riesz_ksp_type"],
        "pc_type": FROZEN_GATES["riesz_pc_type"],
        "requested_max_it": FROZEN_GATES["riesz_ksp_max_it"],
        "effective_max_it": FROZEN_GATES["riesz_ksp_max_it"],
    }
    for key, expected in expected_metric.items():
        if metric.get(key) != expected:
            raise AdmissionError(f"effective Riesz {key} changed from {expected!r}")
    for key, expected in {
        "requested_rtol": FROZEN_GATES["riesz_ksp_rtol"],
        "effective_rtol": FROZEN_GATES["riesz_ksp_rtol"],
        "requested_atol": FROZEN_GATES["riesz_ksp_atol"],
        "effective_atol": FROZEN_GATES["riesz_ksp_atol"],
        "true_residual_rtol_gate": FROZEN_GATES["riesz_true_residual_rtol"],
    }.items():
        _exact_float(metric.get(key), expected, f"effective Riesz {key}")

    provenance = dict(metric.get("provenance") or {})
    if provenance.get("problem") != "Plasticity3D":
        raise AdmissionError("Riesz provenance has the wrong problem")
    if provenance.get("operator_source") != "elastic_tangent_at_zero_displacement":
        raise AdmissionError("Riesz operator is not the frozen zero-state elastic tangent")
    if provenance.get("constitutive_mode") != "elastic":
        raise AdmissionError("Riesz operator provenance is not elastic")
    if provenance.get("reference_operator_tangent_mode") != "element_ad_for_all_routes":
        raise AdmissionError("Riesz operator was not built by the common reference route")
    if provenance.get("constraint_variant") != "glued_bottom":
        raise AdmissionError("Riesz operator does not use the glued-bottom free space")
    if provenance.get("free_space") != "glued_free_dofs":
        raise AdmissionError("Riesz operator provenance has the wrong constrained free space")
    if provenance.get("ordering") != "backend_mix_reordered_constrained_free_dofs":
        raise AdmissionError("Riesz operator provenance has the wrong DOF ordering")
    if provenance.get("ownership") != "petsc_distributed_rows":
        raise AdmissionError("Riesz operator provenance has the wrong ownership model")
    if provenance.get("mesh_name") != row.get("mesh_name"):
        raise AdmissionError("Riesz mesh provenance disagrees with the matrix")
    if int(provenance.get("element_degree", -1)) != int(row["element_degree"]):
        raise AdmissionError("Riesz element-degree provenance disagrees with the matrix")
    if provenance.get("quadrature_rule_id") != row.get("quadrature_rule"):
        raise AdmissionError("Riesz quadrature provenance disagrees with the matrix")

    free_dofs = _integer(provenance.get("free_dofs"), "Riesz free_dofs", minimum=1)
    owned = _integer(
        dict(payload.get("parallel_setup") or {}).get("owned_free_dofs_sum"),
        "distributed free_dofs",
        minimum=1,
    )
    certificate = dict(provenance.get("spd_certificate") or {})
    inertia = dict(certificate.get("inertia") or {})
    if certificate.get("certified_spd") is not True:
        raise AdmissionError("Riesz operator lacks an SPD certificate")
    if certificate.get("method") != "symmetric_direct_factorization_inertia":
        raise AdmissionError("Riesz SPD certificate used an unknown method")
    if certificate.get("symmetry_checked") is not True:
        raise AdmissionError("Riesz SPD certificate did not explicitly check symmetry")
    if str(certificate.get("factor_solver_type", "")).lower() != "mumps":
        raise AdmissionError("Riesz SPD certificate did not use MUMPS")
    _exact_float(
        certificate.get("symmetry_relative_tolerance"),
        FROZEN_GATES["riesz_symmetry_relative_tolerance"],
        "Riesz certificate symmetry tolerance",
    )
    if (
        _integer(certificate.get("matrix_rows"), "Riesz certificate rows", minimum=1)
        != free_dofs
        or _integer(
            certificate.get("matrix_columns"),
            "Riesz certificate columns",
            minimum=1,
        )
        != free_dofs
    ):
        raise AdmissionError("Riesz SPD certificate has the wrong matrix dimensions")
    matrix_norm = _finite(
        certificate.get("matrix_infinity_norm"),
        "Riesz certificate matrix infinity norm",
        nonnegative=True,
    )
    symmetry_absolute = _finite(
        certificate.get("symmetry_absolute_tolerance"),
        "Riesz certificate absolute symmetry tolerance",
        nonnegative=True,
    )
    if not math.isclose(
        symmetry_absolute,
        FROZEN_GATES["riesz_symmetry_relative_tolerance"]
        * max(1.0, matrix_norm),
        rel_tol=1.0e-12,
        abs_tol=1.0e-18,
    ):
        raise AdmissionError("Riesz SPD symmetry tolerances are internally inconsistent")
    if (
        _integer(inertia.get("negative"), "negative inertia") != 0
        or _integer(inertia.get("zero"), "zero inertia") != 0
        or _integer(inertia.get("positive"), "positive inertia", minimum=1)
        != free_dofs
        or free_dofs != owned
    ):
        raise AdmissionError("Riesz inertia does not certify the complete distributed free space")

    identity = dict(provenance.get("input_identity") or {})
    array_hashes = dict(identity.get("array_sha256") or {})
    for name in (
        "nodes",
        "elements_scalar",
        "material_id",
        "free_dofs",
        "free_mask",
        "free_dof_permutation",
    ):
        if not _is_sha256(array_hashes.get(name)):
            raise AdmissionError(f"Riesz input hash {name!r} is missing or invalid")
    hdf5 = dict(identity.get("hdf5") or {})
    dataset_hashes = dict(hdf5.get("dataset_sha256") or {})
    for name in ("shear_q", "bulk_q", "lame_q", "quad_weight"):
        if not _is_sha256(dataset_hashes.get(name)):
            raise AdmissionError(f"Riesz HDF5 hash {name!r} is missing or invalid")
    tangent = dict(identity.get("tangent_route") or {})
    if tangent.get("constitutive_mode") != "elastic":
        raise AdmissionError("Riesz tangent route is not explicitly elastic")

    initial = _finite(
        dict(convergence.get("initial_absolute_dual_residual") or {}).get("value"),
        "initial absolute dual residual",
        nonnegative=True,
    )
    residual = _finite(
        dict(convergence.get("absolute_dual_residual") or {}).get("value"),
        "terminal absolute dual residual",
        nonnegative=True,
    )
    state_norm = _finite(
        dict(convergence.get("state_norm") or {}).get("value"),
        "terminal Riesz state norm",
        nonnegative=True,
    )
    correction = _finite(
        dict(convergence.get("relative_correction") or {}).get("value"),
        "terminal relative correction",
        nonnegative=True,
    )
    coefficient_gradient = _finite(
        convergence.get("coefficient_gradient_l2"),
        "terminal coefficient-gradient norm",
        nonnegative=True,
    )
    root_gradient = _finite(
        payload.get("final_grad_norm"), "root coefficient-gradient norm", nonnegative=True
    )
    if not math.isclose(
        coefficient_gradient, root_gradient, rel_tol=1.0e-12, abs_tol=1.0e-14
    ):
        raise AdmissionError("root and convergence coefficient-gradient norms disagree")

    last = dict(convergence.get("last_riesz_solve") or {})
    if last.get("riesz_solve") != "iterative":
        raise AdmissionError("terminal dual norm did not use an audited iterative Riesz solve")
    if _integer(last.get("reason"), "terminal Riesz reason") <= 0:
        raise AdmissionError("terminal Riesz solve did not converge")
    if _integer(last.get("iterations"), "terminal Riesz iterations", minimum=0) > int(
        FROZEN_GATES["riesz_ksp_max_it"]
    ):
        raise AdmissionError("terminal Riesz solve exceeded the frozen cap")
    for key, expected in {
        "ksp_type": FROZEN_GATES["riesz_ksp_type"],
        "pc_type": FROZEN_GATES["riesz_pc_type"],
        "requested_max_it": FROZEN_GATES["riesz_ksp_max_it"],
        "effective_max_it": FROZEN_GATES["riesz_ksp_max_it"],
    }.items():
        if last.get(key) != expected:
            raise AdmissionError(f"terminal Riesz {key} changed from {expected!r}")
    for key, expected in {
        "requested_rtol": FROZEN_GATES["riesz_ksp_rtol"],
        "effective_rtol": FROZEN_GATES["riesz_ksp_rtol"],
        "requested_atol": FROZEN_GATES["riesz_ksp_atol"],
        "effective_atol": FROZEN_GATES["riesz_ksp_atol"],
        "true_residual_rtol_gate": FROZEN_GATES["riesz_true_residual_rtol"],
    }.items():
        _exact_float(last.get(key), expected, f"terminal Riesz {key}")
    true_relative = _finite(
        last.get("relative_true_residual"), "terminal Riesz true residual", nonnegative=True
    )
    if true_relative > FROZEN_GATES["riesz_true_residual_rtol"]:
        raise AdmissionError("terminal Riesz solve failed its independent true-residual gate")
    rhs_norm = _finite(last.get("rhs_norm"), "terminal Riesz rhs norm", nonnegative=True)
    if not math.isclose(rhs_norm, root_gradient, rel_tol=1.0e-12, abs_tol=1.0e-14):
        raise AdmissionError("terminal Riesz evidence is stale relative to the final gradient")
    residual_gate = dict(convergence.get("residual_gate") or {})
    if residual_gate.get("passed") is not True:
        raise AdmissionError("completed output did not pass its nonlinear residual gate")
    _exact_float(
        residual_gate.get("absolute_tolerance"),
        FROZEN_GATES["absolute_dual_residual_target"],
        "nonlinear absolute residual target",
    )
    effective = _finite(
        residual_gate.get("effective_absolute_target"),
        "effective nonlinear residual target",
        nonnegative=True,
    )
    if residual >= effective or residual >= FROZEN_GATES["absolute_dual_residual_target"]:
        raise AdmissionError("terminal dual residual is not below the frozen target")
    if correction >= FROZEN_GATES["relative_correction_target"]:
        raise AdmissionError("terminal relative correction is not below the frozen target")
    return {
        "initial_absolute_dual_residual": initial,
        "absolute_dual_residual": residual,
        "state_norm": state_norm,
        "relative_correction": correction,
        "state_scale": state_scale,
        "coefficient_gradient_l2": coefficient_gradient,
        "relative_true_residual": true_relative,
        "free_dofs": free_dofs,
        "provenance_invariants": {
            "array_sha256": array_hashes,
            "hdf5_dataset_sha256": dataset_hashes,
            "hdf5_size_bytes": hdf5.get("size_bytes"),
            "hdf5_path": hdf5.get("path"),
            "assembly_backend": provenance.get("assembly_backend"),
            "matrix_type": provenance.get("matrix_type"),
            "matrix_nonzeros": provenance.get("matrix_nonzeros"),
            "material_parameter_ranges": provenance.get("material_parameter_ranges"),
            "reference_operator_tangent_mode": provenance.get(
                "reference_operator_tangent_mode"
            ),
            "tangent_assembly_backend": tangent.get("assembly_backend"),
        },
        "route_evidence": {
            "backend_mix_route": provenance.get("backend_mix_route"),
            "autodiff_tangent_mode": provenance.get("autodiff_tangent_mode"),
            "local_hessian_mode": provenance.get("local_hessian_mode"),
            "tangent_route": tangent,
        },
    }


def _validate_linear_work(payload: dict[str, Any]) -> dict[str, Any]:
    def validate_effective(value: object, label: str) -> dict[str, Any]:
        effective = dict(value or {})
        if effective.get("captured_after_set_from_options") is not True:
            raise AdmissionError(f"{label} lacks post-options effective KSP evidence")
        if effective.get("ksp_type") not in {"fgmres", "stcg"}:
            raise AdmissionError(f"{label} effective KSP type is outside the frozen policy")
        if effective.get("pc_type") not in {"mg", "hypre"}:
            raise AdmissionError(f"{label} effective PC type is outside the frozen policy")
        _exact_float(
            effective.get("rtol"),
            FROZEN_GATES["solver_ksp_rtol"],
            f"{label} effective KSP rtol",
        )
        if _integer(effective.get("max_it"), f"{label} effective KSP cap") != int(
            FROZEN_GATES["solver_ksp_max_it"]
        ):
            raise AdmissionError(f"{label} effective KSP cap changed")
        return effective

    nit = _integer(payload.get("nit"), "nonlinear iteration count", minimum=0)
    if nit > int(FROZEN_GATES["nonlinear_max_it"]):
        raise AdmissionError("nonlinear iteration count exceeded the frozen cap")
    rows = payload.get("linear_history")
    if not isinstance(rows, list) or not rows:
        raise AdmissionError("linear_history is missing")
    sequence: list[int] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise AdmissionError("linear_history contains a non-object")
        its = _integer(raw.get("ksp_its"), f"linear_history[{index}].ksp_its", minimum=0)
        reason = _integer(raw.get("ksp_reason_code"), f"linear_history[{index}].reason")
        if its > int(FROZEN_GATES["solver_ksp_max_it"]):
            raise AdmissionError("a nonlinear KSP solve exceeded the frozen cap")
        if reason <= 0:
            raise AdmissionError("a nonlinear KSP solve did not report convergence")
        validate_effective(raw.get("effective_ksp"), f"linear_history[{index}]")
        sequence.append(its)
    total = _integer(payload.get("linear_iterations_total"), "total Krylov iterations", minimum=0)
    if total != sum(sequence):
        raise AdmissionError("total Krylov iterations disagree with linear_history")
    initial = dict(payload.get("initial_guess") or {})
    if initial.get("success") is not True:
        raise AdmissionError("elastic initial guess did not converge")
    initial_its = _integer(initial.get("ksp_iterations"), "initial-guess Krylov iterations", minimum=0)
    if initial_its > int(FROZEN_GATES["solver_ksp_max_it"]):
        raise AdmissionError("initial-guess KSP exceeded the frozen cap")
    reason = initial.get("ksp_reason_code")
    if reason is not None and _integer(reason, "initial-guess KSP reason") <= 0:
        raise AdmissionError("initial-guess KSP did not report convergence")
    validate_effective(initial.get("effective_ksp"), "initial guess")
    return {
        "nonlinear_iterations": nit,
        "krylov_iterations_total": total,
        "krylov_iterations_per_solve": sequence,
        "initial_guess_krylov_iterations": initial_its,
    }


def _np_scalar(archive: Any, name: str) -> Any:
    if name not in archive:
        raise AdmissionError(f"state archive lacks {name!r}")
    value = np.asarray(archive[name])
    if value.size != 1:
        raise AdmissionError(f"state metadata {name!r} is not scalar")
    return value.reshape(()).item()


def _load_state(path: Path, payload: dict[str, Any], row: dict[str, str], route: str) -> dict[str, Any]:
    if not path.is_file():
        raise AdmissionError(f"state archive is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required_arrays = (
            "coords_ref",
            "coords_final",
            "displacement",
            "free_displacement_reordered",
            "reference_elastic_action",
            "reference_elastic_state_quadratic",
            "tetrahedra",
            "surface_faces",
            "boundary_label",
        )
        for name in required_arrays:
            if name not in archive:
                raise AdmissionError(f"state archive lacks {name!r}")
        coords_ref = np.asarray(archive["coords_ref"], dtype=np.float64)
        coords_final = np.asarray(archive["coords_final"], dtype=np.float64)
        displacement = np.asarray(archive["displacement"], dtype=np.float64)
        free_displacement = np.asarray(
            archive["free_displacement_reordered"], dtype=np.float64
        ).reshape(-1)
        elastic_action = np.asarray(
            archive["reference_elastic_action"], dtype=np.float64
        ).reshape(-1)
        elastic_quadratic = _finite(
            _np_scalar(archive, "reference_elastic_state_quadratic"),
            "reference-elastic state quadratic",
            nonnegative=True,
        )
        tetrahedra = np.asarray(archive["tetrahedra"])
        surface_faces = np.asarray(archive["surface_faces"])
        boundary_label = np.asarray(archive["boundary_label"])
        if coords_ref.ndim != 2 or coords_ref.shape[1] != 3 or coords_ref.shape != coords_final.shape:
            raise AdmissionError("state coordinates have an invalid shape")
        if displacement.shape != coords_ref.shape:
            raise AdmissionError("state displacement shape disagrees with coordinates")
        if not all(np.all(np.isfinite(array)) for array in (coords_ref, coords_final, displacement)):
            raise AdmissionError("state contains nonfinite floating-point values")
        if (
            free_displacement.size == 0
            or free_displacement.shape != elastic_action.shape
            or not np.all(np.isfinite(free_displacement))
            or not np.all(np.isfinite(elastic_action))
        ):
            raise AdmissionError("reference-elastic state/action arrays are malformed")
        recomputed_quadratic = float(np.dot(free_displacement, elastic_action))
        if not math.isclose(
            recomputed_quadratic,
            elastic_quadratic,
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        ):
            raise AdmissionError("reference-elastic state quadratic is internally inconsistent")
        if not np.array_equal(displacement, coords_final - coords_ref):
            raise AdmissionError("saved displacement is not exactly coords_final-coords_ref")
        if tetrahedra.ndim != 2 or tetrahedra.size == 0:
            raise AdmissionError("state tetrahedral topology is empty or malformed")
        if not np.issubdtype(tetrahedra.dtype, np.integer):
            raise AdmissionError("state tetrahedral topology is not integral")
        if int(np.min(tetrahedra)) < 0 or int(np.max(tetrahedra)) >= coords_ref.shape[0]:
            raise AdmissionError("state tetrahedral topology indexes outside the coordinate array")
        if surface_faces.ndim != 2 or boundary_label.shape != (surface_faces.shape[0],):
            raise AdmissionError("state surface topology and boundary labels disagree")
        if _np_scalar(archive, "mesh_name") != row["mesh_name"]:
            raise AdmissionError("state mesh metadata disagrees with the matrix")
        if int(_np_scalar(archive, "element_degree")) != int(row["element_degree"]):
            raise AdmissionError("state element degree disagrees with the matrix")
        if _np_scalar(archive, "quadrature_rule_id") != row["quadrature_rule"]:
            raise AdmissionError("state quadrature rule disagrees with the matrix")
        if _np_scalar(archive, "constraint_variant") != "glued_bottom":
            raise AdmissionError("state does not use the glued-bottom constraint")
        if int(_np_scalar(archive, "mpi_ranks")) != int(row["total_ranks"]):
            raise AdmissionError("state rank count disagrees with the matrix")
        if _np_scalar(archive, "assembly_backend") != ROUTE_BACKENDS[route]:
            raise AdmissionError("state route metadata disagrees with its route directory")
        _exact_float(_np_scalar(archive, "lambda_target"), 1.55, "state load factor")
        state_energy = _finite(_np_scalar(archive, "energy"), "state energy")
    energy = _finite(payload.get("energy"), "root energy")
    if not math.isclose(state_energy, energy, rel_tol=1.0e-13, abs_tol=1.0e-13):
        raise AdmissionError("state and output energies disagree")
    u_max_recomputed = float(np.max(np.linalg.norm(displacement, axis=1)))
    u_max = _finite(payload.get("u_max"), "root maximum displacement", nonnegative=True)
    if not math.isclose(u_max_recomputed, u_max, rel_tol=1.0e-12, abs_tol=1.0e-13):
        raise AdmissionError("state and output maximum displacements disagree")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "coords_ref": coords_ref,
        "displacement": displacement,
        "free_displacement_reordered": free_displacement,
        "reference_elastic_action": elastic_action,
        "reference_elastic_state_quadratic": elastic_quadratic,
        "tetrahedra": tetrahedra,
        "surface_faces": surface_faces,
        "boundary_label": boundary_label,
    }


def _validate_branch_diagnostics(payload: dict[str, Any], route: str) -> dict[str, Any]:
    diagnostics = dict(payload.get("branch_diagnostics") or {})
    if diagnostics.get("definition") != "mohr_coulomb_owned_quadrature_branch_v2":
        raise AdmissionError(f"{route} lacks the frozen endpoint branch diagnostic")
    labels = ("elastic", "shear", "left_edge", "right_edge", "apex")
    counts_raw = dict(diagnostics.get("counts") or {})
    if set(counts_raw) != set(labels):
        raise AdmissionError(f"{route} endpoint branch counts have the wrong labels")
    counts = {
        label: _integer(counts_raw.get(label), f"{route} {label} branch count", minimum=0)
        for label in labels
    }
    total = _integer(
        diagnostics.get("owned_quadrature_points"),
        f"{route} owned endpoint quadrature points",
        minimum=1,
    )
    if sum(counts.values()) != total:
        raise AdmissionError(f"{route} endpoint branch counts do not sum to the owned total")
    margin = _finite(
        diagnostics.get("normalized_boundary_margin_min"),
        f"{route} normalized endpoint branch margin",
        nonnegative=True,
    )
    _exact_float(
        diagnostics.get("near_boundary_threshold"),
        FROZEN_GATES["minimum_normalized_branch_margin"],
        f"{route} near-branch threshold",
    )
    near = _finite(
        diagnostics.get("near_boundary_fraction"),
        f"{route} near-branch fraction",
        nonnegative=True,
    )
    if margin < float(FROZEN_GATES["minimum_normalized_branch_margin"]):
        raise AdmissionError(f"{route} endpoint lies too close to a branch boundary")
    if near > float(FROZEN_GATES["maximum_near_branch_fraction"]):
        raise AdmissionError(f"{route} endpoint contains near-boundary quadrature points")
    if (
        diagnostics.get("canonical_map_definition")
        != "global_element_id_then_quadrature_index_int8_v1"
        or not _is_sha256(diagnostics.get("canonical_map_sha256"))
    ):
        raise AdmissionError(f"{route} lacks a canonical pointwise branch-map hash")
    return {
        "counts": counts,
        "owned_quadrature_points": total,
        "minimum_margin": margin,
        "canonical_map_sha256": str(diagnostics["canonical_map_sha256"]),
    }


def _validate_route_output(
    path: Path,
    state_path: Path,
    row: dict[str, str],
    route: str,
    *,
    expected_commit: str,
    expected_job_id: str,
) -> dict[str, Any]:
    payload = _read_json(path)
    exact = {
        "status": "completed",
        "solver_success": True,
        "assembly_backend": ROUTE_BACKENDS[route],
        "solver_backend": "local_pmg",
        "mesh_name": row["mesh_name"],
        "elem_degree": int(row["element_degree"]),
        "quadrature_rule_id": row["quadrature_rule"],
        "constraint_variant": "glued_bottom",
        "pmg_strategy": row["pmg_strategy"],
        "ranks": int(row["total_ranks"]),
        "maxit": int(row["maxit"]),
        "ksp_max_it": int(row["ksp_max_it"]),
        "line_search": "armijo",
        "use_trust_region": True,
        "trust_subproblem_line_search": True,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise AdmissionError(
                f"{route} output {key}={payload.get(key)!r}, expected {expected!r}"
            )
    git = dict(payload.get("git") or {})
    if git.get("dirty") is not False or git.get("commit") != expected_commit:
        raise AdmissionError(f"{route} output does not match the clean prepared source commit")
    job_metadata = dict(payload.get("job_metadata") or {})
    if str(job_metadata.get("slurm_job_id", "")) != str(expected_job_id):
        raise AdmissionError(f"{route} output Slurm job identity is missing or stale")
    if "converged" not in str(payload.get("message", "")).lower():
        raise AdmissionError(f"{route} output does not carry a converged terminal message")
    _exact_float(payload.get("lambda_target"), 1.55, f"{route} load factor")
    _exact_float(payload.get("ksp_rtol"), FROZEN_GATES["solver_ksp_rtol"], f"{route} KSP rtol")
    _exact_float(payload.get("stop_tol"), FROZEN_GATES["relative_correction_target"], f"{route} correction target")
    _exact_float(payload.get("grad_stop_tol"), FROZEN_GATES["absolute_dual_residual_target"], f"{route} residual target")
    _exact_float(payload.get("linesearch_tol"), 1.0e-3, f"{route} line-search tolerance")
    expected_quadrature_points = 24 if int(row["element_degree"]) == 4 else 1
    if _integer(payload.get("quadrature_points"), f"{route} quadrature points") != expected_quadrature_points:
        raise AdmissionError(f"{route} output has the wrong number of quadrature points")
    state_out = Path(str(payload.get("state_out", "")))
    if state_out.is_absolute():
        raise AdmissionError(f"{route} state_out must be relative to its output record")
    declared_state = (path.parent / state_out).resolve()
    try:
        declared_state.relative_to(path.parent.resolve())
    except ValueError as exc:
        raise AdmissionError(f"{route} state_out escapes its output record") from exc
    if declared_state != state_path.resolve():
        raise AdmissionError(f"{route} output points to a stale or mislabeled state archive")
    energy = _finite(payload.get("energy"), f"{route} energy")
    work = _finite(payload.get("omega"), f"{route} external work")
    u_max = _finite(payload.get("u_max"), f"{route} u_max", nonnegative=True)
    internal_time = _finite(payload.get("total_time"), f"{route} internal total time")
    if internal_time <= 0.0:
        raise AdmissionError(f"{route} internal total time must be positive")
    if payload.get("total_time_reduction") != "mpi_collective_max":
        raise AdmissionError(f"{route} solver did not use collective-max total timing")
    raw_rank_times = payload.get("total_time_by_rank_s")
    if not isinstance(raw_rank_times, list) or len(raw_rank_times) != int(row["total_ranks"]):
        raise AdmissionError(f"{route} solver lacks one wall time per MPI rank")
    rank_times = np.asarray(raw_rank_times, dtype=np.float64)
    if not np.all(np.isfinite(rank_times)) or np.any(rank_times <= 0.0):
        raise AdmissionError(f"{route} solver rank wall times must be finite and positive")
    if not math.isclose(
        internal_time, float(np.max(rank_times)), rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise AdmissionError(f"{route} solver collective maximum disagrees with rank timings")
    riesz = _validate_riesz_evidence(payload, row)
    work_counts = _validate_linear_work(payload)
    branch = _validate_branch_diagnostics(payload, route)
    state = _load_state(state_path, payload, row, route)
    return {
        "output_path": str(path),
        "output_sha256": _sha256_file(path),
        "state_path": str(state_path),
        "state_sha256": state["sha256"],
        "energy": energy,
        "work": work,
        "u_max": u_max,
        "riesz": riesz,
        "work_counts": work_counts,
        "branch_diagnostics": branch,
        "solver_rank_wall_times_s": rank_times,
        "solver_collective_max_wall_time_s": internal_time,
        "state": state,
    }


def _route_provenance_gate(left: dict[str, Any], right: dict[str, Any]) -> None:
    left_inv = left["riesz"]["provenance_invariants"]
    right_inv = right["riesz"]["provenance_invariants"]
    if left_inv != right_inv:
        raise AdmissionError("route Riesz operators do not share identical input identities")
    for route, record in zip(ROUTES, (left, right), strict=True):
        evidence = record["riesz"]["route_evidence"]
        expected_mode = "element" if route == "element_ad" else "constitutive"
        if evidence.get("backend_mix_route") != ROUTE_BACKENDS[route]:
            raise AdmissionError(f"{route} Riesz provenance has the wrong backend-mix route")
        if evidence.get("autodiff_tangent_mode") != expected_mode:
            raise AdmissionError(f"{route} Riesz provenance has the wrong AD tangent mode")
        if evidence.get("local_hessian_mode") != "element":
            raise AdmissionError(f"{route} Riesz provenance has the wrong local Hessian mode")
        tangent = dict(evidence.get("tangent_route") or {})
        if tangent.get("autodiff_tangent_mode") != "element":
            raise AdmissionError(f"{route} Riesz operator did not use the common element route")
        if tangent.get("reference_operator_forced_common") is not True:
            raise AdmissionError(f"{route} Riesz operator lacks the common-route marker")
        if tangent.get("solve_route_autodiff_tangent_mode") != expected_mode:
            raise AdmissionError(f"{route} Riesz solve-route identity is stale")
        if tangent.get("local_hessian_mode") != "element":
            raise AdmissionError(f"{route} Riesz tangent-route local mode is stale")


def _endpoint_equivalence(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    left = records[ROUTES[0]]
    right = records[ROUTES[1]]
    _route_provenance_gate(left, right)
    ls = left["state"]
    rs = right["state"]
    for name in ("coords_ref", "tetrahedra", "surface_faces", "boundary_label"):
        if not np.array_equal(ls[name], rs[name]):
            raise AdmissionError(f"route state archives disagree in {name}")
    left_u = np.asarray(ls["free_displacement_reordered"], dtype=np.float64)
    right_u = np.asarray(rs["free_displacement_reordered"], dtype=np.float64)
    left_action = np.asarray(ls["reference_elastic_action"], dtype=np.float64)
    right_action = np.asarray(rs["reference_elastic_action"], dtype=np.float64)
    if left_u.shape != right_u.shape or left_action.shape != right_action.shape:
        raise AdmissionError("route reference-elastic state/action dimensions disagree")
    difference = right_u - left_u
    difference_action = right_action - left_action
    distance_squared = float(np.dot(difference, difference_action))
    quadratic_scale = max(
        float(ls["reference_elastic_state_quadratic"]),
        float(rs["reference_elastic_state_quadratic"]),
        np.finfo(np.float64).tiny,
    )
    if distance_squared < -1.0e-12 * quadratic_scale:
        raise AdmissionError("reference-elastic endpoint difference has a negative quadratic")
    distance = float(np.sqrt(max(0.0, distance_squared)))
    left_norm = float(np.sqrt(max(0.0, ls["reference_elastic_state_quadratic"])))
    right_norm = float(np.sqrt(max(0.0, rs["reference_elastic_state_quadratic"])))
    state_relative = distance / max(left_norm, right_norm, np.finfo(np.float64).tiny)
    physical_difference = np.asarray(rs["displacement"], dtype=np.float64) - np.asarray(
        ls["displacement"], dtype=np.float64
    )
    state_max = float(np.max(np.abs(physical_difference)))
    if state_relative > FROZEN_GATES["state_relative_riesz_max"]:
        raise AdmissionError("endpoint state relative reference-elastic Riesz gate failed")
    if state_max > FROZEN_GATES["state_max_absolute_max"]:
        raise AdmissionError("endpoint state maximum-absolute gate failed")
    if left["branch_diagnostics"]["counts"] != right["branch_diagnostics"]["counts"]:
        raise AdmissionError("endpoint active-branch counts differ between routes")
    if (
        left["branch_diagnostics"]["owned_quadrature_points"]
        != right["branch_diagnostics"]["owned_quadrature_points"]
    ):
        raise AdmissionError("endpoint owned quadrature-point counts differ between routes")
    if (
        left["branch_diagnostics"]["canonical_map_sha256"]
        != right["branch_diagnostics"]["canonical_map_sha256"]
    ):
        raise AdmissionError("endpoint pointwise branch maps differ between routes")

    scalar_gates: dict[str, Any] = {}
    for name, relative_key, absolute_key in (
        ("energy", "energy_relative_max", "energy_absolute_max"),
        ("work", "work_relative_max", "work_absolute_max"),
        ("u_max", "u_max_relative_max", "u_max_absolute_max"),
    ):
        absolute, relative, passed = _combined_scalar_gate(
            float(left[name]),
            float(right[name]),
            relative=float(FROZEN_GATES[relative_key]),
            absolute=float(FROZEN_GATES[absolute_key]),
        )
        scalar_gates[name] = {
            "absolute_difference": absolute,
            "relative_difference": relative,
            "passed": passed,
        }
        if not passed:
            raise AdmissionError(f"endpoint {name} equivalence gate failed")

    residual_relative = _relative_difference(
        left["riesz"]["absolute_dual_residual"],
        right["riesz"]["absolute_dual_residual"],
    )
    correction_relative = _relative_difference(
        left["riesz"]["relative_correction"],
        right["riesz"]["relative_correction"],
    )
    if residual_relative > FROZEN_GATES["dual_residual_relative_difference_max"]:
        raise AdmissionError("route terminal dual residuals differ beyond the frozen gate")
    if correction_relative > FROZEN_GATES[
        "relative_correction_relative_difference_max"
    ]:
        raise AdmissionError("route terminal corrections differ beyond the frozen gate")
    left_work = left["work_counts"]
    right_work = right["work_counts"]
    if left_work["nonlinear_iterations"] != right_work["nonlinear_iterations"]:
        raise AdmissionError("route nonlinear iteration counts differ")
    if left_work["krylov_iterations_total"] != right_work["krylov_iterations_total"]:
        raise AdmissionError("route total Krylov iteration counts differ")
    if left_work["krylov_iterations_per_solve"] != right_work[
        "krylov_iterations_per_solve"
    ]:
        raise AdmissionError("route per-solve Krylov iteration sequences differ")
    if left_work["initial_guess_krylov_iterations"] != right_work[
        "initial_guess_krylov_iterations"
    ]:
        raise AdmissionError("route initial-guess Krylov iteration counts differ")
    return {
        "state_relative_reference_elastic_riesz": state_relative,
        "state_reference_elastic_riesz_distance": distance,
        "state_max_absolute": state_max,
        "branch_counts": left["branch_diagnostics"]["counts"],
        "branch_map_sha256": left["branch_diagnostics"]["canonical_map_sha256"],
        "scalars": scalar_gates,
        "dual_residual_relative_difference": residual_relative,
        "relative_correction_relative_difference": correction_relative,
        "nonlinear_iterations": left_work["nonlinear_iterations"],
        "total_krylov_iterations": left_work["krylov_iterations_total"],
        "per_solve_krylov_iterations": left_work["krylov_iterations_per_solve"],
        "initial_guess_krylov_iterations": left_work[
            "initial_guess_krylov_iterations"
        ],
    }


def _expected_block(row: dict[str, str]) -> dict[str, Any]:
    repetition = None
    try:
        repetition = int(row.get("block_repetition", ""))
    except ValueError:
        pass
    return {
        "case_id": str(row.get("case_id", "")),
        "comparison_id": str(row.get("comparison_id", "")),
        "block_repetition": repetition,
        "tier": str(row.get("tier", "")),
        "mesh_name": str(row.get("mesh_name", "")),
        "element_degree": _safe_int(row.get("element_degree")),
        "quadrature_rule": str(row.get("quadrature_rule", "")),
        "rank_count": _safe_int(row.get("total_ranks")),
        "planned_route_order": str(row.get("route_order", "")),
        "status": "missing",
        "reason": "no_job_output",
        "job_path": "",
        "endpoint_gates": None,
        "routes": {
            route: {
                "status": "missing",
                "reason": "no_job_output",
                "timing_exposed": False,
                "admitted_collective_max_wall_time_s": None,
            }
            for route in ROUTES
        },
    }


def _job_directories(campaign_root: Path, case_id: str) -> list[Path]:
    root = campaign_root / "cases" / case_id
    if not root.is_dir():
        return []
    return sorted(path for path in root.glob("job_*" ) if path.is_dir())


def _matrix_rows_equal(planned: dict[str, str], observed: dict[str, Any]) -> bool:
    return all(str(observed.get(key, "")) == str(value) for key, value in planned.items())


def _analyze_job(
    row: dict[str, str],
    job: Path,
    block: dict[str, Any],
    *,
    expected_commit: str,
) -> dict[str, float]:
    block["job_path"] = str(job)
    campaign_root = job.parents[2]
    expected_job_id = job.name.removeprefix("job_")
    block["batch_evidence"] = _cluster_batch_evidence(
        campaign_root=campaign_root,
        case_id=row["case_id"],
        job_id=expected_job_id,
        expected_commit=expected_commit,
    )
    matrix_path = job / "matrix_row.json"
    if not matrix_path.is_file() or not _matrix_rows_equal(row, _read_json(matrix_path)):
        raise AdmissionError("executed matrix row does not exactly match the reviewed matrix")
    run_records = _read_json_list(job / "run_records.json")
    planned_order = _normalize_route_order(row["route_order"])
    route_run_records = [record for record in run_records if record.get("route") in ROUTES]
    if len(route_run_records) != 2 or tuple(
        str(record.get("route")) for record in route_run_records
    ) != planned_order:
        raise AdmissionError("wrapper run records do not preserve the preregistered route order")
    failed_wrappers = [
        record
        for record in route_run_records
        if _integer(record.get("returncode"), "route wrapper returncode") != 0
    ]
    if failed_wrappers:
        failed = failed_wrappers[0]
        returncode = int(failed["returncode"])
        if returncode == 86 and int(failed.get("process_returncode", -1)) == 0:
            validation = dict(failed.get("scientific_validation") or {})
            raise AdmissionError(
                "executor scientific validation rejected route evidence: "
                f"{validation.get('reason', 'unspecified reason')}"
            )
        reason = "runner_timeout" if bool(failed.get("timed_out")) else f"runner_nonzero_exit_{returncode}"
        block["status"] = "censored"
        block["reason"] = reason
        for route in ROUTES:
            block["routes"][route].update({"status": "censored", "reason": reason})
        return {}
    for record in route_run_records:
        validation = dict(record.get("scientific_validation") or {})
        if validation.get("status") != "passed":
            raise AdmissionError(
                f"wrapper scientific validation failed for {record.get('route')}"
            )

    measure = job / "measure_01"
    block_result = _read_json(measure / "block_result.json")
    if int(block_result.get("schema_version", -1)) != 1:
        raise AdmissionError("block_result has the wrong schema version")
    if block_result.get("experiment_id") != EXPERIMENT_ID:
        raise AdmissionError("block_result has the wrong experiment ID")
    if block_result.get("tier") != row["tier"]:
        raise AdmissionError("block_result tier disagrees with the matrix")
    if block_result.get("comparison_id") != row["comparison_id"]:
        raise AdmissionError("block_result comparison_id disagrees with the matrix")
    if int(block_result.get("block_repetition", -1)) != int(row["block_repetition"]):
        raise AdmissionError("block_result repetition disagrees with the matrix")
    planned_order = _normalize_route_order(row["route_order"])
    actual_order = _normalize_route_order(block_result.get("route_order"))
    if actual_order != planned_order:
        raise AdmissionError("executed route order disagrees with the preregistered matrix")
    if block_result.get("timing_reduction") != "mpi_collective_max":
        raise AdmissionError("block timing is not the collective maximum over MPI ranks")
    if block_result.get("route_order_policy") != "seeded_balanced_alternating_v1":
        raise AdmissionError("block_result has the wrong route-order policy")
    if block_result.get("status") != "routes_completed_pending_endpoint_analysis":
        raise AdmissionError("block_result is not awaiting endpoint analysis")
    if block_result.get("timing_claim_released") is not False:
        raise AdmissionError("executor improperly released a timing claim before analysis")
    if str(dict(block_result.get("job_metadata") or {}).get("slurm_job_id", "")) != expected_job_id:
        raise AdmissionError("block_result Slurm job identity disagrees with its job directory")
    route_results = block_result.get("routes")
    if not isinstance(route_results, dict) or set(route_results) != set(ROUTES):
        raise AdmissionError("block_result must contain exactly the two frozen routes")

    timings: dict[str, float] = {}
    records: dict[str, dict[str, Any]] = {}
    for route in ROUTES:
        declared = route_results[route]
        if not isinstance(declared, dict):
            raise AdmissionError(f"block_result route {route!r} is not an object")
        if declared.get("status") != "completed":
            reason = str(declared.get("reason") or f"{route}_not_completed")
            block["status"] = "censored"
            block["reason"] = reason
            block["routes"][route].update({"status": "censored", "reason": reason})
            other = ROUTES[1] if route == ROUTES[0] else ROUTES[0]
            block["routes"][other].update(
                {"status": "timing_withheld", "reason": "paired_route_censored"}
            )
            return {}
        if declared.get("solver_success") is not True:
            raise AdmissionError(f"{route} block result does not record solver success")
        timing = _finite(
            declared.get("collective_max_wall_time_s"),
            f"{route} collective maximum wall time",
        )
        if timing <= 0.0:
            raise AdmissionError(f"{route} collective maximum wall time must be positive")
        if declared.get("timing_provenance") != "solver_allgather_then_MPI_MAX":
            raise AdmissionError(f"{route} timing lacks solver all-gather/MPI-max provenance")
        timing_rank_count = _integer(
            declared.get("timing_rank_count"), f"{route} timing rank count", minimum=1
        )
        raw_rank_times = declared.get("per_rank_wall_times_s")
        if (
            not isinstance(raw_rank_times, list)
            or timing_rank_count != int(row["total_ranks"])
            or len(raw_rank_times) != timing_rank_count
        ):
            raise AdmissionError(f"{route} timing does not contain one value per MPI rank")
        rank_times = np.asarray(raw_rank_times, dtype=np.float64)
        if not np.all(np.isfinite(rank_times)) or np.any(rank_times <= 0.0):
            raise AdmissionError(f"{route} rank timings must be finite and positive")
        if not math.isclose(
            timing, float(np.max(rank_times)), rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise AdmissionError(f"{route} declared timing is not the rank-wise maximum")
        output_path = _path_within(
            measure,
            declared.get("output_json"),
            measure / route / "output.json",
        )
        state_path = _path_within(
            measure,
            declared.get("state_npz"),
            measure / route / "state.npz",
        )
        try:
            record = _validate_route_output(
                output_path,
                state_path,
                row,
                route,
                expected_commit=expected_commit,
                expected_job_id=expected_job_id,
            )
        except (AdmissionError, OSError, ValueError, json.JSONDecodeError) as exc:
            block["routes"][route].update({"status": "invalid", "reason": str(exc)})
            raise AdmissionError(f"{route}: {exc}") from exc
        records[route] = record
        if not np.array_equal(rank_times, record["solver_rank_wall_times_s"]):
            raise AdmissionError(f"{route} block and solver rank timings disagree")
        if not math.isclose(
            timing,
            record["solver_collective_max_wall_time_s"],
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise AdmissionError(f"{route} block and solver collective timings disagree")
        timings[route] = timing
        block["routes"][route].update(
            {
                "status": "endpoint_valid",
                "reason": "route_policy_and_accuracy_evidence_passed",
                "output_path": record["output_path"],
                "output_sha256": record["output_sha256"],
                "state_path": record["state_path"],
                "state_sha256": record["state_sha256"],
                "energy": record["energy"],
                "work": record["work"],
                "u_max": record["u_max"],
                "absolute_dual_residual": record["riesz"]["absolute_dual_residual"],
                "relative_correction": record["riesz"]["relative_correction"],
                "nonlinear_iterations": record["work_counts"]["nonlinear_iterations"],
                "total_krylov_iterations": record["work_counts"][
                    "krylov_iterations_total"
                ],
            }
        )
    endpoint_gates = _endpoint_equivalence(records)
    block["endpoint_gates"] = endpoint_gates
    block["executed_route_order"] = list(actual_order)
    block["status"] = "endpoint_admitted_timing_withheld"
    block["reason"] = "block_passed_waiting_for_complete_campaign"
    for route in ROUTES:
        block["routes"][route].update(
            {
                "status": "endpoint_admitted_timing_withheld",
                "reason": "complete_campaign_gate_pending",
            }
        )
    return timings


def _coverage_reasons(rows: list[dict[str, str]], blocks: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    expected_repetitions = set(range(1, int(FROZEN_GATES["expected_block_repetitions"]) + 1))
    expected_shapes = {
        (tier, scope["mesh_name"], int(scope["element_degree"]), scope["quadrature_rule"], rank)
        for tier, scope in EXPECTED_SCOPES.items()
        for rank in scope["ranks"]
    }
    groups: dict[tuple[str, str, int, str, int], list[dict[str, str]]] = {}
    for row in rows:
        try:
            key = (
                row["tier"],
                row["mesh_name"],
                int(row["element_degree"]),
                row["quadrature_rule"],
                int(row["total_ranks"]),
            )
        except (KeyError, ValueError):
            continue
        groups.setdefault(key, []).append(row)
    if set(groups) != expected_shapes:
        reasons.append("matrix_scope_does_not_match_full_and_low_order_design")
    for key in sorted(expected_shapes):
        group = groups.get(key, [])
        repetitions: set[int] = set()
        first_counts = {route: 0 for route in ROUTES}
        comparison_ids: set[str] = set()
        for row in group:
            try:
                repetition = int(row["block_repetition"])
                order = _normalize_route_order(row["route_order"])
            except (KeyError, AdmissionError, ValueError):
                continue
            repetitions.add(repetition)
            first_counts[order[0]] += 1
            comparison_ids.add(str(row.get("comparison_id", "")))
        if repetitions != expected_repetitions:
            reasons.append(f"incomplete_block_repetitions:{key}")
        if len(comparison_ids) != 1 or "" in comparison_ids:
            reasons.append(f"comparison_id_not_constant_within_group:{key}")
        if first_counts != FROZEN_GATES["expected_first_route_count"]:
            reasons.append(f"route_order_not_balanced:{key}")
    if len(blocks) != len(rows):
        reasons.append("analysis_block_count_disagrees_with_matrix")
    for block in blocks:
        if block["status"] != "endpoint_admitted_timing_withheld":
            reasons.append(
                f"block_not_endpoint_admitted:{block['case_id']}:{block['status']}"
            )
    return reasons


def _bootstrap_median_interval(
    values: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float]:
    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    if sample.size < 2 or not np.all(np.isfinite(sample)):
        raise AdmissionError("bootstrap median interval requires at least two finite values")
    resamples = int(FROZEN_GATES["bootstrap_resamples"])
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    indices = rng.integers(0, sample.size, size=(resamples, sample.size))
    medians = np.median(sample[indices], axis=1)
    alpha = 0.5 * (1.0 - float(FROZEN_GATES["bootstrap_confidence_level"]))
    return (
        float(np.quantile(medians, alpha)),
        float(np.quantile(medians, 1.0 - alpha)),
    )


def _timing_summary(blocks: list[dict[str, Any]], timings: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str, int], list[dict[str, Any]]] = {}
    for block in blocks:
        key = (
            block["tier"],
            block["mesh_name"],
            int(block["element_degree"]),
            block["quadrature_rule"],
            int(block["rank_count"]),
        )
        groups.setdefault(key, []).append(block)
    summary: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        route_values = {
            route: np.asarray(
                [timings[block["case_id"]][route] for block in group], dtype=np.float64
            )
            for route in ROUTES
        }
        ratios = route_values["constitutive_ad"] / route_values["element_ad"]
        group_seed = int(FROZEN_GATES["bootstrap_seed"]) + int(
            hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:8], 16
        )
        route_intervals = {
            route: _bootstrap_median_interval(values, seed=group_seed + index)
            for index, (route, values) in enumerate(route_values.items())
        }
        ratio_interval = _bootstrap_median_interval(ratios, seed=group_seed + 2)
        tie_ratio = float(FROZEN_GATES["practical_ranking_tie_ratio"])
        inverse_tie = 1.0 / tie_ratio
        if ratio_interval[0] > tie_ratio:
            overall_winner = "element_ad"
        elif ratio_interval[1] < inverse_tie:
            overall_winner = "constitutive_ad"
        else:
            overall_winner = None
        order_strata: dict[str, dict[str, Any]] = {}
        stratum_winners: list[str | None] = []
        for order_index, first_route in enumerate(ROUTES):
            stratum = np.asarray(
                [
                    ratios[index]
                    for index, block in enumerate(group)
                    if tuple(block["executed_route_order"])[0] == first_route
                ],
                dtype=np.float64,
            )
            interval: tuple[float, float] | None = None
            winner: str | None = None
            if stratum.size >= int(FROZEN_GATES["minimum_order_stratum_blocks"]):
                interval = _bootstrap_median_interval(
                    stratum, seed=group_seed + 10 + order_index
                )
                if interval[0] > tie_ratio:
                    winner = "element_ad"
                elif interval[1] < inverse_tie:
                    winner = "constitutive_ad"
            stratum_winners.append(winner)
            order_strata[first_route] = {
                "first_route": first_route,
                "blocks": int(stratum.size),
                "median_constitutive_over_element_ratio": (
                    float(np.median(stratum)) if stratum.size else None
                ),
                "bootstrap_median_confidence_interval": (
                    list(interval) if interval is not None else None
                ),
                "winner_beyond_practical_tie_band": winner,
            }
        ranking_admissible = bool(
            overall_winner is not None
            and all(winner == overall_winner for winner in stratum_winners)
        )
        summary.append(
            {
                "tier": key[0],
                "mesh_name": key[1],
                "element_degree": key[2],
                "quadrature_rule": key[3],
                "rank_count": key[4],
                "repetitions": len(group),
                "routes": {
                    route: {
                        "median_collective_max_wall_time_s": float(np.median(values)),
                        "q25_collective_max_wall_time_s": float(np.quantile(values, 0.25)),
                        "q75_collective_max_wall_time_s": float(np.quantile(values, 0.75)),
                        "bootstrap_median_confidence_interval_s": list(
                            route_intervals[route]
                        ),
                    }
                    for route, values in route_values.items()
                },
                "paired_constitutive_over_element_ratio": {
                    "median": float(np.median(ratios)),
                    "q25": float(np.quantile(ratios, 0.25)),
                    "q75": float(np.quantile(ratios, 0.75)),
                    "bootstrap_median_confidence_interval": list(ratio_interval),
                },
                "comparative_ranking": {
                    "practical_tie_ratio": tie_ratio,
                    "overall_winner_beyond_tie_band": overall_winner,
                    "route_order_strata": order_strata,
                    "order_sensitivity_passed": bool(
                        overall_winner is not None
                        and all(winner == overall_winner for winner in stratum_winners)
                    ),
                    "ranking_admissible": ranking_admissible,
                    "reason": (
                        "paired_interval_and_both_order_strata_clear_practical_tie_band"
                        if ranking_admissible
                        else "paired_or_order_stratified_interval_does_not_clear_practical_tie_band"
                    ),
                },
                "uncertainty_method": {
                    "name": "paired_nonparametric_bootstrap_of_independent_cold_process_blocks",
                    "confidence_level": FROZEN_GATES["bootstrap_confidence_level"],
                    "resamples": FROZEN_GATES["bootstrap_resamples"],
                    "base_seed": FROZEN_GATES["bootstrap_seed"],
                },
            }
        )
    return summary


def analyze(matrix_path: Path, campaign_root: Path, manifest_path: Path) -> dict[str, Any]:
    matrix_path = matrix_path.resolve()
    campaign_root = campaign_root.resolve()
    rows, matrix_violations = _load_matrix(matrix_path)
    manifest = _validate_manifest(manifest_path.resolve(), matrix_path)
    blocks: list[dict[str, Any]] = []
    timing_values: dict[str, dict[str, float]] = {}
    violations_by_case = {row["case_id"]: row["reason"] for row in matrix_violations}
    for row in rows:
        block = _expected_block(row)
        case_id = block["case_id"]
        if case_id in violations_by_case:
            block.update({"status": "invalid", "reason": violations_by_case[case_id]})
            for route in ROUTES:
                block["routes"][route].update(
                    {"status": "invalid", "reason": "matrix_policy_invalid"}
                )
            blocks.append(block)
            continue
        jobs = _job_directories(campaign_root, case_id)
        if len(jobs) > 1:
            block.update({"status": "invalid", "reason": "duplicate_job_directories"})
            for route in ROUTES:
                block["routes"][route].update(
                    {"status": "invalid", "reason": "duplicate_job_directories"}
                )
        elif len(jobs) == 1:
            try:
                timing_values[case_id] = _analyze_job(
                    row,
                    jobs[0],
                    block,
                    expected_commit=str(manifest.get("source_commit", "")),
                )
            except (AdmissionError, OSError, ValueError, json.JSONDecodeError) as exc:
                if block["status"] not in {"censored"}:
                    block.update({"status": "invalid", "reason": str(exc)})
                    for route in ROUTES:
                        if block["routes"][route]["status"] not in {"invalid"}:
                            block["routes"][route].update(
                                {"status": "invalid", "reason": "paired_block_invalid"}
                            )
                timing_values.pop(case_id, None)
        blocks.append(block)

    coverage_reasons = _coverage_reasons(rows, blocks)
    campaign_reasons = [row["reason"] for row in matrix_violations]
    campaign_reasons.extend(coverage_reasons)
    if manifest.get("eligible") is not True:
        campaign_reasons.append(str(manifest.get("reason")))
    campaign_reasons = list(dict.fromkeys(campaign_reasons))
    endpoint_correct_timing_admissible = not campaign_reasons
    timing_summary: list[dict[str, Any]] = []
    if endpoint_correct_timing_admissible:
        for block in blocks:
            block["status"] = "timing_admitted"
            block["reason"] = "all_tier_b_and_low_order_gates_passed"
            for route in ROUTES:
                value = timing_values[block["case_id"]][route]
                block["routes"][route].update(
                    {
                        "status": "timing_admitted",
                        "reason": "all_tier_b_and_low_order_gates_passed",
                        "timing_exposed": True,
                        "admitted_collective_max_wall_time_s": value,
                    }
                )
        timing_summary = _timing_summary(blocks, timing_values)
    else:
        for block in blocks:
            if block["status"] == "endpoint_admitted_timing_withheld":
                block["reason"] = "campaign_or_low_order_gate_failed"
                for route in ROUTES:
                    block["routes"][route]["reason"] = "campaign_or_low_order_gate_failed"

    structural_censors = [
        {
            "tier": "full_solve_confirmation",
            "mesh_name": EXPECTED_SCOPES["full_solve_confirmation"]["mesh_name"],
            "element_degree": 4,
            "quadrature_rule": "tetra_24point",
            "rank_count": ranks,
            "route": "colored_sfd",
            "status": "censored",
            "reason": "prespecified_not_attempted_memory_risk_no_threshold_claim",
            "timing_exposed": False,
            "admitted_collective_max_wall_time_s": None,
        }
        for ranks in EXPECTED_SCOPES["full_solve_confirmation"]["ranks"]
    ]
    status_counts: dict[str, int] = {}
    for block in blocks:
        status_counts[block["status"]] = status_counts.get(block["status"], 0) + 1
    required_rows = len(rows)
    admitted_rows = sum(block["status"] == "timing_admitted" for block in blocks)
    analysis_git = _git_metadata()
    publication_admissible = bool(
        endpoint_correct_timing_admissible
        and required_rows == 30
        and admitted_rows == required_rows
        and manifest.get("eligible") is True
        and analysis_git.get("dirty") is False
        and analysis_git.get("commit") == manifest.get("source_commit")
    )
    return {
        "schema": {"id": SCHEMA_ID, "version": SCHEMA_VERSION},
        "experiment_id": EXPERIMENT_ID,
        "analysis_tier": "full_solve_cross_route_endpoint_admission",
        "matrix_path": str(matrix_path),
        "matrix_sha256": _sha256_file(matrix_path),
        "analysis_contract_path": str(ROUTE_ANALYSIS_CONTRACT),
        "analysis_contract_sha256": _sha256_file(ROUTE_ANALYSIS_CONTRACT),
        "analysis_script_path": str(Path(__file__).resolve()),
        "analysis_script_sha256": _sha256_file(Path(__file__).resolve()),
        "campaign_root": str(campaign_root),
        "manifest": manifest,
        "frozen_gates": FROZEN_GATES,
        "expected_scopes": EXPECTED_SCOPES,
        "matrix_policy_violations": matrix_violations,
        "coverage_and_campaign_failure_reasons": campaign_reasons,
        "required_rows": required_rows,
        "admitted_rows": admitted_rows,
        "publication_admissible": publication_admissible,
        "provenance": {"git": analysis_git},
        "representative_low_order_confirmation_required": True,
        "representative_low_order_confirmation_passed": bool(
            endpoint_correct_timing_admissible
            and all(
                block["status"] == "timing_admitted"
                for block in blocks
                if block["tier"] == "low_order_confirmation"
            )
        ),
        "endpoint_correct_timing_admissible": endpoint_correct_timing_admissible,
        "descriptive_timing_available": endpoint_correct_timing_admissible,
        "comparative_ranking_admissible": bool(
            endpoint_correct_timing_admissible
            and timing_summary
            and all(
                item["comparative_ranking"]["ranking_admissible"]
                for item in timing_summary
            )
        ),
        "timing_admissible": endpoint_correct_timing_admissible,
        "terminal_decision": (
            (
                "tier_b_comparative_ranking_admissible"
                if timing_summary
                and all(
                    item["comparative_ranking"]["ranking_admissible"]
                    for item in timing_summary
                )
                else "tier_b_descriptive_timing_only"
            )
            if endpoint_correct_timing_admissible
            else "timing_withheld"
        ),
        "status_counts": status_counts,
        "blocks": blocks,
        "structural_censors": structural_censors,
        "timing_summary": timing_summary,
    }


def _write_csv(path: Path, analysis: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "case_id",
        "comparison_id",
        "block_repetition",
        "tier",
        "mesh_name",
        "element_degree",
        "quadrature_rule",
        "rank_count",
        "route",
        "block_status",
        "route_status",
        "reason",
        "state_sha256",
        "energy",
        "work",
        "u_max",
        "absolute_dual_residual",
        "relative_correction",
        "nonlinear_iterations",
        "total_krylov_iterations",
        "timing_exposed",
        "admitted_collective_max_wall_time_s",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for block in analysis["blocks"]:
            for route in ROUTES:
                route_row = block["routes"][route]
                writer.writerow(
                    {
                        "case_id": block["case_id"],
                        "comparison_id": block["comparison_id"],
                        "block_repetition": block["block_repetition"],
                        "tier": block["tier"],
                        "mesh_name": block["mesh_name"],
                        "element_degree": block["element_degree"],
                        "quadrature_rule": block["quadrature_rule"],
                        "rank_count": block["rank_count"],
                        "route": route,
                        "block_status": block["status"],
                        "route_status": route_row["status"],
                        "reason": route_row["reason"],
                        "state_sha256": route_row.get("state_sha256", ""),
                        "energy": route_row.get("energy", ""),
                        "work": route_row.get("work", ""),
                        "u_max": route_row.get("u_max", ""),
                        "absolute_dual_residual": route_row.get(
                            "absolute_dual_residual", ""
                        ),
                        "relative_correction": route_row.get("relative_correction", ""),
                        "nonlinear_iterations": route_row.get("nonlinear_iterations", ""),
                        "total_krylov_iterations": route_row.get(
                            "total_krylov_iterations", ""
                        ),
                        "timing_exposed": route_row["timing_exposed"],
                        "admitted_collective_max_wall_time_s": route_row[
                            "admitted_collective_max_wall_time_s"
                        ],
                    }
                )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Defaults to CAMPAIGN_ROOT/prepared_manifest.json.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--require-timing-admissible",
        action="store_true",
        help="Exit 2 after writing the complete map if any gate withholds timing.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    campaign_root = Path(args.campaign_root).resolve()
    manifest = (
        Path(args.manifest).resolve()
        if args.manifest is not None
        else campaign_root / "prepared_manifest.json"
    )
    result = analyze(Path(args.matrix), campaign_root, manifest)
    atomic_write_json(Path(args.output_json), result, nonfinite_as_null=False)
    if args.output_csv is not None:
        _write_csv(Path(args.output_csv), result)
    print(
        json.dumps(
            {
                "terminal_decision": result["terminal_decision"],
                "timing_admissible": result["timing_admissible"],
                "status_counts": result["status_counts"],
                "output_json": str(Path(args.output_json).resolve()),
            },
            indent=2,
            allow_nan=False,
        )
    )
    if args.require_timing_admissible and not result["timing_admissible"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

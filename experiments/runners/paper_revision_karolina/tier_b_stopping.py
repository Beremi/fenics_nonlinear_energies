"""Frozen EXP-ROUTE-001 Tier-B stopping policy and STOP gate validation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = REPO_ROOT / "paper/protocols/EXP-ROUTE-001-tier-b-stopping-policy.json"
TIER_B_TIERS = frozenset({"full_solve_confirmation", "low_order_confirmation"})
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


class TierBStoppingError(ValueError):
    """The Tier-B policy or its prerequisite STOP evidence is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TierBStoppingError(f"{path} must contain a JSON object")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise TierBStoppingError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TierBStoppingError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise TierBStoppingError(f"{label} must be finite")
    return result


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise TierBStoppingError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TierBStoppingError(f"{label} must be an integer") from exc
    if isinstance(value, float) and value != float(result):
        raise TierBStoppingError(f"{label} must be an integer")
    if isinstance(value, str) and value.strip() != str(result):
        raise TierBStoppingError(f"{label} must be an integer")
    return result


def _require_exact_number(value: object, expected: float, *, label: str) -> None:
    if _finite(value, label=label) != float(expected):
        raise TierBStoppingError(f"{label} changed from the frozen value {expected!r}")


def load_policy(path: Path = POLICY_PATH) -> dict[str, object]:
    policy = _read_object(Path(path).resolve())
    if set(policy) != {
        "schema_id",
        "schema_version",
        "status",
        "local_calibration",
        "nonlinear_solver",
        "riesz_solver",
        "submission_gate",
        "claim_boundary",
    }:
        raise TierBStoppingError("Tier-B stopping policy top-level shape is invalid")
    if (
        policy.get("schema_id")
        != "fenics-nonlinear-energies.exp-route-001-tier-b-stopping-policy"
        or policy.get("schema_version") != 1
        or policy.get("status") != "frozen_conditional_on_cluster_stop_adjudication"
    ):
        raise TierBStoppingError("Tier-B stopping policy identity is invalid")
    local = policy.get("local_calibration")
    nonlinear = policy.get("nonlinear_solver")
    riesz = policy.get("riesz_solver")
    gate = policy.get("submission_gate")
    if not all(isinstance(value, dict) for value in (local, nonlinear, riesz, gate)):
        raise TierBStoppingError("Tier-B stopping policy sections are incomplete")
    assert isinstance(local, dict)
    assert isinstance(nonlinear, dict)
    assert isinstance(riesz, dict)
    assert isinstance(gate, dict)
    if set(local) != {
        "analysis_path",
        "analysis_sha256",
        "plan_path",
        "plan_sha256",
        "source_commit",
        "terminal_decision",
    }:
        raise TierBStoppingError("Tier-B local-calibration binding is malformed")
    for key in ("analysis_sha256", "plan_sha256"):
        if HEX64.fullmatch(str(local.get(key, ""))) is None:
            raise TierBStoppingError(f"local calibration {key} is invalid")
    if HEX40.fullmatch(str(local.get("source_commit", ""))) is None:
        raise TierBStoppingError("local calibration source commit is invalid")
    if (
        local.get("analysis_path")
        != "artifacts/reproduction/exp_stop_001_local_5b2f3b5/analysis.json"
        or local.get("plan_path")
        != "artifacts/reproduction/exp_stop_001_local_5b2f3b5/plan.json"
        or local.get("terminal_decision")
        != "local_calibration_complete_cluster_computations_deferred"
    ):
        raise TierBStoppingError("Tier-B local-calibration identity is invalid")
    if set(nonlinear) != {
        "convergence_metric",
        "convergence_mode",
        "nonlinear_max_it",
        "ksp_rtol",
        "ksp_max_it",
        "absolute_residual_tolerance",
        "relative_correction_is_termination_gate",
        "relative_correction_diagnostic_limit",
        "targets_by_degree",
    }:
        raise TierBStoppingError("Tier-B nonlinear policy shape is invalid")
    if (
        nonlinear.get("convergence_metric") != "reference_elastic_energy"
        or nonlinear.get("convergence_mode") != "gradient_only"
        or nonlinear.get("relative_correction_is_termination_gate") is not False
    ):
        raise TierBStoppingError("Tier-B nonlinear stopping semantics are invalid")
    targets = nonlinear.get("targets_by_degree")
    if not isinstance(targets, dict) or set(targets) != {"1", "4"}:
        raise TierBStoppingError("Tier-B degree-target grid is invalid")
    _require_exact_number(nonlinear.get("ksp_rtol"), 1.0e-8, label="Tier-B KSP rtol")
    if _integer(nonlinear.get("nonlinear_max_it"), label="Tier-B nonlinear cap") != 80:
        raise TierBStoppingError("Tier-B nonlinear cap changed from the frozen value")
    if _integer(nonlinear.get("ksp_max_it"), label="Tier-B KSP cap") != 1000:
        raise TierBStoppingError("Tier-B KSP cap changed from the frozen value")
    _require_exact_number(
        nonlinear.get("absolute_residual_tolerance"),
        0.0,
        label="Tier-B absolute residual tolerance",
    )
    _require_exact_number(
        nonlinear.get("relative_correction_diagnostic_limit"),
        2.0e-3,
        label="Tier-B correction diagnostic limit",
    )
    target_contracts = {
        "1": {
            "relative_dual_residual_target": 1.0e-6,
            "status": "selected_from_admitted_local_calibration",
            "reference_row_id": "p3d_p1_nonlinear_1em07",
            "selected_row_id": "p3d_p1_nonlinear_1em06",
        },
        "4": {
            "relative_dual_residual_target": 1.0e-7,
            "status": "tight_reference_requires_cluster_adjudication",
            "reference_row_id": "p3d_p4_nonlinear_1em07_cluster",
            "selected_row_id": None,
        },
    }
    for degree, expected in target_contracts.items():
        target = targets.get(degree)
        if not isinstance(target, dict) or set(target) != set(expected):
            raise TierBStoppingError(f"Tier-B P{degree} target shape is invalid")
        _require_exact_number(
            target.get("relative_dual_residual_target"),
            float(expected["relative_dual_residual_target"]),
            label=f"Tier-B P{degree} relative target",
        )
        for key in ("status", "reference_row_id", "selected_row_id"):
            if target.get(key) != expected[key]:
                raise TierBStoppingError(f"Tier-B P{degree} {key} is invalid")
    if set(riesz) != {
        "ksp_type",
        "pc_type",
        "norm_type",
        "rtol",
        "atol",
        "max_it",
        "true_residual_rtol",
        "spd_factor_solver_type",
        "symmetry_relative_tolerance",
        "set_from_petsc_options",
    }:
        raise TierBStoppingError("Tier-B Riesz policy shape is invalid")
    if (
        riesz.get("ksp_type") != "cg"
        or riesz.get("pc_type") != "jacobi"
        or riesz.get("norm_type") != "unpreconditioned"
        or riesz.get("set_from_petsc_options") is not False
    ):
        raise TierBStoppingError("Tier-B Riesz solver semantics are invalid")
    for key, expected in {
        "rtol": 1.0e-10,
        "atol": 0.0,
        "true_residual_rtol": 1.0e-8,
        "symmetry_relative_tolerance": 1.0e-12,
    }.items():
        _require_exact_number(riesz.get(key), expected, label=f"Tier-B Riesz {key}")
    if _integer(riesz.get("max_it"), label="Tier-B Riesz cap") != 5000:
        raise TierBStoppingError("Tier-B Riesz cap changed from the frozen value")
    if riesz.get("spd_factor_solver_type") != "mumps":
        raise TierBStoppingError("Tier-B Riesz SPD factor route is invalid")
    if set(gate) != {
        "required_schema_id",
        "required_schema_version",
        "required_calibration_scope_passed",
        "required_mpi_comparison_ids",
        "required_p4_reference_comparison_status",
        "required_p4_reference_row_id",
        "required_p4_selected_policy_key",
        "required_selected_policy_status",
        "submission_without_gate_admissible",
    }:
        raise TierBStoppingError("Tier-B STOP gate shape is invalid")
    if (
        gate.get("required_schema_id")
        != "fenics-nonlinear-energies.exp-stop-001.final-adjudication"
        or gate.get("required_schema_version") != 3
        or gate.get("required_calibration_scope_passed") is not True
        or gate.get("submission_without_gate_admissible") is not False
    ):
        raise TierBStoppingError("Tier-B STOP submission gate is invalid")
    if (
        gate.get("required_p4_reference_comparison_status") != "accepted"
        or gate.get("required_p4_reference_row_id")
        != "p3d_p4_nonlinear_1em07_cluster"
        or gate.get("required_mpi_comparison_ids")
        != [
            "ginzburg_landau_mpi_consistency_cluster",
            "hyperelasticity_mpi_consistency_cluster",
            "plasticity3d_mpi_consistency_cluster",
        ]
        or gate.get("required_p4_selected_policy_key")
        != "p3d_p4_nonlinear_cluster"
        or gate.get("required_selected_policy_status")
        != "selected_loosest_accepted_same_discretization_policy"
    ):
        raise TierBStoppingError("Tier-B P4 STOP gate identity is invalid")
    return policy


def is_tier_b_row(row: Mapping[str, str]) -> bool:
    return (
        row.get("experiment_id") == "EXP-ROUTE-001"
        and row.get("tier") in TIER_B_TIERS
        and row.get("runner") == "p3d_solve_block"
    )


def row_contract(row: Mapping[str, str]) -> dict[str, object]:
    if not is_tier_b_row(row):
        raise TierBStoppingError("row is not an EXP-ROUTE-001 Tier-B solve block")
    frozen = load_policy()
    nonlinear = dict(frozen["nonlinear_solver"])
    riesz = dict(frozen["riesz_solver"])
    degree = str(_integer(row.get("element_degree"), label="matrix element degree"))
    try:
        target_row = dict(dict(nonlinear["targets_by_degree"])[degree])
    except (KeyError, TypeError, ValueError) as exc:
        raise TierBStoppingError(f"P{degree} is outside the Tier-B degree policy") from exc
    target = _finite(
        target_row.get("relative_dual_residual_target"),
        label=f"P{degree} relative dual-residual target",
    )
    expected = {
        "ksp_rtol": _finite(nonlinear.get("ksp_rtol"), label="Tier-B KSP rtol"),
        "ksp_max_it": _integer(nonlinear["ksp_max_it"], label="Tier-B KSP cap"),
        "maxit": _integer(
            nonlinear["nonlinear_max_it"], label="Tier-B nonlinear cap"
        ),
        "stop_tol": target,
        "grad_stop_tol": _finite(
            nonlinear.get("absolute_residual_tolerance"),
            label="Tier-B absolute residual tolerance",
        ),
    }
    observed = {
        "ksp_rtol": _finite(row.get("ksp_rtol"), label="matrix KSP rtol"),
        "ksp_max_it": _integer(row.get("ksp_max_it"), label="matrix KSP cap"),
        "maxit": _integer(row.get("maxit"), label="matrix nonlinear cap"),
        "stop_tol": _finite(row.get("stop_tol"), label="matrix relative target"),
        "grad_stop_tol": _finite(
            row.get("grad_stop_tol"), label="matrix absolute target"
        ),
    }
    if observed != expected:
        mismatches = sorted(
            key for key, expected_value in expected.items() if observed[key] != expected_value
        )
        raise TierBStoppingError(
            f"{row.get('case_id')}: matrix stopping values differ from the Tier-B "
            f"policy ({', '.join(mismatches)})"
        )
    if row.get("convergence_metric") != nonlinear["convergence_metric"]:
        raise TierBStoppingError(
            f"{row.get('case_id')}: matrix convergence metric differs from the Tier-B policy"
        )
    return {
        "degree": int(degree),
        "relative_dual_residual_target": target,
        "target_status": target_row.get("status"),
        "reference_row_id": target_row.get("reference_row_id"),
        "nonlinear_solver": nonlinear,
        "riesz_solver": riesz,
        "policy_path": str(POLICY_PATH.relative_to(REPO_ROOT)),
        "policy_sha256": sha256_file(POLICY_PATH),
    }


def validate_stop_adjudication(path: Path) -> dict[str, object]:
    frozen = load_policy()
    gate = dict(frozen["submission_gate"])
    local = dict(frozen["local_calibration"])
    source = Path(path).resolve()
    payload = _read_object(source)
    if (
        payload.get("schema_id") != gate["required_schema_id"]
        or payload.get("schema_version") != gate["required_schema_version"]
        or payload.get("experiment_id") != "EXP-STOP-001"
        or payload.get("calibration_scope_passed")
        is not gate["required_calibration_scope_passed"]
    ):
        raise TierBStoppingError("STOP adjudication has not passed the calibration scope")
    if (
        payload.get("terminal_decision")
        not in {
            "CALIBRATION_PASS_PENDING_DISCRETIZATION_GATE",
            "CALIBRATION_SCOPED_PASS_PENDING_DISCRETIZATION_GATE",
        }
        or payload.get("complete_exp_stop_pass") is not False
        or payload.get("publication_timing_admissible") is not False
        or payload.get("cluster_case_count") != 7
        or payload.get("required_gate_failures") != []
    ):
        raise TierBStoppingError("STOP adjudication terminal decision is inconsistent")
    if (
        payload.get("computation_source_commit") != local["source_commit"]
        or payload.get("local_analysis_sha256") != local["analysis_sha256"]
    ):
        raise TierBStoppingError("STOP adjudication is not bound to the frozen local calibration")
    reference_id = str(gate["required_p4_reference_row_id"])
    comparisons = payload.get("comparisons")
    expected_comparison_ids = {
        "p3d_p4_nonlinear_1em02_cluster",
        "p3d_p4_nonlinear_1em04_cluster",
        "p3d_p4_nonlinear_1em06_cluster",
        "p3d_p4_nonlinear_1em07_cluster",
        *gate["required_mpi_comparison_ids"],
    }
    if not isinstance(comparisons, dict) or set(comparisons) != expected_comparison_ids:
        raise TierBStoppingError("STOP adjudication comparison grid is incomplete")
    comparison_raw = comparisons.get(reference_id)
    if not isinstance(comparison_raw, dict):
        raise TierBStoppingError("STOP adjudication lacks the P4 tight reference")
    comparison = dict(comparison_raw)
    if (
        comparison.get("status") != gate["required_p4_reference_comparison_status"]
        or comparison.get("reference_row_id") != reference_id
        or dict(comparison.get("gates") or {}).get("passed") is not True
    ):
        raise TierBStoppingError("STOP adjudication did not admit the P4 tight reference")
    for case_id in gate["required_mpi_comparison_ids"]:
        mpi_comparison = comparisons.get(str(case_id))
        if not isinstance(mpi_comparison, dict) or mpi_comparison.get("status") != "accepted":
            raise TierBStoppingError("STOP adjudication did not pass every MPI consistency gate")
    selected_policies = payload.get("selected_policies")
    if not isinstance(selected_policies, dict):
        raise TierBStoppingError("STOP adjudication lacks selected policies")
    p4_selected = selected_policies.get(str(gate["required_p4_selected_policy_key"]))
    selected_row_id = (
        str(p4_selected.get("row_id", "")) if isinstance(p4_selected, dict) else ""
    )
    selected_comparison = comparisons.get(selected_row_id)
    if (
        not isinstance(p4_selected, dict)
        or p4_selected.get("status") != gate["required_selected_policy_status"]
        or not isinstance(selected_comparison, dict)
        or selected_comparison.get("status") != "accepted"
    ):
        raise TierBStoppingError("STOP adjudication lacks an accepted P4 selected policy")
    rejected = payload.get("rejected_or_censored_cases")
    expected_rejected = sorted(
        case_id
        for case_id, value in comparisons.items()
        if not isinstance(value, dict) or value.get("status") != "accepted"
    )
    if rejected != expected_rejected:
        raise TierBStoppingError("STOP adjudication rejection inventory is inconsistent")
    adjudicator = payload.get("adjudicator")
    adjudicator_path = REPO_ROOT / "experiments/runners/prepare_exp_stop_001_karolina.py"
    if (
        not isinstance(adjudicator, dict)
        or HEX40.fullmatch(str(adjudicator.get("source_commit", ""))) is None
        or adjudicator.get("source_dirty") is not False
        or adjudicator.get("path")
        != "experiments/runners/prepare_exp_stop_001_karolina.py"
        or HEX64.fullmatch(str(adjudicator.get("sha256", ""))) is None
        or adjudicator.get("sha256") != sha256_file(adjudicator_path)
    ):
        raise TierBStoppingError("STOP adjudication code provenance is invalid")
    checksum = str(payload.get("cluster_archive_checksum_sha256", ""))
    if HEX64.fullmatch(checksum) is None:
        raise TierBStoppingError("STOP adjudication lacks a sealed cluster checksum")
    return {
        "schema_id": str(payload["schema_id"]),
        "schema_version": int(payload["schema_version"]),
        "path": str(source),
        "sha256": sha256_file(source),
        "computation_source_commit": str(payload["computation_source_commit"]),
        "adjudicator_source_commit": str(adjudicator["source_commit"]),
        "adjudicator_sha256": str(adjudicator["sha256"]),
        "local_analysis_sha256": str(payload["local_analysis_sha256"]),
        "cluster_archive_checksum_sha256": checksum,
        "p4_reference_row_id": reference_id,
        "p4_reference_status": str(comparison["status"]),
    }

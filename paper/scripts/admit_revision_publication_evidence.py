#!/usr/bin/env python3
"""Audit and admit the 14 source files used by the revision evidence tables.

The default ``audit`` mode is read-only and diagnostic.  ``admit`` writes a
versioned publication source manifest only when every configured input passes
the same fail-closed checks that the table generator and submission checker
subsequently repeat.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.benchmark.run_record import (  # noqa: E402
    RunRecordValidationError,
    atomic_write_json,
    validate_run_record,
)
from experiments.runners.paper_revision_karolina.tier_b_stopping import (  # noqa: E402
    POLICY_PATH as TIER_B_STOPPING_POLICY,
    sha256_file as stopping_sha256_file,
    validate_stop_adjudication,
)


SCHEMA_ID = "fenics-nonlinear-energies.revision-publication-evidence-source"
SCHEMA_VERSION = 2
AUDIT_SCHEMA_ID = "fenics-nonlinear-energies.revision-publication-evidence-audit"
AUDIT_SCHEMA_VERSION = 1
ADMITTED_STATUS = "admitted_clean_publication_evidence"
DEFAULT_EVIDENCE_ROOT = (
    REPO_ROOT / "artifacts/reproduction/paper_revision_2026_07_10/pilots"
)
DEFAULT_MANIFEST_NAME = "publication_evidence_manifest.json"
TABLE_GENERATOR = Path("paper/scripts/generate_revision_evidence_tables.py")
HEX40_RE = re.compile(r"[0-9a-f]{40}")
HEX64_RE = re.compile(r"[0-9a-f]{64}")
ROUTE_ENDPOINT_SCHEMA_ID = "fenics-nonlinear-energies.exp-route-001.tier-b-endpoints"
ROUTE_ENDPOINT_PUBLICATION_PATH = (
    "EXP-ROUTE-001/analysis_contract_v1/tier_b_endpoint_analysis.json"
)
ROUTE_STOPPING_PUBLICATION_PATH = (
    "EXP-ROUTE-001/analysis_contract_v1/stopping_adjudication.json"
)
ROUTE_ENDPOINT_SUMMARY_KEYS = frozenset(
    {
        "path",
        "sha256",
        "schema_id",
        "schema_version",
        "terminal_decision",
        "comparative_ranking_admissible",
        "publication_admissible",
        "required_rows",
        "admitted_rows",
        "stopping_policy",
        "stopping_adjudication",
        "stopping_binding_matches_manifest",
    }
)
ROUTE_STOPPING_BINDING_KEYS = frozenset(
    {
        "schema_id",
        "schema_version",
        "path",
        "sha256",
        "computation_source_commit",
        "adjudicator_source_commit",
        "adjudicator_sha256",
        "local_analysis_sha256",
        "cluster_archive_checksum_sha256",
        "p4_reference_row_id",
        "p4_reference_status",
    }
)
QUADRATURE_RULE_IDS = (
    "tetra_1point",
    "tetra_11point",
    "tetra_24point",
    "tetra_duffy_125point",
)
QUADRATURE_ARTIFACT_FIELDS = {
    "hessian_action_artifact": (
        "hessian_action",
        "hessian_action_content_sha256",
        "float64",
    ),
    "residual_artifact": ("residual", "residual_content_sha256", "float64"),
    "branch_map_artifact": ("branch_map", "branch_map_content_sha256", "int8"),
}


@dataclass(frozen=True, slots=True)
class EvidenceSpec:
    key: str
    relative_path: Path
    producer_path: Path
    family: str
    terminal_statuses: tuple[str, ...]
    companion_manifest: Path
    run_records: tuple[Path, ...] = ()


EVIDENCE_SPECS: tuple[EvidenceSpec, ...] = (
    EvidenceSpec(
        "plaplace",
        Path("EXP-VAL-001/plaplace_manufactured.json"),
        Path("experiments/runners/run_manufactured_plaplace_verification.py"),
        "manufactured_scalar",
        ("passed",),
        Path("EXP-VAL-001/pilot_manifest.json"),
    ),
    EvidenceSpec(
        "ginzburg_landau",
        Path("EXP-VAL-001/ginzburg_landau_manufactured.json"),
        Path("experiments/runners/run_manufactured_ginzburg_landau_verification.py"),
        "manufactured_scalar",
        ("passed",),
        Path("EXP-VAL-001/pilot_manifest.json"),
    ),
    EvidenceSpec(
        "hyperelastic_patch",
        Path("EXP-VAL-001/hyperelastic_affine_patch.json"),
        Path("experiments/runners/run_hyperelastic_affine_patch_verification.py"),
        "affine_patch",
        ("passed",),
        Path("EXP-VAL-001/pilot_manifest.json"),
    ),
    EvidenceSpec(
        "hyperelastic_nonaffine",
        Path("EXP-VAL-001/hyperelastic_nonaffine_quadrature_refinement_v2/result.json"),
        Path("experiments/runners/run_manufactured_hyperelastic_verification.py"),
        "hyperelastic_nonaffine",
        ("passed",),
        Path("EXP-VAL-001/pilot_manifest.json"),
    ),
    EvidenceSpec(
        "smooth_derivatives",
        Path("EXP-DERIV-001/smooth_fixed_element_v1.json"),
        Path("experiments/runners/run_smooth_element_derivative_verification.py"),
        "derivative",
        ("passed",),
        Path("EXP-DERIV-001/pilot_manifest.json"),
    ),
    EvidenceSpec(
        "p1_derivatives",
        Path("EXP-DERIV-001/p1_l1_fixed_element_v2.json"),
        Path("experiments/runners/run_paper_derivative_verification.py"),
        "derivative",
        ("passed",),
        Path("EXP-DERIV-001/pilot_manifest.json"),
    ),
    EvidenceSpec(
        "p2_derivatives",
        Path("EXP-DERIV-001/p2_l1_fixed_element_v2.json"),
        Path("experiments/runners/run_paper_derivative_verification.py"),
        "derivative",
        ("passed",),
        Path("EXP-DERIV-001/pilot_manifest.json"),
    ),
    EvidenceSpec(
        "p4_derivatives",
        Path("EXP-DERIV-001/p4_l1_fixed_element_v2.json"),
        Path("experiments/runners/run_paper_derivative_verification.py"),
        "derivative",
        ("passed",),
        Path("EXP-DERIV-001/pilot_manifest.json"),
    ),
    EvidenceSpec(
        "material_point",
        Path("EXP-MC-001/material_point_verification.json"),
        Path("experiments/runners/run_plasticity3d_material_point_verification.py"),
        "material_point",
        ("passed",),
        Path("EXP-MC-001/pilot_manifest.json"),
        (Path("EXP-MC-001/run_record.json"),),
    ),
    EvidenceSpec(
        "distribution",
        Path("EXP-DIST-001/distribution_equivalence.json"),
        Path("experiments/runners/run_hyperelasticity_distribution_equivalence.py"),
        "distribution",
        ("passed",),
        Path("EXP-DIST-001/pilot_manifest.json"),
        (
            Path("EXP-DIST-001/run_record_np1.json"),
            Path("EXP-DIST-001/run_record_np2.json"),
            Path("EXP-DIST-001/run_record_np4.json"),
        ),
    ),
    EvidenceSpec(
        "p1_quadrature",
        Path("EXP-DISC-001/p1_l1_fixed_state_quadrature_v2.json"),
        Path("experiments/runners/run_plasticity3d_fixed_state_quadrature.py"),
        "quadrature",
        ("completed",),
        Path("EXP-DISC-001/pilot_manifest.json"),
    ),
    EvidenceSpec(
        "p2_quadrature",
        Path("EXP-DISC-001/p2_l1_fixed_state_quadrature_v2.json"),
        Path("experiments/runners/run_plasticity3d_fixed_state_quadrature.py"),
        "quadrature",
        ("completed",),
        Path("EXP-DISC-001/pilot_manifest.json"),
    ),
    EvidenceSpec(
        "p4_quadrature",
        Path("EXP-DISC-001/p4_l1_fixed_state_quadrature_v2.json"),
        Path("experiments/runners/run_plasticity3d_fixed_state_quadrature.py"),
        "quadrature",
        ("completed",),
        Path("EXP-DISC-001/pilot_manifest.json"),
    ),
    EvidenceSpec(
        "route_analysis",
        Path("EXP-ROUTE-001/analysis_contract_v1/analysis.json"),
        Path("experiments/analysis/analyze_plasticity3d_route_cost_model.py"),
        "route_analysis",
        ("predictive_selector_admissible", "finite_empirical_map_only"),
        Path("EXP-ROUTE-001/analysis_contract_v1/manifest.json"),
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def _display_path(path: Path, *, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "worktree_clean": not bool(status.strip())}


def _git_is_ancestor(repo_root: Path, older: str, newer: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", older, newer],
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0


def _is_contained(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _check(check_id: str, passed: bool, detail: str, *, applicable: bool = True) -> dict[str, Any]:
    return {
        "id": check_id,
        "applicable": bool(applicable),
        "passed": bool(passed),
        "detail": detail,
    }


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return all(_finite_tree(item) for item in value)
    return False


def _boolean_leaves(value: Any) -> list[bool]:
    leaves: list[bool] = []
    if isinstance(value, bool):
        leaves.append(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            leaves.extend(_boolean_leaves(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            leaves.extend(_boolean_leaves(item))
    return leaves


def _number(
    value: Any,
    label: str,
    errors: list[str],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label} must be a finite number")
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        errors.append(f"{label} must be finite")
        return None
    if integer and not isinstance(value, int):
        errors.append(f"{label} must be an integer")
    if minimum is not None and numeric < minimum:
        errors.append(f"{label} must be at least {minimum}")
    if maximum is not None and numeric > maximum:
        errors.append(f"{label} must be at most {maximum}")
    return numeric


def _exact_keys(
    value: Any, expected: set[str], label: str, errors: list[str]
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object with keys {sorted(expected)}")
        return None
    actual = {str(key) for key in value}
    if actual != expected:
        errors.append(f"{label} keys {sorted(actual)} != {sorted(expected)}")
    return value


def _source_schema_errors(spec: EvidenceSpec, payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    publication = payload.get("publication_provenance")
    if not isinstance(publication, Mapping):
        errors.append("publication_provenance object is required")
    else:
        if publication.get("schema_id") != (
            "fenics-nonlinear-energies.revision-publication-source-provenance"
        ) or publication.get("schema_version") != 1:
            errors.append("publication_provenance schema id/version is invalid")
        if publication.get("run_kind") != "publication":
            errors.append("publication_provenance.run_kind must be publication")
        experiment_commit = publication.get("experiment_commit")
        if not isinstance(experiment_commit, str) or not HEX40_RE.fullmatch(
            experiment_commit
        ):
            errors.append("publication_provenance.experiment_commit is invalid")
        producer = publication.get("producer")
        if not isinstance(producer, Mapping) or producer.get("path") != spec.producer_path.as_posix():
            errors.append(
                f"publication_provenance.producer.path must equal {spec.producer_path.as_posix()}"
            )
        elif not isinstance(producer.get("sha256"), str) or not HEX64_RE.fullmatch(
            str(producer.get("sha256"))
        ):
            errors.append("publication_provenance.producer.sha256 is malformed")
    wrapper = payload.get("source_schema")
    expected_schema_id = f"fenics-nonlinear-energies.revision-source.{spec.key}"
    if not isinstance(wrapper, Mapping):
        errors.append("source_schema object is required for publication admission")
    else:
        if wrapper.get("id") != expected_schema_id:
            errors.append(f"source_schema.id must equal {expected_schema_id}")
        if wrapper.get("version") != 1:
            errors.append("source_schema.version must equal 1")
    expected_experiment = {
        "plaplace": "EXP-VAL-001-PLAPLACE-MANUFACTURED",
        "ginzburg_landau": "EXP-VAL-001-GINZBURG-LANDAU-MANUFACTURED",
        "hyperelastic_patch": "EXP-VAL-001-HYPERELASTIC-AFFINE-PATCH",
        "hyperelastic_nonaffine": "EXP-VAL-001-HYPERELASTIC-NONAFFINE-MANUFACTURED",
        "smooth_derivatives": "EXP-DERIV-001-SMOOTH-FIXED-ELEMENT",
        "p1_derivatives": "EXP-DERIV-001-P3D-FIXED-ELEMENT",
        "p2_derivatives": "EXP-DERIV-001-P3D-FIXED-ELEMENT",
        "p4_derivatives": "EXP-DERIV-001-P3D-FIXED-ELEMENT",
        "material_point": "EXP-MC-001",
        "distribution": "EXP-DIST-001",
        "p1_quadrature": "EXP-DISC-001-P3D-FIXED-STATE-QUADRATURE",
        "p2_quadrature": "EXP-DISC-001-P3D-FIXED-STATE-QUADRATURE",
        "p4_quadrature": "EXP-DISC-001-P3D-FIXED-STATE-QUADRATURE",
        "route_analysis": "EXP-ROUTE-001",
    }[spec.key]
    experiment_value = (
        payload.get("experiment") if spec.key == "distribution" else payload.get("experiment_id")
    )
    if experiment_value != expected_experiment:
        errors.append(f"experiment identifier must equal {expected_experiment}")
    native_version = {
        "plaplace": 1,
        "ginzburg_landau": 1,
        "hyperelastic_patch": 1,
        "hyperelastic_nonaffine": 2,
    }.get(spec.key)
    if native_version is not None and payload.get("schema_version") != native_version:
        errors.append(f"native schema_version must equal {native_version}")
    if spec.key == "material_point" and (
        payload.get("schema_name") != "plasticity3d_material_point_verification"
        or payload.get("schema_version") != 1
    ):
        errors.append("material-point native schema name/version is invalid")
    if spec.key == "distribution":
        schema = payload.get("schema")
        if not isinstance(schema, Mapping) or schema.get("id") != (
            "fenics-nonlinear-energies.exp-dist-he-comparison"
        ) or schema.get("version") != 1:
            errors.append("distribution native schema id/version is invalid")
    if spec.key == "route_analysis" and payload.get("analysis_schema_version") != 1:
        errors.append("route analysis_schema_version must equal 1")
    return errors


def _manufactured_scalar_errors(
    spec: EvidenceSpec, payload: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    levels = payload.get("levels")
    rates = payload.get("rates")
    solver = payload.get("solver_contract")
    if not isinstance(levels, list) or len(levels) != 4:
        return ["manufactured study must contain exactly four mesh levels"]
    if not isinstance(rates, list) or len(rates) != 3:
        errors.append("manufactured study must contain exactly three adjacent-mesh rates")
        rates = []
    if not isinstance(solver, Mapping):
        errors.append("solver_contract is missing")
        solver = {}
    tolerance_max = 1.0e-8 if spec.key == "plaplace" else 1.0e-9
    residual_tolerance = _number(
        solver.get("relative_residual_tolerance"),
        "solver relative_residual_tolerance",
        errors,
        minimum=0.0,
        maximum=tolerance_max,
    )
    _number(
        solver.get("maximum_iterations"),
        "solver maximum_iterations",
        errors,
        minimum=1,
        maximum=50,
        integer=True,
    )
    subdivisions: list[int] = []
    mesh_sizes: list[float] = []
    for index, row in enumerate(levels):
        if not isinstance(row, Mapping):
            errors.append(f"levels[{index}] must be an object")
            continue
        if row.get("status") != "converged":
            errors.append(f"levels[{index}].status must be converged")
        subdivision = _number(
            row.get("subdivisions"),
            f"levels[{index}].subdivisions",
            errors,
            minimum=2,
            integer=True,
        )
        mesh_size = _number(row.get("h"), f"levels[{index}].h", errors, minimum=0.0)
        if subdivision is not None:
            subdivisions.append(int(subdivision))
        if mesh_size is not None:
            mesh_sizes.append(mesh_size)
        for field in ("l2_error", "h1_seminorm_error"):
            _number(row.get(field), f"levels[{index}].{field}", errors, minimum=0.0)
        _number(
            row.get("final_relative_residual"),
            f"levels[{index}].final_relative_residual",
            errors,
            minimum=0.0,
            maximum=residual_tolerance,
        )
        _number(
            row.get("tangent_symmetry_defect"),
            f"levels[{index}].tangent_symmetry_defect",
            errors,
            minimum=0.0,
            maximum=1.0e-12,
        )
        if spec.key == "plaplace":
            _number(
                row.get("minimum_element_gradient_norm"),
                f"levels[{index}].minimum_element_gradient_norm",
                errors,
                minimum=0.5,
            )
        else:
            _number(
                row.get("minimum_nodal_value"),
                f"levels[{index}].minimum_nodal_value",
                errors,
                minimum=1.0 / math.sqrt(3.0),
            )
    if subdivisions != [8, 16, 32, 64]:
        errors.append("manufactured subdivisions must equal [8, 16, 32, 64]")
    if len(mesh_sizes) == 4 and any(
        fine >= coarse for coarse, fine in zip(mesh_sizes[:-1], mesh_sizes[1:], strict=True)
    ):
        errors.append("manufactured mesh sizes must decrease strictly")
    for index, row in enumerate(rates):
        if not isinstance(row, Mapping):
            errors.append(f"rates[{index}] must be an object")
            continue
        if len(subdivisions) == 4 and (
            row.get("coarse_subdivisions") != subdivisions[index]
            or row.get("fine_subdivisions") != subdivisions[index + 1]
        ):
            errors.append(f"rates[{index}] does not identify its adjacent mesh pair")
        _number(row.get("l2_rate"), f"rates[{index}].l2_rate", errors, minimum=0.0)
        _number(
            row.get("h1_seminorm_rate"),
            f"rates[{index}].h1_seminorm_rate",
            errors,
            minimum=0.0,
        )
    if rates and isinstance(rates[-1], Mapping):
        _number(rates[-1].get("l2_rate"), "last l2_rate", errors, minimum=1.75)
        _number(
            rates[-1].get("h1_seminorm_rate"),
            "last h1_seminorm_rate",
            errors,
            minimum=0.85,
        )
    if spec.key == "plaplace":
        required = {
            "minimum_last_l2_rate",
            "minimum_last_h1_seminorm_rate",
            "maximum_symmetry_defect",
            "minimum_discrete_gradient_norm",
        }
        contract = _exact_keys(
            payload.get("acceptance_contract"), required, "acceptance_contract", errors
        ) or {}
        _number(contract.get("minimum_last_l2_rate"), "minimum_last_l2_rate", errors, minimum=1.75)
        _number(contract.get("minimum_last_h1_seminorm_rate"), "minimum_last_h1_seminorm_rate", errors, minimum=0.85)
        _number(contract.get("maximum_symmetry_defect"), "maximum_symmetry_defect", errors, minimum=0.0, maximum=1.0e-12)
        _number(contract.get("minimum_discrete_gradient_norm"), "minimum_discrete_gradient_norm", errors, minimum=0.5)
    return errors


def _affine_patch_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        errors.append("affine-patch contract is missing")
        contract = {}
    tolerance = _number(
        contract.get("relative_tolerance"),
        "affine relative_tolerance",
        errors,
        minimum=0.0,
        maximum=2.0e-11,
    )
    required = {
        "energy_relative_error",
        "residual_relative_error",
        "hessian_relative_error",
        "hessian_symmetry_defect",
        "traction_balance_relative_error",
        "net_internal_force_norm",
        "objectivity_energy_relative_error",
        "piola_rotation_covariance_relative_error",
        "translation_mode_hessian_action_norms",
    }
    metrics = _exact_keys(payload.get("metrics"), required, "affine-patch metrics", errors) or {}
    for field in required - {"translation_mode_hessian_action_norms"}:
        _number(metrics.get(field), f"metrics.{field}", errors, minimum=0.0, maximum=tolerance)
    translations = metrics.get("translation_mode_hessian_action_norms")
    if not isinstance(translations, list) or len(translations) != 3:
        errors.append("translation_mode_hessian_action_norms must contain three modes")
    else:
        for index, value in enumerate(translations):
            _number(value, f"translation mode {index}", errors, minimum=0.0, maximum=tolerance)
    case = payload.get("case")
    if not isinstance(case, Mapping):
        errors.append("affine-patch case is missing")
    else:
        _number(case.get("determinant"), "case.determinant", errors, minimum=1.0e-12)
    return errors


def _hyperelastic_nonaffine_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        return ["nonaffine hyperelastic contract is missing"]
    if contract.get("subdivisions") != [4, 8, 16, 24]:
        errors.append("nonaffine subdivisions must equal [4, 8, 16, 24]")
    if contract.get("load_quadrature_orders") != [4, 6, 8]:
        errors.append("load quadrature orders must equal [4, 6, 8]")
    residual_gate = _number(
        contract.get("relative_algebraic_residual"),
        "relative_algebraic_residual",
        errors,
        minimum=0.0,
        maximum=1.0e-10,
    )
    symmetry_gate = _number(
        contract.get("tangent_symmetry_tolerance"),
        "tangent_symmetry_tolerance",
        errors,
        minimum=0.0,
        maximum=1.0e-11,
    )
    determinant_gate = _number(
        contract.get("minimum_determinant"),
        "minimum_determinant",
        errors,
        minimum=0.5,
    )
    fraction_gate = _number(
        contract.get("maximum_load_quadrature_error_fraction"),
        "maximum_load_quadrature_error_fraction",
        errors,
        minimum=0.0,
        maximum=0.01,
    )
    rate_contract = _exact_keys(
        contract.get("last_pair_minimum_rates"),
        {"first_piola_l2", "h1_deformation", "l2_displacement"},
        "last_pair_minimum_rates",
        errors,
    ) or {}
    for field, minimum in {
        "first_piola_l2": 0.75,
        "h1_deformation": 0.75,
        "l2_displacement": 1.75,
    }.items():
        _number(rate_contract.get(field), f"last_pair_minimum_rates.{field}", errors, minimum=minimum)
    required_gates = {
        "algebraic_residual",
        "all_levels_converged",
        "h1_rate",
        "l2_rate",
        "load_quadrature_below_consistency_error",
        "load_quadrature_below_fe_error",
        "load_quadrature_reference_stable",
        "load_quadrature_refined_solves_converged",
        "load_quadrature_refined_solves_resolve_load_change",
        "orientation",
        "stress_rate",
        "tangent_symmetry",
    }
    gates = _exact_keys(payload.get("gates"), required_gates, "nonaffine gates", errors)
    if gates is not None and any(gates.get(key) is not True for key in required_gates):
        errors.append("every nonaffine named gate must be true")
    levels = payload.get("levels")
    if not isinstance(levels, list) or len(levels) != 4:
        errors.append("nonaffine study must contain four levels")
        levels = []
    for index, row in enumerate(levels):
        if not isinstance(row, Mapping):
            errors.append(f"nonaffine levels[{index}] must be an object")
            continue
        if row.get("subdivisions") != [4, 8, 16, 24][index] or row.get("status") != "converged":
            errors.append(f"nonaffine levels[{index}] has wrong subdivision/status")
        for field in ("l2_displacement_error", "h1_deformation_error", "first_piola_l2_error"):
            _number(row.get(field), f"levels[{index}].{field}", errors, minimum=0.0)
        _number(
            row.get("final_relative_residual"),
            f"levels[{index}].final_relative_residual",
            errors,
            minimum=0.0,
            maximum=residual_gate,
        )
        _number(
            row.get("minimum_discrete_determinant"),
            f"levels[{index}].minimum_discrete_determinant",
            errors,
            minimum=determinant_gate,
        )
        _number(
            row.get("tangent_symmetry_defect"),
            f"levels[{index}].tangent_symmetry_defect",
            errors,
            minimum=0.0,
            maximum=symmetry_gate,
        )
        load = row.get("load_quadrature_check")
        if not isinstance(load, Mapping):
            errors.append(f"levels[{index}].load_quadrature_check is missing")
        else:
            if (
                load.get("below_fe_error") is not True
                or load.get("reference_load_stable") is not True
                or load.get("refined_solution_status") != "converged"
                or load.get("refined_solution_resolves_load_change") is not True
            ):
                errors.append(f"levels[{index}] load-quadrature status gates failed")
            _number(
                load.get("maximum_fraction_of_fe_error"),
                f"levels[{index}].maximum_fraction_of_fe_error",
                errors,
                minimum=0.0,
                maximum=fraction_gate,
            )
    rates = payload.get("rates")
    if not isinstance(rates, list) or len(rates) != 3:
        errors.append("nonaffine study must contain three adjacent rates")
    else:
        for index, row in enumerate(rates):
            if not isinstance(row, Mapping):
                errors.append(f"rates[{index}] must be an object")
                continue
            for field in ("l2_displacement_error", "h1_deformation_error", "first_piola_l2_error"):
                _number(row.get(field), f"rates[{index}].{field}", errors, minimum=0.0)
        last = rates[-1]
        if isinstance(last, Mapping):
            _number(last.get("l2_displacement_error"), "last displacement rate", errors, minimum=1.75)
            _number(last.get("h1_deformation_error"), "last deformation rate", errors, minimum=0.75)
            _number(last.get("first_piola_l2_error"), "last Piola rate", errors, minimum=0.75)
    return errors


def _assembled_derivative_errors(
    spec: EvidenceSpec, payload: Mapping[str, Any]
) -> list[str]:
    """Validate the managed serial assembled-route comparison fail-closed."""
    errors: list[str] = []
    expected_degree = {
        "p1_derivatives": 1,
        "p2_derivatives": 2,
        "p4_derivatives": 4,
    }[spec.key]
    top_case = payload.get("case")
    if not isinstance(top_case, Mapping):
        errors.append("derivative case must be an object")
    else:
        if top_case.get("degree") != expected_degree:
            errors.append(f"derivative case.degree must equal {expected_degree}")
        if top_case.get("mesh_name") != "hetero_ssr_L1":
            errors.append("derivative case.mesh_name must equal hetero_ssr_L1")

    assembled = payload.get("assembled_route_equivalence")
    if not isinstance(assembled, Mapping):
        return errors + [
            "assembled_route_equivalence must be an object for managed P1/P2/P4 evidence"
        ]
    if assembled.get("status") != "passed":
        errors.append("assembled_route_equivalence.status must be passed")
    if assembled.get("all_values_finite") is not True:
        errors.append("assembled_route_equivalence.all_values_finite must be true")
    if assembled.get("all_hessians_symmetric_within_tolerance") is not True:
        errors.append(
            "assembled_route_equivalence.all_hessians_symmetric_within_tolerance must be true"
        )

    case_keys = {
        "constraint_variant",
        "degree",
        "elements",
        "free_dofs",
        "lambda_target",
        "mesh_name",
        "state_definition",
        "state_norm",
        "state_scale",
    }
    case = _exact_keys(
        assembled.get("case"), case_keys, "assembled_route_equivalence.case", errors
    ) or {}
    if case.get("degree") != expected_degree:
        errors.append(f"assembled route case.degree must equal {expected_degree}")
    if case.get("mesh_name") != "hetero_ssr_L1":
        errors.append("assembled route case.mesh_name must equal hetero_ssr_L1")
    if case.get("constraint_variant") != "glued_bottom":
        errors.append("assembled route constraint_variant must equal glued_bottom")
    lambda_target = _number(
        case.get("lambda_target"), "assembled route lambda_target", errors
    )
    if lambda_target is not None and lambda_target != 1.5:
        errors.append("assembled route lambda_target must equal 1.5")
    for field in ("free_dofs", "elements"):
        _number(
            case.get(field),
            f"assembled route {field}",
            errors,
            minimum=1,
            integer=True,
        )
    state_scale = _number(
        case.get("state_scale"), "assembled route state_scale", errors, minimum=0.0
    )
    if state_scale is not None and state_scale != 1.0e-8:
        errors.append("assembled route state_scale must equal frozen value 1e-08")
    state_norm = _number(
        case.get("state_norm"), "assembled route state_norm", errors, minimum=0.0
    )
    if state_norm is not None and state_norm <= 0.0:
        errors.append("assembled route state_norm must be positive")
    if not isinstance(case.get("state_definition"), str) or not case.get(
        "state_definition"
    ):
        errors.append("assembled route state_definition must be nonempty")

    expected_contract = {
        "value_atol": 1.0e-12,
        "value_rtol": 1.0e-12,
        "gradient_norm_atol": 1.0e-10,
        "gradient_norm_rtol": 1.0e-12,
        "hessian_maximum_entry_atol": 1.0e-8,
        "hessian_frobenius_rtol": 1.0e-12,
        "hessian_symmetry_tolerance": 1.0e-12,
    }
    contract = _exact_keys(
        assembled.get("contract"),
        set(expected_contract) | {"branch_gate"},
        "assembled_route_equivalence.contract",
        errors,
    ) or {}
    for field, expected in expected_contract.items():
        value = _number(
            contract.get(field), f"assembled route contract.{field}", errors, minimum=0.0
        )
        if value is not None and value != expected:
            errors.append(
                f"assembled route contract.{field} must equal frozen value {expected}"
            )
    if contract.get("branch_gate") != "every quadrature point must satisfy trial_yield < 0":
        errors.append("assembled route branch_gate differs from the frozen contract")

    branch_keys = {
        "all_quadrature_points_strictly_elastic",
        "elastic_quadrature_points",
        "interpretation",
        "maximum_trial_yield",
        "minimum_normalized_elastic_margin",
        "minimum_trial_yield",
        "plastic_quadrature_points",
        "quadrature_points",
    }
    branch = _exact_keys(
        assembled.get("branch_diagnostics"),
        branch_keys,
        "assembled_route_equivalence.branch_diagnostics",
        errors,
    ) or {}
    quadrature_points = _number(
        branch.get("quadrature_points"),
        "assembled route quadrature_points",
        errors,
        minimum=1,
        integer=True,
    )
    elastic_points = _number(
        branch.get("elastic_quadrature_points"),
        "assembled route elastic_quadrature_points",
        errors,
        minimum=1,
        integer=True,
    )
    plastic_points = _number(
        branch.get("plastic_quadrature_points"),
        "assembled route plastic_quadrature_points",
        errors,
        minimum=0,
        integer=True,
    )
    if branch.get("all_quadrature_points_strictly_elastic") is not True:
        errors.append("assembled route must be strictly elastic at every quadrature point")
    if plastic_points is not None and plastic_points != 0:
        errors.append("assembled route plastic_quadrature_points must equal zero")
    if (
        quadrature_points is not None
        and elastic_points is not None
        and elastic_points != quadrature_points
    ):
        errors.append("assembled route elastic count must equal quadrature-point count")
    maximum_trial = _number(
        branch.get("maximum_trial_yield"), "assembled route maximum_trial_yield", errors
    )
    if maximum_trial is not None and maximum_trial >= 0.0:
        errors.append("assembled route maximum_trial_yield must be negative")
    margin = _number(
        branch.get("minimum_normalized_elastic_margin"),
        "assembled route minimum_normalized_elastic_margin",
        errors,
        minimum=0.0,
    )
    if margin is not None and margin <= 0.0:
        errors.append("assembled route normalized elastic margin must be positive")
    _number(branch.get("minimum_trial_yield"), "assembled route minimum_trial_yield", errors)

    route_names = {"element_ad", "local_sfd", "constitutive_ad"}
    routes = _exact_keys(
        assembled.get("routes"),
        route_names,
        "assembled_route_equivalence.routes",
        errors,
    ) or {}
    route_values: dict[str, dict[str, float]] = {}
    route_keys = {
        "assembly_mode",
        "energy",
        "gradient_norm",
        "hessian_frobenius_norm",
        "hessian_nonzeros",
        "hessian_symmetry_defect",
    }
    for route_name in sorted(route_names):
        route = _exact_keys(
            routes.get(route_name),
            route_keys,
            f"assembled route {route_name}",
            errors,
        ) or {}
        energy = _number(route.get("energy"), f"{route_name}.energy", errors)
        gradient_norm = _number(
            route.get("gradient_norm"),
            f"{route_name}.gradient_norm",
            errors,
            minimum=0.0,
        )
        hessian_norm = _number(
            route.get("hessian_frobenius_norm"),
            f"{route_name}.hessian_frobenius_norm",
            errors,
            minimum=0.0,
        )
        _number(
            route.get("hessian_symmetry_defect"),
            f"{route_name}.hessian_symmetry_defect",
            errors,
            minimum=0.0,
            maximum=expected_contract["hessian_symmetry_tolerance"],
        )
        _number(
            route.get("hessian_nonzeros"),
            f"{route_name}.hessian_nonzeros",
            errors,
            minimum=1,
            integer=True,
        )
        if not isinstance(route.get("assembly_mode"), str) or not route.get(
            "assembly_mode"
        ):
            errors.append(f"{route_name}.assembly_mode must be nonempty")
        if energy is not None and gradient_norm is not None and hessian_norm is not None:
            route_values[route_name] = {
                "energy": energy,
                "gradient_norm": gradient_norm,
                "hessian_norm": hessian_norm,
            }

    comparisons = assembled.get("pairwise_comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 3:
        errors.append("assembled route comparison must contain exactly three route pairs")
        comparisons = []
    comparison_keys = {
        "energy_absolute_error",
        "energy_relative_error",
        "gradient_absolute_error",
        "gradient_relative_error",
        "hessian_absolute_error",
        "hessian_csr_structure_equal",
        "hessian_maximum_entry_error",
        "hessian_relative_error",
        "left",
        "passed",
        "right",
    }
    observed_pairs: set[frozenset[str]] = set()
    for index, raw_comparison in enumerate(comparisons):
        comparison = _exact_keys(
            raw_comparison,
            comparison_keys,
            f"assembled route pairwise_comparisons[{index}]",
            errors,
        ) or {}
        left = comparison.get("left")
        right = comparison.get("right")
        if left not in route_names or right not in route_names or left == right:
            errors.append(f"assembled route comparison {index} has invalid route names")
        else:
            observed_pairs.add(frozenset((str(left), str(right))))
        if comparison.get("passed") is not True:
            errors.append(f"assembled route comparison {index} must pass")
        if comparison.get("hessian_csr_structure_equal") is not True:
            errors.append(f"assembled route comparison {index} must have equal CSR structure")
        values: dict[str, float | None] = {}
        for field in (
            "energy_absolute_error",
            "energy_relative_error",
            "gradient_absolute_error",
            "gradient_relative_error",
            "hessian_absolute_error",
            "hessian_maximum_entry_error",
            "hessian_relative_error",
        ):
            values[field] = _number(
                comparison.get(field),
                f"assembled route comparison {index}.{field}",
                errors,
                minimum=0.0,
            )
        hessian_relative = values["hessian_relative_error"]
        if (
            hessian_relative is not None
            and hessian_relative > expected_contract["hessian_frobenius_rtol"]
        ):
            errors.append(f"assembled route comparison {index} fails Hessian relative gate")
        hessian_maximum = values["hessian_maximum_entry_error"]
        if (
            hessian_maximum is not None
            and hessian_maximum > expected_contract["hessian_maximum_entry_atol"]
        ):
            errors.append(f"assembled route comparison {index} fails Hessian entry gate")
        if left in route_values and right in route_values:
            energy_gate = expected_contract["value_atol"] + expected_contract[
                "value_rtol"
            ] * max(
                abs(route_values[str(left)]["energy"]),
                abs(route_values[str(right)]["energy"]),
            )
            energy_absolute = values["energy_absolute_error"]
            if energy_absolute is not None and energy_absolute > energy_gate:
                errors.append(f"assembled route comparison {index} fails energy gate")
            gradient_gate = expected_contract["gradient_norm_atol"] + expected_contract[
                "gradient_norm_rtol"
            ] * max(
                route_values[str(left)]["gradient_norm"],
                route_values[str(right)]["gradient_norm"],
            )
            gradient_absolute = values["gradient_absolute_error"]
            if gradient_absolute is not None and gradient_absolute > gradient_gate:
                errors.append(f"assembled route comparison {index} fails gradient gate")
    expected_pairs = {
        frozenset(("element_ad", "local_sfd")),
        frozenset(("element_ad", "constitutive_ad")),
        frozenset(("local_sfd", "constitutive_ad")),
    }
    if observed_pairs != expected_pairs:
        errors.append("assembled route comparison does not cover each route pair exactly once")

    scope = _exact_keys(
        assembled.get("algebraic_scope"),
        {
            "interpretation",
            "ksp_tolerance_used_for_comparison",
            "linear_solver_called",
            "local_sfd_meaning",
            "nonlinear_solver_called",
        },
        "assembled_route_equivalence.algebraic_scope",
        errors,
    ) or {}
    if scope.get("linear_solver_called") is not False:
        errors.append("assembled route comparison must not call a linear solver")
    if scope.get("nonlinear_solver_called") is not False:
        errors.append("assembled route comparison must not call a nonlinear solver")
    if scope.get("ksp_tolerance_used_for_comparison") is not None:
        errors.append("assembled route comparison must not depend on a KSP tolerance")
    return errors


def _derivative_errors(spec: EvidenceSpec, payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    contract = payload.get("contract")
    summary = payload.get("summary")
    if not isinstance(contract, Mapping) or not isinstance(summary, Mapping):
        return ["derivative contract and summary must be objects"]
    smooth = spec.key == "smooth_derivatives"
    expected_contract = {
        "route_relative_tolerance": 1.0e-10 if smooth else 1.0e-9,
        "symmetry_tolerance": 1.0e-12 if smooth else 1.0e-10,
        "centered_fd_tolerance": 1.0e-7,
        "centered_fd_gate_index": 3 if smooth else 2,
        "centered_fd_gate_step": 3.0e-5 if smooth else 1.0e-7,
    }
    for field, expected in expected_contract.items():
        value = _number(contract.get(field), f"contract.{field}", errors, minimum=0.0)
        if value is not None and value != expected:
            errors.append(f"contract.{field} must equal frozen value {expected}")
    required = (
        {
            "cases",
            "maximum_gradient_relative_error",
            "maximum_hessian_relative_error",
            "maximum_hessian_symmetry_defect",
            "maximum_fd_gradient_error_at_gate",
            "maximum_fd_hvp_error_at_gate",
        }
        if smooth
        else {
            "states",
            "maximum_residual_relative_error",
            "maximum_hessian_relative_error",
            "maximum_hessian_symmetry_defect",
            "maximum_centered_fd_energy_error_at_gate",
            "maximum_centered_fd_hvp_error_at_gate",
            "all_states_branch_stable_at_fd_gate",
            "fixed_element_status",
            "assembled_route_equivalence_status",
        }
    )
    summary = _exact_keys(summary, required, "derivative summary", errors) or {}
    count_field = "cases" if smooth else "states"
    count = _number(
        summary.get(count_field),
        f"summary.{count_field}",
        errors,
        minimum=5,
        integer=True,
    )
    route_field = "maximum_gradient_relative_error" if smooth else "maximum_residual_relative_error"
    fd_fields = (
        ("maximum_fd_gradient_error_at_gate", "maximum_fd_hvp_error_at_gate")
        if smooth
        else ("maximum_centered_fd_energy_error_at_gate", "maximum_centered_fd_hvp_error_at_gate")
    )
    _number(summary.get(route_field), f"summary.{route_field}", errors, minimum=0.0, maximum=expected_contract["route_relative_tolerance"])
    _number(summary.get("maximum_hessian_relative_error"), "summary.maximum_hessian_relative_error", errors, minimum=0.0, maximum=expected_contract["route_relative_tolerance"])
    _number(summary.get("maximum_hessian_symmetry_defect"), "summary.maximum_hessian_symmetry_defect", errors, minimum=0.0, maximum=expected_contract["symmetry_tolerance"])
    for field in fd_fields:
        _number(summary.get(field), f"summary.{field}", errors, minimum=0.0, maximum=expected_contract["centered_fd_tolerance"])
    if not smooth and summary.get("all_states_branch_stable_at_fd_gate") is not True:
        errors.append("all_states_branch_stable_at_fd_gate must be true")
    if not smooth and summary.get("fixed_element_status") != "passed":
        errors.append("summary.fixed_element_status must be passed")
    if not smooth and summary.get("assembled_route_equivalence_status") != "passed":
        errors.append("summary.assembled_route_equivalence_status must be passed")
    records = payload.get("records")
    if not isinstance(records, list) or count is None or len(records) != int(count):
        errors.append("derivative record count must equal the declared case/state count")
    if not smooth:
        errors.extend(_assembled_derivative_errors(spec, payload))
    return errors


def _material_point_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    contract = payload.get("contract")
    summary = payload.get("summary")
    if not isinstance(contract, Mapping) or not isinstance(summary, Mapping):
        return ["material-point contract and summary must be objects"]
    bounds = {
        "centered_fd_scaled_error_tolerance": 1.0e-7,
        "hessian_symmetry_tolerance": 1.0e-10,
        "numpy_energy_transcription_relative_tolerance": 1.0e-12,
        "rotation_absolute_tolerance_for_near_zero_tangent_actions": 1.0e-9,
        "rotation_scaled_tolerance": 1.0e-9,
    }
    numeric_contract: dict[str, float | None] = {}
    for field, maximum in bounds.items():
        numeric_contract[field] = _number(
            contract.get(field),
            f"contract.{field}",
            errors,
            minimum=0.0,
            maximum=maximum,
        )
    margin_gate = _number(
        contract.get("minimum_normalized_active_branch_margin"),
        "contract.minimum_normalized_active_branch_margin",
        errors,
        minimum=1.0e-3,
    )
    branches = {"elastic", "shear", "left_edge", "right_edge", "apex"}
    required_branches = contract.get("required_branches")
    if not isinstance(required_branches, list) or any(
        not isinstance(value, str) for value in required_branches
    ) or set(required_branches) != branches:
        errors.append("required_branches must equal the five frozen branches")
    for field in (
        "cpu_fp64_execution_passed",
        "degeneracy_finiteness_checks_passed",
        "interface_sweeps_passed",
        "interior_checks_passed",
        "rotation_checks_passed",
    ):
        if summary.get(field) is not True:
            errors.append(f"summary.{field} must be true")
    counts = _exact_keys(
        summary.get("branch_interior_counts"), branches, "branch_interior_counts", errors
    ) or {}
    for branch in branches:
        _number(
            counts.get(branch),
            f"branch_interior_counts.{branch}",
            errors,
            minimum=1,
            integer=True,
        )
    for field, minimum in (
        ("degeneracy_case_count", 7),
        ("interface_count", 5),
        ("interface_pair_count", 15),
        ("rotation_check_count", 15),
    ):
        _number(summary.get(field), f"summary.{field}", errors, minimum=minimum, integer=True)
    metric_bounds = {
        "maximum_centered_energy_directional_error_at_gate": numeric_contract["centered_fd_scaled_error_tolerance"],
        "maximum_centered_hvp_error_at_gate": numeric_contract["centered_fd_scaled_error_tolerance"],
        "maximum_hessian_symmetry_defect": numeric_contract["hessian_symmetry_tolerance"],
        "maximum_numpy_energy_transcription_relative_error": numeric_contract["numpy_energy_transcription_relative_tolerance"],
        "maximum_interface_numpy_energy_transcription_relative_error": numeric_contract["numpy_energy_transcription_relative_tolerance"],
        "maximum_degeneracy_numpy_energy_transcription_relative_error": numeric_contract["numpy_energy_transcription_relative_tolerance"],
        "maximum_rotation_energy_invariance_scaled_error": numeric_contract["rotation_scaled_tolerance"],
        "maximum_rotation_stress_covariance_scaled_error": numeric_contract["rotation_scaled_tolerance"],
        "maximum_rotation_tangent_action_covariance_scaled_error": numeric_contract["rotation_scaled_tolerance"],
        "maximum_rotation_tangent_action_covariance_absolute_error": numeric_contract["rotation_absolute_tolerance_for_near_zero_tangent_actions"],
    }
    for field, maximum in metric_bounds.items():
        _number(summary.get(field), f"summary.{field}", errors, minimum=0.0, maximum=maximum)
    _number(
        summary.get("minimum_normalized_active_branch_margin"),
        "summary.minimum_normalized_active_branch_margin",
        errors,
        minimum=margin_gate,
    )
    return errors


def _distribution_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    derivative_keys = {
        "energy_relative",
        "matrix_action_relative",
        "matrix_relative",
        "residual_relative",
    }
    exact_object_keys = {"direction", "matrix_indices", "matrix_indptr", "state"}
    topology_keys = {
        "affine_lift",
        "connectivity",
        "coordinates",
        "freedofs",
        "right_boundary_nodes",
    }
    linear_keys = {
        "candidate_true_residual",
        "linear_correction",
        "reference_true_residual",
    }
    expected_factors = {
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
    }
    if payload.get("controlled_factors") != expected_factors:
        errors.append("controlled_factors must equal the frozen EXP-DIST-001 rank-count design")
    if payload.get("varied_factor") != {"name": "mpi_ranks", "levels": [1, 2, 4]}:
        errors.append("varied_factor must contain exactly mpi_ranks levels [1, 2, 4]")

    def validate_comparison(value: Any, label: str) -> Mapping[str, Any] | None:
        if not isinstance(value, Mapping):
            errors.append(f"{label} must be an object")
            return None
        for block_name, expected in (
            ("derivative_gates", derivative_keys),
            ("exact_object_gates", exact_object_keys),
            ("exact_topology_gates", topology_keys),
            ("linear_solve_gates", linear_keys),
        ):
            block = _exact_keys(
                value.get(block_name), expected, f"{label}.{block_name}", errors
            )
            if block is not None and any(block.get(key) is not True for key in expected):
                errors.append(f"every {label}.{block_name} gate must be true")
        if value.get("algebraic_gate_passed") is not True:
            errors.append(f"{label}.algebraic_gate_passed must be true")
        derivative_tolerance = _number(
            value.get("derivative_tolerance"),
            f"{label}.derivative_tolerance",
            errors,
            minimum=0.0,
            maximum=1.0e-8,
        )
        solve_tolerance = _number(
            value.get("solve_tolerance"),
            f"{label}.solve_tolerance",
            errors,
            minimum=0.0,
            maximum=1.0e-8,
        )
        relative_keys = derivative_keys | {"linear_correction_relative"}
        relative = _exact_keys(
            value.get("relative_errors"),
            relative_keys,
            f"{label}.relative_errors",
            errors,
        ) or {}
        for field in derivative_keys:
            _number(
                relative.get(field),
                f"{label}.relative_errors.{field}",
                errors,
                minimum=0.0,
                maximum=derivative_tolerance,
            )
        _number(
            relative.get("linear_correction_relative"),
            f"{label}.relative_errors.linear_correction_relative",
            errors,
            minimum=0.0,
            maximum=solve_tolerance,
        )
        return value

    comparison = validate_comparison(payload.get("comparison"), "comparison")
    rank_comparisons = _exact_keys(
        payload.get("rank_comparisons"), {"np2", "np4"}, "rank_comparisons", errors
    )
    parsed_rank_comparisons: dict[str, Mapping[str, Any]] = {}
    if rank_comparisons is not None:
        for key in ("np2", "np4"):
            parsed = validate_comparison(rank_comparisons.get(key), f"rank_comparisons.{key}")
            if parsed is not None:
                parsed_rank_comparisons[key] = parsed

    if comparison is not None and set(parsed_rank_comparisons) == {"np2", "np4"}:
        candidates = [parsed_rank_comparisons["np2"], parsed_rank_comparisons["np4"]]
        if comparison.get("algebraic_gate_passed") is not all(
            candidate.get("algebraic_gate_passed") is True for candidate in candidates
        ):
            errors.append("comparison algebraic gate is inconsistent with rank_comparisons")
        for block_name in (
            "derivative_gates",
            "exact_object_gates",
            "exact_topology_gates",
            "linear_solve_gates",
        ):
            aggregate_block = comparison.get(block_name)
            if not isinstance(aggregate_block, Mapping):
                continue
            for key in aggregate_block:
                expected = all(
                    isinstance(candidate.get(block_name), Mapping)
                    and candidate[block_name].get(key) is True
                    for candidate in candidates
                )
                if aggregate_block.get(key) is not expected:
                    errors.append(
                        f"comparison.{block_name}.{key} is inconsistent with rank_comparisons"
                    )
        aggregate_relative = comparison.get("relative_errors")
        if isinstance(aggregate_relative, Mapping):
            for key, value in aggregate_relative.items():
                candidate_values = [
                    candidate.get("relative_errors", {}).get(key)
                    for candidate in candidates
                    if isinstance(candidate.get("relative_errors"), Mapping)
                ]
                if len(candidate_values) != 2 or any(
                    isinstance(item, bool) or not isinstance(item, (int, float))
                    for item in candidate_values
                ):
                    continue
                expected = max(float(item) for item in candidate_values)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isclose(
                    float(value), expected, rel_tol=0.0, abs_tol=0.0
                ):
                    errors.append(
                        f"comparison.relative_errors.{key} is not the rank-comparison maximum"
                    )
        for tolerance in ("derivative_tolerance", "solve_tolerance"):
            values = [comparison.get(tolerance), *(candidate.get(tolerance) for candidate in candidates)]
            if any(value != values[0] for value in values[1:]):
                errors.append(f"{tolerance} differs across rank comparisons")

    workers = _exact_keys(payload.get("workers"), {"np1", "np2", "np4"}, "workers", errors)
    if workers is not None:
        for key in ("np1", "np2", "np4"):
            if not isinstance(workers.get(key), Mapping) or workers[key].get("status") != "passed":
                errors.append(f"workers.{key}.status must be passed")
    return errors


def _walk_artifact_references(
    value: Any,
    *,
    trail: tuple[str, ...] = (),
) -> Iterable[tuple[tuple[str, ...], str, Any, Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_trail = (*trail, str(key))
            if isinstance(key, str) and key.endswith("_artifact"):
                yield child_trail, key, child, value
            yield from _walk_artifact_references(child, trail=child_trail)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_artifact_references(
                child,
                trail=(*trail, f"[{index}]"),
            )


def _array_content_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _quadrature_referenced_artifact_errors(
    spec: EvidenceSpec,
    payload: Mapping[str, Any],
    *,
    evidence_root: Path,
) -> list[str]:
    """Independently verify every publication-critical DISC array reference."""
    errors: list[str] = []
    degree = {"p1_quadrature": 1, "p2_quadrature": 2, "p4_quadrature": 4}[spec.key]
    expected = {
        (rule, field): Path(
            f"_publication_staging/EXP-DISC-001/actions/p{degree}_l1/"
            f"{rule}_{suffix}.npy"
        )
        for rule in QUADRATURE_RULE_IDS
        for field, (suffix, _content_field, _dtype) in QUADRATURE_ARTIFACT_FIELDS.items()
    }
    found: set[tuple[str, str]] = set()
    descriptor_hashes: dict[str, str] = {}
    for trail, field, descriptor, container in _walk_artifact_references(payload):
        location = ".".join(trail)
        if field not in QUADRATURE_ARTIFACT_FIELDS:
            errors.append(f"{location} is an unsupported publication artifact reference")
            continue
        if not isinstance(descriptor, Mapping):
            errors.append(f"{location} must be an artifact object")
            continue
        identity = (str(container.get("quadrature_rule_id")), field)
        expected_relative = expected.get(identity)
        if expected_relative is None:
            errors.append(f"{location} has an unknown quadrature rule/artifact identity")
            continue
        if identity in found:
            errors.append(f"{location} duplicates quadrature artifact {identity}")
            continue
        found.add(identity)
        raw_path = descriptor.get("path")
        if not isinstance(raw_path, str):
            errors.append(f"{location}.path must be a string")
            continue
        relative = Path(raw_path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "." in relative.parts
            or relative.as_posix() != raw_path
        ):
            errors.append(f"{location}.path must be canonical and relative")
            continue
        if relative != expected_relative:
            errors.append(
                f"{location}.path must be {expected_relative.as_posix()}"
            )
            continue
        file_digest = descriptor.get("sha256")
        if not isinstance(file_digest, str) or not HEX64_RE.fullmatch(file_digest):
            errors.append(f"{location}.sha256 is malformed")
            continue
        descriptor_hashes[relative.as_posix()] = file_digest
        candidate = evidence_root / relative
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            errors.append(f"{location}.path is missing")
            continue
        if (
            not _is_contained(resolved, evidence_root.resolve())
            or candidate.is_symlink()
            or resolved != candidate.absolute()
        ):
            errors.append(f"{location}.path escapes or traverses a symlink")
            continue
        if not candidate.is_file() or sha256_file(candidate) != file_digest:
            errors.append(f"{location}.sha256 mismatch")
            continue
        _suffix, content_field, expected_dtype = QUADRATURE_ARTIFACT_FIELDS[field]
        content_digest = descriptor.get("content_sha256")
        if not isinstance(content_digest, str) or not HEX64_RE.fullmatch(content_digest):
            errors.append(f"{location}.content_sha256 is malformed")
            continue
        if container.get(content_field) != content_digest:
            errors.append(f"{location} disagrees with {content_field}")
        try:
            array = np.load(candidate, allow_pickle=False)
        except (OSError, ValueError) as exc:
            errors.append(f"{location} is not a safe NPY array: {exc}")
            continue
        if not isinstance(array, np.ndarray) or array.dtype.hasobject:
            errors.append(f"{location} must contain one non-object array")
            continue
        shape = descriptor.get("shape")
        if (
            not isinstance(shape, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in shape
            )
            or tuple(shape) != array.shape
        ):
            errors.append(f"{location}.shape does not match its array")
        if descriptor.get("dtype") != expected_dtype or str(array.dtype) != expected_dtype:
            errors.append(f"{location}.dtype does not match its array")
        if _array_content_sha256(array) != content_digest:
            errors.append(f"{location}.content_sha256 mismatch")

    if set(expected) != found:
        missing = sorted(set(expected) - found)
        errors.append(
            "quadrature payload does not reference the complete residual/action/branch-map "
            f"array set; missing={missing}"
        )
    provenance = payload.get("publication_provenance")
    declared = (
        provenance.get("referenced_artifact_hashes")
        if isinstance(provenance, Mapping)
        else None
    )
    if not isinstance(declared, Mapping):
        errors.append(
            "publication_provenance.referenced_artifact_hashes must be a path/hash map"
        )
    elif dict(declared) != dict(sorted(descriptor_hashes.items())):
        errors.append(
            "publication_provenance.referenced_artifact_hashes differs from recursive "
            "quadrature references"
        )
    return errors


def _quadrature_errors(spec: EvidenceSpec, payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    degree = {"p1_quadrature": 1, "p2_quadrature": 2, "p4_quadrature": 4}[spec.key]
    solve_rule = {1: "tetra_1point", 2: "tetra_11point", 4: "tetra_24point"}[degree]
    if payload.get("element_degree") != degree:
        errors.append(f"element_degree must equal {degree}")
    if payload.get("solve_quadrature_rule_id") != solve_rule:
        errors.append(f"solve_quadrature_rule_id must equal {solve_rule}")
    if payload.get("reference_rule_id") != "tetra_duffy_125point":
        errors.append("reference_rule_id must equal tetra_duffy_125point")
    if payload.get("constraint_variant") != "glued_bottom" or payload.get("mesh_name") != "hetero_ssr_L1":
        errors.append("quadrature mesh/constraint protocol changed")
    _number(payload.get("lambda_target"), "lambda_target", errors, minimum=1.55, maximum=1.55)
    _number(payload.get("reference_energy_scale"), "reference_energy_scale", errors, minimum=1.0e-30)
    if payload.get("common_free_dof_set") is not True:
        errors.append("common_free_dof_set must be true")
    direction_hash = payload.get("common_direction_content_sha256")
    if not isinstance(direction_hash, str) or not HEX64_RE.fullmatch(direction_hash):
        errors.append("common_direction_content_sha256 must be a SHA-256 digest")
    expected_rules = {
        "tetra_1point": 1,
        "tetra_11point": 11,
        "tetra_24point": 24,
        "tetra_duffy_125point": 125,
    }
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 4:
        return errors + ["quadrature evaluations must contain exactly four frozen rules"]
    by_rule = {
        str(row.get("quadrature_rule_id")): row
        for row in evaluations
        if isinstance(row, Mapping)
    }
    if set(by_rule) != set(expected_rules):
        errors.append("quadrature evaluation rule set is incomplete or changed")
    for rule, points in expected_rules.items():
        row = by_rule.get(rule)
        if not isinstance(row, Mapping):
            continue
        if row.get("quadrature_points_per_element") != points or row.get("element_degree") != degree:
            errors.append(f"{rule} point count/degree is invalid")
        for field in (
            "elements",
            "degrees_of_freedom",
            "free_degrees_of_freedom",
            "full_residual_l2_norm",
            "free_residual_l2_norm",
            "full_hessian_action_l2_norm",
            "free_hessian_action_l2_norm",
            "minimum_normalized_active_branch_margin",
            "minimum_normalized_constitutive_denominator",
            "relative_total_potential_difference_from_last_rule",
        ):
            minimum = 1.0 if field in {
                "elements",
                "degrees_of_freedom",
                "free_degrees_of_freedom",
            } else 0.0
            _number(row.get(field), f"{rule}.{field}", errors, minimum=minimum)
        action = row.get("free_hessian_action_vector_comparison_to_last_rule")
        if not isinstance(action, Mapping):
            errors.append(f"{rule}.free_hessian_action_vector_comparison_to_last_rule is missing")
        else:
            for field in (
                "absolute_l2_difference",
                "relative_l2_difference",
                "absolute_linf_difference",
            ):
                _number(action.get(field), f"{rule}.action.{field}", errors, minimum=0.0)
        fractions = row.get("branch_point_fractions")
        branch_keys = {"elastic", "shear", "left_edge", "right_edge", "apex"}
        if not isinstance(fractions, Mapping) or set(fractions) != branch_keys:
            errors.append(f"{rule}.branch_point_fractions is invalid")
        else:
            values = [
                _number(value, f"{rule}.branch fraction", errors, minimum=0.0, maximum=1.0)
                for value in fractions.values()
            ]
            if all(value is not None for value in values) and not math.isclose(
                sum(float(value) for value in values), 1.0, rel_tol=0.0, abs_tol=1.0e-12
            ):
                errors.append(f"{rule}.branch_point_fractions do not sum to one")
    reference = by_rule.get("tetra_duffy_125point")
    if isinstance(reference, Mapping):
        _number(
            reference.get("relative_total_potential_difference_from_last_rule"),
            "reference energy difference",
            errors,
            minimum=0.0,
            maximum=1.0e-15,
        )
    return errors


def _route_expected_slots(
    contract: Mapping[str, Any], hardware_ids: set[str]
) -> tuple[
    set[tuple[str, str, str, int, str]],
    set[tuple[str, str, str, int, str]],
]:
    configurations = {
        str(row["configuration_id"]): row for row in contract["configurations"]
    }
    slots: set[tuple[str, str, str, int, str]] = set()
    for hardware in hardware_ids & set(contract["expected_scope"]):
        scope = contract["expected_scope"][hardware]
        config_ids = list(scope["configuration_ids"]) + list(
            scope.get("factor_configuration_ids", [])
        )
        for config_id in config_ids:
            ranks = configurations[config_id].get("hardware_ranks", {}).get(
                hardware, scope["ranks"]
            )
            for state in contract["states"]:
                for rank in ranks:
                    for route in contract["route_order"]:
                        slots.add(
                            (
                                hardware,
                                config_id,
                                str(state["state_id"]),
                                int(rank),
                                str(route),
                            )
                        )
    censors = {
        slot
        for slot in slots
        for rule in contract["structural_censors"]
        if slot[0] == rule["hardware_id"]
        and slot[1] == rule["configuration_id"]
        and slot[4] == rule["route"]
    }
    return slots, censors


PREDICTIVE_SELECTOR_TERMINAL = "predictive_selector_admissible"
FINITE_EMPIRICAL_MAP_TERMINAL = "finite_empirical_map_only"


def _factorized_diagnostic_errors(
    factor: Any, contract: Mapping[str, Any]
) -> list[str]:
    """Validate a reportable mechanism diagnostic without gating the selector."""
    if not isinstance(factor, Mapping):
        return ["factorized diagnostic is missing"]
    errors: list[str] = []
    policy = contract["factorized_calibration_policy"]
    if policy.get("required_for_selector_claim") is not False:
        errors.append("factorized diagnostic contract was changed into a selector gate")
    if factor.get("calibration_integrated") is not False:
        errors.append("factorized diagnostic must remain non-integrated")
    if factor.get("selector_use") != policy.get("current_status"):
        errors.append("factorized diagnostic selector-use label differs from contract")
    if factor.get("selector_blockers") != []:
        errors.append("factorized diagnostic must not declare selector blockers")
    if factor.get("required_ranks") != [1, 8, 32]:
        errors.append("factorized diagnostic required_ranks changed")
    blocks = _number(
        factor.get("independent_blocks_per_rank"),
        "factor independent_blocks_per_rank",
        errors,
        minimum=3,
        integer=True,
    )
    if blocks is not None and int(blocks) != int(policy["independent_blocks_per_rank"]):
        errors.append("factorized diagnostic block count differs from contract")
    passed = factor.get("passed")
    failures = factor.get("failures")
    if not isinstance(passed, bool):
        errors.append("factorized diagnostic passed flag must be Boolean")
    if not isinstance(failures, list) or any(
        not isinstance(value, str) or not value for value in failures
    ):
        errors.append("factorized diagnostic failures must be a list of nonempty strings")
        failures = []
    calibration = factor.get("calibration_model")
    calibration_passed = (
        isinstance(calibration, Mapping) and calibration.get("status") == "passed"
    )
    if passed is True and (failures != [] or not calibration_passed):
        errors.append("passed factorized diagnostic is internally inconsistent")
    if passed is False and (not failures or calibration_passed):
        errors.append("failed factorized diagnostic is internally inconsistent")
    return errors


def _negative_route_leakage_errors(
    payload: Mapping[str, Any], model: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
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
        "source_schema",
        "publication_evidence",
        "publication_provenance",
        "run_kind",
        "experiment_commit",
    }
    unexpected_top_level = set(payload) - allowed_top_level
    if unexpected_top_level:
        errors.append(
            "finite-map-only payload contains uncontracted top-level fields: "
            f"{sorted(unexpected_top_level)}"
        )
    allowed_model_keys = {
        "status",
        "selector_claim_admissible",
        "feature_order",
        "training_rows",
        "holdout_rows",
        "preflight_failures",
        "failed_gates",
    }
    if set(model) != allowed_model_keys:
        errors.append(
            "finite-map-only cost_model must contain exactly the nonpredictive "
            f"decision fields {sorted(allowed_model_keys)}"
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
    def leaked_keys(value: Any, *, prefix: str = "") -> list[str]:
        leaked: list[str] = []
        if isinstance(value, Mapping):
            for key, nested in value.items():
                name = str(key)
                path = f"{prefix}.{name}" if prefix else name
                lowered = name.lower()
                if any(token in lowered for token in prohibited_tokens) or lowered in {
                    "best_route",
                    "selected_route",
                    "recommended_route",
                    "route_choice",
                }:
                    leaked.append(path)
                leaked.extend(leaked_keys(nested, prefix=path))
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                leaked.extend(leaked_keys(nested, prefix=f"{prefix}[{index}]"))
        return leaked

    rows = payload.get("empirical_map")
    if isinstance(rows, list):
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            leaked = sorted(set(leaked_keys(row)))
            if leaked:
                errors.append(
                    f"finite-map-only empirical_map[{index}] leaks predictive fields {leaked}"
                )
    return errors


def _route_stopping_binding_errors(
    binding: object, *, prefix: str
) -> list[str]:
    if not isinstance(binding, Mapping):
        return [f"{prefix} must be a hash-bound STOP adjudication object"]
    errors: list[str] = []
    if set(binding) != ROUTE_STOPPING_BINDING_KEYS:
        errors.append(
            f"{prefix} has unexpected or missing fields: "
            f"{sorted(str(key) for key in set(binding) ^ ROUTE_STOPPING_BINDING_KEYS)}"
        )
    if (
        binding.get("schema_id")
        != "fenics-nonlinear-energies.exp-stop-001.final-adjudication"
        or binding.get("schema_version") != 3
    ):
        errors.append(f"{prefix} schema must be final EXP-STOP-001 version 3")
    path = binding.get("path")
    if (
        not isinstance(path, str)
        or path != ROUTE_STOPPING_PUBLICATION_PATH
    ):
        errors.append(
            f"{prefix}.path must be the canonical publication STOP path"
        )
    for key in (
        "sha256",
        "local_analysis_sha256",
        "cluster_archive_checksum_sha256",
        "adjudicator_sha256",
    ):
        if not isinstance(binding.get(key), str) or not HEX64_RE.fullmatch(
            str(binding.get(key))
        ):
            errors.append(f"{prefix}.{key} must be a SHA-256 digest")
    for key in ("computation_source_commit", "adjudicator_source_commit"):
        if not isinstance(binding.get(key), str) or not HEX40_RE.fullmatch(
            str(binding.get(key))
        ):
            errors.append(f"{prefix}.{key} must be a Git commit")
    if (
        binding.get("p4_reference_row_id")
        != "p3d_p4_nonlinear_1em07_cluster"
        or binding.get("p4_reference_status") != "accepted"
    ):
        errors.append(f"{prefix} does not admit the fixed P4 tight reference")
    return errors


def _route_analysis_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    contract_path = REPO_ROOT / "paper/protocols/EXP-ROUTE-001-analysis-contract.json"
    try:
        contract = _read_json(contract_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"cannot read frozen route contract: {exc}"]
    expected_terminal_policy = {
        "selector_claim_requires_all_model_gates": True,
        "selector_admitted": PREDICTIVE_SELECTOR_TERMINAL,
        "otherwise": FINITE_EMPIRICAL_MAP_TERMINAL,
        "never_impute_censored_or_missing_timings": True,
    }
    if contract.get("contract_version") != 3:
        errors.append("route analysis contract_version must be 3")
    if contract.get("terminal_policy") != expected_terminal_policy:
        errors.append("route terminal policy differs from the frozen two-branch contract")
    terminal = payload.get("terminal_decision")
    if terminal not in {PREDICTIVE_SELECTOR_TERMINAL, FINITE_EMPIRICAL_MAP_TERMINAL}:
        errors.append(
            "route terminal_decision must be predictive_selector_admissible or "
            "finite_empirical_map_only"
        )
    predictive_branch = terminal == PREDICTIVE_SELECTOR_TERMINAL
    finite_map_only_branch = terminal == FINITE_EMPIRICAL_MAP_TERMINAL
    if "post_fit_confirmation" in payload:
        errors.append(
            "post_fit_confirmation is uncontracted and cannot enter publication evidence"
        )
    if payload.get("invalid_records") != []:
        errors.append("route invalid_records must be empty")
    endpoint = payload.get("endpoint_analysis")
    if not isinstance(endpoint, Mapping):
        errors.append("mandatory Tier-B endpoint_analysis binding is missing")
    else:
        if set(endpoint) != ROUTE_ENDPOINT_SUMMARY_KEYS:
            errors.append(
                "endpoint_analysis has unexpected or missing fields: "
                f"{sorted(str(key) for key in set(endpoint) ^ ROUTE_ENDPOINT_SUMMARY_KEYS)}"
            )
        if endpoint.get("schema_id") != ROUTE_ENDPOINT_SCHEMA_ID:
            errors.append("endpoint_analysis schema_id is invalid")
        if endpoint.get("schema_version") != 2:
            errors.append("endpoint_analysis schema_version must be 2")
        if endpoint.get("terminal_decision") not in {
            "tier_b_descriptive_timing_only",
            "tier_b_comparative_ranking_admissible",
        }:
            errors.append("endpoint_analysis terminal_decision is not publication-admissible")
        comparative = endpoint.get("comparative_ranking_admissible")
        expected_comparative_terminal = (
            "tier_b_comparative_ranking_admissible"
            if comparative is True
            else "tier_b_descriptive_timing_only"
        )
        if not isinstance(comparative, bool) or endpoint.get(
            "terminal_decision"
        ) != expected_comparative_terminal:
            errors.append(
                "endpoint_analysis terminal decision disagrees with its comparative-ranking flag"
            )
        if endpoint.get("publication_admissible") is not True:
            errors.append("endpoint_analysis publication_admissible must be true")
        required_rows = _number(
            endpoint.get("required_rows"),
            "endpoint_analysis.required_rows",
            errors,
            minimum=30,
            integer=True,
        )
        admitted_rows = _number(
            endpoint.get("admitted_rows"),
            "endpoint_analysis.admitted_rows",
            errors,
            minimum=30,
            integer=True,
        )
        if required_rows != 30 or admitted_rows != required_rows:
            errors.append("endpoint_analysis must admit all 30 required Tier-B rows")
        endpoint_path = endpoint.get("path")
        if endpoint_path != ROUTE_ENDPOINT_PUBLICATION_PATH:
            errors.append("endpoint_analysis.path must be the canonical publication path")
        if not isinstance(endpoint.get("sha256"), str) or not HEX64_RE.fullmatch(
            str(endpoint.get("sha256"))
        ):
            errors.append("endpoint_analysis.sha256 must be a SHA-256 digest")
        policy = endpoint.get("stopping_policy")
        expected_policy = {
            "path": contract["publication_model_input_gates"][
                "tier_b_stopping_policy_path"
            ],
            "sha256": contract["publication_model_input_gates"][
                "tier_b_stopping_policy_sha256"
            ],
        }
        if expected_policy != {
            "path": str(TIER_B_STOPPING_POLICY.relative_to(REPO_ROOT)),
            "sha256": stopping_sha256_file(TIER_B_STOPPING_POLICY),
        }:
            errors.append("frozen route contract has a stale Tier-B stopping policy")
        if policy != expected_policy:
            errors.append("endpoint_analysis stopping-policy binding is stale")
        errors.extend(
            _route_stopping_binding_errors(
                endpoint.get("stopping_adjudication"),
                prefix="endpoint_analysis.stopping_adjudication",
            )
        )
        if endpoint.get("stopping_binding_matches_manifest") is not True:
            errors.append(
                "endpoint_analysis must confirm its pre-submission STOP binding"
            )
    sources = payload.get("sources")
    hardware_ids: set[str] = set()
    if not isinstance(sources, list):
        errors.append("route sources must be a list")
    else:
        for index, source in enumerate(sources):
            if not isinstance(source, Mapping):
                errors.append(f"sources[{index}] must be an object")
                continue
            hardware = str(source.get("hardware_id", ""))
            hardware_ids.add(hardware)
            gate = source.get("publication_provenance_gate")
            if not isinstance(gate, Mapping) or gate.get("eligible") is not True:
                errors.append(f"source {hardware} publication provenance is not eligible")
    if (
        hardware_ids != {"workstation_local", "karolina_cpu"}
        or not isinstance(sources, list)
        or len(sources) != 2
    ):
        errors.append("route sources must cover workstation_local and karolina_cpu exactly")
    expected_slots, censor_slots = _route_expected_slots(contract, hardware_ids)
    rows = payload.get("empirical_map")
    if not isinstance(rows, list):
        return errors + ["route empirical_map must be a list"]
    row_by_slot: dict[
        tuple[str, str, str, int, str], Mapping[str, Any]
    ] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"empirical_map[{index}] must be an object")
            continue
        try:
            slot = (
                str(row["hardware_id"]),
                str(row["configuration_id"]),
                str(row["state_id"]),
                int(row["rank_count"]),
                str(row["route"]),
            )
        except (KeyError, TypeError, ValueError):
            errors.append(f"empirical_map[{index}] has malformed slot identity")
            continue
        if slot in row_by_slot:
            errors.append(f"duplicate empirical route slot {slot}")
        row_by_slot[slot] = row
    if set(row_by_slot) != expected_slots:
        errors.append(
            f"route empirical scope has {len(row_by_slot)} slots; expected exact {len(expected_slots)}"
        )
    equivalence = contract["equivalence_gates"]
    for slot in sorted(expected_slots & set(row_by_slot)):
        row = row_by_slot[slot]
        if slot in censor_slots:
            expected_reason = next(
                str(rule["reason"])
                for rule in contract["structural_censors"]
                if slot[0] == rule["hardware_id"]
                and slot[1] == rule["configuration_id"]
                and slot[4] == rule["route"]
            )
            if (
                row.get("status") != "censored"
                or row.get("publication_model_eligible") is not False
            ):
                errors.append(
                    f"structural censor slot {slot} is not an excluded censor"
                )
            if row.get("reason") != expected_reason or row.get(
                "model_exclusion_reason"
            ) != expected_reason:
                errors.append(f"structural censor slot {slot} reason differs from contract")
            for field in (
                "admitted_wall_time_median_s",
                "paired_block_medians_s",
                "paired_block_repetitions",
                "paired_block_route_positions",
                "model_covariates",
                "action_relative_l2_error",
                "action_relative_l2_errors",
                "action_max_absolute_error",
                "gradient_residual_relative_error",
            ):
                if row.get(field) is not None:
                    errors.append(f"structural censor slot {slot} improperly imputes {field}")
            continue
        if row.get("status") != "admitted":
            errors.append(f"active route slot {slot} is not admitted to the finite map")
        model_eligible = row.get("publication_model_eligible")
        if model_eligible is not True:
            errors.append(
                f"active route slot {slot} is not publication-model eligible under the frozen paired design"
            )
        expected_split = (
            "training"
            if slot[0] == "workstation_local"
            or slot[3] in contract["hardware"]["karolina_cpu"]["training_ranks"]
            else "holdout"
        )
        if row.get("split") != expected_split:
            errors.append(f"route slot {slot} has wrong train/holdout split")
        _number(
            row.get("admitted_wall_time_median_s"),
            f"route slot {slot} timing",
            errors,
            minimum=1.0e-300,
        )
        _number(
            row.get("action_relative_l2_error"),
            f"route slot {slot} action error",
            errors,
            minimum=0.0,
            maximum=float(equivalence["action_relative_l2_max"]),
        )
        action_errors = row.get("action_relative_l2_errors")
        if not isinstance(action_errors, list) or len(action_errors) < 4:
            errors.append(f"route slot {slot} lacks four tangent-action errors")
        else:
            for value in action_errors:
                _number(
                    value,
                    f"route slot {slot} probe error",
                    errors,
                    minimum=0.0,
                    maximum=float(equivalence["action_relative_l2_max"]),
                )
        _number(
            row.get("action_max_absolute_error"),
            f"route slot {slot} max action error",
            errors,
            minimum=0.0,
        )
        _number(
            row.get("gradient_residual_relative_error"),
            f"route slot {slot} gradient error",
            errors,
            minimum=0.0,
            maximum=1.0e-12,
        )
        for field in ("state_sha256", "action_sha256"):
            value = row.get(field)
            if not isinstance(value, str) or not HEX64_RE.fullmatch(value):
                errors.append(f"route slot {slot} {field} is invalid")
        commit = row.get("source_commit")
        if not isinstance(commit, str) or not HEX40_RE.fullmatch(commit):
            errors.append(f"route slot {slot} source_commit is invalid")
        if (
            not isinstance(row.get("model_covariates"), Mapping)
            or not row["model_covariates"]
        ):
            errors.append(f"route slot {slot} model_covariates are missing")

    model = payload.get("cost_model")
    if not isinstance(model, Mapping):
        return errors + ["route cost_model must be an object"]
    policy = contract["cost_model"]
    training_rows = _number(
        model.get("training_rows"),
        "cost_model.training_rows",
        errors,
        minimum=(
            float(policy["minimum_training_rows"]) if predictive_branch else 0.0
        ),
        integer=True,
    )
    holdout_rows = _number(
        model.get("holdout_rows"),
        "cost_model.holdout_rows",
        errors,
        minimum=(float(policy["minimum_holdout_rows"]) if predictive_branch else 0.0),
        integer=True,
    )
    actual_training = sum(
        1
        for row in row_by_slot.values()
        if row.get("publication_model_eligible") is True
        and row.get("split") == "training"
    )
    actual_holdout = sum(
        1
        for row in row_by_slot.values()
        if row.get("publication_model_eligible") is True
        and row.get("split") == "holdout"
    )
    expected_training = sum(
        1
        for slot in expected_slots - censor_slots
        if slot[0] == "workstation_local"
        or slot[3] in contract["hardware"]["karolina_cpu"]["training_ranks"]
    )
    expected_holdout = len(expected_slots - censor_slots) - expected_training
    if actual_training != expected_training or actual_holdout != expected_holdout:
        errors.append(
            "publication-model eligibility must cover every active route slot "
            f"({expected_training} training and {expected_holdout} holdout)"
        )
    if training_rows is not None and int(training_rows) != actual_training:
        errors.append("cost_model.training_rows differs from admitted empirical rows")
    if holdout_rows is not None and int(holdout_rows) != actual_holdout:
        errors.append("cost_model.holdout_rows differs from admitted empirical rows")
    if model.get("feature_order") != policy["features_in_order"]:
        errors.append("cost_model.feature_order differs from frozen contract")
    gate_keys = {
        "median_absolute_percentage_error",
        "p90_absolute_percentage_error",
        "minimum_resolved_holdout_groups",
        "resolved_ordering_accuracy",
        "distinct_observed_holdout_winners",
    }
    if predictive_branch:
        if (
            model.get("status") != "selection_rule_passed"
            or model.get("selector_claim_admissible") is not True
        ):
            errors.append("route cost model is not selection_rule_passed/admissible")
        model_gates = _exact_keys(
            model.get("gate_results"), gate_keys, "cost_model.gate_results", errors
        )
        if model_gates is not None and any(
            model_gates.get(key) is not True for key in gate_keys
        ):
            errors.append("every cost-model named gate must pass")
        _number(
            model.get("holdout_median_absolute_percentage_error"),
            "holdout median APE",
            errors,
            minimum=0.0,
            maximum=float(policy["median_absolute_percentage_error_max"]),
        )
        _number(
            model.get("holdout_p90_absolute_percentage_error"),
            "holdout p90 APE",
            errors,
            minimum=0.0,
            maximum=float(policy["p90_absolute_percentage_error_max"]),
        )
        _number(
            model.get("resolved_holdout_groups"),
            "resolved_holdout_groups",
            errors,
            minimum=float(policy["minimum_resolved_holdout_groups"]),
            integer=True,
        )
        _number(
            model.get("resolved_ordering_accuracy"),
            "resolved_ordering_accuracy",
            errors,
            minimum=float(policy["resolved_ordering_accuracy_min"]),
            maximum=1.0,
        )
        winners = model.get("distinct_observed_holdout_winners")
        if (
            not isinstance(winners, list)
            or any(not isinstance(value, str) for value in winners)
            or len(set(winners))
            < int(policy["minimum_distinct_observed_holdout_winners"])
        ):
            errors.append("too few distinct observed holdout winners")
        coefficients = model.get("coefficients")
        if not isinstance(coefficients, Mapping) or set(coefficients) != set(
            policy["features_in_order"]
        ):
            errors.append("cost-model coefficient set differs from frozen feature set")
        else:
            for key, value in coefficients.items():
                _number(value, f"coefficient.{key}", errors)
    elif finite_map_only_branch:
        if model.get("selector_claim_admissible") is not False:
            errors.append("finite-map-only result must explicitly reject the selector claim")
        allowed_statuses = {
            "not_fit_invalid_design",
            "fit_gate_failed",
        }
        if model.get("status") not in allowed_statuses:
            errors.append(
                "finite-map-only cost_model has no admissible negative terminal status"
            )
        preflight_failures = model.get("preflight_failures")
        failed_gates = model.get("failed_gates")
        if not isinstance(preflight_failures, list) or any(
            not isinstance(value, str) or not value for value in preflight_failures
        ):
            errors.append("finite-map-only preflight_failures are malformed")
            preflight_failures = []
        if not isinstance(failed_gates, list) or any(
            value not in gate_keys for value in failed_gates
        ) or len(set(failed_gates)) != len(failed_gates):
            errors.append("finite-map-only failed_gates are malformed")
            failed_gates = []
        if model.get("status") == "fit_gate_failed" and (
            preflight_failures != [] or not failed_gates
        ):
            errors.append(
                "fit_gate_failed must contain only a nonempty subset of the five frozen failed_gates"
            )
        if model.get("status") == "not_fit_invalid_design" and (
            preflight_failures != ["rank_deficient_or_ill_conditioned_design"]
            or failed_gates != []
        ):
            errors.append(
                "not_fit_invalid_design must contain only the frozen design-rank failure"
            )
        errors.extend(_negative_route_leakage_errors(payload, model))

    errors.extend(
        _factorized_diagnostic_errors(
            payload.get("factorized_microbenchmark_gate"), contract
        )
    )
    return errors


def _semantic_gate_errors(spec: EvidenceSpec, payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = _source_schema_errors(spec, payload)
    status_value = (
        payload.get("terminal_decision") if spec.family == "route_analysis" else payload.get("status")
    )
    if status_value not in spec.terminal_statuses:
        errors.append(
            f"terminal status {status_value!r} is not one of {list(spec.terminal_statuses)!r}"
        )
    if not _finite_tree(payload):
        errors.append("payload contains a non-finite or non-JSON scientific value")
    if spec.family == "manufactured_scalar":
        errors.extend(_manufactured_scalar_errors(spec, payload))
    elif spec.family == "affine_patch":
        errors.extend(_affine_patch_errors(payload))
    elif spec.family == "hyperelastic_nonaffine":
        errors.extend(_hyperelastic_nonaffine_errors(payload))
    elif spec.family == "derivative":
        errors.extend(_derivative_errors(spec, payload))
    elif spec.family == "material_point":
        errors.extend(_material_point_errors(payload))
    elif spec.family == "distribution":
        errors.extend(_distribution_errors(payload))
    elif spec.family == "quadrature":
        errors.extend(_quadrature_errors(spec, payload))
    elif spec.family == "route_analysis":
        errors.extend(_route_analysis_errors(payload))
    return errors


def _git_declarations(
    payload: Mapping[str, Any],
    companion: Mapping[str, Any],
    records: Sequence[tuple[Path, Mapping[str, Any]]],
) -> list[tuple[str, str | None, bool | None]]:
    declarations: list[tuple[str, str | None, bool | None]] = []

    def add(label: str, value: Any, *, dirty_key: bool = False) -> None:
        if not isinstance(value, Mapping):
            return
        commit = value.get("commit", value.get("git_commit"))
        if "worktree_clean" in value:
            clean = value.get("worktree_clean")
        elif "git_clean" in value:
            clean = value.get("git_clean")
        elif "dirty" in value:
            clean = not value.get("dirty") if isinstance(value.get("dirty"), bool) else None
        elif "git_dirty" in value:
            clean = not value.get("git_dirty") if isinstance(value.get("git_dirty"), bool) else None
        else:
            clean = None
        if dirty_key and "dirty" in value and isinstance(value.get("dirty"), bool):
            clean = not bool(value["dirty"])
        if commit is not None or clean is not None:
            declarations.append((label, None if commit is None else str(commit).lower(), clean if isinstance(clean, bool) else None))

    provenance = payload.get("provenance")
    add("payload.provenance", provenance)
    add("payload.provenance.git", _nested(payload, "provenance", "git"), dirty_key=True)
    publication = payload.get("publication_provenance")
    if isinstance(publication, Mapping):
        add(
            "payload.publication_provenance",
            {
                "git_commit": publication.get("experiment_commit"),
                "git_clean": publication.get("git_clean"),
            },
        )
        add(
            "payload.publication_provenance.git",
            publication.get("git"),
            dirty_key=True,
        )
    add("companion", companion)
    add("companion.preflight", companion.get("preflight"))
    add("companion.provenance", companion.get("provenance"))
    add("companion.provenance.git", _nested(companion, "provenance", "git"), dirty_key=True)
    for path, record in records:
        add(f"run-record:{path.name}", record.get("provenance"))
    return declarations


def _command_sources(
    payload: Mapping[str, Any],
    companion: Mapping[str, Any],
    records: Sequence[tuple[Path, Mapping[str, Any]]],
) -> list[str]:
    values = [
        _nested(payload, "provenance", "command"),
        _nested(payload, "provenance", "command_argv"),
        _nested(payload, "publication_provenance", "command_argv"),
        companion.get("command"),
        companion.get("command_template"),
        companion.get("commands"),
    ]
    values.extend(_nested(record, "provenance", "command_argv") for _path, record in records)
    return [json.dumps(value, sort_keys=True) if not isinstance(value, str) else value for value in values if value]


def _environment_sources(
    payload: Mapping[str, Any],
    companion: Mapping[str, Any],
    records: Sequence[tuple[Path, Mapping[str, Any]]],
) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = []
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        environment = provenance.get("environment")
        if isinstance(environment, Mapping) and environment:
            sources.append(environment)
        versions = {
            key: provenance[key]
            for key in ("python", "platform", "jax", "numpy", "scipy", "jax_backend", "jax_enable_x64")
            if key in provenance and provenance[key] not in (None, "")
        }
        if versions:
            sources.append(versions)
    publication = payload.get("publication_provenance")
    if isinstance(publication, Mapping):
        environment = publication.get("environment")
        if isinstance(environment, Mapping) and environment:
            sources.append(environment)
    if isinstance(companion.get("environment"), Mapping) and companion["environment"]:
        sources.append(companion["environment"])
    for _path, record in records:
        if isinstance(record.get("environment"), Mapping) and record["environment"]:
            sources.append(record["environment"])
    return sources


def _path_matches(candidate: str, expected: Path) -> bool:
    path = Path(candidate)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and path.as_posix() == expected.as_posix()
    )


def _producer_hash_declarations(
    spec: EvidenceSpec,
    payload: Mapping[str, Any],
    companion: Mapping[str, Any],
    records: Sequence[tuple[Path, Mapping[str, Any]]],
) -> list[tuple[str, str]]:
    declarations: list[tuple[str, str]] = []
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        for field in ("runner_sha256", "analysis_script_sha256", "producer_sha256"):
            value = provenance.get(field)
            if isinstance(value, str):
                declarations.append((f"payload.provenance.{field}", value.lower()))
    publication_producer = _nested(payload, "publication_provenance", "producer")
    if isinstance(publication_producer, Mapping) and _path_matches(
        str(publication_producer.get("path", "")), spec.producer_path
    ) and isinstance(publication_producer.get("sha256"), str):
        declarations.append(
            (
                "payload.publication_provenance.producer.sha256",
                str(publication_producer["sha256"]).lower(),
            )
        )
    maps: list[tuple[str, Any]] = [
        ("companion.code_hashes", companion.get("code_hashes")),
        ("companion.code.entry_points", _nested(companion, "code", "entry_points")),
        ("companion.output_hashes", companion.get("output_hashes")),
    ]
    for path, record in records:
        maps.append((f"run-record:{path.name}.provenance.code_hashes", _nested(record, "provenance", "code_hashes")))
    for label, mapping in maps:
        if not isinstance(mapping, Mapping):
            continue
        for path_value, digest in mapping.items():
            if _path_matches(str(path_value), spec.producer_path) and isinstance(digest, str):
                declarations.append((f"{label}[{path_value}]", digest.lower()))
    return declarations


def _resolve_declared_path(
    raw_path: str,
    *,
    section: str,
    repo_root: Path,
    evidence_root: Path,
    companion_path: Path,
    spec: EvidenceSpec,
) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != raw_path:
        raise ValueError("path must be canonical, relative, and contain no '..'")
    if section.startswith(
        ("code", "configuration_hashes", "inputs.mesh", "input_hashes")
    ):
        candidates = [repo_root / path, evidence_root / path, companion_path.parent / path]
    else:
        candidates = [companion_path.parent / path, evidence_root / path, repo_root / path]
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate.resolve()
            if not _is_contained(resolved, repo_root):
                raise ValueError("resolved path escapes repository")
            if section.startswith(
                ("output", "outputs", "raw_output", "execution_receipts", "artifacts")
            ) and not _is_contained(resolved, evidence_root):
                raise ValueError("output/artifact path escapes evidence_root")
            return resolved
    return candidates[0].resolve()


def _declared_hashes(companion: Mapping[str, Any]) -> Iterable[tuple[str, str, Any]]:
    mapping_paths = (
        ("code_hashes", companion.get("code_hashes")),
        ("code.entry_points", _nested(companion, "code", "entry_points")),
        ("configuration_hashes", companion.get("configuration_hashes")),
        ("input_hashes", companion.get("input_hashes")),
        ("inputs.mesh_or_data_hashes", _nested(companion, "inputs", "mesh_or_data_hashes")),
        ("raw_output_hashes", companion.get("raw_output_hashes")),
        ("execution_receipts", companion.get("execution_receipts")),
        ("output_hashes", companion.get("output_hashes")),
        ("outputs", companion.get("outputs")),
    )
    for section, mapping in mapping_paths:
        if not isinstance(mapping, Mapping):
            continue
        for raw_path, digest in mapping.items():
            yield section, str(raw_path), digest
    artifacts = companion.get("artifacts")
    if isinstance(artifacts, list):
        for index, row in enumerate(artifacts):
            if not isinstance(row, Mapping):
                continue
            path = row.get("path")
            digest = row.get("sha256")
            if isinstance(path, str):
                yield f"artifacts[{index}]", path, digest


def _hash_contract_errors(
    spec: EvidenceSpec,
    *,
    input_path: Path,
    companion_path: Path,
    companion: Mapping[str, Any],
    repo_root: Path,
    evidence_root: Path,
) -> tuple[list[str], int, bool]:
    errors: list[str] = []
    checked = 0
    target_declared = False
    input_resolved = input_path.resolve()
    for section, raw_path, expected in _declared_hashes(companion):
        if not isinstance(expected, str) or not HEX64_RE.fullmatch(expected.lower()):
            errors.append(f"{section} has malformed SHA-256 for {raw_path}")
            continue
        expected = expected.lower()
        try:
            resolved = _resolve_declared_path(
                raw_path,
                section=section,
                repo_root=repo_root,
                evidence_root=evidence_root,
                companion_path=companion_path,
                spec=spec,
            ).resolve()
        except ValueError as exc:
            errors.append(f"{section} invalid path {raw_path}: {exc}")
            continue
        if resolved == input_resolved:
            target_declared = True
            allowed_target_paths = {spec.relative_path.as_posix()}
            try:
                allowed_target_paths.add(
                    input_resolved.relative_to(repo_root.resolve()).as_posix()
                )
            except ValueError:
                pass
            if raw_path not in allowed_target_paths:
                errors.append(
                    f"{section} binds the table input through a non-canonical alias {raw_path}"
                )
        if not resolved.is_file():
            errors.append(f"{section} declares missing file {raw_path}")
            continue
        checked += 1
        actual = sha256_file(resolved)
        if actual != expected:
            errors.append(f"{section} SHA-256 mismatch for {raw_path}")
    if not target_declared:
        errors.append("companion manifest does not bind this table input by SHA-256")
    return errors, checked, target_declared


def _payload_hash_errors(
    spec: EvidenceSpec,
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
    evidence_root: Path,
) -> list[str]:
    """Verify payload-level publication and analyzer evidence recursively."""
    errors: list[str] = []

    def verify(path_value: Any, digest_value: Any, label: str, *, evidence_only: bool) -> None:
        if not isinstance(path_value, str):
            errors.append(f"{label}.path must be a string")
            return
        relative = Path(path_value)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != path_value:
            errors.append(f"{label}.path must be canonical and relative")
            return
        if not isinstance(digest_value, str) or not HEX64_RE.fullmatch(digest_value.lower()):
            errors.append(f"{label}.sha256 is malformed")
            return
        candidates = (
            [
                evidence_root / spec.relative_path.parent / relative,
                evidence_root / relative,
            ]
            if evidence_only
            else [repo_root / relative, evidence_root / relative]
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        allowed_root = evidence_root if evidence_only else repo_root
        if not _is_contained(path, allowed_root):
            errors.append(f"{label}.path escapes its allowed root")
        elif not path.is_file():
            errors.append(f"{label}.path is missing")
        elif sha256_file(path) != digest_value.lower():
            errors.append(f"{label}.sha256 mismatch")

    publication = payload.get("publication_provenance")
    if isinstance(publication, Mapping):
        for map_name in ("configuration_hashes", "input_hashes"):
            mapping = publication.get(map_name)
            allow_empty = (
                map_name == "input_hashes"
                and publication.get("input_policy") == "no_external_file_inputs"
            )
            if not isinstance(mapping, Mapping) or (not mapping and not allow_empty):
                errors.append(
                    f"publication_provenance.{map_name} must be a path/hash map"
                    + (" or declare no_external_file_inputs" if map_name == "input_hashes" else "")
                )
                continue
            for path, digest in mapping.items():
                verify(path, digest, f"publication_provenance.{map_name}[{path}]", evidence_only=False)
        for object_name in ("raw_output", "execution_receipt"):
            row = publication.get(object_name)
            if not isinstance(row, Mapping):
                errors.append(f"publication_provenance.{object_name} is missing")
            else:
                verify(
                    row.get("path"),
                    row.get("sha256"),
                    f"publication_provenance.{object_name}",
                    evidence_only=True,
                )
        receipt_row = publication.get("execution_receipt")
        if isinstance(receipt_row, Mapping) and isinstance(receipt_row.get("path"), str):
            receipt_relative = Path(str(receipt_row["path"]))
            receipt_path = evidence_root / receipt_relative
            if receipt_path.is_file() and _is_contained(receipt_path, evidence_root):
                try:
                    receipt = _read_json(receipt_path)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"execution receipt cannot be parsed: {exc}")
                else:
                    if (
                        receipt.get("schema_id")
                        != "fenics-nonlinear-energies.revision-publication-execution-receipt"
                        or receipt.get("schema_version") != 1
                        or receipt.get("status") != "completed"
                        or receipt.get("experiment_commit")
                        != publication.get("experiment_commit")
                    ):
                        errors.append("execution receipt schema/status/commit is invalid")
                    for phase in ("preflight", "postflight"):
                        row = receipt.get(phase)
                        if not isinstance(row, Mapping) or row.get("git_commit") != publication.get(
                            "experiment_commit"
                        ) or row.get("git_clean") is not True:
                            errors.append(f"execution receipt {phase} is not clean/commit-consistent")
                    producer = receipt.get("producer")
                    expected_producer = publication.get("producer")
                    if producer != expected_producer:
                        errors.append("execution receipt producer differs from publication provenance")
                    declared_references = publication.get("referenced_artifact_hashes")
                    receipt_references = receipt.get("referenced_artifact_hashes")
                    raw_outputs = receipt.get("raw_output_hashes")
                    if not isinstance(declared_references, Mapping):
                        errors.append(
                            "publication provenance referenced_artifact_hashes is missing"
                        )
                    elif receipt_references != declared_references:
                        errors.append(
                            "execution receipt referenced artifacts differ from publication provenance"
                        )
                    elif not isinstance(raw_outputs, Mapping) or any(
                        raw_outputs.get(path) != digest
                        for path, digest in declared_references.items()
                    ):
                        errors.append(
                            "execution receipt raw outputs do not bind every referenced artifact"
                        )
                    if receipt.get("artifact_validation_errors") != []:
                        errors.append("execution receipt records artifact validation errors")
                    for map_name, evidence_only in (
                        ("configuration_hashes", False),
                        ("input_hashes", False),
                        ("raw_output_hashes", True),
                        ("referenced_artifact_hashes", True),
                        ("logs", True),
                    ):
                        mapping = receipt.get(map_name)
                        if not isinstance(mapping, Mapping):
                            errors.append(f"execution receipt {map_name} is missing")
                            continue
                        for path, digest in mapping.items():
                            verify(
                                path,
                                digest,
                                f"execution_receipt.{map_name}[{path}]",
                                evidence_only=evidence_only,
                            )
                    fingerprint = receipt.get("receipt_fingerprint_sha256")
                    fingerprint_payload = dict(receipt)
                    fingerprint_payload.pop("receipt_fingerprint_sha256", None)
                    if not isinstance(fingerprint, str) or fingerprint != _json_sha256(
                        fingerprint_payload
                    ):
                        errors.append("execution receipt fingerprint is stale")
    else:
        errors.append("publication_provenance object is required")

    if spec.family == "quadrature":
        errors.extend(
            _quadrature_referenced_artifact_errors(
                spec,
                payload,
                evidence_root=evidence_root,
            )
        )

    if spec.family == "route_analysis":
        entries = _nested(payload, "provenance", "input_files")
        stopping_role_entries: list[Mapping[str, Any]] = []
        if not isinstance(entries, list) or not entries:
            errors.append("route provenance.input_files must be a nonempty evidence inventory")
        else:
            roles: set[str] = set()
            for index, row in enumerate(entries):
                if not isinstance(row, Mapping):
                    errors.append(f"route provenance.input_files[{index}] must be an object")
                    continue
                role = str(row.get("role", ""))
                roles.add(role)
                if role == "tier_b_stopping_adjudication":
                    stopping_role_entries.append(row)
                verify(
                    row.get("path"),
                    row.get("sha256"),
                    f"route provenance.input_files[{index}]",
                    evidence_only=True,
                )
            required_roles = {
                "route_campaign_master",
                "route_tranche_manifest",
                "route_submission_ledger",
                "route_release_authorization",
                "tier_b_stopping_adjudication",
            }
            missing_roles = sorted(required_roles - roles)
            if missing_roles:
                errors.append("route evidence inventory lacks roles: " + ", ".join(missing_roles))
        endpoint = payload.get("endpoint_analysis")
        if isinstance(endpoint, Mapping):
            verify(
                endpoint.get("path"),
                endpoint.get("sha256"),
                "endpoint_analysis",
                evidence_only=True,
            )
            endpoint_relative = Path(str(endpoint.get("path", "")))
            endpoint_candidates = [
                evidence_root / spec.relative_path.parent / endpoint_relative,
                evidence_root / endpoint_relative,
            ]
            endpoint_path = next(
                (candidate for candidate in endpoint_candidates if candidate.is_file()),
                endpoint_candidates[0],
            )
            if endpoint_path.is_file() and _is_contained(endpoint_path, evidence_root):
                try:
                    endpoint_payload = _read_json(endpoint_path)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"endpoint analysis JSON cannot be parsed: {exc}")
                else:
                    schema = endpoint_payload.get("schema")
                    if not isinstance(schema, Mapping) or schema.get("id") != (
                        "fenics-nonlinear-energies.exp-route-001.tier-b-endpoints"
                    ) or schema.get("version") != 2:
                        errors.append("endpoint analysis native schema is invalid")
                    if endpoint_payload.get("experiment_id") != "EXP-ROUTE-001":
                        errors.append("endpoint analysis experiment_id is invalid")
                    comparative = endpoint_payload.get(
                        "comparative_ranking_admissible"
                    )
                    expected_endpoint_terminal = (
                        "tier_b_comparative_ranking_admissible"
                        if comparative is True
                        else "tier_b_descriptive_timing_only"
                    )
                    if (
                        endpoint_payload.get("terminal_decision")
                        != endpoint.get("terminal_decision")
                        or not isinstance(comparative, bool)
                        or endpoint_payload.get("terminal_decision")
                        != expected_endpoint_terminal
                        or endpoint.get("comparative_ranking_admissible")
                        is not comparative
                        or endpoint_payload.get("publication_admissible") is not True
                        or endpoint_payload.get("endpoint_correct_timing_admissible") is not True
                        or endpoint_payload.get("required_rows") != 30
                        or endpoint_payload.get("admitted_rows") != 30
                        or endpoint_payload.get("matrix_policy_violations") != []
                        or endpoint_payload.get("coverage_and_campaign_failure_reasons") != []
                    ):
                        errors.append("endpoint analysis terminal/coverage gates disagree with binding")
                    contract = _read_json(
                        REPO_ROOT
                        / "paper/protocols/EXP-ROUTE-001-analysis-contract.json"
                    )
                    expected_policy = {
                        "path": contract["publication_model_input_gates"][
                            "tier_b_stopping_policy_path"
                        ],
                        "sha256": contract["publication_model_input_gates"][
                            "tier_b_stopping_policy_sha256"
                        ],
                    }
                    if endpoint_payload.get("stopping_policy") != expected_policy:
                        errors.append("endpoint native stopping-policy binding is stale")
                    if endpoint_payload.get("stopping_binding_matches_manifest") is not True:
                        errors.append(
                            "endpoint native evidence does not confirm its submission STOP binding"
                        )
                    native_stop = endpoint_payload.get("stopping_adjudication")
                    declared_stop = endpoint.get("stopping_adjudication")
                    errors.extend(
                        _route_stopping_binding_errors(
                            native_stop,
                            prefix="endpoint native stopping_adjudication",
                        )
                    )
                    if isinstance(native_stop, Mapping) and isinstance(
                        declared_stop, Mapping
                    ):
                        if any(
                            native_stop.get(key) != declared_stop.get(key)
                            for key in ROUTE_STOPPING_BINDING_KEYS
                        ):
                            errors.append(
                                "endpoint native and route-summary STOP bindings disagree"
                            )
                    blocks = endpoint_payload.get("blocks")
                    if not isinstance(blocks, list) or len(blocks) != 30 or any(
                        not isinstance(block, Mapping)
                        or block.get("status") != "timing_admitted"
                        for block in blocks
                    ):
                        errors.append("endpoint analysis does not contain 30 timing-admitted blocks")
                    structural = endpoint_payload.get("structural_censors")
                    if not isinstance(structural, list) or len(structural) != 2 or any(
                        not isinstance(row, Mapping)
                        or row.get("status") != "censored"
                        or row.get("reason")
                        != "prespecified_not_attempted_memory_risk_no_threshold_claim"
                        or row.get("route") != "colored_sfd"
                        or row.get("timing_exposed") is not False
                        or row.get("admitted_collective_max_wall_time_s") is not None
                        for row in structural
                    ):
                        errors.append("endpoint analysis structural censors are invalid")
            declared_stop = endpoint.get("stopping_adjudication")
            if isinstance(declared_stop, Mapping):
                if not stopping_role_entries:
                    errors.append(
                        "route evidence inventory lacks a STOP artifact bound to the "
                        "endpoint summary"
                    )
                elif any(
                    row.get("sha256") != declared_stop.get("sha256")
                    for row in stopping_role_entries
                ):
                    errors.append(
                        "route STOP provenance digest disagrees with endpoint_analysis"
                    )
                verify(
                    declared_stop.get("path"),
                    declared_stop.get("sha256"),
                    "endpoint_analysis.stopping_adjudication",
                    evidence_only=True,
                )
                stop_relative = Path(str(declared_stop.get("path", "")))
                stop_candidates = [
                    evidence_root / spec.relative_path.parent / stop_relative,
                    evidence_root / stop_relative,
                ]
                stop_path = next(
                    (candidate for candidate in stop_candidates if candidate.is_file()),
                    stop_candidates[0],
                )
                if stop_path.is_file() and _is_contained(stop_path, evidence_root):
                    try:
                        validated_stop = validate_stop_adjudication(stop_path)
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        errors.append(f"STOP adjudication cannot be validated: {exc}")
                    else:
                        for key, value in validated_stop.items():
                            if key != "path" and declared_stop.get(key) != value:
                                errors.append(
                                    "route-summary STOP binding disagrees with the "
                                    f"validated artifact at {key}"
                                )
                        if declared_stop.get("sha256") != sha256_file(stop_path):
                            errors.append("route-summary STOP artifact hash is stale")
    return errors


def _audit_run_records(
    spec: EvidenceSpec,
    *,
    repo_root: Path,
    evidence_root: Path,
) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    def resolve_confined_file(raw_path: Any) -> Path | None:
        if not isinstance(raw_path, str):
            return None
        declared = Path(raw_path)
        if (
            declared.is_absolute()
            or ".." in declared.parts
            or declared.as_posix() != raw_path
        ):
            return None
        for root in (repo_root, evidence_root):
            candidate = root / declared
            if _is_contained(candidate, root) and candidate.is_file():
                return candidate.resolve()
        return None

    records: list[tuple[Path, dict[str, Any]]] = []
    errors: list[str] = []
    for relative in spec.run_records:
        path = evidence_root / relative
        if not _is_contained(path, evidence_root):
            errors.append(f"run record path escapes evidence_root: {relative.as_posix()}")
            continue
        if not path.is_file():
            errors.append(f"required publication run record is missing: {relative.as_posix()}")
            continue
        try:
            record = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"cannot parse run record {relative.as_posix()}: {exc}")
            continue
        records.append((path, record))
        try:
            validate_run_record(record, require_publication_ready=True)
        except RunRecordValidationError as exc:
            errors.extend(f"{relative.as_posix()}: {item}" for item in exc.errors)
        provenance = record.get("provenance")
        if isinstance(provenance, Mapping):
            for map_name in ("code_hashes", "input_hashes"):
                mapping = provenance.get(map_name)
                if not isinstance(mapping, Mapping):
                    errors.append(f"{relative.as_posix()}: provenance.{map_name} is missing")
                    continue
                for raw_path, digest in mapping.items():
                    resolved = resolve_confined_file(raw_path)
                    if resolved is None:
                        errors.append(
                            f"{relative.as_posix()}: provenance.{map_name} file is missing/escaped: {raw_path}"
                        )
                        continue
                    if not isinstance(digest, str) or not HEX64_RE.fullmatch(
                        digest.lower()
                    ):
                        errors.append(
                            f"{relative.as_posix()}: provenance.{map_name} digest is malformed: {raw_path}"
                        )
                    elif sha256_file(resolved) != digest.lower():
                        errors.append(
                            f"{relative.as_posix()}: provenance.{map_name} SHA-256 mismatch: {raw_path}"
                        )
        artifacts = record.get("artifacts")
        if isinstance(artifacts, Mapping):
            for category, values in artifacts.items():
                if not isinstance(values, list):
                    continue
                for raw_path in values:
                    if resolve_confined_file(raw_path) is None:
                        errors.append(
                            f"{relative.as_posix()}: artifacts.{category} path is missing/unsafe: {raw_path}"
                        )
    return records, errors


def _audit_fingerprint_payload(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_commit": audit["experiment_commit"],
        "tools": audit["tools"],
        "inputs": audit["inputs"],
        "configured_input_count": audit["configured_input_count"],
    }


def audit_evidence(
    evidence_root: Path,
    *,
    repo_root: Path = REPO_ROOT,
    specs: Sequence[EvidenceSpec] | None = None,
    git_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a complete, non-mutating admission audit."""
    repo_root = repo_root.resolve()
    evidence_root = evidence_root.resolve()
    specs = EVIDENCE_SPECS if specs is None else tuple(specs)
    git = dict(git_metadata) if git_metadata is not None else _git_metadata(repo_root)
    git["commit"] = str(git.get("commit", "")).lower()
    git["worktree_clean"] = git.get("worktree_clean") is True
    admission_tool = Path(__file__).resolve()
    table_generator = repo_root / TABLE_GENERATOR
    tools = {
        "admission_tool": {
            "path": _display_path(admission_tool, repo_root=repo_root),
            "sha256": sha256_file(admission_tool),
        },
        "table_generator": {
            "path": TABLE_GENERATOR.as_posix(),
            "sha256": sha256_file(table_generator) if table_generator.is_file() else None,
        },
    }
    global_checks = [
        _check(
            "evidence-root-containment",
            _is_contained(evidence_root, repo_root),
            "evidence_root resolves inside repository"
            if _is_contained(evidence_root, repo_root)
            else "evidence_root resolves outside repository",
        ),
        _check(
            "configured-input-count",
            len(specs) == 14 and len({spec.key for spec in specs}) == 14,
            f"configured {len(specs)} inputs with {len({spec.key for spec in specs})} unique keys; exactly 14 are required",
        ),
        _check(
            "current-git-commit",
            bool(HEX40_RE.fullmatch(git["commit"])),
            f"current HEAD is {git['commit'] or '<unavailable>'}",
        ),
        _check(
            "current-worktree-clean",
            git["worktree_clean"],
            "current worktree is clean" if git["worktree_clean"] else "current worktree is dirty",
        ),
        _check(
            "table-generator-identity",
            table_generator.is_file() and bool(tools["table_generator"]["sha256"]),
            f"table generator: {_display_path(table_generator, repo_root=repo_root)}",
        ),
    ]

    rows: dict[str, Any] = {}
    for spec in specs:
        input_path = evidence_root / spec.relative_path
        companion_path = evidence_root / spec.companion_manifest
        checks: list[dict[str, Any]] = []
        payload: dict[str, Any] = {}
        companion: dict[str, Any] = {}
        row_source_commit: str | None = None

        input_contained = _is_contained(input_path, evidence_root)
        checks.append(
            _check(
                "input-path-containment",
                input_contained,
                "input resolves inside evidence_root"
                if input_contained
                else "input path or symlink escapes evidence_root",
            )
        )
        exists = input_contained and input_path.is_file()
        checks.append(
            _check("input-exists", exists, f"source path: {spec.relative_path.as_posix()}")
        )
        parsed = False
        if exists:
            try:
                payload = _read_json(input_path)
                parsed = True
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                checks.append(_check("input-json", False, f"cannot parse source JSON: {exc}"))
        if parsed:
            checks.append(_check("input-json", True, "source is a JSON object with finite-value checks pending"))

        companion_contained = _is_contained(companion_path, evidence_root)
        checks.append(
            _check(
                "companion-path-containment",
                companion_contained,
                "companion resolves inside evidence_root"
                if companion_contained
                else "companion path or symlink escapes evidence_root",
            )
        )
        companion_exists = companion_contained and companion_path.is_file()
        checks.append(
            _check(
                "companion-manifest-exists",
                companion_exists,
                f"companion: {spec.companion_manifest.as_posix()}",
            )
        )
        companion_parsed = False
        if companion_exists:
            try:
                companion = _read_json(companion_path)
                companion_parsed = True
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                checks.append(_check("companion-manifest-json", False, f"cannot parse companion manifest: {exc}"))
        if companion_parsed:
            checks.append(_check("companion-manifest-json", True, "companion manifest is a JSON object"))

        records, record_errors = _audit_run_records(
            spec,
            repo_root=repo_root,
            evidence_root=evidence_root,
        )
        checks.append(
            _check(
                "publication-run-records",
                not record_errors,
                (
                    f"{len(records)} publication-ready run record(s) validated"
                    if not record_errors
                    else "; ".join(record_errors)
                ),
                applicable=bool(spec.run_records),
            )
        )

        if parsed:
            try:
                semantic_errors = _semantic_gate_errors(spec, payload)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                semantic_errors = [
                    f"semantic validator rejected malformed payload without admission: {exc}"
                ]
            checks.append(
                _check(
                    "terminal-and-scientific-gates",
                    not semantic_errors,
                    "all source-specific terminal/scientific gates pass" if not semantic_errors else "; ".join(semantic_errors),
                )
            )

            if "publication_evidence" in payload:
                payload_publication = payload.get("publication_evidence") is True
                checks.append(
                    _check(
                        "payload-publication-status",
                        payload_publication,
                        f"payload publication_evidence={payload.get('publication_evidence')!r}",
                    )
                )
            else:
                checks.append(
                    _check(
                        "payload-publication-status",
                        True,
                        "payload schema has no publication_evidence field; companion/run-record admission remains mandatory",
                        applicable=False,
                    )
                )

        if companion_parsed:
            companion_schema_ok = (
                companion.get("schema_id")
                == "fenics-nonlinear-energies.revision-publication-companion"
                and companion.get("schema_version") == 1
            )
            checks.append(
                _check(
                    "companion-schema",
                    companion_schema_ok,
                    "companion schema id/version is valid"
                    if companion_schema_ok
                    else "companion schema id/version is missing or invalid",
                )
            )
            companion_publication = companion.get("publication_evidence") is True
            checks.append(
                _check(
                    "companion-publication-status",
                    companion_publication,
                    f"companion publication_evidence={companion.get('publication_evidence')!r}",
                )
            )
            checks.append(
                _check(
                    "companion-run-kind",
                    companion.get("run_kind") == "publication",
                    f"companion run_kind={companion.get('run_kind')!r}",
                )
            )

        if parsed and companion_parsed:
            commands = _command_sources(payload, companion, records)
            checks.append(
                _check(
                    "command-provenance",
                    bool(commands),
                    f"found {len(commands)} non-empty command record(s)" if commands else "no command or command_argv provenance found",
                )
            )
            environments = _environment_sources(payload, companion, records)
            checks.append(
                _check(
                    "environment-provenance",
                    bool(environments),
                    f"found {len(environments)} non-empty environment record(s)" if environments else "no environment/version provenance found",
                )
            )

            declarations = _git_declarations(payload, companion, records)
            git_errors: list[str] = []
            if not declarations:
                git_errors.append("no source Git declaration found")
            declared_commits: set[str] = set()
            for label, commit, clean in declarations:
                if commit is None or not HEX40_RE.fullmatch(commit):
                    git_errors.append(f"{label} has no valid 40-digit Git commit")
                else:
                    declared_commits.add(commit)
                if clean is not True:
                    git_errors.append(f"{label} does not record a clean source worktree")
            if len(declared_commits) != 1:
                git_errors.append("source Git declarations do not name one experiment commit")
            else:
                row_source_commit = next(iter(declared_commits))
            checks.append(
                _check(
                    "clean-consistent-git",
                    not git_errors,
                    f"all source declarations are clean at experiment commit {row_source_commit}"
                    if not git_errors
                    else "; ".join(git_errors),
                )
            )

            producer = repo_root / spec.producer_path
            producer_actual = sha256_file(producer) if producer.is_file() else None
            declared_producers = _producer_hash_declarations(spec, payload, companion, records)
            producer_errors: list[str] = []
            if producer_actual is None:
                producer_errors.append(f"configured producer is missing: {spec.producer_path.as_posix()}")
            if not declared_producers:
                producer_errors.append("no producer/analyzer SHA-256 declaration found")
            for label, declared in declared_producers:
                if not HEX64_RE.fullmatch(declared):
                    producer_errors.append(f"{label} is not a SHA-256 digest")
                elif producer_actual is not None and declared != producer_actual:
                    producer_errors.append(f"{label} does not match the current producer")
            checks.append(
                _check(
                    "producer-identity",
                    not producer_errors,
                    f"producer {spec.producer_path.as_posix()} is hash-bound" if not producer_errors else "; ".join(producer_errors),
                )
            )

            hash_errors, checked_hashes, _target_declared = _hash_contract_errors(
                spec,
                input_path=input_path,
                companion_path=companion_path,
                companion=companion,
                repo_root=repo_root,
                evidence_root=evidence_root,
            )
            checks.append(
                _check(
                    "declared-file-hashes",
                    not hash_errors,
                    f"verified {checked_hashes} companion-declared file hash(es), including the table input" if not hash_errors else "; ".join(hash_errors),
                )
            )
            payload_hash_errors = _payload_hash_errors(
                spec,
                payload,
                repo_root=repo_root,
                evidence_root=evidence_root,
            )
            checks.append(
                _check(
                    "payload-evidence-hashes",
                    not payload_hash_errors,
                    "payload publication/analyzer evidence hashes verified"
                    if not payload_hash_errors
                    else "; ".join(payload_hash_errors),
                )
            )

            if spec.family == "route_analysis":
                contract_value = payload.get("contract_path")
                contract_hash = payload.get("contract_sha256")
                contract_path = Path(str(contract_value))
                if not contract_path.is_absolute():
                    contract_path = repo_root / contract_path
                contract_ok = (
                    contract_path.is_file()
                    and isinstance(contract_hash, str)
                    and sha256_file(contract_path) == contract_hash.lower()
                )
                checks.append(
                    _check(
                        "analysis-contract-identity",
                        contract_ok,
                        "route analysis contract is hash-bound" if contract_ok else "route analysis contract path/hash is missing or stale",
                    )
                )

        row_admitted = all(check["passed"] for check in checks if check["applicable"])
        rows[spec.key] = {
            "path": spec.relative_path.as_posix(),
            "sha256": sha256_file(input_path) if input_path.is_file() else None,
            "producer": {
                "path": spec.producer_path.as_posix(),
                "sha256": sha256_file(repo_root / spec.producer_path)
                if (repo_root / spec.producer_path).is_file()
                else None,
            },
            "companion_manifest": {
                "path": spec.companion_manifest.as_posix(),
                "sha256": sha256_file(companion_path) if companion_path.is_file() else None,
            },
            "run_records": [
                {
                    "path": relative.as_posix(),
                    "sha256": sha256_file(evidence_root / relative)
                    if (evidence_root / relative).is_file()
                    else None,
                }
                for relative in spec.run_records
            ],
            "experiment_commit": row_source_commit,
            "admitted": row_admitted,
            "checks": checks,
            "blockers": [
                check["detail"] for check in checks if check["applicable"] and not check["passed"]
            ],
        }

    experiment_commits = {
        str(row["experiment_commit"])
        for row in rows.values()
        if isinstance(row.get("experiment_commit"), str)
    }
    common_experiment_commit = (
        next(iter(experiment_commits)) if len(experiment_commits) == 1 else None
    )
    all_rows_share_commit = bool(
        common_experiment_commit
        and all(
            row.get("experiment_commit") == common_experiment_commit
            for row in rows.values()
        )
    )
    global_checks.append(
        _check(
            "common-experiment-commit",
            all_rows_share_commit,
            f"all 14 inputs name experiment commit {common_experiment_commit}"
            if all_rows_share_commit
            else "the 14 inputs do not name one common experiment commit",
        )
    )
    experiment_is_ancestor = bool(
        common_experiment_commit
        and HEX40_RE.fullmatch(common_experiment_commit)
        and HEX40_RE.fullmatch(git["commit"])
        and _git_is_ancestor(repo_root, common_experiment_commit, git["commit"])
    )
    global_checks.append(
        _check(
            "experiment-commit-ancestry",
            experiment_is_ancestor,
            "immutable experiment commit is an ancestor of release HEAD"
            if experiment_is_ancestor
            else "experiment commit is not an ancestor of release HEAD",
        )
    )
    eligible = all(check["passed"] for check in global_checks) and all(
        row["admitted"] for row in rows.values()
    )
    audit: dict[str, Any] = {
        "schema_id": AUDIT_SCHEMA_ID,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "publication_evidence": False,
        "status": "eligible_for_manifest_creation" if eligible else "publication_admission_blocked",
        "evidence_root": _display_path(evidence_root, repo_root=repo_root),
        "configured_input_count": len(specs),
        "admitted_input_count": sum(1 for row in rows.values() if row["admitted"]),
        "git": git,
        "experiment_commit": common_experiment_commit,
        "tools": tools,
        "global_checks": global_checks,
        "inputs": rows,
        "eligible": eligible,
    }
    audit["audit_sha256"] = _json_sha256(_audit_fingerprint_payload(audit))
    return audit


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, payload)


def build_publication_manifest(audit: Mapping[str, Any]) -> dict[str, Any]:
    if audit.get("eligible") is not True:
        raise ValueError("publication evidence audit is blocked; no manifest may be created")
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "publication_evidence": True,
        "status": ADMITTED_STATUS,
        "created_at_utc": _utc_now(),
        "evidence_root": audit["evidence_root"],
        "git_commit": audit["experiment_commit"],
        "experiment_commit": audit["experiment_commit"],
        "admission_head": audit["git"]["commit"],
        "worktree_clean": True,
        "git": {
            "experiment_commit": audit["experiment_commit"],
            "admission_head": audit["git"]["commit"],
            "worktree_clean": True,
        },
        "admission_tool": audit["tools"]["admission_tool"],
        "table_generator": audit["tools"]["table_generator"],
        "audit_schema_id": audit["schema_id"],
        "audit_schema_version": audit["schema_version"],
        "audit_sha256": audit["audit_sha256"],
        "configured_input_count": audit["configured_input_count"],
        "inputs": audit["inputs"],
        "interpretation": (
            "Every configured table input passed independent publication-status, clean-commit, "
            "command/environment, terminal/scientific-gate, run-record, file-hash, and producer-identity checks."
        ),
    }


def validate_publication_source_manifest(
    manifest_path: Path,
    *,
    evidence_root: Path,
    repo_root: Path = REPO_ROOT,
    expected_inputs: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Deeply revalidate an admitted source manifest and all 14 source files."""
    manifest_path = manifest_path.resolve()
    evidence_root = evidence_root.resolve()
    repo_root = repo_root.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Publication evidence source manifest does not exist: {manifest_path}")
    manifest = _read_json(manifest_path)
    errors: list[str] = []
    if manifest.get("schema_id") != SCHEMA_ID:
        errors.append(f"schema_id must be {SCHEMA_ID}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if manifest.get("publication_evidence") is not True:
        errors.append("publication_evidence must be true")
    if manifest.get("status") != ADMITTED_STATUS:
        errors.append(f"status must be {ADMITTED_STATUS}")
    if manifest.get("worktree_clean") is not True:
        errors.append("worktree_clean must be true")
    recorded_root = str(manifest.get("evidence_root", ""))
    expected_root = _display_path(evidence_root, repo_root=repo_root)
    if recorded_root != expected_root:
        errors.append(f"evidence_root must equal {expected_root}")

    audit = audit_evidence(evidence_root, repo_root=repo_root)
    if audit.get("eligible") is not True:
        errors.append("fresh independent source audit is blocked")
        for check in audit["global_checks"]:
            if check["applicable"] and not check["passed"]:
                errors.append(f"global:{check['id']}: {check['detail']}")
        for key, row in audit["inputs"].items():
            for blocker in row["blockers"]:
                errors.append(f"input {key}: {blocker}")

    if str(manifest.get("experiment_commit", "")).lower() != str(
        audit.get("experiment_commit", "")
    ):
        errors.append("manifest experiment_commit differs from the fresh source audit")
    if str(manifest.get("git_commit", "")).lower() != str(
        audit.get("experiment_commit", "")
    ):
        errors.append("legacy git_commit must equal experiment_commit")
    git_manifest = manifest.get("git")
    if not isinstance(git_manifest, Mapping) or git_manifest.get(
        "experiment_commit"
    ) != audit.get("experiment_commit") or git_manifest.get("worktree_clean") is not True:
        errors.append("manifest Git metadata does not bind the clean experiment commit")
    for tool_name in ("admission_tool", "table_generator"):
        if manifest.get(tool_name) != audit["tools"][tool_name]:
            errors.append(f"{tool_name} path/hash differs from the fresh audit")
    if manifest.get("audit_sha256") != audit["audit_sha256"]:
        errors.append("audit_sha256 differs from the fresh audit")

    declared_inputs = manifest.get("inputs")
    if not isinstance(declared_inputs, Mapping):
        errors.append("inputs must be an object")
        declared_inputs = {}
    expected_keys = {spec.key for spec in EVIDENCE_SPECS}
    if set(declared_inputs) != expected_keys:
        errors.append("manifest inputs must contain exactly the 14 configured keys")
    for key in sorted(expected_keys):
        row = declared_inputs.get(key)
        fresh = audit["inputs"].get(key)
        if row != fresh:
            errors.append(f"input {key} admission record differs from the fresh audit")
        if not isinstance(row, Mapping) or row.get("admitted") is not True:
            errors.append(f"input {key} is not admitted")
        elif any(
            check.get("applicable") is True and check.get("passed") is not True
            for check in row.get("checks", [])
            if isinstance(check, Mapping)
        ):
            errors.append(f"input {key} contains a failed applicable check")

    if expected_inputs is not None:
        if set(expected_inputs) != expected_keys:
            errors.append("table generator input keys differ from the admission contract")
        for key, expected_path in expected_inputs.items():
            if key not in audit["inputs"]:
                continue
            if expected_path.resolve() != (evidence_root / audit["inputs"][key]["path"]).resolve():
                errors.append(f"table generator path for {key} differs from the admission contract")

    if errors:
        raise ValueError("Publication evidence source manifest rejected:\n- " + "\n- ".join(errors))
    return manifest


def _print_audit(audit: Mapping[str, Any]) -> None:
    print(
        f"Revision publication evidence: {audit['status']} "
        f"({audit['admitted_input_count']}/{audit['configured_input_count']} inputs admitted)."
    )
    for check in audit["global_checks"]:
        marker = "PASS" if check["passed"] else "BLOCK"
        print(f"[{marker}] global {check['id']}: {check['detail']}")
    for key, row in audit["inputs"].items():
        marker = "PASS" if row["admitted"] else "BLOCK"
        print(f"[{marker}] {key}: {row['path']}")
        for blocker in row["blockers"]:
            print(f"  - {blocker}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("audit", "admit"), default="audit")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument(
        "--audit-json",
        type=Path,
        help="optional diagnostic audit output; never a publication source manifest",
    )
    args = parser.parse_args(argv)
    evidence_root = args.evidence_root.resolve()
    audit = audit_evidence(evidence_root)
    _print_audit(audit)
    if args.audit_json is not None:
        _atomic_write_json(args.audit_json.resolve(), audit)
        print(f"Diagnostic audit written to {args.audit_json.resolve()}")
    if args.mode == "audit":
        return 0
    if audit["eligible"] is not True:
        print("Admission refused; no publication source manifest was written.", file=sys.stderr)
        return 1
    manifest_out = (
        args.manifest_out.resolve()
        if args.manifest_out is not None
        else evidence_root / DEFAULT_MANIFEST_NAME
    )
    manifest = build_publication_manifest(audit)
    _atomic_write_json(manifest_out, manifest)
    validate_publication_source_manifest(
        manifest_out,
        evidence_root=evidence_root,
    )
    print(f"Publication source manifest written and revalidated: {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

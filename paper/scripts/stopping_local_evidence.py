#!/usr/bin/env python3
"""Independent fail-closed admission for local EXP-STOP-001 calibration.

All 45 locally required rows must have completed, hash-bound receipts.  This
module independently reloads the raw JSON/NPZ outputs, reconstructs endpoint
admission, recomputes every same-discretization comparison and policy choice,
and retains the seven parallel-cluster rows as explicit censors.  Timing is
never admitted and local completion can never be promoted to a complete
EXP-STOP-001 protocol pass.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_ID = "fenics-nonlinear-energies.exp-stop-001-local-paper-admission"
SCHEMA_VERSION = 1
PLAN_SCHEMA_ID = "fenics-nonlinear-energies.exp-stop-001.local-plan"
RECEIPT_SCHEMA_ID = "fenics-nonlinear-energies.exp-stop-001.local-receipt"
ANALYSIS_SCHEMA_ID = "fenics-nonlinear-energies.exp-stop-001.local-analysis"
P3D_RESULT_SCHEMA_ID = "fenics-nonlinear-energies.exp-stop-001.p3d-fixed-state"
CAMPAIGN_ID = "exp_stop_001_local_calibration_v1"
PLAN_NAME = "plan.json"
ANALYSIS_NAME = "analysis.json"
TABLE_NAME = "stopping_local_status.tex"
MANIFEST_NAME = "stopping_local_manifest.json"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")

EXPECTED_ENVIRONMENT = {
    "FNE_SKIP_REORDERED_WARMUP": "1",
    "JAX_ENABLE_X64": "True",
    "JAX_PLATFORMS": "cpu",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1",
}

EXPECTED_CONTRACT = {
    "gl_lumped_l2_relative_state_difference_max": 1.0e-5,
    "gl_energy_absolute_difference_max": 1.0e-10,
    "he_dual_norm_relative_difference_max": 1.0e-6,
    "he_state_scale_relative_difference_max": 1.0e-12,
    "he_true_residual_factor": 20.0,
    "he_true_residual_floor": 1.0e-8,
    "he_nonlinear_displacement_relative_difference_max": 1.0e-5,
    "he_nonlinear_reference_elastic_relative_state_difference_max": 1.0e-5,
    "he_nonlinear_energy_absolute_difference_max": 1.0e-8,
    "p3d_correction_relative_difference_max": 1.0e-4,
    "p3d_reference_elastic_relative_difference_max": 1.0e-4,
    "p3d_true_residual_factor": 20.0,
    "p3d_true_residual_floor": 1.0e-10,
    "p3d_nonlinear_reference_elastic_relative_state_difference_max": 1.0e-5,
    "p3d_nonlinear_energy_absolute_difference_max": 1.0e-6,
    "p3d_nonlinear_omega_absolute_difference_max": 1.0e-6,
    "p3d_nonlinear_u_max_absolute_difference_max": 1.0e-8,
}


class AdmissionError(ValueError):
    """Raised when local stopping-calibration evidence is inadmissible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _assert_finite_json(value: object, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AdmissionError(f"{label} contains a nonfinite value")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, label=f"{label}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_json(item, label=f"{label}.{key}")


def read_strict_json(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise AdmissionError(f"{path}: nonfinite JSON constant {value!r}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AdmissionError(f"{path}: top-level JSON value must be an object")
    _assert_finite_json(payload, label=str(path))
    return payload


def _finite(value: object, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdmissionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise AdmissionError(f"{label} must be {qualifier}")
    return result


def _assert_no_symlink(path: Path, *, root: Path, label: str) -> None:
    root = root.absolute()
    path = path.absolute()
    if root.is_symlink():
        raise AdmissionError(f"{label}: root must not be a symlink")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise AdmissionError(f"{label}: path is outside its required root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise AdmissionError(f"{label}: symlinks are forbidden ({current})")


def _safe_artifact(
    raw: object,
    *,
    repo_root: Path,
    evidence_root: Path,
    label: str,
    expected: Path | None = None,
    must_exist: bool = True,
) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise AdmissionError(f"{label} must be a nonempty path string")
    lexical = Path(raw)
    if ".." in lexical.parts:
        raise AdmissionError(f"{label} must not contain '..'")
    path = lexical if lexical.is_absolute() else repo_root / lexical
    path = path.absolute()
    _assert_no_symlink(path, root=evidence_root, label=label)
    resolved = path.resolve()
    try:
        resolved.relative_to(evidence_root.resolve())
    except ValueError as exc:
        raise AdmissionError(f"{label} resolves outside the evidence root") from exc
    if expected is not None and resolved != expected.resolve():
        raise AdmissionError(f"{label} does not identify the canonical path")
    if must_exist and not resolved.is_file():
        raise AdmissionError(f"{label} is missing: {resolved}")
    return resolved


def _git_output(repo_root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise AdmissionError(process.stderr.strip() or "Git command failed")
    return process.stdout.strip()


def _validate_commit(
    raw: object, *, repo_root: Path, require_release_clean: bool
) -> str:
    commit = str(raw).lower()
    if HEX40.fullmatch(commit) is None:
        raise AdmissionError("experiment source commit must be one exact 40-character SHA-1")
    head = _git_output(repo_root, "rev-parse", "HEAD").lower()
    process = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", commit, head],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise AdmissionError("experiment source commit is not an ancestor of release HEAD")
    if require_release_clean and _git_output(
        repo_root, "status", "--porcelain=v1", "--untracked-files=all"
    ):
        raise AdmissionError("release worktree must be clean during evidence admission")
    return commit


def _git_blob_sha256(repo_root: Path, commit: str, relative: str) -> str:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise AdmissionError(f"unsafe Git-bound path: {relative!r}")
    process = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{commit}:{relative}"],
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        raise AdmissionError(f"path is absent from experiment commit {commit}: {relative}")
    return hashlib.sha256(process.stdout).hexdigest()


def _verify_git_inventory(
    raw: object, *, repo_root: Path, commit: str, label: str
) -> dict[str, str]:
    if not isinstance(raw, dict) or not raw:
        raise AdmissionError(f"{label} must be a nonempty hash map")
    result: dict[str, str] = {}
    for relative, digest in sorted(raw.items()):
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise AdmissionError(f"{label} contains a malformed entry")
        if HEX64.fullmatch(digest) is None:
            raise AdmissionError(f"{label} contains a malformed SHA-256")
        if _git_blob_sha256(repo_root, commit, relative) != digest:
            raise AdmissionError(f"{label} differs from experiment-commit blob: {relative}")
        result[relative] = digest
    return result


def _load_npz(path: Path, *, required: Sequence[str] = ()) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set(required) - set(archive.files))
            if missing:
                raise AdmissionError(f"{path}: missing arrays {missing}")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise AdmissionError(f"cannot load safe NPZ {path}: {exc}") from exc
    for name, values in arrays.items():
        if values.dtype.kind == "O":
            raise AdmissionError(f"{path}: object array {name} is forbidden")
        if values.dtype.kind in "fc" and not np.all(np.isfinite(values)):
            raise AdmissionError(f"{path}: array {name} contains nonfinite values")
    return arrays


def _suffix(value: float) -> str:
    return f"{float(value):.0e}".replace("+", "").replace("-", "m")


def expected_design() -> dict[str, dict[str, object]]:
    """Return the frozen 45-local/7-cluster row design."""

    rows: dict[str, dict[str, object]] = {}

    def local(
        row_id: str,
        family: str,
        group: str,
        *,
        reference: bool,
        parameter: str,
        tolerance: float,
        outputs: Sequence[str],
        parameters: Mapping[str, object],
    ) -> None:
        rows[row_id] = {
            "execution_class": "required_local",
            "family": family,
            "group_id": group,
            "reference_row": reference,
            "selection_parameter": parameter,
            "selection_tolerance": float(tolerance),
            "expected_outputs": tuple(outputs),
            "parameters": dict(parameters),
        }

    gl_targets = (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
    for level in (5, 6):
        for target in gl_targets:
            row_id = f"gl_l{level}_residual_{_suffix(target)}"
            root = f"raw/gl/{row_id}"
            local(
                row_id,
                "ginzburg_landau",
                f"gl_l{level}",
                reference=target == 1.0e-8,
                parameter="relative_dual_residual_target",
                tolerance=target,
                outputs=(f"{root}/result.json", f"{root}/state.npz"),
                parameters={
                    "mesh_level": level,
                    "relative_dual_residual_target": target,
                    "linear_ksp_rtol": 1.0e-10,
                },
            )
    for level in (1, 2):
        for tolerance in (1.0e-8, 1.0e-10, 1.0e-12):
            row_id = f"he_l{level}_riesz_{_suffix(tolerance)}"
            local(
                row_id,
                "hyperelasticity_reference_riesz",
                f"he_l{level}",
                reference=tolerance == 1.0e-12,
                parameter="riesz_ksp_rtol",
                tolerance=tolerance,
                outputs=(f"raw/he/{row_id}/result.json",),
                parameters={
                    "mesh_level": level,
                    "riesz_ksp_rtol": tolerance,
                    "nonlinear_max_iterations": 0,
                },
            )
    for level in (1, 2):
        for target in gl_targets:
            row_id = f"he_l{level}_nonlinear_{_suffix(target)}"
            root = f"raw/he_nonlinear/{row_id}"
            local(
                row_id,
                "hyperelasticity_nonlinear_stopping",
                f"he_l{level}_nonlinear",
                reference=target == 1.0e-8,
                parameter="relative_dual_residual_target",
                tolerance=target,
                outputs=(f"{root}/result.json", f"{root}/state.npz"),
                parameters={
                    "mesh_level": level,
                    "relative_dual_residual_target": target,
                    "load_steps": 1,
                    "total_steps": 24,
                },
            )
    quadrature = {1: "tetra_1point", 2: "tetra_11point", 4: "tetra_24point"}
    for degree in (1, 2, 4):
        for tolerance in (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10):
            row_id = f"p3d_p{degree}_linear_{_suffix(tolerance)}"
            root = f"raw/p3d/{row_id}"
            local(
                row_id,
                "plasticity3d_fixed_state_linear",
                f"p3d_p{degree}",
                reference=tolerance == 1.0e-10,
                parameter="ksp_rtol",
                tolerance=tolerance,
                outputs=(f"{root}/result.json", f"{root}/state_and_correction.npz"),
                parameters={
                    "element_degree": degree,
                    "quadrature_rule_id": quadrature[degree],
                    "ksp_rtol": tolerance,
                },
            )
    for degree in (1, 2):
        for target in gl_targets:
            row_id = f"p3d_p{degree}_nonlinear_{_suffix(target)}"
            root = f"raw/p3d_nonlinear/{row_id}"
            local(
                row_id,
                "plasticity3d_nonlinear_stopping",
                f"p3d_p{degree}_nonlinear",
                reference=target == 1.0e-8,
                parameter="relative_dual_residual_target",
                tolerance=target,
                outputs=(f"{root}/result.json", f"{root}/state.npz"),
                parameters={
                    "element_degree": degree,
                    "quadrature_rule_id": quadrature[degree],
                    "relative_dual_residual_target": target,
                },
            )
    for target in gl_targets:
        row_id = f"p3d_p4_nonlinear_{_suffix(target)}_cluster"
        rows[row_id] = {
            "execution_class": "deferred_cluster_computation",
            "family": "plasticity3d_nonlinear_stopping",
            "group_id": "p3d_p4_nonlinear_cluster",
            "reference_row": False,
            "parameters": {
                "element_degree": 4,
                "relative_residual_target": target,
            },
        }
    for family in ("ginzburg_landau", "hyperelasticity", "plasticity3d"):
        row_id = f"{family}_mpi_consistency_cluster"
        rows[row_id] = {
            "execution_class": "deferred_cluster_computation",
            "family": f"{family}_mpi_consistency",
            "group_id": f"{family}_mpi_cluster",
            "reference_row": False,
            "parameters": {
                "rank_counts": "publication_rank_counts_from_dependent_protocols"
            },
        }
    return rows


def _validate_plan(
    plan_path: Path,
    *,
    repo_root: Path,
    evidence_root: Path,
    require_release_clean: bool,
) -> tuple[dict[str, object], str, dict[str, dict[str, object]]]:
    plan = read_strict_json(plan_path)
    if plan.get("schema_id") != PLAN_SCHEMA_ID or plan.get("schema_version") != 1:
        raise AdmissionError("local stopping plan schema is invalid")
    if plan.get("experiment_id") != "EXP-STOP-001" or plan.get("campaign_id") != CAMPAIGN_ID:
        raise AdmissionError("local stopping plan identity is invalid")
    if plan.get("run_kind") != "publication" or plan.get("publication_evidence_candidate") is not True:
        raise AdmissionError("local stopping plan is not publication evidence")
    output_root = _safe_artifact(
        plan.get("output_root"), repo_root=repo_root, evidence_root=evidence_root,
        label="plan output_root", expected=evidence_root, must_exist=False
    )
    if output_root != evidence_root.resolve():
        raise AdmissionError("plan output_root differs from evidence_root")
    source = plan.get("source")
    if not isinstance(source, dict) or source.get("dirty") is not False:
        raise AdmissionError("plan does not record a clean source")
    source_commit = _validate_commit(
        source.get("commit"), repo_root=repo_root,
        require_release_clean=require_release_clean
    )
    _verify_git_inventory(
        source.get("relevant_file_hashes"), repo_root=repo_root,
        commit=source_commit, label="plan source hashes"
    )
    inputs = plan.get("inputs")
    if not isinstance(inputs, dict):
        raise AdmissionError("plan input inventory is missing")
    _verify_git_inventory(
        inputs.get("file_hashes"), repo_root=repo_root,
        commit=source_commit, label="plan input hashes"
    )
    environment = plan.get("environment")
    if not isinstance(environment, dict) or environment.get("command_environment") != EXPECTED_ENVIRONMENT:
        raise AdmissionError("plan command environment differs from the frozen policy")
    policies = plan.get("policies")
    if not isinstance(policies, dict):
        raise AdmissionError("plan policy record is missing")
    if policies.get("p4_fixed_state") != "local" or policies.get("p4_local_feasibility_attested") is not True:
        raise AdmissionError("P4 fixed-state local feasibility is not frozen and attested")
    if policies.get("timing_claims_admissible") is not False:
        raise AdmissionError("plan improperly admits timing claims")
    if policies.get("analysis_contract") != EXPECTED_CONTRACT:
        raise AdmissionError("analysis contract differs from the frozen thresholds")
    if plan.get("row_counts") != {
        "total": 52,
        "required_local": 45,
        "deferred_cluster_computation": 7,
    }:
        raise AdmissionError("plan row counts differ from the frozen design")
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != 52:
        raise AdmissionError("plan must contain exactly 52 rows")
    actual = {
        str(row.get("row_id")): row for row in rows if isinstance(row, dict)
    }
    expected = expected_design()
    if len(actual) != 52 or set(actual) != set(expected):
        raise AdmissionError("plan row grid differs from the frozen design")
    for row_id, spec in expected.items():
        row = actual[row_id]
        for field in ("execution_class", "family", "group_id", "reference_row"):
            if row.get(field) != spec[field]:
                raise AdmissionError(f"{row_id}: field {field} differs from frozen design")
        parameters = row.get("parameters")
        if not isinstance(parameters, dict):
            raise AdmissionError(f"{row_id}: parameters are missing")
        for key, value in spec["parameters"].items():
            if parameters.get(key) != value:
                raise AdmissionError(f"{row_id}: parameter {key} differs from frozen design")
        if spec["execution_class"] == "deferred_cluster_computation":
            censor = row.get("censor")
            if row.get("command") is not None or row.get("expected_outputs") != []:
                raise AdmissionError(f"{row_id}: cluster censor unexpectedly has a command/output")
            if not isinstance(censor, dict) or censor.get("status") != "censored":
                raise AdmissionError(f"{row_id}: cluster censor is malformed")
            if censor.get("timing_admissible") is not False or censor.get("accuracy_claim_admissible") is not False:
                raise AdmissionError(f"{row_id}: cluster censor promotes unsupported evidence")
            continue
        expected_outputs = [evidence_root / raw for raw in spec["expected_outputs"]]
        outputs = row.get("expected_outputs")
        if not isinstance(outputs, list) or len(outputs) != len(expected_outputs):
            raise AdmissionError(f"{row_id}: expected-output inventory is malformed")
        for index, (raw, canonical) in enumerate(zip(outputs, expected_outputs)):
            _safe_artifact(
                raw, repo_root=repo_root, evidence_root=evidence_root,
                label=f"{row_id}.expected_outputs[{index}]", expected=canonical,
                must_exist=False
            )
        command = row.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(token, str) and token for token in command):
            raise AdmissionError(f"{row_id}: frozen command is malformed")
        if row.get("environment") != EXPECTED_ENVIRONMENT:
            raise AdmissionError(f"{row_id}: command environment differs from frozen policy")
        resolved_tokens: set[Path] = set()
        for token in command:
            candidate = Path(token)
            if candidate.is_absolute():
                resolved_tokens.add(candidate.resolve())
        if not set(path.resolve() for path in expected_outputs).issubset(resolved_tokens):
            raise AdmissionError(f"{row_id}: command does not bind every frozen output")
    claim_boundary = plan.get("claim_boundary")
    if not isinstance(claim_boundary, dict):
        raise AdmissionError("plan claim boundary is missing")
    cannot = claim_boundary.get("local_completion_cannot_establish")
    if not isinstance(cannot, list) or not any("terminal PASS" in str(item) for item in cannot):
        raise AdmissionError("plan does not refuse a complete local protocol pass")
    return plan, source_commit, actual


def _result_path(row: Mapping[str, object], *, repo_root: Path, evidence_root: Path) -> Path:
    outputs = row.get("expected_outputs")
    matches = [raw for raw in outputs if isinstance(raw, str) and Path(raw).suffix == ".json"] if isinstance(outputs, list) else []
    if len(matches) != 1:
        raise AdmissionError(f"{row.get('row_id')}: exactly one result JSON is required")
    return _safe_artifact(
        matches[0], repo_root=repo_root, evidence_root=evidence_root,
        label=f"{row.get('row_id')} result"
    )


def _state_path(row: Mapping[str, object], *, repo_root: Path, evidence_root: Path) -> Path:
    outputs = row.get("expected_outputs")
    matches = [raw for raw in outputs if isinstance(raw, str) and Path(raw).suffix == ".npz"] if isinstance(outputs, list) else []
    if len(matches) != 1:
        raise AdmissionError(f"{row.get('row_id')}: exactly one state NPZ is required")
    return _safe_artifact(
        matches[0], repo_root=repo_root, evidence_root=evidence_root,
        label=f"{row.get('row_id')} state"
    )


def _extract_gl(row: Mapping[str, object], *, repo_root: Path, evidence_root: Path) -> dict[str, object]:
    payload = read_strict_json(_result_path(row, repo_root=repo_root, evidence_root=evidence_root))
    result = payload.get("result")
    steps = result.get("steps") if isinstance(result, dict) else None
    if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], dict):
        raise AdmissionError("GL result must contain exactly one step")
    step = steps[0]
    convergence = step.get("convergence")
    metadata = result.get("metadata") if isinstance(result, dict) else None
    selection = (metadata.get("convergence") or {}).get("selection") if isinstance(metadata, dict) else None
    if not isinstance(convergence, dict):
        raise AdmissionError("GL convergence payload is missing")
    state_path = _state_path(row, repo_root=repo_root, evidence_root=evidence_root)
    arrays = _load_npz(state_path, required=("coords", "triangles", "u"))
    status = "endpoint_admitted" if step.get("success") is True and selection == "lumped_l2" else "censored_solver_nonconvergence"
    relative = _finite(convergence.get("dual_residual_relative"), label="GL relative residual", nonnegative=True)
    if status == "endpoint_admitted" and relative > float(row["parameters"]["relative_dual_residual_target"]):
        raise AdmissionError(f"{row['row_id']}: GL endpoint exceeds its stopping target")
    return {
        "status": status,
        "message": str(step.get("message", "")),
        "energy": _finite(step.get("energy"), label="GL energy"),
        "dual_residual_relative": relative,
        "correction_norm": _finite(convergence.get("correction_norm"), label="GL correction", nonnegative=True),
        "relative_correction": _finite(convergence.get("relative_correction"), label="GL relative correction", nonnegative=True),
        "state_sha256": array_sha256(np.asarray(arrays["u"], dtype=np.float64)),
        "state_file_sha256": sha256_file(state_path),
    }


def _extract_he_reference(row: Mapping[str, object], *, repo_root: Path, evidence_root: Path) -> dict[str, object]:
    payload = read_strict_json(_result_path(row, repo_root=repo_root, evidence_root=evidence_root))
    result = payload.get("result")
    steps = result.get("steps") if isinstance(result, dict) else None
    if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], dict):
        raise AdmissionError("HE reference result must contain one setup-only step")
    convergence = steps[0].get("convergence")
    if not isinstance(convergence, dict):
        raise AdmissionError("HE reference convergence payload is missing")
    metric = convergence.get("metric")
    norm_solve = convergence.get("dual_residual_metadata")
    certificate = ((metric.get("provenance") or {}).get("spd_certificate") if isinstance(metric, dict) else None)
    certified = isinstance(certificate, dict) and certificate.get("certified_spd") is True
    if not isinstance(norm_solve, dict):
        raise AdmissionError("HE reference norm-solve metadata is missing")
    true_residual = _finite(norm_solve.get("relative_true_residual"), label="HE true residual", nonnegative=True)
    true_gate = _finite(norm_solve.get("true_residual_rtol_gate"), label="HE true-residual gate", nonnegative=True)
    reason = int(norm_solve.get("reason", 0))
    status = "reference_metric_check_admitted" if certified and reason > 0 and true_residual <= true_gate else "reference_metric_check_failed"
    return {
        "status": status,
        "scope": "maxit_zero_reference_metric_check_not_nonlinear_convergence",
        "energy": _finite(steps[0].get("energy"), label="HE setup energy"),
        "dual_residual_norm": _finite(convergence.get("dual_residual_norm"), label="HE residual", nonnegative=True),
        "dual_residual_relative": _finite(convergence.get("dual_residual_relative"), label="HE relative residual", nonnegative=True),
        "state_scale": _finite(convergence.get("state_scale"), label="HE state scale", nonnegative=True),
        "riesz_iterations": int(norm_solve.get("iterations", 0)),
        "riesz_reason": reason,
        "relative_true_residual": true_residual,
        "true_residual_rtol_gate": true_gate,
        "certified_spd": certified,
    }


def _extract_he_nonlinear(row: Mapping[str, object], *, repo_root: Path, evidence_root: Path) -> dict[str, object]:
    payload = read_strict_json(_result_path(row, repo_root=repo_root, evidence_root=evidence_root))
    result = payload.get("result")
    steps = result.get("steps") if isinstance(result, dict) else None
    if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], dict):
        raise AdmissionError("HE nonlinear result must contain one load step")
    step = steps[0]
    convergence = step.get("convergence")
    if not isinstance(convergence, dict):
        raise AdmissionError("HE nonlinear convergence payload is missing")
    metric = convergence.get("metric")
    norm_solve = convergence.get("dual_residual_metadata")
    certificate = ((metric.get("provenance") or {}).get("spd_certificate") if isinstance(metric, dict) else None)
    certified = isinstance(certificate, dict) and certificate.get("certified_spd") is True
    if not isinstance(norm_solve, dict):
        raise AdmissionError("HE nonlinear norm-solve metadata is missing")
    true_residual = _finite(norm_solve.get("relative_true_residual"), label="HE nonlinear true residual", nonnegative=True)
    true_gate = _finite(norm_solve.get("true_residual_rtol_gate"), label="HE nonlinear true-residual gate", nonnegative=True)
    relative = _finite(convergence.get("dual_residual_relative"), label="HE nonlinear relative residual", nonnegative=True)
    target = float(row["parameters"]["relative_dual_residual_target"])
    state_path = _state_path(row, repo_root=repo_root, evidence_root=evidence_root)
    arrays = _load_npz(
        state_path,
        required=("coords_ref", "tetrahedra", "displacement", "free_deformation_original", "reference_elastic_action"),
    )
    free = np.asarray(arrays["free_deformation_original"], dtype=np.float64).reshape(-1)
    action = np.asarray(arrays["reference_elastic_action"], dtype=np.float64).reshape(-1)
    if free.shape != action.shape or free.size == 0:
        raise AdmissionError("HE nonlinear reference-elastic arrays are not aligned")
    quadratic = float(np.dot(free, action))
    tolerance = 256.0 * np.finfo(np.float64).eps * max(1.0, abs(quadratic))
    if not math.isfinite(quadratic) or quadratic < -tolerance:
        raise AdmissionError("HE nonlinear reference-elastic state norm is invalid")
    admitted = bool(step.get("success") is True and certified and int(norm_solve.get("reason", 0)) > 0 and true_residual <= true_gate and relative <= target)
    return {
        "status": "endpoint_admitted" if admitted else "censored_solver_nonconvergence",
        "scope": "one_load_step_full_nonlinear_reference_riesz_endpoint",
        "message": str(step.get("message", "")),
        "energy": _finite(step.get("energy"), label="HE nonlinear energy"),
        "dual_residual_norm": _finite(convergence.get("dual_residual_norm"), label="HE nonlinear residual", nonnegative=True),
        "dual_residual_relative": relative,
        "relative_correction": _finite(convergence.get("relative_correction"), label="HE nonlinear correction", nonnegative=True),
        "state_scale": _finite(convergence.get("state_scale"), label="HE nonlinear state scale", nonnegative=True),
        "relative_true_residual": true_residual,
        "true_residual_rtol_gate": true_gate,
        "certified_spd": certified,
        "state_sha256": array_sha256(np.asarray(arrays["displacement"], dtype=np.float64)),
        "state_file_sha256": sha256_file(state_path),
        "reference_elastic_state_norm": math.sqrt(max(0.0, quadratic)),
        "reference_elastic_action_sha256": array_sha256(action),
        "riesz_state_difference_available": True,
    }


def _extract_p3d_fixed(row: Mapping[str, object], *, repo_root: Path, evidence_root: Path) -> dict[str, object]:
    payload = read_strict_json(_result_path(row, repo_root=repo_root, evidence_root=evidence_root))
    if payload.get("schema_id") != P3D_RESULT_SCHEMA_ID or payload.get("schema_version") != 1:
        raise AdmissionError("P3D fixed-state result schema is invalid")
    linear = payload.get("linear_solve")
    if not isinstance(linear, dict):
        raise AdmissionError("P3D fixed-state linear solve is missing")
    state_path = _state_path(row, repo_root=repo_root, evidence_root=evidence_root)
    arrays = _load_npz(
        state_path, required=("state", "rhs", "correction", "reference_elastic_action")
    )
    file_row = payload.get("state_file")
    if not isinstance(file_row, dict) or file_row.get("sha256") != sha256_file(state_path):
        raise AdmissionError(f"{row['row_id']}: P3D state-file hash differs")
    _safe_artifact(
        file_row.get("path"),
        repo_root=repo_root,
        evidence_root=evidence_root,
        label=f"{row['row_id']}.state_file.path",
        expected=state_path,
    )
    expected_hashes = {
        "state_sha256": array_sha256(np.asarray(arrays["state"], dtype=np.float64)),
        "rhs_sha256": array_sha256(np.asarray(arrays["rhs"], dtype=np.float64)),
        "correction_sha256": array_sha256(np.asarray(arrays["correction"], dtype=np.float64)),
    }
    for key, digest in expected_hashes.items():
        if payload.get(key) != digest:
            raise AdmissionError(f"{row['row_id']}: {key} differs from NPZ content")
    status = str(payload.get("status", "failed"))
    reason = int(linear.get("reason", 0))
    relative_true = _finite(linear.get("relative_true_residual"), label="P3D true residual", nonnegative=True)
    true_gate = _finite(linear.get("true_residual_gate"), label="P3D true-residual gate", nonnegative=True)
    if status == "passed" and (reason <= 0 or relative_true > true_gate):
        raise AdmissionError(f"{row['row_id']}: P3D passed status violates the true-residual gate")
    return {
        "status": status,
        "scope": "fixed_state_linear_system_not_nonlinear_convergence",
        "ksp_reason": reason,
        "ksp_iterations": int(linear.get("iterations", 0)),
        "recursive_residual_norm": _finite(linear.get("recursive_residual_norm"), label="P3D recursive residual", nonnegative=True),
        "true_residual_norm": _finite(linear.get("true_residual_norm"), label="P3D true residual norm", nonnegative=True),
        "relative_true_residual": relative_true,
        "true_residual_gate": true_gate,
        "correction_norm_2": _finite(linear.get("correction_norm_2"), label="P3D correction norm", nonnegative=True),
        "reference_elastic_correction_norm": _finite(linear.get("reference_elastic_correction_norm"), label="P3D reference correction norm", nonnegative=True),
        "state_sha256": str(payload.get("state_sha256", "")),
        "rhs_sha256": str(payload.get("rhs_sha256", "")),
        "branch_diagnostics": dict(payload.get("branch_diagnostics", {})),
    }


def _extract_p3d_nonlinear(row: Mapping[str, object], *, repo_root: Path, evidence_root: Path) -> dict[str, object]:
    payload = read_strict_json(_result_path(row, repo_root=repo_root, evidence_root=evidence_root))
    convergence = payload.get("nonlinear_convergence")
    if not isinstance(convergence, dict):
        raise AdmissionError("P3D nonlinear convergence payload is missing")
    configuration = convergence.get("configuration")
    metric = convergence.get("metric")
    norm_solve = convergence.get("last_riesz_solve")
    if not all(isinstance(value, dict) for value in (configuration, metric, norm_solve)):
        raise AdmissionError("P3D nonlinear Riesz contract is incomplete")
    assert isinstance(configuration, dict)
    assert isinstance(metric, dict)
    assert isinstance(norm_solve, dict)
    certificate = ((metric.get("provenance") or {}).get("spd_certificate") if isinstance(metric.get("provenance"), dict) else None)
    certified = isinstance(certificate, dict) and certificate.get("certified_spd") is True
    true_residual = _finite(norm_solve.get("relative_true_residual"), label="P3D nonlinear true residual", nonnegative=True)
    true_gate = _finite(norm_solve.get("true_residual_rtol_gate"), label="P3D nonlinear true-residual gate", nonnegative=True)
    relative_row = convergence.get("initial_relative_dual_residual")
    correction_row = convergence.get("relative_correction")
    if not isinstance(relative_row, dict) or not isinstance(correction_row, dict):
        raise AdmissionError("P3D nonlinear residual/correction records are missing")
    relative = _finite(relative_row.get("value"), label="P3D nonlinear relative residual", nonnegative=True)
    target = float(row["parameters"]["relative_dual_residual_target"])
    residual_gate = convergence.get("residual_gate")
    residual_passed = isinstance(residual_gate, dict) and residual_gate.get("passed") is True
    state_path = _state_path(row, repo_root=repo_root, evidence_root=evidence_root)
    arrays = _load_npz(
        state_path,
        required=("coords_ref", "tetrahedra", "free_displacement_reordered", "reference_elastic_action"),
    )
    free = np.asarray(arrays["free_displacement_reordered"], dtype=np.float64).reshape(-1)
    action = np.asarray(arrays["reference_elastic_action"], dtype=np.float64).reshape(-1)
    if free.shape != action.shape or free.size == 0:
        raise AdmissionError("P3D nonlinear reference-elastic arrays are not aligned")
    admitted = bool(
        payload.get("status") == "completed"
        and payload.get("solver_success") is True
        and configuration.get("selection") == "reference_elastic_energy"
        and certified
        and int(norm_solve.get("reason", 0)) > 0
        and true_residual <= true_gate
        and residual_passed
        and relative <= target
    )
    return {
        "status": "endpoint_admitted" if admitted else "censored_solver_nonconvergence",
        "scope": "full_nonlinear_reference_riesz_endpoint_on_one_serial_rank",
        "message": str(payload.get("message", "")),
        "energy": _finite(payload.get("energy"), label="P3D nonlinear energy"),
        "omega": _finite(payload.get("omega"), label="P3D nonlinear omega"),
        "u_max": _finite(payload.get("u_max"), label="P3D nonlinear u_max", nonnegative=True),
        "dual_residual_relative": relative,
        "relative_correction": _finite(correction_row.get("value"), label="P3D nonlinear correction", nonnegative=True),
        "relative_true_residual": true_residual,
        "true_residual_rtol_gate": true_gate,
        "certified_spd": certified,
        "branch_diagnostics": dict(payload.get("branch_diagnostics", {})),
        "free_state_sha256": array_sha256(free),
        "reference_elastic_action_sha256": array_sha256(action),
        "state_file_sha256": sha256_file(state_path),
    }


def extract_endpoint(
    row: Mapping[str, object], *, repo_root: Path, evidence_root: Path
) -> dict[str, object]:
    family = row.get("family")
    if family == "ginzburg_landau":
        return _extract_gl(row, repo_root=repo_root, evidence_root=evidence_root)
    if family == "hyperelasticity_reference_riesz":
        return _extract_he_reference(row, repo_root=repo_root, evidence_root=evidence_root)
    if family == "hyperelasticity_nonlinear_stopping":
        return _extract_he_nonlinear(row, repo_root=repo_root, evidence_root=evidence_root)
    if family == "plasticity3d_fixed_state_linear":
        return _extract_p3d_fixed(row, repo_root=repo_root, evidence_root=evidence_root)
    if family == "plasticity3d_nonlinear_stopping":
        return _extract_p3d_nonlinear(row, repo_root=repo_root, evidence_root=evidence_root)
    raise AdmissionError(f"unsupported local endpoint family: {family}")


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), np.finfo(np.float64).tiny)


def _triangle_weights(coords: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.ndim == 2 and triangles.shape[1] != 3 and triangles.shape[0] == 3:
        triangles = triangles.T
    if triangles.ndim != 2 or triangles.shape[1] != 3 or coords.ndim != 2 or coords.shape[1] < 2:
        raise AdmissionError("GL mesh arrays have unsupported shapes")
    points = coords[triangles, :2]
    cross = (points[:, 1, 0] - points[:, 0, 0]) * (points[:, 2, 1] - points[:, 0, 1]) - (points[:, 1, 1] - points[:, 0, 1]) * (points[:, 2, 0] - points[:, 0, 0])
    areas = 0.5 * np.abs(cross)
    if np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
        raise AdmissionError("GL mesh contains a degenerate triangle")
    weights = np.zeros(coords.shape[0], dtype=np.float64)
    for local in range(3):
        np.add.at(weights, triangles[:, local], areas / 3.0)
    return weights


def recompute_comparison(
    row: Mapping[str, object],
    endpoint: Mapping[str, object],
    reference_row: Mapping[str, object],
    reference: Mapping[str, object],
    *,
    repo_root: Path,
    evidence_root: Path,
) -> dict[str, object]:
    family = str(row["family"])
    contract = EXPECTED_CONTRACT
    if family == "ginzburg_landau":
        if endpoint.get("status") != "endpoint_admitted" or reference.get("status") != "endpoint_admitted":
            return {"status": "censored", "reason": "candidate_or_reference_endpoint_not_converged"}
        candidate = _load_npz(_state_path(row, repo_root=repo_root, evidence_root=evidence_root), required=("coords", "triangles", "u"))
        ref = _load_npz(_state_path(reference_row, repo_root=repo_root, evidence_root=evidence_root), required=("coords", "triangles", "u"))
        coords = np.asarray(candidate["coords"], dtype=np.float64)
        triangles = np.asarray(candidate["triangles"], dtype=np.int64)
        if not np.array_equal(coords, np.asarray(ref["coords"], dtype=np.float64)) or not np.array_equal(triangles, np.asarray(ref["triangles"], dtype=np.int64)):
            raise AdmissionError("GL same-level comparison changed the mesh")
        difference = np.asarray(candidate["u"], dtype=np.float64).reshape(-1) - np.asarray(ref["u"], dtype=np.float64).reshape(-1)
        reference_u = np.asarray(ref["u"], dtype=np.float64).reshape(-1)
        weights = _triangle_weights(coords, triangles)
        norm = float(np.sqrt(np.dot(weights, difference * difference)))
        ref_norm = float(np.sqrt(np.dot(weights, reference_u * reference_u)))
        relative = norm / max(ref_norm, np.finfo(np.float64).tiny)
        energy = abs(float(endpoint["energy"]) - float(reference["energy"]))
        passed = relative <= contract["gl_lumped_l2_relative_state_difference_max"] and energy <= contract["gl_energy_absolute_difference_max"]
        return {
            "status": "accepted" if passed else "rejected",
            "reference_row_id": reference_row["row_id"],
            "lumped_l2_state_difference": norm,
            "lumped_l2_relative_state_difference": relative,
            "energy_absolute_difference": energy,
            "gates": {
                "relative_state_max": contract["gl_lumped_l2_relative_state_difference_max"],
                "energy_absolute_max": contract["gl_energy_absolute_difference_max"],
                "passed": passed,
            },
        }
    if family == "hyperelasticity_reference_riesz":
        if endpoint.get("status") != "reference_metric_check_admitted" or reference.get("status") != "reference_metric_check_admitted":
            return {"status": "censored", "reason": "candidate_or_reference_metric_check_failed"}
        dual = _relative_difference(float(endpoint["dual_residual_norm"]), float(reference["dual_residual_norm"]))
        scale = _relative_difference(float(endpoint["state_scale"]), float(reference["state_scale"]))
        tolerance = float(row["parameters"]["riesz_ksp_rtol"])
        true_gate = max(contract["he_true_residual_floor"], contract["he_true_residual_factor"] * tolerance)
        passed = dual <= contract["he_dual_norm_relative_difference_max"] and scale <= contract["he_state_scale_relative_difference_max"] and float(endpoint["relative_true_residual"]) <= true_gate
        return {
            "status": "accepted" if passed else "rejected",
            "reference_row_id": reference_row["row_id"],
            "dual_residual_norm_relative_difference": dual,
            "state_scale_relative_difference": scale,
            "relative_true_residual": endpoint["relative_true_residual"],
            "gates": {
                "dual_norm_relative_max": contract["he_dual_norm_relative_difference_max"],
                "state_scale_relative_max": contract["he_state_scale_relative_difference_max"],
                "true_residual_max": true_gate,
                "passed": passed,
            },
        }
    if family == "hyperelasticity_nonlinear_stopping":
        if endpoint.get("status") != "endpoint_admitted" or reference.get("status") != "endpoint_admitted":
            return {"status": "censored", "reason": "candidate_or_reference_endpoint_not_converged"}
        candidate = _load_npz(_state_path(row, repo_root=repo_root, evidence_root=evidence_root), required=("coords_ref", "tetrahedra", "displacement", "free_deformation_original", "reference_elastic_action"))
        ref = _load_npz(_state_path(reference_row, repo_root=repo_root, evidence_root=evidence_root), required=("coords_ref", "tetrahedra", "displacement", "free_deformation_original", "reference_elastic_action"))
        if not np.array_equal(candidate["coords_ref"], ref["coords_ref"]) or not np.array_equal(candidate["tetrahedra"], ref["tetrahedra"]):
            raise AdmissionError("HE same-level nonlinear comparison changed the mesh")
        displacement = np.asarray(candidate["displacement"], dtype=np.float64).reshape(-1)
        ref_displacement = np.asarray(ref["displacement"], dtype=np.float64).reshape(-1)
        coefficient_relative = float(np.linalg.norm(displacement - ref_displacement) / max(np.linalg.norm(ref_displacement), np.finfo(np.float64).tiny))
        state_diff = np.asarray(candidate["free_deformation_original"], dtype=np.float64).reshape(-1) - np.asarray(ref["free_deformation_original"], dtype=np.float64).reshape(-1)
        action_diff = np.asarray(candidate["reference_elastic_action"], dtype=np.float64).reshape(-1) - np.asarray(ref["reference_elastic_action"], dtype=np.float64).reshape(-1)
        squared = float(np.dot(state_diff, action_diff))
        ref_squared = float(np.dot(np.asarray(ref["free_deformation_original"], dtype=np.float64).reshape(-1), np.asarray(ref["reference_elastic_action"], dtype=np.float64).reshape(-1)))
        tolerance = 256.0 * np.finfo(np.float64).eps * max(1.0, abs(squared), abs(ref_squared))
        if squared < -tolerance or ref_squared < -tolerance:
            raise AdmissionError("HE reference-elastic comparison norm is invalid")
        difference = math.sqrt(max(0.0, squared))
        relative = difference / max(math.sqrt(max(0.0, ref_squared)), np.finfo(np.float64).tiny)
        energy = abs(float(endpoint["energy"]) - float(reference["energy"]))
        passed = coefficient_relative <= contract["he_nonlinear_displacement_relative_difference_max"] and relative <= contract["he_nonlinear_reference_elastic_relative_state_difference_max"] and energy <= contract["he_nonlinear_energy_absolute_difference_max"]
        return {
            "status": "accepted" if passed else "rejected",
            "reference_row_id": reference_row["row_id"],
            "coefficient_displacement_relative_difference": coefficient_relative,
            "reference_elastic_state_difference": difference,
            "reference_elastic_relative_state_difference": relative,
            "energy_absolute_difference": energy,
            "riesz_state_difference_available": True,
            "gates": {
                "coefficient_displacement_relative_max": contract["he_nonlinear_displacement_relative_difference_max"],
                "reference_elastic_relative_state_max": contract["he_nonlinear_reference_elastic_relative_state_difference_max"],
                "energy_absolute_max": contract["he_nonlinear_energy_absolute_difference_max"],
                "passed": passed,
            },
            "interpretation": "same-mesh endpoint difference in the frozen reference-elastic Riesz metric; the coefficient displacement difference is retained as a secondary diagnostic",
        }
    if family == "plasticity3d_fixed_state_linear":
        if endpoint.get("status") != "passed" or reference.get("status") != "passed":
            return {"status": "censored", "reason": "candidate_or_reference_linear_solve_failed"}
        if endpoint.get("state_sha256") != reference.get("state_sha256") or endpoint.get("rhs_sha256") != reference.get("rhs_sha256"):
            raise AdmissionError("P3D same-degree comparison changed fixed state or RHS")
        candidate = _load_npz(_state_path(row, repo_root=repo_root, evidence_root=evidence_root), required=("correction", "reference_elastic_action"))
        ref = _load_npz(_state_path(reference_row, repo_root=repo_root, evidence_root=evidence_root), required=("correction", "reference_elastic_action"))
        difference = np.asarray(candidate["correction"], dtype=np.float64).reshape(-1) - np.asarray(ref["correction"], dtype=np.float64).reshape(-1)
        action_difference = np.asarray(candidate["reference_elastic_action"], dtype=np.float64).reshape(-1) - np.asarray(ref["reference_elastic_action"], dtype=np.float64).reshape(-1)
        coefficient = float(np.linalg.norm(difference) / max(np.linalg.norm(np.asarray(ref["correction"], dtype=np.float64)), np.finfo(np.float64).tiny))
        squared = float(np.dot(difference, action_difference))
        if squared < -1.0e-10 * max(float(np.linalg.norm(difference) * np.linalg.norm(action_difference)), 1.0):
            raise AdmissionError("P3D reference-elastic correction difference is invalid")
        difference_norm = math.sqrt(max(0.0, squared))
        relative = difference_norm / max(float(reference["reference_elastic_correction_norm"]), np.finfo(np.float64).tiny)
        tolerance = float(row["parameters"]["ksp_rtol"])
        true_gate = max(contract["p3d_true_residual_floor"], contract["p3d_true_residual_factor"] * tolerance)
        passed = coefficient <= contract["p3d_correction_relative_difference_max"] and relative <= contract["p3d_reference_elastic_relative_difference_max"] and float(endpoint["relative_true_residual"]) <= true_gate
        return {
            "status": "accepted" if passed else "rejected",
            "reference_row_id": reference_row["row_id"],
            "coefficient_l2_relative_correction_difference": coefficient,
            "reference_elastic_correction_difference": difference_norm,
            "reference_elastic_relative_correction_difference": relative,
            "relative_true_residual": endpoint["relative_true_residual"],
            "gates": {
                "coefficient_relative_max": contract["p3d_correction_relative_difference_max"],
                "reference_elastic_relative_max": contract["p3d_reference_elastic_relative_difference_max"],
                "true_residual_max": true_gate,
                "passed": passed,
            },
        }
    if family == "plasticity3d_nonlinear_stopping":
        if endpoint.get("status") != "endpoint_admitted" or reference.get("status") != "endpoint_admitted":
            return {"status": "censored", "reason": "candidate_or_reference_endpoint_not_converged"}
        candidate = _load_npz(_state_path(row, repo_root=repo_root, evidence_root=evidence_root), required=("coords_ref", "tetrahedra", "free_displacement_reordered", "reference_elastic_action"))
        ref = _load_npz(_state_path(reference_row, repo_root=repo_root, evidence_root=evidence_root), required=("coords_ref", "tetrahedra", "free_displacement_reordered", "reference_elastic_action"))
        if not np.array_equal(candidate["coords_ref"], ref["coords_ref"]) or not np.array_equal(candidate["tetrahedra"], ref["tetrahedra"]):
            raise AdmissionError("P3D same-degree nonlinear comparison changed the mesh")
        difference = np.asarray(candidate["free_displacement_reordered"], dtype=np.float64).reshape(-1) - np.asarray(ref["free_displacement_reordered"], dtype=np.float64).reshape(-1)
        action_difference = np.asarray(candidate["reference_elastic_action"], dtype=np.float64).reshape(-1) - np.asarray(ref["reference_elastic_action"], dtype=np.float64).reshape(-1)
        squared = float(np.dot(difference, action_difference))
        if squared < -1.0e-10 * max(float(np.linalg.norm(difference) * np.linalg.norm(action_difference)), 1.0):
            raise AdmissionError("P3D nonlinear reference-elastic difference is invalid")
        difference_norm = math.sqrt(max(0.0, squared))
        reference_squared = float(np.dot(np.asarray(ref["free_displacement_reordered"], dtype=np.float64).reshape(-1), np.asarray(ref["reference_elastic_action"], dtype=np.float64).reshape(-1)))
        relative = difference_norm / max(math.sqrt(max(0.0, reference_squared)), np.finfo(np.float64).tiny)
        energy = abs(float(endpoint["energy"]) - float(reference["energy"]))
        omega = abs(float(endpoint["omega"]) - float(reference["omega"]))
        u_max = abs(float(endpoint["u_max"]) - float(reference["u_max"]))
        candidate_counts = (endpoint.get("branch_diagnostics") or {}).get("counts") if isinstance(endpoint.get("branch_diagnostics"), dict) else None
        reference_counts = (reference.get("branch_diagnostics") or {}).get("counts") if isinstance(reference.get("branch_diagnostics"), dict) else None
        branch_equal = isinstance(candidate_counts, dict) and candidate_counts == reference_counts
        passed = relative <= contract["p3d_nonlinear_reference_elastic_relative_state_difference_max"] and energy <= contract["p3d_nonlinear_energy_absolute_difference_max"] and omega <= contract["p3d_nonlinear_omega_absolute_difference_max"] and u_max <= contract["p3d_nonlinear_u_max_absolute_difference_max"] and branch_equal
        return {
            "status": "accepted" if passed else "rejected",
            "reference_row_id": reference_row["row_id"],
            "reference_elastic_state_difference": difference_norm,
            "reference_elastic_relative_state_difference": relative,
            "energy_absolute_difference": energy,
            "omega_absolute_difference": omega,
            "u_max_absolute_difference": u_max,
            "branch_counts_equal": branch_equal,
            "gates": {
                "reference_elastic_relative_state_max": contract["p3d_nonlinear_reference_elastic_relative_state_difference_max"],
                "energy_absolute_max": contract["p3d_nonlinear_energy_absolute_difference_max"],
                "omega_absolute_max": contract["p3d_nonlinear_omega_absolute_difference_max"],
                "u_max_absolute_max": contract["p3d_nonlinear_u_max_absolute_difference_max"],
                "branch_counts_exact": True,
                "passed": passed,
            },
        }
    raise AdmissionError(f"unsupported comparison family: {family}")


def select_policy(
    rows: Sequence[Mapping[str, object]], comparisons: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    accepted = [row for row in rows if comparisons.get(str(row["row_id"]), {}).get("status") == "accepted"]
    if not accepted:
        return {"status": "no_acceptable_policy", "row_id": None, "tolerance": None}
    design = expected_design()
    selected = max(accepted, key=lambda row: float(design[str(row["row_id"])]["selection_tolerance"]))
    spec = design[str(selected["row_id"])]
    return {
        "status": "selected_loosest_accepted_same_discretization_policy",
        "row_id": selected["row_id"],
        "parameter": spec["selection_parameter"],
        "tolerance": spec["selection_tolerance"],
    }


def _validate_receipts(
    *,
    plan: Mapping[str, object],
    rows: Mapping[str, Mapping[str, object]],
    plan_path: Path,
    source_commit: str,
    repo_root: Path,
    evidence_root: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], set[Path]]:
    receipts: dict[str, dict[str, object]] = {}
    endpoints: dict[str, dict[str, object]] = {}
    referenced: set[Path] = {plan_path}
    for row_id, row in sorted(rows.items()):
        if row.get("execution_class") != "required_local":
            continue
        receipt_path = evidence_root / "receipts" / f"{row_id}.json"
        _assert_no_symlink(receipt_path, root=evidence_root, label=f"{row_id} receipt")
        if not receipt_path.is_file():
            raise AdmissionError(f"{row_id}: required local receipt is missing")
        receipt = read_strict_json(receipt_path)
        referenced.add(receipt_path)
        if receipt.get("schema_id") != RECEIPT_SCHEMA_ID or receipt.get("schema_version") != 1:
            raise AdmissionError(f"{row_id}: receipt schema is invalid")
        if receipt.get("experiment_id") != "EXP-STOP-001" or receipt.get("campaign_id") != CAMPAIGN_ID:
            raise AdmissionError(f"{row_id}: receipt experiment identity is invalid")
        if receipt.get("row_id") != row_id or receipt.get("source_commit") != source_commit:
            raise AdmissionError(f"{row_id}: receipt row/source identity differs")
        if receipt.get("run_kind") != "publication" or receipt.get("plan_sha256") != sha256_file(plan_path):
            raise AdmissionError(f"{row_id}: receipt plan binding is invalid")
        plan_binding = _safe_artifact(
            receipt.get("plan_path"), repo_root=repo_root, evidence_root=evidence_root,
            label=f"{row_id}.plan_path", expected=plan_path
        )
        referenced.add(plan_binding)
        if receipt.get("command") != row.get("command") or receipt.get("environment_overrides") != row.get("environment"):
            raise AdmissionError(f"{row_id}: receipt command/environment differs from plan")
        if receipt.get("status") != "completed" or receipt.get("returncode") != 0 or receipt.get("timed_out") is not False or receipt.get("verification_error") is not None:
            raise AdmissionError(f"{row_id}: local execution did not complete cleanly")
        _finite(receipt.get("wall_time_s"), label=f"{row_id}.wall_time_s", nonnegative=True)
        logs = receipt.get("logs")
        if not isinstance(logs, dict) or set(logs) != {"stdout", "stderr"}:
            raise AdmissionError(f"{row_id}: receipt log inventory is malformed")
        for name in ("stdout", "stderr"):
            path = _safe_artifact(
                logs[name], repo_root=repo_root, evidence_root=evidence_root,
                label=f"{row_id}.{name}", expected=evidence_root / "logs" / row_id / f"{name}.log"
            )
            referenced.add(path)
        expected_outputs = [
            _safe_artifact(
                raw, repo_root=repo_root, evidence_root=evidence_root,
                label=f"{row_id}.expected_output"
            )
            for raw in row.get("expected_outputs", [])
        ]
        output_hashes = receipt.get("output_hashes")
        if not isinstance(output_hashes, dict) or len(output_hashes) != len(expected_outputs):
            raise AdmissionError(f"{row_id}: receipt output-hash inventory is incomplete")
        matched: set[Path] = set()
        for raw, digest in output_hashes.items():
            path = _safe_artifact(
                raw, repo_root=repo_root, evidence_root=evidence_root,
                label=f"{row_id}.output_hash"
            )
            if path not in expected_outputs or path in matched:
                raise AdmissionError(f"{row_id}: receipt output path is noncanonical or duplicated")
            if not isinstance(digest, str) or HEX64.fullmatch(digest) is None or sha256_file(path) != digest:
                raise AdmissionError(f"{row_id}: receipt output SHA-256 mismatch")
            matched.add(path)
            referenced.add(path)
            if path.suffix == ".json":
                read_strict_json(path)
            elif path.suffix == ".npz":
                _load_npz(path)
        if set(expected_outputs) != matched:
            raise AdmissionError(f"{row_id}: receipt does not close every frozen output")
        receipts[row_id] = receipt
        endpoints[row_id] = extract_endpoint(
            row, repo_root=repo_root, evidence_root=evidence_root
        )
    if len(receipts) != 45 or len(endpoints) != 45:
        raise AdmissionError("all 45 local receipts/endpoints are required")
    return receipts, endpoints, referenced


def _validate_analysis(
    *,
    analysis_path: Path,
    plan: Mapping[str, object],
    plan_path: Path,
    source_commit: str,
    rows: Mapping[str, Mapping[str, object]],
    receipts: Mapping[str, Mapping[str, object]],
    endpoints: Mapping[str, Mapping[str, object]],
    repo_root: Path,
    evidence_root: Path,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    analysis = read_strict_json(analysis_path)
    if analysis.get("schema_id") != ANALYSIS_SCHEMA_ID or analysis.get("schema_version") != 2:
        raise AdmissionError("local stopping analysis schema is invalid")
    if analysis.get("experiment_id") != "EXP-STOP-001" or analysis.get("campaign_id") != CAMPAIGN_ID:
        raise AdmissionError("local stopping analysis identity is invalid")
    if analysis.get("terminal_decision") != "local_calibration_complete_cluster_computations_deferred":
        raise AdmissionError("local analysis is not complete with cluster computations deferred")
    if analysis.get("complete_exp_stop_pass") is not False or analysis.get("publication_timing_admissible") is not False:
        raise AdmissionError("local analysis overclaims complete-protocol or timing evidence")
    plan_row = analysis.get("plan")
    if not isinstance(plan_row, dict):
        raise AdmissionError("analysis plan binding is missing")
    _safe_artifact(
        plan_row.get("path"), repo_root=repo_root, evidence_root=evidence_root,
        label="analysis.plan.path", expected=plan_path
    )
    if plan_row.get("sha256") != sha256_file(plan_path) or plan_row.get("run_kind") != "publication" or plan_row.get("source_commit") != source_commit:
        raise AdmissionError("analysis plan hash/source binding is invalid")
    expected_counts = {
        "required_local": 45,
        "completed_endpoint_records": 45,
        "missing_local": 0,
        "invalid_local": 0,
        "runtime_censored_local": 0,
        "reference_failures": 0,
        "policy_gate_failures": 0,
        "deferred_cluster_computations": 7,
    }
    if analysis.get("counts") != expected_counts:
        raise AdmissionError("analysis counts do not prove complete local execution")
    for field in (
        "missing_local_rows",
        "invalid_local_rows",
        "runtime_censored_local_rows",
        "reference_failures",
        "runtime_censors",
    ):
        if analysis.get(field) != []:
            raise AdmissionError(f"analysis field {field} must be empty")
    source_audit = analysis.get("audit")
    if not isinstance(source_audit, dict) or set(source_audit) != set(receipts):
        raise AdmissionError("analysis receipt audit grid is incomplete")
    for row_id, record in source_audit.items():
        if not isinstance(record, dict) or record.get("receipt_status") != "completed" or record.get("errors") != []:
            raise AdmissionError(f"{row_id}: analysis receipt audit is not clean")
        _safe_artifact(
            record.get("receipt"), repo_root=repo_root, evidence_root=evidence_root,
            label=f"{row_id}.analysis_receipt", expected=evidence_root / "receipts" / f"{row_id}.json"
        )
    if analysis.get("endpoints") != endpoints:
        raise AdmissionError("stored endpoints differ from independent raw-output extraction")
    comparisons: dict[str, dict[str, object]] = {}
    local_rows = [row for row in rows.values() if row.get("execution_class") == "required_local"]
    for group in sorted({str(row["group_id"]) for row in local_rows}):
        group_rows = [row for row in local_rows if row["group_id"] == group]
        references = [row for row in group_rows if row.get("reference_row") is True]
        if len(references) != 1:
            raise AdmissionError(f"{group}: group does not have exactly one reference")
        reference_row = references[0]
        reference = endpoints[str(reference_row["row_id"])]
        admitted_status = {
            "ginzburg_landau": "endpoint_admitted",
            "hyperelasticity_reference_riesz": "reference_metric_check_admitted",
            "hyperelasticity_nonlinear_stopping": "endpoint_admitted",
            "plasticity3d_fixed_state_linear": "passed",
            "plasticity3d_nonlinear_stopping": "endpoint_admitted",
        }[str(reference_row["family"])]
        if reference.get("status") != admitted_status:
            raise AdmissionError(f"{group}: tight reference endpoint is not admitted")
        for row in group_rows:
            row_id = str(row["row_id"])
            comparisons[row_id] = recompute_comparison(
                row,
                endpoints[row_id],
                reference_row,
                reference,
                repo_root=repo_root,
                evidence_root=evidence_root,
            )
    if analysis.get("same_discretization_reference_comparisons") != comparisons:
        raise AdmissionError("stored comparisons differ from independent gate recomputation")
    selected: dict[str, dict[str, object]] = {}
    for group in sorted({str(row["group_id"]) for row in local_rows}):
        group_rows = [row for row in local_rows if row["group_id"] == group]
        selected[group] = select_policy(group_rows, comparisons)
    if analysis.get("selected_local_policies") != selected:
        raise AdmissionError("stored policy selections differ from independent adjudication")
    expected_groups = sorted(selected)
    if len(expected_groups) != 11 or any(
        record.get("status")
        != "selected_loosest_accepted_same_discretization_policy"
        for record in selected.values()
    ):
        raise AdmissionError(
            "every one of the 11 required local groups must select an accepted policy"
        )
    policy_grid = {
        "expected_groups": expected_groups,
        "observed_groups": expected_groups,
        "missing_groups": [],
        "unexpected_groups": [],
        "missing_policy_records": [],
        "unexpected_policy_records": [],
        "rejected_policy_groups": [],
        "invalid_selected_rows": [],
        "complete": True,
    }
    if analysis.get("required_local_policy_grid") != policy_grid:
        raise AdmissionError(
            "required_local_policy_grid does not prove the exact complete 11-group gate"
        )
    deferred = analysis.get("deferred_cluster_computations")
    expected_deferred = [row for row in rows.values() if row.get("execution_class") == "deferred_cluster_computation"]
    if not isinstance(deferred, list) or len(deferred) != 7 or {
        str(item.get("row_id")) for item in deferred if isinstance(item, dict)
    } != {str(row["row_id"]) for row in expected_deferred}:
        raise AdmissionError("analysis cluster-deferred row grid is incomplete")
    expected_deferred_by_id = {
        str(row["row_id"]): row for row in expected_deferred
    }
    for item in deferred:
        if not isinstance(item, dict) or not isinstance(item.get("censor"), dict):
            raise AdmissionError("analysis cluster censor is malformed")
        expected_row = expected_deferred_by_id[str(item["row_id"])]
        if item.get("family") != expected_row.get("family") or item.get(
            "parameters"
        ) != expected_row.get("parameters") or item.get("censor") != expected_row.get(
            "censor"
        ):
            raise AdmissionError("analysis cluster censor differs from the frozen plan")
        censor = item["censor"]
        if censor.get("status") != "censored" or censor.get("timing_admissible") is not False or censor.get("accuracy_claim_admissible") is not False:
            raise AdmissionError("analysis cluster censor promotes unsupported evidence")
    return analysis, comparisons


def _tree_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AdmissionError(f"symlinks are forbidden in evidence tree: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        hashes[relative] = sha256_file(path)
        if path.suffix == ".json":
            read_strict_json(path)
        elif path.suffix == ".npz":
            _load_npz(path)
    return hashes


def audit_campaign(
    evidence_root: Path,
    *,
    repo_root: Path,
    require_release_clean: bool = True,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    evidence_root = evidence_root.absolute()
    reproduction = (repo_root / "artifacts/reproduction").resolve()
    _assert_no_symlink(evidence_root, root=reproduction, label="evidence_root")
    try:
        evidence_root.resolve().relative_to(reproduction)
    except ValueError as exc:
        raise AdmissionError("evidence_root must be below artifacts/reproduction") from exc
    plan_path = evidence_root / PLAN_NAME
    analysis_path = evidence_root / ANALYSIS_NAME
    if not plan_path.is_file() or not analysis_path.is_file():
        raise AdmissionError("canonical plan.json and analysis.json are both required")
    plan, source_commit, rows = _validate_plan(
        plan_path,
        repo_root=repo_root,
        evidence_root=evidence_root,
        require_release_clean=require_release_clean,
    )
    receipts, endpoints, referenced = _validate_receipts(
        plan=plan,
        rows=rows,
        plan_path=plan_path,
        source_commit=source_commit,
        repo_root=repo_root,
        evidence_root=evidence_root,
    )
    analysis, comparisons = _validate_analysis(
        analysis_path=analysis_path,
        plan=plan,
        plan_path=plan_path,
        source_commit=source_commit,
        rows=rows,
        receipts=receipts,
        endpoints=endpoints,
        repo_root=repo_root,
        evidence_root=evidence_root,
    )
    referenced.add(analysis_path)
    hashes = _tree_hashes(evidence_root)
    for path in referenced:
        relative = path.resolve().relative_to(evidence_root.resolve()).as_posix()
        if relative not in hashes:
            raise AdmissionError(f"referenced artifact is absent from hash closure: {relative}")
    families: list[dict[str, object]] = []
    for family in (
        "ginzburg_landau",
        "hyperelasticity_reference_riesz",
        "hyperelasticity_nonlinear_stopping",
        "plasticity3d_fixed_state_linear",
        "plasticity3d_nonlinear_stopping",
    ):
        family_rows = [row for row in rows.values() if row.get("execution_class") == "required_local" and row.get("family") == family]
        admitted_label = {
            "ginzburg_landau": "endpoint_admitted",
            "hyperelasticity_reference_riesz": "reference_metric_check_admitted",
            "hyperelasticity_nonlinear_stopping": "endpoint_admitted",
            "plasticity3d_fixed_state_linear": "passed",
            "plasticity3d_nonlinear_stopping": "endpoint_admitted",
        }[family]
        families.append(
            {
                "family": family,
                "completed_receipts": len(family_rows),
                "required_local": len(family_rows),
                "admitted_endpoints": sum(endpoints[str(row["row_id"])].get("status") == admitted_label for row in family_rows),
                "accepted_comparisons": sum(comparisons[str(row["row_id"])].get("status") == "accepted" for row in family_rows),
                "comparison_rows": len(family_rows),
            }
        )
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "admitted_local_calibration_cluster_deferred",
        "experiment_id": "EXP-STOP-001",
        "source_commit": source_commit,
        "plan_sha256": sha256_file(plan_path),
        "analysis_sha256": sha256_file(analysis_path),
        "artifact_hashes": hashes,
        "artifact_hashes_sha256": json_sha256(hashes),
        "artifact_count": len(hashes),
        "scientific_adjudication": {
            "required_local_rows": 45,
            "completed_local_rows": 45,
            "deferred_cluster_rows": 7,
            "complete_exp_stop_pass": False,
            "timing_claim_admissible": False,
            "population_robustness_claim_admissible": False,
            "family_summaries": families,
            "claim_scope": "Deterministic same-discretization local stopping calibration only.",
            "claim_refusals": [
                (
                    "Closure of the seven deferred cluster rows is necessary but not "
                    "sufficient for a complete EXP-STOP-001 pass; a separately hash-bound "
                    "EXP-DISC gate and final adjudication are also required."
                ),
                "This local artifact does not bind or adjudicate EXP-DISC evidence.",
                "No timing, scaling, or performance claim is admitted.",
                "No population robustness claim is admitted by this deterministic calibration.",
            ],
        },
    }


def render_table(audit: Mapping[str, object]) -> str:
    adjudication = audit.get("scientific_adjudication")
    if not isinstance(adjudication, Mapping):
        raise AdmissionError("scientific adjudication is missing")
    rows = adjudication.get("family_summaries")
    if not isinstance(rows, list) or len(rows) != 5:
        raise AdmissionError("family summary is incomplete")
    labels = {
        "ginzburg_landau": "Ginzburg--Landau endpoints",
        "hyperelasticity_reference_riesz": "Hyperelasticity metric checks",
        "hyperelasticity_nonlinear_stopping": "Hyperelasticity nonlinear endpoints",
        "plasticity3d_fixed_state_linear": "Plasticity3D fixed-state solves",
        "plasticity3d_nonlinear_stopping": "Plasticity3D nonlinear endpoints",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Fail-closed status of the local stopping-calibration tranche.}",
        r"\label{tab:stopping-local-status}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Family & Receipts & Admitted endpoints & Accepted comparisons \\",
        r"\midrule",
    ]
    for row in rows:
        family = str(row.get("family"))
        if family not in labels:
            raise AdmissionError("family summary contains an unknown family")
        lines.append(
            f"{labels[family]} & {int(row['completed_receipts'])}/{int(row['required_local'])} & "
            f"{int(row['admitted_endpoints'])}/{int(row['required_local'])} & "
            f"{int(row['accepted_comparisons'])}/{int(row['comparison_rows'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.96\linewidth}\small",
            (
                r"All 45 serial rows are hash-bound and independently rechecked. "
                r"Four P4 nonlinear rows and three MPI-consistency rows remain "
                r"cluster-deferred. Closing them is necessary but not sufficient for a "
                r"complete EXP-STOP-001 pass: separately hash-bound EXP-DISC evidence "
                r"and final adjudication are also required. Timing, "
                r"scaling, performance, and population claims are excluded."
            ),
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)

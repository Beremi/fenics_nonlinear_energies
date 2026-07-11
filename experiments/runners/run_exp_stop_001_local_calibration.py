#!/usr/bin/env python3
"""Prepare, execute, and analyze the local EXP-STOP-001 calibration campaign.

The campaign is deliberately staged.  ``prepare`` freezes commands and input
hashes into a fresh output root, ``execute`` runs either one frozen local row
or the complete local tranche, and ``analyze`` compares every endpoint with
the tightest successful same-discretization reference.  Full nonlinear P4
Plasticity3D rows and all MPI-consistency rows remain explicit
cluster-deferred censors in this local campaign.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import socket
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = Path("experiments/runners/run_exp_stop_001_local_calibration.py")
TRUST_RUNNER_PATH = Path("experiments/runners/run_trust_region_case.py")
P3D_BACKEND_PATH = Path("experiments/runners/run_plasticity3d_backend_mix_case.py")
P3D_ROUTE_PATH = Path("experiments/runners/run_plasticity3d_fixed_state_route_screen.py")
PROTOCOL_PATH = Path("paper/protocols/EXP-STOP-001.md")
P3D_MESH_MANIFEST_PATH = Path(
    "data/meshes/SlopeStability3D/hetero_ssr/publication_mesh_manifest.json"
)
P3D_MESH_GENERATOR_SOURCES = (
    Path("data/meshes/SlopeStability3D/hetero_ssr/SSR_hetero_ada_L1.msh"),
    Path("data/meshes/SlopeStability3D/hetero_ssr/definition.py"),
    Path("src/problems/slope_stability_3d/support/materials.py"),
    Path("src/problems/slope_stability_3d/support/mesh.py"),
    Path("src/problems/slope_stability_3d/support/simplex_lagrange.py"),
)

PLAN_SCHEMA_ID = "fenics-nonlinear-energies.exp-stop-001.local-plan"
PLAN_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_ID = "fenics-nonlinear-energies.exp-stop-001.local-receipt"
RECEIPT_SCHEMA_VERSION = 1
ANALYSIS_SCHEMA_ID = "fenics-nonlinear-energies.exp-stop-001.local-analysis"
ANALYSIS_SCHEMA_VERSION = 2
P3D_RESULT_SCHEMA_ID = "fenics-nonlinear-energies.exp-stop-001.p3d-fixed-state"
P3D_RESULT_SCHEMA_VERSION = 1

GL_LEVELS = (5, 6)
GL_RESIDUAL_TARGETS = (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
HE_LEVELS = (1, 2)
HE_RIESZ_RTOLS = (1.0e-8, 1.0e-10, 1.0e-12)
P3D_LINEAR_RTOLS = (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10)
P3D_QUADRATURE = {
    1: "tetra_1point",
    2: "tetra_11point",
    4: "tetra_24point",
}
HE_NONLINEAR_TARGETS = (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
P3D_NONLINEAR_TARGETS = (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-7)
ACCEPTED_POLICY_STATUS = "selected_loosest_accepted_same_discretization_policy"
COMPLETE_REQUIRED_LOCAL_GROUPS = frozenset(
    {
        *(f"gl_l{level}" for level in GL_LEVELS),
        *(f"he_l{level}" for level in HE_LEVELS),
        *(f"he_l{level}_nonlinear" for level in HE_LEVELS),
        *(f"p3d_p{degree}" for degree in (1, 2, 4)),
        *(f"p3d_p{degree}_nonlinear" for degree in (1, 2)),
    }
)

SAFE_ROW_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
)


class CampaignError(RuntimeError):
    """Raised when a fail-closed campaign contract is violated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CampaignError(f"JSON root must be an object: {path}")
    return payload


def _manifested_p3d_meshes() -> dict[str, dict[str, Any]]:
    """Verify and bind the ignored P1/P2/P4 generated HDF5 caches."""

    from src.problems.slope_stability_3d.support.mesh import (
        _same_mesh_hdf5_is_current,
    )

    manifest_path = REPO_ROOT / P3D_MESH_MANIFEST_PATH
    payload = _read_json(manifest_path)
    expected_paths = {
        (
            "data/meshes/SlopeStability3D/hetero_ssr/"
            f"hetero_ssr_L1_p{degree}_same_mesh_glued_bottom.h5"
        ): degree
        for degree in (1, 2, 4)
    }
    if (
        set(payload) != {"algorithm", "files", "generator", "schema_id", "schema_version"}
        or payload.get("schema_id")
        != "fenics-nonlinear-energies.manifested-generated-meshes"
        or payload.get("schema_version") != 1
        or payload.get("algorithm") != "sha256"
        or not isinstance(payload.get("files"), dict)
        or set(payload["files"]) != set(expected_paths)
        or payload.get("generator")
        != {
            "function": (
                "src.problems.slope_stability_3d.support.mesh."
                "ensure_same_mesh_case_hdf5"
            ),
            "tracked_sources": [
                path.as_posix() for path in P3D_MESH_GENERATOR_SOURCES
            ],
        }
    ):
        raise CampaignError("publication mesh manifest identity is invalid")
    bindings: dict[str, dict[str, Any]] = {}
    for relative, degree in expected_paths.items():
        record = payload["files"][relative]
        if not isinstance(record, dict) or set(record) != {
            "bytes",
            "constraint_variant",
            "element_degree",
            "mesh_name",
            "same_mesh_hdf5_schema_version",
            "sha256",
        }:
            raise CampaignError(f"publication mesh record is malformed: {relative}")
        path = REPO_ROOT / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or not isinstance(record.get("bytes"), int)
            or path.stat().st_size != int(record["bytes"])
            or record.get("constraint_variant") != "glued_bottom"
            or record.get("element_degree") != degree
            or record.get("mesh_name") != "hetero_ssr_L1"
            or record.get("same_mesh_hdf5_schema_version") != 7
            or not isinstance(record.get("sha256"), str)
            or len(str(record["sha256"])) != 64
            or _sha256_file(path) != record["sha256"]
            or not _same_mesh_hdf5_is_current(
                path,
                mesh_name="hetero_ssr_L1",
                degree=degree,
                constraint_variant="glued_bottom",
            )
        ):
            raise CampaignError(f"publication mesh cache is missing or stale: {relative}")
        bindings[relative] = {
            **record,
            "manifest": P3D_MESH_MANIFEST_PATH.as_posix(),
        }
    return bindings


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise CampaignError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _git_metadata() -> dict[str, Any]:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain=v1", "--untracked-files=all")),
    }


def _package_snapshot() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("h5py", "jax", "mpi4py", "numpy", "petsc4py", "scipy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _safe_row_id(value: str) -> str:
    if not value or any(character not in SAFE_ROW_CHARS for character in value):
        raise CampaignError(f"unsafe row id {value!r}")
    return value


def _relative_to_repo(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise CampaignError(f"required source/input is outside repository: {resolved}") from exc


def _confined(root: Path, path: Path | str) -> Path:
    root = Path(root).resolve()
    candidate = Path(path)
    candidate = candidate if candidate.is_absolute() else root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CampaignError(f"campaign path escapes output root: {candidate}") from exc
    return candidate


def _float_id(value: float) -> str:
    return f"{float(value):.0e}".replace("+", "").replace("-", "m")


def _input_hashes(paths: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in paths:
        path = (REPO_ROOT / relative).resolve()
        if not path.is_file():
            raise CampaignError(f"required source/input is missing: {relative}")
        result[_relative_to_repo(path)] = _sha256_file(path)
    return dict(sorted(result.items()))


def _base_environment() -> dict[str, str]:
    return {
        "BLIS_NUM_THREADS": "1",
        "FNE_SKIP_REORDERED_WARMUP": "1",
        "JAX_ENABLE_X64": "True",
        "JAX_PLATFORMS": "cpu",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "VECLIB_MAXIMUM_THREADS": "1",
        "XLA_FLAGS": (
            "--xla_cpu_multi_thread_eigen=false "
            "intra_op_parallelism_threads=1 "
            "--xla_force_host_platform_device_count=1"
        ),
    }


def _local_row(
    *,
    row_id: str,
    family: str,
    group_id: str,
    scope: str,
    parameters: Mapping[str, Any],
    command: Sequence[str],
    outputs: Sequence[Path],
    reference: bool,
) -> dict[str, Any]:
    return {
        "row_id": _safe_row_id(row_id),
        "family": str(family),
        "group_id": str(group_id),
        "execution_class": "required_local",
        "scientific_scope": str(scope),
        "parameters": dict(parameters),
        "reference_row": bool(reference),
        "command": [str(value) for value in command],
        "environment": _base_environment(),
        "expected_outputs": [str(Path(value).resolve()) for value in outputs],
    }


def _deferred_row(
    *,
    row_id: str,
    family: str,
    group_id: str,
    scope: str,
    parameters: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "row_id": _safe_row_id(row_id),
        "family": str(family),
        "group_id": str(group_id),
        "execution_class": "deferred_cluster_computation",
        "scientific_scope": str(scope),
        "parameters": dict(parameters),
        "reference_row": False,
        "command": None,
        "environment": {},
        "expected_outputs": [],
        "censor": {
            "status": "censored",
            "reason": str(reason),
            "timing_admissible": False,
            "accuracy_claim_admissible": False,
        },
    }


def _gl_rows(output_root: Path, python: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in GL_LEVELS:
        for target in GL_RESIDUAL_TARGETS:
            tolerance_id = _float_id(target)
            row_id = f"gl_l{level}_residual_{tolerance_id}"
            row_root = output_root / "raw" / "gl" / row_id
            result = row_root / "result.json"
            state = row_root / "state.npz"
            command = [
                python,
                TRUST_RUNNER_PATH.as_posix(),
                "--problem",
                "gl",
                "--backend",
                "element",
                "--level",
                str(level),
                "--out",
                str(result),
                "--state-out",
                str(state),
                "--profile",
                "reference",
                "--ksp-type",
                "gmres",
                "--pc-type",
                "hypre",
                "--ksp-rtol",
                "1e-10",
                "--ksp-max-it",
                "1000",
                "--gamg-threshold",
                "0.05",
                "--gamg-agg-nsmooths",
                "1",
                "--element-reorder-mode",
                "block_xyz",
                "--local-hessian-mode",
                "element",
                "--convergence-metric",
                "lumped_l2",
                "--tolf",
                "1e300",
                "--tolg",
                "0",
                "--tolg-rel",
                f"{target:.0e}",
                "--tolx-rel",
                "1e300",
                "--tolx-abs",
                "1e300",
                "--maxit",
                "100",
                "--line-search",
                "armijo",
                "--use-trust-region",
                "--trust-radius-init",
                "1",
                "--trust-radius-min",
                "1e-8",
                "--trust-radius-max",
                "1e6",
                "--trust-shrink",
                "0.5",
                "--trust-expand",
                "1.5",
                "--trust-eta-shrink",
                "0.05",
                "--trust-eta-expand",
                "0.75",
                "--trust-max-reject",
                "6",
                "--no-retry-on-failure",
                "--save-history",
                "--save-linear-timing",
                "--quiet",
            ]
            rows.append(
                _local_row(
                    row_id=row_id,
                    family="ginzburg_landau",
                    group_id=f"gl_l{level}",
                    scope="deterministic_nonlinear_endpoint_with_lumped_l2_stopping",
                    parameters={
                        "mesh_level": level,
                        "relative_dual_residual_target": target,
                        "linear_ksp_rtol": 1.0e-10,
                        "globalization": "rho_based_trust_region_with_armijo_model_steps",
                        "nonprimary_energy_and_correction_gates": "finite_nonbinding_caps",
                    },
                    command=command,
                    outputs=(result, state),
                    reference=target == min(GL_RESIDUAL_TARGETS),
                )
            )
    return rows


def _he_rows(output_root: Path, python: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in HE_LEVELS:
        for tolerance in HE_RIESZ_RTOLS:
            tolerance_id = _float_id(tolerance)
            row_id = f"he_l{level}_riesz_{tolerance_id}"
            row_root = output_root / "raw" / "he" / row_id
            result = row_root / "result.json"
            command = [
                python,
                TRUST_RUNNER_PATH.as_posix(),
                "--problem",
                "he",
                "--backend",
                "element",
                "--level",
                str(level),
                "--out",
                str(result),
                "--steps",
                "1",
                "--total-steps",
                "24",
                "--maxit",
                "0",
                "--problem-build-mode",
                "rank_local",
                "--he-mesh-source",
                "procedural",
                "--he-element-degree",
                "1",
                "--distribution-strategy",
                "overlap_p2p",
                "--assembly-backend",
                "coo_local",
                "--element-reorder-mode",
                "block_xyz",
                "--local-hessian-mode",
                "element",
                "--ksp-type",
                "gmres",
                "--pc-type",
                "hypre",
                "--ksp-rtol",
                "1e-10",
                "--ksp-max-it",
                "1000",
                "--convergence-metric",
                "reference_elastic_energy",
                "--riesz-ksp-type",
                "cg",
                "--riesz-pc-type",
                "jacobi",
                "--riesz-ksp-rtol",
                f"{tolerance:.0e}",
                "--riesz-ksp-atol",
                "0",
                "--riesz-ksp-max-it",
                "5000",
                "--riesz-true-residual-rtol",
                "1e-6",
                "--riesz-spd-factor-solver-type",
                "mumps",
                "--riesz-symmetry-tol",
                "1e-12",
                "--no-retry-on-failure",
                "--quiet",
            ]
            rows.append(
                _local_row(
                    row_id=row_id,
                    family="hyperelasticity_reference_riesz",
                    group_id=f"he_l{level}",
                    scope=(
                        "reference_metric_setup_and_terminal_residual_only; "
                        "maxit_zero_is_not_nonlinear_convergence"
                    ),
                    parameters={
                        "mesh_level": level,
                        "riesz_ksp_type": "cg",
                        "riesz_pc_type": "jacobi",
                        "riesz_ksp_norm_type": "unpreconditioned",
                        "riesz_ksp_rtol": tolerance,
                        "riesz_ksp_atol": 0.0,
                        "riesz_ksp_max_it": 5000,
                        "riesz_true_residual_safety_gate": 1.0e-6,
                        "nonlinear_max_iterations": 0,
                    },
                    command=command,
                    outputs=(result,),
                    reference=tolerance == min(HE_RIESZ_RTOLS),
                )
            )
    return rows


def _he_nonlinear_rows(output_root: Path, python: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in HE_LEVELS:
        for target in HE_NONLINEAR_TARGETS:
            tolerance_id = _float_id(target)
            row_id = f"he_l{level}_nonlinear_{tolerance_id}"
            row_root = output_root / "raw" / "he_nonlinear" / row_id
            result = row_root / "result.json"
            state = row_root / "state.npz"
            command = [
                python,
                TRUST_RUNNER_PATH.as_posix(),
                "--problem",
                "he",
                "--backend",
                "element",
                "--level",
                str(level),
                "--out",
                str(result),
                "--state-out",
                str(state),
                "--steps",
                "1",
                "--total-steps",
                "24",
                "--maxit",
                "80",
                "--problem-build-mode",
                "rank_local",
                "--he-mesh-source",
                "procedural",
                "--he-element-degree",
                "1",
                "--distribution-strategy",
                "overlap_p2p",
                "--assembly-backend",
                "coo_local",
                "--element-reorder-mode",
                "block_xyz",
                "--local-hessian-mode",
                "element",
                "--ksp-type",
                "gmres",
                "--pc-type",
                "hypre",
                "--ksp-rtol",
                "1e-8",
                "--ksp-max-it",
                "1000",
                "--convergence-metric",
                "reference_elastic_energy",
                "--riesz-ksp-type",
                "cg",
                "--riesz-pc-type",
                "jacobi",
                "--riesz-ksp-rtol",
                "1e-10",
                "--riesz-ksp-atol",
                "0",
                "--riesz-ksp-max-it",
                "5000",
                "--riesz-true-residual-rtol",
                "1e-8",
                "--riesz-spd-factor-solver-type",
                "mumps",
                "--riesz-symmetry-tol",
                "1e-12",
                "--tolf",
                "1e300",
                "--tolg",
                "0",
                "--tolg-rel",
                f"{target:.0e}",
                "--tolx-rel",
                "1e300",
                "--tolx-abs",
                "1e300",
                "--line-search",
                "armijo",
                "--no-retry-on-failure",
                "--save-history",
                "--save-linear-timing",
                "--quiet",
            ]
            rows.append(
                _local_row(
                    row_id=row_id,
                    family="hyperelasticity_nonlinear_stopping",
                    group_id=f"he_l{level}_nonlinear",
                    scope=(
                        "one_load_step_full_nonlinear_reference_riesz_endpoint; "
                        "same_mesh_state_export_has_no_reference_operator_action"
                    ),
                    parameters={
                        "mesh_level": level,
                        "relative_dual_residual_target": target,
                        "linear_ksp_rtol": 1.0e-8,
                        "riesz_ksp_type": "cg",
                        "riesz_pc_type": "jacobi",
                        "riesz_ksp_norm_type": "unpreconditioned",
                        "riesz_ksp_rtol": 1.0e-10,
                        "riesz_ksp_atol": 0.0,
                        "riesz_ksp_max_it": 5000,
                        "riesz_true_residual_rtol": 1.0e-8,
                        "load_steps": 1,
                        "total_steps": 24,
                    },
                    command=command,
                    outputs=(result, state),
                    reference=target == min(HE_NONLINEAR_TARGETS),
                )
            )
    return rows


def _p3d_rows(
    output_root: Path,
    python: str,
    *,
    p4_policy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for degree in (1, 2, 4):
        if degree == 4 and p4_policy == "deferred_cluster":
            for tolerance in P3D_LINEAR_RTOLS:
                rows.append(
                    _deferred_row(
                        row_id=f"p3d_p4_linear_{_float_id(tolerance)}",
                        family="plasticity3d_fixed_state_linear",
                        group_id="p3d_p4",
                        scope="fixed_elastic_state_linear_tolerance_and_true_residual",
                        parameters={
                            "mesh_name": "hetero_ssr_L1",
                            "element_degree": 4,
                            "ksp_rtol": tolerance,
                        },
                        reason=(
                            "p4_local_resource_feasibility_not_attested; prepare with "
                            "--p4-policy local --confirm-p4-local-feasible to make these "
                            "frozen required-local rows"
                        ),
                    )
                )
            continue
        quadrature = P3D_QUADRATURE[degree]
        for tolerance in P3D_LINEAR_RTOLS:
            tolerance_id = _float_id(tolerance)
            row_id = f"p3d_p{degree}_linear_{tolerance_id}"
            row_root = output_root / "raw" / "p3d" / row_id
            result = row_root / "result.json"
            state = row_root / "state_and_correction.npz"
            command = [
                python,
                RUNNER_PATH.as_posix(),
                "p3d-fixed-state",
                "--degree",
                str(degree),
                "--quadrature-rule",
                quadrature,
                "--state-amplitude",
                "0.0002",
                "--ksp-type",
                "gmres",
                "--pc-type",
                "hypre",
                "--ksp-rtol",
                f"{tolerance:.0e}",
                "--ksp-max-it",
                "4000",
                "--true-residual-factor",
                "20",
                "--output",
                str(result),
                "--state-out",
                str(state),
            ]
            rows.append(
                _local_row(
                    row_id=row_id,
                    family="plasticity3d_fixed_state_linear",
                    group_id=f"p3d_p{degree}",
                    scope=(
                        "fixed_elastic_state_single_linear_system_only; "
                        "not_a_nonlinear_endpoint"
                    ),
                    parameters={
                        "mesh_name": "hetero_ssr_L1",
                        "element_degree": degree,
                        "quadrature_rule_id": quadrature,
                        "state_label": "elastic",
                        "state_amplitude": 0.0002,
                        "ksp_type": "gmres",
                        "pc_type": "hypre",
                        "ksp_rtol": tolerance,
                        "true_residual_factor": 20.0,
                    },
                    command=command,
                    outputs=(result, state),
                    reference=tolerance == min(P3D_LINEAR_RTOLS),
                )
            )
    return rows


def _p3d_nonlinear_rows(output_root: Path, python: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for degree in (1, 2):
        quadrature = P3D_QUADRATURE[degree]
        for target in P3D_NONLINEAR_TARGETS:
            tolerance_id = _float_id(target)
            row_id = f"p3d_p{degree}_nonlinear_{tolerance_id}"
            row_root = output_root / "raw" / "p3d_nonlinear" / row_id
            result = row_root / "result.json"
            state = row_root / "state.npz"
            work = row_root / "work"
            command = [
                python,
                P3D_BACKEND_PATH.as_posix(),
                "--assembly-backend",
                "local",
                "--solver-backend",
                "local",
                "--out-dir",
                str(work),
                "--output-json",
                str(result),
                "--state-out",
                str(state),
                "--mesh-name",
                "hetero_ssr_L1",
                "--elem-degree",
                str(degree),
                "--quadrature-rule",
                quadrature,
                "--constraint-variant",
                "glued_bottom",
                "--lambda-target",
                "1.55",
                "--ksp-rtol",
                "1e-8",
                "--ksp-max-it",
                "1000",
                "--convergence-mode",
                "gradient_only",
                "--grad-stop-tol",
                "0",
                "--grad-stop-rtol",
                f"{target:.0e}",
                "--stop-tol",
                f"{target:.0e}",
                "--convergence-metric",
                "reference_elastic_energy",
                "--riesz-ksp-type",
                "cg",
                "--riesz-pc-type",
                "jacobi",
                "--riesz-ksp-rtol",
                "1e-10",
                "--riesz-ksp-atol",
                "0",
                "--riesz-ksp-max-it",
                "5000",
                "--riesz-true-residual-rtol",
                "1e-8",
                "--riesz-spd-factor-solver-type",
                "mumps",
                "--riesz-symmetry-tol",
                "1e-12",
                "--maxit",
                "80",
                "--line-search",
                "armijo",
                "--armijo-alpha0",
                "1",
                "--armijo-c1",
                "1e-4",
                "--armijo-shrink",
                "0.5",
                "--armijo-max-ls",
                "40",
            ]
            rows.append(
                _local_row(
                    row_id=row_id,
                    family="plasticity3d_nonlinear_stopping",
                    group_id=f"p3d_p{degree}_nonlinear",
                    scope="full_nonlinear_reference_riesz_endpoint_on_one_serial_rank",
                    parameters={
                        "mesh_name": "hetero_ssr_L1",
                        "element_degree": degree,
                        "quadrature_rule_id": quadrature,
                        "relative_dual_residual_target": target,
                        "linear_ksp_rtol": 1.0e-8,
                        "riesz_ksp_type": "cg",
                        "riesz_pc_type": "jacobi",
                        "riesz_ksp_norm_type": "unpreconditioned",
                        "riesz_ksp_rtol": 1.0e-10,
                        "riesz_ksp_atol": 0.0,
                        "riesz_ksp_max_it": 5000,
                        "riesz_true_residual_rtol": 1.0e-8,
                    },
                    command=command,
                    outputs=(result, state),
                    reference=target == min(P3D_NONLINEAR_TARGETS),
                )
            )
    return rows


def _cluster_censors() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in P3D_NONLINEAR_TARGETS:
        rows.append(
            _deferred_row(
                row_id=f"p3d_p4_nonlinear_{_float_id(target)}_cluster",
                family="plasticity3d_nonlinear_stopping",
                group_id="p3d_p4_nonlinear_cluster",
                scope="full_nonlinear_reference_riesz_endpoint",
                parameters={
                    "mesh_name": "hetero_ssr_L1",
                    "element_degree": 4,
                    "relative_residual_target": target,
                },
                reason=(
                    "P4 full nonlinear reference-Riesz endpoints exceed the default local "
                    "resource scope and remain reviewed parallel-cluster computations"
                ),
            )
        )
    for family in ("ginzburg_landau", "hyperelasticity", "plasticity3d"):
        rows.append(
            _deferred_row(
                row_id=f"{family}_mpi_consistency_cluster",
                family=f"{family}_mpi_consistency",
                group_id=f"{family}_mpi_cluster",
                scope="publication_rank_count_consistency",
                parameters={"rank_counts": "publication_rank_counts_from_dependent_protocols"},
                reason=(
                    "multi-rank consistency is outside the serial local tranche and remains "
                    "a parallel-cluster computation"
                ),
            )
        )
    return rows


def build_plan(
    output_root: Path,
    *,
    run_kind: str,
    allow_dirty: bool,
    p4_policy: str,
    confirm_p4_local_feasible: bool,
) -> dict[str, Any]:
    """Build an immutable command plan without creating files."""
    output_root = Path(output_root).resolve()
    git = _git_metadata()
    if run_kind not in {"publication", "diagnostic"}:
        raise CampaignError("run_kind must be publication or diagnostic")
    if run_kind == "publication" and allow_dirty:
        raise CampaignError("publication preparation cannot use --allow-dirty")
    if bool(git["dirty"]) and not allow_dirty:
        raise CampaignError("source worktree is dirty; use diagnostic --allow-dirty or clean it")
    if run_kind == "publication" and bool(git["dirty"]):
        raise CampaignError("publication preparation requires a clean worktree")
    if p4_policy == "local" and not confirm_p4_local_feasible:
        raise CampaignError(
            "--p4-policy local requires --confirm-p4-local-feasible before commands are frozen"
        )
    if p4_policy != "local" and confirm_p4_local_feasible:
        raise CampaignError("--confirm-p4-local-feasible is valid only with --p4-policy local")
    if p4_policy not in {"local", "deferred_cluster"}:
        raise CampaignError("p4_policy must be local or deferred_cluster")

    # Keep the invoked virtual-environment path. Resolving its interpreter
    # symlink can bypass the venv prefix and silently lose mpi4py/petsc4py.
    python = str(Path(sys.executable).absolute())
    rows = [
        *_gl_rows(output_root, python),
        *_he_rows(output_root, python),
        *_he_nonlinear_rows(output_root, python),
        *_p3d_rows(output_root, python, p4_policy=p4_policy),
        *_p3d_nonlinear_rows(output_root, python),
        *_cluster_censors(),
    ]
    row_ids = [str(row["row_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise CampaignError("campaign row ids are not unique")

    source_paths = (
        RUNNER_PATH,
        TRUST_RUNNER_PATH,
        P3D_BACKEND_PATH,
        P3D_ROUTE_PATH,
        Path("src/core/cli/threading.py"),
        Path("src/core/petsc/metrics.py"),
        Path("src/core/petsc/scalar_problem_driver.py"),
        Path("src/problems/ginzburg_landau/jax_petsc/solver.py"),
        Path("src/problems/hyperelasticity/jax_petsc/solver.py"),
        Path("src/problems/slope_stability_3d/jax_petsc/solver.py"),
        Path("src/problems/slope_stability_3d/support/fixed_state.py"),
    )
    input_paths = [
        PROTOCOL_PATH,
        *(Path(f"data/meshes/GinzburgLandau/GL_level{level}.h5") for level in GL_LEVELS),
        P3D_MESH_MANIFEST_PATH,
        *P3D_MESH_GENERATOR_SOURCES,
    ]
    manifested_meshes = _manifested_p3d_meshes()
    required_local = sum(row["execution_class"] == "required_local" for row in rows)
    deferred = len(rows) - required_local
    return {
        "schema_id": PLAN_SCHEMA_ID,
        "schema_version": PLAN_SCHEMA_VERSION,
        "experiment_id": "EXP-STOP-001",
        "campaign_id": "exp_stop_001_local_calibration_v1",
        "created_utc": _utc_now(),
        "run_kind": run_kind,
        "publication_evidence_candidate": run_kind == "publication",
        "output_root": str(output_root),
        "source": {
            "commit": str(git["commit"]),
            "dirty": bool(git["dirty"]),
            "relevant_file_hashes": _input_hashes(source_paths),
        },
        "inputs": {
            "file_hashes": _input_hashes(input_paths),
            "manifested_file_hashes": manifested_meshes,
            "procedural_he_mesh": {
                "levels": list(HE_LEVELS),
                "source": "rank_local_procedural_he_p1_mesh_builder",
                "note": "no HDF5 mesh is consumed by the frozen HE commands",
            },
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": python,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": _package_snapshot(),
            "command_environment": _base_environment(),
        },
        "policies": {
            "p4_fixed_state": p4_policy,
            "p4_local_feasibility_attested": bool(confirm_p4_local_feasible),
            "fresh_output_root_required": True,
            "command_mutation_forbidden_after_prepare": True,
            "timing_claims_admissible": False,
            "analysis_contract": {
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
            },
        },
        "row_counts": {
            "total": len(rows),
            "required_local": required_local,
            "deferred_cluster_computation": deferred,
        },
        "claim_boundary": {
            "local_completion_can_establish": [
                "deterministic GL same-mesh endpoint sensitivity on two mesh levels",
                "HE reference-Riesz setup and terminal norm-solve sensitivity on two levels",
                "HE L1/L2 one-load-step nonlinear endpoint sensitivity",
                "P1/P2 fixed-state Plasticity3D linear true-residual sensitivity",
                "P1/P2 Plasticity3D full nonlinear endpoint sensitivity",
                "P4 fixed-state sensitivity only when locally attested in this frozen plan",
            ],
            "local_completion_cannot_establish": [
                "HyperElasticity behavior beyond the frozen one-load-step L1/L2 cases",
                "full nonlinear P4 Plasticity3D convergence",
                "publication-rank MPI consistency",
                "timing or scaling claims",
                "a terminal PASS for the complete EXP-STOP-001 protocol",
            ],
        },
        "rows": rows,
    }


def prepare_campaign(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise CampaignError(f"fresh output root already exists: {output_root}")
    plan = build_plan(
        output_root,
        run_kind=str(args.run_kind),
        allow_dirty=bool(args.allow_dirty),
        p4_policy=str(args.p4_policy),
        confirm_p4_local_feasible=bool(args.confirm_p4_local_feasible),
    )
    output_root.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "raw", "receipts"):
        (output_root / name).mkdir()
    plan_path = output_root / "plan.json"
    _atomic_json(plan_path, plan)
    return plan_path


def _load_plan(plan_path: Path) -> tuple[dict[str, Any], Path]:
    plan_path = Path(plan_path).resolve()
    plan = _read_json(plan_path)
    if plan.get("schema_id") != PLAN_SCHEMA_ID or plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise CampaignError("invalid EXP-STOP-001 local plan schema")
    output_root = Path(str(plan.get("output_root", ""))).resolve()
    if plan_path != output_root / "plan.json":
        raise CampaignError("plan path does not equal the frozen output_root/plan.json")
    rows = plan.get("rows")
    if not isinstance(rows, list) or not rows:
        raise CampaignError("plan has no rows")
    row_ids = [row.get("row_id") for row in rows if isinstance(row, Mapping)]
    if len(row_ids) != len(rows) or len(row_ids) != len(set(row_ids)):
        raise CampaignError("plan rows are malformed or duplicated")
    return plan, output_root


def _verify_frozen_plan_design(
    plan: Mapping[str, Any],
    *,
    output_root: Path,
) -> None:
    policies = plan.get("policies")
    source = plan.get("source")
    if not isinstance(policies, Mapping) or not isinstance(source, Mapping):
        raise CampaignError("plan policy or source identity is missing")
    run_kind = str(plan.get("run_kind", ""))
    p4_policy = str(policies.get("p4_fixed_state", ""))
    p4_attested = policies.get("p4_local_feasibility_attested") is True
    canonical = build_plan(
        output_root,
        run_kind=run_kind,
        allow_dirty=run_kind == "diagnostic" and bool(source.get("dirty")),
        p4_policy=p4_policy,
        confirm_p4_local_feasible=p4_attested,
    )
    for key in ("rows", "row_counts", "policies", "claim_boundary"):
        if plan.get(key) != canonical.get(key):
            raise CampaignError(f"frozen plan {key} differs from the canonical design")
    canonical_source = canonical["source"]["relevant_file_hashes"]
    if source.get("relevant_file_hashes") != canonical_source:
        raise CampaignError("frozen plan source inventory differs from the canonical design")
    if plan.get("inputs") != canonical.get("inputs"):
        raise CampaignError("frozen plan input inventory differs from the canonical design")


def _verify_source_identity(plan: Mapping[str, Any]) -> None:
    git = _git_metadata()
    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise CampaignError("plan source identity is missing")
    if str(git["commit"]) != str(source.get("commit")):
        raise CampaignError("current HEAD differs from the frozen experiment commit")
    if plan.get("run_kind") == "publication" and bool(git["dirty"]):
        raise CampaignError("publication execution requires a clean worktree")
    for section_name, hashes in (
        ("source", source.get("relevant_file_hashes")),
        ("input", (plan.get("inputs") or {}).get("file_hashes")),
    ):
        if not isinstance(hashes, Mapping):
            raise CampaignError(f"plan {section_name} hashes are missing")
        for relative, expected in hashes.items():
            path = (REPO_ROOT / str(relative)).resolve()
            if not path.is_file() or _sha256_file(path) != expected:
                raise CampaignError(f"frozen {section_name} changed or is missing: {relative}")


def _row_by_id(plan: Mapping[str, Any], row_id: str) -> dict[str, Any]:
    matches = [row for row in plan["rows"] if row.get("row_id") == row_id]
    if len(matches) != 1:
        raise CampaignError(f"unknown or duplicate row id {row_id!r}")
    return dict(matches[0])


def _verify_completed_outputs(row: Mapping[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for raw in row.get("expected_outputs", []):
        path = Path(str(raw)).resolve()
        if not path.is_file():
            raise CampaignError(f"row {row['row_id']} did not create {path}")
        hashes[str(path)] = _sha256_file(path)
    json_outputs = [Path(path) for path in hashes if Path(path).suffix == ".json"]
    if len(json_outputs) != 1:
        raise CampaignError(f"row {row['row_id']} must create exactly one JSON result")
    _read_json(json_outputs[0])
    return hashes


def execute_row(
    *,
    plan_path: Path,
    row_id: str,
    timeout_s: float | None,
) -> Path:
    plan, output_root = _load_plan(plan_path)
    _verify_source_identity(plan)
    _verify_frozen_plan_design(plan, output_root=output_root)
    row = _row_by_id(plan, row_id)
    if row.get("execution_class") != "required_local":
        raise CampaignError(f"row {row_id} is a frozen cluster-deferred censor")
    receipt_path = output_root / "receipts" / f"{_safe_row_id(row_id)}.json"
    if receipt_path.exists():
        raise CampaignError(f"receipt already exists; row cannot be overwritten: {row_id}")
    for output in row.get("expected_outputs", []):
        if Path(str(output)).exists():
            raise CampaignError(f"row output already exists and cannot be overwritten: {output}")

    command = [str(value) for value in row.get("command", [])]
    if not command:
        raise CampaignError(f"row {row_id} has no frozen command")
    environment = dict(os.environ)
    environment.update({str(key): str(value) for key, value in row.get("environment", {}).items()})
    log_root = output_root / "logs" / _safe_row_id(row_id)
    log_root.mkdir(parents=True, exist_ok=False)
    started_utc = _utc_now()
    started = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        returncode = int(completed.returncode)
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -1
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "") + f"\nexecution timed out after {timeout_s} seconds"
    (log_root / "stdout.log").write_text(stdout, encoding="utf-8")
    (log_root / "stderr.log").write_text(stderr, encoding="utf-8")

    output_hashes: dict[str, str] = {}
    verification_error: str | None = None
    if returncode == 0 and not timed_out:
        try:
            output_hashes = _verify_completed_outputs(row)
        except Exception as exc:
            verification_error = f"{type(exc).__name__}: {exc}"
    status = (
        "completed"
        if returncode == 0 and not timed_out and verification_error is None
        else "failed"
    )
    receipt = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "experiment_id": "EXP-STOP-001",
        "campaign_id": plan["campaign_id"],
        "row_id": row_id,
        "status": status,
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "wall_time_s": float(time.perf_counter() - started),
        "host": socket.gethostname(),
        "plan_path": str(Path(plan_path).resolve()),
        "plan_sha256": _sha256_file(Path(plan_path).resolve()),
        "source_commit": plan["source"]["commit"],
        "run_kind": plan["run_kind"],
        "command": command,
        "environment_overrides": dict(row.get("environment", {})),
        "returncode": returncode,
        "timed_out": timed_out,
        "verification_error": verification_error,
        "output_hashes": output_hashes,
        "logs": {
            "stdout": str(log_root / "stdout.log"),
            "stderr": str(log_root / "stderr.log"),
        },
    }
    _atomic_json(receipt_path, receipt)
    if status != "completed":
        raise CampaignError(f"row {row_id} failed; inspect {receipt_path}")
    return receipt_path


def execute_campaign(args: argparse.Namespace) -> list[Path]:
    plan, _output_root = _load_plan(Path(args.plan))
    if args.timeout_s is not None and float(args.timeout_s) <= 0.0:
        raise CampaignError("--timeout-s must be positive")
    if args.all_local:
        if not args.confirm_all_local_execution:
            raise CampaignError("--all-local requires --confirm-all-local-execution")
        row_ids = [
            str(row["row_id"])
            for row in plan["rows"]
            if row.get("execution_class") == "required_local"
        ]
    else:
        if not args.row:
            raise CampaignError("select --row ROW_ID or --all-local")
        row_ids = [str(args.row)]
    return [
        execute_row(
            plan_path=Path(args.plan),
            row_id=row_id,
            timeout_s=args.timeout_s,
        )
        for row_id in row_ids
    ]


def _receipt_for_row(
    *, plan_path: Path, output_root: Path, row: Mapping[str, Any]
) -> tuple[str, dict[str, Any] | None, list[str]]:
    receipt_path = output_root / "receipts" / f"{row['row_id']}.json"
    if not receipt_path.is_file():
        return "missing", None, ["required local receipt is missing"]
    errors: list[str] = []
    receipt = _read_json(receipt_path)
    if receipt.get("schema_id") != RECEIPT_SCHEMA_ID or receipt.get(
        "schema_version"
    ) != RECEIPT_SCHEMA_VERSION:
        errors.append("receipt schema is invalid")
    if receipt.get("row_id") != row.get("row_id"):
        errors.append("receipt row id differs from plan")
    if receipt.get("plan_sha256") != _sha256_file(plan_path):
        errors.append("receipt plan hash differs from current frozen plan")
    if receipt.get("command") != row.get("command"):
        errors.append("receipt command differs from frozen command")
    if errors:
        return "invalid", receipt, errors
    if receipt.get("status") == "failed":
        reason = (
            "frozen local command failed or timed out; retained as an unclassified "
            "runtime censor and not as convergence evidence"
        )
        return "runtime_censored", receipt, [reason]
    if receipt.get("status") != "completed":
        return "invalid", receipt, ["receipt has an unsupported terminal status"]
    expected_outputs = {str(Path(path).resolve()) for path in row.get("expected_outputs", [])}
    output_hashes = receipt.get("output_hashes")
    if not isinstance(output_hashes, Mapping) or set(output_hashes) != expected_outputs:
        errors.append("receipt output hash inventory differs from frozen outputs")
    else:
        for raw_path, expected_hash in output_hashes.items():
            path = Path(str(raw_path)).resolve()
            if not path.is_file() or _sha256_file(path) != expected_hash:
                errors.append(f"output hash mismatch: {path}")
    return ("invalid" if errors else "completed"), receipt, errors


def _result_json_path(row: Mapping[str, Any]) -> Path:
    matches = [Path(path) for path in row.get("expected_outputs", []) if Path(path).suffix == ".json"]
    if len(matches) != 1:
        raise CampaignError(f"row {row['row_id']} does not identify one result JSON")
    return matches[0].resolve()


def _state_npz_path(row: Mapping[str, Any]) -> Path:
    matches = [Path(path) for path in row.get("expected_outputs", []) if Path(path).suffix == ".npz"]
    if len(matches) != 1:
        raise CampaignError(f"row {row['row_id']} does not identify one state NPZ")
    return matches[0].resolve()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignError(f"{label} is nonfinite")
    return result


def _require_riesz_solver_contract(
    row: Mapping[str, Any],
    *,
    metric: Mapping[str, Any],
    norm_solve: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = row.get("parameters")
    if not isinstance(parameters, Mapping):
        raise CampaignError(f"row {row.get('row_id')} has no Riesz parameters")
    gate_keys = [
        key
        for key in (
            "riesz_true_residual_rtol",
            "riesz_true_residual_safety_gate",
        )
        if key in parameters
    ]
    if len(gate_keys) != 1:
        raise CampaignError(
            f"row {row.get('row_id')} must freeze exactly one Riesz true-residual gate"
        )
    expected_strings = {
        "ksp_type": str(parameters.get("riesz_ksp_type", "")),
        "pc_type": str(parameters.get("riesz_pc_type", "")),
        "norm_type": str(parameters.get("riesz_ksp_norm_type", "")),
    }
    string_observations = {
        "metric.ksp_type": metric.get("ksp_type"),
        "metric.pc_type": metric.get("pc_type"),
        "metric.requested_norm_type": metric.get("requested_norm_type"),
        "metric.effective_norm_type": metric.get("effective_norm_type"),
        "solve.ksp_type": norm_solve.get("ksp_type"),
        "solve.pc_type": norm_solve.get("pc_type"),
        "solve.requested_norm_type": norm_solve.get("requested_norm_type"),
        "solve.effective_norm_type": norm_solve.get("effective_norm_type"),
        "solve.reported_residual_norm_type": norm_solve.get(
            "reported_residual_norm_type"
        ),
    }
    expected_by_field = {
        "metric.ksp_type": expected_strings["ksp_type"],
        "metric.pc_type": expected_strings["pc_type"],
        "metric.requested_norm_type": expected_strings["norm_type"],
        "metric.effective_norm_type": expected_strings["norm_type"],
        "solve.ksp_type": expected_strings["ksp_type"],
        "solve.pc_type": expected_strings["pc_type"],
        "solve.requested_norm_type": expected_strings["norm_type"],
        "solve.effective_norm_type": expected_strings["norm_type"],
        "solve.reported_residual_norm_type": expected_strings["norm_type"],
    }
    string_mismatches = [
        name
        for name, observed in string_observations.items()
        if observed != expected_by_field[name]
    ]
    if string_mismatches:
        raise CampaignError(
            f"row {row.get('row_id')} Riesz solver provenance differs from the "
            f"frozen plan: {', '.join(string_mismatches)}"
        )
    if metric.get("set_from_petsc_options") is not False:
        raise CampaignError(
            f"row {row.get('row_id')} allowed PETSc options to mutate the Riesz solve"
        )

    expected_rtol = _finite_number(
        parameters.get("riesz_ksp_rtol"), "frozen Riesz relative tolerance"
    )
    expected_atol = _finite_number(
        parameters.get("riesz_ksp_atol"), "frozen Riesz absolute tolerance"
    )
    expected_max_it = int(parameters.get("riesz_ksp_max_it", 0))
    if expected_max_it <= 0:
        raise CampaignError(f"row {row.get('row_id')} has an invalid Riesz iteration cap")
    expected_gate = _finite_number(
        parameters.get(gate_keys[0]), "frozen Riesz true-residual gate"
    )
    numeric_expectations = {
        "metric.requested_rtol": expected_rtol,
        "metric.effective_rtol": expected_rtol,
        "solve.requested_rtol": expected_rtol,
        "solve.effective_rtol": expected_rtol,
        "metric.requested_atol": expected_atol,
        "metric.effective_atol": expected_atol,
        "solve.requested_atol": expected_atol,
        "solve.effective_atol": expected_atol,
        "metric.requested_max_it": float(expected_max_it),
        "metric.effective_max_it": float(expected_max_it),
        "solve.requested_max_it": float(expected_max_it),
        "solve.effective_max_it": float(expected_max_it),
        "metric.true_residual_rtol_gate": expected_gate,
        "solve.true_residual_rtol_gate": expected_gate,
    }
    numeric_sources = {
        "metric.requested_rtol": metric.get("requested_rtol"),
        "metric.effective_rtol": metric.get("effective_rtol"),
        "solve.requested_rtol": norm_solve.get("requested_rtol"),
        "solve.effective_rtol": norm_solve.get("effective_rtol"),
        "metric.requested_atol": metric.get("requested_atol"),
        "metric.effective_atol": metric.get("effective_atol"),
        "solve.requested_atol": norm_solve.get("requested_atol"),
        "solve.effective_atol": norm_solve.get("effective_atol"),
        "metric.requested_max_it": metric.get("requested_max_it"),
        "metric.effective_max_it": metric.get("effective_max_it"),
        "solve.requested_max_it": norm_solve.get("requested_max_it"),
        "solve.effective_max_it": norm_solve.get("effective_max_it"),
        "metric.true_residual_rtol_gate": metric.get("true_residual_rtol_gate"),
        "solve.true_residual_rtol_gate": norm_solve.get("true_residual_rtol_gate"),
    }
    numeric_mismatches: list[str] = []
    for name, expected in numeric_expectations.items():
        observed = _finite_number(numeric_sources[name], name)
        if observed != expected:
            numeric_mismatches.append(name)
    if numeric_mismatches:
        raise CampaignError(
            f"row {row.get('row_id')} Riesz tolerances differ from the frozen plan: "
            f"{', '.join(numeric_mismatches)}"
        )
    return {
        "ksp_type": expected_strings["ksp_type"],
        "pc_type": expected_strings["pc_type"],
        "norm_type": expected_strings["norm_type"],
        "rtol": expected_rtol,
        "atol": expected_atol,
        "max_it": expected_max_it,
        "true_residual_rtol_gate": expected_gate,
    }


def _gl_endpoint(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = _read_json(_result_json_path(row))
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise CampaignError("GL result object is missing")
    steps = result.get("steps")
    if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], Mapping):
        raise CampaignError("GL result must contain exactly one step")
    step = steps[0]
    convergence = step.get("convergence")
    if not isinstance(convergence, Mapping):
        raise CampaignError("GL convergence payload is missing")
    state_path = _state_npz_path(row)
    with np.load(state_path, allow_pickle=False) as state:
        state_sha = _sha256_array(np.asarray(state["u"], dtype=np.float64))
    metadata = result.get("metadata")
    configuration = metadata.get("convergence") if isinstance(metadata, Mapping) else None
    selection = configuration.get("selection") if isinstance(configuration, Mapping) else None
    status = "endpoint_admitted" if step.get("success") is True and selection == "lumped_l2" else "censored_solver_nonconvergence"
    return {
        "status": status,
        "message": str(step.get("message", "")),
        "energy": _finite_number(step.get("energy"), "GL energy"),
        "dual_residual_relative": _finite_number(
            convergence.get("dual_residual_relative"), "GL relative dual residual"
        ),
        "correction_norm": _finite_number(
            convergence.get("correction_norm"), "GL correction norm"
        ),
        "relative_correction": _finite_number(
            convergence.get("relative_correction"), "GL relative correction"
        ),
        "state_sha256": state_sha,
        "state_file_sha256": _sha256_file(state_path),
    }


def _he_endpoint(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = _read_json(_result_json_path(row))
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise CampaignError("HE result object is missing")
    steps = result.get("steps")
    if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], Mapping):
        raise CampaignError("HE result must contain exactly one setup-only step")
    convergence = steps[0].get("convergence")
    if not isinstance(convergence, Mapping):
        raise CampaignError("HE terminal convergence payload is missing")
    metric = convergence.get("metric")
    metadata = convergence.get("dual_residual_metadata")
    if not isinstance(metric, Mapping) or not isinstance(metadata, Mapping):
        raise CampaignError("HE metric or norm-solve metadata is missing")
    solver_contract = _require_riesz_solver_contract(
        row,
        metric=metric,
        norm_solve=metadata,
    )
    provenance = metric.get("provenance")
    certificate = provenance.get("spd_certificate") if isinstance(provenance, Mapping) else None
    certified_spd = isinstance(certificate, Mapping) and certificate.get("certified_spd") is True
    true_residual = _finite_number(
        metadata.get("relative_true_residual"), "HE Riesz true residual"
    )
    gate = _finite_number(metadata.get("true_residual_rtol_gate"), "HE true-residual gate")
    status = (
        "reference_metric_check_admitted"
        if certified_spd and int(metadata.get("reason", 0)) > 0 and true_residual <= gate
        else "reference_metric_check_failed"
    )
    return {
        "status": status,
        "scope": "maxit_zero_reference_metric_check_not_nonlinear_convergence",
        "energy": _finite_number(steps[0].get("energy"), "HE setup energy"),
        "dual_residual_norm": _finite_number(
            convergence.get("dual_residual_norm"), "HE dual residual norm"
        ),
        "dual_residual_relative": _finite_number(
            convergence.get("dual_residual_relative"), "HE relative dual residual"
        ),
        "state_scale": _finite_number(convergence.get("state_scale"), "HE state scale"),
        "riesz_iterations": int(metadata.get("iterations", 0)),
        "riesz_reason": int(metadata.get("reason", 0)),
        "relative_true_residual": true_residual,
        "true_residual_rtol_gate": gate,
        "riesz_solver_contract": solver_contract,
        "certified_spd": certified_spd,
    }


def _he_nonlinear_endpoint(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = _read_json(_result_json_path(row))
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise CampaignError("HE nonlinear result object is missing")
    steps = result.get("steps")
    if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], Mapping):
        raise CampaignError("HE nonlinear result must contain exactly one load step")
    step = steps[0]
    convergence = step.get("convergence")
    if not isinstance(convergence, Mapping):
        raise CampaignError("HE nonlinear convergence payload is missing")
    metadata = convergence.get("dual_residual_metadata")
    metric = convergence.get("metric")
    if not isinstance(metadata, Mapping) or not isinstance(metric, Mapping):
        raise CampaignError("HE nonlinear Riesz metadata is missing")
    solver_contract = _require_riesz_solver_contract(
        row,
        metric=metric,
        norm_solve=metadata,
    )
    provenance = metric.get("provenance")
    certificate = provenance.get("spd_certificate") if isinstance(provenance, Mapping) else None
    certified_spd = isinstance(certificate, Mapping) and certificate.get("certified_spd") is True
    true_residual = _finite_number(
        metadata.get("relative_true_residual"), "HE nonlinear Riesz true residual"
    )
    true_gate = _finite_number(
        metadata.get("true_residual_rtol_gate"), "HE nonlinear Riesz true-residual gate"
    )
    target = float(row["parameters"]["relative_dual_residual_target"])
    relative_dual = _finite_number(
        convergence.get("dual_residual_relative"), "HE nonlinear relative dual residual"
    )
    state_path = _state_npz_path(row)
    with np.load(state_path, allow_pickle=False) as state:
        required = {
            "displacement",
            "free_deformation_original",
            "reference_elastic_action",
        }
        missing = required - set(state.files)
        if missing:
            raise CampaignError(
                "HE nonlinear state export is missing reference-Riesz arrays: "
                f"{sorted(missing)}"
            )
        displacement = np.asarray(state["displacement"], dtype=np.float64)
        free_deformation = np.asarray(
            state["free_deformation_original"], dtype=np.float64
        ).reshape(-1)
        reference_action = np.asarray(
            state["reference_elastic_action"], dtype=np.float64
        ).reshape(-1)
        if (
            free_deformation.shape != reference_action.shape
            or free_deformation.size == 0
            or not np.all(np.isfinite(free_deformation))
            or not np.all(np.isfinite(reference_action))
        ):
            raise CampaignError("HE nonlinear reference-Riesz arrays are invalid")
        state_quadratic = float(np.dot(free_deformation, reference_action))
        quadratic_tolerance = 256.0 * np.finfo(np.float64).eps * max(
            1.0, abs(state_quadratic)
        )
        if not np.isfinite(state_quadratic) or state_quadratic < -quadratic_tolerance:
            raise CampaignError("HE nonlinear reference-Riesz state norm is invalid")
        state_sha = _sha256_array(displacement)
    admitted = bool(
        step.get("success") is True
        and certified_spd
        and int(metadata.get("reason", 0)) > 0
        and true_residual <= true_gate
        and relative_dual <= target
    )
    return {
        "status": "endpoint_admitted" if admitted else "censored_solver_nonconvergence",
        "scope": "one_load_step_full_nonlinear_reference_riesz_endpoint",
        "message": str(step.get("message", "")),
        "energy": _finite_number(step.get("energy"), "HE nonlinear energy"),
        "dual_residual_norm": _finite_number(
            convergence.get("dual_residual_norm"), "HE nonlinear dual residual norm"
        ),
        "dual_residual_relative": relative_dual,
        "relative_correction": _finite_number(
            convergence.get("relative_correction"), "HE nonlinear relative correction"
        ),
        "state_scale": _finite_number(
            convergence.get("state_scale"), "HE nonlinear state scale"
        ),
        "relative_true_residual": true_residual,
        "true_residual_rtol_gate": true_gate,
        "riesz_solver_contract": solver_contract,
        "certified_spd": certified_spd,
        "state_sha256": state_sha,
        "state_file_sha256": _sha256_file(state_path),
        "reference_elastic_state_norm": math.sqrt(max(0.0, state_quadratic)),
        "reference_elastic_action_sha256": _sha256_array(reference_action),
        "riesz_state_difference_available": True,
    }


def _p3d_endpoint(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = _read_json(_result_json_path(row))
    if payload.get("schema_id") != P3D_RESULT_SCHEMA_ID or payload.get(
        "schema_version"
    ) != P3D_RESULT_SCHEMA_VERSION:
        raise CampaignError("P3D fixed-state result schema is invalid")
    linear = payload.get("linear_solve")
    if not isinstance(linear, Mapping):
        raise CampaignError("P3D linear solve payload is missing")
    return {
        "status": str(payload.get("status", "failed")),
        "scope": "fixed_state_linear_system_not_nonlinear_convergence",
        "ksp_reason": int(linear.get("reason", 0)),
        "ksp_iterations": int(linear.get("iterations", 0)),
        "recursive_residual_norm": _finite_number(
            linear.get("recursive_residual_norm"), "P3D recursive residual"
        ),
        "true_residual_norm": _finite_number(
            linear.get("true_residual_norm"), "P3D true residual"
        ),
        "relative_true_residual": _finite_number(
            linear.get("relative_true_residual"), "P3D relative true residual"
        ),
        "true_residual_gate": _finite_number(
            linear.get("true_residual_gate"), "P3D true-residual gate"
        ),
        "correction_norm_2": _finite_number(
            linear.get("correction_norm_2"), "P3D correction norm"
        ),
        "reference_elastic_correction_norm": _finite_number(
            linear.get("reference_elastic_correction_norm"),
            "P3D reference-elastic correction norm",
        ),
        "state_sha256": str(payload.get("state_sha256", "")),
        "rhs_sha256": str(payload.get("rhs_sha256", "")),
        "branch_diagnostics": dict(payload.get("branch_diagnostics", {})),
    }


def _p3d_nonlinear_endpoint(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = _read_json(_result_json_path(row))
    convergence = payload.get("nonlinear_convergence")
    if not isinstance(convergence, Mapping):
        raise CampaignError("P3D nonlinear convergence payload is missing")
    configuration = convergence.get("configuration")
    metric = convergence.get("metric")
    norm_solve = convergence.get("last_riesz_solve")
    if not all(isinstance(value, Mapping) for value in (configuration, metric, norm_solve)):
        raise CampaignError("P3D nonlinear Riesz contract is incomplete")
    solver_contract = _require_riesz_solver_contract(
        row,
        metric=metric,
        norm_solve=norm_solve,
    )
    provenance = metric.get("provenance")
    certificate = provenance.get("spd_certificate") if isinstance(provenance, Mapping) else None
    certified_spd = isinstance(certificate, Mapping) and certificate.get("certified_spd") is True
    true_residual = _finite_number(
        norm_solve.get("relative_true_residual"), "P3D nonlinear Riesz true residual"
    )
    true_gate = _finite_number(
        norm_solve.get("true_residual_rtol_gate"),
        "P3D nonlinear Riesz true-residual gate",
    )
    relative = convergence.get("initial_relative_dual_residual")
    relative_value = (
        relative.get("value") if isinstance(relative, Mapping) else None
    )
    target = float(row["parameters"]["relative_dual_residual_target"])
    state_path = _state_npz_path(row)
    with np.load(state_path, allow_pickle=False) as state:
        free_state = np.asarray(
            state["free_displacement_reordered"], dtype=np.float64
        ).reshape(-1)
        reference_action = np.asarray(
            state["reference_elastic_action"], dtype=np.float64
        ).reshape(-1)
    residual_gate = convergence.get("residual_gate")
    residual_gate_passed = (
        isinstance(residual_gate, Mapping) and residual_gate.get("passed") is True
    )
    admitted = bool(
        payload.get("status") == "completed"
        and payload.get("solver_success") is True
        and configuration.get("selection") == "reference_elastic_energy"
        and certified_spd
        and int(norm_solve.get("reason", 0)) > 0
        and true_residual <= true_gate
        and residual_gate_passed
        and relative_value is not None
        and float(relative_value) <= target
    )
    return {
        "status": "endpoint_admitted" if admitted else "censored_solver_nonconvergence",
        "scope": "full_nonlinear_reference_riesz_endpoint_on_one_serial_rank",
        "message": str(payload.get("message", "")),
        "energy": _finite_number(payload.get("energy"), "P3D nonlinear energy"),
        "omega": _finite_number(payload.get("omega"), "P3D nonlinear omega"),
        "u_max": _finite_number(payload.get("u_max"), "P3D nonlinear u_max"),
        "dual_residual_relative": _finite_number(
            relative_value, "P3D nonlinear relative dual residual"
        ),
        "relative_correction": _finite_number(
            (convergence.get("relative_correction") or {}).get("value"),
            "P3D nonlinear relative correction",
        ),
        "relative_true_residual": true_residual,
        "true_residual_rtol_gate": true_gate,
        "riesz_solver_contract": solver_contract,
        "certified_spd": certified_spd,
        "branch_diagnostics": dict(payload.get("branch_diagnostics", {})),
        "free_state_sha256": _sha256_array(free_state),
        "reference_elastic_action_sha256": _sha256_array(reference_action),
        "state_file_sha256": _sha256_file(state_path),
    }


def _relative_difference(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(
        abs(float(left)), abs(float(right)), np.finfo(np.float64).tiny
    )


def _triangle_lumped_weights(coords: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.ndim != 2:
        raise CampaignError("GL triangle array is not two-dimensional")
    if triangles.shape[1] != 3 and triangles.shape[0] == 3:
        triangles = triangles.T
    if triangles.shape[1] != 3 or coords.ndim != 2 or coords.shape[1] < 2:
        raise CampaignError("GL state mesh has an unsupported shape")
    points = coords[triangles, :2]
    cross = (points[:, 1, 0] - points[:, 0, 0]) * (
        points[:, 2, 1] - points[:, 0, 1]
    ) - (points[:, 1, 1] - points[:, 0, 1]) * (
        points[:, 2, 0] - points[:, 0, 0]
    )
    areas = 0.5 * np.abs(cross)
    if np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
        raise CampaignError("GL state mesh contains a degenerate triangle")
    weights = np.zeros(coords.shape[0], dtype=np.float64)
    for local in range(3):
        np.add.at(weights, triangles[:, local], areas / 3.0)
    return weights


def _compare_gl(
    row: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    reference_row: Mapping[str, Any],
    reference: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if endpoint.get("status") != "endpoint_admitted" or reference.get("status") != "endpoint_admitted":
        return {"status": "censored", "reason": "candidate_or_reference_endpoint_not_converged"}
    with np.load(_state_npz_path(row), allow_pickle=False) as candidate_state, np.load(
        _state_npz_path(reference_row), allow_pickle=False
    ) as reference_state:
        coords = np.asarray(candidate_state["coords"], dtype=np.float64)
        triangles = np.asarray(candidate_state["triangles"], dtype=np.int64)
        if not np.array_equal(coords, np.asarray(reference_state["coords"], dtype=np.float64)):
            raise CampaignError("GL same-level comparison has different coordinates")
        if not np.array_equal(
            triangles, np.asarray(reference_state["triangles"], dtype=np.int64)
        ):
            raise CampaignError("GL same-level comparison has different connectivity")
        candidate_u = np.asarray(candidate_state["u"], dtype=np.float64).reshape(-1)
        reference_u = np.asarray(reference_state["u"], dtype=np.float64).reshape(-1)
    weights = _triangle_lumped_weights(coords, triangles)
    difference = candidate_u - reference_u
    difference_norm = float(np.sqrt(np.dot(weights, difference * difference)))
    reference_norm = float(np.sqrt(np.dot(weights, reference_u * reference_u)))
    relative_state = difference_norm / max(reference_norm, np.finfo(np.float64).tiny)
    energy_difference = abs(float(endpoint["energy"]) - float(reference["energy"]))
    passed = bool(
        relative_state <= float(contract["gl_lumped_l2_relative_state_difference_max"])
        and energy_difference <= float(contract["gl_energy_absolute_difference_max"])
    )
    return {
        "status": "accepted" if passed else "rejected",
        "reference_row_id": reference_row["row_id"],
        "lumped_l2_state_difference": difference_norm,
        "lumped_l2_relative_state_difference": relative_state,
        "energy_absolute_difference": energy_difference,
        "gates": {
            "relative_state_max": contract[
                "gl_lumped_l2_relative_state_difference_max"
            ],
            "energy_absolute_max": contract["gl_energy_absolute_difference_max"],
            "passed": passed,
        },
    }


def _compare_he(
    row: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    reference_row: Mapping[str, Any],
    reference: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if endpoint.get("status") != "reference_metric_check_admitted" or reference.get(
        "status"
    ) != "reference_metric_check_admitted":
        return {"status": "censored", "reason": "candidate_or_reference_metric_check_failed"}
    dual_difference = _relative_difference(
        float(endpoint["dual_residual_norm"]), float(reference["dual_residual_norm"])
    )
    scale_difference = _relative_difference(
        float(endpoint["state_scale"]), float(reference["state_scale"])
    )
    tolerance = float(row["parameters"]["riesz_ksp_rtol"])
    true_gate = max(
        float(contract["he_true_residual_floor"]),
        float(contract["he_true_residual_factor"]) * tolerance,
    )
    passed = bool(
        dual_difference <= float(contract["he_dual_norm_relative_difference_max"])
        and scale_difference <= float(contract["he_state_scale_relative_difference_max"])
        and float(endpoint["relative_true_residual"]) <= true_gate
    )
    return {
        "status": "accepted" if passed else "rejected",
        "reference_row_id": reference_row["row_id"],
        "dual_residual_norm_relative_difference": dual_difference,
        "state_scale_relative_difference": scale_difference,
        "relative_true_residual": endpoint["relative_true_residual"],
        "gates": {
            "dual_norm_relative_max": contract[
                "he_dual_norm_relative_difference_max"
            ],
            "state_scale_relative_max": contract[
                "he_state_scale_relative_difference_max"
            ],
            "true_residual_max": true_gate,
            "passed": passed,
        },
    }


def _compare_he_nonlinear(
    row: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    reference_row: Mapping[str, Any],
    reference: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if endpoint.get("status") != "endpoint_admitted" or reference.get(
        "status"
    ) != "endpoint_admitted":
        return {"status": "censored", "reason": "candidate_or_reference_endpoint_not_converged"}
    with np.load(_state_npz_path(row), allow_pickle=False) as candidate_state, np.load(
        _state_npz_path(reference_row), allow_pickle=False
    ) as reference_state:
        candidate_coords = np.asarray(candidate_state["coords_ref"], dtype=np.float64)
        reference_coords = np.asarray(reference_state["coords_ref"], dtype=np.float64)
        candidate_tetrahedra = np.asarray(candidate_state["tetrahedra"], dtype=np.int64)
        reference_tetrahedra = np.asarray(reference_state["tetrahedra"], dtype=np.int64)
        if not np.array_equal(candidate_coords, reference_coords) or not np.array_equal(
            candidate_tetrahedra, reference_tetrahedra
        ):
            raise CampaignError("HE same-level nonlinear comparison changed the mesh")
        candidate_displacement = np.asarray(
            candidate_state["displacement"], dtype=np.float64
        ).reshape(-1)
        reference_displacement = np.asarray(
            reference_state["displacement"], dtype=np.float64
        ).reshape(-1)
        required = {"free_deformation_original", "reference_elastic_action"}
        if required - set(candidate_state.files) or required - set(reference_state.files):
            raise CampaignError(
                "HE same-level nonlinear comparison lacks reference-Riesz arrays"
            )
        candidate_free = np.asarray(
            candidate_state["free_deformation_original"], dtype=np.float64
        ).reshape(-1)
        reference_free = np.asarray(
            reference_state["free_deformation_original"], dtype=np.float64
        ).reshape(-1)
        candidate_action = np.asarray(
            candidate_state["reference_elastic_action"], dtype=np.float64
        ).reshape(-1)
        reference_action = np.asarray(
            reference_state["reference_elastic_action"], dtype=np.float64
        ).reshape(-1)
    if not (
        candidate_free.shape
        == reference_free.shape
        == candidate_action.shape
        == reference_action.shape
    ) or candidate_free.size == 0:
        raise CampaignError("HE same-level reference-Riesz arrays are not aligned")
    if not all(
        np.all(np.isfinite(values))
        for values in (candidate_free, reference_free, candidate_action, reference_action)
    ):
        raise CampaignError("HE same-level reference-Riesz arrays are nonfinite")
    displacement_relative = float(
        np.linalg.norm(candidate_displacement - reference_displacement)
        / max(np.linalg.norm(reference_displacement), np.finfo(np.float64).tiny)
    )
    state_difference = candidate_free - reference_free
    action_difference = candidate_action - reference_action
    squared_difference = float(np.dot(state_difference, action_difference))
    difference_tolerance = 256.0 * np.finfo(np.float64).eps * max(
        1.0,
        float(np.linalg.norm(state_difference) * np.linalg.norm(action_difference)),
    )
    if not np.isfinite(squared_difference) or squared_difference < -difference_tolerance:
        raise CampaignError("HE reference-elastic state difference is invalid")
    reference_squared = float(np.dot(reference_free, reference_action))
    reference_tolerance = 256.0 * np.finfo(np.float64).eps * max(
        1.0, abs(reference_squared)
    )
    if not np.isfinite(reference_squared) or reference_squared < -reference_tolerance:
        raise CampaignError("HE reference-elastic reference norm is invalid")
    riesz_difference = math.sqrt(max(0.0, squared_difference))
    riesz_relative = riesz_difference / max(
        math.sqrt(max(0.0, reference_squared)), np.finfo(np.float64).tiny
    )
    energy_difference = abs(float(endpoint["energy"]) - float(reference["energy"]))
    passed = bool(
        displacement_relative
        <= float(contract["he_nonlinear_displacement_relative_difference_max"])
        and riesz_relative
        <= float(
            contract[
                "he_nonlinear_reference_elastic_relative_state_difference_max"
            ]
        )
        and energy_difference
        <= float(contract["he_nonlinear_energy_absolute_difference_max"])
    )
    return {
        "status": "accepted" if passed else "rejected",
        "reference_row_id": reference_row["row_id"],
        "coefficient_displacement_relative_difference": displacement_relative,
        "reference_elastic_state_difference": riesz_difference,
        "reference_elastic_relative_state_difference": riesz_relative,
        "energy_absolute_difference": energy_difference,
        "riesz_state_difference_available": True,
        "gates": {
            "coefficient_displacement_relative_max": contract[
                "he_nonlinear_displacement_relative_difference_max"
            ],
            "reference_elastic_relative_state_max": contract[
                "he_nonlinear_reference_elastic_relative_state_difference_max"
            ],
            "energy_absolute_max": contract[
                "he_nonlinear_energy_absolute_difference_max"
            ],
            "passed": passed,
        },
        "interpretation": (
            "same-mesh endpoint difference in the frozen reference-elastic Riesz metric; "
            "the coefficient displacement difference is retained as a secondary diagnostic"
        ),
    }


def _compare_p3d(
    row: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    reference_row: Mapping[str, Any],
    reference: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if endpoint.get("status") != "passed" or reference.get("status") != "passed":
        return {"status": "censored", "reason": "candidate_or_reference_linear_solve_failed"}
    if endpoint.get("state_sha256") != reference.get("state_sha256") or endpoint.get(
        "rhs_sha256"
    ) != reference.get("rhs_sha256"):
        raise CampaignError("P3D same-degree comparison changed the fixed state or RHS")
    with np.load(_state_npz_path(row), allow_pickle=False) as candidate_state, np.load(
        _state_npz_path(reference_row), allow_pickle=False
    ) as reference_state:
        candidate_correction = np.asarray(
            candidate_state["correction"], dtype=np.float64
        ).reshape(-1)
        reference_correction = np.asarray(
            reference_state["correction"], dtype=np.float64
        ).reshape(-1)
        candidate_action = np.asarray(
            candidate_state["reference_elastic_action"], dtype=np.float64
        ).reshape(-1)
        reference_action = np.asarray(
            reference_state["reference_elastic_action"], dtype=np.float64
        ).reshape(-1)
    correction_difference = candidate_correction - reference_correction
    action_difference = candidate_action - reference_action
    coefficient_relative = float(
        np.linalg.norm(correction_difference)
        / max(np.linalg.norm(reference_correction), np.finfo(np.float64).tiny)
    )
    squared_riesz = float(np.dot(correction_difference, action_difference))
    if squared_riesz < -1.0e-10 * max(
        float(np.linalg.norm(correction_difference) * np.linalg.norm(action_difference)),
        1.0,
    ):
        raise CampaignError("P3D reference-elastic difference norm is negative")
    riesz_difference = math.sqrt(max(0.0, squared_riesz))
    riesz_relative = riesz_difference / max(
        float(reference["reference_elastic_correction_norm"]),
        np.finfo(np.float64).tiny,
    )
    tolerance = float(row["parameters"]["ksp_rtol"])
    true_gate = max(
        float(contract["p3d_true_residual_floor"]),
        float(contract["p3d_true_residual_factor"]) * tolerance,
    )
    passed = bool(
        coefficient_relative
        <= float(contract["p3d_correction_relative_difference_max"])
        and riesz_relative
        <= float(contract["p3d_reference_elastic_relative_difference_max"])
        and float(endpoint["relative_true_residual"]) <= true_gate
    )
    return {
        "status": "accepted" if passed else "rejected",
        "reference_row_id": reference_row["row_id"],
        "coefficient_l2_relative_correction_difference": coefficient_relative,
        "reference_elastic_correction_difference": riesz_difference,
        "reference_elastic_relative_correction_difference": riesz_relative,
        "relative_true_residual": endpoint["relative_true_residual"],
        "gates": {
            "coefficient_relative_max": contract[
                "p3d_correction_relative_difference_max"
            ],
            "reference_elastic_relative_max": contract[
                "p3d_reference_elastic_relative_difference_max"
            ],
            "true_residual_max": true_gate,
            "passed": passed,
        },
    }


def _compare_p3d_nonlinear(
    row: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    reference_row: Mapping[str, Any],
    reference: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if endpoint.get("status") != "endpoint_admitted" or reference.get(
        "status"
    ) != "endpoint_admitted":
        return {"status": "censored", "reason": "candidate_or_reference_endpoint_not_converged"}
    with np.load(_state_npz_path(row), allow_pickle=False) as candidate_state, np.load(
        _state_npz_path(reference_row), allow_pickle=False
    ) as reference_state:
        candidate_coords = np.asarray(candidate_state["coords_ref"], dtype=np.float64)
        reference_coords = np.asarray(reference_state["coords_ref"], dtype=np.float64)
        candidate_tetrahedra = np.asarray(candidate_state["tetrahedra"], dtype=np.int64)
        reference_tetrahedra = np.asarray(reference_state["tetrahedra"], dtype=np.int64)
        if not np.array_equal(candidate_coords, reference_coords) or not np.array_equal(
            candidate_tetrahedra, reference_tetrahedra
        ):
            raise CampaignError("P3D same-degree nonlinear comparison changed the mesh")
        candidate_free = np.asarray(
            candidate_state["free_displacement_reordered"], dtype=np.float64
        ).reshape(-1)
        reference_free = np.asarray(
            reference_state["free_displacement_reordered"], dtype=np.float64
        ).reshape(-1)
        candidate_action = np.asarray(
            candidate_state["reference_elastic_action"], dtype=np.float64
        ).reshape(-1)
        reference_action = np.asarray(
            reference_state["reference_elastic_action"], dtype=np.float64
        ).reshape(-1)
    difference = candidate_free - reference_free
    action_difference = candidate_action - reference_action
    squared_difference = float(np.dot(difference, action_difference))
    if squared_difference < -1.0e-10 * max(
        float(np.linalg.norm(difference) * np.linalg.norm(action_difference)), 1.0
    ):
        raise CampaignError("P3D nonlinear Riesz state difference is negative")
    riesz_difference = math.sqrt(max(0.0, squared_difference))
    reference_squared = float(np.dot(reference_free, reference_action))
    reference_norm = math.sqrt(max(0.0, reference_squared))
    riesz_relative = riesz_difference / max(
        reference_norm, np.finfo(np.float64).tiny
    )
    energy_difference = abs(float(endpoint["energy"]) - float(reference["energy"]))
    omega_difference = abs(float(endpoint["omega"]) - float(reference["omega"]))
    u_max_difference = abs(float(endpoint["u_max"]) - float(reference["u_max"]))
    candidate_counts = (endpoint.get("branch_diagnostics") or {}).get("counts")
    reference_counts = (reference.get("branch_diagnostics") or {}).get("counts")
    branch_counts_equal = isinstance(candidate_counts, Mapping) and dict(
        candidate_counts
    ) == dict(reference_counts or {})
    passed = bool(
        riesz_relative
        <= float(
            contract[
                "p3d_nonlinear_reference_elastic_relative_state_difference_max"
            ]
        )
        and energy_difference
        <= float(contract["p3d_nonlinear_energy_absolute_difference_max"])
        and omega_difference
        <= float(contract["p3d_nonlinear_omega_absolute_difference_max"])
        and u_max_difference
        <= float(contract["p3d_nonlinear_u_max_absolute_difference_max"])
        and branch_counts_equal
    )
    return {
        "status": "accepted" if passed else "rejected",
        "reference_row_id": reference_row["row_id"],
        "reference_elastic_state_difference": riesz_difference,
        "reference_elastic_relative_state_difference": riesz_relative,
        "energy_absolute_difference": energy_difference,
        "omega_absolute_difference": omega_difference,
        "u_max_absolute_difference": u_max_difference,
        "branch_counts_equal": branch_counts_equal,
        "gates": {
            "reference_elastic_relative_state_max": contract[
                "p3d_nonlinear_reference_elastic_relative_state_difference_max"
            ],
            "energy_absolute_max": contract[
                "p3d_nonlinear_energy_absolute_difference_max"
            ],
            "omega_absolute_max": contract[
                "p3d_nonlinear_omega_absolute_difference_max"
            ],
            "u_max_absolute_max": contract[
                "p3d_nonlinear_u_max_absolute_difference_max"
            ],
            "branch_counts_exact": True,
            "passed": passed,
        },
    }


def _selected_group_policy(
    rows: Sequence[Mapping[str, Any]], comparisons: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    accepted = [
        row
        for row in rows
        if comparisons.get(str(row["row_id"]), {}).get("status") == "accepted"
    ]
    if not accepted:
        return {"status": "no_acceptable_policy", "row_id": None, "tolerance": None}
    family = str(accepted[0]["family"])
    parameter = {
        "ginzburg_landau": "relative_dual_residual_target",
        "hyperelasticity_reference_riesz": "riesz_ksp_rtol",
        "hyperelasticity_nonlinear_stopping": "relative_dual_residual_target",
        "plasticity3d_fixed_state_linear": "ksp_rtol",
        "plasticity3d_nonlinear_stopping": "relative_dual_residual_target",
    }[family]
    selected = max(accepted, key=lambda row: float(row["parameters"][parameter]))
    return {
        "status": ACCEPTED_POLICY_STATUS,
        "row_id": selected["row_id"],
        "parameter": parameter,
        "tolerance": float(selected["parameters"][parameter]),
    }


def _required_local_policy_grid(
    local_rows: Sequence[Mapping[str, Any]],
    selected_policies: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit the exact local group grid needed before cluster completion.

    A completed process inventory is not a successful calibration.  Every
    required same-discretization group must also expose one policy selected
    from an accepted comparison, and that row must belong to the named group.
    """

    rows_by_id = {str(row.get("row_id")): row for row in local_rows}
    observed_groups = {str(row.get("group_id")) for row in local_rows}
    selected_groups = {str(group) for group in selected_policies}
    missing_groups = sorted(COMPLETE_REQUIRED_LOCAL_GROUPS - observed_groups)
    unexpected_groups = sorted(observed_groups - COMPLETE_REQUIRED_LOCAL_GROUPS)
    missing_policy_records = sorted(COMPLETE_REQUIRED_LOCAL_GROUPS - selected_groups)
    unexpected_policy_records = sorted(selected_groups - COMPLETE_REQUIRED_LOCAL_GROUPS)
    rejected_policy_groups: list[str] = []
    invalid_selected_rows: list[str] = []
    for group in sorted(COMPLETE_REQUIRED_LOCAL_GROUPS & selected_groups):
        record = selected_policies.get(group)
        if not isinstance(record, Mapping) or record.get("status") != ACCEPTED_POLICY_STATUS:
            rejected_policy_groups.append(group)
            continue
        row_id = str(record.get("row_id", ""))
        row = rows_by_id.get(row_id)
        if row is None or str(row.get("group_id")) != group:
            invalid_selected_rows.append(group)
    complete = not any(
        (
            missing_groups,
            unexpected_groups,
            missing_policy_records,
            unexpected_policy_records,
            rejected_policy_groups,
            invalid_selected_rows,
        )
    )
    return {
        "expected_groups": sorted(COMPLETE_REQUIRED_LOCAL_GROUPS),
        "observed_groups": sorted(observed_groups),
        "missing_groups": missing_groups,
        "unexpected_groups": unexpected_groups,
        "missing_policy_records": missing_policy_records,
        "unexpected_policy_records": unexpected_policy_records,
        "rejected_policy_groups": rejected_policy_groups,
        "invalid_selected_rows": invalid_selected_rows,
        "complete": complete,
    }


def _local_terminal_decision(
    *,
    missing: Sequence[str],
    invalid: Sequence[str],
    runtime_censored: Sequence[str],
    reference_failures: Sequence[str],
    policy_grid: Mapping[str, Any],
) -> str:
    if missing:
        return "incomplete_local_execution"
    if invalid or reference_failures:
        return "invalid_local_evidence"
    if runtime_censored:
        return "censored_local_execution"
    if policy_grid.get("missing_groups") or policy_grid.get("unexpected_groups"):
        return "incomplete_local_group_scope"
    if policy_grid.get("complete") is not True:
        return "local_calibration_policy_gate_failed"
    return "local_calibration_complete_cluster_computations_deferred"


def analyze_plan(plan_path: Path) -> dict[str, Any]:
    plan, output_root = _load_plan(plan_path)
    _verify_source_identity(plan)
    _verify_frozen_plan_design(plan, output_root=output_root)
    contract = plan["policies"]["analysis_contract"]
    local_rows = [row for row in plan["rows"] if row["execution_class"] == "required_local"]
    deferred_rows = [
        row
        for row in plan["rows"]
        if row["execution_class"] == "deferred_cluster_computation"
    ]
    audit: dict[str, dict[str, Any]] = {}
    endpoints: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    invalid: list[str] = []
    runtime_censored: list[str] = []
    for row in local_rows:
        status, receipt, errors = _receipt_for_row(
            plan_path=Path(plan_path).resolve(), output_root=output_root, row=row
        )
        audit[str(row["row_id"])] = {
            "receipt_status": status,
            "receipt": None if receipt is None else str(
                output_root / "receipts" / f"{row['row_id']}.json"
            ),
            "errors": errors,
        }
        if status == "missing":
            missing.append(str(row["row_id"]))
            continue
        if status == "runtime_censored":
            runtime_censored.append(str(row["row_id"]))
            continue
        if status != "completed":
            invalid.append(str(row["row_id"]))
            continue
        try:
            if row["family"] == "ginzburg_landau":
                endpoints[str(row["row_id"])] = _gl_endpoint(row)
            elif row["family"] == "hyperelasticity_reference_riesz":
                endpoints[str(row["row_id"])] = _he_endpoint(row)
            elif row["family"] == "hyperelasticity_nonlinear_stopping":
                endpoints[str(row["row_id"])] = _he_nonlinear_endpoint(row)
            elif row["family"] == "plasticity3d_fixed_state_linear":
                endpoints[str(row["row_id"])] = _p3d_endpoint(row)
            elif row["family"] == "plasticity3d_nonlinear_stopping":
                endpoints[str(row["row_id"])] = _p3d_nonlinear_endpoint(row)
            else:
                raise CampaignError(f"unsupported local family {row['family']}")
        except Exception as exc:
            invalid.append(str(row["row_id"]))
            audit[str(row["row_id"])]["errors"].append(
                f"endpoint parse failed: {type(exc).__name__}: {exc}"
            )

    comparisons: dict[str, dict[str, Any]] = {}
    selected_policies: dict[str, dict[str, Any]] = {}
    groups = sorted({str(row["group_id"]) for row in local_rows})
    reference_failures: list[str] = []
    for group_id in groups:
        group_rows = [row for row in local_rows if row["group_id"] == group_id]
        references = [row for row in group_rows if row.get("reference_row") is True]
        if len(references) != 1:
            invalid.append(group_id)
            continue
        reference_row = references[0]
        reference_endpoint = endpoints.get(str(reference_row["row_id"]))
        if reference_endpoint is None:
            if str(reference_row["row_id"]) not in missing:
                reference_failures.append(str(reference_row["row_id"]))
            continue
        if reference_endpoint.get("status") not in {
            "endpoint_admitted",
            "reference_metric_check_admitted",
            "passed",
        }:
            reference_failures.append(str(reference_row["row_id"]))
        for row in group_rows:
            endpoint = endpoints.get(str(row["row_id"]))
            if endpoint is None:
                continue
            if row["family"] == "ginzburg_landau":
                comparison = _compare_gl(
                    row, endpoint, reference_row, reference_endpoint, contract
                )
            elif row["family"] == "hyperelasticity_reference_riesz":
                comparison = _compare_he(
                    row, endpoint, reference_row, reference_endpoint, contract
                )
            elif row["family"] == "hyperelasticity_nonlinear_stopping":
                comparison = _compare_he_nonlinear(
                    row, endpoint, reference_row, reference_endpoint, contract
                )
            elif row["family"] == "plasticity3d_fixed_state_linear":
                comparison = _compare_p3d(
                    row, endpoint, reference_row, reference_endpoint, contract
                )
            else:
                comparison = _compare_p3d_nonlinear(
                    row, endpoint, reference_row, reference_endpoint, contract
                )
            comparisons[str(row["row_id"])] = comparison
        selected_policies[group_id] = _selected_group_policy(group_rows, comparisons)

    policy_grid = _required_local_policy_grid(local_rows, selected_policies)
    terminal = _local_terminal_decision(
        missing=missing,
        invalid=invalid,
        runtime_censored=runtime_censored,
        reference_failures=reference_failures,
        policy_grid=policy_grid,
    )
    return {
        "schema_id": ANALYSIS_SCHEMA_ID,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "experiment_id": "EXP-STOP-001",
        "campaign_id": plan["campaign_id"],
        "created_utc": _utc_now(),
        "terminal_decision": terminal,
        "complete_exp_stop_pass": False,
        "publication_timing_admissible": False,
        "scope_statement": (
            "The local tranche calibrates deterministic GL endpoints, HE reference-Riesz "
            "setup/norm solves and L1/L2 nonlinear endpoints, P1/P2 fixed-state P3D linear "
            "solves, and P1/P2 nonlinear P3D endpoints. P4 nonlinear and MPI consistency "
            "remain cluster-deferred. Same-mesh HE and P3D endpoint comparisons use the "
            "action of the frozen reference-elastic operator; coefficient-space differences "
            "are retained only as secondary diagnostics."
        ),
        "plan": {
            "path": str(Path(plan_path).resolve()),
            "sha256": _sha256_file(Path(plan_path).resolve()),
            "run_kind": plan["run_kind"],
            "source_commit": plan["source"]["commit"],
        },
        "counts": {
            "required_local": len(local_rows),
            "completed_endpoint_records": len(endpoints),
            "missing_local": len(missing),
            "invalid_local": len(set(invalid)),
            "runtime_censored_local": len(set(runtime_censored)),
            "reference_failures": len(set(reference_failures)),
            "policy_gate_failures": len(policy_grid["rejected_policy_groups"])
            + len(policy_grid["missing_policy_records"])
            + len(policy_grid["invalid_selected_rows"]),
            "deferred_cluster_computations": len(deferred_rows),
        },
        "missing_local_rows": sorted(set(missing)),
        "invalid_local_rows": sorted(set(invalid)),
        "runtime_censored_local_rows": sorted(set(runtime_censored)),
        "reference_failures": sorted(set(reference_failures)),
        "audit": audit,
        "endpoints": endpoints,
        "same_discretization_reference_comparisons": comparisons,
        "selected_local_policies": selected_policies,
        "required_local_policy_grid": policy_grid,
        "cross_mesh_summary": {
            "ginzburg_landau": {
                str(level): selected_policies.get(f"gl_l{level}") for level in GL_LEVELS
            },
            "hyperelasticity_reference_riesz": {
                str(level): selected_policies.get(f"he_l{level}") for level in HE_LEVELS
            },
            "hyperelasticity_nonlinear_stopping": {
                str(level): selected_policies.get(f"he_l{level}_nonlinear")
                for level in HE_LEVELS
            },
            "plasticity3d_nonlinear_stopping": {
                str(degree): selected_policies.get(f"p3d_p{degree}_nonlinear")
                for degree in (1, 2)
            },
            "interpretation": (
                "Cross-mesh rows compare selected policies and scalar observables only; "
                "no state-vector difference is formed across nonmatching meshes."
            ),
        },
        "deferred_cluster_computations": [
            {
                "row_id": row["row_id"],
                "family": row["family"],
                "parameters": row["parameters"],
                "censor": row["censor"],
            }
            for row in deferred_rows
        ],
        "runtime_censors": [
            {
                "row_id": row_id,
                "receipt": audit[row_id]["receipt"],
                "reason": audit[row_id]["errors"][0],
                "classification": "unclassified_runtime_failure",
                "publication_evidence": False,
            }
            for row_id in sorted(set(runtime_censored))
        ],
    }


def analyze_campaign(args: argparse.Namespace) -> Path:
    plan, output_root = _load_plan(Path(args.plan))
    del plan
    output = (
        Path(args.output).resolve()
        if args.output
        else output_root / "analysis.json"
    )
    _confined(output_root, output)
    analysis = analyze_plan(Path(args.plan))
    _atomic_json(output, analysis)
    return output


def _validate_p3d_worker_args(args: argparse.Namespace) -> None:
    degree = int(args.degree)
    if str(args.quadrature_rule) != P3D_QUADRATURE[degree]:
        raise CampaignError(
            f"P{degree} fixed-state calibration requires {P3D_QUADRATURE[degree]}"
        )
    if not (0.0 < float(args.ksp_rtol) < 1.0):
        raise CampaignError("--ksp-rtol must lie strictly between zero and one")
    if int(args.ksp_max_it) <= 0 or float(args.true_residual_factor) <= 0.0:
        raise CampaignError("KSP cap and true-residual factor must be positive")
    if float(args.state_amplitude) <= 0.0:
        raise CampaignError("--state-amplitude must be positive")


def _p3d_fixed_state(args: argparse.Namespace) -> Path:
    """Solve one frozen P3D tangent system and recompute its true residual."""
    _validate_p3d_worker_args(args)

    from mpi4py import MPI
    from petsc4py import PETSc

    from experiments.runners.run_plasticity3d_backend_mix_case import (
        _build_local_assembly_backend,
    )
    from experiments.runners.run_plasticity3d_fixed_state_route_screen import (
        _branch_diagnostics,
    )
    from src.problems.slope_stability_3d.support.fixed_state import (
        prescribed_analytic_displacement,
    )

    if MPI.COMM_WORLD.size != 1:
        raise CampaignError("p3d-fixed-state is a serial local row; MPI rows are cluster-deferred")
    degree = int(args.degree)
    output = Path(args.output).resolve()
    state_out = Path(args.state_out).resolve()
    if state_out.parent != output.parent:
        raise CampaignError("--state-out and --output must share one row directory")
    if output.exists() or state_out.exists():
        raise CampaignError("P3D fixed-state outputs already exist")
    output.parent.mkdir(parents=True, exist_ok=True)

    backend = _build_local_assembly_backend(
        mesh_name="hetero_ssr_L1",
        elem_degree=degree,
        constraint_variant="glued_bottom",
        quadrature_rule_id=str(args.quadrature_rule),
        lambda_target=1.55,
        local_hessian_mode="element",
        autodiff_tangent_mode="element",
        ksp_rtol=float(args.ksp_rtol),
        ksp_max_it=int(args.ksp_max_it),
    )
    state = gradient = rhs = correction = residual = elastic_action = None
    tangent = ksp = None
    try:
        coords = np.asarray(backend.coords_ref, dtype=np.float64)
        full = prescribed_analytic_displacement(
            coords, amplitude=float(args.state_amplitude)
        )
        original_free = full.reshape(-1)[np.asarray(backend.freedofs, dtype=np.int64)]
        state_global = np.asarray(
            original_free[np.asarray(backend.perm, dtype=np.int64)], dtype=np.float64
        )
        state = backend.create_vec(state_global)
        gradient = state.duplicate()
        rhs = state.duplicate()
        correction = state.duplicate()
        residual = state.duplicate()
        elastic_action = state.duplicate()
        backend.vec_gradient(state, gradient)
        gradient.copy(rhs)
        rhs.scale(-1.0)
        tangent = backend.vec_tangent(state).copy()

        ksp = PETSc.KSP().create(comm=tangent.getComm())
        ksp.setType(str(args.ksp_type))
        ksp.getPC().setType(str(args.pc_type))
        ksp.setOperators(tangent)
        ksp.setTolerances(
            rtol=float(args.ksp_rtol), atol=0.0, max_it=int(args.ksp_max_it)
        )
        ksp.setInitialGuessNonzero(False)
        ksp.setUp()
        started = time.perf_counter()
        ksp.solve(rhs, correction)
        solve_time = float(time.perf_counter() - started)

        tangent.mult(correction, residual)
        residual.axpy(-1.0, rhs)
        rhs_norm = float(rhs.norm(PETSc.NormType.NORM_2))
        true_norm = float(residual.norm(PETSc.NormType.NORM_2))
        relative_true = true_norm / max(rhs_norm, np.finfo(np.float64).tiny)
        recursive = float(ksp.getResidualNorm())
        reason = int(ksp.getConvergedReason())
        correction_global = np.asarray(backend.global_from_vec(correction), dtype=np.float64)
        rhs_global = np.asarray(backend.global_from_vec(rhs), dtype=np.float64)

        reference_elastic = backend.elastic_matrix()
        reference_elastic.mult(correction, elastic_action)
        elastic_action_global = np.asarray(
            backend.global_from_vec(elastic_action), dtype=np.float64
        )
        squared_reference_norm = float(correction.dot(elastic_action))
        reference_norm_valid = squared_reference_norm >= -1.0e-10 * max(
            float(correction.norm() * elastic_action.norm()), 1.0
        )
        reference_norm = math.sqrt(max(0.0, squared_reference_norm))
        branch = _branch_diagnostics(backend, state)
        branch_gate = bool(
            float(branch.get("normalized_boundary_margin_min", 0.0)) >= 1.0e-8
            and float(branch.get("near_boundary_fraction", 1.0)) == 0.0
        )
        true_gate = max(
            1.0e-12, float(args.true_residual_factor) * float(args.ksp_rtol)
        )
        finite = all(
            math.isfinite(value)
            for value in (
                rhs_norm,
                true_norm,
                relative_true,
                recursive,
                reference_norm,
            )
        )
        passed = bool(
            reason > 0
            and finite
            and relative_true <= true_gate
            and reference_norm_valid
            and branch_gate
        )

        temporary_npz = state_out.with_name(f".{state_out.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            temporary_npz,
            state=state_global,
            rhs=rhs_global,
            correction=correction_global,
            reference_elastic_action=elastic_action_global,
            element_degree=np.asarray(degree, dtype=np.int64),
            ksp_rtol=np.asarray(float(args.ksp_rtol), dtype=np.float64),
        )
        temporary_npz.replace(state_out)
        tolerances = ksp.getTolerances()
        payload = {
            "schema_id": P3D_RESULT_SCHEMA_ID,
            "schema_version": P3D_RESULT_SCHEMA_VERSION,
            "experiment_id": "EXP-STOP-001",
            "status": "passed" if passed else "failed",
            "scope": {
                "fixed_state": True,
                "linear_system_only": True,
                "nonlinear_iterations": 0,
                "nonlinear_convergence_claim_admissible": False,
                "timing_claim_admissible": False,
            },
            "case": {
                "mesh_name": "hetero_ssr_L1",
                "element_degree": degree,
                "quadrature_rule_id": str(args.quadrature_rule),
                "constraint_variant": "glued_bottom",
                "lambda_target": 1.55,
                "state_family": "analytic_mesh_field_v1",
                "state_label": "elastic",
                "state_amplitude": float(args.state_amplitude),
                "free_dofs": int(state_global.size),
            },
            "state_sha256": _sha256_array(state_global),
            "rhs_sha256": _sha256_array(rhs_global),
            "correction_sha256": _sha256_array(correction_global),
            "state_file": {
                "path": str(state_out),
                "sha256": _sha256_file(state_out),
            },
            "branch_diagnostics": branch,
            "branch_gate_passed": branch_gate,
            "linear_solve": {
                "ksp_type": str(ksp.getType()),
                "pc_type": str(ksp.getPC().getType()),
                "requested_rtol": float(args.ksp_rtol),
                "effective_rtol": float(tolerances[0]),
                "effective_atol": float(tolerances[1]),
                "effective_dtol": float(tolerances[2]),
                "effective_max_it": int(tolerances[3]),
                "reason": reason,
                "iterations": int(ksp.getIterationNumber()),
                "rhs_norm": rhs_norm,
                "recursive_residual_norm": recursive,
                "true_residual_norm": true_norm,
                "relative_true_residual": relative_true,
                "true_residual_gate": true_gate,
                "true_residual_gate_passed": relative_true <= true_gate,
                "correction_norm_2": float(np.linalg.norm(correction_global)),
                "reference_elastic_correction_norm": reference_norm,
                "reference_elastic_norm_valid": reference_norm_valid,
                "solve_time_diagnostic_s": solve_time,
            },
            "provenance": {
                "git": _git_metadata(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "command": shlex.join([sys.executable, *sys.argv]),
            },
        }
        _atomic_json(output, payload)
        return output
    finally:
        if ksp is not None:
            ksp.destroy()
        if tangent is not None:
            tangent.destroy()
        for vector in (elastic_action, residual, correction, rhs, gradient, state):
            if vector is not None:
                vector.destroy()
        backend.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="freeze a fresh local campaign plan")
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--run-kind", choices=("publication", "diagnostic"), default="publication")
    prepare.add_argument("--allow-dirty", action="store_true")
    prepare.add_argument(
        "--p4-policy",
        choices=("deferred_cluster", "local"),
        default="deferred_cluster",
    )
    prepare.add_argument("--confirm-p4-local-feasible", action="store_true")

    execute = subparsers.add_parser("execute", help="execute frozen local row(s)")
    execute.add_argument("--plan", type=Path, required=True)
    selection = execute.add_mutually_exclusive_group(required=True)
    selection.add_argument("--row")
    selection.add_argument("--all-local", action="store_true")
    execute.add_argument("--confirm-all-local-execution", action="store_true")
    execute.add_argument("--timeout-s", type=float, default=None)

    analyze = subparsers.add_parser("analyze", help="audit and compare local endpoints")
    analyze.add_argument("--plan", type=Path, required=True)
    analyze.add_argument("--output", type=Path, default=None)

    worker = subparsers.add_parser(
        "p3d-fixed-state", help="internal fixed-state P3D linear calibration worker"
    )
    worker.add_argument("--degree", type=int, choices=(1, 2, 4), required=True)
    worker.add_argument(
        "--quadrature-rule",
        choices=tuple(P3D_QUADRATURE.values()),
        required=True,
    )
    worker.add_argument("--state-amplitude", type=float, required=True)
    worker.add_argument("--ksp-type", choices=("gmres",), default="gmres")
    worker.add_argument("--pc-type", choices=("hypre",), default="hypre")
    worker.add_argument("--ksp-rtol", type=float, required=True)
    worker.add_argument("--ksp-max-it", type=int, default=4000)
    worker.add_argument("--true-residual-factor", type=float, default=20.0)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--state-out", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        if args.command == "prepare":
            path = prepare_campaign(args)
            print(path)
        elif args.command == "execute":
            for path in execute_campaign(args):
                print(path)
        elif args.command == "analyze":
            print(analyze_campaign(args))
        else:
            print(_p3d_fixed_state(args))
    except CampaignError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

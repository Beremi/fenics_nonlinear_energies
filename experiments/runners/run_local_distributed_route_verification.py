#!/usr/bin/env python3
"""Run the local 1/2/4-rank colored-recovery correctness matrix.

This campaign is a correctness experiment, not a timing benchmark.  It runs
element AD, colored sparse finite differences, and constitutive AD at the same
prescribed P1/P2 states and compares canonical states, gradients, four tangent
actions, branch diagnostics, ownership summaries, and feasible direct CSR
matrices across both routes and MPI decompositions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid
from typing import Mapping, Sequence

import numpy as np

from src.core.benchmark.run_record import (
    atomic_write_json,
    check_experiment_preflight,
    sha256_file,
    strict_json_dumps,
    utc_now_iso,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CHILD_RUNNER = REPO_ROOT / "experiments/runners/run_plasticity3d_fixed_state_route_screen.py"
BACKEND_RUNNER = REPO_ROOT / "experiments/runners/run_plasticity3d_backend_mix_case.py"
PROTOCOL_PATH = REPO_ROOT / "paper/protocols/EXP-DIST-001.md"
MESH_ROOT = REPO_ROOT / "data/meshes/SlopeStability3D/hetero_ssr"
MANIFEST_NAME = "manifest.json"
PROCESS_RECORD_NAME = "process_record.json"
CAMPAIGN_SCHEMA_ID = "fenics-nonlinear-energies.exp-dist-001-colored-recovery-manifest"
CAMPAIGN_SCHEMA_VERSION = 2
PLAN_SCHEMA_ID = "fenics-nonlinear-energies.exp-dist-001-colored-recovery-plan"
PLAN_SCHEMA_VERSION = 2
VERIFICATION_SCHEMA_ID = "fenics-nonlinear-energies.exp-dist-001-colored-recovery"
VERIFICATION_SCHEMA_VERSION = 2
COMMIT_RE = re.compile(r"[0-9a-fA-F]{40,64}")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
ROUTES = ("element_ad", "colored_sfd", "constitutive_ad")
DEGREES = (1, 2)
RANKS = (1, 2, 4)
STATES = (("elastic", 2.0e-4), ("mixed", 2.0e-2))
RULE_BY_DEGREE = {1: "tetra_1point", 2: "tetra_11point"}
RELATIVE_TOLERANCE = 1.0e-8
ABSOLUTE_TOLERANCE = 1.0e-10
THREAD_ENVIRONMENT = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "True",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
}


class DistributedCampaignError(RuntimeError):
    """Raised when the local distributed publication contract is violated."""


def _json_sha256(value: object) -> str:
    encoded = strict_json_dumps(value, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _read_strict_json(path: Path) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _git_metadata() -> dict[str, object]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise DistributedCampaignError(
                f"git {' '.join(arguments)} failed: {detail}"
            )
        return completed.stdout.strip()

    status = tuple(
        line
        for line in run("status", "--porcelain=v1", "--untracked-files=all").splitlines()
        if line
    )
    return {
        "commit": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": list(status),
    }


def _resolve_python(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    # Preserve the venv launcher spelling.  Resolving its symlink to the base
    # interpreter changes Python prefix discovery and can drop site-packages.
    path = Path(os.path.abspath(path))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise DistributedCampaignError(
            f"Python executable is missing or not executable: {path}"
        )
    if path.resolve() != Path(sys.executable).resolve():
        raise DistributedCampaignError(
            "publication driver and worker must use the same Python executable"
        )
    return path


def _resolve_mpiexec(raw: str | Path) -> Path:
    text = str(raw)
    located = shutil.which(text) if not Path(text).is_absolute() else text
    if not located:
        raise DistributedCampaignError(f"MPI launcher not found: {raw}")
    path = Path(located).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise DistributedCampaignError(f"MPI launcher is not executable: {path}")
    return path


def _relative_key(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _collect_code_hashes() -> dict[str, str]:
    tracked = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--", "src"],
        check=False,
        capture_output=True,
    )
    if tracked.returncode != 0:
        raise DistributedCampaignError("could not enumerate tracked source files")
    paths = {
        REPO_ROOT / raw.decode("utf-8")
        for raw in tracked.stdout.split(b"\0")
        if raw
    }
    paths.update({Path(__file__).resolve(), CHILD_RUNNER, BACKEND_RUNNER})
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise DistributedCampaignError(f"hashed code file is missing: {missing[0]}")
    return {
        _relative_key(path): sha256_file(path)
        for path in sorted(paths, key=_relative_key)
    }


def _collect_configuration_hashes() -> dict[str, str]:
    if not PROTOCOL_PATH.is_file():
        raise DistributedCampaignError(f"protocol is missing: {PROTOCOL_PATH}")
    return {_relative_key(PROTOCOL_PATH): sha256_file(PROTOCOL_PATH)}


def _collect_input_hashes() -> dict[str, str]:
    from src.problems.slope_stability_3d.support.mesh import (
        _same_mesh_hdf5_is_current,
    )

    paths = {MESH_ROOT / "SSR_hetero_ada_L1.msh", MESH_ROOT / "definition.py"}
    for degree in DEGREES:
        path = MESH_ROOT / f"hetero_ssr_L1_p{degree}_same_mesh_glued_bottom.h5"
        if not _same_mesh_hdf5_is_current(
            path,
            mesh_name="hetero_ssr_L1",
            degree=degree,
            constraint_variant="glued_bottom",
            quadrature_rule_id=RULE_BY_DEGREE[degree],
        ):
            raise DistributedCampaignError(
                "frozen distributed HDF5 input is absent or stale and would be "
                f"regenerated: {path}"
            )
        paths.add(path)
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise DistributedCampaignError(f"frozen input is missing: {missing[0]}")
    return {
        _relative_key(path): sha256_file(path)
        for path in sorted(paths, key=_relative_key)
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("h5py", "jax", "jaxlib", "mpi4py", "numpy", "petsc4py", "scipy"):
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _capture_environment(python: Path, mpiexec: Path) -> dict[str, object]:
    version = subprocess.run(
        [str(mpiexec), "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "captured_at_utc": utc_now_iso(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "python_executable": str(python),
        "python_executable_sha256": sha256_file(python),
        "mpi_launcher": str(mpiexec),
        "mpi_launcher_sha256": sha256_file(mpiexec),
        "mpi_launcher_version": (version.stdout or version.stderr).strip(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": _package_versions(),
        "cpu_affinity": (
            sorted(int(value) for value in os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else []
        ),
        "thread_environment": dict(THREAD_ENVIRONMENT),
    }


def _tree_hashes(root: Path, *, exclude: set[Path] | None = None) -> dict[str, str]:
    excluded = {path.resolve() for path in (exclude or set())}
    hashes: dict[str, str] = {}
    if not root.is_dir():
        return hashes
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() in excluded or path.name.endswith(".tmp"):
            continue
        hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def _output_hash_closure(out_root: Path) -> dict[str, object]:
    hashes = _tree_hashes(out_root, exclude={out_root / MANIFEST_NAME})
    return {
        "algorithm": "sha256",
        "scope": "all_regular_files_below_output_root_except_manifest",
        "excluded_paths": [MANIFEST_NAME],
        "file_count": len(hashes),
        "files": hashes,
        "files_map_sha256": _json_sha256(hashes),
    }


def _verify_hash_inventory(inventory: Mapping[str, str]) -> list[str]:
    errors: list[str] = []
    for raw, expected in inventory.items():
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            errors.append(f"missing: {raw}")
        elif sha256_file(path) != expected:
            errors.append(f"hash mismatch: {raw}")
    return errors


@dataclass(frozen=True, slots=True)
class Block:
    degree: int
    state_label: str
    state_amplitude: float
    ranks: int
    route_order: tuple[str, ...]

    @property
    def block_id(self) -> str:
        return f"p{self.degree}_{self.state_label}_np{self.ranks}"


def build_blocks() -> tuple[Block, ...]:
    blocks: list[Block] = []
    index = 0
    for degree in DEGREES:
        for state_label, amplitude in STATES:
            for ranks in RANKS:
                shift = index % len(ROUTES)
                route_order = ROUTES[shift:] + ROUTES[:shift]
                blocks.append(
                    Block(
                        degree=degree,
                        state_label=state_label,
                        state_amplitude=float(amplitude),
                        ranks=ranks,
                        route_order=route_order,
                    )
                )
                index += 1
    return tuple(blocks)


def _relative_error(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return float("inf"), float("inf")
    absolute = float(np.linalg.norm(left - right))
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), np.finfo(float).tiny)
    return absolute, float(absolute / scale)


def _load_route(route_dir: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    payload = _read_strict_json(route_dir / "output.json")
    if not isinstance(payload, dict):
        raise ValueError("route output must be a JSON object")
    with np.load(route_dir / "tangent_action.npz", allow_pickle=False) as archive:
        arrays = {
            "state": np.asarray(archive["state"], dtype=np.float64),
            "tangent_actions": np.asarray(archive["tangent_actions"], dtype=np.float64),
            "gradient": np.asarray(archive["gradient"], dtype=np.float64),
        }
    return payload, arrays


def _validate_child_identity(
    errors: list[str],
    *,
    block: Block,
    route: str,
    route_position: int,
    payload: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    source_commit: str,
    run_id: str,
    expected_child_argv: Sequence[str],
) -> None:
    label = f"{block.block_id}/{route}"
    exact = {
        "experiment_id": "EXP-ROUTE-001",
        "tier": "fixed_state_screen",
        "status": "completed",
        "route": route,
        "mesh_name": "hetero_ssr_L1",
        "element_degree": block.degree,
        "quadrature_rule_id": RULE_BY_DEGREE[block.degree],
        "constraint_variant": "glued_bottom",
        "lambda_target": 1.55,
        "state_family": "analytic_mesh_field_v1",
        "state_label": block.state_label,
        "state_amplitude": block.state_amplitude,
        "probe_count": 4,
        "mpi_ranks": block.ranks,
        "warmup_repetitions": 1,
        "measured_repetitions": 5,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            errors.append(f"{label}: child identity field {key} differs from the plan")

    design = payload.get("comparison_design")
    if not isinstance(design, dict):
        errors.append(f"{label}: comparison_design is missing")
    else:
        expected_design = {
            "comparison_id": block.block_id,
            "block_repetition": 1,
            "route_order_position": route_position,
            "route_order_policy": "local_distributed_correctness_v2",
            "timing_reduction": "mpi_collective_max",
            "independent_process_block": True,
        }
        for key, expected in expected_design.items():
            if design.get(key) != expected:
                errors.append(f"{label}: child comparison field {key} differs from the plan")

    git = payload.get("git")
    if not isinstance(git, dict):
        errors.append(f"{label}: child Git identity is missing")
    elif git.get("commit") != source_commit or git.get("dirty") is not False:
        errors.append(f"{label}: child Git identity differs from the clean campaign source")
    job = payload.get("job_metadata")
    if not isinstance(job, dict) or job.get("workstation_run_id") != run_id:
        errors.append(f"{label}: child run identity differs from the campaign")

    try:
        child_argv = shlex.split(str(payload.get("command", "")))
    except ValueError:
        child_argv = []
    if len(child_argv) < 2 or child_argv[1:] != list(expected_child_argv):
        errors.append(f"{label}: child-recorded argv differs from the frozen command")

    state = np.asarray(arrays["state"], dtype=np.float64)
    actions = np.asarray(arrays["tangent_actions"], dtype=np.float64)
    gradient = np.asarray(arrays["gradient"], dtype=np.float64)
    if state.ndim != 1 or gradient.shape != state.shape:
        errors.append(f"{label}: state and gradient arrays have inconsistent shapes")
    if actions.ndim != 2 or actions.shape != (4, state.size):
        errors.append(f"{label}: tangent-action array must have shape (4, number of dofs)")
    if not all(np.all(np.isfinite(value)) for value in (state, gradient, actions)):
        errors.append(f"{label}: saved arrays contain non-finite values")
    first_action = actions[0] if actions.ndim == 2 and actions.shape[0] else actions
    for key, values in (
        ("state_sha256", state),
        ("gradient_sha256", gradient),
        ("action_sha256", first_action),
    ):
        if payload.get(key) != _array_sha256(values):
            errors.append(f"{label}: {key} does not bind the saved array")
    action_hashes = [_array_sha256(row) for row in actions] if actions.ndim == 2 else []
    if payload.get("action_sha256_by_probe") != action_hashes:
        errors.append(f"{label}: action_sha256_by_probe does not bind all probes")


def _compare(
    errors: list[str],
    maxima: dict[str, float],
    *,
    label: str,
    left: np.ndarray,
    right: np.ndarray,
    exact: bool = False,
) -> None:
    if exact:
        if not np.array_equal(left, right):
            errors.append(f"{label}: exact equality failed")
        return
    absolute, relative = _relative_error(left, right)
    maxima[f"{label}.absolute"] = max(maxima.get(f"{label}.absolute", 0.0), absolute)
    maxima[f"{label}.relative"] = max(maxima.get(f"{label}.relative", 0.0), relative)
    if relative > RELATIVE_TOLERANCE and absolute > ABSOLUTE_TOLERANCE:
        errors.append(
            f"{label}: absolute error {absolute:.6e}, relative error {relative:.6e}"
        )


def validate_campaign(
    out_root: Path,
    *,
    source_commit: str = "",
    run_id: str = "",
    commands: Mapping[tuple[str, str], Sequence[str]] | None = None,
) -> dict[str, object]:
    errors: list[str] = []
    maxima: dict[str, float] = {}
    loaded: dict[tuple[int, str, int, str], tuple[dict[str, object], dict[str, np.ndarray]]] = {}
    for block in build_blocks():
        route_data: dict[str, tuple[dict[str, object], dict[str, np.ndarray]]] = {}
        for route in ROUTES:
            route_dir = out_root / "blocks" / block.block_id / route
            try:
                payload, arrays = _load_route(route_dir)
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{block.block_id}/{route}: unreadable output ({exc})")
                continue
            route_data[route] = (payload, arrays)
            loaded[(block.degree, block.state_label, block.ranks, route)] = (payload, arrays)
            if payload.get("status") != "completed" or payload.get("route") != route:
                errors.append(f"{block.block_id}/{route}: invalid terminal identity")
            if int(payload.get("mpi_ranks", -1)) != block.ranks:
                errors.append(f"{block.block_id}/{route}: MPI rank count mismatch")
            if int(payload.get("probe_count", -1)) != 4 or arrays["tangent_actions"].shape[0] != 4:
                errors.append(f"{block.block_id}/{route}: four tangent probes are required")
            rank_rows = payload.get("rank_summaries")
            if not isinstance(rank_rows, list) or len(rank_rows) != block.ranks:
                errors.append(f"{block.block_id}/{route}: incomplete rank summaries")
            else:
                rank_ids = sorted(int(row.get("rank", -1)) for row in rank_rows if isinstance(row, dict))
                owned = sum(int(row.get("owned_dofs", 0)) for row in rank_rows if isinstance(row, dict))
                expected_owned = int(payload.get("model_covariates", {}).get("global_free_dofs", -1))
                if rank_ids != list(range(block.ranks)) or owned != expected_owned:
                    errors.append(f"{block.block_id}/{route}: ownership partition is inconsistent")
            if source_commit and commands is not None:
                command = list(commands[(block.block_id, route)])
                _validate_child_identity(
                    errors,
                    block=block,
                    route=route,
                    route_position=block.route_order.index(route),
                    payload=payload,
                    arrays=arrays,
                    source_commit=source_commit,
                    run_id=run_id,
                    expected_child_argv=command[5:],
                )

        if set(route_data) != set(ROUTES):
            continue
        reference_payload, reference_arrays = route_data["element_ad"]
        for route in ("colored_sfd", "constitutive_ad"):
            payload, arrays = route_data[route]
            prefix = f"{block.block_id}/{route}-vs-element"
            _compare(errors, maxima, label=f"{prefix}/state", left=arrays["state"], right=reference_arrays["state"], exact=True)
            _compare(errors, maxima, label=f"{prefix}/gradient", left=arrays["gradient"], right=reference_arrays["gradient"])
            _compare(errors, maxima, label=f"{prefix}/actions", left=arrays["tangent_actions"], right=reference_arrays["tangent_actions"])
            if payload.get("branch_diagnostics") != reference_payload.get("branch_diagnostics"):
                errors.append(f"{prefix}: branch diagnostics differ")
        if block.ranks == 1:
            matrices: dict[str, dict[str, np.ndarray]] = {}
            for route in ROUTES:
                matrix_path = out_root / "blocks" / block.block_id / route / "tangent_matrix_csr.npz"
                try:
                    with np.load(matrix_path, allow_pickle=False) as matrix:
                        matrices[route] = {
                            "indptr": np.asarray(matrix["indptr"], dtype=np.int64),
                            "indices": np.asarray(matrix["indices"], dtype=np.int64),
                            "values": np.asarray(matrix["values"], dtype=np.float64),
                            "shape": np.asarray(matrix["shape"], dtype=np.int64),
                        }
                except (OSError, KeyError, ValueError) as exc:
                    errors.append(f"{block.block_id}/{route}: missing direct CSR ({exc})")
            if set(matrices) == set(ROUTES):
                reference = matrices["element_ad"]
                for route in ("colored_sfd", "constitutive_ad"):
                    candidate = matrices[route]
                    prefix = f"{block.block_id}/{route}-vs-element/csr"
                    for field in ("indptr", "indices", "shape"):
                        _compare(errors, maxima, label=f"{prefix}/{field}", left=candidate[field], right=reference[field], exact=True)
                    _compare(errors, maxima, label=f"{prefix}/values", left=candidate["values"], right=reference["values"])

    for degree in DEGREES:
        for state_label, _amplitude in STATES:
            for route in ROUTES:
                reference = loaded.get((degree, state_label, 1, route))
                if reference is None:
                    continue
                reference_payload, reference_arrays = reference
                for ranks in (2, 4):
                    candidate = loaded.get((degree, state_label, ranks, route))
                    if candidate is None:
                        continue
                    payload, arrays = candidate
                    prefix = f"p{degree}_{state_label}/{route}/np{ranks}-vs-np1"
                    _compare(errors, maxima, label=f"{prefix}/state", left=arrays["state"], right=reference_arrays["state"], exact=True)
                    _compare(errors, maxima, label=f"{prefix}/gradient", left=arrays["gradient"], right=reference_arrays["gradient"])
                    _compare(errors, maxima, label=f"{prefix}/actions", left=arrays["tangent_actions"], right=reference_arrays["tangent_actions"])
                    if payload.get("branch_diagnostics") != reference_payload.get("branch_diagnostics"):
                        errors.append(f"{prefix}: branch diagnostics differ across ranks")

    return {
        "schema_id": VERIFICATION_SCHEMA_ID,
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "passed" if not errors else "failed",
        "comparison_tolerances": {
            "relative": RELATIVE_TOLERANCE,
            "absolute": ABSOLUTE_TOLERANCE,
            "integer_topology_and_state": "exact",
        },
        "planned_blocks": len(build_blocks()),
        "planned_route_processes": len(build_blocks()) * len(ROUTES),
        "maximum_observed_errors": dict(sorted(maxima.items())),
        "errors": errors,
        "timing_claim_admissible": False,
    }


def _command(
    block: Block,
    route: str,
    route_position: int,
    route_dir: Path,
    *,
    python: Path,
    mpiexec: Path,
) -> list[str]:
    command = [
        str(mpiexec),
        "--oversubscribe",
        "-n",
        str(block.ranks),
        str(python),
        "experiments/runners/run_plasticity3d_fixed_state_route_screen.py",
        "--route",
        route,
        "--tier",
        "fixed_state_screen",
        "--mesh-name",
        "hetero_ssr_L1",
        "--element-degree",
        str(block.degree),
        "--quadrature-rule",
        RULE_BY_DEGREE[block.degree],
        "--constraint-variant",
        "glued_bottom",
        "--lambda-target",
        "1.55",
        "--state-label",
        block.state_label,
        "--state-amplitude",
        str(block.state_amplitude),
        "--warmup-repetitions",
        "1",
        "--measured-repetitions",
        "5",
        "--probe-count",
        "4",
        "--comparison-id",
        block.block_id,
        "--block-repetition",
        "1",
        "--route-order-position",
        str(route_position),
        "--route-order-policy",
        "local_distributed_correctness_v2",
        "--ksp-rtol",
        "1e-8",
        "--ksp-max-it",
        "500",
        "--output",
        str(route_dir / "output.json"),
        "--action-out",
        str(route_dir / "tangent_action.npz"),
    ]
    if block.ranks == 1:
        command.append("--save-direct-matrix")
    return command


def _normalize_command(
    command: Sequence[str], *, out_root: Path, python: Path, mpiexec: Path
) -> list[str]:
    normalized: list[str] = []
    for index, raw in enumerate(command):
        token = str(raw)
        if index == 0 and Path(token).resolve() == mpiexec.resolve():
            normalized.append("${MPIEXEC}")
            continue
        if index == 4 and Path(token).resolve() == python.resolve():
            normalized.append("${PYTHON}")
            continue
        candidate = Path(token)
        if candidate.is_absolute():
            resolved = candidate.resolve(strict=False)
            try:
                relative = resolved.relative_to(out_root)
            except ValueError:
                try:
                    relative = resolved.relative_to(REPO_ROOT)
                except ValueError:
                    normalized.append(token)
                else:
                    normalized.append("${REPO_ROOT}/" + relative.as_posix())
            else:
                normalized.append("${OUTPUT_ROOT}/" + relative.as_posix())
        else:
            normalized.append(token)
    return normalized


def _route_hash_closure(route_dir: Path) -> dict[str, object]:
    hashes = _tree_hashes(route_dir, exclude={route_dir / PROCESS_RECORD_NAME})
    return {
        "algorithm": "sha256",
        "scope": "all_regular_files_below_route_directory_except_process_record",
        "excluded_paths": [PROCESS_RECORD_NAME],
        "file_count": len(hashes),
        "files": hashes,
        "files_map_sha256": _json_sha256(hashes),
    }


def _write_process_record(route_dir: Path, record: dict[str, object]) -> dict[str, object]:
    record["artifact_hash_closure"] = _route_hash_closure(route_dir)
    path = route_dir / PROCESS_RECORD_NAME
    atomic_write_json(path, record)
    returned = dict(record)
    returned["process_record"] = path.name
    returned["process_record_sha256"] = sha256_file(path)
    return returned


def run_campaign(args: argparse.Namespace) -> dict[str, object]:
    raw_out_root = Path(args.out_root).expanduser()
    if not raw_out_root.is_absolute():
        raw_out_root = Path.cwd() / raw_out_root
    out_root = raw_out_root.resolve(strict=False)
    if os.path.lexists(raw_out_root) or os.path.lexists(out_root):
        raise FileExistsError(f"refusing to reuse output root {out_root}")
    if str(args.run_kind) == "publication":
        reproduction_root = (REPO_ROOT / "artifacts/reproduction").resolve()
        try:
            relative_output = out_root.relative_to(reproduction_root)
        except ValueError as exc:
            raise DistributedCampaignError(
                "publication output root must be below artifacts/reproduction"
            ) from exc
        if not relative_output.parts:
            raise DistributedCampaignError(
                "publication output root must name one campaign below artifacts/reproduction"
            )

    run_id = str(args.run_id or uuid.uuid4())
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise DistributedCampaignError(
            "--run-id must use only letters, digits, dot, underscore, and hyphen"
        )
    preflight = check_experiment_preflight(
        REPO_ROOT,
        run_kind=str(args.run_kind),
        pilot_dirty_override=bool(args.pilot_dirty_override),
        pilot_override_reason=args.pilot_override_reason,
    )
    git = _git_metadata()
    if git.get("commit") != preflight.git_commit or bool(git.get("dirty")) != (
        not preflight.git_clean
    ):
        raise DistributedCampaignError("Git preflight and frozen source identity disagree")
    expected_commit = str(args.expected_commit or "").strip()
    if expected_commit and COMMIT_RE.fullmatch(expected_commit) is None:
        raise DistributedCampaignError(
            "--expected-commit must be a full 40--64 digit hexadecimal hash"
        )
    if expected_commit and expected_commit.lower() != preflight.git_commit.lower():
        raise DistributedCampaignError("current HEAD differs from --expected-commit")
    if args.execute:
        if os.environ.get("LOCAL_DISTRIBUTED_RUN_CONFIRMED") != "YES":
            raise DistributedCampaignError(
                "execution requires LOCAL_DISTRIBUTED_RUN_CONFIRMED=YES"
            )
    if args.execute and str(args.run_kind) == "publication":
        if not expected_commit:
            raise DistributedCampaignError(
                "publication execution requires --expected-commit"
            )
    if int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) != 1:
        raise DistributedCampaignError("campaign driver itself must run as one process")

    python = _resolve_python(args.python)
    mpiexec = _resolve_mpiexec(args.mpiexec)
    code_hashes = _collect_code_hashes()
    configuration_hashes = _collect_configuration_hashes()
    input_hashes = _collect_input_hashes()
    environment_record = _capture_environment(python, mpiexec)
    command_map: dict[tuple[str, str], list[str]] = {}
    command_plan: list[dict[str, object]] = []
    for block in build_blocks():
        for position, route in enumerate(block.route_order):
            route_dir = out_root / "blocks" / block.block_id / route
            command = _command(
                block,
                route,
                position,
                route_dir,
                python=python,
                mpiexec=mpiexec,
            )
            command_map[(block.block_id, route)] = command
            normalized = _normalize_command(
                command, out_root=out_root, python=python, mpiexec=mpiexec
            )
            command_plan.append(
                {
                    "block_id": block.block_id,
                    "route": route,
                    "route_order_position": position,
                    "normalized_argv": normalized,
                    "normalized_argv_sha256": _json_sha256(normalized),
                }
            )
    if len(command_plan) != 36:
        raise DistributedCampaignError("frozen command plan must contain 36 route processes")

    out_root.mkdir(parents=True, exist_ok=False)
    environment_path = out_root / "environment.json"
    atomic_write_json(environment_path, environment_record)
    created_at_utc = utc_now_iso()
    plan: dict[str, object] = {
        "schema_id": PLAN_SCHEMA_ID,
        "schema_version": PLAN_SCHEMA_VERSION,
        "experiment_id": "EXP-DIST-001",
        "run_id": run_id,
        "created_at_utc": created_at_utc,
        "source_commit": preflight.git_commit,
        "source_clean": preflight.git_clean,
        "source_tree": git.get("tree", ""),
        "source_branch": git.get("branch", ""),
        "source_status_porcelain": git.get("status_porcelain", []),
        "expected_commit": expected_commit,
        "run_kind": str(args.run_kind),
        "blocks": [
            {
                "block_id": block.block_id,
                "degree": block.degree,
                "state_label": block.state_label,
                "state_amplitude": block.state_amplitude,
                "ranks": block.ranks,
                "route_order": list(block.route_order),
            }
            for block in build_blocks()
        ],
        "normalized_commands": command_plan,
        "normalized_commands_sha256": _json_sha256(command_plan),
        "comparison_tolerances": {
            "relative": RELATIVE_TOLERANCE,
            "absolute": ABSOLUTE_TOLERANCE,
            "integer_topology_and_state": "exact",
        },
        "timing_claim_admissible": False,
    }
    atomic_write_json(out_root / "plan.json", plan)
    manifest: dict[str, object] = {
        "schema_id": CAMPAIGN_SCHEMA_ID,
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "experiment_id": "EXP-DIST-001",
        "run_kind": str(args.run_kind),
        "run_id": run_id,
        "status": "running" if args.execute else "prepared_not_executed",
        "created_at_utc": created_at_utc,
        "source_commit": preflight.git_commit,
        "source_clean": preflight.git_clean,
        "source_tree": git.get("tree", ""),
        "source_branch": git.get("branch", ""),
        "source_status_porcelain": git.get("status_porcelain", []),
        "expected_commit": expected_commit,
        "plan_path": "plan.json",
        "plan_sha256": sha256_file(out_root / "plan.json"),
        "environment_path": environment_path.name,
        "environment_sha256": sha256_file(environment_path),
        "code_hashes": code_hashes,
        "code_hashes_sha256": _json_sha256(code_hashes),
        "configuration_hashes": configuration_hashes,
        "configuration_hashes_sha256": _json_sha256(configuration_hashes),
        "input_hashes": input_hashes,
        "input_hashes_sha256": _json_sha256(input_hashes),
        "normalized_commands_sha256": _json_sha256(command_plan),
        "planned_blocks": 12,
        "planned_route_processes": 36,
        "process_records": [],
        "timing_claim_admissible": False,
        "output_hash_closure": {"status": "open_during_execution"},
    }
    if not args.execute:
        manifest["output_hash_closure"] = _output_hash_closure(out_root)
        atomic_write_json(out_root / MANIFEST_NAME, manifest)
        return manifest

    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    environment["WORKSTATION_RUN_ID"] = run_id
    records: list[dict[str, object]] = []
    atomic_write_json(out_root / MANIFEST_NAME, manifest)
    try:
        for block in build_blocks():
            for position, route in enumerate(block.route_order):
                route_dir = out_root / "blocks" / block.block_id / route
                route_dir.mkdir(parents=True, exist_ok=False)
                command = command_map[(block.block_id, route)]
                normalized = _normalize_command(
                    command, out_root=out_root, python=python, mpiexec=mpiexec
                )
                (route_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
                started_at_utc = utc_now_iso()
                started = time.perf_counter()
                with (route_dir / "stdout.txt").open("xb") as stdout, (route_dir / "stderr.txt").open("xb") as stderr:
                    completed = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                    )
                wall_time_s = float(time.perf_counter() - started)
                record = _write_process_record(
                    route_dir,
                    {
                        "schema_id": "fenics-nonlinear-energies.exp-dist-001-route-process",
                        "schema_version": 1,
                        "experiment_id": "EXP-DIST-001",
                        "run_kind": str(args.run_kind),
                        "run_id": run_id,
                        "block_id": block.block_id,
                        "route": route,
                        "route_order_position": position,
                        "status": "completed" if completed.returncode == 0 else "failed",
                        "returncode": int(completed.returncode),
                        "started_at_utc": started_at_utc,
                        "finished_at_utc": utc_now_iso(),
                        "launcher_wall_time_s": wall_time_s,
                        "timing_claim_admissible": False,
                        "source_commit": preflight.git_commit,
                        "command_argv": command,
                        "normalized_command_argv": normalized,
                        "normalized_command_sha256": _json_sha256(normalized),
                    },
                )
                records.append(record)
                manifest["process_records"] = records
                atomic_write_json(out_root / MANIFEST_NAME, manifest)
                if completed.returncode != 0:
                    raise DistributedCampaignError(
                        f"route process failed: {block.block_id}/{route}; see stderr.txt"
                    )
        verification = validate_campaign(
            out_root,
            source_commit=preflight.git_commit,
            run_id=run_id,
            commands=command_map,
        )
        atomic_write_json(out_root / "verification_summary.json", verification)
        terminal_git = _git_metadata()
        frozen_errors = {
            label: errors
            for label, errors in (
                ("code", _verify_hash_inventory(code_hashes)),
                ("configuration", _verify_hash_inventory(configuration_hashes)),
                ("input", _verify_hash_inventory(input_hashes)),
            )
            if errors
        }
        manifest.update(
            {
                "status": (
                    "completed"
                    if verification["status"] == "passed"
                    and not frozen_errors
                    and terminal_git.get("commit") == preflight.git_commit
                    and terminal_git.get("dirty") is False
                    else "failed"
                ),
                "finished_at_utc": utc_now_iso(),
                "terminal_source": terminal_git,
                "terminal_frozen_hash_verification": {
                    "passed": not frozen_errors,
                    "errors": frozen_errors,
                },
                "verification_summary": "verification_summary.json",
                "verification_sha256": sha256_file(
                    out_root / "verification_summary.json"
                ),
            }
        )
        manifest["output_hash_closure"] = _output_hash_closure(out_root)
        atomic_write_json(out_root / MANIFEST_NAME, manifest)
        if manifest["status"] != "completed":
            raise DistributedCampaignError("distributed route verification failed")
        return manifest
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = utc_now_iso()
        manifest["runner_exception"] = f"{type(exc).__name__}: {exc}"
        manifest["process_records"] = records
        manifest["output_hash_closure"] = _output_hash_closure(out_root)
        atomic_write_json(out_root / MANIFEST_NAME, manifest)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--python", default="./.venv/bin/python")
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--expected-commit", default="")
    parser.add_argument("--run-kind", choices=("publication", "pilot"), default="publication")
    parser.add_argument("--pilot-dirty-override", action="store_true")
    parser.add_argument("--pilot-override-reason")
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        result = run_campaign(args)
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

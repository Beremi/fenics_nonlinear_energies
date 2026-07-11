#!/usr/bin/env python3
"""Independent admission for the local distributed colored-recovery evidence.

This module deliberately does not import the experiment launcher.  It reparses
the campaign envelope, verifies every recorded SHA-256 closure, reloads the
NumPy arrays and rank-one CSR matrices, and recomputes the numerical gates.
Timing samples are outside this correctness-only admission contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from typing import Mapping, Sequence

import numpy as np


SCHEMA_ID = "fenics-nonlinear-energies.distributed-colored-paper-admission"
SCHEMA_VERSION = 1
CAMPAIGN_SCHEMA_ID = "fenics-nonlinear-energies.exp-dist-001-colored-recovery-manifest"
PLAN_SCHEMA_ID = "fenics-nonlinear-energies.exp-dist-001-colored-recovery-plan"
VERIFICATION_SCHEMA_ID = "fenics-nonlinear-energies.exp-dist-001-colored-recovery"
PROCESS_SCHEMA_ID = "fenics-nonlinear-energies.exp-dist-001-route-process"
ROUTES = ("element_ad", "colored_sfd", "constitutive_ad")
DEGREES = (1, 2)
RANKS = (1, 2, 4)
STATES = (("elastic", 2.0e-4), ("mixed", 2.0e-2))
RULE_BY_DEGREE = {1: "tetra_1point", 2: "tetra_11point"}
RELATIVE_TOLERANCE = 1.0e-8
ABSOLUTE_TOLERANCE = 1.0e-10
EXPECTED_THREAD_ENVIRONMENT = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "True",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
}
TABLE_NAME = "distributed_colored_verification.tex"
MANIFEST_NAME = "distributed_colored_manifest.json"


class AdmissionError(ValueError):
    """Raised when distributed-colored evidence is not publication-admissible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def read_strict_json(path: Path) -> object:
    def reject_constant(value: str) -> None:
        raise AdmissionError(f"{path}: non-finite JSON constant {value!r}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"cannot read strict JSON {path}: {exc}") from exc


def _object(path: Path) -> dict[str, object]:
    value = read_strict_json(path)
    if not isinstance(value, dict):
        raise AdmissionError(f"{path}: JSON payload must be an object")
    return value


def _relative_file(
    raw: object,
    *,
    root: Path,
    label: str,
    expected: str | None = None,
) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute() or ".." in Path(raw).parts:
        raise AdmissionError(f"{label} must be a safe relative path")
    path = (root / raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise AdmissionError(f"{label} resolves outside its root") from exc
    if expected is not None and Path(raw).as_posix() != expected:
        raise AdmissionError(f"{label} must equal {expected!r}")
    if not path.is_file():
        raise AdmissionError(f"{label} is missing: {path}")
    return path


def _safe_repo_file(raw: object, *, repo_root: Path, label: str) -> Path:
    return _relative_file(raw, root=repo_root, label=label)


def _tree_hashes(root: Path, *, excluded: set[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        hashes[relative] = sha256_file(path)
    return hashes


def verify_tree_closure(
    root: Path,
    closure: object,
    *,
    expected_scope: str,
    expected_excluded: Sequence[str],
    label: str,
) -> dict[str, str]:
    if not isinstance(closure, dict):
        raise AdmissionError(f"{label}: hash closure is missing")
    if closure.get("algorithm") != "sha256" or closure.get("scope") != expected_scope:
        raise AdmissionError(f"{label}: hash-closure algorithm or scope is invalid")
    excluded = closure.get("excluded_paths")
    if excluded != list(expected_excluded):
        raise AdmissionError(f"{label}: excluded path list is not canonical")
    recorded = closure.get("files")
    if not isinstance(recorded, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in recorded.items()
    ):
        raise AdmissionError(f"{label}: file-hash map is malformed")
    actual = _tree_hashes(root, excluded=set(expected_excluded))
    if recorded != actual:
        missing = sorted(set(recorded) - set(actual))
        extra = sorted(set(actual) - set(recorded))
        changed = sorted(
            key for key in set(actual) & set(recorded) if actual[key] != recorded[key]
        )
        raise AdmissionError(
            f"{label}: hash closure differs (missing={missing}, extra={extra}, changed={changed})"
        )
    if closure.get("file_count") != len(recorded):
        raise AdmissionError(f"{label}: file count differs from the hash map")
    if closure.get("files_map_sha256") != json_sha256(recorded):
        raise AdmissionError(f"{label}: file-map digest is invalid")
    return dict(recorded)


def _verify_inventory(
    manifest: Mapping[str, object],
    *,
    field: str,
    repo_root: Path,
) -> dict[str, str]:
    raw = manifest.get(field)
    if not isinstance(raw, dict) or not raw:
        raise AdmissionError(f"campaign {field} must be a nonempty object")
    inventory: dict[str, str] = {}
    for key, digest in sorted(raw.items()):
        if not isinstance(key, str) or not isinstance(digest, str) or len(digest) != 64:
            raise AdmissionError(f"campaign {field} contains a malformed entry")
        path = _safe_repo_file(key, repo_root=repo_root, label=f"{field}.{key}")
        if sha256_file(path) != digest:
            raise AdmissionError(f"campaign {field} hash is stale: {key}")
        inventory[key] = digest
    if manifest.get(f"{field}_sha256") != json_sha256(inventory):
        raise AdmissionError(f"campaign {field}_sha256 is invalid")
    return inventory


def _expected_blocks() -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    index = 0
    for degree in DEGREES:
        for state_label, amplitude in STATES:
            for ranks in RANKS:
                shift = index % len(ROUTES)
                blocks.append(
                    {
                        "block_id": f"p{degree}_{state_label}_np{ranks}",
                        "degree": degree,
                        "state_label": state_label,
                        "state_amplitude": float(amplitude),
                        "ranks": ranks,
                        "route_order": list(ROUTES[shift:] + ROUTES[:shift]),
                    }
                )
                index += 1
    return blocks


def _expected_normalized_command(
    block: Mapping[str, object], route: str, position: int
) -> list[str]:
    degree = int(block["degree"])
    block_id = str(block["block_id"])
    route_dir = f"${{OUTPUT_ROOT}}/blocks/{block_id}/{route}"
    command = [
        "${MPIEXEC}",
        "--oversubscribe",
        "-n",
        str(block["ranks"]),
        "${PYTHON}",
        "experiments/runners/run_plasticity3d_fixed_state_route_screen.py",
        "--route",
        route,
        "--tier",
        "fixed_state_screen",
        "--mesh-name",
        "hetero_ssr_L1",
        "--element-degree",
        str(degree),
        "--quadrature-rule",
        RULE_BY_DEGREE[degree],
        "--constraint-variant",
        "glued_bottom",
        "--lambda-target",
        "1.55",
        "--state-label",
        str(block["state_label"]),
        "--state-amplitude",
        str(block["state_amplitude"]),
        "--warmup-repetitions",
        "1",
        "--measured-repetitions",
        "5",
        "--probe-count",
        "4",
        "--comparison-id",
        block_id,
        "--block-repetition",
        "1",
        "--route-order-position",
        str(position),
        "--route-order-policy",
        "local_distributed_correctness_v2",
        "--ksp-rtol",
        "1e-8",
        "--ksp-max-it",
        "500",
        "--output",
        f"{route_dir}/output.json",
        "--action-out",
        f"{route_dir}/tangent_action.npz",
    ]
    if int(block["ranks"]) == 1:
        command.append("--save-direct-matrix")
    return command


def _validate_command_plan(plan: Mapping[str, object]) -> dict[tuple[str, str], list[str]]:
    blocks = plan.get("blocks")
    if blocks != _expected_blocks():
        raise AdmissionError("plan blocks differ from the frozen 12-block design")
    rows = plan.get("normalized_commands")
    if not isinstance(rows, list) or len(rows) != 36:
        raise AdmissionError("plan must contain exactly 36 normalized commands")
    expected_rows: list[dict[str, object]] = []
    commands: dict[tuple[str, str], list[str]] = {}
    for block in _expected_blocks():
        for position, route in enumerate(block["route_order"]):
            command = _expected_normalized_command(block, str(route), position)
            expected_rows.append(
                {
                    "block_id": block["block_id"],
                    "route": route,
                    "route_order_position": position,
                    "normalized_argv": command,
                    "normalized_argv_sha256": json_sha256(command),
                }
            )
            commands[(str(block["block_id"]), str(route))] = command
    if rows != expected_rows or plan.get("normalized_commands_sha256") != json_sha256(rows):
        raise AdmissionError("normalized command plan differs from the frozen design")
    return commands


def _git_is_ancestor(repo_root: Path, older: str) -> bool:
    current = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if current.returncode != 0:
        return False
    return subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", older, current.stdout.strip()],
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0


def validate_campaign_envelope(
    evidence_root: Path, *, repo_root: Path
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    evidence_root = evidence_root.resolve()
    reproduction = (repo_root / "artifacts/reproduction").resolve()
    try:
        relative_root = evidence_root.relative_to(reproduction)
    except ValueError as exc:
        raise AdmissionError("evidence root must be below artifacts/reproduction") from exc
    if not relative_root.parts or not evidence_root.is_dir():
        raise AdmissionError("evidence root must identify one existing campaign")

    manifest_path = evidence_root / "manifest.json"
    manifest = _object(manifest_path)
    exact_manifest = {
        "schema_id": CAMPAIGN_SCHEMA_ID,
        "schema_version": 2,
        "experiment_id": "EXP-DIST-001",
        "run_kind": "publication",
        "status": "completed",
        "source_clean": True,
        "timing_claim_admissible": False,
        "planned_blocks": 12,
        "planned_route_processes": 36,
    }
    for key, expected in exact_manifest.items():
        if manifest.get(key) != expected:
            raise AdmissionError(f"campaign manifest field {key} must equal {expected!r}")
    commit = str(manifest.get("source_commit", ""))
    if len(commit) not in {40, 64} or any(char not in "0123456789abcdefABCDEF" for char in commit):
        raise AdmissionError("campaign source_commit is not a full hexadecimal hash")
    if manifest.get("expected_commit") != commit:
        raise AdmissionError("campaign expected_commit differs from source_commit")
    for field in ("created_at_utc", "finished_at_utc"):
        timestamp = manifest.get(field)
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            raise AdmissionError(f"campaign {field} is missing or invalid")
    terminal = manifest.get("terminal_source")
    if not isinstance(terminal, dict) or terminal.get("commit") != commit or terminal.get("dirty") is not False:
        raise AdmissionError("terminal source identity differs from the clean launch identity")
    frozen = manifest.get("terminal_frozen_hash_verification")
    if not isinstance(frozen, dict) or frozen.get("passed") is not True or frozen.get("errors") != {}:
        raise AdmissionError("terminal frozen-hash verification did not pass")
    if not _git_is_ancestor(repo_root, commit):
        raise AdmissionError("campaign source commit is not an ancestor of current HEAD")

    plan_path = _relative_file(
        manifest.get("plan_path"), root=evidence_root, label="plan_path", expected="plan.json"
    )
    if manifest.get("plan_sha256") != sha256_file(plan_path):
        raise AdmissionError("plan SHA-256 differs from the campaign manifest")
    plan = _object(plan_path)
    if plan.get("schema_id") != PLAN_SCHEMA_ID or plan.get("schema_version") != 2:
        raise AdmissionError("campaign plan has the wrong schema")
    for key in (
        "experiment_id",
        "run_kind",
        "run_id",
        "source_commit",
        "source_clean",
        "created_at_utc",
    ):
        if plan.get(key) != manifest.get(key):
            raise AdmissionError(f"plan field {key} differs from the manifest")
    if plan.get("timing_claim_admissible") is not False:
        raise AdmissionError("plan must explicitly prohibit timing claims")
    commands = _validate_command_plan(plan)
    if manifest.get("normalized_commands_sha256") != plan.get("normalized_commands_sha256"):
        raise AdmissionError("manifest and plan command digests differ")

    environment_path = _relative_file(
        manifest.get("environment_path"),
        root=evidence_root,
        label="environment_path",
        expected="environment.json",
    )
    if manifest.get("environment_sha256") != sha256_file(environment_path):
        raise AdmissionError("environment SHA-256 differs from the manifest")
    environment = _object(environment_path)
    if environment.get("thread_environment") != EXPECTED_THREAD_ENVIRONMENT:
        raise AdmissionError("captured thread environment differs from the frozen CPU policy")
    packages = environment.get("packages")
    if not isinstance(packages, dict) or set(packages) != {
        "h5py", "jax", "jaxlib", "mpi4py", "numpy", "petsc4py", "scipy"
    }:
        raise AdmissionError("captured package-version inventory is incomplete")
    for executable, digest in (
        (environment.get("python_executable"), environment.get("python_executable_sha256")),
        (environment.get("mpi_launcher"), environment.get("mpi_launcher_sha256")),
    ):
        path = Path(str(executable))
        if not path.is_file() or sha256_file(path) != digest:
            raise AdmissionError(f"captured executable is missing or changed: {path}")

    inventories = {
        field: _verify_inventory(manifest, field=field, repo_root=repo_root)
        for field in ("code_hashes", "configuration_hashes", "input_hashes")
    }
    production_requirements = {
        "code_hashes": {
            "experiments/runners/run_local_distributed_route_verification.py",
            "experiments/runners/run_plasticity3d_fixed_state_route_screen.py",
            "experiments/runners/run_plasticity3d_backend_mix_case.py",
        },
        "configuration_hashes": {"paper/protocols/EXP-DIST-001.md"},
        "input_hashes": {
            "data/meshes/SlopeStability3D/hetero_ssr/SSR_hetero_ada_L1.msh",
            "data/meshes/SlopeStability3D/hetero_ssr/definition.py",
            "data/meshes/SlopeStability3D/hetero_ssr/hetero_ssr_L1_p1_same_mesh_glued_bottom.h5",
            "data/meshes/SlopeStability3D/hetero_ssr/hetero_ssr_L1_p2_same_mesh_glued_bottom.h5",
        },
    }
    for field, required in production_requirements.items():
        # Unit fixtures and extracted archives may not contain a source tree.
        # In a live repository, every required path exists and omission is a
        # publication-blocking incomplete inventory.
        if all((repo_root / path).is_file() for path in required):
            missing = sorted(required - set(inventories[field]))
            if missing:
                raise AdmissionError(f"campaign {field} omits required files: {missing}")
    tracked_src = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--", "src"],
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked_src.returncode == 0:
        expected_src = {line for line in tracked_src.stdout.splitlines() if line}
        missing_src = sorted(expected_src - set(inventories["code_hashes"]))
        if missing_src:
            raise AdmissionError(
                f"campaign code_hashes omits tracked source files: {missing_src[:5]}"
            )
    process_rows = manifest.get("process_records")
    if not isinstance(process_rows, list) or len(process_rows) != 36:
        raise AdmissionError("manifest must bind exactly 36 completed process records")
    by_key: dict[tuple[str, str], dict[str, object]] = {}
    for row in process_rows:
        if not isinstance(row, dict):
            raise AdmissionError("process-record inventory contains a non-object")
        key = (str(row.get("block_id", "")), str(row.get("route", "")))
        if key in by_key or key not in commands:
            raise AdmissionError(f"unexpected or duplicate process record: {key}")
        if (
            row.get("schema_id") != PROCESS_SCHEMA_ID
            or row.get("schema_version") != 1
            or row.get("experiment_id") != "EXP-DIST-001"
            or row.get("run_kind") != "publication"
            or row.get("run_id") != manifest.get("run_id")
            or row.get("source_commit") != commit
            or row.get("status") != "completed"
            or row.get("returncode") != 0
            or row.get("timing_claim_admissible") is not False
        ):
            raise AdmissionError(f"process record has invalid terminal identity: {key}")
        for field in ("started_at_utc", "finished_at_utc"):
            timestamp = row.get(field)
            if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
                raise AdmissionError(f"process record has invalid {field}: {key}")
        launcher_wall = row.get("launcher_wall_time_s")
        if (
            isinstance(launcher_wall, bool)
            or not isinstance(launcher_wall, (int, float))
            or not np.isfinite(float(launcher_wall))
            or float(launcher_wall) < 0.0
        ):
            raise AdmissionError(f"process record has invalid launcher wall time: {key}")
        if row.get("normalized_command_argv") != commands[key]:
            raise AdmissionError(f"process normalized command differs from the plan: {key}")
        if row.get("normalized_command_sha256") != json_sha256(commands[key]):
            raise AdmissionError(f"process normalized command hash is invalid: {key}")
        record_path = evidence_root / "blocks" / key[0] / key[1] / "process_record.json"
        if row.get("process_record") != "process_record.json" or not record_path.is_file():
            raise AdmissionError(f"process-record file is missing: {key}")
        if row.get("process_record_sha256") != sha256_file(record_path):
            raise AdmissionError(f"process-record file hash differs: {key}")
        record = _object(record_path)
        for field, value in record.items():
            if row.get(field) != value:
                raise AdmissionError(f"manifest copy differs from process record {key}: {field}")
        route_dir = record_path.parent
        verify_tree_closure(
            route_dir,
            record.get("artifact_hash_closure"),
            expected_scope="all_regular_files_below_route_directory_except_process_record",
            expected_excluded=["process_record.json"],
            label=f"process {key}",
        )
        try:
            child_argv = shlex.split(str(_object(route_dir / "output.json").get("command", "")))
        except ValueError as exc:
            raise AdmissionError(f"child command is not valid shell syntax: {key}") from exc
        raw_command = row.get("command_argv")
        if not isinstance(raw_command, list) or len(raw_command) < 6:
            raise AdmissionError(f"process raw command is malformed: {key}")
        materialized: list[str] = []
        for token in commands[key]:
            if token == "${MPIEXEC}":
                materialized.append(str(environment["mpi_launcher"]))
            elif token == "${PYTHON}":
                materialized.append(str(environment["python_executable"]))
            elif token.startswith("${OUTPUT_ROOT}/"):
                materialized.append(
                    str(evidence_root / token.removeprefix("${OUTPUT_ROOT}/"))
                )
            else:
                materialized.append(token)
        if raw_command != materialized:
            raise AdmissionError(f"process raw command differs from the frozen plan: {key}")
        command_text = (route_dir / "command.txt").read_text(encoding="utf-8")
        if command_text != shlex.join(raw_command) + "\n":
            raise AdmissionError(f"command.txt differs from the process record: {key}")
        if child_argv[1:] != raw_command[5:]:
            raise AdmissionError(f"child-recorded argv differs from its process record: {key}")
        by_key[key] = row
    if set(by_key) != set(commands):
        raise AdmissionError("process-record inventory is incomplete")

    verification_path = _relative_file(
        manifest.get("verification_summary"),
        root=evidence_root,
        label="verification_summary",
        expected="verification_summary.json",
    )
    if manifest.get("verification_sha256") != sha256_file(verification_path):
        raise AdmissionError("verification-summary hash differs from the manifest")
    verification = _object(verification_path)
    if (
        verification.get("schema_id") != VERIFICATION_SCHEMA_ID
        or verification.get("schema_version") != 2
        or verification.get("status") != "passed"
        or verification.get("errors") != []
        or verification.get("timing_claim_admissible") is not False
    ):
        raise AdmissionError("runner verification summary is not a passed correctness-only record")

    output_files = verify_tree_closure(
        evidence_root,
        manifest.get("output_hash_closure"),
        expected_scope="all_regular_files_below_output_root_except_manifest",
        expected_excluded=["manifest.json"],
        label="campaign",
    )
    return {
        "source_commit": commit,
        "run_id": manifest.get("run_id"),
        "campaign_manifest_sha256": sha256_file(manifest_path),
        "output_files": output_files,
        "output_files_sha256": json_sha256(output_files),
        "inventories": inventories,
        "commands": commands,
    }


def _relative_errors(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise AdmissionError("compared arrays have incompatible shapes or non-finite entries")
    absolute = float(np.linalg.norm(left - right))
    denominator = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(np.float64).tiny,
    )
    return absolute, float(absolute / denominator)


def _require_close(left: np.ndarray, right: np.ndarray, *, label: str) -> float:
    absolute, relative = _relative_errors(left, right)
    if relative > RELATIVE_TOLERANCE and absolute > ABSOLUTE_TOLERANCE:
        raise AdmissionError(
            f"{label} exceeds the mixed tolerance: absolute={absolute:.6e}, relative={relative:.6e}"
        )
    return relative


def _load_csr(path: Path, *, route: str, size: int) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"indptr", "indices", "values", "shape", "route"}:
                raise AdmissionError(f"{path}: CSR archive members are not canonical")
            result = {
                "indptr": np.asarray(archive["indptr"], dtype=np.int64),
                "indices": np.asarray(archive["indices"], dtype=np.int64),
                "values": np.asarray(archive["values"], dtype=np.float64),
                "shape": np.asarray(archive["shape"], dtype=np.int64),
            }
            archived_route = str(np.asarray(archive["route"]).item())
    except (OSError, KeyError, ValueError) as exc:
        raise AdmissionError(f"cannot load CSR archive {path}: {exc}") from exc
    if archived_route != route:
        raise AdmissionError(f"{path}: archived route identity differs")
    indptr, indices, values, shape = (
        result["indptr"], result["indices"], result["values"], result["shape"]
    )
    if (
        shape.shape != (2,)
        or tuple(int(value) for value in shape) != (size, size)
        or indptr.shape != (size + 1,)
        or indices.ndim != 1
        or values.shape != indices.shape
        or indptr[0] != 0
        or indptr[-1] != indices.size
        or np.any(indptr[1:] < indptr[:-1])
        or np.any(indices < 0)
        or np.any(indices >= size)
        or not np.all(np.isfinite(values))
    ):
        raise AdmissionError(f"{path}: CSR structure is invalid")
    for start, stop in zip(indptr[:-1], indptr[1:]):
        row = indices[int(start) : int(stop)]
        if row.size and np.any(row[1:] <= row[:-1]):
            raise AdmissionError(f"{path}: CSR row indices are not strictly increasing")
    return result


def _load_route(
    evidence_root: Path,
    *,
    block: Mapping[str, object],
    route: str,
    source_commit: str,
    run_id: str,
) -> dict[str, object]:
    block_id = str(block["block_id"])
    route_dir = evidence_root / "blocks" / block_id / route
    payload = _object(route_dir / "output.json")
    position = list(block["route_order"]).index(route)
    exact = {
        "experiment_id": "EXP-ROUTE-001",
        "tier": "fixed_state_screen",
        "status": "completed",
        "route": route,
        "mesh_name": "hetero_ssr_L1",
        "element_degree": block["degree"],
        "quadrature_rule_id": RULE_BY_DEGREE[int(block["degree"])],
        "constraint_variant": "glued_bottom",
        "lambda_target": 1.55,
        "state_family": "analytic_mesh_field_v1",
        "state_label": block["state_label"],
        "state_amplitude": block["state_amplitude"],
        "probe_count": 4,
        "mpi_ranks": block["ranks"],
        "warmup_repetitions": 1,
        "measured_repetitions": 5,
        "action_out": "tangent_action.npz",
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise AdmissionError(f"{block_id}/{route}: child identity {key} differs")
    design = payload.get("comparison_design")
    expected_design = {
        "comparison_id": block_id,
        "block_repetition": 1,
        "route_order_position": position,
        "route_order_policy": "local_distributed_correctness_v2",
        "timing_reduction": "mpi_collective_max",
        "independent_process_block": True,
    }
    if not isinstance(design, dict) or any(design.get(key) != value for key, value in expected_design.items()):
        raise AdmissionError(f"{block_id}/{route}: child comparison identity differs")
    git = payload.get("git")
    if not isinstance(git, dict) or git.get("commit") != source_commit or git.get("dirty") is not False:
        raise AdmissionError(f"{block_id}/{route}: child Git identity differs")
    job = payload.get("job_metadata")
    if not isinstance(job, dict) or job.get("workstation_run_id") != run_id:
        raise AdmissionError(f"{block_id}/{route}: child run identity is missing")

    action_path = route_dir / "tangent_action.npz"
    try:
        with np.load(action_path, allow_pickle=False) as archive:
            expected_members = {
                "state", "tangent_action", "tangent_actions", "gradient", "route", "state_label"
            }
            if set(archive.files) != expected_members:
                raise AdmissionError(f"{action_path}: action archive members are not canonical")
            state = np.asarray(archive["state"], dtype=np.float64)
            first_action = np.asarray(archive["tangent_action"], dtype=np.float64)
            actions = np.asarray(archive["tangent_actions"], dtype=np.float64)
            gradient = np.asarray(archive["gradient"], dtype=np.float64)
            archived_route = str(np.asarray(archive["route"]).item())
            archived_state = str(np.asarray(archive["state_label"]).item())
    except (OSError, KeyError, ValueError) as exc:
        raise AdmissionError(f"cannot load action archive {action_path}: {exc}") from exc
    if (
        state.ndim != 1
        or gradient.shape != state.shape
        or actions.shape != (4, state.size)
        or first_action.shape != state.shape
        or not np.array_equal(first_action, actions[0])
        or archived_route != route
        or archived_state != block["state_label"]
        or not all(np.all(np.isfinite(value)) for value in (state, gradient, actions))
    ):
        raise AdmissionError(f"{block_id}/{route}: saved action arrays are inconsistent")
    if (
        payload.get("state_sha256") != array_sha256(state)
        or payload.get("gradient_sha256") != array_sha256(gradient)
        or payload.get("action_sha256") != array_sha256(actions[0])
        or payload.get("action_sha256_by_probe") != [array_sha256(row) for row in actions]
    ):
        raise AdmissionError(f"{block_id}/{route}: child array hashes differ from the archive")
    covariates = payload.get("model_covariates")
    if not isinstance(covariates, dict) or covariates.get("global_free_dofs") != state.size:
        raise AdmissionError(f"{block_id}/{route}: global degree count differs from the arrays")
    rank_rows = payload.get("rank_summaries")
    if not isinstance(rank_rows, list) or len(rank_rows) != int(block["ranks"]):
        raise AdmissionError(f"{block_id}/{route}: rank-summary inventory is incomplete")
    rank_ids = sorted(int(row.get("rank", -1)) for row in rank_rows if isinstance(row, dict))
    owned = sum(int(row.get("owned_dofs", 0)) for row in rank_rows if isinstance(row, dict))
    if rank_ids != list(range(int(block["ranks"]))) or owned != state.size:
        raise AdmissionError(f"{block_id}/{route}: owned-row partition is inconsistent")

    csr = None
    if int(block["ranks"]) == 1:
        if payload.get("direct_matrix_out") != "tangent_matrix_csr.npz":
            raise AdmissionError(f"{block_id}/{route}: rank-one direct CSR identity is missing")
        csr = _load_csr(
            route_dir / "tangent_matrix_csr.npz", route=route, size=state.size
        )
        if (
            payload.get("direct_matrix_nonzeros") != int(csr["values"].size)
            or payload.get("direct_matrix_value_sha256") != array_sha256(csr["values"])
        ):
            raise AdmissionError(f"{block_id}/{route}: child CSR hashes differ from the archive")
    elif payload.get("direct_matrix_out") != "":
        raise AdmissionError(f"{block_id}/{route}: distributed row unexpectedly reports direct CSR")
    return {
        "payload": payload,
        "state": state,
        "gradient": gradient,
        "actions": actions,
        "csr": csr,
    }


def revalidate_numerical_evidence(
    evidence_root: Path, *, source_commit: str, run_id: str
) -> dict[str, object]:
    loaded: dict[tuple[int, str, int, str], dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    maxima = {
        "gradient_relative": 0.0,
        "action_relative": 0.0,
        "csr_value_relative": 0.0,
        "rank_gradient_relative": 0.0,
        "rank_action_relative": 0.0,
    }
    for block in _expected_blocks():
        degree = int(block["degree"])
        state = str(block["state_label"])
        ranks = int(block["ranks"])
        route_data = {
            route: _load_route(
                evidence_root,
                block=block,
                route=route,
                source_commit=source_commit,
                run_id=run_id,
            )
            for route in ROUTES
        }
        for route, value in route_data.items():
            loaded[(degree, state, ranks, route)] = value
        reference = route_data["element_ad"]
        colored = route_data["colored_sfd"]
        for route in ("colored_sfd", "constitutive_ad"):
            candidate = route_data[route]
            if not np.array_equal(reference["state"], candidate["state"]):
                raise AdmissionError(f"{block['block_id']}/{route}: states are not exactly paired")
            gradient_error = _require_close(
                reference["gradient"], candidate["gradient"],
                label=f"{block['block_id']}/{route} gradient",
            )
            action_error = _require_close(
                reference["actions"], candidate["actions"],
                label=f"{block['block_id']}/{route} tangent actions",
            )
            if candidate["payload"].get("branch_diagnostics") != reference["payload"].get("branch_diagnostics"):
                raise AdmissionError(f"{block['block_id']}/{route}: branch diagnostics differ")
            if route == "colored_sfd":
                colored_gradient = gradient_error
                colored_action = action_error
        csr_error: float | None = None
        if ranks == 1:
            reference_csr = reference["csr"]
            if not isinstance(reference_csr, dict):
                raise AdmissionError(f"{block['block_id']}: reference CSR is missing")
            for route in ("colored_sfd", "constitutive_ad"):
                candidate_csr = route_data[route]["csr"]
                if not isinstance(candidate_csr, dict):
                    raise AdmissionError(f"{block['block_id']}/{route}: CSR is missing")
                for field in ("indptr", "indices", "shape"):
                    if not np.array_equal(reference_csr[field], candidate_csr[field]):
                        raise AdmissionError(f"{block['block_id']}/{route}: CSR {field} differs")
                value_error = _require_close(
                    reference_csr["values"], candidate_csr["values"],
                    label=f"{block['block_id']}/{route} CSR values",
                )
                if route == "colored_sfd":
                    csr_error = value_error
        rows.append(
            {
                "degree": degree,
                "state": state,
                "ranks": ranks,
                "state_exact": True,
                "colored_gradient_relative_error": colored_gradient,
                "colored_action_relative_error": colored_action,
                "colored_csr_relative_error": csr_error,
            }
        )
        maxima["gradient_relative"] = max(maxima["gradient_relative"], colored_gradient)
        maxima["action_relative"] = max(maxima["action_relative"], colored_action)
        if csr_error is not None:
            maxima["csr_value_relative"] = max(maxima["csr_value_relative"], csr_error)

    for degree in DEGREES:
        for state, _amplitude in STATES:
            for route in ROUTES:
                reference = loaded[(degree, state, 1, route)]
                for ranks in (2, 4):
                    candidate = loaded[(degree, state, ranks, route)]
                    if not np.array_equal(reference["state"], candidate["state"]):
                        raise AdmissionError(f"P{degree}/{state}/{route}: state differs across ranks")
                    gradient_error = _require_close(
                        reference["gradient"], candidate["gradient"],
                        label=f"P{degree}/{state}/{route}/np{ranks} gradient",
                    )
                    action_error = _require_close(
                        reference["actions"], candidate["actions"],
                        label=f"P{degree}/{state}/{route}/np{ranks} actions",
                    )
                    if candidate["payload"].get("branch_diagnostics") != reference["payload"].get("branch_diagnostics"):
                        raise AdmissionError(f"P{degree}/{state}/{route}: branch diagnostics differ across ranks")
                    maxima["rank_gradient_relative"] = max(
                        maxima["rank_gradient_relative"], gradient_error
                    )
                    maxima["rank_action_relative"] = max(
                        maxima["rank_action_relative"], action_error
                    )
    return {
        "status": "passed",
        "comparison_tolerances": {
            "relative": RELATIVE_TOLERANCE,
            "absolute": ABSOLUTE_TOLERANCE,
            "integer_topology_and_state": "exact",
        },
        "rows": rows,
        "maxima": maxima,
        "timing_claim_admissible": False,
    }


def audit_campaign(evidence_root: Path, *, repo_root: Path) -> dict[str, object]:
    envelope = validate_campaign_envelope(evidence_root, repo_root=repo_root)
    numerical = revalidate_numerical_evidence(
        evidence_root.resolve(),
        source_commit=str(envelope["source_commit"]),
        run_id=str(envelope["run_id"]),
    )
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "admitted_correctness_only",
        "experiment_id": "EXP-DIST-001",
        "source_commit": envelope["source_commit"],
        "run_id": envelope["run_id"],
        "campaign_manifest_sha256": envelope["campaign_manifest_sha256"],
        "campaign_output_files_sha256": envelope["output_files_sha256"],
        "numerical_revalidation": numerical,
        "timing_claim_admissible": False,
    }


def _tex_number(value: float | None) -> str:
    if value is None:
        return "--"
    if value == 0.0:
        return "$0$"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"


def render_table(audit: Mapping[str, object]) -> str:
    if audit.get("status") != "admitted_correctness_only" or audit.get("timing_claim_admissible") is not False:
        raise AdmissionError("only admitted correctness-only evidence can be rendered")
    numerical = audit.get("numerical_revalidation")
    if not isinstance(numerical, dict) or numerical.get("timing_claim_admissible") is not False:
        raise AdmissionError("numerical revalidation does not prohibit timing claims")
    rows = numerical.get("rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise AdmissionError("numerical revalidation must contain 12 table rows")
    lines = [
        "\\begin{table}[t]",
        "  \\caption{Fixed-state verification of distributed colored sparse recovery. "
        "State vectors agree exactly in all 12 blocks. The common element-AD residual is "
        "retained as a state and assembly consistency check. The four one-rank blocks "
        "additionally compare complete CSR patterns and values; a dash denotes that no "
        "direct distributed CSR comparison was performed. The tabulated route defects "
        "compare colored recovery with element AD at the same state and rank; "
        "constitutive-route and cross-rank discrepancies are summarized in the text. "
        "Each tabulated numerical comparison is "
        "accepted when its relative error is at most $1 \\times 10^{-8}$ or its absolute "
        "error is at most $1 \\times 10^{-10}$. Timings are not included.}",
        "  \\label{tab:distributed-colored-verification}",
        "  \\centering",
        "  \\begin{tabularx}{\\linewidth}{C{0.70}C{1.05}C{0.65}C{1.20}C{1.20}C{1.20}}",
        "    \\toprule",
        "    Element & State & Ranks & Common residual rel. defect & Colored/element action rel. defect & Colored/element CSR-value rel. defect \\\\",
        "    \\midrule",
    ]
    for row in rows:
        if not isinstance(row, dict):
            raise AdmissionError("table row must be an object")
        lines.append(
            "    "
            f"$P_{int(row['degree'])}(L_1)$ & {str(row['state']).capitalize()} & {int(row['ranks'])} & "
            f"{_tex_number(float(row['colored_gradient_relative_error']))} & "
            f"{_tex_number(float(row['colored_action_relative_error']))} & "
            f"{_tex_number(None if row['colored_csr_relative_error'] is None else float(row['colored_csr_relative_error']))} \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabularx}",
            "\\end{table}",
            "",
        ]
    )
    text = "\n".join(lines)
    forbidden = ("wall time", "speedup", "faster", "timing result")
    if any(token in text.lower() for token in forbidden):
        raise AdmissionError("correctness table contains timing-claim language")
    return text

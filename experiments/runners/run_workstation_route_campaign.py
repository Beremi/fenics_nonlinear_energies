#!/usr/bin/env python3
"""Prepare or execute the frozen clean-workstation EXP-ROUTE paired blocks.

The publication path is deliberately fail-before-write.  It verifies the
confirmation, clean source commit, frozen configuration, executable, and
immutable inputs before creating the campaign root.  During execution every
route process receives the smaller of the remaining row and campaign wall
budgets.  A cap is retained as an explicit censor instead of erasing already
completed process evidence.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
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


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.runners.paper_revision_karolina.execute_case import (
    _run,
    p3d_fixed_block_commands,
    validate_fixed_state_block,
)
from src.core.benchmark.run_record import atomic_write_json, strict_json_dumps


DEFAULT_PLAN = REPO_ROOT / "paper/protocols/EXP-ROUTE-001-workstation-plan.json"
MATRIX = REPO_ROOT / "experiments/runners/paper_revision_karolina/campaign_matrix.csv"
FIXED_ROUTE_RUNNER = REPO_ROOT / "experiments/runners/run_plasticity3d_fixed_state_route_screen.py"
EXECUTE_CASE = REPO_ROOT / "experiments/runners/paper_revision_karolina/execute_case.py"
BACKEND_RUNNER = REPO_ROOT / "experiments/runners/run_plasticity3d_backend_mix_case.py"
MESH_ROOT = REPO_ROOT / "data/meshes/SlopeStability3D/hetero_ssr"
MANIFEST_NAME = "workstation_manifest.json"
PROCESS_RECORD_NAME = "process_record.json"
COMMIT_RE = re.compile(r"[0-9a-fA-F]{40,64}")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class WorkstationCampaignError(RuntimeError):
    """Raised when the workstation campaign contract cannot be satisfied."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = strict_json_dumps(value, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_strict_json(path: Path) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _git_metadata() -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise WorkstationCampaignError(f"git {' '.join(args)} failed: {detail}")
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


def _load_plan(path: Path) -> tuple[dict[str, object], list[dict[str, str]]]:
    plan = _read_strict_json(path)
    if not isinstance(plan, dict) or plan.get("schema_id") != (
        "fenics-nonlinear-energies.exp-route-001-workstation-plan"
    ):
        raise ValueError("workstation plan has the wrong schema")
    if int(plan.get("schema_version", -1)) != 1:
        raise ValueError("workstation plan has the wrong schema version")
    if plan.get("source_matrix_sha256") != _sha256(MATRIX):
        raise ValueError("workstation plan source-matrix hash is stale")
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        by_case = {row["case_id"]: dict(row) for row in csv.DictReader(handle)}
    case_ids = [str(value) for value in plan.get("case_ids", [])]
    if len(case_ids) != 12 or len(set(case_ids)) != 12:
        raise ValueError("workstation plan must contain 12 unique paired blocks")
    try:
        rows = [by_case[case_id] for case_id in case_ids]
    except KeyError as exc:
        raise ValueError(f"workstation plan names an unknown matrix row: {exc.args[0]}") from exc
    if any(
        row["experiment_id"] != "EXP-ROUTE-001"
        or row["tier"] != "fixed_state_screen"
        or row["runner"] != "p3d_fixed_state_block"
        or int(row["total_ranks"]) != 1
        or int(row["element_degree"]) not in {1, 2}
        or row["mesh_name"] != "hetero_ssr_L1"
        for row in rows
    ):
        raise ValueError("workstation plan selects an out-of-scope matrix row")
    expected = {
        (degree, state, block)
        for degree in (1, 2)
        for state in ("elastic", "mixed")
        for block in (1, 2, 3)
    }
    actual = {
        (int(row["element_degree"]), row["state_label"], int(row["block_repetition"]))
        for row in rows
    }
    if actual != expected:
        raise ValueError("workstation plan does not cover the frozen P1/P2 state blocks")
    return plan, rows


def _resolve_python(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    # Keep the invoked venv path intact. Resolving a venv's ``python`` symlink
    # to the base interpreter changes Python's prefix discovery and can silently
    # drop the venv site-packages even though both paths name the same binary.
    path = Path(os.path.abspath(path))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise WorkstationCampaignError(f"Python executable is missing or not executable: {path}")
    if path.resolve() != Path(sys.executable).resolve():
        raise WorkstationCampaignError(
            "publication driver and worker must use the same Python executable"
        )
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
        raise WorkstationCampaignError("could not enumerate tracked source files")
    paths = {
        (REPO_ROOT / raw.decode("utf-8"))
        for raw in tracked.stdout.split(b"\0")
        if raw
    }
    paths.update({Path(__file__).resolve(), EXECUTE_CASE, FIXED_ROUTE_RUNNER, BACKEND_RUNNER})
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise WorkstationCampaignError(f"hashed code file is missing: {missing[0]}")
    return {
        _relative_key(path): _sha256(path)
        for path in sorted(paths, key=lambda value: _relative_key(value))
    }


def _collect_configuration_hashes(plan_path: Path) -> dict[str, str]:
    return {
        _relative_key(plan_path): _sha256(plan_path),
        _relative_key(MATRIX): _sha256(MATRIX),
    }


def _collect_input_hashes(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    from src.problems.slope_stability_3d.support.mesh import (
        _same_mesh_hdf5_is_current,
    )

    paths = {
        MESH_ROOT / "SSR_hetero_ada_L1.msh",
        MESH_ROOT / "definition.py",
    }
    for row in rows:
        degree = int(row["element_degree"])
        mesh_name = str(row["mesh_name"])
        hdf5_path = MESH_ROOT / f"{mesh_name}_p{degree}_same_mesh_glued_bottom.h5"
        if not _same_mesh_hdf5_is_current(
            hdf5_path,
            mesh_name=mesh_name,
            degree=degree,
            constraint_variant="glued_bottom",
            quadrature_rule_id=str(row["quadrature_rule"]),
        ):
            raise WorkstationCampaignError(
                "frozen workstation HDF5 is absent or stale and would be regenerated: "
                f"{hdf5_path}"
            )
        paths.add(hdf5_path)
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise WorkstationCampaignError(f"frozen workstation input is missing: {missing[0]}")
    return {
        _relative_key(path): _sha256(path)
        for path in sorted(paths, key=lambda value: _relative_key(value))
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("h5py", "jax", "jaxlib", "mpi4py", "numpy", "petsc4py", "scipy"):
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _capture_environment(python_path: Path) -> dict[str, object]:
    return {
        "captured_at_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "python_executable": str(python_path),
        "python_executable_sha256": _sha256(python_path),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "packages": _package_versions(),
        "cpu_affinity": (
            sorted(int(value) for value in os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else []
        ),
        "thread_environment": {
            name: os.environ.get(name, "")
            for name in (
                "JAX_PLATFORMS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "XLA_FLAGS",
            )
        },
    }


def _normalize_command(
    command: Sequence[str], *, out_root: Path, python_path: Path
) -> list[str]:
    normalized: list[str] = []
    for index, raw in enumerate(command):
        token = str(raw)
        if index == 0 and Path(token).resolve() == python_path.resolve():
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


def _command_plan(
    rows: Sequence[Mapping[str, str]],
    *,
    out_root: Path,
    python_path: Path,
    run_id: str,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in rows:
        measure_dir = out_root / "cases" / row["case_id"] / f"job_{run_id}" / "measure_01"
        for route, command in p3d_fixed_block_commands(
            dict(row),
            python=str(python_path),
            run_dir=measure_dir,
            use_srun=False,
        ):
            normalized = _normalize_command(
                command, out_root=out_root, python_path=python_path
            )
            result.append(
                {
                    "case_id": row["case_id"],
                    "route": route,
                    "normalized_argv": normalized,
                    "normalized_argv_sha256": _json_sha256(normalized),
                }
            )
    return result


def _preflight(
    args: argparse.Namespace,
    *,
    out_root: Path,
    git: Mapping[str, object],
) -> None:
    raw_out_root = Path(args.out_root).expanduser()
    if os.path.lexists(raw_out_root) or os.path.lexists(out_root):
        raise WorkstationCampaignError(
            f"campaign output root must be fresh and nonexisting: {out_root}"
        )
    commit = str(git.get("commit", ""))
    if COMMIT_RE.fullmatch(commit) is None:
        raise WorkstationCampaignError("git did not return a full hexadecimal commit")
    if bool(git.get("dirty", True)):
        detail = "\n".join(str(value) for value in git.get("status_porcelain", []))
        suffix = f"\n{detail}" if detail else ""
        raise WorkstationCampaignError(
            "publication workstation campaign requires a clean worktree" + suffix
        )
    expected_commit = str(args.expected_commit or "").strip()
    if expected_commit and COMMIT_RE.fullmatch(expected_commit) is None:
        raise WorkstationCampaignError("--expected-commit must be a full 40--64 digit hash")
    if expected_commit and expected_commit.lower() != commit.lower():
        raise WorkstationCampaignError("current HEAD differs from --expected-commit")
    if args.execute:
        if os.environ.get("WORKSTATION_RUN_CONFIRMED") != "YES":
            raise WorkstationCampaignError(
                "execution requires WORKSTATION_RUN_CONFIRMED=YES"
            )
        if not expected_commit:
            raise WorkstationCampaignError(
                "publication execution requires --expected-commit"
            )
        if int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) != 1:
            raise WorkstationCampaignError(
                "workstation driver itself must run as one process"
            )
        for name in ("row_wall_s", "campaign_wall_s"):
            value = float(getattr(args, name))
            if not np.isfinite(value) or value <= 0.0:
                raise WorkstationCampaignError(
                    f"--{name.replace('_', '-')} must be finite and positive"
                )


def _tree_hashes(root: Path, *, exclude: set[Path] | None = None) -> dict[str, str]:
    excluded = {path.resolve() for path in (exclude or set())}
    hashes: dict[str, str] = {}
    if not root.is_dir():
        return hashes
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() in excluded or path.name.endswith(".tmp"):
            continue
        hashes[path.relative_to(root).as_posix()] = _sha256(path)
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
        elif _sha256(path) != expected:
            errors.append(f"hash mismatch: {raw}")
    return errors


def _route_artifact_hashes(route_dir: Path) -> dict[str, str]:
    return _tree_hashes(route_dir, exclude={route_dir / PROCESS_RECORD_NAME})


def _verify_route_artifacts(route_dir: Path) -> None:
    output = route_dir / "output.json"
    action = route_dir / "tangent_action.npz"
    if not output.is_file() or not action.is_file():
        raise WorkstationCampaignError(
            f"route process did not create output.json and tangent_action.npz: {route_dir}"
        )
    payload = _read_strict_json(output)
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        raise WorkstationCampaignError(f"route result is not a completed JSON object: {output}")


def _write_process_record(route_dir: Path, record: dict[str, object]) -> dict[str, object]:
    record["artifact_hash_closure"] = {
        "algorithm": "sha256",
        "scope": "all_regular_files_below_route_directory_except_process_record",
        "excluded_paths": [PROCESS_RECORD_NAME],
        "files": _route_artifact_hashes(route_dir),
    }
    record["artifact_hash_closure"]["files_map_sha256"] = _json_sha256(
        record["artifact_hash_closure"]["files"]
    )
    path = route_dir / PROCESS_RECORD_NAME
    atomic_write_json(path, record)
    returned = dict(record)
    returned["process_record"] = path.name
    returned["process_record_sha256"] = _sha256(path)
    return returned


def _terminal_counts(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    statuses = ("completed", "censored", "failed")
    return {
        status: sum(record.get("status") == status for record in records)
        for status in statuses
    }


def _checkpoint_manifest(
    manifest_path: Path,
    manifest: dict[str, object],
    records: Sequence[Mapping[str, object]],
    case_statuses: Mapping[str, str],
) -> None:
    manifest["route_terminal_counts"] = _terminal_counts(records)
    manifest["route_processes_launched"] = sum(
        bool(record.get("launched")) for record in records
    )
    manifest["case_statuses"] = dict(case_statuses)
    atomic_write_json(manifest_path, manifest)


def _censor_record(
    *,
    case_id: str,
    route: str,
    command: Sequence[str],
    normalized: Sequence[str],
    reason: str,
    timeout_s: float | None,
    source_commit: str,
) -> dict[str, object]:
    now = _utc_now()
    return {
        "schema_id": "fenics-nonlinear-energies.workstation-route-process-record",
        "schema_version": 1,
        "experiment_id": "EXP-ROUTE-001",
        "run_kind": "publication",
        "case_id": case_id,
        "route": route,
        "status": "censored",
        "censor_reason": reason,
        "launched": False,
        "timed_out": False,
        "returncode": None,
        "wall_time_s": 0.0,
        "timeout_s": timeout_s,
        "started_at_utc": now,
        "finished_at_utc": now,
        "command_argv": list(command),
        "normalized_command_argv": list(normalized),
        "normalized_command_sha256": _json_sha256(list(normalized)),
        "source_commit": source_commit,
    }


def prepare_or_execute(args: argparse.Namespace) -> dict[str, object]:
    plan_path = Path(args.plan).expanduser().resolve()
    plan, rows = _load_plan(plan_path)
    raw_out_root = Path(args.out_root).expanduser()
    if not raw_out_root.is_absolute():
        raw_out_root = Path.cwd() / raw_out_root
    out_root = raw_out_root.resolve(strict=False)
    git = _git_metadata()
    _preflight(args, out_root=out_root, git=git)
    python_path = _resolve_python(args.python)
    run_id = str(args.run_id or uuid.uuid4())
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise WorkstationCampaignError(
            "--run-id must use only letters, digits, dot, underscore, and hyphen"
        )

    # Every potentially expensive or failure-prone read happens before mkdir.
    code_hashes = _collect_code_hashes()
    configuration_hashes = _collect_configuration_hashes(plan_path)
    input_hashes = _collect_input_hashes(rows)
    environment = _capture_environment(python_path)
    command_plan = _command_plan(
        rows,
        out_root=out_root,
        python_path=python_path,
        run_id=run_id,
    )
    if len(command_plan) != sum(len(row["route_order"].split("|")) for row in rows):
        raise WorkstationCampaignError("normalized command plan has the wrong size")

    out_root.mkdir(parents=True, exist_ok=False)
    archived_plan = out_root / "workstation_plan.json"
    shutil.copy2(plan_path, archived_plan)
    environment_path = out_root / "environment.json"
    atomic_write_json(environment_path, environment)
    manifest: dict[str, object] = {
        "schema_id": "fenics-nonlinear-energies.exp-route-001-workstation-manifest",
        "schema_version": 1,
        "status": "running" if args.execute else "prepared_not_executed",
        "experiment_id": "EXP-ROUTE-001",
        "run_kind": "publication",
        "hardware_id": "workstation_local",
        "created_at_utc": _utc_now(),
        "plan_path": archived_plan.name,
        "plan_sha256": _sha256(archived_plan),
        "matrix_path": str(MATRIX.relative_to(REPO_ROOT)),
        "matrix_sha256": _sha256(MATRIX),
        "source_commit": git["commit"],
        "source_tree": git.get("tree", ""),
        "source_branch": git.get("branch", ""),
        "source_dirty": False,
        "source_status_porcelain": [],
        "expected_commit": str(args.expected_commit or ""),
        "run_id": run_id,
        "case_count": len(rows),
        "route_process_executions": len(command_plan),
        "case_ids": [row["case_id"] for row in rows],
        "row_wall_cap_s": float(args.row_wall_s),
        "campaign_wall_cap_s": float(args.campaign_wall_s),
        "environment_path": environment_path.name,
        "environment_sha256": _sha256(environment_path),
        "code_hashes": code_hashes,
        "code_hashes_sha256": _json_sha256(code_hashes),
        "configuration_hashes": configuration_hashes,
        "configuration_hashes_sha256": _json_sha256(configuration_hashes),
        "input_hashes": input_hashes,
        "input_hashes_sha256": _json_sha256(input_hashes),
        "normalized_commands": command_plan,
        "normalized_commands_sha256": _json_sha256(command_plan),
        "output_hash_closure": {"status": "open_during_execution"},
        "route_terminal_counts": {"completed": 0, "censored": 0, "failed": 0},
        "route_processes_launched": 0,
        "case_statuses": {},
    }
    manifest_path = out_root / MANIFEST_NAME
    if not args.execute:
        manifest["output_hash_closure"] = _output_hash_closure(out_root)
        atomic_write_json(manifest_path, manifest)
        return manifest

    records: list[dict[str, object]] = []
    case_statuses: dict[str, str] = {}
    atomic_write_json(manifest_path, manifest)
    campaign_started = time.perf_counter()
    previous_run_id = os.environ.get("WORKSTATION_RUN_ID")
    os.environ["WORKSTATION_RUN_ID"] = run_id
    try:
        for row in rows:
            case_id = row["case_id"]
            job_dir = out_root / "cases" / case_id / f"job_{run_id}"
            measure_dir = job_dir / "measure_01"
            measure_dir.mkdir(parents=True, exist_ok=False)
            atomic_write_json(job_dir / "matrix_row.json", dict(row))
            route_dirs: dict[str, Path] = {}
            job_records: list[dict[str, object]] = []
            row_started = time.perf_counter()
            incomplete_reason: str | None = None
            commands = p3d_fixed_block_commands(
                dict(row),
                python=str(python_path),
                run_dir=measure_dir,
                use_srun=False,
            )
            for route, command in commands:
                route_dir = measure_dir / route
                route_dir.mkdir(parents=True, exist_ok=False)
                route_dirs[route] = route_dir
                normalized = _normalize_command(
                    command, out_root=out_root, python_path=python_path
                )
                (route_dir / "command.txt").write_text(
                    shlex.join(command) + "\n", encoding="utf-8"
                )
                now = time.perf_counter()
                campaign_remaining = float(args.campaign_wall_s) - (
                    now - campaign_started
                )
                row_remaining = float(args.row_wall_s) - (now - row_started)
                process_cap = min(campaign_remaining, row_remaining)
                if incomplete_reason is not None:
                    raw_record = _censor_record(
                        case_id=case_id,
                        route=route,
                        command=command,
                        normalized=normalized,
                        reason=incomplete_reason,
                        timeout_s=max(0.0, process_cap),
                        source_commit=str(git["commit"]),
                    )
                elif campaign_remaining <= 0.0:
                    incomplete_reason = "campaign_wall_cap_exhausted_before_launch"
                    raw_record = _censor_record(
                        case_id=case_id,
                        route=route,
                        command=command,
                        normalized=normalized,
                        reason=incomplete_reason,
                        timeout_s=0.0,
                        source_commit=str(git["commit"]),
                    )
                elif row_remaining <= 0.0:
                    incomplete_reason = "row_wall_cap_exhausted_before_launch"
                    raw_record = _censor_record(
                        case_id=case_id,
                        route=route,
                        command=command,
                        normalized=normalized,
                        reason=incomplete_reason,
                        timeout_s=0.0,
                        source_commit=str(git["commit"]),
                    )
                else:
                    started_utc = _utc_now()
                    run_result = _run(
                        command,
                        stdout=route_dir / "stdout.txt",
                        stderr=route_dir / "stderr.txt",
                        timeout_s=float(process_cap),
                    )
                    timed_out = bool(run_result["timed_out"])
                    returncode = int(run_result["returncode"])
                    verification_error: str | None = None
                    if returncode == 0 and not timed_out:
                        try:
                            _verify_route_artifacts(route_dir)
                        except (OSError, ValueError, WorkstationCampaignError) as exc:
                            verification_error = f"{type(exc).__name__}: {exc}"
                    if timed_out:
                        status = "censored"
                        reason = (
                            "campaign_wall_cap_timeout"
                            if campaign_remaining <= row_remaining
                            else "row_wall_cap_timeout"
                        )
                        incomplete_reason = "not_launched_after_route_timeout"
                    elif returncode != 0:
                        status = "failed"
                        reason = "route_process_nonzero_exit"
                        incomplete_reason = "not_launched_after_route_failure"
                    elif verification_error is not None:
                        status = "failed"
                        reason = "route_artifact_verification_failed"
                        incomplete_reason = "not_launched_after_route_failure"
                    else:
                        status = "completed"
                        reason = None
                    raw_record = {
                        "schema_id": "fenics-nonlinear-energies.workstation-route-process-record",
                        "schema_version": 1,
                        "experiment_id": "EXP-ROUTE-001",
                        "run_kind": "publication",
                        "case_id": case_id,
                        "route": route,
                        "status": status,
                        "censor_reason": reason,
                        "launched": True,
                        "timed_out": timed_out,
                        "returncode": returncode,
                        "wall_time_s": float(run_result["wall_time_s"]),
                        "timeout_s": float(process_cap),
                        "started_at_utc": started_utc,
                        "finished_at_utc": _utc_now(),
                        "verification_error": verification_error,
                        "command_argv": list(command),
                        "normalized_command_argv": normalized,
                        "normalized_command_sha256": _json_sha256(normalized),
                        "source_commit": str(git["commit"]),
                    }
                record = _write_process_record(route_dir, raw_record)
                records.append(record)
                job_records.append(record)
                atomic_write_json(job_dir / "run_records.json", job_records)
                _checkpoint_manifest(manifest_path, manifest, records, case_statuses)

            route_statuses = {str(record["status"]) for record in job_records}
            if route_statuses == {"completed"} and len(job_records) == len(commands):
                try:
                    block = validate_fixed_state_block(dict(row), route_dirs)
                except Exception as exc:  # retain all process evidence and continue
                    case_statuses[case_id] = "failed_validation"
                    atomic_write_json(
                        measure_dir / "block_terminal.json",
                        {
                            "status": "failed_validation",
                            "error": f"{type(exc).__name__}: {exc}",
                            "route_statuses": [record["status"] for record in job_records],
                        },
                    )
                else:
                    block["job_metadata"] = {"workstation_run_id": run_id}
                    atomic_write_json(measure_dir / "block_result.json", block)
                    case_statuses[case_id] = "completed"
            else:
                case_statuses[case_id] = (
                    "censored" if "censored" in route_statuses else "failed"
                )
                atomic_write_json(
                    measure_dir / "block_terminal.json",
                    {
                        "status": case_statuses[case_id],
                        "route_statuses": [record["status"] for record in job_records],
                        "timing_admissible": False,
                    },
                )
            _checkpoint_manifest(manifest_path, manifest, records, case_statuses)

        terminal_git = _git_metadata()
        manifest["finished_at_utc"] = _utc_now()
        manifest["terminal_source"] = terminal_git
        frozen_hash_errors = {
            name: errors
            for name, errors in (
                ("code", _verify_hash_inventory(code_hashes)),
                ("configuration", _verify_hash_inventory(configuration_hashes)),
                ("input", _verify_hash_inventory(input_hashes)),
            )
            if errors
        }
        manifest["terminal_frozen_hash_verification"] = {
            "passed": not frozen_hash_errors,
            "errors": frozen_hash_errors,
        }
        counts = _terminal_counts(records)
        validation_failures = sum(
            status == "failed_validation" for status in case_statuses.values()
        )
        if (
            terminal_git.get("commit") != git.get("commit")
            or bool(terminal_git.get("dirty", True))
        ):
            manifest["status"] = "failed_source_changed_during_execution"
        elif frozen_hash_errors:
            manifest["status"] = "failed_frozen_hash_changed_during_execution"
        elif counts["failed"] or validation_failures:
            manifest["status"] = "completed_with_failures"
        elif counts["censored"]:
            manifest["status"] = "completed_with_censors"
        elif counts["completed"] == len(command_plan) and set(case_statuses.values()) == {
            "completed"
        }:
            manifest["status"] = "completed"
        else:
            manifest["status"] = "failed_incomplete_terminal_inventory"
        manifest["output_hash_closure"] = _output_hash_closure(out_root)
        _checkpoint_manifest(manifest_path, manifest, records, case_statuses)
        if manifest["status"] in {
            "failed_source_changed_during_execution",
            "failed_frozen_hash_changed_during_execution",
        }:
            raise WorkstationCampaignError(
                "source commit, worktree, or frozen hash changed during execution"
            )
        return manifest
    except BaseException as exc:
        if manifest.get("status") == "running":
            manifest["status"] = "failed_runner_exception"
        manifest["runner_exception"] = f"{type(exc).__name__}: {exc}"
        manifest["finished_at_utc"] = _utc_now()
        manifest["output_hash_closure"] = _output_hash_closure(out_root)
        _checkpoint_manifest(manifest_path, manifest, records, case_statuses)
        raise
    finally:
        if previous_run_id is None:
            os.environ.pop("WORKSTATION_RUN_ID", None)
        else:
            os.environ["WORKSTATION_RUN_ID"] = previous_run_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--python", default="./.venv/bin/python")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--expected-commit",
        default="",
        help="full clean HEAD hash; required with --execute",
    )
    parser.add_argument(
        "--row-wall-s",
        type=float,
        default=3600.0,
        help="wall cap shared by the three independent route processes in one block",
    )
    parser.add_argument(
        "--campaign-wall-s",
        type=float,
        default=43200.0,
        help="wall cap shared by the complete 12-block campaign",
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        result = prepare_or_execute(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(strict_json_dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

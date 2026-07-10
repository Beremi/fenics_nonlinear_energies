#!/usr/bin/env python3
"""Shared fail-closed contracts for small reviewed Karolina campaigns.

This module is intentionally scheduler-free.  It writes and validates a
relocatable plan, source freeze, and shell-quoted ``sbatch`` command inventory;
it never executes those commands.  The experiment-specific preparers own the
scientific matrix and final adjudication.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = "fta-26-40"
QOS = "3571_6328"
PLAN_SCHEMA_ID = "fenics-nonlinear-energies.reviewed-karolina-plan"
MANIFEST_SCHEMA_ID = "fenics-nonlinear-energies.reviewed-karolina-manifest"
SOURCE_FREEZE_SCHEMA_ID = "fenics-nonlinear-energies.reviewed-karolina-source-freeze"
SCHEMA_VERSION = 1
RUN_SCRIPT = Path("experiments/runners/run_reviewed_karolina_case.sbatch")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class CampaignContractError(ValueError):
    """A prepared campaign is incomplete, stale, or unsafe."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignContractError(f"{path} must contain one JSON object")
    return value


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise CampaignContractError(completed.stderr.strip() or "git failed")
        return completed.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain=v1", "--untracked-files=all")),
    }


def _repo_relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise CampaignContractError(f"reviewed file is outside the repository: {path}") from exc


def _archive_member(root: Path, raw: object, *, label: str) -> Path:
    relative = Path(str(raw))
    if relative.is_absolute():
        raise CampaignContractError(f"{label} must be archive-relative")
    member = (root / relative).resolve()
    try:
        member.relative_to(root)
    except ValueError as exc:
        raise CampaignContractError(f"{label} escapes the campaign archive") from exc
    return member


def _seconds(walltime: str) -> int:
    match = re.fullmatch(r"([0-9]{2}):([0-5][0-9]):([0-5][0-9])", walltime)
    if match is None:
        raise CampaignContractError(f"invalid walltime {walltime!r}; expected HH:MM:SS")
    hours, minutes, seconds = (int(value) for value in match.groups())
    if hours < 1 and minutes == 0 and seconds == 0:
        raise CampaignContractError("walltime must be positive")
    return hours * 3600 + minutes * 60 + seconds


def validate_case(case: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "case_id",
        "family",
        "nodes",
        "total_ranks",
        "ranks_per_node",
        "partition",
        "walltime",
        "payload_argv",
        "expected_outputs",
        "scientific_contract",
    }
    if set(case) != required:
        raise CampaignContractError(
            f"case keys differ from the reviewed schema: {sorted(set(case) ^ required)}"
        )
    case_id = str(case["case_id"])
    if _SAFE_ID.fullmatch(case_id) is None:
        raise CampaignContractError(f"unsafe case ID {case_id!r}")
    nodes = int(case["nodes"])
    total_ranks = int(case["total_ranks"])
    ranks_per_node = int(case["ranks_per_node"])
    if nodes < 1 or not 1 <= ranks_per_node <= 128 or total_ranks != nodes * ranks_per_node:
        raise CampaignContractError(f"{case_id} has an invalid Karolina CPU resource shape")
    expected_partition = "qcpu_exp" if nodes <= 2 else "qcpu"
    if case["partition"] != expected_partition:
        raise CampaignContractError(
            f"{case_id} must use {expected_partition} for {nodes} node(s)"
        )
    _seconds(str(case["walltime"]))
    argv = case["payload_argv"]
    if not isinstance(argv, list) or not argv or not all(
        isinstance(value, str) and value for value in argv
    ):
        raise CampaignContractError(f"{case_id} payload_argv must be a nonempty string list")
    if argv[0] != "{PYTHON}":
        raise CampaignContractError(f"{case_id} payload must start with {{PYTHON}}")
    allowed_placeholders = {
        "{PYTHON}", "{REPO_ROOT}", "{CAMPAIGN_ROOT}", "{JOB_ROOT}"
    }
    for token in argv:
        for placeholder in re.findall(r"\{[A-Z_]+\}", token):
            if placeholder not in allowed_placeholders:
                raise CampaignContractError(f"{case_id} uses unknown placeholder {placeholder}")
    outputs = case["expected_outputs"]
    if not isinstance(outputs, list) or not outputs or len(set(outputs)) != len(outputs):
        raise CampaignContractError(f"{case_id} expected_outputs are empty or duplicated")
    for raw in outputs:
        relative = Path(str(raw))
        if relative.is_absolute() or ".." in relative.parts or str(relative) in {"", "."}:
            raise CampaignContractError(f"{case_id} has an unsafe expected output path")
        if "{JOB_ROOT}" not in " ".join(argv):
            raise CampaignContractError(f"{case_id} payload does not bind outputs to JOB_ROOT")
    contract = case["scientific_contract"]
    if not isinstance(contract, dict) or not contract:
        raise CampaignContractError(f"{case_id} lacks a scientific contract")
    return {
        **case,
        "nodes": nodes,
        "total_ranks": total_ranks,
        "ranks_per_node": ranks_per_node,
        "walltime_seconds": _seconds(str(case["walltime"])),
    }


def _source_freeze(
    *, source_commit: str, reviewed_sources: Iterable[Path]
) -> dict[str, Any]:
    records: dict[str, dict[str, str]] = {}
    for path in sorted({Path(value).resolve() for value in reviewed_sources}):
        if not path.is_file():
            raise CampaignContractError(f"reviewed source is missing: {path}")
        relative = _repo_relative(path)
        records[relative] = {"path": relative, "sha256": sha256_file(path)}
    if not records:
        raise CampaignContractError("reviewed source freeze is empty")
    return {
        "schema_id": SOURCE_FREEZE_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "source_commit": source_commit,
        "reviewed_sources": records,
    }


def _environment_contract(
    root: Path, *, env_setup: Path | None, env_lock: Path | None
) -> dict[str, Any]:
    if (env_setup is None) != (env_lock is None):
        raise CampaignContractError("provide both --env-setup and --env-lock, or neither")
    if env_setup is None:
        return {
            "status": "unbound_preparation_only",
            "submission_admissible": False,
            "reason": "reviewed compute environment setup and lock were not supplied",
        }
    records: dict[str, dict[str, str]] = {}
    for label, source in (("setup", Path(env_setup)), ("lock", Path(env_lock))):
        source = source.resolve()
        if not source.is_file():
            raise CampaignContractError(f"environment {label} file is missing: {source}")
        destination = root / f"environment_{label}{source.suffix}"
        destination.write_bytes(source.read_bytes())
        records[label] = {"path": destination.name, "sha256": sha256_file(destination)}
    return {"status": "hash_bound", "submission_admissible": True, **records}


def _sbatch_command(
    *, root: Path, case: Mapping[str, Any], plan_sha256: str, freeze_sha256: str,
    environment: Mapping[str, Any],
) -> list[str]:
    env_setup = (
        str(root / str(environment["setup"]["path"]))
        if environment.get("status") == "hash_bound"
        else "ENV_SETUP_REQUIRED_FOR_SUBMISSION"
    )
    env_lock = (
        str(root / str(environment["lock"]["path"]))
        if environment.get("status") == "hash_bound"
        else "ENV_LOCK_REQUIRED_FOR_SUBMISSION"
    )
    return [
        "sbatch",
        f"--account={ACCOUNT}",
        f"--qos={QOS}",
        f"--job-name={case['case_id']}",
        f"--partition={case['partition']}",
        f"--nodes={case['nodes']}",
        f"--ntasks={case['total_ranks']}",
        f"--ntasks-per-node={case['ranks_per_node']}",
        "--cpus-per-task=1",
        f"--time={case['walltime']}",
        "--distribution=block:block",
        str(REPO_ROOT / RUN_SCRIPT),
        str(root),
        str(case["case_id"]),
        plan_sha256,
        freeze_sha256,
        env_setup,
        env_lock,
    ]


def prepare_campaign(
    *,
    output_root: Path,
    experiment_id: str,
    campaign_id: str,
    cases: Sequence[Mapping[str, Any]],
    protocol: Path,
    reviewed_sources: Iterable[Path],
    env_setup: Path | None = None,
    env_lock: Path | None = None,
    git: Mapping[str, Any] | None = None,
    external_bindings: Mapping[str, Any] | None = None,
    bound_inputs: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    if root.exists() or root.is_symlink():
        raise CampaignContractError(f"fresh campaign root already exists: {root}")
    metadata = dict(git_metadata() if git is None else git)
    if _HEX40.fullmatch(str(metadata.get("commit", ""))) is None or bool(metadata.get("dirty")):
        raise CampaignContractError("publication preparation requires one clean SHA-1 commit")
    if not cases:
        raise CampaignContractError("campaign has no cases")
    normalized = [validate_case(case) for case in cases]
    case_ids = [str(case["case_id"]) for case in normalized]
    if len(case_ids) != len(set(case_ids)):
        raise CampaignContractError("campaign case IDs are duplicated")
    root.mkdir(parents=True, exist_ok=False)
    environment = _environment_contract(root, env_setup=env_setup, env_lock=env_lock)
    archived_bindings: dict[str, dict[str, str]] = {}
    for label, source in sorted((bound_inputs or {}).items()):
        if _SAFE_ID.fullmatch(str(label)) is None:
            raise CampaignContractError(f"unsafe bound-input label {label!r}")
        source = Path(source).resolve()
        if not source.is_file() or source.is_symlink():
            raise CampaignContractError(f"bound input is missing or symlinked: {source}")
        destination = root / "bound_inputs" / f"{label}{source.suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        archived_bindings[str(label)] = {
            "path": destination.relative_to(root).as_posix(),
            "sha256": sha256_file(destination),
        }
    source_paths = {
        Path(protocol).resolve(),
        (REPO_ROOT / RUN_SCRIPT).resolve(),
        Path(__file__).resolve(),
        *(Path(path).resolve() for path in reviewed_sources),
    }
    freeze_path = root / "reviewed_source_freeze.json"
    atomic_json(
        freeze_path,
        _source_freeze(
            source_commit=str(metadata["commit"]), reviewed_sources=source_paths
        ),
    )
    plan_path = root / "prepared_plan.json"
    plan = {
        "schema_id": PLAN_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "campaign_id": campaign_id,
        "source_commit": str(metadata["commit"]),
        "account": ACCOUNT,
        "qos": QOS,
        "protocol": {"path": _repo_relative(protocol), "sha256": sha256_file(protocol)},
        "external_bindings": {
            "metadata": dict(external_bindings or {}),
            "archived_inputs": archived_bindings,
        },
        "cases": normalized,
    }
    atomic_json(plan_path, plan)
    plan_sha256 = sha256_file(plan_path)
    freeze_sha256 = sha256_file(freeze_path)
    commands = [
        _sbatch_command(
            root=root,
            case=case,
            plan_sha256=plan_sha256,
            freeze_sha256=freeze_sha256,
            environment=environment,
        )
        for case in normalized
    ]
    commands_path = root / "sbatch_commands.txt"
    commands_path.write_text(
        "".join(shlex.join(command) + "\n" for command in commands), encoding="utf-8"
    )
    manifest_path = root / "prepared_manifest.json"
    manifest = {
        "schema_id": MANIFEST_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_not_submitted",
        "scheduler_contact": False,
        "experiment_id": experiment_id,
        "campaign_id": campaign_id,
        "prepared_at_utc": utc_now(),
        "source_commit": str(metadata["commit"]),
        "source_dirty": False,
        "case_count": len(normalized),
        "estimated_node_hours_ceiling": sum(
            case["nodes"] * case["walltime_seconds"] / 3600.0 for case in normalized
        ),
        "plan": {"path": plan_path.name, "sha256": plan_sha256},
        "commands": {"path": commands_path.name, "sha256": sha256_file(commands_path)},
        "source_freeze": {"path": freeze_path.name, "sha256": freeze_sha256},
        "environment_contract": environment,
        "submission_admissible": environment.get("status") == "hash_bound",
        "claim_admissible": False,
        "claim_boundary": (
            "This is a scheduler-free preparation record. It contains no computed result "
            "and supports no scientific or performance claim."
        ),
    }
    atomic_json(manifest_path, manifest)
    receipt = offline_preflight(root)
    manifest["offline_preflight"] = receipt
    atomic_json(manifest_path, manifest)
    return manifest


def offline_preflight(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest = read_object(root / "prepared_manifest.json")
    if (
        manifest.get("schema_id") != MANIFEST_SCHEMA_ID
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") not in {
            "prepared_not_submitted",
            "submitting",
            "submitted",
            "partial_submission",
        }
        or manifest.get("source_dirty") is not False
    ):
        raise CampaignContractError("prepared manifest identity or status is invalid")
    status = str(manifest["status"])
    scheduler_contact = manifest.get("scheduler_contact")
    if (
        status == "prepared_not_submitted" and scheduler_contact is not False
    ) or (
        status in {"submitting", "submitted", "partial_submission"}
        and scheduler_contact is not True
    ):
        raise CampaignContractError("prepared manifest scheduler-contact state is inconsistent")
    plan_record = manifest.get("plan")
    command_record = manifest.get("commands")
    freeze_record = manifest.get("source_freeze")
    for label, record in (
        ("plan", plan_record), ("commands", command_record), ("source freeze", freeze_record)
    ):
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise CampaignContractError(f"manifest {label} record is malformed")
        path = _archive_member(root, record["path"], label=label)
        if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
            raise CampaignContractError(f"manifest {label} is missing or stale")
    plan_path = _archive_member(root, plan_record["path"], label="plan")
    plan = read_object(plan_path)
    if (
        plan.get("schema_id") != PLAN_SCHEMA_ID
        or plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("source_commit") != manifest.get("source_commit")
        or plan.get("experiment_id") != manifest.get("experiment_id")
        or plan.get("account") != ACCOUNT
        or plan.get("qos") != QOS
    ):
        raise CampaignContractError("prepared plan identity is stale")
    cases = plan.get("cases")
    if not isinstance(cases, list) or len(cases) != int(manifest.get("case_count", -1)):
        raise CampaignContractError("prepared plan case count is stale")
    normalized = [validate_case({key: value for key, value in case.items() if key != "walltime_seconds"}) for case in cases]
    if [case["case_id"] for case in normalized] != [case["case_id"] for case in cases]:
        raise CampaignContractError("prepared plan case ordering changed")
    bindings = plan.get("external_bindings")
    if not isinstance(bindings, dict) or set(bindings) != {"metadata", "archived_inputs"}:
        raise CampaignContractError("prepared plan external bindings are malformed")
    archived_inputs = bindings["archived_inputs"]
    if not isinstance(archived_inputs, dict):
        raise CampaignContractError("prepared plan archived inputs are malformed")
    for label, record in archived_inputs.items():
        if _SAFE_ID.fullmatch(str(label)) is None or not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise CampaignContractError("prepared plan bound-input record is malformed")
        artifact = _archive_member(root, record["path"], label=f"bound input {label}")
        if not artifact.is_file() or artifact.is_symlink() or sha256_file(artifact) != record["sha256"]:
            raise CampaignContractError(f"bound input {label} is missing or stale")
    freeze_path = _archive_member(root, freeze_record["path"], label="source freeze")
    freeze = read_object(freeze_path)
    if (
        freeze.get("schema_id") != SOURCE_FREEZE_SCHEMA_ID
        or freeze.get("schema_version") != SCHEMA_VERSION
        or freeze.get("source_commit") != manifest.get("source_commit")
    ):
        raise CampaignContractError("source freeze identity is stale")
    reviewed = freeze.get("reviewed_sources")
    if not isinstance(reviewed, dict) or not reviewed:
        raise CampaignContractError("source freeze is empty")
    for key, record in reviewed.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"} or key != record["path"]:
            raise CampaignContractError("source freeze record is malformed")
        source = (REPO_ROOT / key).resolve()
        try:
            source.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise CampaignContractError("source freeze path escapes the repository") from exc
        if not source.is_file() or sha256_file(source) != record["sha256"]:
            raise CampaignContractError(f"reviewed source is missing or stale: {key}")
    command_path = _archive_member(root, command_record["path"], label="commands")
    command_lines = [line for line in command_path.read_text(encoding="utf-8").splitlines() if line]
    if len(command_lines) != len(cases):
        raise CampaignContractError("sbatch command inventory does not cover the plan")
    for case, line in zip(cases, command_lines, strict=True):
        tokens = shlex.split(line)
        if not tokens or tokens[0] != "sbatch" or str(case["case_id"]) not in tokens:
            raise CampaignContractError("sbatch command inventory is malformed or reordered")
        forbidden = {"--exclusive", "--mem", "--mem-per-cpu"}
        if forbidden.intersection(tokens):
            raise CampaignContractError("sbatch command inventory contains forbidden options")
    environment = manifest.get("environment_contract")
    if not isinstance(environment, dict) or environment.get("status") not in {
        "unbound_preparation_only", "hash_bound"
    }:
        raise CampaignContractError("environment contract is absent")
    if environment["status"] == "hash_bound":
        for label in ("setup", "lock"):
            record = environment.get(label)
            if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                raise CampaignContractError(f"environment {label} record is malformed")
            artifact = _archive_member(root, record["path"], label=f"environment {label}")
            if not artifact.is_file() or artifact.is_symlink() or sha256_file(artifact) != record["sha256"]:
                raise CampaignContractError(f"environment {label} is missing or stale")
    return {
        "status": "passed_without_scheduler_contact",
        "experiment_id": plan["experiment_id"],
        "case_count": len(cases),
        "source_commit": plan["source_commit"],
        "plan_sha256": sha256_file(plan_path),
        "commands_sha256": sha256_file(command_path),
        "submission_admissible": environment["status"] == "hash_bound",
    }


def load_plan(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    offline_preflight(root)
    manifest = read_object(Path(root) / "prepared_manifest.json")
    plan = read_object(Path(root) / str(manifest["plan"]["path"]))
    return manifest, plan

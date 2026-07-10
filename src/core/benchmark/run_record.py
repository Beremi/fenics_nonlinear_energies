"""Versioned publication run records and durable JSON checkpoint writes.

The contract in this module is intentionally independent of a particular
solver.  Runners can therefore record a successful solve, a numerical failure,
an intentional resource/iteration cap, or a launcher timeout in one shape.
No third-party schema package is required at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Literal, Mapping, Sequence


RUN_RECORD_SCHEMA_ID = "fenics-nonlinear-energies.publication-run-record"
RUN_RECORD_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_ID = "fenics-nonlinear-energies.experiment-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1

TERMINATION_STATUSES = frozenset({"success", "failure", "capped", "timeout"})
RUN_KINDS = frozenset({"publication", "pilot"})

TOP_LEVEL_FIELDS = (
    "schema",
    "record_id",
    "run_kind",
    "identifiers",
    "problem",
    "solver",
    "termination",
    "accuracy",
    "counts",
    "timing",
    "resources",
    "diagnostics",
    "environment",
    "provenance",
    "artifacts",
)

SECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "schema": ("id", "version"),
    "identifiers": ("campaign", "experiment", "case", "method", "route", "repetition"),
    "problem": (
        "name",
        "mesh",
        "degree",
        "quadrature",
        "total_degrees_of_freedom",
        "free_degrees_of_freedom",
        "notes",
    ),
    "solver": ("algorithm", "implementation", "parameters", "preconditioner", "stopping_contract"),
    "termination": (
        "status",
        "reason",
        "exit_code",
        "started_at_utc",
        "finished_at_utc",
        "limit_kind",
        "limit_value",
        "censored",
    ),
    "accuracy": (
        "contract_id",
        "gate_passed",
        "absolute_residual",
        "relative_residual",
        "scaled_residual",
        "relative_correction",
        "energy_change",
        "custom_metrics",
        "notes",
    ),
    "counts": (
        "nonlinear_iterations",
        "krylov_iterations",
        "function_evaluations",
        "gradient_evaluations",
        "hessian_evaluations",
        "hvp_evaluations",
        "preconditioner_setups",
        "notes",
    ),
    "timing": (
        "aggregation",
        "cold_process",
        "barrier_policy",
        "synchronization_policy",
        "phases_overlap",
        "relation_to_total",
        "process_startup_s",
        "jit_compilation_s",
        "coloring_s",
        "derivative_evaluation_s",
        "constitutive_contraction_s",
        "assembly_s",
        "communication_s",
        "preconditioner_setup_s",
        "krylov_solve_s",
        "globalization_s",
        "state_output_s",
        "total_s",
        "notes",
    ),
    "resources": (
        "nodes",
        "ranks",
        "threads_per_rank",
        "peak_memory_per_rank_bytes",
        "peak_memory_per_node_bytes",
        "tracked_allocations_bytes",
        "measurement_method",
        "notes",
    ),
    "diagnostics": ("state", "branch", "feasibility", "kkt"),
    "environment": (
        "python",
        "packages",
        "platform",
        "jax",
        "xla",
        "jax_enable_x64",
        "petsc",
        "mpi",
        "compiler",
        "blas",
        "cpu_model",
        "node_model",
        "memory_model",
        "scheduler",
        "scheduler_job_id",
        "affinity",
    ),
    "provenance": (
        "git_commit",
        "git_clean",
        "git_status_porcelain",
        "pilot_override",
        "pilot_override_reason",
        "command_argv",
        "working_directory",
        "code_hashes",
        "configuration_hashes",
        "input_hashes",
        "dirty_patch_sha256",
        "seed",
        "deterministic_policy",
        "recorded_at_utc",
    ),
    "artifacts": ("raw_outputs", "states", "logs", "tables", "figures", "reports"),
}

TIMING_VALUE_FIELDS = (
    "process_startup_s",
    "jit_compilation_s",
    "coloring_s",
    "derivative_evaluation_s",
    "constitutive_contraction_s",
    "assembly_s",
    "communication_s",
    "preconditioner_setup_s",
    "krylov_solve_s",
    "globalization_s",
    "state_output_s",
    "total_s",
)

ACCURACY_VALUE_FIELDS = (
    "absolute_residual",
    "relative_residual",
    "scaled_residual",
    "relative_correction",
    "energy_change",
)

COUNT_VALUE_FIELDS = (
    "nonlinear_iterations",
    "krylov_iterations",
    "function_evaluations",
    "gradient_evaluations",
    "hessian_evaluations",
    "hvp_evaluations",
    "preconditioner_setups",
)

MEMORY_VALUE_FIELDS = (
    "peak_memory_per_rank_bytes",
    "peak_memory_per_node_bytes",
    "tracked_allocations_bytes",
)

# This exported JSON-Schema summary is deliberately small.  The Python
# validator below is the authoritative implementation of conditional rules,
# including publication-versus-pilot provenance and status-specific gates.
RUN_RECORD_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": f"urn:{RUN_RECORD_SCHEMA_ID}:v{RUN_RECORD_SCHEMA_VERSION}",
    "title": "Publication experiment run record",
    "type": "object",
    "required": list(TOP_LEVEL_FIELDS),
    "properties": {
        **{
            section: {"type": "object", "required": list(fields)}
            for section, fields in SECTION_FIELDS.items()
        },
        "record_id": {"type": "string", "minLength": 1},
        "run_kind": {"enum": sorted(RUN_KINDS)},
    },
    "additionalProperties": True,
}
RUN_RECORD_JSON_SCHEMA["properties"]["schema"]["properties"] = {
    "id": {"const": RUN_RECORD_SCHEMA_ID},
    "version": {"const": RUN_RECORD_SCHEMA_VERSION},
}
RUN_RECORD_JSON_SCHEMA["properties"]["termination"]["properties"] = {
    "status": {"enum": sorted(TERMINATION_STATUSES)}
}


class RunRecordValidationError(ValueError):
    """Raised when a run record violates the versioned contract."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("Invalid publication run record:\n- " + "\n- ".join(self.errors))


class ExperimentPreflightError(RuntimeError):
    """Raised when the repository state is unsuitable for the requested run."""


@dataclass(frozen=True, slots=True)
class ExperimentPreflight:
    """Git evidence captured immediately before an experiment starts."""

    run_kind: Literal["publication", "pilot"]
    git_commit: str
    git_clean: bool
    git_status_porcelain: tuple[str, ...]
    pilot_override: bool
    pilot_override_reason: str | None
    checked_at_utc: str

    def provenance_fields(self) -> dict[str, Any]:
        """Return fields that can be merged into the record provenance block."""
        return {
            "git_commit": self.git_commit,
            "git_clean": self.git_clean,
            "git_status_porcelain": list(self.git_status_porcelain),
            "pilot_override": self.pilot_override,
            "pilot_override_reason": self.pilot_override_reason,
        }


def utc_now_iso() -> str:
    """Return a seconds-resolution UTC timestamp ending in ``Z``."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "unknown git error"
        raise ExperimentPreflightError(f"Git preflight failed: {detail}")
    return proc.stdout


def check_experiment_preflight(
    repo_root: str | Path,
    *,
    run_kind: Literal["publication", "pilot"] = "publication",
    pilot_dirty_override: bool = False,
    pilot_override_reason: str | None = None,
) -> ExperimentPreflight:
    """Require a clean commit for publication evidence.

    A dirty worktree is allowed only when all three facts are explicit:
    ``run_kind="pilot"``, ``pilot_dirty_override=True``, and a non-empty reason.
    Such a run remains a pilot in its persisted record and cannot validate as
    publication evidence.
    """
    root = Path(repo_root).resolve()
    if run_kind not in RUN_KINDS:
        raise ExperimentPreflightError(f"Unknown run kind {run_kind!r}; use 'publication' or 'pilot'")
    reason = str(pilot_override_reason).strip() if pilot_override_reason is not None else None
    if run_kind == "publication" and (pilot_dirty_override or reason):
        raise ExperimentPreflightError(
            "A pilot override cannot be attached to a publication run; select run_kind='pilot'"
        )
    if pilot_dirty_override and not reason:
        raise ExperimentPreflightError("A dirty-pilot override requires a non-empty reason")
    if reason and not pilot_dirty_override:
        raise ExperimentPreflightError("A pilot override reason requires pilot_dirty_override=True")

    commit = _git(root, "rev-parse", "--verify", "HEAD").strip()
    status_text = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    status = tuple(line for line in status_text.splitlines() if line)
    clean = not status

    if not clean and run_kind == "publication":
        preview = "\n".join(status[:20])
        suffix = "\n..." if len(status) > 20 else ""
        raise ExperimentPreflightError(
            "Publication experiments require an empty git status --porcelain. "
            "Commit or remove the following changes before running:\n"
            f"{preview}{suffix}"
        )
    if not clean and not pilot_dirty_override:
        raise ExperimentPreflightError(
            "The pilot worktree is dirty. Re-run with pilot_dirty_override=True and record why; "
            "the resulting evidence cannot be used as publication data."
        )

    return ExperimentPreflight(
        run_kind=run_kind,
        git_commit=commit,
        git_clean=clean,
        git_status_porcelain=status,
        pilot_override=bool(pilot_dirty_override),
        pilot_override_reason=reason,
        checked_at_utc=utc_now_iso(),
    )


def atomic_write_json(
    path: str | Path,
    payload: Mapping[str, Any] | Sequence[Any],
    *,
    mode: int = 0o644,
    nonfinite_as_null: bool = False,
) -> None:
    """Durably replace ``path`` with one complete, standards-compliant JSON value.

    Serialization is completed before a same-directory temporary file is
    created.  The temporary file is flushed and fsynced before ``os.replace``;
    the containing directory is then fsynced.  A serialization or write error
    therefore leaves any previous checkpoint untouched.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = strict_json_dumps(
        payload,
        indent=2,
        sort_keys=True,
        nonfinite_as_null=bool(nonfinite_as_null),
    ) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, destination)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(destination.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def sanitize_json_value(value: Any) -> Any:
    """Return a JSON-shaped copy with every non-finite real replaced by ``None``.

    Solver histories use non-finite sentinels for quantities that are not
    applicable at a particular iteration (for example, a missing relative
    step before the first Newton update).  Bare ``NaN`` and ``Infinity`` are
    not valid JSON.  This recursive conversion preserves finite scientific
    values and container structure while representing unavailable optional
    quantities as JSON ``null``.

    Values outside the ordinary JSON type system are deliberately left for
    :func:`json.dumps` to reject.  Real and integral scalar subclasses (such as
    NumPy scalars) are normalized to Python scalars along the way.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, Mapping):
        return {key: sanitize_json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(child) for child in value]
    return value


def strict_json_dumps(
    payload: Any,
    *,
    indent: int | None = None,
    sort_keys: bool = False,
    nonfinite_as_null: bool = False,
) -> str:
    """Serialize standards-compliant JSON, optionally mapping non-finite reals to null.

    ``allow_nan=False`` is unconditional.  Callers writing raw iterative
    histories should opt into ``nonfinite_as_null``; validated publication run
    records retain the stricter default so accidental non-finite evidence is
    rejected rather than silently repaired.
    """
    normalized = sanitize_json_value(payload) if nonfinite_as_null else payload
    return json.dumps(
        normalized,
        indent=indent,
        sort_keys=sort_keys,
        allow_nan=False,
    )


def atomic_write_checkpoint(
    path: str | Path,
    *,
    record_id: str,
    sequence: int,
    progress: Mapping[str, Any],
    written_at_utc: str | None = None,
) -> dict[str, Any]:
    """Write one versioned periodic progress checkpoint atomically."""
    if not str(record_id).strip():
        raise ValueError("record_id must be non-empty")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    checkpoint = {
        "schema": {"id": CHECKPOINT_SCHEMA_ID, "version": CHECKPOINT_SCHEMA_VERSION},
        "record_id": str(record_id),
        "sequence": int(sequence),
        "written_at_utc": written_at_utc or utc_now_iso(),
        "progress": dict(progress),
    }
    atomic_write_json(path, checkpoint)
    return checkpoint


def atomic_write_run_record(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    require_publication_ready: bool | None = None,
) -> None:
    """Validate and atomically persist one terminal run record."""
    validate_run_record(record, require_publication_ready=require_publication_ready)
    atomic_write_json(path, record)


def _mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be an object")
        return None
    return value


def _required(mapping: Mapping[str, Any], fields: Sequence[str], path: str, errors: list[str]) -> None:
    for field in fields:
        if field not in mapping:
            errors.append(f"{path}.{field} is required")


def _nonempty_string(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path} must be a non-empty string")


def _nullable_finite_number(
    value: Any,
    path: str,
    errors: list[str],
    *,
    nonnegative: bool = False,
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        errors.append(f"{path} must be null or a finite number")
    elif nonnegative and float(value) < 0.0:
        errors.append(f"{path} must be non-negative")


def _nullable_nonnegative_integer(value: Any, path: str, errors: list[str]) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{path} must be null or a non-negative integer")


def _positive_integer(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        errors.append(f"{path} must be a positive integer")


def _timestamp(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        errors.append(f"{path} must be an ISO-8601 UTC timestamp ending in Z")
        return
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        errors.append(f"{path} must be a valid ISO-8601 timestamp")
        return
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        errors.append(f"{path} must be in UTC")


def _string_list(value: Any, path: str, errors: list[str], *, nonempty: bool = False) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        errors.append(f"{path} must be an array of strings")
    elif nonempty and not value:
        errors.append(f"{path} must not be empty")


def _hash_mapping(value: Any, path: str, errors: list[str]) -> None:
    mapping = _mapping(value, path, errors)
    if mapping is None:
        return
    for key, digest in mapping.items():
        if not isinstance(key, str) or not key:
            errors.append(f"{path} keys must be non-empty strings")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            errors.append(f"{path}.{key} must be a SHA-256 hexadecimal digest")


def _finite_json(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{path} contains a non-finite number")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _finite_json(child, f"{path}.{key}", errors)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_json(child, f"{path}[{index}]", errors)


def validate_run_record(
    record: Mapping[str, Any],
    *,
    require_publication_ready: bool | None = None,
) -> None:
    """Validate a terminal record and raise all discovered contract errors.

    ``require_publication_ready`` defaults to true for records whose
    ``run_kind`` is ``publication``.  Passing it explicitly is useful to make a
    publication ingestion boundary reject pilot records as well.
    """
    errors: list[str] = []
    if not isinstance(record, Mapping):
        raise RunRecordValidationError(("record must be an object",))
    _required(record, TOP_LEVEL_FIELDS, "record", errors)
    if errors:
        raise RunRecordValidationError(errors)

    _nonempty_string(record["record_id"], "record.record_id", errors)
    run_kind = record["run_kind"]
    if run_kind not in RUN_KINDS:
        errors.append("record.run_kind must be 'publication' or 'pilot'")
    # A record that labels itself as publication evidence is always checked at
    # publication strictness.  The explicit flag can only strengthen a pilot
    # ingestion boundary; it cannot weaken a publication record.
    publication_ready = run_kind == "publication" or require_publication_ready is True
    if publication_ready and run_kind != "publication":
        errors.append("record.run_kind must be 'publication' at a publication ingestion boundary")

    sections: dict[str, dict[str, Any]] = {}
    for section_name, required_fields in SECTION_FIELDS.items():
        section = _mapping(record[section_name], f"record.{section_name}", errors)
        normalized = dict(section or {})
        _required(normalized, required_fields, f"record.{section_name}", errors)
        # Populate missing values locally so validation can report independent
        # errors in other sections instead of stopping at the first omission.
        for field in required_fields:
            normalized.setdefault(field, None)
        sections[section_name] = normalized

    schema = sections["schema"]
    if schema["id"] != RUN_RECORD_SCHEMA_ID:
        errors.append(f"record.schema.id must equal {RUN_RECORD_SCHEMA_ID!r}")
    if schema["version"] != RUN_RECORD_SCHEMA_VERSION:
        errors.append(f"record.schema.version must equal {RUN_RECORD_SCHEMA_VERSION}")

    identifiers = sections["identifiers"]
    for field in ("campaign", "experiment", "case", "method", "route"):
        _nonempty_string(identifiers[field], f"record.identifiers.{field}", errors)
    _positive_integer(identifiers["repetition"], "record.identifiers.repetition", errors)

    problem = sections["problem"]
    for field in ("name", "mesh", "quadrature", "notes"):
        _nonempty_string(problem[field], f"record.problem.{field}", errors)
    _nullable_nonnegative_integer(problem["degree"], "record.problem.degree", errors)
    _nullable_nonnegative_integer(
        problem["total_degrees_of_freedom"], "record.problem.total_degrees_of_freedom", errors
    )
    _nullable_nonnegative_integer(
        problem["free_degrees_of_freedom"], "record.problem.free_degrees_of_freedom", errors
    )

    solver = sections["solver"]
    for field in ("algorithm", "implementation", "stopping_contract"):
        _nonempty_string(solver[field], f"record.solver.{field}", errors)
    _mapping(solver["parameters"], "record.solver.parameters", errors)
    _mapping(solver["preconditioner"], "record.solver.preconditioner", errors)

    termination = sections["termination"]
    status = termination["status"]
    if status not in TERMINATION_STATUSES:
        errors.append(f"record.termination.status must be one of {sorted(TERMINATION_STATUSES)}")
    _nonempty_string(termination["reason"], "record.termination.reason", errors)
    if termination["exit_code"] is not None and (
        isinstance(termination["exit_code"], bool) or not isinstance(termination["exit_code"], int)
    ):
        errors.append("record.termination.exit_code must be null or an integer")
    _timestamp(termination["started_at_utc"], "record.termination.started_at_utc", errors)
    _timestamp(termination["finished_at_utc"], "record.termination.finished_at_utc", errors)
    if not isinstance(termination["censored"], bool):
        errors.append("record.termination.censored must be a boolean")
    if status in {"capped", "timeout"}:
        _nonempty_string(termination["limit_kind"], "record.termination.limit_kind", errors)
        if termination["limit_value"] is None:
            errors.append("record.termination.limit_value is required for capped or timed-out runs")
        if termination["censored"] is not True:
            errors.append("record.termination.censored must be true for capped or timed-out runs")
    elif termination["limit_kind"] is not None and not isinstance(termination["limit_kind"], str):
        errors.append("record.termination.limit_kind must be null or a string")
    if status == "success" and termination["censored"] is not False:
        errors.append("record.termination.censored must be false for successful runs")

    accuracy = sections["accuracy"]
    _nonempty_string(accuracy["contract_id"], "record.accuracy.contract_id", errors)
    if accuracy["gate_passed"] is not None and not isinstance(accuracy["gate_passed"], bool):
        errors.append("record.accuracy.gate_passed must be null or a boolean")
    if status == "success" and accuracy["gate_passed"] is not True:
        errors.append("record.accuracy.gate_passed must be true for a successful run")
    if status in {"failure", "capped", "timeout"} and accuracy["gate_passed"] is True:
        errors.append("record.accuracy.gate_passed cannot be true for a non-success terminal status")
    for field in ACCURACY_VALUE_FIELDS:
        _nullable_finite_number(
            accuracy[field],
            f"record.accuracy.{field}",
            errors,
            nonnegative=field != "energy_change",
        )
    _mapping(accuracy["custom_metrics"], "record.accuracy.custom_metrics", errors)
    _nonempty_string(accuracy["notes"], "record.accuracy.notes", errors)

    counts = sections["counts"]
    for field in COUNT_VALUE_FIELDS:
        _nullable_nonnegative_integer(counts[field], f"record.counts.{field}", errors)
    _nonempty_string(counts["notes"], "record.counts.notes", errors)

    timing = sections["timing"]
    for field in ("aggregation", "barrier_policy", "synchronization_policy", "relation_to_total", "notes"):
        _nonempty_string(timing[field], f"record.timing.{field}", errors)
    if not isinstance(timing["cold_process"], bool):
        errors.append("record.timing.cold_process must be a boolean")
    if not isinstance(timing["phases_overlap"], bool):
        errors.append("record.timing.phases_overlap must be a boolean")
    for field in TIMING_VALUE_FIELDS:
        _nullable_finite_number(timing[field], f"record.timing.{field}", errors, nonnegative=True)
    if timing["total_s"] is None:
        errors.append("record.timing.total_s must be measured for every terminal status")

    resources = sections["resources"]
    for field in ("nodes", "ranks", "threads_per_rank"):
        _positive_integer(resources[field], f"record.resources.{field}", errors)
    for field in MEMORY_VALUE_FIELDS:
        _nullable_nonnegative_integer(resources[field], f"record.resources.{field}", errors)
    for field in ("measurement_method", "notes"):
        _nonempty_string(resources[field], f"record.resources.{field}", errors)

    diagnostics = sections["diagnostics"]
    for field in SECTION_FIELDS["diagnostics"]:
        _mapping(diagnostics[field], f"record.diagnostics.{field}", errors)

    environment = sections["environment"]
    for field in (
        "python",
        "platform",
        "jax",
        "xla",
        "petsc",
        "mpi",
        "compiler",
        "blas",
        "cpu_model",
        "node_model",
        "memory_model",
        "scheduler",
        "affinity",
    ):
        _nonempty_string(environment[field], f"record.environment.{field}", errors)
    packages = _mapping(environment["packages"], "record.environment.packages", errors)
    if publication_ready and packages is not None and not packages:
        errors.append("record.environment.packages must not be empty for publication evidence")
    if environment["jax_enable_x64"] is not None and not isinstance(environment["jax_enable_x64"], bool):
        errors.append("record.environment.jax_enable_x64 must be null or a boolean")
    if environment["scheduler_job_id"] is not None and not isinstance(environment["scheduler_job_id"], str):
        errors.append("record.environment.scheduler_job_id must be null or a string")

    provenance = sections["provenance"]
    commit = provenance["git_commit"]
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-fA-F]{40,64}", commit) is None:
        errors.append("record.provenance.git_commit must be a full 40--64 digit hexadecimal commit")
    if not isinstance(provenance["git_clean"], bool):
        errors.append("record.provenance.git_clean must be a boolean")
    _string_list(provenance["git_status_porcelain"], "record.provenance.git_status_porcelain", errors)
    if not isinstance(provenance["pilot_override"], bool):
        errors.append("record.provenance.pilot_override must be a boolean")
    pilot_reason = provenance["pilot_override_reason"]
    if provenance["pilot_override"]:
        _nonempty_string(pilot_reason, "record.provenance.pilot_override_reason", errors)
    elif pilot_reason is not None:
        errors.append("record.provenance.pilot_override_reason must be null without an override")
    if provenance["git_clean"] and provenance["git_status_porcelain"]:
        errors.append("record.provenance.git_status_porcelain must be empty when git_clean is true")
    if not provenance["git_clean"] and not provenance["git_status_porcelain"]:
        errors.append("record.provenance.git_status_porcelain must describe a dirty tree")
    if publication_ready:
        if provenance["git_clean"] is not True or provenance["git_status_porcelain"]:
            errors.append("publication evidence must come from a clean worktree")
        if provenance["pilot_override"] is not False:
            errors.append("publication evidence cannot use a pilot override")
    elif run_kind == "pilot" and not provenance["git_clean"] and not provenance["pilot_override"]:
        errors.append("a dirty pilot record requires an explicit pilot override")
    _string_list(provenance["command_argv"], "record.provenance.command_argv", errors, nonempty=True)
    _nonempty_string(provenance["working_directory"], "record.provenance.working_directory", errors)
    for field in ("code_hashes", "configuration_hashes", "input_hashes"):
        _hash_mapping(provenance[field], f"record.provenance.{field}", errors)
    dirty_digest = provenance["dirty_patch_sha256"]
    if dirty_digest is not None and (
        not isinstance(dirty_digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", dirty_digest) is None
    ):
        errors.append("record.provenance.dirty_patch_sha256 must be null or a SHA-256 digest")
    if provenance["git_clean"] and dirty_digest is not None:
        errors.append("record.provenance.dirty_patch_sha256 must be null for a clean run")
    if provenance["seed"] is not None and (
        isinstance(provenance["seed"], bool) or not isinstance(provenance["seed"], int)
    ):
        errors.append("record.provenance.seed must be null or an integer")
    _nonempty_string(provenance["deterministic_policy"], "record.provenance.deterministic_policy", errors)
    _timestamp(provenance["recorded_at_utc"], "record.provenance.recorded_at_utc", errors)

    artifacts = sections["artifacts"]
    for field in SECTION_FIELDS["artifacts"]:
        _string_list(artifacts[field], f"record.artifacts.{field}", errors)
    if publication_ready and not (artifacts["raw_outputs"] or artifacts["logs"]):
        errors.append("publication evidence must retain at least one raw output or log")

    _finite_json(record, "record", errors)
    if errors:
        raise RunRecordValidationError(errors)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one provenance input."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

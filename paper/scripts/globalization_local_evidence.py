#!/usr/bin/env python3
"""Fail-closed admission for the local controlled EXP-GLOB-001 campaign.

The admission deliberately ignores all recorded timings.  It closes every
campaign JSON, NPZ, CSV, and log file by SHA-256, validates the clean source
commit against Git, reloads every state archive, and independently adjudicates
the common-start, completion, and same-endpoint gates.  The three prescribed
starts are a deterministic sensitivity set, not a sampled population.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_ID = "fenics-nonlinear-energies.exp-glob-001-local-paper-admission"
SCHEMA_VERSION = 1
CAMPAIGN_SCHEMA_ID = "fenics-nonlinear-energies.exp-glob-001-campaign"
COMMON_START_SCHEMA_ID = "fenics-nonlinear-energies.exp-glob-001-common-starts"
RUN_RECORD_SCHEMA_ID = "fenics-nonlinear-energies.publication-run-record"
CAMPAIGN_ID = "paper-revision-exp-glob-001-local-v1"
RAW_RELATIVE = Path("raw/controlled/smoke")
REPORT_RELATIVE = Path("reports/controlled")
CAMPAIGN_MANIFEST_RELATIVE = RAW_RELATIVE / "campaign_manifest.json"
START_MANIFEST_RELATIVE = RAW_RELATIVE / "_canonical_starts/manifest.json"
SUMMARY_JSON_RELATIVE = REPORT_RELATIVE / "smoke_summary.json"
SUMMARY_CSV_RELATIVE = REPORT_RELATIVE / "smoke_summary.csv"
IDENTITY_AUDIT_RELATIVE = REPORT_RELATIVE / "smoke_identity_audit.json"
BENCHMARKS = ("gl_l5_np2", "he_l2_np2_step1")
METHODS = ("newton_armijo", "reduced_trust_armijo")
INSTANCES = ("nominal", "mode_plus", "mode_minus")
REPETITIONS = (1, 2, 3, 4, 5)
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
TABLE_NAME = "globalization_local_status.tex"
MANIFEST_NAME = "globalization_local_manifest.json"


class AdmissionError(ValueError):
    """Raised when local globalization evidence is not publication-admissible."""


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
    """Match the solver's dtype-and-shape-aware state digest."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(int(value) for value in array.shape)).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _assert_finite_json(value: object, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AdmissionError(f"{label} contains a nonfinite number")
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
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdmissionError(f"{path}: top-level JSON value must be an object")
    _assert_finite_json(value, label=str(path))
    return value


def _assert_no_symlink(path: Path, *, root: Path, label: str) -> None:
    root = root.absolute()
    path = path.absolute()
    if root.is_symlink():
        raise AdmissionError(f"{label}: evidence root must not be a symlink")
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
        raise AdmissionError(f"{label} does not identify the canonical artifact")
    if not resolved.is_file():
        raise AdmissionError(f"{label} is missing: {resolved}")
    return resolved


def _canonical_artifact(
    relative: Path, *, evidence_root: Path, label: str
) -> Path:
    path = (evidence_root / relative).absolute()
    _assert_no_symlink(path, root=evidence_root, label=label)
    if not path.is_file():
        raise AdmissionError(f"{label} is missing: {path}")
    return path


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AdmissionError(completed.stderr.strip() or "Git command failed")
    return completed.stdout.strip()


def _validate_commit(
    commit: object, *, repo_root: Path, require_release_clean: bool
) -> str:
    source_commit = str(commit).lower()
    if HEX40.fullmatch(source_commit) is None:
        raise AdmissionError("experiment source commit must be one exact 40-character SHA-1")
    head = _git_output(repo_root, "rev-parse", "HEAD").lower()
    ancestor = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", source_commit, head],
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise AdmissionError("experiment source commit is not an ancestor of release HEAD")
    if require_release_clean and _git_output(
        repo_root, "status", "--porcelain=v1", "--untracked-files=all"
    ):
        raise AdmissionError("release worktree must be clean during evidence admission")
    return source_commit


def _git_blob_sha256(repo_root: Path, commit: str, relative: str) -> str:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise AdmissionError(f"unsafe Git-bound path: {relative!r}")
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{commit}:{relative}"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise AdmissionError(f"path is absent from experiment commit {commit}: {relative}")
    return hashlib.sha256(completed.stdout).hexdigest()


def _verify_git_inventory(
    raw: object, *, repo_root: Path, commit: str, label: str
) -> dict[str, str]:
    if not isinstance(raw, dict) or not raw:
        raise AdmissionError(f"{label} must be a nonempty hash map")
    inventory: dict[str, str] = {}
    for relative, digest in sorted(raw.items()):
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise AdmissionError(f"{label} has a malformed entry")
        if HEX64.fullmatch(digest) is None:
            raise AdmissionError(f"{label} has a malformed SHA-256 for {relative}")
        if _git_blob_sha256(repo_root, commit, relative) != digest:
            raise AdmissionError(f"{label} differs from experiment-commit blob: {relative}")
        inventory[relative] = digest
    return inventory


def _load_npz(path: Path, *, required: Sequence[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set(required) - set(archive.files))
            if missing:
                raise AdmissionError(f"{path}: missing arrays {missing}")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise AdmissionError(f"cannot load safe NPZ {path}: {exc}") from exc
    for name, values in arrays.items():
        if values.dtype.kind in "fc" and not np.all(np.isfinite(values)):
            raise AdmissionError(f"{path}: array {name} contains nonfinite values")
    return arrays


def _state_digest(
    arrays: Mapping[str, np.ndarray], benchmark: str, *, flatten_he: bool
) -> str:
    if benchmark.startswith("gl_"):
        return array_sha256(np.asarray(arrays["u"], dtype=np.float64).reshape(-1))
    values = np.asarray(arrays["coords_final"], dtype=np.float64)
    return array_sha256(values.reshape(-1) if flatten_he else values)


def _expected_case_ids() -> set[str]:
    return {
        f"{benchmark}_{instance}_{method}_r{repetition:02d}"
        for benchmark in BENCHMARKS
        for instance in INSTANCES
        for method in METHODS
        for repetition in REPETITIONS
    }


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdmissionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise AdmissionError(f"{label} must be finite and nonnegative")
    return result


def _nested(mapping: Mapping[str, object], *keys: str, label: str) -> object:
    value: object = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise AdmissionError(f"{label} is missing")
        value = value[key]
    return value


def _validate_configuration(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise AdmissionError("campaign configuration is missing")
    if configuration.get("campaign_id") != CAMPAIGN_ID:
        raise AdmissionError("campaign configuration id is invalid")
    if configuration.get("mode") != "smoke" or configuration.get("comparison_tier") != "controlled":
        raise AdmissionError("only the controlled local smoke design is admissible")
    if configuration.get("maximum_local_ranks") != 4:
        raise AdmissionError("local-rank ceiling differs from the frozen design")
    if configuration.get("machine_noise_repetitions") != 5:
        raise AdmissionError("campaign must retain exactly five process repetitions")
    environment = configuration.get("controlled_child_environment")
    expected_environment = {
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
    }
    if environment != expected_environment:
        raise AdmissionError("controlled child environment differs from the frozen design")
    cases = configuration.get("cases")
    if not isinstance(cases, list) or len(cases) != 60:
        raise AdmissionError("campaign configuration must contain exactly 60 cases")
    by_id: dict[str, dict[str, object]] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise AdmissionError("campaign case is not an object")
        benchmark = case.get("benchmark")
        method = case.get("method")
        instance = case.get("robustness_instance")
        if not isinstance(benchmark, dict) or not isinstance(method, dict) or not isinstance(instance, dict):
            raise AdmissionError("campaign case identity is incomplete")
        benchmark_id = str(benchmark.get("key"))
        method_id = str(method.get("key"))
        instance_id = str(instance.get("instance_id"))
        repetition = int(case.get("timing_repetition", 0))
        case_id = f"{benchmark_id}_{instance_id}_{method_id}_r{repetition:02d}"
        if case_id in by_id:
            raise AdmissionError(f"duplicate campaign case: {case_id}")
        if case.get("mode") != "smoke" or case.get("comparison_tier") != "controlled":
            raise AdmissionError(f"{case_id}: case is outside the controlled smoke tier")
        if benchmark.get("ranks") != 2:
            raise AdmissionError(f"{case_id}: local case must use exactly two ranks")
        by_id[case_id] = case
    if set(by_id) != _expected_case_ids():
        raise AdmissionError("campaign case grid differs from the frozen 60-run design")
    if manifest.get("configuration_sha256") != json_sha256(configuration):
        raise AdmissionError("campaign configuration digest is invalid")
    return by_id


def _validate_start_manifest(
    path: Path, *, repo_root: Path, evidence_root: Path
) -> tuple[dict[tuple[str, str], dict[str, str]], set[Path]]:
    payload = read_strict_json(path)
    if payload.get("schema_id") != COMMON_START_SCHEMA_ID or payload.get("schema_version") != 2:
        raise AdmissionError("canonical-start manifest schema is invalid")
    if payload.get("status") != "prepared":
        raise AdmissionError("canonical-start manifest is not prepared")
    instances = payload.get("instances")
    if not isinstance(instances, dict) or set(instances) != {
        f"{benchmark}::{instance}" for benchmark in BENCHMARKS for instance in INSTANCES
    }:
        raise AdmissionError("canonical-start instance grid is incomplete")
    identities: dict[tuple[str, str], dict[str, str]] = {}
    files: set[Path] = {path}
    for key, raw in sorted(instances.items()):
        if not isinstance(raw, dict):
            raise AdmissionError(f"canonical start {key} is malformed")
        benchmark, instance = key.split("::", 1)
        expected = RAW_RELATIVE / "_canonical_starts" / benchmark / f"{instance}.npz"
        state_path = _safe_artifact(
            raw.get("path"),
            repo_root=repo_root,
            evidence_root=evidence_root,
            label=f"canonical start {key}",
            expected=evidence_root / expected,
        )
        digest = str(raw.get("file_sha256", ""))
        if HEX64.fullmatch(digest) is None or sha256_file(state_path) != digest:
            raise AdmissionError(f"canonical start {key} file SHA-256 mismatch")
        required = ("u",) if benchmark.startswith("gl_") else ("coords_final",)
        arrays = _load_npz(state_path, required=required)
        content = _state_digest(arrays, benchmark, flatten_he=False)
        if raw.get("state_sha256") != content:
            raise AdmissionError(f"canonical start {key} content SHA-256 mismatch")
        if raw.get("benchmark") != benchmark or raw.get("robustness_instance") != instance:
            raise AdmissionError(f"canonical start {key} identity mismatch")
        identities[(benchmark, instance)] = {"file": digest, "content": content}
        files.add(state_path)
    return identities, files


def _validate_report_rows(path: Path) -> dict[str, dict[str, object]]:
    payload = read_strict_json(path)
    if payload.get("mode") != "smoke" or payload.get("comparison_tier") != "controlled":
        raise AdmissionError("summary report is outside the controlled smoke tier")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 60:
        raise AdmissionError("summary report must contain exactly 60 rows")
    by_id: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise AdmissionError("summary row is not an object")
        case_id = (
            f"{row.get('benchmark')}_{row.get('robustness_instance')}_"
            f"{row.get('method')}_r{int(row.get('timing_repetition', 0)):02d}"
        )
        if case_id in by_id:
            raise AdmissionError(f"duplicate summary row: {case_id}")
        for field in (
            "wall_time_s",
            "line_search_time_s",
            "independent_dual_residual",
            "independent_coefficient_residual",
        ):
            if row.get(field) not in (None, ""):
                _finite_nonnegative(row[field], label=f"{case_id}.{field}")
        if row.get("result") not in {"completed", "failed", "timeout", "launcher_failed"}:
            raise AdmissionError(f"{case_id}: unsupported result classification")
        by_id[case_id] = row
    if set(by_id) != _expected_case_ids():
        raise AdmissionError("summary row grid differs from the frozen 60-run design")
    return by_id


def _validate_summary_csv(path: Path, rows: Mapping[str, Mapping[str, object]]) -> None:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise AdmissionError(f"cannot parse summary CSV: {exc}") from exc
    if len(csv_rows) != 60:
        raise AdmissionError("summary CSV must contain exactly 60 rows")
    observed: dict[str, Mapping[str, str]] = {}
    for row in csv_rows:
        case_id = (
            f"{row.get('benchmark')}_{row.get('robustness_instance')}_"
            f"{row.get('method')}_r{int(row.get('timing_repetition') or 0):02d}"
        )
        observed[case_id] = row
        source = rows.get(case_id)
        if source is None or row.get("result") != str(source.get("result")):
            raise AdmissionError(f"summary CSV differs from JSON for {case_id}")
        for field in (
            "initial_state_file_sha256",
            "initial_state_content_sha256",
            "final_state_file_sha256",
            "final_state_content_sha256",
            "endpoint_state_sha256",
            "independent_residual_sha256",
        ):
            if row.get(field, "") != str(source.get(field, "")):
                raise AdmissionError(f"summary CSV identity differs for {case_id}.{field}")
    if set(observed) != _expected_case_ids():
        raise AdmissionError("summary CSV case grid is incomplete")


def _run_artifact_paths(
    run_record: Mapping[str, object],
    *,
    repo_root: Path,
    evidence_root: Path,
    case_id: str,
) -> tuple[Path, Path, Path, Path]:
    artifacts = run_record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise AdmissionError(f"{case_id}: run-record artifact map is missing")
    raw_outputs = artifacts.get("raw_outputs")
    logs = artifacts.get("logs")
    states = artifacts.get("states")
    if not isinstance(raw_outputs, list) or len(raw_outputs) != 1:
        raise AdmissionError(f"{case_id}: exactly one raw output is required")
    if not isinstance(logs, list) or len(logs) != 1:
        raise AdmissionError(f"{case_id}: exactly one log is required")
    if not isinstance(states, list) or len(states) != 2:
        raise AdmissionError(f"{case_id}: canonical and terminal states are required")
    case_root = evidence_root / RAW_RELATIVE / case_id
    output = _safe_artifact(
        raw_outputs[0], repo_root=repo_root, evidence_root=evidence_root,
        label=f"{case_id} raw output", expected=case_root / "output.json"
    )
    log = _safe_artifact(
        logs[0], repo_root=repo_root, evidence_root=evidence_root,
        label=f"{case_id} log", expected=case_root / "run.log"
    )
    terminal = _safe_artifact(
        states[1], repo_root=repo_root, evidence_root=evidence_root,
        label=f"{case_id} terminal state", expected=case_root / "final_state.npz"
    )
    initial = _safe_artifact(
        states[0], repo_root=repo_root, evidence_root=evidence_root,
        label=f"{case_id} initial state"
    )
    return output, log, initial, terminal


def _validate_one_run(
    *,
    case_id: str,
    row: Mapping[str, object],
    planned: Mapping[str, object],
    configuration: Mapping[str, object],
    record_path: Path,
    record_hash: str,
    source_commit: str,
    starts: Mapping[tuple[str, str], Mapping[str, str]],
    repo_root: Path,
    evidence_root: Path,
) -> tuple[dict[str, object], set[Path]]:
    if sha256_file(record_path) != record_hash:
        raise AdmissionError(f"{case_id}: run-record SHA-256 mismatch")
    record = read_strict_json(record_path)
    schema = record.get("schema")
    if schema != {"id": RUN_RECORD_SCHEMA_ID, "version": 1}:
        raise AdmissionError(f"{case_id}: run-record schema is invalid")
    if record.get("run_kind") != "publication":
        raise AdmissionError(f"{case_id}: run record is not publication evidence")
    provenance = record.get("provenance")
    identifiers = record.get("identifiers")
    diagnostics = record.get("diagnostics")
    accuracy = record.get("accuracy")
    termination = record.get("termination")
    if not all(isinstance(value, dict) for value in (provenance, identifiers, diagnostics, accuracy, termination)):
        raise AdmissionError(f"{case_id}: run-record scientific fields are incomplete")
    assert isinstance(provenance, dict)
    assert isinstance(identifiers, dict)
    assert isinstance(diagnostics, dict)
    assert isinstance(accuracy, dict)
    assert isinstance(termination, dict)
    if provenance.get("git_commit") != source_commit or provenance.get("git_clean") is not True:
        raise AdmissionError(f"{case_id}: run record lacks the clean exact source commit")
    if provenance.get("git_status_porcelain") != [] or provenance.get("dirty_patch_sha256") is not None:
        raise AdmissionError(f"{case_id}: run record contains dirty-source provenance")
    benchmark = str(row.get("benchmark"))
    instance = str(row.get("robustness_instance"))
    method = str(row.get("method"))
    repetition = int(row.get("timing_repetition", 0))
    if identifiers.get("experiment") != "EXP-GLOB-001" or identifiers.get("campaign") != CAMPAIGN_ID:
        raise AdmissionError(f"{case_id}: run-record experiment identity is invalid")
    if identifiers.get("method") != method or identifiers.get("repetition") != repetition:
        raise AdmissionError(f"{case_id}: run-record method/repetition identity differs")
    if planned.get("case_id") != case_id or planned.get("benchmark") != benchmark:
        raise AdmissionError(f"{case_id}: planned-run identity differs")
    if planned.get("method") != method or planned.get("robustness_instance") != instance:
        raise AdmissionError(f"{case_id}: planned-run method/instance differs")
    if planned.get("timing_repetition") != repetition:
        raise AdmissionError(f"{case_id}: planned-run repetition differs")
    expected_command = _nested(provenance, "command_argv", label=f"{case_id}.command_argv")
    if planned.get("command_argv") != expected_command or planned.get("command_sha256") != json_sha256(expected_command):
        raise AdmissionError(f"{case_id}: planned command binding is invalid")
    _verify_git_inventory(
        provenance.get("code_hashes"), repo_root=repo_root, commit=source_commit,
        label=f"{case_id}.code_hashes"
    )
    configuration_hashes = provenance.get("configuration_hashes")
    if not isinstance(configuration_hashes, dict):
        raise AdmissionError(f"{case_id}: configuration hashes are missing")
    protocol_hashes = {
        key: value
        for key, value in configuration_hashes.items()
        if isinstance(key, str) and key.endswith(".md")
    }
    _verify_git_inventory(
        protocol_hashes, repo_root=repo_root, commit=source_commit,
        label=f"{case_id}.protocol_hashes"
    )
    if configuration_hashes.get("campaign_configuration") != json_sha256(configuration):
        raise AdmissionError(f"{case_id}: campaign configuration hash differs")
    output, log, initial, terminal = _run_artifact_paths(
        record, repo_root=repo_root, evidence_root=evidence_root, case_id=case_id
    )
    start = starts[(benchmark, instance)]
    if sha256_file(initial) != start["file"]:
        raise AdmissionError(f"{case_id}: initial-state file differs from canonical start")
    state_diag = diagnostics.get("state")
    if not isinstance(state_diag, dict):
        raise AdmissionError(f"{case_id}: state diagnostics are missing")
    if state_diag.get("initial_file_sha256") != start["file"] or state_diag.get("initial_content_sha256") != start["content"]:
        raise AdmissionError(f"{case_id}: canonical start identity differs in run record")
    terminal_file_hash = sha256_file(terminal)
    if state_diag.get("final_file_sha256") != terminal_file_hash:
        raise AdmissionError(f"{case_id}: terminal-state file hash differs")
    required = ("u",) if benchmark.startswith("gl_") else ("coords_final",)
    arrays = _load_npz(terminal, required=required)
    terminal_content_hash = _state_digest(arrays, benchmark, flatten_he=True)
    if state_diag.get("final_content_sha256") != terminal_content_hash:
        raise AdmissionError(f"{case_id}: terminal-state content hash differs")
    output_payload = read_strict_json(output)
    metadata = _nested(output_payload, "result", "metadata", label=f"{case_id}.metadata")
    if not isinstance(metadata, Mapping):
        raise AdmissionError(f"{case_id}: solver metadata is malformed")
    initial_identity = _nested(metadata, "initial_state_input", label=f"{case_id}.initial_state_input")
    state_output = _nested(metadata, "state_output", label=f"{case_id}.state_output")
    endpoint_identity = _nested(metadata, "endpoint_identity", label=f"{case_id}.endpoint_identity")
    if not all(isinstance(value, Mapping) for value in (initial_identity, state_output, endpoint_identity)):
        raise AdmissionError(f"{case_id}: terminal identities are malformed")
    assert isinstance(initial_identity, Mapping)
    assert isinstance(state_output, Mapping)
    assert isinstance(endpoint_identity, Mapping)
    independent = endpoint_identity.get("independent_residual")
    if not isinstance(independent, Mapping):
        raise AdmissionError(f"{case_id}: independently evaluated residual is missing")
    residual = _finite_nonnegative(independent.get("dual_norm"), label=f"{case_id}.dual_residual")
    coefficient = _finite_nonnegative(
        independent.get("coefficient_l2_norm"), label=f"{case_id}.coefficient_residual"
    )
    residual_hash = str(independent.get("owned_reordered_gradient_sha256", ""))
    endpoint_hash = str(endpoint_identity.get("owned_reordered_state_sha256", ""))
    if HEX64.fullmatch(residual_hash) is None or HEX64.fullmatch(endpoint_hash) is None:
        raise AdmissionError(f"{case_id}: endpoint/residual identity hash is malformed")
    if independent.get("evaluated_after_solver_termination") is not True:
        raise AdmissionError(f"{case_id}: residual was not independently evaluated after termination")
    if initial_identity.get("file_sha256") != start["file"] or initial_identity.get("state_sha256") != start["content"]:
        raise AdmissionError(f"{case_id}: output canonical-start identity differs")
    if state_output.get("file_sha256") != terminal_file_hash or state_output.get("state_sha256") != terminal_content_hash:
        raise AdmissionError(f"{case_id}: output terminal-state identity differs")
    expected_row = {
        "initial_state_file_sha256": start["file"],
        "initial_state_content_sha256": start["content"],
        "final_state_file_sha256": terminal_file_hash,
        "final_state_content_sha256": terminal_content_hash,
        "endpoint_state_sha256": endpoint_hash,
        "independent_residual_sha256": residual_hash,
    }
    for field, expected in expected_row.items():
        if row.get(field) != expected:
            raise AdmissionError(f"{case_id}: summary identity differs for {field}")
    if float(row.get("independent_dual_residual")) != residual or float(
        row.get("independent_coefficient_residual")
    ) != coefficient:
        raise AdmissionError(f"{case_id}: summary residual differs from raw output")
    if accuracy.get("absolute_residual") != residual:
        raise AdmissionError(f"{case_id}: run-record residual differs from raw output")
    result = str(row.get("result"))
    completed = result == "completed"
    if termination.get("status") != ("success" if completed else "failure"):
        raise AdmissionError(f"{case_id}: termination classification differs from summary")
    if accuracy.get("gate_passed") is not completed:
        raise AdmissionError(f"{case_id}: accuracy gate differs from completion status")
    if row.get("run_record_sha256") != record_hash:
        raise AdmissionError(f"{case_id}: summary run-record hash differs")
    return (
        {
            "case_id": case_id,
            "benchmark": benchmark,
            "instance": instance,
            "method": method,
            "repetition": repetition,
            "result": result,
            "residual": residual,
            "start_file_sha256": start["file"],
            "start_content_sha256": start["content"],
            "terminal_file_sha256": terminal_file_hash,
            "terminal_content_sha256": terminal_content_hash,
            "endpoint_state_sha256": endpoint_hash,
        },
        {record_path, output, log, initial, terminal},
    )


def _adjudicate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_unit: dict[tuple[str, str, int], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["benchmark"]), str(row["instance"]), int(row["repetition"]))
        by_unit.setdefault(key, []).append(row)
    units: list[dict[str, object]] = []
    common_start_passed = True
    endpoint_comparison_passed = True
    all_completed = True
    for (benchmark, instance, repetition), group in sorted(by_unit.items()):
        methods = {str(row["method"]) for row in group}
        starts = {
            (str(row["start_file_sha256"]), str(row["start_content_sha256"]))
            for row in group
        }
        completed = all(row["result"] == "completed" for row in group)
        endpoints = {
            (str(row["terminal_content_sha256"]), str(row["endpoint_state_sha256"]))
            for row in group
        }
        common = methods == set(METHODS) and len(starts) == 1
        comparable = completed and len(endpoints) == 1
        common_start_passed &= common
        endpoint_comparison_passed &= comparable
        all_completed &= completed
        units.append(
            {
                "benchmark": benchmark,
                "instance": instance,
                "repetition": repetition,
                "common_start_passed": common,
                "all_methods_completed": completed,
                "same_endpoint_passed": comparable,
                "results": {str(row["method"]): str(row["result"]) for row in group},
            }
        )
    summaries: list[dict[str, object]] = []
    for benchmark in BENCHMARKS:
        for method in METHODS:
            selected = [
                row for row in rows
                if row["benchmark"] == benchmark and row["method"] == method
            ]
            summaries.append(
                {
                    "benchmark": benchmark,
                    "method": method,
                    "completed": sum(row["result"] == "completed" for row in selected),
                    "failed": sum(row["result"] != "completed" for row in selected),
                    "total": len(selected),
                }
            )
    return {
        "status": "bounded_outcomes_admitted",
        "common_start_gate_passed": common_start_passed,
        "all_methods_completed_gate_passed": all_completed,
        "same_endpoint_comparison_gate_passed": endpoint_comparison_passed,
        "tested_instance_method_comparison_admissible": bool(
            common_start_passed and all_completed and endpoint_comparison_passed
        ),
        "timing_claim_admissible": False,
        "population_robustness_claim_admissible": False,
        "claim_scope": (
            "Observed completion/failure counts for the frozen two-problem, "
            "three-start, five-repetition deterministic local design only."
        ),
        "claim_refusals": [
            "No timing, speedup, or performance-ordering claim is admitted.",
            "No robustness generalization is admitted because the starts are not population samples.",
            "A paired method comparison requires successful termination at one exact endpoint identity.",
        ],
        "method_summaries": summaries,
        "paired_units": units,
    }


def _verify_source_report_audit(
    path: Path, *, adjudication: Mapping[str, object]
) -> None:
    payload = read_strict_json(path)
    if payload.get("schema_id") != "fenics-nonlinear-energies.exp-glob-001-identity-audit" or payload.get("schema_version") != 2:
        raise AdmissionError("source identity-audit schema is invalid")
    if payload.get("robustness_generalization_claim_admissible") is not False:
        raise AdmissionError("source identity audit promotes an unsupported robustness claim")
    if bool(payload.get("tested_instance_comparison_admissible")) != bool(
        adjudication["tested_instance_method_comparison_admissible"]
    ):
        raise AdmissionError("source identity audit differs from independent comparison adjudication")
    if bool(payload.get("timing_claim_admissible")) and not bool(
        adjudication["tested_instance_method_comparison_admissible"]
    ):
        raise AdmissionError("source identity audit promotes timing despite failed identity gates")


def _tree_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AdmissionError(f"symlinks are forbidden in evidence tree: {path}")
        if path.is_file():
            hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def audit_campaign(
    evidence_root: Path,
    *,
    repo_root: Path,
    require_release_clean: bool = True,
) -> dict[str, object]:
    """Audit one completed local campaign and return deterministic adjudication."""

    repo_root = repo_root.resolve()
    evidence_root = evidence_root.absolute()
    reproduction_root = (repo_root / "artifacts/reproduction").resolve()
    _assert_no_symlink(evidence_root, root=reproduction_root, label="evidence_root")
    try:
        evidence_root.resolve().relative_to(reproduction_root)
    except ValueError as exc:
        raise AdmissionError("evidence_root must be below artifacts/reproduction") from exc
    campaign_path = _canonical_artifact(
        CAMPAIGN_MANIFEST_RELATIVE, evidence_root=evidence_root, label="campaign manifest"
    )
    campaign = read_strict_json(campaign_path)
    if campaign.get("schema") != {"id": CAMPAIGN_SCHEMA_ID, "version": 1}:
        raise AdmissionError("campaign manifest schema is invalid")
    if campaign.get("campaign_id") != CAMPAIGN_ID:
        raise AdmissionError("campaign id is invalid")
    if campaign.get("status") not in {"completed", "completed_with_failed_identity_gate"}:
        raise AdmissionError("campaign is not terminal")
    preflight = campaign.get("publication_preflight")
    if not isinstance(preflight, dict) or preflight.get("git_clean") is not True:
        raise AdmissionError("campaign was not launched from a clean publication preflight")
    if preflight.get("git_status_porcelain") != [] or preflight.get("pilot_override") is not False:
        raise AdmissionError("campaign publication preflight is dirty or overridden")
    source_commit = _validate_commit(
        preflight.get("git_commit"), repo_root=repo_root,
        require_release_clean=require_release_clean
    )
    configuration = _validate_configuration(campaign)
    _verify_git_inventory(
        campaign.get("source_hashes"), repo_root=repo_root, commit=source_commit,
        label="campaign source_hashes"
    )
    _verify_git_inventory(
        campaign.get("protocol_hashes"), repo_root=repo_root, commit=source_commit,
        label="campaign protocol_hashes"
    )
    start_manifest_path = _canonical_artifact(
        START_MANIFEST_RELATIVE, evidence_root=evidence_root, label="start manifest"
    )
    start_binding = campaign.get("common_start_manifest")
    if not isinstance(start_binding, dict):
        raise AdmissionError("campaign start-manifest binding is missing")
    _safe_artifact(
        start_binding.get("path"), repo_root=repo_root, evidence_root=evidence_root,
        label="campaign start manifest", expected=start_manifest_path
    )
    if start_binding.get("sha256") != sha256_file(start_manifest_path):
        raise AdmissionError("campaign start-manifest SHA-256 mismatch")
    starts, expected_files = _validate_start_manifest(
        start_manifest_path, repo_root=repo_root, evidence_root=evidence_root
    )
    expected_files.add(campaign_path)
    report_paths = {
        SUMMARY_JSON_RELATIVE: _canonical_artifact(
            SUMMARY_JSON_RELATIVE, evidence_root=evidence_root, label="summary JSON"
        ),
        SUMMARY_CSV_RELATIVE: _canonical_artifact(
            SUMMARY_CSV_RELATIVE, evidence_root=evidence_root, label="summary CSV"
        ),
        IDENTITY_AUDIT_RELATIVE: _canonical_artifact(
            IDENTITY_AUDIT_RELATIVE, evidence_root=evidence_root, label="identity audit"
        ),
    }
    recorded_reports = campaign.get("reports")
    if not isinstance(recorded_reports, dict) or len(recorded_reports) != 3:
        raise AdmissionError("campaign must hash exactly the three canonical reports")
    for relative, path in report_paths.items():
        matching = []
        for raw, digest in recorded_reports.items():
            candidate = _safe_artifact(
                raw, repo_root=repo_root, evidence_root=evidence_root,
                label=f"report {raw}"
            )
            if candidate == path:
                matching.append(digest)
        if matching != [sha256_file(path)]:
            raise AdmissionError(f"campaign report binding differs: {relative}")
        expected_files.add(path)
    summary_rows = _validate_report_rows(report_paths[SUMMARY_JSON_RELATIVE])
    _validate_summary_csv(report_paths[SUMMARY_CSV_RELATIVE], summary_rows)
    planned_runs = campaign.get("planned_runs")
    if not isinstance(planned_runs, list) or len(planned_runs) != 60:
        raise AdmissionError("campaign planned-run inventory must contain 60 rows")
    planned_by_id = {
        str(row.get("case_id")): row for row in planned_runs if isinstance(row, dict)
    }
    if set(planned_by_id) != _expected_case_ids() or len(planned_by_id) != 60:
        raise AdmissionError("planned-run grid differs from the frozen design")
    run_records = campaign.get("run_records")
    if not isinstance(run_records, list) or len(run_records) != 60:
        raise AdmissionError("campaign must bind exactly 60 run records")
    records_by_case: dict[str, tuple[Path, str]] = {}
    for binding in run_records:
        if not isinstance(binding, dict):
            raise AdmissionError("run-record binding is malformed")
        raw_path = binding.get("path")
        path = _safe_artifact(
            raw_path, repo_root=repo_root, evidence_root=evidence_root,
            label="run-record binding"
        )
        case_id = path.parent.name
        expected = evidence_root / RAW_RELATIVE / case_id / "run_record.json"
        if path != expected.resolve() or case_id in records_by_case:
            raise AdmissionError("run-record binding path is noncanonical or duplicated")
        digest = str(binding.get("sha256", ""))
        if HEX64.fullmatch(digest) is None:
            raise AdmissionError(f"{case_id}: malformed run-record digest")
        records_by_case[case_id] = (path, digest)
    if set(records_by_case) != _expected_case_ids():
        raise AdmissionError("run-record case grid is incomplete")
    rows: list[dict[str, object]] = []
    for case_id in sorted(_expected_case_ids()):
        record_path, record_hash = records_by_case[case_id]
        validated, files = _validate_one_run(
            case_id=case_id,
            row=summary_rows[case_id],
            planned=planned_by_id[case_id],
            configuration=campaign["configuration"],
            record_path=record_path,
            record_hash=record_hash,
            source_commit=source_commit,
            starts=starts,
            repo_root=repo_root,
            evidence_root=evidence_root,
        )
        rows.append(validated)
        expected_files.update(files)
    adjudication = _adjudicate(rows)
    _verify_source_report_audit(
        report_paths[IDENTITY_AUDIT_RELATIVE], adjudication=adjudication
    )
    claims = campaign.get("claim_admission")
    if not isinstance(claims, dict):
        raise AdmissionError("campaign claim-admission record is missing")
    if claims.get("robustness_generalization_claim_admissible") is not False:
        raise AdmissionError("campaign promotes unsupported population robustness")
    if bool(claims.get("tested_instance_comparison_admissible")) != bool(
        adjudication["tested_instance_method_comparison_admissible"]
    ):
        raise AdmissionError("campaign comparison claim differs from independent adjudication")
    if bool(claims.get("timing_claim_admissible")) and not bool(
        adjudication["tested_instance_method_comparison_admissible"]
    ):
        raise AdmissionError("campaign promotes timing despite failed scientific gates")
    actual_hashes = _tree_hashes(evidence_root)
    expected_relatives = {
        path.resolve().relative_to(evidence_root.resolve()).as_posix()
        for path in expected_files
    }
    if set(actual_hashes) != expected_relatives:
        missing = sorted(expected_relatives - set(actual_hashes))
        extra = sorted(set(actual_hashes) - expected_relatives)
        raise AdmissionError(
            f"campaign artifact tree is not closed (missing={missing}, extra={extra})"
        )
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "admitted_bounded_local_outcomes",
        "experiment_id": "EXP-GLOB-001",
        "source_commit": source_commit,
        "campaign_manifest_sha256": sha256_file(campaign_path),
        "artifact_hashes": actual_hashes,
        "artifact_hashes_sha256": json_sha256(actual_hashes),
        "artifact_count": len(actual_hashes),
        "scientific_adjudication": adjudication,
        "timing_claim_admissible": False,
        "population_robustness_claim_admissible": False,
    }


def _tex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def render_table(audit: Mapping[str, object]) -> str:
    adjudication = audit.get("scientific_adjudication")
    if not isinstance(adjudication, Mapping):
        raise AdmissionError("audit scientific adjudication is missing")
    summaries = adjudication.get("method_summaries")
    if not isinstance(summaries, list) or len(summaries) != 4:
        raise AdmissionError("audit method summary is incomplete")
    labels = {
        "gl_l5_np2": r"Ginzburg--Landau, $L_5$, 2 ranks",
        "he_l2_np2_step1": r"Hyperelasticity, $L_2$, step 1, 2 ranks",
        "newton_armijo": r"Newton--Armijo",
        "reduced_trust_armijo": r"Reduced trust--Armijo",
    }
    lines = [
        r"\begin{table}[t]",
        r"  \caption{Observed terminal outcomes for the frozen local controlled-globalization design.}",
        r"  \label{tab:globalization-local-status}",
        r"  \centering",
        r"  \begin{tabularx}{\linewidth}{C{1.55}C{1.15}C{0.65}C{0.65}}",
        r"    \toprule",
        r"    Problem & Method & Completed & Failed \\",
        r"    \midrule",
    ]
    for row in summaries:
        benchmark = str(row.get("benchmark"))
        method = str(row.get("method"))
        if benchmark not in labels or method not in labels:
            raise AdmissionError("method summary contains an unknown label")
        lines.append(
            f"    {labels[benchmark]} & {labels[method]} & "
            f"{int(row.get('completed', -1))}/{int(row.get('total', -1))} & "
            f"{int(row.get('failed', -1))}/{int(row.get('total', -1))} \\\\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabularx}",
            r"  \begin{minipage}{0.96\linewidth}\small",
            (
                r"    Counts are outcomes for three prescribed starts and five process "
                r"repetitions per method. Timing and performance ordering are excluded. "
                r"The starts are deterministic sensitivity instances, so no robustness "
                r"generalization is made. A paired method comparison is admissible only "
                r"when both methods terminate at the same independently identified endpoint."
            ),
            r"  \end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)

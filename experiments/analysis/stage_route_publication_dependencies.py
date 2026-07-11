#!/usr/bin/env python3
"""Stage complete EXP-ROUTE evidence for the managed publication finalizer.

This program performs local file validation and copying only.  It does not
submit jobs, contact a scheduler, or access a remote host.  The ``plan`` mode
validates completed workstation and copied-back Karolina evidence, freezes a
recursive SHA-256 inventory of both source trees, and writes a dependency-
preparation plan consumable by ``finalize_revision_publication_campaign.py``.
The four staging modes are invoked by that managed executor to produce its
normal fingerprinted receipts at the exact clean experiment commit.  The
endpoint-bound final STOP adjudication is independently copied and attested.

The managed executor intentionally creates parents for every declared output
before invoking a producer.  Whole-tree staging therefore accepts an existing
destination only when it is a directory tree containing no files or links;
that empty skeleton is replaced atomically by the validated copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import uuid
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analysis import (  # noqa: E402
    analyze_plasticity3d_route_cost_model as route_analysis,
)
from experiments.analysis import finalize_revision_publication_campaign as finalizer  # noqa: E402
from experiments.runners.paper_revision_karolina.tier_b_stopping import (  # noqa: E402
    POLICY_PATH as TIER_B_STOPPING_POLICY_PATH,
    validate_stop_adjudication,
)
from src.core.benchmark.run_record import atomic_write_json, strict_json_dumps  # noqa: E402


SCRIPT_PATH = Path("experiments/analysis/stage_route_publication_dependencies.py")
DEFAULT_CONTRACT = Path("paper/protocols/EXP-ROUTE-001-analysis-contract.json")
WORKSTATION_TARGET = Path("EXP-ROUTE-001/source_archives/workstation")
KAROLINA_TARGET = Path("EXP-ROUTE-001/source_archives/karolina")
CANONICAL_ENDPOINT = (
    KAROLINA_TARGET / "reviewed_inputs/tier_b_endpoint_analysis.json"
)
CANONICAL_STOPPING_ADJUDICATION = (
    KAROLINA_TARGET / "reviewed_inputs/stopping_adjudication.json"
)
DEPENDENCY_CAMPAIGN_ID = "paper_revision_route_dependencies_v1"
HEX40 = frozenset("0123456789abcdef")
HASH_BOUND_VALIDATOR_FILES = (
    DEFAULT_CONTRACT,
    Path("paper/protocols/EXP-ROUTE-001-workstation-plan.json"),
    Path("experiments/runners/paper_revision_karolina/campaign_matrix.csv"),
    Path("experiments/analysis/analyze_plasticity3d_route_cost_model.py"),
    Path("experiments/analysis/analyze_plasticity3d_route_endpoints.py"),
    Path("experiments/analysis/aggregate_route_tranche_manifests.py"),
    Path("experiments/analysis/aggregate_route_tier_b_manifests.py"),
    TIER_B_STOPPING_POLICY_PATH.relative_to(REPO_ROOT),
    Path("experiments/runners/paper_revision_karolina/tier_b_stopping.py"),
    Path("experiments/runners/prepare_exp_stop_001_karolina.py"),
    Path("src/core/benchmark/run_record.py"),
)


class RouteDependencyError(RuntimeError):
    """Raised when route evidence cannot enter publication staging."""


def _read_object(path: Path) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token!r}")

    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise RouteDependencyError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        strict_json_dumps(value, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _commit(value: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 40 or any(
        character not in HEX40 for character in normalized
    ):
        raise RouteDependencyError("expected commit must be a full 40-digit SHA-1")
    return normalized


def _relative_path(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RouteDependencyError(f"{label} must be a canonical relative path")
    return path


def _require_regular_tree(root: Path, *, label: str) -> Path:
    expanded = root.expanduser()
    absolute = Path(os.path.abspath(expanded))
    if absolute.is_symlink() or not absolute.is_dir():
        raise RouteDependencyError(f"{label} must be a real directory: {absolute}")
    root = absolute.resolve(strict=True)
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RouteDependencyError(f"{label} contains a symbolic link: {path}")
        mode = path.stat().st_mode
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise RouteDependencyError(f"{label} contains a non-regular entry: {path}")
    return root


def _tree_inventory(root: Path, *, label: str) -> dict[str, str]:
    root = _require_regular_tree(root, label=label)
    inventory = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if not inventory:
        raise RouteDependencyError(f"{label} contains no regular files")
    return inventory


def _inventory_fingerprint(inventory: Mapping[str, str]) -> str:
    return _json_sha256(dict(sorted(inventory.items())))


def _require_inventory(
    root: Path,
    *,
    expected_sha256: str,
    label: str,
) -> dict[str, str]:
    inventory = _tree_inventory(root, label=label)
    actual = _inventory_fingerprint(inventory)
    if actual != str(expected_sha256).lower():
        raise RouteDependencyError(
            f"{label} recursive inventory changed after dependency-plan review: "
            f"expected {expected_sha256}, got {actual}"
        )
    return inventory


def _validate_workstation_hash_closure(root: Path) -> dict[str, str]:
    manifest_path = root / "workstation_manifest.json"
    manifest = _read_object(manifest_path)
    closure = manifest.get("output_hash_closure")
    if not isinstance(closure, Mapping):
        raise RouteDependencyError(
            "workstation manifest lacks its recursive output closure"
        )
    actual = _tree_inventory(root, label="workstation archive")
    actual_without_manifest = dict(actual)
    actual_without_manifest.pop("workstation_manifest.json", None)
    declared = closure.get("files")
    if (
        closure.get("algorithm") != "sha256"
        or closure.get("scope")
        != "all_regular_files_below_output_root_except_manifest"
        or closure.get("excluded_paths") != ["workstation_manifest.json"]
        or not isinstance(declared, Mapping)
        or dict(declared) != actual_without_manifest
        or int(closure.get("file_count", -1)) != len(actual_without_manifest)
        or closure.get("files_map_sha256")
        != _inventory_fingerprint(actual_without_manifest)
    ):
        raise RouteDependencyError(
            "workstation recursive output closure is stale or incomplete"
        )
    return actual


def _validate_workstation_frozen_inputs(
    root: Path,
    *,
    expected_commit: str,
) -> None:
    manifest = _read_object(root / "workstation_manifest.json")
    exact_counts = {"completed": 36, "censored": 0, "failed": 0}
    terminal = manifest.get("terminal_source")
    frozen = manifest.get("terminal_frozen_hash_verification")
    case_statuses = manifest.get("case_statuses")
    if (
        manifest.get("route_terminal_counts") != exact_counts
        or int(manifest.get("route_processes_launched", -1)) != 36
        or not isinstance(case_statuses, Mapping)
        or len(case_statuses) != 12
        or set(case_statuses.values()) != {"completed"}
        or not isinstance(terminal, Mapping)
        or str(terminal.get("commit", "")).lower() != expected_commit
        or terminal.get("dirty") is not False
        or not isinstance(frozen, Mapping)
        or frozen.get("passed") is not True
        or frozen.get("errors") not in ({}, None)
    ):
        raise RouteDependencyError(
            "workstation manifest does not prove 36 completed route processes "
            "and unchanged frozen inputs"
        )
    for field, fingerprint_field in (
        ("code_hashes", "code_hashes_sha256"),
        ("configuration_hashes", "configuration_hashes_sha256"),
        ("input_hashes", "input_hashes_sha256"),
    ):
        inventory = manifest.get(field)
        if (
            not isinstance(inventory, Mapping)
            or not inventory
            or manifest.get(fingerprint_field) != _inventory_fingerprint(inventory)
        ):
            raise RouteDependencyError(
                f"workstation {field} inventory is missing or stale"
            )
        for raw_path, expected_hash in inventory.items():
            path = Path(str(raw_path))
            if path.is_absolute():
                try:
                    path.resolve().relative_to(REPO_ROOT.resolve())
                except ValueError as exc:
                    raise RouteDependencyError(
                        f"workstation {field} path escapes the repository: {raw_path}"
                    ) from exc
            else:
                path = REPO_ROOT / path
            if not path.is_file() or _sha256(path) != expected_hash:
                raise RouteDependencyError(
                    f"workstation {field} file is missing or changed: {raw_path}"
                )


def _load_contract(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    canonical = (REPO_ROOT / DEFAULT_CONTRACT).resolve()
    if resolved != canonical:
        raise RouteDependencyError(
            f"route dependency staging requires the canonical contract {canonical}"
        )
    contract = _read_object(resolved)
    if contract.get("experiment_id") != "EXP-ROUTE-001":
        raise RouteDependencyError("route analysis contract has the wrong experiment")
    return resolved, contract


def _stopping_identity(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Return the relocation-independent identity of one STOP adjudication."""

    return {str(key): value for key, value in binding.items() if key != "path"}


def _regular_file_within(
    root: Path,
    raw_path: str | Path,
    *,
    label: str,
) -> tuple[Path, Path]:
    root = _require_regular_tree(root, label=f"{label} archive")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise RouteDependencyError(f"{label} is not a regular file: {candidate}")
    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise RouteDependencyError(f"{label} escapes the Karolina archive") from exc
    return resolved, relative


def _declared_stopping_adjudication(
    endpoint_path: Path,
    *,
    karolina_root: Path,
) -> tuple[Path, Path]:
    """Locate the real STOP JSON declared by the endpoint analysis.

    The endpoint stores the complete validated STOP binding.  The path is the
    only relocation-sensitive field, so dependency planning resolves it while
    the copied-back archive is still in its reviewed location and then freezes
    the archive-relative path and exact file hash.
    """

    endpoint = _read_object(endpoint_path)
    binding = endpoint.get("stopping_adjudication")
    if not isinstance(binding, Mapping):
        raise RouteDependencyError(
            "Tier-B endpoint lacks its validated STOP adjudication binding"
        )
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RouteDependencyError(
            "Tier-B endpoint STOP adjudication path is missing"
        )
    return _regular_file_within(
        karolina_root,
        raw_path,
        label="Tier-B STOP adjudication",
    )


def _require_endpoint_gate(
    endpoint_path: Path,
    stopping_adjudication_path: Path,
    *,
    sources: list[tuple[str, Path]],
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the shared endpoint and STOP validators and cross-check identities."""

    endpoint = route_analysis._endpoint_analysis_gate(
        endpoint_path,
        stopping_adjudication_path=stopping_adjudication_path,
        sources=sources,
        contract=contract,
    )
    if endpoint.get("publication_admissible") is not True:
        raise RouteDependencyError(
            "Tier-B endpoint/STOP gate failed: "
            f"{endpoint.get('reason', 'unknown')}"
        )
    validated = validate_stop_adjudication(stopping_adjudication_path)
    gate_binding = endpoint.get("stopping_adjudication")
    if (
        not isinstance(gate_binding, Mapping)
        or _stopping_identity(gate_binding) != _stopping_identity(validated)
        or endpoint.get("stopping_binding_matches_manifest") is not True
    ):
        raise RouteDependencyError(
            "Tier-B endpoint and actual STOP adjudication identities differ"
        )
    return endpoint, dict(validated)


def _require_source_gate(
    hardware_id: str,
    root: Path,
    contract: dict[str, Any],
    *,
    expected_commit: str,
) -> dict[str, object]:
    gate = route_analysis._source_provenance_gate(hardware_id, root, contract)
    if gate.get("eligible") is not True:
        raise RouteDependencyError(
            f"{hardware_id} provenance gate failed: {gate.get('reason', 'unknown')}"
        )
    if str(gate.get("source_commit", "")).lower() != expected_commit:
        raise RouteDependencyError(
            f"{hardware_id} source commit differs from the dependency plan"
        )
    return gate


def validate_workstation_archive(
    root: Path,
    *,
    contract: dict[str, Any],
    expected_commit: str,
) -> dict[str, Any]:
    root = _require_regular_tree(root, label="workstation archive")
    inventory = _validate_workstation_hash_closure(root)
    _validate_workstation_frozen_inputs(root, expected_commit=expected_commit)
    gate = _require_source_gate(
        "workstation_local",
        root,
        contract,
        expected_commit=expected_commit,
    )
    observed, censors, invalid = route_analysis._scan_source(
        "workstation_local", root, contract=contract, source_provenance=gate
    )
    rows = route_analysis.build_empirical_map(
        contract=contract,
        hardware_ids=["workstation_local"],
        observed=observed,
        runtime_censors=censors,
    )
    if invalid or not rows or any(
        row.get("status") != "admitted"
        or row.get("publication_model_eligible") is not True
        for row in rows
    ):
        raise RouteDependencyError(
            "workstation evidence is not a complete publication-eligible empirical map"
        )
    return {
        "hardware_id": "workstation_local",
        "source_commit": expected_commit,
        "file_count": len(inventory),
        "inventory_sha256": _inventory_fingerprint(inventory),
        "admitted_rows": len(rows),
    }


def validate_complete_route_evidence(
    *,
    workstation_root: Path,
    karolina_root: Path,
    endpoint_relative: Path,
    stopping_adjudication_path: Path | None = None,
    contract: dict[str, Any],
    expected_commit: str,
) -> dict[str, Any]:
    workstation_root = _require_regular_tree(
        workstation_root, label="workstation archive"
    )
    karolina_root = _require_regular_tree(karolina_root, label="Karolina archive")
    workstation_summary = validate_workstation_archive(
        workstation_root,
        contract=contract,
        expected_commit=expected_commit,
    )
    sources = [
        ("workstation_local", workstation_root),
        ("karolina_cpu", karolina_root),
    ]
    gates = {
        hardware: _require_source_gate(
            hardware,
            root,
            contract,
            expected_commit=expected_commit,
        )
        for hardware, root in sources
    }
    observed: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    censors: dict[tuple[str, str, str, int, str], str] = {}
    invalid: list[dict[str, str]] = []
    for hardware, root in sources:
        new_observed, new_censors, new_invalid = route_analysis._scan_source(
            hardware,
            root,
            contract=contract,
            source_provenance=gates[hardware],
        )
        if set(observed).intersection(new_observed):
            raise RouteDependencyError(
                "route source archives contain overlapping evidence slots"
            )
        observed.update(new_observed)
        censors.update(new_censors)
        invalid.extend(new_invalid)
    rows = route_analysis.build_empirical_map(
        contract=contract,
        hardware_ids=[hardware for hardware, _root in sources],
        observed=observed,
        runtime_censors=censors,
    )
    factorized = route_analysis._factorized_microbenchmark_gate(sources, contract)
    endpoint_path = (karolina_root / endpoint_relative).resolve()
    try:
        endpoint_path.relative_to(karolina_root.resolve())
    except ValueError as exc:
        raise RouteDependencyError(
            "Tier-B endpoint path escapes the Karolina archive"
        ) from exc
    if endpoint_path.is_symlink() or not endpoint_path.is_file():
        raise RouteDependencyError(
            "Tier-B endpoint analysis is not a regular file in the Karolina archive"
        )
    if stopping_adjudication_path is None:
        stopping_path, stopping_relative = _declared_stopping_adjudication(
            endpoint_path,
            karolina_root=karolina_root,
        )
    else:
        stopping_path, stopping_relative = _regular_file_within(
            karolina_root,
            stopping_adjudication_path,
            label="Tier-B STOP adjudication",
        )
    endpoint, stopping = _require_endpoint_gate(
        endpoint_path,
        stopping_path,
        sources=sources,
        contract=contract,
    )
    fitted = route_analysis.fit_cost_model(
        rows,
        contract,
        factorized_gate=factorized,
        endpoint_gate=endpoint,
    )
    terminal = (
        route_analysis.PREDICTIVE_SELECTOR_TERMINAL
        if fitted.get("selector_claim_admissible") is True
        else route_analysis.FINITE_EMPIRICAL_MAP_TERMINAL
    )
    model = route_analysis._publication_safe_cost_model(fitted)
    admissible = route_analysis._publication_evidence_is_admissible(
        clean_committed_analysis=True,
        terminal_decision=terminal,
        empirical_rows=rows,
        cost_model=model,
        endpoint_gate=endpoint,
        factorized_gate=factorized,
        invalid_records=invalid,
        contract=contract,
    )
    if not admissible:
        statuses: dict[str, int] = {}
        for row in rows:
            status = str(row.get("status", "unknown"))
            statuses[status] = statuses.get(status, 0) + 1
        raise RouteDependencyError(
            "combined workstation/Karolina route evidence is not publication-admissible: "
            f"statuses={dict(sorted(statuses.items()))}, invalid={len(invalid)}, "
            f"model_status={model.get('status')}, "
            f"endpoint={endpoint.get('reason')}, "
            f"factor_failures={len(factorized.get('failures') or [])}"
        )
    karolina_inventory = _tree_inventory(
        karolina_root, label="Karolina archive"
    )
    return {
        "experiment_id": "EXP-ROUTE-001",
        "source_commit": expected_commit,
        "terminal_decision": terminal,
        "publication_admissible": True,
        "empirical_rows": len(rows),
        "training_rows": int(model["training_rows"]),
        "holdout_rows": int(model["holdout_rows"]),
        "workstation": workstation_summary,
        "karolina": {
            "file_count": len(karolina_inventory),
            "inventory_sha256": _inventory_fingerprint(karolina_inventory),
        },
        "endpoint": {
            "path": endpoint_relative.as_posix(),
            "sha256": _sha256(endpoint_path),
            "terminal_decision": endpoint.get("terminal_decision"),
            "schema_version": endpoint.get("schema_version"),
            "stopping_binding_matches_manifest": endpoint.get(
                "stopping_binding_matches_manifest"
            ),
            "stopping_adjudication": {
                **_stopping_identity(stopping),
                "path": stopping_relative.as_posix(),
            },
        },
    }


def _empty_directory_skeleton(path: Path) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    for child in path.rglob("*"):
        if child.is_symlink() or not child.is_dir():
            return False
    return True


def _require_destination_suffix(destination: Path, suffix: Path) -> None:
    if tuple(destination.parts[-len(suffix.parts) :]) != suffix.parts:
        raise RouteDependencyError(
            f"destination must end in the canonical staging path {suffix.as_posix()}"
        )


def _stage_tree(
    source: Path,
    destination: Path,
    *,
    expected_inventory_sha256: str,
    label: str,
) -> dict[str, str]:
    source = _require_regular_tree(source, label=label)
    destination = destination.expanduser().resolve(strict=False)
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise RouteDependencyError(f"{label} source and destination trees overlap")
    expected_inventory = _require_inventory(
        source,
        expected_sha256=expected_inventory_sha256,
        label=label,
    )
    if destination.exists() or destination.is_symlink():
        if not _empty_directory_skeleton(destination):
            raise RouteDependencyError(
                f"refusing to replace nonempty or non-directory destination: {destination}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.stage-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, temporary, copy_function=shutil.copy2)
        copied = _tree_inventory(temporary, label=f"staged {label}")
        if copied != expected_inventory:
            raise RouteDependencyError(f"{label} changed while it was being copied")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        final_inventory = _tree_inventory(destination, label=f"staged {label}")
        if final_inventory != expected_inventory:
            raise RouteDependencyError(f"staged {label} failed post-copy hash verification")
        return final_inventory
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def stage_workstation(
    *,
    source: Path,
    destination: Path,
    contract_path: Path,
    expected_commit: str,
    expected_inventory_sha256: str,
) -> dict[str, Any]:
    expected_commit = _commit(expected_commit)
    _contract_path, contract = _load_contract(contract_path)
    _require_destination_suffix(destination, WORKSTATION_TARGET)
    validate_workstation_archive(
        source,
        contract=contract,
        expected_commit=expected_commit,
    )
    inventory = _stage_tree(
        source,
        destination,
        expected_inventory_sha256=expected_inventory_sha256,
        label="workstation archive",
    )
    validate_workstation_archive(
        destination,
        contract=contract,
        expected_commit=expected_commit,
    )
    return {
        "status": "staged",
        "target": WORKSTATION_TARGET.as_posix(),
        "file_count": len(inventory),
        "inventory_sha256": _inventory_fingerprint(inventory),
    }


def stage_karolina(
    *,
    source: Path,
    destination: Path,
    workstation_root: Path,
    endpoint_relative: Path,
    stopping_relative: Path,
    contract_path: Path,
    expected_commit: str,
    expected_inventory_sha256: str,
) -> dict[str, Any]:
    expected_commit = _commit(expected_commit)
    endpoint_relative = _relative_path(
        endpoint_relative, label="Tier-B endpoint relative path"
    )
    stopping_relative = _relative_path(
        stopping_relative, label="Tier-B STOP adjudication relative path"
    )
    _contract_path, contract = _load_contract(contract_path)
    _require_destination_suffix(destination, KAROLINA_TARGET)
    validate_complete_route_evidence(
        workstation_root=workstation_root,
        karolina_root=source,
        endpoint_relative=endpoint_relative,
        stopping_adjudication_path=source / stopping_relative,
        contract=contract,
        expected_commit=expected_commit,
    )
    inventory = _stage_tree(
        source,
        destination,
        expected_inventory_sha256=expected_inventory_sha256,
        label="Karolina archive",
    )
    validate_complete_route_evidence(
        workstation_root=workstation_root,
        karolina_root=destination,
        endpoint_relative=endpoint_relative,
        stopping_adjudication_path=destination / stopping_relative,
        contract=contract,
        expected_commit=expected_commit,
    )
    return {
        "status": "staged",
        "target": KAROLINA_TARGET.as_posix(),
        "file_count": len(inventory),
        "inventory_sha256": _inventory_fingerprint(inventory),
    }


def stage_stopping_adjudication(
    *,
    workstation_root: Path,
    karolina_root: Path,
    endpoint_relative: Path,
    stopping_relative: Path,
    destination: Path,
    contract_path: Path,
    expected_commit: str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Freeze the validated STOP JSON at its canonical staging path."""

    expected_commit = _commit(expected_commit)
    endpoint_relative = _relative_path(
        endpoint_relative, label="Tier-B endpoint relative path"
    )
    stopping_relative = _relative_path(
        stopping_relative, label="Tier-B STOP adjudication relative path"
    )
    _contract_path, contract = _load_contract(contract_path)
    _require_destination_suffix(destination, CANONICAL_STOPPING_ADJUDICATION)
    validation = validate_complete_route_evidence(
        workstation_root=workstation_root,
        karolina_root=karolina_root,
        endpoint_relative=endpoint_relative,
        stopping_adjudication_path=karolina_root / stopping_relative,
        contract=contract,
        expected_commit=expected_commit,
    )
    source, _source_relative = _regular_file_within(
        karolina_root,
        stopping_relative,
        label="Tier-B STOP adjudication",
    )
    if _sha256(source) != expected_sha256:
        raise RouteDependencyError(
            "Tier-B STOP adjudication changed after dependency-plan review"
        )
    if source == destination.expanduser().resolve(strict=False):
        raise RouteDependencyError(
            "copied-back STOP adjudication already occupies its reserved canonical path"
        )
    if destination.exists() or destination.is_symlink():
        raise RouteDependencyError(
            f"refusing to overwrite staged STOP adjudication: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.stage-{uuid.uuid4().hex}")
    try:
        shutil.copy2(source, temporary)
        if _sha256(temporary) != expected_sha256:
            raise RouteDependencyError(
                "Tier-B STOP adjudication changed while it was being copied"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    if _sha256(destination) != expected_sha256:
        raise RouteDependencyError(
            "staged STOP adjudication failed post-copy hash verification"
        )
    endpoint_path = (karolina_root / endpoint_relative).resolve()
    endpoint_gate, stopping = _require_endpoint_gate(
        endpoint_path,
        destination,
        sources=[
            ("workstation_local", workstation_root),
            ("karolina_cpu", karolina_root),
        ],
        contract=contract,
    )
    if (
        validation.get("endpoint", {}).get("stopping_binding_matches_manifest")
        is not True
        or endpoint_gate.get("stopping_binding_matches_manifest") is not True
    ):
        raise RouteDependencyError(
            "canonical staged STOP adjudication failed endpoint identity validation"
        )
    return {
        "status": "staged",
        "target": CANONICAL_STOPPING_ADJUDICATION.as_posix(),
        "sha256": expected_sha256,
        "schema_id": stopping["schema_id"],
        "schema_version": stopping["schema_version"],
    }


def stage_endpoint(
    *,
    workstation_root: Path,
    karolina_root: Path,
    endpoint_relative: Path,
    stopping_adjudication: Path,
    destination: Path,
    contract_path: Path,
    expected_commit: str,
    expected_sha256: str,
) -> dict[str, Any]:
    expected_commit = _commit(expected_commit)
    endpoint_relative = _relative_path(
        endpoint_relative, label="Tier-B endpoint relative path"
    )
    _contract_path, contract = _load_contract(contract_path)
    _require_destination_suffix(destination, CANONICAL_ENDPOINT)
    validate_complete_route_evidence(
        workstation_root=workstation_root,
        karolina_root=karolina_root,
        endpoint_relative=endpoint_relative,
        stopping_adjudication_path=stopping_adjudication,
        contract=contract,
        expected_commit=expected_commit,
    )
    source = (karolina_root / endpoint_relative).resolve()
    if not source.is_file() or source.is_symlink() or _sha256(source) != expected_sha256:
        raise RouteDependencyError(
            "Tier-B endpoint changed after dependency-plan review"
        )
    if destination.exists() or destination.is_symlink():
        raise RouteDependencyError(f"refusing to overwrite staged endpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.stage-{uuid.uuid4().hex}")
    try:
        shutil.copy2(source, temporary)
        if _sha256(temporary) != expected_sha256:
            raise RouteDependencyError("Tier-B endpoint changed while it was being copied")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    if _sha256(destination) != expected_sha256:
        raise RouteDependencyError(
            "staged Tier-B endpoint failed post-copy hash verification"
        )
    endpoint_gate, _stopping = _require_endpoint_gate(
        destination,
        stopping_adjudication,
        sources=[
            ("workstation_local", workstation_root),
            ("karolina_cpu", karolina_root),
        ],
        contract=contract,
    )
    if endpoint_gate.get("publication_admissible") is not True:
        raise RouteDependencyError(
            "canonical staged Tier-B endpoint failed semantic validation"
        )
    return {
        "status": "staged",
        "target": CANONICAL_ENDPOINT.as_posix(),
        "sha256": expected_sha256,
    }


def _staging_input(path: Path, *, attestation: str) -> dict[str, Any]:
    return {
        "scope": "staging",
        "path": path.as_posix(),
        "attestation": {"path": attestation},
    }


def _preparation_command(
    command_id: str,
    argv: Sequence[str],
    *,
    expected_artifacts: Sequence[Path],
    inputs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "id": command_id,
        "source_keys": [],
        "role": "preparation",
        "producer": SCRIPT_PATH.as_posix(),
        "argv": list(argv),
        "environment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "configuration_files": [
            path.as_posix() for path in HASH_BOUND_VALIDATOR_FILES
        ],
        "input_files": [dict(value) for value in inputs],
        "expected_artifacts": [path.as_posix() for path in expected_artifacts],
    }


def build_dependency_plan(
    *,
    expected_commit: str,
    workstation_source: Path,
    karolina_source: Path,
    endpoint_relative: Path,
    stopping_relative: Path,
    contract_path: Path = REPO_ROOT / DEFAULT_CONTRACT,
) -> dict[str, Any]:
    expected_commit = _commit(expected_commit)
    workstation_source = _require_regular_tree(
        workstation_source, label="workstation archive"
    )
    karolina_source = _require_regular_tree(
        karolina_source, label="Karolina archive"
    )
    endpoint_relative = _relative_path(
        endpoint_relative, label="Tier-B endpoint relative path"
    )
    stopping_relative = _relative_path(
        stopping_relative, label="Tier-B STOP adjudication relative path"
    )
    _resolved_contract, contract = _load_contract(contract_path)
    if endpoint_relative == Path("reviewed_inputs/tier_b_endpoint_analysis.json"):
        raise RouteDependencyError(
            "the copied-back Karolina endpoint may not occupy the reserved canonical staging path"
        )
    if stopping_relative == Path("reviewed_inputs/stopping_adjudication.json"):
        raise RouteDependencyError(
            "the copied-back STOP adjudication may not occupy the reserved canonical staging path"
        )
    stopping_source, observed_stopping_relative = _regular_file_within(
        karolina_source,
        stopping_relative,
        label="Tier-B STOP adjudication",
    )
    if observed_stopping_relative != stopping_relative:
        raise RouteDependencyError(
            "Tier-B STOP adjudication path is not canonical within the Karolina archive"
        )
    validation = validate_complete_route_evidence(
        workstation_root=workstation_source,
        karolina_root=karolina_source,
        endpoint_relative=endpoint_relative,
        stopping_adjudication_path=stopping_source,
        contract=contract,
        expected_commit=expected_commit,
    )
    endpoint_validation = validation.get("endpoint")
    if not isinstance(endpoint_validation, Mapping):
        raise RouteDependencyError(
            "complete route validation lacks its Tier-B endpoint result"
        )
    stopping_binding = endpoint_validation.get("stopping_adjudication")
    if not isinstance(stopping_binding, Mapping):
        raise RouteDependencyError(
            "complete route validation lacks its STOP adjudication binding"
        )
    stopping_validated = validate_stop_adjudication(stopping_source)
    if _stopping_identity(stopping_validated) != _stopping_identity(stopping_binding):
        raise RouteDependencyError(
            "dependency validation STOP identity changed before plan construction"
        )
    workstation_inventory = _tree_inventory(
        workstation_source, label="workstation archive"
    )
    karolina_inventory = _tree_inventory(karolina_source, label="Karolina archive")
    workstation_fingerprint = _inventory_fingerprint(workstation_inventory)
    karolina_fingerprint = _inventory_fingerprint(karolina_inventory)
    endpoint_sha256 = _sha256(karolina_source / endpoint_relative)
    stopping_sha256 = _sha256(stopping_source)
    workstation_outputs = [
        WORKSTATION_TARGET / relative for relative in sorted(workstation_inventory)
    ]
    karolina_outputs = [
        KAROLINA_TARGET / relative for relative in sorted(karolina_inventory)
    ]
    workstation_receipt = (
        f"{finalizer.RECEIPT_DIRECTORY}/prepare_workstation_archive.json"
    )
    karolina_receipt = (
        f"{finalizer.RECEIPT_DIRECTORY}/prepare_route_campaign_master.json"
    )
    stopping_receipt = (
        f"{finalizer.RECEIPT_DIRECTORY}/prepare_route_stopping_adjudication.json"
    )
    commands = [
        _preparation_command(
            "prepare_workstation_archive",
            [
                "{python}",
                SCRIPT_PATH.as_posix(),
                "stage-workstation",
                "--source",
                str(workstation_source),
                "--destination",
                f"{{staging_root}}/{WORKSTATION_TARGET.as_posix()}",
                "--contract",
                f"{{repo_root}}/{DEFAULT_CONTRACT.as_posix()}",
                "--expected-commit",
                expected_commit,
                "--expected-inventory-sha256",
                workstation_fingerprint,
            ],
            expected_artifacts=workstation_outputs,
        ),
        _preparation_command(
            "prepare_route_campaign_master",
            [
                "{python}",
                SCRIPT_PATH.as_posix(),
                "stage-karolina",
                "--source",
                str(karolina_source),
                "--destination",
                f"{{staging_root}}/{KAROLINA_TARGET.as_posix()}",
                "--workstation-root",
                f"{{staging_root}}/{WORKSTATION_TARGET.as_posix()}",
                "--endpoint-relative",
                endpoint_relative.as_posix(),
                "--stopping-relative",
                stopping_relative.as_posix(),
                "--contract",
                f"{{repo_root}}/{DEFAULT_CONTRACT.as_posix()}",
                "--expected-commit",
                expected_commit,
                "--expected-inventory-sha256",
                karolina_fingerprint,
            ],
            expected_artifacts=karolina_outputs,
            inputs=[
                _staging_input(path, attestation=workstation_receipt)
                for path in workstation_outputs
            ],
        ),
        _preparation_command(
            "prepare_route_stopping_adjudication",
            [
                "{python}",
                SCRIPT_PATH.as_posix(),
                "stage-stopping-adjudication",
                "--workstation-root",
                f"{{staging_root}}/{WORKSTATION_TARGET.as_posix()}",
                "--karolina-root",
                f"{{staging_root}}/{KAROLINA_TARGET.as_posix()}",
                "--endpoint-relative",
                endpoint_relative.as_posix(),
                "--stopping-relative",
                stopping_relative.as_posix(),
                "--destination",
                f"{{staging_root}}/{CANONICAL_STOPPING_ADJUDICATION.as_posix()}",
                "--contract",
                f"{{repo_root}}/{DEFAULT_CONTRACT.as_posix()}",
                "--expected-commit",
                expected_commit,
                "--expected-sha256",
                stopping_sha256,
            ],
            expected_artifacts=[CANONICAL_STOPPING_ADJUDICATION],
            inputs=[
                _staging_input(
                    KAROLINA_TARGET / stopping_relative,
                    attestation=karolina_receipt,
                )
            ],
        ),
        _preparation_command(
            "prepare_tier_b_endpoint_analysis",
            [
                "{python}",
                SCRIPT_PATH.as_posix(),
                "stage-endpoint",
                "--workstation-root",
                f"{{staging_root}}/{WORKSTATION_TARGET.as_posix()}",
                "--karolina-root",
                f"{{staging_root}}/{KAROLINA_TARGET.as_posix()}",
                "--endpoint-relative",
                endpoint_relative.as_posix(),
                "--stopping-adjudication",
                f"{{staging_root}}/{CANONICAL_STOPPING_ADJUDICATION.as_posix()}",
                "--destination",
                f"{{staging_root}}/{CANONICAL_ENDPOINT.as_posix()}",
                "--contract",
                f"{{repo_root}}/{DEFAULT_CONTRACT.as_posix()}",
                "--expected-commit",
                expected_commit,
                "--expected-sha256",
                endpoint_sha256,
            ],
            expected_artifacts=[CANONICAL_ENDPOINT],
            inputs=[
                _staging_input(
                    KAROLINA_TARGET / endpoint_relative,
                    attestation=karolina_receipt,
                ),
                _staging_input(
                    CANONICAL_STOPPING_ADJUDICATION,
                    attestation=stopping_receipt,
                ),
            ],
        ),
    ]
    plan = {
        "schema_id": finalizer.PLAN_SCHEMA_ID,
        "schema_version": finalizer.PLAN_SCHEMA_VERSION,
        "campaign_id": DEPENDENCY_CAMPAIGN_ID,
        "plan_kind": "dependency_preparation",
        "experiment_commit": expected_commit,
        "commands": commands,
        "execution_order": [command["id"] for command in commands],
        "source_archives": {
            "workstation": {
                "source": str(workstation_source),
                "target": WORKSTATION_TARGET.as_posix(),
                "file_count": len(workstation_inventory),
                "inventory_sha256": workstation_fingerprint,
                "files": workstation_inventory,
            },
            "karolina": {
                "source": str(karolina_source),
                "target": KAROLINA_TARGET.as_posix(),
                "file_count": len(karolina_inventory),
                "inventory_sha256": karolina_fingerprint,
                "files": karolina_inventory,
            },
        },
        "tier_b_endpoint": {
            "source_relative": endpoint_relative.as_posix(),
            "source_sha256": endpoint_sha256,
            "canonical_target": CANONICAL_ENDPOINT.as_posix(),
        },
        "tier_b_stopping_adjudication": {
            "source_relative": stopping_relative.as_posix(),
            "source_sha256": stopping_sha256,
            "canonical_target": CANONICAL_STOPPING_ADJUDICATION.as_posix(),
            "identity": _stopping_identity(stopping_validated),
        },
        "semantic_validation": validation,
        "safety": {
            "scheduler_commands": False,
            "remote_access": False,
            "copy_policy": "validated_local_regular_files_only",
            "overwrite_policy": "empty_managed_executor_skeleton_only",
        },
    }
    finalizer._plan_command_map(plan)
    return plan


def write_dependency_plan(
    *,
    output: Path,
    expected_commit: str,
    workstation_source: Path,
    karolina_source: Path,
    endpoint_relative: Path,
    stopping_relative: Path,
    contract_path: Path,
) -> Path:
    output = output.expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise RouteDependencyError(f"refusing to overwrite dependency plan: {output}")
    normalized_commit = _commit(expected_commit)
    current_commit = finalizer._require_clean_head(REPO_ROOT)
    if current_commit != normalized_commit:
        raise RouteDependencyError(
            "dependency plan must be generated at its exact clean experiment commit"
        )
    plan = build_dependency_plan(
        expected_commit=normalized_commit,
        workstation_source=workstation_source,
        karolina_source=karolina_source,
        endpoint_relative=endpoint_relative,
        stopping_relative=stopping_relative,
        contract_path=contract_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, plan)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    plan = subparsers.add_parser(
        "plan",
        help="validate local/copy-back evidence and freeze a managed dependency plan",
    )
    plan.add_argument("--expected-commit", required=True)
    plan.add_argument("--workstation-source", type=Path, required=True)
    plan.add_argument("--karolina-source", type=Path, required=True)
    plan.add_argument(
        "--endpoint-relative",
        type=Path,
        required=True,
        help="Tier-B endpoint analysis path relative to --karolina-source",
    )
    plan.add_argument(
        "--stopping-relative",
        type=Path,
        required=True,
        help="final STOP adjudication path relative to --karolina-source",
    )
    plan.add_argument("--contract", type=Path, default=REPO_ROOT / DEFAULT_CONTRACT)
    plan.add_argument("--output", type=Path, required=True)

    workstation = subparsers.add_parser(
        "stage-workstation",
        help="copy one recursively frozen workstation archive into managed staging",
    )
    workstation.add_argument("--source", type=Path, required=True)
    workstation.add_argument("--destination", type=Path, required=True)
    workstation.add_argument("--contract", type=Path, required=True)
    workstation.add_argument("--expected-commit", required=True)
    workstation.add_argument("--expected-inventory-sha256", required=True)

    karolina = subparsers.add_parser(
        "stage-karolina",
        help="copy one complete, semantically admitted Karolina archive into staging",
    )
    karolina.add_argument("--source", type=Path, required=True)
    karolina.add_argument("--destination", type=Path, required=True)
    karolina.add_argument("--workstation-root", type=Path, required=True)
    karolina.add_argument("--endpoint-relative", type=Path, required=True)
    karolina.add_argument("--stopping-relative", type=Path, required=True)
    karolina.add_argument("--contract", type=Path, required=True)
    karolina.add_argument("--expected-commit", required=True)
    karolina.add_argument("--expected-inventory-sha256", required=True)

    stopping = subparsers.add_parser(
        "stage-stopping-adjudication",
        help="copy the endpoint-bound STOP adjudication to its canonical staging path",
    )
    stopping.add_argument("--workstation-root", type=Path, required=True)
    stopping.add_argument("--karolina-root", type=Path, required=True)
    stopping.add_argument("--endpoint-relative", type=Path, required=True)
    stopping.add_argument("--stopping-relative", type=Path, required=True)
    stopping.add_argument("--destination", type=Path, required=True)
    stopping.add_argument("--contract", type=Path, required=True)
    stopping.add_argument("--expected-commit", required=True)
    stopping.add_argument("--expected-sha256", required=True)

    endpoint = subparsers.add_parser(
        "stage-endpoint",
        help="copy the admitted Tier-B endpoint to its canonical in-archive path",
    )
    endpoint.add_argument("--workstation-root", type=Path, required=True)
    endpoint.add_argument("--karolina-root", type=Path, required=True)
    endpoint.add_argument("--endpoint-relative", type=Path, required=True)
    endpoint.add_argument("--stopping-adjudication", type=Path, required=True)
    endpoint.add_argument("--destination", type=Path, required=True)
    endpoint.add_argument("--contract", type=Path, required=True)
    endpoint.add_argument("--expected-commit", required=True)
    endpoint.add_argument("--expected-sha256", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.mode == "plan":
            result: Any = write_dependency_plan(
                output=args.output,
                expected_commit=args.expected_commit,
                workstation_source=args.workstation_source,
                karolina_source=args.karolina_source,
                endpoint_relative=args.endpoint_relative,
                stopping_relative=args.stopping_relative,
                contract_path=args.contract,
            )
        elif args.mode == "stage-workstation":
            result = stage_workstation(
                source=args.source,
                destination=args.destination,
                contract_path=args.contract,
                expected_commit=args.expected_commit,
                expected_inventory_sha256=args.expected_inventory_sha256,
            )
        elif args.mode == "stage-karolina":
            result = stage_karolina(
                source=args.source,
                destination=args.destination,
                workstation_root=args.workstation_root,
                endpoint_relative=args.endpoint_relative,
                stopping_relative=args.stopping_relative,
                contract_path=args.contract,
                expected_commit=args.expected_commit,
                expected_inventory_sha256=args.expected_inventory_sha256,
            )
        elif args.mode == "stage-stopping-adjudication":
            result = stage_stopping_adjudication(
                workstation_root=args.workstation_root,
                karolina_root=args.karolina_root,
                endpoint_relative=args.endpoint_relative,
                stopping_relative=args.stopping_relative,
                destination=args.destination,
                contract_path=args.contract,
                expected_commit=args.expected_commit,
                expected_sha256=args.expected_sha256,
            )
        else:
            result = stage_endpoint(
                workstation_root=args.workstation_root,
                karolina_root=args.karolina_root,
                endpoint_relative=args.endpoint_relative,
                stopping_adjudication=args.stopping_adjudication,
                destination=args.destination,
                contract_path=args.contract,
                expected_commit=args.expected_commit,
                expected_sha256=args.expected_sha256,
            )
    except (OSError, ValueError, RouteDependencyError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(
        result
        if isinstance(result, Path)
        else json.dumps(result, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()

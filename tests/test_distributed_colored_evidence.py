from __future__ import annotations

import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "paper/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import distributed_colored_evidence as evidence  # noqa: E402
import check_distributed_colored_manifest as checker  # noqa: E402
import generate_distributed_colored_table as generator  # noqa: E402
import validate_paper_assets as asset_validator  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _materialize_command(normalized: list[str], root: Path) -> list[str]:
    command: list[str] = []
    for token in normalized:
        if token in {"${MPIEXEC}", "${PYTHON}"}:
            command.append(sys.executable)
        elif token.startswith("${OUTPUT_ROOT}/"):
            command.append(str(root / token.removeprefix("${OUTPUT_ROOT}/")))
        else:
            command.append(token)
    return command


def _write_route(
    root: Path,
    *,
    block: dict[str, object],
    route: str,
    position: int,
    commit: str,
    run_id: str,
    normalized: list[str],
) -> dict[str, object]:
    block_id = str(block["block_id"])
    ranks = int(block["ranks"])
    route_dir = root / "blocks" / block_id / route
    route_dir.mkdir(parents=True)
    state = np.linspace(0.0, 0.8, 9)
    gradient = np.linspace(-1.0, 1.0, 9)
    actions = np.vstack([gradient + float(index) for index in range(4)])
    raw_command = _materialize_command(normalized, root)
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": "EXP-ROUTE-001",
        "tier": "fixed_state_screen",
        "status": "completed",
        "route": route,
        "mesh_name": "hetero_ssr_L1",
        "element_degree": block["degree"],
        "quadrature_rule_id": evidence.RULE_BY_DEGREE[int(block["degree"])],
        "constraint_variant": "glued_bottom",
        "lambda_target": 1.55,
        "state_family": "analytic_mesh_field_v1",
        "state_label": block["state_label"],
        "state_amplitude": block["state_amplitude"],
        "state_sha256": evidence.array_sha256(state),
        "probe_count": 4,
        "action_sha256": evidence.array_sha256(actions[0]),
        "action_sha256_by_probe": [evidence.array_sha256(row) for row in actions],
        "gradient_sha256": evidence.array_sha256(gradient),
        "branch_diagnostics": {
            "counts": {"elastic": 7, "shear": 2},
            "normalized_boundary_margin_min": 0.1,
        },
        "model_covariates": {"global_free_dofs": 9},
        "mpi_ranks": ranks,
        "warmup_repetitions": 1,
        "measured_repetitions": 5,
        "rank_summaries": [
            {
                "rank": rank,
                "owned_dofs": (9 // ranks) + (1 if rank < (9 % ranks) else 0),
            }
            for rank in range(ranks)
        ],
        "command": shlex.join([sys.executable, *raw_command[5:]]),
        "git": {"commit": commit, "dirty": False},
        "job_metadata": {"workstation_run_id": run_id},
        "comparison_design": {
            "comparison_id": block_id,
            "block_repetition": 1,
            "route_order_position": position,
            "route_order_policy": "local_distributed_correctness_v2",
            "timing_reduction": "mpi_collective_max",
            "independent_process_block": True,
        },
        "action_out": "tangent_action.npz",
    }
    np.savez_compressed(
        route_dir / "tangent_action.npz",
        state=state,
        tangent_action=actions[0],
        tangent_actions=actions,
        gradient=gradient,
        route=np.asarray(route),
        state_label=np.asarray(str(block["state_label"])),
    )
    if ranks == 1:
        indptr = np.arange(10, dtype=np.int64)
        indices = np.arange(9, dtype=np.int64)
        values = np.linspace(1.0, 2.0, 9)
        np.savez_compressed(
            route_dir / "tangent_matrix_csr.npz",
            indptr=indptr,
            indices=indices,
            values=values,
            shape=np.asarray([9, 9], dtype=np.int64),
            route=np.asarray(route),
        )
        payload.update(
            {
                "direct_matrix_out": "tangent_matrix_csr.npz",
                "direct_matrix_nonzeros": 9,
                "direct_matrix_value_sha256": evidence.array_sha256(values),
            }
        )
    else:
        payload["direct_matrix_out"] = ""
    _write_json(route_dir / "output.json", payload)
    (route_dir / "command.txt").write_text(shlex.join(raw_command) + "\n", encoding="utf-8")
    (route_dir / "stdout.txt").write_text("completed\n", encoding="utf-8")
    (route_dir / "stderr.txt").write_text("", encoding="utf-8")
    hashes = evidence._tree_hashes(route_dir, excluded={"process_record.json"})
    record: dict[str, object] = {
        "schema_id": evidence.PROCESS_SCHEMA_ID,
        "schema_version": 1,
        "experiment_id": "EXP-DIST-001",
        "run_kind": "publication",
        "run_id": run_id,
        "block_id": block_id,
        "route": route,
        "route_order_position": position,
        "status": "completed",
        "returncode": 0,
        "started_at_utc": "2026-07-10T12:00:00Z",
        "finished_at_utc": "2026-07-10T12:00:01Z",
        "launcher_wall_time_s": 1.0,
        "timing_claim_admissible": False,
        "source_commit": commit,
        "command_argv": raw_command,
        "normalized_command_argv": normalized,
        "normalized_command_sha256": evidence.json_sha256(normalized),
        "artifact_hash_closure": {
            "algorithm": "sha256",
            "scope": "all_regular_files_below_route_directory_except_process_record",
            "excluded_paths": ["process_record.json"],
            "file_count": len(hashes),
            "files": hashes,
            "files_map_sha256": evidence.json_sha256(hashes),
        },
    }
    process_path = route_dir / "process_record.json"
    _write_json(process_path, record)
    return {
        **record,
        "process_record": "process_record.json",
        "process_record_sha256": evidence.sha256_file(process_path),
    }


def _synthetic_campaign(tmp_path: Path) -> tuple[Path, Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    for name, content in (
        ("code.py", "VALUE = 1\n"),
        ("protocol.md", "# Frozen protocol\n"),
        ("input.dat", "immutable input\n"),
    ):
        (repo / name).write_text(content, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")

    root = repo / "artifacts/reproduction/exp-dist-fixture"
    root.mkdir(parents=True)
    run_id = "fixture-run"
    blocks = evidence._expected_blocks()
    command_rows: list[dict[str, object]] = []
    process_rows: list[dict[str, object]] = []
    for block in blocks:
        for position, raw_route in enumerate(block["route_order"]):
            route = str(raw_route)
            normalized = evidence._expected_normalized_command(block, route, position)
            command_rows.append(
                {
                    "block_id": block["block_id"],
                    "route": route,
                    "route_order_position": position,
                    "normalized_argv": normalized,
                    "normalized_argv_sha256": evidence.json_sha256(normalized),
                }
            )
            process_rows.append(
                _write_route(
                    root,
                    block=block,
                    route=route,
                    position=position,
                    commit=commit,
                    run_id=run_id,
                    normalized=normalized,
                )
            )
    plan = {
        "schema_id": evidence.PLAN_SCHEMA_ID,
        "schema_version": 2,
        "experiment_id": "EXP-DIST-001",
        "run_id": run_id,
        "created_at_utc": "2026-07-10T12:00:00Z",
        "source_commit": commit,
        "source_clean": True,
        "run_kind": "publication",
        "blocks": blocks,
        "normalized_commands": command_rows,
        "normalized_commands_sha256": evidence.json_sha256(command_rows),
        "timing_claim_admissible": False,
    }
    _write_json(root / "plan.json", plan)
    packages = {
        name: "fixture"
        for name in ("h5py", "jax", "jaxlib", "mpi4py", "numpy", "petsc4py", "scipy")
    }
    environment = {
        "python_executable": sys.executable,
        "python_executable_sha256": evidence.sha256_file(Path(sys.executable)),
        "mpi_launcher": sys.executable,
        "mpi_launcher_sha256": evidence.sha256_file(Path(sys.executable)),
        "packages": packages,
        "thread_environment": evidence.EXPECTED_THREAD_ENVIRONMENT,
    }
    _write_json(root / "environment.json", environment)
    verification = {
        "schema_id": evidence.VERIFICATION_SCHEMA_ID,
        "schema_version": 2,
        "status": "passed",
        "errors": [],
        "timing_claim_admissible": False,
    }
    _write_json(root / "verification_summary.json", verification)
    inventories = {
        "code_hashes": {"code.py": evidence.sha256_file(repo / "code.py")},
        "configuration_hashes": {
            "protocol.md": evidence.sha256_file(repo / "protocol.md")
        },
        "input_hashes": {"input.dat": evidence.sha256_file(repo / "input.dat")},
    }
    manifest: dict[str, object] = {
        "schema_id": evidence.CAMPAIGN_SCHEMA_ID,
        "schema_version": 2,
        "experiment_id": "EXP-DIST-001",
        "run_kind": "publication",
        "run_id": run_id,
        "status": "completed",
        "created_at_utc": "2026-07-10T12:00:00Z",
        "finished_at_utc": "2026-07-10T12:01:00Z",
        "source_commit": commit,
        "source_clean": True,
        "expected_commit": commit,
        "plan_path": "plan.json",
        "plan_sha256": evidence.sha256_file(root / "plan.json"),
        "environment_path": "environment.json",
        "environment_sha256": evidence.sha256_file(root / "environment.json"),
        "normalized_commands_sha256": evidence.json_sha256(command_rows),
        "planned_blocks": 12,
        "planned_route_processes": 36,
        "process_records": process_rows,
        "timing_claim_admissible": False,
        "terminal_source": {"commit": commit, "dirty": False},
        "terminal_frozen_hash_verification": {"passed": True, "errors": {}},
        "verification_summary": "verification_summary.json",
        "verification_sha256": evidence.sha256_file(root / "verification_summary.json"),
        **inventories,
    }
    for name, inventory in inventories.items():
        manifest[f"{name}_sha256"] = evidence.json_sha256(inventory)
    files = evidence._tree_hashes(root, excluded={"manifest.json"})
    manifest["output_hash_closure"] = {
        "algorithm": "sha256",
        "scope": "all_regular_files_below_output_root_except_manifest",
        "excluded_paths": ["manifest.json"],
        "file_count": len(files),
        "files": files,
        "files_map_sha256": evidence.json_sha256(files),
    }
    _write_json(root / "manifest.json", manifest)
    return repo, root, commit


def test_independent_admission_accepts_closed_campaign_and_renders_deterministically(
    tmp_path: Path,
) -> None:
    repo, root, _commit = _synthetic_campaign(tmp_path)
    audit = evidence.audit_campaign(root, repo_root=repo)
    assert audit["status"] == "admitted_correctness_only"
    assert audit["timing_claim_admissible"] is False
    numerical = audit["numerical_revalidation"]
    assert numerical["status"] == "passed"
    assert len(numerical["rows"]) == 12
    first = evidence.render_table(audit)
    second = evidence.render_table(audit)
    assert first.encode("utf-8") == second.encode("utf-8")
    assert "No timing measurements enter" in first
    assert "speedup" not in first.lower()


def test_admission_rejects_unrecorded_file_and_hash_tampering(tmp_path: Path) -> None:
    repo, root, _commit = _synthetic_campaign(tmp_path)
    (root / "unrecorded.txt").write_text("not closed\n", encoding="utf-8")
    with pytest.raises(evidence.AdmissionError, match="hash closure differs"):
        evidence.validate_campaign_envelope(root, repo_root=repo)


def test_independent_revalidation_rejects_validly_rehashed_action_drift(
    tmp_path: Path,
) -> None:
    _repo, root, commit = _synthetic_campaign(tmp_path)
    route_dir = root / "blocks/p2_mixed_np4/colored_sfd"
    with np.load(route_dir / "tangent_action.npz", allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    actions = np.asarray(values["tangent_actions"], dtype=np.float64)
    actions[3, 4] += 1.0e-3
    values["tangent_actions"] = actions
    np.savez_compressed(route_dir / "tangent_action.npz", **values)
    payload = evidence._object(route_dir / "output.json")
    payload["action_sha256_by_probe"] = [evidence.array_sha256(row) for row in actions]
    _write_json(route_dir / "output.json", payload)
    with pytest.raises(evidence.AdmissionError, match="tangent actions"):
        evidence.revalidate_numerical_evidence(
            root, source_commit=commit, run_id="fixture-run"
        )


def test_independent_revalidation_rejects_child_identity_and_malformed_csr(
    tmp_path: Path,
) -> None:
    _repo, root, commit = _synthetic_campaign(tmp_path)
    output = root / "blocks/p1_elastic_np1/colored_sfd/output.json"
    payload = evidence._object(output)
    payload["git"] = {"commit": "0" * 40, "dirty": False}
    _write_json(output, payload)
    with pytest.raises(evidence.AdmissionError, match="child Git identity"):
        evidence.revalidate_numerical_evidence(
            root, source_commit=commit, run_id="fixture-run"
        )

    payload["git"] = {"commit": commit, "dirty": False}
    _write_json(output, payload)
    csr = root / "blocks/p1_elastic_np1/colored_sfd/tangent_matrix_csr.npz"
    np.savez_compressed(
        csr,
        indptr=np.asarray([0, 2, 2, 3, 4, 5, 6, 7, 8, 10], dtype=np.int64),
        indices=np.asarray([0, 0, 2, 3, 4, 5, 6, 7, 8, 8], dtype=np.int64),
        values=np.ones(10),
        shape=np.asarray([9, 9], dtype=np.int64),
        route=np.asarray("colored_sfd"),
    )
    with pytest.raises(evidence.AdmissionError, match="strictly increasing"):
        evidence.revalidate_numerical_evidence(
            root, source_commit=commit, run_id="fixture-run"
        )


def test_paper_asset_hook_binds_independent_manifest_without_revision_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    tables = repo / "paper/tables/generated"
    tables.mkdir(parents=True)
    source = repo / "artifacts/reproduction/campaign/manifest.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}\n", encoding="utf-8")
    table = tables / evidence.TABLE_NAME
    table.write_text("table bytes\n", encoding="utf-8")
    tools: dict[str, object] = {}
    for name in ("validator", "generator", "checker"):
        path = repo / f"paper/scripts/{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n", encoding="utf-8")
        tools[name] = {
            "path": path.relative_to(repo).as_posix(),
            "sha256": evidence.sha256_file(path),
        }
    _write_json(
        tables / evidence.MANIFEST_NAME,
        {
            "schema_id": "fenics-nonlinear-energies.distributed-colored-table-manifest",
            "schema_version": 1,
            "status": "admitted_correctness_only",
            "publication_evidence": True,
            "experiment_id": "EXP-DIST-001",
            "timing_claim_admissible": False,
            "allow_unreferenced_tables": True,
            "outputs": {evidence.TABLE_NAME: evidence.sha256_file(table)},
            "tools": tools,
            "source_campaign_manifest": {
                "path": source.relative_to(repo).as_posix(),
                "sha256": evidence.sha256_file(source),
            },
        },
    )
    monkeypatch.setattr(asset_validator, "REPO_ROOT", repo)
    assert asset_validator._distributed_colored_manifest_tables(tables) == {
        evidence.TABLE_NAME
    }
    asset_validator._validate_distributed_colored_manifest(tables)

    table.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="SHA-256 hash is stale"):
        asset_validator._validate_distributed_colored_manifest(tables)


def test_generator_and_checker_regenerate_table_and_manifest_byte_for_byte(
    tmp_path: Path,
) -> None:
    repo, root, _commit = _synthetic_campaign(tmp_path)
    scripts = repo / "paper/scripts"
    scripts.mkdir(parents=True)
    for name in (
        "distributed_colored_evidence.py",
        "generate_distributed_colored_table.py",
        "check_distributed_colored_manifest.py",
    ):
        shutil.copy2(SCRIPTS / name, scripts / name)
    out = repo / "paper/tables/generated"
    manifest = generator.generate(root, out, repo_root=repo)
    assert manifest["timing_claim_admissible"] is False
    assert set(manifest["outputs"]) == {evidence.TABLE_NAME}
    assert checker.validate_manifest(
        out / evidence.MANIFEST_NAME,
        repo_root=repo,
        require_canonical=True,
    ) == []

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "paper/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_globalization_local_manifest as checker  # noqa: E402
import generate_globalization_local_status as generator  # noqa: E402
import globalization_local_evidence as evidence  # noqa: E402
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


def _relative(path: Path, repo: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def _write_state(path: Path, benchmark: str, value: float, *, canonical: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if benchmark.startswith("gl_"):
        state = np.full(4, value, dtype=np.float64)
        np.savez(
            path,
            coords=np.zeros((4, 2), dtype=np.float64),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
            u=state,
        )
        return evidence.array_sha256(state)
    coords = np.full((4, 3), value, dtype=np.float64)
    np.savez(
        path,
        coords_ref=np.zeros((4, 3), dtype=np.float64),
        coords_final=coords,
        displacement=coords,
        tetrahedra=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
    )
    return evidence.array_sha256(coords if canonical else coords.reshape(-1))


def _fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / ".gitignore").write_text(
        "artifacts/\npaper/tables/generated/\n__pycache__/\n", encoding="utf-8"
    )
    (repo / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "protocol.md").write_text("# Frozen protocol\n", encoding="utf-8")
    scripts = repo / "paper/scripts"
    scripts.mkdir(parents=True)
    for name in (
        "globalization_local_evidence.py",
        "generate_globalization_local_status.py",
        "check_globalization_local_manifest.py",
    ):
        shutil.copy2(SCRIPTS / name, scripts / name)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")

    root = repo / "artifacts/reproduction/exp-glob-fixture"
    raw = root / evidence.RAW_RELATIVE
    reports = root / evidence.REPORT_RELATIVE
    raw.mkdir(parents=True)
    reports.mkdir(parents=True)

    start_rows: dict[str, object] = {}
    starts: dict[tuple[str, str], tuple[Path, str, str]] = {}
    for benchmark_index, benchmark in enumerate(evidence.BENCHMARKS):
        for instance_index, instance in enumerate(evidence.INSTANCES):
            state_path = raw / "_canonical_starts" / benchmark / f"{instance}.npz"
            content = _write_state(
                state_path,
                benchmark,
                0.01 * (1 + benchmark_index + instance_index),
                canonical=True,
            )
            file_hash = evidence.sha256_file(state_path)
            starts[(benchmark, instance)] = (state_path, file_hash, content)
            start_rows[f"{benchmark}::{instance}"] = {
                "benchmark": benchmark,
                "problem": "gl" if benchmark.startswith("gl_") else "he",
                "level": 5 if benchmark.startswith("gl_") else 2,
                "robustness_instance": instance,
                "robustness_parameters": {"instance_id": instance},
                "path": _relative(state_path, repo),
                "file_sha256": file_hash,
                "state_sha256": content,
            }
    start_manifest = raw / "_canonical_starts/manifest.json"
    _write_json(
        start_manifest,
        {
            "schema_id": evidence.COMMON_START_SCHEMA_ID,
            "schema_version": 2,
            "status": "prepared",
            "created_at_utc": "2026-01-01T00:00:00Z",
            "instances": start_rows,
        },
    )

    cases: list[dict[str, object]] = []
    planned: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    run_bindings: list[dict[str, str]] = []
    for benchmark_index, benchmark in enumerate(evidence.BENCHMARKS):
        for instance_index, instance in enumerate(evidence.INSTANCES):
            for method in evidence.METHODS:
                for repetition in evidence.REPETITIONS:
                    case_id = f"{benchmark}_{instance}_{method}_r{repetition:02d}"
                    case = {
                        "mode": "smoke",
                        "comparison_tier": "controlled",
                        "benchmark": {"key": benchmark, "ranks": 2},
                        "method": {"key": method},
                        "robustness_instance": {"instance_id": instance},
                        "timing_repetition": repetition,
                    }
                    cases.append(case)
                    command = ["python", "source.py", case_id]
                    planned.append(
                        {
                            "case_id": case_id,
                            "benchmark": benchmark,
                            "method": method,
                            "robustness_instance": instance,
                            "timing_repetition": repetition,
                            "command_argv": command,
                            "command_sha256": evidence.json_sha256(command),
                            "input_hashes": {},
                        }
                    )
                    case_root = raw / case_id
                    case_root.mkdir()
                    terminal = case_root / "final_state.npz"
                    terminal_content = _write_state(
                        terminal,
                        benchmark,
                        1.0 + benchmark_index + instance_index,
                        canonical=False,
                    )
                    terminal_file = evidence.sha256_file(terminal)
                    endpoint = "a" * 64
                    residual_hash = "b" * 64
                    residual = 1.0e-8
                    start_path, start_file, start_content = starts[(benchmark, instance)]
                    output = case_root / "output.json"
                    _write_json(
                        output,
                        {
                            "result": {
                                "metadata": {
                                    "initial_state_input": {
                                        "file_sha256": start_file,
                                        "state_sha256": start_content,
                                    },
                                    "state_output": {
                                        "file_sha256": terminal_file,
                                        "state_sha256": terminal_content,
                                    },
                                    "endpoint_identity": {
                                        "owned_reordered_state_sha256": endpoint,
                                        "independent_residual": {
                                            "dual_norm": residual,
                                            "coefficient_l2_norm": residual,
                                            "evaluated_after_solver_termination": True,
                                            "owned_reordered_gradient_sha256": residual_hash,
                                        },
                                    },
                                }
                            }
                        },
                    )
                    log = case_root / "run.log"
                    log.write_text("completed\n", encoding="utf-8")
                    record = {
                        "schema": {"id": evidence.RUN_RECORD_SCHEMA_ID, "version": 1},
                        "run_kind": "publication",
                        "identifiers": {
                            "campaign": evidence.CAMPAIGN_ID,
                            "experiment": "EXP-GLOB-001",
                            "method": method,
                            "repetition": repetition,
                        },
                        "provenance": {
                            "git_commit": commit,
                            "git_clean": True,
                            "git_status_porcelain": [],
                            "dirty_patch_sha256": None,
                            "command_argv": command,
                            "code_hashes": {
                                "source.py": evidence.sha256_file(repo / "source.py")
                            },
                            "configuration_hashes": {
                                "protocol.md": evidence.sha256_file(repo / "protocol.md"),
                                "campaign_configuration": "pending",
                            },
                        },
                        "diagnostics": {
                            "state": {
                                "initial_file_sha256": start_file,
                                "initial_content_sha256": start_content,
                                "final_file_sha256": terminal_file,
                                "final_content_sha256": terminal_content,
                            }
                        },
                        "accuracy": {
                            "absolute_residual": residual,
                            "gate_passed": True,
                        },
                        "termination": {"status": "success"},
                        "artifacts": {
                            "raw_outputs": [_relative(output, repo)],
                            "logs": [_relative(log, repo)],
                            "states": [_relative(start_path, repo), _relative(terminal, repo)],
                        },
                    }
                    record_path = case_root / "run_record.json"
                    # The campaign-configuration digest is filled after the grid is built.
                    _write_json(record_path, record)
                    summary_rows.append(
                        {
                            "mode": "smoke",
                            "comparison_tier": "controlled",
                            "benchmark": benchmark,
                            "robustness_instance": instance,
                            "method": method,
                            "timing_repetition": repetition,
                            "result": "completed",
                            "wall_time_s": 1.0,
                            "line_search_time_s": 0.1,
                            "independent_dual_residual": residual,
                            "independent_coefficient_residual": residual,
                            "initial_state_file_sha256": start_file,
                            "initial_state_content_sha256": start_content,
                            "final_state_file_sha256": terminal_file,
                            "final_state_content_sha256": terminal_content,
                            "endpoint_state_sha256": endpoint,
                            "independent_residual_sha256": residual_hash,
                            "run_record_sha256": "pending",
                        }
                    )

    configuration = {
        "campaign_id": evidence.CAMPAIGN_ID,
        "mode": "smoke",
        "comparison_tier": "controlled",
        "maximum_local_ranks": 4,
        "machine_noise_repetitions": 5,
        "controlled_child_environment": {
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
        },
        "cases": cases,
    }
    configuration_hash = evidence.json_sha256(configuration)
    for row in summary_rows:
        case_id = (
            f"{row['benchmark']}_{row['robustness_instance']}_{row['method']}_"
            f"r{int(row['timing_repetition']):02d}"
        )
        record_path = raw / case_id / "run_record.json"
        record = evidence.read_strict_json(record_path)
        record["provenance"]["configuration_hashes"][
            "campaign_configuration"
        ] = configuration_hash
        _write_json(record_path, record)
        record_hash = evidence.sha256_file(record_path)
        row["run_record_sha256"] = record_hash
        run_bindings.append({"path": _relative(record_path, repo), "sha256": record_hash})

    summary_json = reports / "smoke_summary.json"
    _write_json(
        summary_json,
        {
            "mode": "smoke",
            "comparison_tier": "controlled",
            "generated_at": "2026-01-01T00:00:00Z",
            "rows": summary_rows,
        },
    )
    summary_csv = reports / "smoke_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    identity = reports / "smoke_identity_audit.json"
    _write_json(
        identity,
        {
            "schema_id": "fenics-nonlinear-energies.exp-glob-001-identity-audit",
            "schema_version": 2,
            "status": "passed",
            "timing_claim_admissible": False,
            "tested_instance_comparison_admissible": True,
            "robustness_generalization_claim_admissible": False,
        },
    )
    campaign = {
        "schema": {"id": evidence.CAMPAIGN_SCHEMA_ID, "version": 1},
        "campaign_id": evidence.CAMPAIGN_ID,
        "status": "completed",
        "publication_preflight": {
            "git_commit": commit,
            "git_clean": True,
            "git_status_porcelain": [],
            "pilot_override": False,
        },
        "configuration": configuration,
        "configuration_sha256": configuration_hash,
        "source_hashes": {"source.py": evidence.sha256_file(repo / "source.py")},
        "protocol_hashes": {"protocol.md": evidence.sha256_file(repo / "protocol.md")},
        "common_start_manifest": {
            "path": _relative(start_manifest, repo),
            "sha256": evidence.sha256_file(start_manifest),
        },
        "planned_runs": planned,
        "run_records": run_bindings,
        "reports": {
            _relative(path, repo): evidence.sha256_file(path)
            for path in (summary_json, summary_csv, identity)
        },
        "claim_admission": {
            "timing_claim_admissible": False,
            "tested_instance_comparison_admissible": True,
            "robustness_generalization_claim_admissible": False,
        },
    }
    _write_json(raw / "campaign_manifest.json", campaign)
    return repo, root, commit


def test_audit_closes_campaign_and_renders_deterministically(tmp_path: Path) -> None:
    repo, root, commit = _fixture(tmp_path)
    audit = evidence.audit_campaign(root, repo_root=repo)
    assert audit["source_commit"] == commit
    assert audit["artifact_count"] == 251
    adjudication = audit["scientific_adjudication"]
    assert adjudication["same_endpoint_comparison_gate_passed"] is True
    assert adjudication["timing_claim_admissible"] is False
    assert adjudication["population_robustness_claim_admissible"] is False
    first = evidence.render_table(audit)
    second = evidence.render_table(audit)
    assert first.encode() == second.encode()
    assert r"Ginzburg--Landau, $L_5$, 2 ranks" in first
    assert r"Hyperelasticity, $L_2$, step 1, 2 ranks" in first
    assert r"\begin{tabularx}{\linewidth}" in first
    assert "performance ordering are excluded" in first
    assert "speedup" not in first.lower()


def test_admission_rejects_missing_row_and_nonfinite_endpoint(tmp_path: Path) -> None:
    repo, root, _commit = _fixture(tmp_path)
    manifest_path = root / evidence.CAMPAIGN_MANIFEST_RELATIVE
    manifest = evidence.read_strict_json(manifest_path)
    manifest["configuration"]["cases"].pop()
    manifest["configuration_sha256"] = evidence.json_sha256(manifest["configuration"])
    _write_json(manifest_path, manifest)
    with pytest.raises(evidence.AdmissionError, match="exactly 60 cases"):
        evidence.audit_campaign(root, repo_root=repo)

    repo, root, _commit = _fixture(tmp_path / "nonfinite")
    output = root / evidence.RAW_RELATIVE / "gl_l5_np2_nominal_newton_armijo_r01/output.json"
    payload = evidence.read_strict_json(output)
    payload["result"]["metadata"]["endpoint_identity"]["independent_residual"][
        "dual_norm"
    ] = 1.0e999
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(evidence.AdmissionError, match="nonfinite"):
        evidence.audit_campaign(root, repo_root=repo)


def test_admission_rejects_path_escape_symlink_and_commit_drift(tmp_path: Path) -> None:
    repo, root, _commit = _fixture(tmp_path)
    case = root / evidence.RAW_RELATIVE / "gl_l5_np2_nominal_newton_armijo_r01"
    terminal = case / "final_state.npz"
    outside = repo / "artifacts/reproduction/outside.npz"
    terminal.rename(outside)
    terminal.symlink_to(outside)
    with pytest.raises(evidence.AdmissionError, match="symlinks are forbidden"):
        evidence.audit_campaign(root, repo_root=repo)

    repo, root, _commit = _fixture(tmp_path / "drift")
    manifest_path = root / evidence.CAMPAIGN_MANIFEST_RELATIVE
    manifest = evidence.read_strict_json(manifest_path)
    manifest["publication_preflight"]["git_commit"] = "f" * 40
    _write_json(manifest_path, manifest)
    with pytest.raises(evidence.AdmissionError, match="not an ancestor"):
        evidence.audit_campaign(root, repo_root=repo)


def test_checker_rejects_hash_tampering_and_semantic_overclaim(tmp_path: Path) -> None:
    repo, root, _commit = _fixture(tmp_path)
    out = repo / "paper/tables/generated"
    generator.generate(root, out, repo_root=repo)
    manifest_path = out / evidence.MANIFEST_NAME
    assert checker.validate_manifest(manifest_path, repo_root=repo) == []

    log = root / evidence.RAW_RELATIVE / "gl_l5_np2_nominal_newton_armijo_r01/run.log"
    log.write_text("tampered\n", encoding="utf-8")
    errors = checker.validate_manifest(manifest_path, repo_root=repo)
    assert any("stored admission audit differs" in error for error in errors)

    repo, root, _commit = _fixture(tmp_path / "overclaim")
    out = repo / "paper/tables/generated"
    generator.generate(root, out, repo_root=repo)
    manifest_path = out / evidence.MANIFEST_NAME
    payload = evidence.read_strict_json(manifest_path)
    payload["timing_claim_admissible"] = True
    payload["population_robustness_claim_admissible"] = True
    _write_json(manifest_path, payload)
    errors = checker.validate_manifest(manifest_path, repo_root=repo)
    assert any("timing_claim_admissible" in error for error in errors)
    assert any("population_robustness_claim_admissible" in error for error in errors)


def test_optional_paper_asset_hook_validates_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, root, _commit = _fixture(tmp_path)
    tables = repo / "paper/tables/generated"
    generator.generate(root, tables, repo_root=repo)
    monkeypatch.setattr(asset_validator, "REPO_ROOT", repo)
    assert asset_validator._globalization_local_manifest_tables(tables) == {
        evidence.TABLE_NAME
    }
    asset_validator._validate_globalization_local_manifest(tables)
    (tables / evidence.TABLE_NAME).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="SHA-256 hash is stale"):
        asset_validator._validate_globalization_local_manifest(tables)

from __future__ import annotations

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

import check_stopping_local_manifest as checker  # noqa: E402
import generate_stopping_local_status as generator  # noqa: E402
import stopping_local_evidence as evidence  # noqa: E402
import validate_paper_assets as asset_validator  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _result_gl(path: Path, target: float) -> None:
    _write_json(
        path,
        {
            "result": {
                "metadata": {"convergence": {"selection": "lumped_l2"}},
                "steps": [
                    {
                        "success": True,
                        "message": "converged",
                        "energy": 1.0,
                        "convergence": {
                            "dual_residual_relative": min(1.0e-9, target / 10.0),
                            "correction_norm": 1.0e-9,
                            "relative_correction": 1.0e-9,
                        },
                    }
                ],
            }
        },
    )


def _result_he_reference(path: Path) -> None:
    _write_json(
        path,
        {
            "result": {
                "steps": [
                    {
                        "energy": 1.0,
                        "convergence": {
                            "dual_residual_norm": 1.0,
                            "dual_residual_relative": 1.0,
                            "state_scale": 1.0,
                            "metric": {
                                "provenance": {
                                    "spd_certificate": {"certified_spd": True}
                                }
                            },
                            "dual_residual_metadata": {
                                "iterations": 2,
                                "reason": 2,
                                "relative_true_residual": 1.0e-12,
                                "true_residual_rtol_gate": 1.0e-8,
                            },
                        },
                    }
                ]
            }
        },
    )


def _result_he_nonlinear(path: Path, target: float) -> None:
    _write_json(
        path,
        {
            "result": {
                "steps": [
                    {
                        "success": True,
                        "message": "converged",
                        "energy": 1.0,
                        "convergence": {
                            "dual_residual_norm": 1.0e-9,
                            "dual_residual_relative": min(1.0e-9, target / 10.0),
                            "relative_correction": 1.0e-9,
                            "state_scale": 1.0,
                            "metric": {
                                "provenance": {
                                    "spd_certificate": {"certified_spd": True}
                                }
                            },
                            "dual_residual_metadata": {
                                "iterations": 2,
                                "reason": 2,
                                "relative_true_residual": 1.0e-12,
                                "true_residual_rtol_gate": 1.0e-8,
                            },
                        },
                    }
                ]
            }
        },
    )


def _result_p3d_fixed(path: Path, state_path: Path, arrays: dict[str, np.ndarray]) -> None:
    _write_json(
        path,
        {
            "schema_id": evidence.P3D_RESULT_SCHEMA_ID,
            "schema_version": 1,
            "status": "passed",
            "state_sha256": evidence.array_sha256(arrays["state"]),
            "rhs_sha256": evidence.array_sha256(arrays["rhs"]),
            "correction_sha256": evidence.array_sha256(arrays["correction"]),
            "state_file": {
                "path": str(state_path),
                "sha256": evidence.sha256_file(state_path),
            },
            "branch_diagnostics": {"counts": {"elastic": 4}},
            "linear_solve": {
                "reason": 2,
                "iterations": 2,
                "recursive_residual_norm": 1.0e-12,
                "true_residual_norm": 1.0e-12,
                "relative_true_residual": 1.0e-12,
                "true_residual_gate": 1.0e-8,
                "correction_norm_2": 2.0,
                "reference_elastic_correction_norm": 2.0,
            },
        },
    )


def _result_p3d_nonlinear(path: Path, target: float) -> None:
    _write_json(
        path,
        {
            "status": "completed",
            "solver_success": True,
            "message": "converged",
            "energy": 1.0,
            "omega": 0.1,
            "u_max": 0.2,
            "branch_diagnostics": {"counts": {"elastic": 4}},
            "nonlinear_convergence": {
                "configuration": {"selection": "reference_elastic_energy"},
                "metric": {
                    "provenance": {
                        "spd_certificate": {"certified_spd": True}
                    }
                },
                "last_riesz_solve": {
                    "reason": 2,
                    "relative_true_residual": 1.0e-12,
                    "true_residual_rtol_gate": 1.0e-8,
                },
                "initial_relative_dual_residual": {
                    "value": min(1.0e-9, target / 10.0)
                },
                "relative_correction": {"value": 1.0e-9},
                "residual_gate": {"passed": True},
            },
        },
    )


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
    (repo / "input.dat").write_text("immutable input\n", encoding="utf-8")
    scripts = repo / "paper/scripts"
    scripts.mkdir(parents=True)
    for name in (
        "stopping_local_evidence.py",
        "generate_stopping_local_status.py",
        "check_stopping_local_manifest.py",
    ):
        shutil.copy2(SCRIPTS / name, scripts / name)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")

    root = repo / "artifacts/reproduction/exp-stop-fixture"
    root.mkdir(parents=True)
    design = evidence.expected_design()
    rows: list[dict[str, object]] = []
    for row_id, spec in design.items():
        row: dict[str, object] = {
            "row_id": row_id,
            "family": spec["family"],
            "group_id": spec["group_id"],
            "execution_class": spec["execution_class"],
            "scientific_scope": "fixture",
            "parameters": dict(spec["parameters"]),
            "reference_row": spec["reference_row"],
        }
        if spec["execution_class"] == "required_local":
            outputs = [str(root / relative) for relative in spec["expected_outputs"]]
            row.update(
                {
                    "command": [sys.executable, str(repo / "source.py"), *outputs],
                    "environment": evidence.EXPECTED_ENVIRONMENT,
                    "expected_outputs": outputs,
                }
            )
        else:
            row.update(
                {
                    "command": None,
                    "environment": {},
                    "expected_outputs": [],
                    "censor": {
                        "status": "censored",
                        "reason": "parallel cluster computation",
                        "timing_admissible": False,
                        "accuracy_claim_admissible": False,
                    },
                }
            )
        rows.append(row)
    plan = {
        "schema_id": evidence.PLAN_SCHEMA_ID,
        "schema_version": 1,
        "experiment_id": "EXP-STOP-001",
        "campaign_id": evidence.CAMPAIGN_ID,
        "run_kind": "publication",
        "publication_evidence_candidate": True,
        "output_root": str(root),
        "source": {
            "commit": commit,
            "dirty": False,
            "relevant_file_hashes": {
                "source.py": evidence.sha256_file(repo / "source.py")
            },
        },
        "inputs": {
            "file_hashes": {"input.dat": evidence.sha256_file(repo / "input.dat")}
        },
        "environment": {"command_environment": evidence.EXPECTED_ENVIRONMENT},
        "policies": {
            "p4_fixed_state": "local",
            "p4_local_feasibility_attested": True,
            "timing_claims_admissible": False,
            "analysis_contract": evidence.EXPECTED_CONTRACT,
        },
        "row_counts": {
            "total": 52,
            "required_local": 45,
            "deferred_cluster_computation": 7,
        },
        "claim_boundary": {
            "local_completion_cannot_establish": [
                "a terminal PASS for the complete EXP-STOP-001 protocol"
            ]
        },
        "rows": rows,
    }
    plan_path = root / evidence.PLAN_NAME
    _write_json(plan_path, plan)
    plan_hash = evidence.sha256_file(plan_path)

    rows_by_id = {str(row["row_id"]): row for row in rows}
    endpoints: dict[str, dict[str, object]] = {}
    audit: dict[str, dict[str, object]] = {}
    for row in rows:
        if row["execution_class"] != "required_local":
            continue
        row_id = str(row["row_id"])
        outputs = [Path(value) for value in row["expected_outputs"]]
        result_path = next(path for path in outputs if path.suffix == ".json")
        state_paths = [path for path in outputs if path.suffix == ".npz"]
        family = str(row["family"])
        target = float(
            row["parameters"].get(
                "relative_dual_residual_target",
                row["parameters"].get("riesz_ksp_rtol", row["parameters"].get("ksp_rtol", 1.0e-8)),
            )
        )
        if family == "ginzburg_landau":
            state_path = state_paths[0]
            state_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                state_path,
                coords=np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
                triangles=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
                u=np.ones(4, dtype=np.float64),
            )
            _result_gl(result_path, target)
        elif family == "hyperelasticity_reference_riesz":
            _result_he_reference(result_path)
        elif family == "hyperelasticity_nonlinear_stopping":
            state_path = state_paths[0]
            state_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                state_path,
                coords_ref=np.zeros((4, 3), dtype=np.float64),
                tetrahedra=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
                displacement=np.full((4, 3), 0.1, dtype=np.float64),
                free_deformation_original=np.ones(4, dtype=np.float64),
                reference_elastic_action=np.ones(4, dtype=np.float64),
            )
            _result_he_nonlinear(result_path, target)
        elif family == "plasticity3d_fixed_state_linear":
            state_path = state_paths[0]
            state_path.parent.mkdir(parents=True, exist_ok=True)
            arrays = {
                "state": np.ones(4, dtype=np.float64),
                "rhs": np.ones(4, dtype=np.float64),
                "correction": np.ones(4, dtype=np.float64),
                "reference_elastic_action": np.ones(4, dtype=np.float64),
            }
            np.savez(state_path, **arrays)
            _result_p3d_fixed(result_path, state_path, arrays)
        else:
            state_path = state_paths[0]
            state_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                state_path,
                coords_ref=np.zeros((4, 3), dtype=np.float64),
                tetrahedra=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
                free_displacement_reordered=np.ones(4, dtype=np.float64),
                reference_elastic_action=np.ones(4, dtype=np.float64),
            )
            _result_p3d_nonlinear(result_path, target)
        log_root = root / "logs" / row_id
        log_root.mkdir(parents=True)
        stdout = log_root / "stdout.log"
        stderr = log_root / "stderr.log"
        stdout.write_text("completed\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        receipt_path = root / "receipts" / f"{row_id}.json"
        _write_json(
            receipt_path,
            {
                "schema_id": evidence.RECEIPT_SCHEMA_ID,
                "schema_version": 1,
                "experiment_id": "EXP-STOP-001",
                "campaign_id": evidence.CAMPAIGN_ID,
                "row_id": row_id,
                "status": "completed",
                "wall_time_s": 1.0,
                "plan_path": str(plan_path),
                "plan_sha256": plan_hash,
                "source_commit": commit,
                "run_kind": "publication",
                "command": row["command"],
                "environment_overrides": row["environment"],
                "returncode": 0,
                "timed_out": False,
                "verification_error": None,
                "output_hashes": {
                    str(path): evidence.sha256_file(path) for path in outputs
                },
                "logs": {"stdout": str(stdout), "stderr": str(stderr)},
            },
        )
        audit[row_id] = {
            "receipt_status": "completed",
            "receipt": str(receipt_path),
            "errors": [],
        }
        endpoints[row_id] = evidence.extract_endpoint(
            row, repo_root=repo, evidence_root=root
        )

    local_rows = [row for row in rows if row["execution_class"] == "required_local"]
    comparisons: dict[str, dict[str, object]] = {}
    selected: dict[str, dict[str, object]] = {}
    groups = sorted({str(row["group_id"]) for row in local_rows})
    for group in groups:
        group_rows = [row for row in local_rows if row["group_id"] == group]
        reference = next(row for row in group_rows if row["reference_row"] is True)
        for row in group_rows:
            comparisons[str(row["row_id"])] = evidence.recompute_comparison(
                row,
                endpoints[str(row["row_id"])],
                reference,
                endpoints[str(reference["row_id"])],
                repo_root=repo,
                evidence_root=root,
            )
        selected[group] = evidence.select_policy(group_rows, comparisons)
    policy_grid = {
        "expected_groups": groups,
        "observed_groups": groups,
        "missing_groups": [],
        "unexpected_groups": [],
        "missing_policy_records": [],
        "unexpected_policy_records": [],
        "rejected_policy_groups": [],
        "invalid_selected_rows": [],
        "complete": True,
    }
    deferred_rows = [row for row in rows if row["execution_class"] == "deferred_cluster_computation"]
    analysis = {
        "schema_id": evidence.ANALYSIS_SCHEMA_ID,
        "schema_version": 2,
        "experiment_id": "EXP-STOP-001",
        "campaign_id": evidence.CAMPAIGN_ID,
        "terminal_decision": "local_calibration_complete_cluster_computations_deferred",
        "complete_exp_stop_pass": False,
        "publication_timing_admissible": False,
        "scope_statement": "fixture",
        "plan": {
            "path": str(plan_path),
            "sha256": plan_hash,
            "run_kind": "publication",
            "source_commit": commit,
        },
        "counts": {
            "required_local": 45,
            "completed_endpoint_records": 45,
            "missing_local": 0,
            "invalid_local": 0,
            "runtime_censored_local": 0,
            "reference_failures": 0,
            "policy_gate_failures": 0,
            "deferred_cluster_computations": 7,
        },
        "missing_local_rows": [],
        "invalid_local_rows": [],
        "runtime_censored_local_rows": [],
        "reference_failures": [],
        "runtime_censors": [],
        "audit": audit,
        "endpoints": endpoints,
        "same_discretization_reference_comparisons": comparisons,
        "selected_local_policies": selected,
        "required_local_policy_grid": policy_grid,
        "cross_mesh_summary": {},
        "deferred_cluster_computations": [
            {
                "row_id": row["row_id"],
                "family": row["family"],
                "parameters": row["parameters"],
                "censor": row["censor"],
            }
            for row in deferred_rows
        ],
    }
    _write_json(root / evidence.ANALYSIS_NAME, analysis)
    return repo, root, commit


def test_audit_recomputes_complete_local_policy_grid_and_table(tmp_path: Path) -> None:
    repo, root, commit = _fixture(tmp_path)
    audit = evidence.audit_campaign(root, repo_root=repo)
    assert audit["source_commit"] == commit
    assert audit["status"] == "admitted_local_calibration_cluster_deferred"
    scientific = audit["scientific_adjudication"]
    assert scientific["completed_local_rows"] == 45
    assert scientific["deferred_cluster_rows"] == 7
    assert scientific["complete_exp_stop_pass"] is False
    assert scientific["timing_claim_admissible"] is False
    table = evidence.render_table(audit)
    assert table == evidence.render_table(audit)
    assert "necessary but not sufficient" in table
    assert "separately hash-bound EXP-DISC" in table


def test_admission_rejects_missing_receipt_and_nonfinite_npz(tmp_path: Path) -> None:
    repo, root, _commit = _fixture(tmp_path)
    (root / "receipts/gl_l5_residual_1em02.json").unlink()
    with pytest.raises(evidence.AdmissionError, match="receipt is missing"):
        evidence.audit_campaign(root, repo_root=repo)

    repo, root, _commit = _fixture(tmp_path / "nonfinite")
    state = root / "raw/gl/gl_l5_residual_1em02/state.npz"
    with np.load(state, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["u"] = np.asarray(arrays["u"], dtype=np.float64)
    arrays["u"][0] = np.nan
    np.savez(state, **arrays)
    receipt_path = root / "receipts/gl_l5_residual_1em02.json"
    receipt = evidence.read_strict_json(receipt_path)
    receipt["output_hashes"][str(state)] = evidence.sha256_file(state)
    _write_json(receipt_path, receipt)
    with pytest.raises(evidence.AdmissionError, match="nonfinite"):
        evidence.audit_campaign(root, repo_root=repo)


def test_admission_rejects_symlink_commit_drift_and_incomplete_policy_grid(
    tmp_path: Path,
) -> None:
    repo, root, _commit = _fixture(tmp_path)
    state = root / "raw/gl/gl_l5_residual_1em02/state.npz"
    outside = repo / "artifacts/reproduction/outside.npz"
    state.rename(outside)
    state.symlink_to(outside)
    with pytest.raises(evidence.AdmissionError, match="symlinks are forbidden"):
        evidence.audit_campaign(root, repo_root=repo)

    repo, root, _commit = _fixture(tmp_path / "escape")
    receipt_path = root / "receipts/gl_l5_residual_1em02.json"
    receipt = evidence.read_strict_json(receipt_path)
    receipt["logs"]["stdout"] = "../../outside.log"
    _write_json(receipt_path, receipt)
    with pytest.raises(evidence.AdmissionError, match="must not contain"):
        evidence.audit_campaign(root, repo_root=repo)

    repo, root, _commit = _fixture(tmp_path / "drift")
    plan_path = root / evidence.PLAN_NAME
    plan = evidence.read_strict_json(plan_path)
    plan["source"]["commit"] = "f" * 40
    _write_json(plan_path, plan)
    with pytest.raises(evidence.AdmissionError, match="not an ancestor"):
        evidence.audit_campaign(root, repo_root=repo)

    repo, root, _commit = _fixture(tmp_path / "policy")
    analysis_path = root / evidence.ANALYSIS_NAME
    analysis = evidence.read_strict_json(analysis_path)
    analysis["selected_local_policies"]["gl_l5"] = {
        "status": "no_acceptable_policy",
        "row_id": None,
        "tolerance": None,
    }
    analysis["required_local_policy_grid"]["rejected_policy_groups"] = ["gl_l5"]
    analysis["required_local_policy_grid"]["complete"] = False
    analysis["counts"]["policy_gate_failures"] = 1
    _write_json(analysis_path, analysis)
    with pytest.raises(evidence.AdmissionError, match="analysis counts"):
        evidence.audit_campaign(root, repo_root=repo)


def test_checker_rejects_hash_tampering_and_semantic_overclaim(tmp_path: Path) -> None:
    repo, root, _commit = _fixture(tmp_path)
    out = repo / "paper/tables/generated"
    generator.generate(root, out, repo_root=repo)
    manifest_path = out / evidence.MANIFEST_NAME
    assert checker.validate_manifest(manifest_path, repo_root=repo) == []

    log = root / "logs/gl_l5_residual_1em02/stdout.log"
    log.write_text("tampered\n", encoding="utf-8")
    errors = checker.validate_manifest(manifest_path, repo_root=repo)
    assert any("stored admission audit differs" in error for error in errors)

    repo, root, _commit = _fixture(tmp_path / "overclaim")
    out = repo / "paper/tables/generated"
    generator.generate(root, out, repo_root=repo)
    manifest_path = out / evidence.MANIFEST_NAME
    payload = evidence.read_strict_json(manifest_path)
    payload["complete_exp_stop_pass"] = True
    payload["timing_claim_admissible"] = True
    payload["population_robustness_claim_admissible"] = True
    _write_json(manifest_path, payload)
    errors = checker.validate_manifest(manifest_path, repo_root=repo)
    assert any("complete_exp_stop_pass" in error for error in errors)
    assert any("timing_claim_admissible" in error for error in errors)
    assert any("population_robustness_claim_admissible" in error for error in errors)


def test_optional_paper_asset_hook_validates_stopping_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, root, _commit = _fixture(tmp_path)
    tables = repo / "paper/tables/generated"
    generator.generate(root, tables, repo_root=repo)
    monkeypatch.setattr(asset_validator, "REPO_ROOT", repo)
    assert asset_validator._stopping_local_manifest_tables(tables) == {
        evidence.TABLE_NAME
    }
    asset_validator._validate_stopping_local_manifest(tables)
    (tables / evidence.TABLE_NAME).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="SHA-256 hash is stale"):
        asset_validator._validate_stopping_local_manifest(tables)

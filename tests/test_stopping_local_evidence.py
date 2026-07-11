from __future__ import annotations

import copy
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
from experiments.runners import run_exp_stop_001_local_calibration as producer  # noqa: E402


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


def _riesz_metric_contract(rtol: float, gate: float) -> dict[str, object]:
    return {
        "ksp_type": "cg",
        "pc_type": "jacobi",
        "requested_norm_type": "unpreconditioned",
        "effective_norm_type": "unpreconditioned",
        "requested_rtol": rtol,
        "effective_rtol": rtol,
        "requested_atol": 0.0,
        "effective_atol": 0.0,
        "requested_max_it": 5000,
        "effective_max_it": 5000,
        "true_residual_rtol_gate": gate,
        "set_from_petsc_options": False,
    }


def _riesz_solve_contract(rtol: float, gate: float) -> dict[str, object]:
    return {
        "ksp_type": "cg",
        "pc_type": "jacobi",
        "requested_norm_type": "unpreconditioned",
        "effective_norm_type": "unpreconditioned",
        "reported_residual_norm_type": "unpreconditioned",
        "requested_rtol": rtol,
        "effective_rtol": rtol,
        "requested_atol": 0.0,
        "effective_atol": 0.0,
        "requested_max_it": 5000,
        "effective_max_it": 5000,
        "true_residual_rtol_gate": gate,
    }


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


def _result_he_reference(path: Path, rtol: float) -> None:
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
                                **_riesz_metric_contract(rtol, 1.0e-6),
                                "provenance": {
                                    "spd_certificate": {"certified_spd": True}
                                }
                            },
                            "dual_residual_metadata": {
                                **_riesz_solve_contract(rtol, 1.0e-6),
                                "iterations": 2,
                                "reason": 2,
                                "relative_true_residual": 1.0e-12,
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
                                **_riesz_metric_contract(1.0e-10, 1.0e-8),
                                "provenance": {
                                    "spd_certificate": {"certified_spd": True}
                                }
                            },
                            "dual_residual_metadata": {
                                **_riesz_solve_contract(1.0e-10, 1.0e-8),
                                "iterations": 2,
                                "reason": 2,
                                "relative_true_residual": 1.0e-12,
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
                    **_riesz_metric_contract(1.0e-10, 1.0e-8),
                    "provenance": {
                        "spd_certificate": {"certified_spd": True}
                    }
                },
                "last_riesz_solve": {
                    **_riesz_solve_contract(1.0e-10, 1.0e-8),
                    "reason": 2,
                    "relative_true_residual": 1.0e-12,
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
        (
            "artifacts/\npaper/tables/generated/\n.venv/\n__pycache__/\n"
            "data/meshes/SlopeStability3D/**/*.h5\n"
        ),
        encoding="utf-8",
    )
    for relative in (*evidence.EXPECTED_SOURCE_PATHS, *evidence.EXPECTED_INPUT_PATHS):
        if relative == evidence.EXPECTED_MESH_MANIFEST:
            continue
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative}\n", encoding="utf-8")
    mesh_records: dict[str, dict[str, object]] = {}
    for relative, degree in evidence.EXPECTED_MANIFESTED_MESH_PATHS.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture mesh degree {degree}\n".encode())
        mesh_records[relative] = {
            "bytes": path.stat().st_size,
            "constraint_variant": "glued_bottom",
            "element_degree": degree,
            "mesh_name": "hetero_ssr_L1",
            "same_mesh_hdf5_schema_version": 7,
            "sha256": evidence.sha256_file(path),
        }
    _write_json(
        repo / evidence.EXPECTED_MESH_MANIFEST,
        {
            "algorithm": "sha256",
            "files": mesh_records,
            "generator": {
                "function": (
                    "src.problems.slope_stability_3d.support.mesh."
                    "ensure_same_mesh_case_hdf5"
                ),
                "tracked_sources": list(evidence.EXPECTED_MESH_GENERATOR_SOURCES),
            },
            "schema_id": "fenics-nonlinear-energies.manifested-generated-meshes",
            "schema_version": 1,
        },
    )
    scripts = repo / "paper/scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for name in (
        "stopping_local_evidence.py",
        "generate_stopping_local_status.py",
        "check_stopping_local_manifest.py",
    ):
        shutil.copy2(SCRIPTS / name, scripts / name)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    python = repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")

    root = repo / "artifacts/reproduction/exp-stop-fixture"
    root.mkdir(parents=True)
    design = evidence.expected_design()
    rows: list[dict[str, object]] = []
    for row_id, spec in design.items():
        row = evidence.expected_plan_row(
            row_id,
            spec,
            evidence_root=root,
            python=str(python.absolute()),
        )
        rows.append(row)
    plan = {
        "schema_id": evidence.PLAN_SCHEMA_ID,
        "schema_version": 1,
        "experiment_id": "EXP-STOP-001",
        "campaign_id": evidence.CAMPAIGN_ID,
        "created_utc": "2026-01-01T00:00:00Z",
        "run_kind": "publication",
        "publication_evidence_candidate": True,
        "output_root": str(root),
        "source": {
            "commit": commit,
            "dirty": False,
            "relevant_file_hashes": {
                relative: evidence.sha256_file(repo / relative)
                for relative in evidence.EXPECTED_SOURCE_PATHS
            },
        },
        "inputs": {
            "file_hashes": {
                relative: evidence.sha256_file(repo / relative)
                for relative in evidence.EXPECTED_INPUT_PATHS
            },
            "manifested_file_hashes": {
                relative: {
                    **record,
                    "manifest": evidence.EXPECTED_MESH_MANIFEST,
                }
                for relative, record in mesh_records.items()
            },
            "procedural_he_mesh": {
                "levels": [1, 2],
                "source": "rank_local_procedural_he_p1_mesh_builder",
                "note": "no HDF5 mesh is consumed by the frozen HE commands",
            },
        },
        "environment": {
            "python": "3.12.0",
            "python_executable": str(python.absolute()),
            "platform": "fixture-platform",
            "machine": "fixture-machine",
            "packages": {
                package: "fixture" for package in evidence.EXPECTED_PACKAGE_NAMES
            },
            "command_environment": evidence.EXPECTED_ENVIRONMENT,
        },
        "policies": {
            "p4_fixed_state": "local",
            "p4_local_feasibility_attested": True,
            "fresh_output_root_required": True,
            "command_mutation_forbidden_after_prepare": True,
            "timing_claims_admissible": False,
            "analysis_contract": evidence.EXPECTED_CONTRACT,
        },
        "row_counts": {
            "total": 52,
            "required_local": 45,
            "deferred_cluster_computation": 7,
        },
        "claim_boundary": {
            "local_completion_can_establish": [
                "deterministic GL same-mesh endpoint sensitivity on two mesh levels",
                "HE reference-Riesz setup and terminal norm-solve sensitivity on two levels",
                "HE L1/L2 one-load-step nonlinear endpoint sensitivity",
                "P1/P2 fixed-state Plasticity3D linear true-residual sensitivity",
                "P1/P2 Plasticity3D full nonlinear endpoint sensitivity",
                "P4 fixed-state sensitivity only when locally attested in this frozen plan",
            ],
            "local_completion_cannot_establish": [
                "HyperElasticity behavior beyond the frozen one-load-step L1/L2 cases",
                "full nonlinear P4 Plasticity3D convergence",
                "publication-rank MPI consistency",
                "timing or scaling claims",
                "a terminal PASS for the complete EXP-STOP-001 protocol",
            ],
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
            _result_he_reference(result_path, target)
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


def test_riesz_solver_contract_rejects_inconsistent_provenance() -> None:
    row = {
        "row_id": "he_l1_nonlinear_1em08",
        "parameters": {
            "riesz_ksp_type": "cg",
            "riesz_pc_type": "jacobi",
            "riesz_ksp_norm_type": "unpreconditioned",
            "riesz_ksp_rtol": 1.0e-10,
            "riesz_ksp_atol": 0.0,
            "riesz_ksp_max_it": 5000,
            "riesz_true_residual_rtol": 1.0e-8,
        },
    }
    metric = _riesz_metric_contract(1.0e-10, 1.0e-8)
    solve = _riesz_solve_contract(1.0e-10, 1.0e-8)
    assert (
        evidence._require_riesz_solver_contract(
            row,
            metric=metric,
            norm_solve=solve,
        )["norm_type"]
        == "unpreconditioned"
    )

    inconsistent = dict(solve)
    inconsistent["effective_norm_type"] = "preconditioned"
    with pytest.raises(evidence.AdmissionError, match="provenance differs"):
        evidence._require_riesz_solver_contract(
            row,
            metric=metric,
            norm_solve=inconsistent,
        )


def test_audit_rejects_hash_consistent_riesz_contract_mutation(tmp_path: Path) -> None:
    repo, root, _commit = _fixture(tmp_path)
    plan = evidence.read_strict_json(root / evidence.PLAN_NAME)
    row = next(
        value
        for value in plan["rows"]
        if value["row_id"] == "he_l1_nonlinear_1em08"
    )
    result_path = next(
        Path(value) for value in row["expected_outputs"] if value.endswith(".json")
    )
    payload = evidence.read_strict_json(result_path)
    convergence = payload["result"]["steps"][0]["convergence"]
    convergence["dual_residual_metadata"]["effective_rtol"] = 1.0e-6
    _write_json(result_path, payload)

    receipt_path = root / "receipts/he_l1_nonlinear_1em08.json"
    receipt = evidence.read_strict_json(receipt_path)
    receipt["output_hashes"][str(result_path)] = evidence.sha256_file(result_path)
    _write_json(receipt_path, receipt)

    with pytest.raises(evidence.AdmissionError, match="tolerances differ"):
        evidence.audit_campaign(root, repo_root=repo)


def test_frozen_admission_design_matches_the_actual_plan_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(
        producer, "_git_metadata", lambda: {"commit": commit, "dirty": False}
    )
    output_root = (tmp_path / "producer-output").resolve()
    plan = producer.build_plan(
        output_root,
        run_kind="publication",
        allow_dirty=False,
        p4_policy="local",
        confirm_p4_local_feasible=True,
    )
    design = evidence.expected_design()
    expected_rows = [
        evidence.expected_plan_row(
            row_id,
            spec,
            evidence_root=output_root,
            python=str(Path(sys.executable).absolute()),
        )
        for row_id, spec in design.items()
    ]
    assert plan["rows"] == expected_rows
    assert set(plan["source"]["relevant_file_hashes"]) == set(
        evidence.EXPECTED_SOURCE_PATHS
    )
    assert set(plan["inputs"]["file_hashes"]) == set(evidence.EXPECTED_INPUT_PATHS)
    assert evidence._validate_manifested_meshes(
        plan["inputs"]["manifested_file_hashes"],
        repo_root=Path(__file__).resolve().parents[1],
        input_inventory=plan["inputs"]["file_hashes"],
    ) == plan["inputs"]["manifested_file_hashes"]


def test_plan_admission_rejects_substitute_inventory_commands_and_semantics(
    tmp_path: Path,
) -> None:
    repo, root, _commit = _fixture(tmp_path)
    plan_path = root / evidence.PLAN_NAME
    canonical = evidence.read_strict_json(plan_path)

    mutated = copy.deepcopy(canonical)
    source_hashes = mutated["source"]["relevant_file_hashes"]
    source_hashes.pop(evidence.EXPECTED_SOURCE_PATHS[0])
    substitute = "paper/scripts/stopping_local_evidence.py"
    source_hashes[substitute] = evidence.sha256_file(repo / substitute)
    _write_json(plan_path, mutated)
    with pytest.raises(evidence.AdmissionError, match="exact canonical set"):
        evidence.audit_campaign(root, repo_root=repo)

    mutated = copy.deepcopy(canonical)
    mutated["inputs"]["file_hashes"].pop(evidence.EXPECTED_INPUT_PATHS[-1])
    _write_json(plan_path, mutated)
    with pytest.raises(evidence.AdmissionError, match="exact canonical set"):
        evidence.audit_campaign(root, repo_root=repo)

    mutated = copy.deepcopy(canonical)
    first_mesh = next(iter(evidence.EXPECTED_MANIFESTED_MESH_PATHS))
    mutated["inputs"]["manifested_file_hashes"][first_mesh]["sha256"] = "0" * 64
    _write_json(plan_path, mutated)
    with pytest.raises(evidence.AdmissionError, match="manifested-file bindings"):
        evidence.audit_campaign(root, repo_root=repo)

    mutated = copy.deepcopy(canonical)
    mutated["rows"][0]["command"][1] = "source.py"
    _write_json(plan_path, mutated)
    with pytest.raises(evidence.AdmissionError, match="full row parameters"):
        evidence.audit_campaign(root, repo_root=repo)

    mutated = copy.deepcopy(canonical)
    mutated["rows"][0]["parameters"]["globalization"] = "substitute-policy"
    _write_json(plan_path, mutated)
    with pytest.raises(evidence.AdmissionError, match="full row parameters"):
        evidence.audit_campaign(root, repo_root=repo)

    mutated = copy.deepcopy(canonical)
    mutated["rows"][0]["scientific_scope"] = "overclaimed-scope"
    _write_json(plan_path, mutated)
    with pytest.raises(evidence.AdmissionError, match="full row parameters"):
        evidence.audit_campaign(root, repo_root=repo)

    mutated = copy.deepcopy(canonical)
    mutated["rows"][-1]["censor"]["reason"] = "silently changed censor"
    _write_json(plan_path, mutated)
    with pytest.raises(evidence.AdmissionError, match="censor semantics"):
        evidence.audit_campaign(root, repo_root=repo)


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

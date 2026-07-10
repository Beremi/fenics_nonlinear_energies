from __future__ import annotations

from dataclasses import replace
import copy
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "paper/scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import admit_revision_publication_evidence as admission  # noqa: E402
import check_revision_evidence_manifest as final_checker  # noqa: E402


PYTHON = REPO_ROOT / ".venv/bin/python"
SCRIPT = SCRIPT_DIR / "admit_revision_publication_evidence.py"
GENERATOR = SCRIPT_DIR / "generate_revision_evidence_tables.py"
CHECKER = SCRIPT_DIR / "check_revision_evidence_manifest.py"
PILOT_ROOT = REPO_ROOT / "artifacts/reproduction/paper_revision_2026_07_10/pilots"


def _source_payload(spec: admission.EvidenceSpec, *, commit: str, producer_hash: str) -> dict:
    payload: dict = {
        "status": spec.terminal_statuses[0],
        "publication_evidence": True,
        "provenance": {
            "command": f"python {spec.producer_path.as_posix()}",
            "python": "3.12.10",
            "runner_sha256": producer_hash,
            "git": {"commit": commit, "dirty": False},
        },
    }
    if spec.family == "manufactured_scalar":
        payload.update(
            {
                "levels": [{"status": "converged", "final_relative_residual": 0.0}],
                "rates": [{"l2_rate": 2.0, "h1_seminorm_rate": 1.0}],
            }
        )
    elif spec.family == "affine_patch":
        payload.update({"contract": {"relative_tolerance": 1.0e-10}, "metrics": {"defect": 0.0}})
    elif spec.family == "hyperelastic_nonaffine":
        payload.update({"gates": {"all": True}, "levels": [{"status": "converged"}]})
    elif spec.family == "derivative":
        payload.update(
            {
                "contract": {
                    "route_relative_tolerance": 1.0e-8,
                    "symmetry_tolerance": 1.0e-8,
                    "centered_fd_tolerance": 1.0e-8,
                },
                "summary": {
                    "maximum_hessian_relative_error": 0.0,
                    "maximum_hessian_symmetry_defect": 0.0,
                    "maximum_fd_hvp_error_at_gate": 0.0,
                },
            }
        )
    elif spec.family == "material_point":
        payload["summary"] = {
            "cpu_fp64_execution_passed": True,
            "degeneracy_finiteness_checks_passed": True,
            "interface_sweeps_passed": True,
            "interior_checks_passed": True,
            "rotation_checks_passed": True,
        }
    elif spec.family == "distribution":
        payload["comparison"] = {
            "algebraic_gate_passed": True,
            "derivative_gates": {"all": True},
            "exact_object_gates": {"all": True},
            "exact_topology_gates": {"all": True},
            "linear_solve_gates": {"all": True},
        }
    elif spec.family == "quadrature":
        payload.update(
            {
                "common_free_dof_set": True,
                "solve_quadrature_rule_id": "q1",
                "reference_rule_id": "q2",
                "evaluations": [
                    {"quadrature_rule_id": "q1"},
                    {"quadrature_rule_id": "q2"},
                ],
            }
        )
    elif spec.family == "route_analysis":
        contract = REPO_ROOT / "paper/protocols/EXP-ROUTE-001-analysis-contract.json"
        payload.pop("status")
        payload.update(
            {
                "terminal_decision": "predictive_selector_admissible",
                "empirical_map": [{"status": "admitted"}],
                "cost_model": {"selector_claim_admissible": True},
                "factorized_microbenchmark_gate": {
                    "passed": True,
                    "calibration_integrated": False,
                    "selector_use": "descriptive_replicated_synthetic_non_route_faithful_proxy",
                },
                "contract_path": str(contract),
                "contract_sha256": admission.sha256_file(contract),
            }
        )
    return payload


def _clean_negative_route_payload() -> dict:
    spec = next(
        spec for spec in admission.EVIDENCE_SPECS if spec.key == "route_analysis"
    )
    contract_path = REPO_ROOT / "paper/protocols/EXP-ROUTE-001-analysis-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    slots, censors = admission._route_expected_slots(
        contract, {"workstation_local", "karolina_cpu"}
    )
    rows: list[dict] = []
    for slot in sorted(slots):
        hardware, configuration, state, ranks, route = slot
        split = (
            "training"
            if hardware == "workstation_local"
            or ranks in contract["hardware"]["karolina_cpu"]["training_ranks"]
            else "holdout"
        )
        if slot in censors:
            reason = "prespecified_not_attempted_memory_risk_no_threshold_claim"
            row = {
                "hardware_id": hardware,
                "configuration_id": configuration,
                "state_id": state,
                "rank_count": ranks,
                "route": route,
                "split": split,
                "status": "censored",
                "reason": reason,
                "publication_model_eligible": False,
                "model_exclusion_reason": reason,
            }
            for field in (
                "admitted_wall_time_median_s",
                "paired_block_medians_s",
                "paired_block_repetitions",
                "paired_block_route_positions",
                "model_covariates",
                "action_relative_l2_error",
                "action_relative_l2_errors",
                "action_max_absolute_error",
                "gradient_residual_relative_error",
            ):
                row[field] = None
        else:
            row = {
                "hardware_id": hardware,
                "configuration_id": configuration,
                "state_id": state,
                "rank_count": ranks,
                "route": route,
                "split": split,
                "status": "admitted",
                "reason": "all_equivalence_gates_passed",
                "publication_model_eligible": True,
                "model_exclusion_reason": "",
                "admitted_wall_time_median_s": 1.0,
                "action_relative_l2_error": 0.0,
                "action_relative_l2_errors": [0.0, 0.0, 0.0, 0.0],
                "action_max_absolute_error": 0.0,
                "gradient_residual_relative_error": 0.0,
                "state_sha256": "1" * 64,
                "action_sha256": "2" * 64,
                "source_commit": "a" * 40,
                "model_covariates": {"route_work_proxy": 1.0},
            }
        rows.append(row)
    training = sum(
        row.get("publication_model_eligible") is True and row["split"] == "training"
        for row in rows
    )
    holdout = sum(
        row.get("publication_model_eligible") is True and row["split"] == "holdout"
        for row in rows
    )
    assert (training, holdout) == (74, 22)
    return {
        "analysis_schema_version": 1,
        "experiment_id": "EXP-ROUTE-001",
        "contract_path": "paper/protocols/EXP-ROUTE-001-analysis-contract.json",
        "contract_sha256": admission.sha256_file(contract_path),
        "sources": [
            {
                "hardware_id": hardware,
                "publication_provenance_gate": {
                    "eligible": True,
                    "source_commit": "a" * 40,
                },
            }
            for hardware in ("workstation_local", "karolina_cpu")
        ],
        "terminal_decision": "finite_empirical_map_only",
        "empirical_map": rows,
        "cost_model": {
            "status": "fit_gate_failed",
            "selector_claim_admissible": False,
            "feature_order": contract["cost_model"]["features_in_order"],
            "training_rows": training,
            "holdout_rows": holdout,
            "preflight_failures": [],
            "failed_gates": ["median_absolute_percentage_error"],
        },
        "endpoint_analysis": {
            "schema_version": 1,
            "terminal_decision": "tier_b_descriptive_timing_only",
            "comparative_ranking_admissible": False,
            "publication_admissible": True,
            "required_rows": 30,
            "admitted_rows": 30,
            "path": "EXP-ROUTE-001/analysis_contract_v1/tier_b_endpoint_analysis.json",
            "sha256": "3" * 64,
        },
        "factorized_microbenchmark_gate": {
            "passed": False,
            "failures": ["factorized calibration holdout gates failed"],
            "calibration_integrated": False,
            "selector_use": "descriptive_replicated_synthetic_non_route_faithful_proxy",
            "selector_blockers": [],
            "required_ranks": [1, 8, 32],
            "independent_blocks_per_rank": 3,
            "calibration_model": None,
        },
        "invalid_records": [],
        "provenance": {"git": {"commit": "a" * 40, "dirty": False}},
        "source_schema": {
            "id": "fenics-nonlinear-energies.revision-source.route_analysis",
            "version": 1,
        },
        "publication_evidence": True,
        "run_kind": "publication",
        "experiment_commit": "a" * 40,
        "publication_provenance": {
            "schema_id": "fenics-nonlinear-energies.revision-publication-source-provenance",
            "schema_version": 1,
            "run_kind": "publication",
            "experiment_commit": "a" * 40,
            "producer": {
                "path": spec.producer_path.as_posix(),
                "sha256": admission.sha256_file(REPO_ROOT / spec.producer_path),
            },
        },
    }


def _clean_synthetic_tree(tmp_path: Path, monkeypatch) -> tuple[Path, tuple[admission.EvidenceSpec, ...], str]:
    commit = "a" * 40
    evidence_root = tmp_path / "evidence"
    specs = tuple(replace(spec, run_records=()) for spec in admission.EVIDENCE_SPECS)
    grouped: dict[Path, list[admission.EvidenceSpec]] = {}
    for spec in specs:
        producer_hash = admission.sha256_file(REPO_ROOT / spec.producer_path)
        payload = _source_payload(spec, commit=commit, producer_hash=producer_hash)
        output = evidence_root / spec.relative_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        grouped.setdefault(spec.companion_manifest, []).append(spec)

    for relative, group in grouped.items():
        companion = evidence_root / relative
        companion.parent.mkdir(parents=True, exist_ok=True)
        companion_payload = {
            "run_kind": "publication",
            "publication_evidence": True,
            "git_commit": commit,
            "git_clean": True,
            "command_template": "python PRODUCER --output OUTPUT",
            "environment": {"python": "3.12.10"},
            "code_hashes": {
                spec.producer_path.as_posix(): admission.sha256_file(REPO_ROOT / spec.producer_path)
                for spec in group
            },
            "output_hashes": {
                spec.relative_path.as_posix(): admission.sha256_file(evidence_root / spec.relative_path)
                for spec in group
            },
        }
        companion.write_text(json.dumps(companion_payload, indent=2) + "\n", encoding="utf-8")

    monkeypatch.setattr(admission, "EVIDENCE_SPECS", specs)
    monkeypatch.setattr(
        admission,
        "_git_metadata",
        lambda _root: {"commit": commit, "worktree_clean": True},
    )
    return evidence_root, specs, commit


def test_admission_contract_configures_exactly_fourteen_distinct_inputs() -> None:
    specs = admission.EVIDENCE_SPECS
    assert len(specs) == 14
    assert len({spec.key for spec in specs}) == 14
    assert len({spec.relative_path for spec in specs}) == 14


def test_actual_shaped_numerical_payloads_pass_family_semantics_before_publication_decoration() -> None:
    for spec in admission.EVIDENCE_SPECS:
        if spec.family == "route_analysis":
            continue
        payload = json.loads((PILOT_ROOT / spec.relative_path).read_text(encoding="utf-8"))
        payload["source_schema"] = {
            "id": f"fenics-nonlinear-energies.revision-source.{spec.key}",
            "version": 1,
        }
        payload["publication_provenance"] = {
            "schema_id": "fenics-nonlinear-energies.revision-publication-source-provenance",
            "schema_version": 1,
            "run_kind": "publication",
            "experiment_commit": "a" * 40,
            "producer": {
                "path": spec.producer_path.as_posix(),
                "sha256": admission.sha256_file(REPO_ROOT / spec.producer_path),
            },
        }
        errors = admission._semantic_gate_errors(spec, payload)
        assert errors == [], f"{spec.key}: {errors}"


def test_complete_clean_finite_map_negative_result_is_publication_admissible() -> None:
    spec = next(
        spec for spec in admission.EVIDENCE_SPECS if spec.key == "route_analysis"
    )
    payload = _clean_negative_route_payload()
    assert admission._semantic_gate_errors(spec, payload) == []
    assert payload["factorized_microbenchmark_gate"]["passed"] is False
    assert payload["cost_model"]["training_rows"] == 74
    assert payload["cost_model"]["holdout_rows"] == 22


def test_finite_map_branch_rejects_relabeling_and_predictive_leakage() -> None:
    spec = next(
        spec for spec in admission.EVIDENCE_SPECS if spec.key == "route_analysis"
    )
    for alias in (
        "finite_map_only",
        "empirical_map_only",
        "tier_b_descriptive_timing_only",
    ):
        payload = _clean_negative_route_payload()
        payload["terminal_decision"] = alias
        errors = admission._semantic_gate_errors(spec, payload)
        assert any("terminal" in error for error in errors), alias

    relabeled = _clean_negative_route_payload()
    relabeled["terminal_decision"] = "predictive_selector_admissible"
    errors = admission._semantic_gate_errors(spec, relabeled)
    assert any("selection_rule_passed" in error for error in errors)

    for mutation in ("coefficients", "holdout_ordering"):
        payload = _clean_negative_route_payload()
        payload["cost_model"][mutation] = {} if mutation == "coefficients" else []
        errors = admission._semantic_gate_errors(spec, payload)
        assert any("nonpredictive decision fields" in error for error in errors)

    nested = _clean_negative_route_payload()
    active = next(
        row for row in nested["empirical_map"] if row["status"] == "admitted"
    )
    active["model_covariates"]["predicted_winner"] = "element_ad"
    errors = admission._semantic_gate_errors(spec, nested)
    assert any("leaks predictive fields" in error for error in errors)

    crossover = _clean_negative_route_payload()
    crossover["post_fit_confirmation"] = {
        "publication_admissible": True,
        "terminal_decision": "post_fit_crossover_confirmed",
    }
    errors = admission._semantic_gate_errors(spec, crossover)
    assert any("post_fit_confirmation is uncontracted" in error for error in errors)


def test_finite_map_branch_rejects_incomplete_or_incoherent_negative_data() -> None:
    spec = next(
        spec for spec in admission.EVIDENCE_SPECS if spec.key == "route_analysis"
    )
    insufficient = _clean_negative_route_payload()
    insufficient["cost_model"].update(
        {
            "status": "not_fit_insufficient_data",
            "training_rows": 0,
            "holdout_rows": 0,
            "preflight_failures": ["insufficient_training_rows"],
            "failed_gates": [],
        }
    )
    errors = admission._semantic_gate_errors(spec, insufficient)
    assert any("admitted empirical rows" in error for error in errors)
    assert any("no admissible negative terminal status" in error for error in errors)

    inconsistent = _clean_negative_route_payload()
    inconsistent["cost_model"]["preflight_failures"] = ["arbitrary_failure"]
    errors = admission._semantic_gate_errors(spec, inconsistent)
    assert any("fit_gate_failed must contain only" in error for error in errors)

    memory_predictor = _clean_negative_route_payload()
    memory_predictor["cost_model"]["feature_order"] = [
        *memory_predictor["cost_model"]["feature_order"],
        "peak_rank_rss_bytes",
    ]
    errors = admission._semantic_gate_errors(spec, memory_predictor)
    assert any("feature_order differs" in error for error in errors)


def test_factor_outcome_is_reportable_but_factor_integrity_is_required() -> None:
    spec = next(
        spec for spec in admission.EVIDENCE_SPECS if spec.key == "route_analysis"
    )
    passed = _clean_negative_route_payload()
    passed["factorized_microbenchmark_gate"].update(
        {"passed": True, "failures": [], "calibration_model": {"status": "passed"}}
    )
    assert admission._semantic_gate_errors(spec, passed) == []

    inconsistent = _clean_negative_route_payload()
    inconsistent["factorized_microbenchmark_gate"].update(
        {"passed": True, "failures": ["failed"], "calibration_model": None}
    )
    errors = admission._semantic_gate_errors(spec, inconsistent)
    assert any("internally inconsistent" in error for error in errors)

    integrated = _clean_negative_route_payload()
    integrated["factorized_microbenchmark_gate"]["calibration_integrated"] = True
    errors = admission._semantic_gate_errors(spec, integrated)
    assert any("non-integrated" in error for error in errors)


def test_endpoint_terminal_and_comparative_flag_must_agree() -> None:
    spec = next(
        spec for spec in admission.EVIDENCE_SPECS if spec.key == "route_analysis"
    )
    for terminal, comparative in (
        ("tier_b_comparative_ranking_admissible", False),
        ("tier_b_descriptive_timing_only", True),
    ):
        payload = _clean_negative_route_payload()
        payload["endpoint_analysis"]["terminal_decision"] = terminal
        payload["endpoint_analysis"]["comparative_ranking_admissible"] = comparative
        errors = admission._semantic_gate_errors(spec, payload)
        assert any("comparative-ranking flag" in error for error in errors)


def test_vacuous_or_self_loosened_family_payloads_are_rejected() -> None:
    by_key = {spec.key: spec for spec in admission.EVIDENCE_SPECS}
    cases = {
        "plaplace": {
            "status": "passed",
            "levels": [{"status": "converged"}],
            "rates": [{}],
        },
        "hyperelastic_patch": {
            "status": "passed",
            "contract": {"relative_tolerance": 1.0e100},
            "metrics": {"defect": -1.0e100},
        },
        "hyperelastic_nonaffine": {
            "status": "passed",
            "gates": {"anything": True},
            "levels": [{"status": "converged"}],
        },
        "p1_derivatives": {
            "status": "passed",
            "contract": {
                "route_relative_tolerance": 1.0e100,
                "symmetry_tolerance": 1.0e100,
                "centered_fd_tolerance": 1.0e100,
                "centered_fd_gate_index": 2,
                "centered_fd_gate_step": 1.0e-7,
            },
            "summary": {
                "maximum_residual_relative_error": -1.0,
                "maximum_hessian_relative_error": -1.0,
                "maximum_hessian_symmetry_defect": -1.0,
            },
        },
        "material_point": {
            "status": "passed",
            "summary": {
                "cpu_fp64_execution_passed": True,
                "degeneracy_finiteness_checks_passed": True,
                "interface_sweeps_passed": True,
                "interior_checks_passed": True,
                "rotation_checks_passed": True,
            },
        },
        "distribution": {
            "status": "passed",
            "comparison": {
                "algebraic_gate_passed": True,
                "derivative_gates": {"anything": True},
                "exact_object_gates": {"anything": True},
                "exact_topology_gates": {"anything": True},
                "linear_solve_gates": {"anything": True},
            },
        },
        "p2_quadrature": {
            "status": "completed",
            "common_free_dof_set": True,
            "solve_quadrature_rule_id": "q1",
            "reference_rule_id": "q2",
            "evaluations": [
                {"quadrature_rule_id": "q1"},
                {"quadrature_rule_id": "q2"},
            ],
        },
        "route_analysis": {
            "terminal_decision": "predictive_selector_admissible",
            "empirical_map": [{"status": "admitted"}],
            "cost_model": {"selector_claim_admissible": True},
            "factorized_microbenchmark_gate": {
                "passed": True,
                "calibration_integrated": False,
                "selector_use": "descriptive_replicated_synthetic_non_route_faithful_proxy",
            },
        },
    }
    for key, payload in cases.items():
        decorated = copy.deepcopy(payload)
        decorated["source_schema"] = {
            "id": f"fenics-nonlinear-energies.revision-source.{key}",
            "version": 1,
        }
        spec = by_key[key]
        decorated["publication_provenance"] = {
            "schema_id": "fenics-nonlinear-energies.revision-publication-source-provenance",
            "schema_version": 1,
            "run_kind": "publication",
            "experiment_commit": "a" * 40,
            "producer": {
                "path": spec.producer_path.as_posix(),
                "sha256": admission.sha256_file(REPO_ROOT / spec.producer_path),
            },
        }
        errors = admission._semantic_gate_errors(spec, decorated)
        assert errors, key
        assert any(
            marker in "; ".join(errors)
            for marker in (
                "level",
                "metrics",
                "contract",
                "summary",
                "comparison",
                "quadrature",
                "route",
                "Tier-B",
            )
        ), (key, errors)


def test_negative_metric_and_inflated_tolerance_are_rejected_on_actual_shapes() -> None:
    spec = next(spec for spec in admission.EVIDENCE_SPECS if spec.key == "p1_derivatives")
    payload = json.loads((PILOT_ROOT / spec.relative_path).read_text(encoding="utf-8"))
    payload["source_schema"] = {
        "id": f"fenics-nonlinear-energies.revision-source.{spec.key}",
        "version": 1,
    }
    payload["publication_provenance"] = {
        "schema_id": "fenics-nonlinear-energies.revision-publication-source-provenance",
        "schema_version": 1,
        "run_kind": "publication",
        "experiment_commit": "a" * 40,
        "producer": {
            "path": spec.producer_path.as_posix(),
            "sha256": admission.sha256_file(REPO_ROOT / spec.producer_path),
        },
    }
    payload["contract"]["centered_fd_tolerance"] = 1.0e100
    payload["summary"]["maximum_hessian_relative_error"] = -1.0
    errors = admission._semantic_gate_errors(spec, payload)
    assert any("frozen value" in error for error in errors)
    assert any("must be at least 0.0" in error for error in errors)


def test_current_pilots_are_independently_blocked() -> None:
    audit = admission.audit_evidence(PILOT_ROOT)

    assert audit["eligible"] is False
    assert audit["configured_input_count"] == 14
    assert set(audit["inputs"]) == {spec.key for spec in admission.EVIDENCE_SPECS}
    assert all(row["admitted"] is False for row in audit["inputs"].values())
    assert "payload publication_evidence=False" in audit["inputs"]["plaplace"]["blockers"]
    assert any(
        "publication ingestion boundary" in blocker
        for blocker in audit["inputs"]["material_point"]["blockers"]
    )
    assert any(
        "finite map" in blocker or "finite-map-only" in blocker
        for blocker in audit["inputs"]["route_analysis"]["blockers"]
    )


def test_admit_mode_writes_no_manifest_when_any_source_is_blocked(tmp_path: Path) -> None:
    destination = tmp_path / "must-not-exist.json"
    result = subprocess.run(
        [
            str(PYTHON),
            str(SCRIPT),
            "admit",
            "--evidence-root",
            str(PILOT_ROOT),
            "--manifest-out",
            str(destination),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Admission refused" in result.stderr
    assert not destination.exists()


def test_clean_semantic_audit_can_create_and_deeply_revalidate_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    audit = admission.audit_evidence(PILOT_ROOT)
    audit["eligible"] = True
    audit["experiment_commit"] = "a" * 40
    audit["git"] = {"commit": "b" * 40, "worktree_clean": True}
    manifest = admission.build_publication_manifest(audit)
    assert manifest["schema_version"] == 2
    assert manifest["publication_evidence"] is True
    assert manifest["experiment_commit"] == "a" * 40
    assert manifest["admission_head"] == "b" * 40


def test_deep_revalidation_rejects_tampered_input(tmp_path: Path, monkeypatch) -> None:
    manifest = {
        "schema_id": admission.SCHEMA_ID,
        "schema_version": admission.SCHEMA_VERSION,
        "publication_evidence": True,
        "status": admission.ADMITTED_STATUS,
        "worktree_clean": True,
        "evidence_root": str(PILOT_ROOT.relative_to(REPO_ROOT)),
        "experiment_commit": "a" * 40,
        "git_commit": "a" * 40,
        "git": {"experiment_commit": "a" * 40, "worktree_clean": True},
        "inputs": {},
    }
    path = tmp_path / "publication_evidence_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    try:
        admission.validate_publication_source_manifest(
            path,
            evidence_root=PILOT_ROOT,
            repo_root=REPO_ROOT,
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("tampered publication input was accepted")
    assert "fresh independent source audit is blocked" in message
    assert "payload publication_evidence=False" in message


def test_table_generator_rejects_handwritten_admission_flags(tmp_path: Path) -> None:
    forged = tmp_path / "forged-source-manifest.json"
    forged.write_text(
        json.dumps(
            {
                "schema_id": admission.SCHEMA_ID,
                "schema_version": admission.SCHEMA_VERSION,
                "publication_evidence": True,
                "status": admission.ADMITTED_STATUS,
                "worktree_clean": True,
                "git_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=REPO_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "evidence_root": str(PILOT_ROOT.relative_to(REPO_ROOT)),
                "inputs": {
                    spec.key: {
                        "path": spec.relative_path.as_posix(),
                        "sha256": admission.sha256_file(PILOT_ROOT / spec.relative_path),
                        "admitted": True,
                        "checks": [],
                    }
                    for spec in admission.EVIDENCE_SPECS
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    generated = subprocess.run(
        [
            str(PYTHON),
            str(GENERATOR),
            "--out-dir",
            str(tmp_path / "tables"),
            "--evidence-root",
            str(PILOT_ROOT),
            "--evidence-class",
            "publication",
            "--evidence-manifest",
            str(forged),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert generated.returncode != 0
    assert "fresh independent source audit is blocked" in generated.stderr
    assert not (tmp_path / "tables/revision_evidence_manifest.json").exists()


def test_final_checker_deeply_revalidates_source_manifest(tmp_path: Path) -> None:
    table_dir = tmp_path / "tables"
    subprocess.run(
        [str(PYTHON), str(GENERATOR), "--out-dir", str(table_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    fake_repo = tmp_path / "repo"
    canonical_dir = fake_repo / "paper/tables/generated"
    evidence_dir = fake_repo / "evidence"
    generator_dir = fake_repo / "paper/scripts"
    canonical_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    generator_dir.mkdir(parents=True)
    (generator_dir / GENERATOR.name).write_bytes(GENERATOR.read_bytes())
    forged = evidence_dir / "forged-source-manifest.json"
    forged.write_text("{}\n", encoding="utf-8")
    table_manifest_path = table_dir / "revision_evidence_manifest.json"
    table_manifest = json.loads(table_manifest_path.read_text(encoding="utf-8"))
    table_manifest.update(
        {
            "evidence_class": "publication",
            "publication_evidence": True,
            "status": "clean_publication_tables",
            "source_evidence_manifest": {
                "path": "evidence/forged-source-manifest.json",
                "sha256": admission.sha256_file(forged),
                "schema_id": admission.SCHEMA_ID,
            },
        }
    )
    table_manifest["evidence_root"] = "evidence"
    table_manifest["git"]["worktree_clean"] = True
    canonical_manifest = canonical_dir / "revision_evidence_manifest.json"
    canonical_manifest.write_text(
        json.dumps(table_manifest, indent=2) + "\n", encoding="utf-8"
    )
    for name in table_manifest["outputs"]:
        (canonical_dir / name).write_bytes((table_dir / name).read_bytes())

    checked = subprocess.run(
        [
            str(PYTHON),
            str(CHECKER),
            "--manifest",
            str(canonical_manifest),
            "--repo-root",
            str(fake_repo),
            "--expect-diagnostic",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert checked.returncode == 0
    assert "source evidence admission failed deep revalidation" in checked.stdout


def test_final_checker_rejects_missing_extra_and_escaping_table_paths(tmp_path: Path) -> None:
    table_dir = tmp_path / "tables"
    subprocess.run(
        [str(PYTHON), str(GENERATOR), "--out-dir", str(table_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest_path = table_dir / "revision_evidence_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["inputs"].pop("plaplace")
    payload["inputs"]["p1_derivatives"]["path"] = "../../etc/passwd"
    payload["outputs"].pop("revision_verification_summary.tex")
    payload["outputs"]["../../unrelated.tex"] = "0" * 64
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    errors = final_checker.validate_revision_evidence_manifest(
        manifest_path,
        repo_root=REPO_ROOT,
        require_clean_worktree=False,
    )
    assert "input hash map must contain exactly the 14 configured input keys" in errors
    assert "output hash map must contain exactly the four manuscript revision tables" in errors
    assert any("must not be absolute, non-canonical, or contain '..'" in error for error in errors)


def test_final_checker_binds_exact_manuscript_consumed_output_names(tmp_path: Path) -> None:
    table_dir = tmp_path / "tables"
    subprocess.run(
        [str(PYTHON), str(GENERATOR), "--out-dir", str(table_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest_path = table_dir / "revision_evidence_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_hash = payload["outputs"].pop("revision_evidence_status.tex")
    unrelated = tmp_path / "unrelated.tex"
    unrelated.write_text("unrelated\n", encoding="utf-8")
    payload["outputs"][str(unrelated)] = original_hash
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    errors = final_checker.validate_revision_evidence_manifest(
        manifest_path,
        repo_root=REPO_ROOT,
        require_clean_worktree=False,
    )
    assert "output hash map must contain exactly the four manuscript revision tables" in errors


def test_final_checker_regenerates_and_byte_compares_canonical_tables(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    table_dir = repo / "paper/tables/generated"
    table_dir.mkdir(parents=True)
    generator = repo / "paper/scripts/generate_revision_evidence_tables.py"
    generator.parent.mkdir(parents=True)
    generator.write_text(
        """#!/usr/bin/env python3
import argparse
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--out-dir', type=Path, required=True)
p.add_argument('--evidence-root')
p.add_argument('--evidence-class')
p.add_argument('--evidence-manifest')
a = p.parse_args()
a.out_dir.mkdir(parents=True, exist_ok=True)
for name in (
    'revision_verification_summary.tex',
    'revision_derivative_checks.tex',
    'revision_quadrature_sensitivity.tex',
    'revision_evidence_status.tex',
):
    (a.out_dir / name).write_text('generated:' + name + '\\n', encoding='utf-8')
""",
        encoding="utf-8",
    )
    evidence_root = repo / "artifacts/evidence"
    source_manifest = evidence_root / "publication_evidence_manifest.json"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text("{}\n", encoding="utf-8")
    source_rows: dict[str, dict[str, str]] = {}
    table_inputs: dict[str, dict[str, str]] = {}
    for spec in admission.EVIDENCE_SPECS:
        path = evidence_root / spec.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{spec.key}\n", encoding="utf-8")
        digest = admission.sha256_file(path)
        source_rows[spec.key] = {"sha256": digest}
        table_inputs[spec.key] = {
            "path": path.relative_to(repo).as_posix(),
            "path_within_evidence_root": spec.relative_path.as_posix(),
            "sha256": digest,
        }
    outputs: dict[str, str] = {}
    for name in final_checker.EXPECTED_OUTPUTS:
        path = table_dir / name
        path.write_text(f"generated:{name}\n", encoding="utf-8")
        outputs[name] = admission.sha256_file(path)
    for name, (tex_relative, literal) in final_checker.MANUSCRIPT_INPUT_BINDINGS.items():
        tex = repo / tex_relative
        tex.parent.mkdir(parents=True, exist_ok=True)
        existing = tex.read_text(encoding="utf-8") if tex.exists() else ""
        tex.write_text(existing + literal + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "evidence_class": "publication",
        "publication_evidence": True,
        "status": "clean_publication_tables",
        "generator": "paper/scripts/generate_revision_evidence_tables.py",
        "generator_sha256": admission.sha256_file(generator),
        "evidence_root": "artifacts/evidence",
        "git": {"commit": "a" * 40, "worktree_clean": True},
        "source_evidence_manifest": {
            "path": source_manifest.relative_to(repo).as_posix(),
            "sha256": admission.sha256_file(source_manifest),
        },
        "inputs": table_inputs,
        "outputs": outputs,
    }
    manifest_path = table_dir / "revision_evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(final_checker, "_git_metadata", lambda _root: ("b" * 40, True))
    monkeypatch.setattr(final_checker, "_git_is_ancestor", lambda *_args: True)
    monkeypatch.setattr(
        final_checker,
        "validate_publication_source_manifest",
        lambda *_args, **_kwargs: {"inputs": source_rows},
    )

    assert final_checker.validate_revision_evidence_manifest(
        manifest_path, repo_root=repo
    ) == []
    (table_dir / "revision_evidence_status.tex").write_text(
        "Predictive selector admitted; crossover admitted\n", encoding="utf-8"
    )
    manifest["outputs"]["revision_evidence_status.tex"] = admission.sha256_file(
        table_dir / "revision_evidence_status.tex"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    errors = final_checker.validate_revision_evidence_manifest(
        manifest_path, repo_root=repo
    )
    assert "regenerated output revision_evidence_status.tex differs byte-for-byte" in errors

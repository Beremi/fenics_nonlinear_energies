from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np
import pytest

from experiments.analysis import finalize_revision_publication_campaign as finalizer
from src.core.benchmark.run_record import RUN_RECORD_SCHEMA_ID, RUN_RECORD_SCHEMA_VERSION


COMMIT = "a" * 40
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper/scripts"))


def _assembled_derivative_block(degree: int) -> dict:
    route_names = ("element_ad", "local_sfd", "constitutive_ad")
    pairs = (
        ("element_ad", "local_sfd"),
        ("element_ad", "constitutive_ad"),
        ("local_sfd", "constitutive_ad"),
    )
    return {
        "status": "passed",
        "case": {
            "mesh_name": "hetero_ssr_L1",
            "degree": degree,
            "constraint_variant": "glued_bottom",
            "lambda_target": 1.5,
            "free_dofs": 3,
            "elements": 1,
            "state_definition": "deterministic test state",
            "state_scale": 1.0e-8,
            "state_norm": 1.0e-8,
        },
        "contract": {
            "value_atol": 1.0e-12,
            "value_rtol": 1.0e-12,
            "gradient_norm_atol": 1.0e-10,
            "gradient_norm_rtol": 1.0e-12,
            "hessian_maximum_entry_atol": 1.0e-8,
            "hessian_frobenius_rtol": 1.0e-12,
            "hessian_symmetry_tolerance": 1.0e-12,
            "branch_gate": "every quadrature point must satisfy trial_yield < 0",
        },
        "branch_diagnostics": {
            "quadrature_points": 1,
            "elastic_quadrature_points": 1,
            "plastic_quadrature_points": 0,
            "minimum_trial_yield": -1.0,
            "maximum_trial_yield": -1.0,
            "minimum_normalized_elastic_margin": 1.0,
            "all_quadrature_points_strictly_elastic": True,
            "interpretation": "test fixture",
        },
        "routes": {
            name: {
                "energy": 1.0,
                "gradient_norm": 1.0,
                "hessian_frobenius_norm": 1.0,
                "hessian_symmetry_defect": 0.0,
                "assembly_mode": f"{name}_test",
                "hessian_nonzeros": 1,
            }
            for name in route_names
        },
        "pairwise_comparisons": [
            {
                "left": left,
                "right": right,
                "energy_absolute_error": 0.0,
                "energy_relative_error": 0.0,
                "gradient_absolute_error": 0.0,
                "gradient_relative_error": 0.0,
                "hessian_csr_structure_equal": True,
                "hessian_absolute_error": 0.0,
                "hessian_relative_error": 0.0,
                "hessian_maximum_entry_error": 0.0,
                "passed": True,
            }
            for left, right in pairs
        ],
        "all_values_finite": True,
        "all_hessians_symmetric_within_tolerance": True,
        "algebraic_scope": {
            "linear_solver_called": False,
            "nonlinear_solver_called": False,
            "ksp_tolerance_used_for_comparison": None,
            "local_sfd_meaning": "exact JVP fixture",
            "interpretation": "fixed-state algebraic comparison fixture",
        },
    }


def _raw_payload(spec: finalizer.SourceSpec, *, commit: str = COMMIT) -> dict:
    payload: dict = {
        "experiment_id": spec.experiment_id,
        "status": "completed" if "quadrature" in spec.key else "passed",
        "provenance": {"git": {"commit": commit, "dirty": False}},
    }
    if spec.key == "distribution":
        payload.pop("experiment_id")
        payload.update(
            {
                "experiment": spec.experiment_id,
                "run_kind": "publication",
                "schema": {
                    "id": "fenics-nonlinear-energies.exp-dist-he-comparison",
                    "version": 1,
                },
            }
        )
    elif spec.key == "material_point":
        payload.update(
            {
                "schema_name": "plasticity3d_material_point_verification",
                "schema_version": 1,
            }
        )
    elif spec.native_version_field is not None:
        payload[spec.native_version_field] = spec.native_version
    if spec.key in {
        "plaplace",
        "ginzburg_landau",
        "hyperelastic_patch",
        "hyperelastic_nonaffine",
    }:
        # These producer schemas conservatively mark raw output as not yet
        # admitted.  The managed clean receipt, not this flag alone, is what
        # permits creation of the separate decorated copy.
        payload["publication_evidence"] = False
    return payload


def _scientific_raw_payload(spec: finalizer.SourceSpec, *, commit: str) -> dict:
    payload = _raw_payload(spec, commit=commit)
    if spec.key in {"plaplace", "ginzburg_landau"}:
        levels = []
        for subdivisions in (8, 16, 32, 64):
            row = {
                "status": "converged",
                "subdivisions": subdivisions,
                "h": 1.0 / subdivisions,
                "l2_error": 1.0 / subdivisions**2,
                "h1_seminorm_error": 1.0 / subdivisions,
                "final_relative_residual": 1.0e-12,
                "tangent_symmetry_defect": 0.0,
            }
            if spec.key == "plaplace":
                row["minimum_element_gradient_norm"] = 0.75
            else:
                row["minimum_nodal_value"] = 0.8
            levels.append(row)
        payload.update(
            {
                "solver_contract": {
                    "relative_residual_tolerance": 1.0e-9,
                    "maximum_iterations": 20,
                },
                "levels": levels,
                "rates": [
                    {
                        "coarse_subdivisions": coarse,
                        "fine_subdivisions": fine,
                        "l2_rate": 2.0,
                        "h1_seminorm_rate": 1.0,
                    }
                    for coarse, fine in ((8, 16), (16, 32), (32, 64))
                ],
            }
        )
        if spec.key == "plaplace":
            payload["acceptance_contract"] = {
                "minimum_last_l2_rate": 1.75,
                "minimum_last_h1_seminorm_rate": 0.85,
                "maximum_symmetry_defect": 1.0e-12,
                "minimum_discrete_gradient_norm": 0.5,
            }
    elif spec.key == "hyperelastic_patch":
        metric_names = {
            "energy_relative_error",
            "residual_relative_error",
            "hessian_relative_error",
            "hessian_symmetry_defect",
            "traction_balance_relative_error",
            "net_internal_force_norm",
            "objectivity_energy_relative_error",
            "piola_rotation_covariance_relative_error",
        }
        payload.update(
            {
                "contract": {"relative_tolerance": 2.0e-11},
                "metrics": {
                    **{name: 0.0 for name in metric_names},
                    "translation_mode_hessian_action_norms": [0.0, 0.0, 0.0],
                },
                "case": {"determinant": 1.0},
            }
        )
    elif spec.key == "hyperelastic_nonaffine":
        gate_names = {
            "algebraic_residual",
            "all_levels_converged",
            "h1_rate",
            "l2_rate",
            "load_quadrature_below_consistency_error",
            "load_quadrature_below_fe_error",
            "load_quadrature_reference_stable",
            "load_quadrature_refined_solves_converged",
            "load_quadrature_refined_solves_resolve_load_change",
            "orientation",
            "stress_rate",
            "tangent_symmetry",
        }
        payload.update(
            {
                "contract": {
                    "subdivisions": [4, 8, 16, 24],
                    "load_quadrature_orders": [4, 6, 8],
                    "relative_algebraic_residual": 1.0e-10,
                    "tangent_symmetry_tolerance": 1.0e-11,
                    "minimum_determinant": 0.5,
                    "maximum_load_quadrature_error_fraction": 0.01,
                    "last_pair_minimum_rates": {
                        "first_piola_l2": 0.75,
                        "h1_deformation": 0.75,
                        "l2_displacement": 1.75,
                    },
                },
                "gates": {name: True for name in gate_names},
                "levels": [
                    {
                        "subdivisions": n,
                        "status": "converged",
                        "l2_displacement_error": 1.0 / n**2,
                        "h1_deformation_error": 1.0 / n,
                        "first_piola_l2_error": 1.0 / n,
                        "final_relative_residual": 1.0e-12,
                        "minimum_discrete_determinant": 0.9,
                        "tangent_symmetry_defect": 0.0,
                        "load_quadrature_check": {
                            "below_fe_error": True,
                            "reference_load_stable": True,
                            "refined_solution_status": "converged",
                            "refined_solution_resolves_load_change": True,
                            "maximum_fraction_of_fe_error": 0.001,
                        },
                    }
                    for n in (4, 8, 16, 24)
                ],
                "rates": [
                    {
                        "l2_displacement_error": 2.0,
                        "h1_deformation_error": 1.0,
                        "first_piola_l2_error": 1.0,
                    }
                    for _ in range(3)
                ],
            }
        )
    elif "derivatives" in spec.key:
        smooth = spec.key == "smooth_derivatives"
        payload["contract"] = {
            "route_relative_tolerance": 1.0e-10 if smooth else 1.0e-9,
            "symmetry_tolerance": 1.0e-12 if smooth else 1.0e-10,
            "centered_fd_tolerance": 1.0e-7,
            "centered_fd_gate_index": 3 if smooth else 2,
            "centered_fd_gate_step": 3.0e-5 if smooth else 1.0e-7,
        }
        if smooth:
            payload["summary"] = {
                "cases": 5,
                "maximum_gradient_relative_error": 0.0,
                "maximum_hessian_relative_error": 0.0,
                "maximum_hessian_symmetry_defect": 0.0,
                "maximum_fd_gradient_error_at_gate": 0.0,
                "maximum_fd_hvp_error_at_gate": 0.0,
            }
        else:
            degree = {"p1_derivatives": 1, "p2_derivatives": 2, "p4_derivatives": 4}[
                spec.key
            ]
            payload["case"] = {"mesh_name": "hetero_ssr_L1", "degree": degree}
            payload["summary"] = {
                "states": 5,
                "maximum_residual_relative_error": 0.0,
                "maximum_hessian_relative_error": 0.0,
                "maximum_hessian_symmetry_defect": 0.0,
                "maximum_centered_fd_energy_error_at_gate": 0.0,
                "maximum_centered_fd_hvp_error_at_gate": 0.0,
                "all_states_branch_stable_at_fd_gate": True,
                "fixed_element_status": "passed",
                "assembled_route_equivalence_status": "passed",
            }
            payload["assembled_route_equivalence"] = _assembled_derivative_block(degree)
        payload["records"] = [{} for _ in range(5)]
    elif spec.key == "material_point":
        branches = ["elastic", "shear", "left_edge", "right_edge", "apex"]
        payload["contract"] = {
            "centered_fd_scaled_error_tolerance": 1.0e-7,
            "hessian_symmetry_tolerance": 1.0e-10,
            "numpy_energy_transcription_relative_tolerance": 1.0e-12,
            "rotation_absolute_tolerance_for_near_zero_tangent_actions": 1.0e-9,
            "rotation_scaled_tolerance": 1.0e-9,
            "minimum_normalized_active_branch_margin": 1.0e-3,
            "required_branches": branches,
        }
        payload["summary"] = {
            "cpu_fp64_execution_passed": True,
            "degeneracy_finiteness_checks_passed": True,
            "interface_sweeps_passed": True,
            "interior_checks_passed": True,
            "rotation_checks_passed": True,
            "branch_interior_counts": {name: 1 for name in branches},
            "degeneracy_case_count": 7,
            "interface_count": 5,
            "interface_pair_count": 15,
            "rotation_check_count": 15,
            "maximum_centered_energy_directional_error_at_gate": 0.0,
            "maximum_centered_hvp_error_at_gate": 0.0,
            "maximum_hessian_symmetry_defect": 0.0,
            "maximum_numpy_energy_transcription_relative_error": 0.0,
            "maximum_interface_numpy_energy_transcription_relative_error": 0.0,
            "maximum_degeneracy_numpy_energy_transcription_relative_error": 0.0,
            "maximum_rotation_energy_invariance_scaled_error": 0.0,
            "maximum_rotation_stress_covariance_scaled_error": 0.0,
            "maximum_rotation_tangent_action_covariance_scaled_error": 0.0,
            "maximum_rotation_tangent_action_covariance_absolute_error": 0.0,
            "minimum_normalized_active_branch_margin": 0.1,
        }
    elif spec.key == "distribution":
        comparison = {
            "derivative_gates": {
                key: True
                for key in (
                    "energy_relative",
                    "matrix_action_relative",
                    "matrix_relative",
                    "residual_relative",
                )
            },
            "exact_object_gates": {
                key: True for key in ("direction", "matrix_indices", "matrix_indptr", "state")
            },
            "exact_topology_gates": {
                key: True
                for key in (
                    "affine_lift",
                    "connectivity",
                    "coordinates",
                    "freedofs",
                    "right_boundary_nodes",
                )
            },
            "linear_solve_gates": {
                key: True
                for key in (
                    "candidate_true_residual",
                    "linear_correction",
                    "reference_true_residual",
                )
            },
            "algebraic_gate_passed": True,
            "derivative_tolerance": 1.0e-8,
            "solve_tolerance": 1.0e-8,
            "relative_errors": {
                "energy_relative": 0.0,
                "matrix_action_relative": 0.0,
                "matrix_relative": 0.0,
                "residual_relative": 0.0,
                "linear_correction_relative": 0.0,
            },
        }
        payload.update(
            {
                "controlled_factors": {
                    "problem": "hyperelasticity",
                    "mesh_source": "procedural",
                    "problem_build_mode": "rank_local",
                    "distribution_strategy": "overlap_p2p",
                    "assembly_backend": "coo_local",
                    "local_hessian_mode": "element",
                    "element_reorder_mode": "block_xyz",
                    "element_degree": 1,
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "factor_solver_type": "mumps",
                    "use_near_nullspace": False,
                },
                "varied_factor": {"name": "mpi_ranks", "levels": [1, 2, 4]},
                "comparison": comparison,
                "rank_comparisons": {"np2": comparison, "np4": comparison},
                "workers": {
                    "np1": {"status": "passed"},
                    "np2": {"status": "passed"},
                    "np4": {"status": "passed"},
                },
            }
        )
    elif "quadrature" in spec.key:
        degree = {"p1_quadrature": 1, "p2_quadrature": 2, "p4_quadrature": 4}[spec.key]
        solve = {1: "tetra_1point", 2: "tetra_11point", 4: "tetra_24point"}[degree]
        rules = {
            "tetra_1point": 1,
            "tetra_11point": 11,
            "tetra_24point": 24,
            "tetra_duffy_125point": 125,
        }
        payload.update(
            {
                "element_degree": degree,
                "solve_quadrature_rule_id": solve,
                "reference_rule_id": "tetra_duffy_125point",
                "constraint_variant": "glued_bottom",
                "mesh_name": "hetero_ssr_L1",
                "lambda_target": 1.55,
                "reference_energy_scale": 1.0,
                "common_free_dof_set": True,
                "common_direction_content_sha256": "d" * 64,
                "evaluations": [
                    {
                        "quadrature_rule_id": rule,
                        "quadrature_points_per_element": points,
                        "element_degree": degree,
                        "elements": 1,
                        "degrees_of_freedom": 1,
                        "free_degrees_of_freedom": 1,
                        "full_residual_l2_norm": 0.0,
                        "free_residual_l2_norm": 0.0,
                        "full_hessian_action_l2_norm": 0.0,
                        "free_hessian_action_l2_norm": 0.0,
                        "minimum_normalized_active_branch_margin": 0.1,
                        "minimum_normalized_constitutive_denominator": 0.1,
                        "relative_total_potential_difference_from_last_rule": 0.0,
                        "free_hessian_action_vector_comparison_to_last_rule": {
                            "absolute_l2_difference": 0.0,
                            "relative_l2_difference": 0.0,
                            "absolute_linf_difference": 0.0,
                        },
                        "branch_point_fractions": {
                            "elastic": 1.0,
                            "shear": 0.0,
                            "left_edge": 0.0,
                            "right_edge": 0.0,
                            "apex": 0.0,
                        },
                    }
                    for rule, points in rules.items()
                ],
            }
        )
    return payload


def _write_quadrature_artifact_fixture(
    spec: finalizer.SourceSpec,
    payload: dict,
    *,
    staging_root: Path,
) -> None:
    degree = finalizer.QUADRATURE_DEGREES[spec.key]
    by_rule = {row["quadrature_rule_id"]: row for row in payload["evaluations"]}
    for rule in finalizer.QUADRATURE_RULE_IDS:
        row = by_rule[rule]
        values = {
            "hessian_action_artifact": np.asarray([degree, 1.0], dtype=np.float64),
            "residual_artifact": np.asarray([degree, 2.0], dtype=np.float64),
            "branch_map_artifact": np.asarray([0, 1], dtype=np.int8),
        }
        for field, array in values.items():
            suffix, content_field, _dtype = finalizer.QUADRATURE_ARTIFACT_FIELDS[field]
            relative = Path(
                f"EXP-DISC-001/actions/p{degree}_l1/{rule}_{suffix}.npy"
            )
            path = staging_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, array, allow_pickle=False)
            content_digest = finalizer._array_content_sha256(array)
            row[content_field] = content_digest
            row[field] = {
                "path": relative.relative_to(spec.relative_path.parent).as_posix(),
                "sha256": finalizer.sha256_file(path),
                "content_sha256": content_digest,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "content": "test fixture",
            }


def _receipt(spec: finalizer.SourceSpec, raw_hash: str) -> dict:
    return {
        "command": {"argv_template": ["python", spec.producer_path.as_posix()]},
        "environment": {"python": "3.12", "packages": {"numpy": "test"}},
        "producer": {"path": spec.producer_path.as_posix(), "sha256": "b" * 64},
        "configuration_hashes": {},
        "input_hashes": {},
        "raw_output_hashes": {
            (Path(finalizer.STAGING_DIRECTORY) / spec.relative_path).as_posix(): raw_hash
        },
        "referenced_artifact_hashes": {},
        "artifact_validation_errors": [],
    }


def test_source_contract_matches_the_14_admission_inputs() -> None:
    from paper.scripts import admit_revision_publication_evidence as admission

    producer_specs = {
        spec.key: (
            spec.relative_path,
            spec.producer_path,
            spec.companion_manifest,
            spec.run_records,
        )
        for spec in finalizer.SOURCE_SPECS
    }
    admission_specs = {
        spec.key: (
            spec.relative_path,
            spec.producer_path,
            spec.companion_manifest,
            spec.run_records,
        )
        for spec in admission.EVIDENCE_SPECS
    }
    assert producer_specs == admission_specs
    assert len(producer_specs) == 14


def test_canonical_template_covers_every_source_and_clean_dependency() -> None:
    plan = finalizer.build_execution_plan_template(experiment_commit=COMMIT)
    commands = finalizer._plan_command_map(plan)
    assert plan["plan_kind"] == "source_campaign"
    assert len(commands) == 17
    assert {
        key for command in commands.values() for key in command["source_keys"]
    } == set(finalizer.SOURCE_BY_KEY)
    material = commands["mc_material_point"]
    distribution = commands["dist_hyperelastic"]
    assert ["--run-kind", "publication"] == material["argv"][2:4]
    assert ["--run-kind", "publication"] == distribution["argv"][2:4]
    for degree in (1, 2, 4):
        preparation = commands[f"prepare_p{degree}_l1_state"]
        assert preparation["source_keys"] == []
        assert preparation["role"] == "preparation"
        assert preparation["producer"].endswith(
            "prepare_plasticity3d_fixed_state.py"
        )
        assert {
            f"EXP-DISC-001/clean_inputs/p{degree}_l1_state.npz",
            f"EXP-DISC-001/clean_inputs/p{degree}_l1_state_manifest.json",
        } == set(preparation["expected_artifacts"])
        derivative_argv = commands[f"deriv_p{degree}"]["argv"]
        assert derivative_argv.count("--assembled-route-equivalence") == 1
        derivative_input = commands[f"deriv_p{degree}"]["input_files"][0]
        assert derivative_input == {
            "scope": "repo_manifested",
            "path": (
                "data/meshes/SlopeStability3D/hetero_ssr/"
                f"hetero_ssr_L1_p{degree}_same_mesh_glued_bottom.h5"
            ),
            "manifest": finalizer.MANIFESTED_MESH_MANIFEST.as_posix(),
        }
        inputs = commands[f"disc_p{degree}"]["input_files"]
        assert inputs[0]["scope"] == "staging"
        assert inputs[0]["attestation"]["path"].endswith(f"p{degree}_l1_state.json")
        assert set(commands[f"disc_p{degree}"]["expected_artifacts"]) == {
            path.as_posix()
            for path in finalizer._quadrature_expected_artifacts(degree)
        }
    route = commands["route_cost_analysis"]
    assert route["route_endpoint_analysis"].endswith("tier_b_endpoint_analysis.json")
    assert len(route["input_files"]) == 3


def test_expand_argv_preserves_virtualenv_python_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_python = tmp_path / "base/bin/python3"
    base_python.parent.mkdir(parents=True)
    base_python.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_python = tmp_path / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)
    monkeypatch.setattr(finalizer.sys, "executable", str(venv_python))

    expanded = finalizer._expand_argv(
        ["{python}"],
        repo_root=tmp_path,
        evidence_root=tmp_path / "evidence",
        staging_root=tmp_path / "evidence/staging",
    )

    assert expanded == [str(venv_python.absolute())]
    assert expanded != [str(venv_python.resolve())]


def test_plan_rejects_missing_or_escaping_quadrature_artifact_declarations() -> None:
    plan = finalizer.build_execution_plan_template(experiment_commit=COMMIT)
    command = next(row for row in plan["commands"] if row["id"] == "disc_p1")
    command["expected_artifacts"].pop()
    with pytest.raises(finalizer.FinalizationError, match="omits publication-critical"):
        finalizer._plan_command_map(plan)

    plan = finalizer.build_execution_plan_template(experiment_commit=COMMIT)
    command = next(row for row in plan["commands"] if row["id"] == "disc_p1")
    command["expected_artifacts"].append("../escaped.npy")
    with pytest.raises(finalizer.FinalizationError, match="canonical relative path"):
        finalizer._plan_command_map(plan)


def test_recursive_quadrature_artifact_contract_rejects_tampering_and_escape(
    tmp_path: Path,
) -> None:
    spec = finalizer.SOURCE_BY_KEY["p1_quadrature"]
    payload = _scientific_raw_payload(spec, commit=COMMIT)
    _write_quadrature_artifact_fixture(spec, payload, staging_root=tmp_path)
    hashes = finalizer._quadrature_referenced_artifact_hashes(
        spec,
        payload,
        staging_root=tmp_path,
    )
    assert set(hashes) == {
        (Path(finalizer.STAGING_DIRECTORY) / path).as_posix()
        for path in finalizer._quadrature_expected_artifacts(1)
    }

    first = tmp_path / finalizer._quadrature_expected_artifacts(1)[0]
    first.write_bytes(first.read_bytes() + b"tamper")
    with pytest.raises(finalizer.FinalizationError, match="SHA-256 mismatch"):
        finalizer._quadrature_referenced_artifact_hashes(
            spec,
            payload,
            staging_root=tmp_path,
        )

    escaped = deepcopy(payload)
    escaped["evaluations"][0]["hessian_action_artifact"]["path"] = "../escape.npy"
    with pytest.raises(finalizer.FinalizationError, match="canonical relative path"):
        finalizer._quadrature_referenced_artifact_hashes(
            spec,
            escaped,
            staging_root=tmp_path,
        )


@pytest.mark.parametrize("spec", finalizer.SOURCE_SPECS, ids=lambda spec: spec.key)
def test_actual_shaped_raw_sources_receive_versioned_clean_provenance(
    tmp_path: Path, spec: finalizer.SourceSpec
) -> None:
    raw = _raw_payload(spec)
    finalizer.validate_raw_source_payload(spec, raw, experiment_commit=COMMIT)
    raw_path = tmp_path / finalizer.STAGING_DIRECTORY / spec.relative_path
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    receipt_path = tmp_path / finalizer.RECEIPT_DIRECTORY / f"{spec.key}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text("{}\n", encoding="utf-8")
    receipt = _receipt(spec, finalizer.sha256_file(raw_path))

    decorated = finalizer._decorate_payload(
        spec,
        raw,
        receipt=receipt,
        receipt_path=receipt_path,
        evidence_root=tmp_path,
        experiment_commit=COMMIT,
    )

    assert decorated["publication_evidence"] is True
    assert decorated["run_kind"] == "publication"
    assert decorated["experiment_commit"] == COMMIT
    assert decorated["source_schema"] == {
        "id": f"fenics-nonlinear-energies.revision-source.{spec.key}",
        "version": 1,
    }
    provenance = decorated["publication_provenance"]
    assert provenance["git_clean"] is True
    assert provenance["git"] == {"commit": COMMIT, "worktree_clean": True}
    assert provenance["raw_output"]["sha256"] == finalizer.sha256_file(raw_path)


def test_dirty_pilot_and_already_decorated_sources_are_never_relabelled() -> None:
    spec = finalizer.SOURCE_BY_KEY["plaplace"]
    pilot = _raw_payload(spec)
    pilot["run_kind"] = "pilot"
    with pytest.raises(finalizer.FinalizationError, match="dirty/pilot"):
        finalizer.validate_raw_source_payload(spec, pilot, experiment_commit=COMMIT)

    dirty = _raw_payload(spec)
    dirty["provenance"]["git"]["dirty"] = True
    with pytest.raises(finalizer.FinalizationError, match="dirty worktree"):
        finalizer.validate_raw_source_payload(spec, dirty, experiment_commit=COMMIT)

    decorated = _raw_payload(spec)
    decorated["source_schema"] = {"id": "forged", "version": 1}
    with pytest.raises(finalizer.FinalizationError, match="already decorated"):
        finalizer.validate_raw_source_payload(spec, decorated, experiment_commit=COMMIT)


def test_strict_run_record_identity_rejects_cross_experiment_substitution() -> None:
    spec = finalizer.SOURCE_BY_KEY["distribution"]
    assert spec.run_records == (
        Path("EXP-DIST-001/run_record_np1.json"),
        Path("EXP-DIST-001/run_record_np2.json"),
        Path("EXP-DIST-001/run_record_np4.json"),
    )
    relative = spec.run_records[0]
    record = _run_record(
        spec=spec,
        relative=relative,
        commit=COMMIT,
        producer=spec.producer_path,
        producer_hash="b" * 64,
        record_id="distribution-1",
    )
    finalizer._validate_run_record_identity(spec, relative, record)
    substituted = deepcopy(record)
    substituted["identifiers"]["experiment"] = "EXP-MC-001"
    with pytest.raises(finalizer.FinalizationError, match="not 'EXP-DIST-001'"):
        finalizer._validate_run_record_identity(spec, relative, substituted)

    np4_relative = spec.run_records[-1]
    np4 = _run_record(
        spec=spec,
        relative=np4_relative,
        commit=COMMIT,
        producer=spec.producer_path,
        producer_hash="b" * 64,
        record_id="distribution-4",
    )
    finalizer._validate_run_record_identity(spec, np4_relative, np4)
    drifted = deepcopy(np4)
    drifted["solver"]["parameters"]["canonical_twist_angle_rad"] = 0.2
    with pytest.raises(finalizer.FinalizationError, match="frozen EXP-DIST-001"):
        finalizer._validate_run_record_identity(spec, np4_relative, drifted)


@pytest.mark.parametrize("path", ["../escape.json", "/absolute.json", "a/./b.json", "a//b.json"])
def test_path_confinement_rejects_noncanonical_paths(tmp_path: Path, path: str) -> None:
    with pytest.raises(finalizer.FinalizationError):
        finalizer._confined(tmp_path, path, label="test")


def test_manifested_generated_mesh_is_hash_and_generator_bound(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        check=True,
    )
    (repo / ".gitignore").write_text("*.h5\n", encoding="utf-8")
    for relative in finalizer.MANIFESTED_MESH_GENERATOR_SOURCES:
        source = repo / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"source {relative.as_posix()}\n", encoding="utf-8")
    records: dict[str, dict[str, object]] = {}
    for relative, degree in finalizer.MANIFESTED_MESH_PATHS.items():
        mesh = repo / relative
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_bytes(f"generated mesh degree {degree}\n".encode())
        records[relative.as_posix()] = {
            "bytes": mesh.stat().st_size,
            "constraint_variant": "glued_bottom",
            "element_degree": degree,
            "mesh_name": "hetero_ssr_L1",
            "same_mesh_hdf5_schema_version": 7,
            "sha256": finalizer.sha256_file(mesh),
        }
    manifest = repo / finalizer.MANIFESTED_MESH_MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "files": records,
                "generator": {
                    "function": (
                        "src.problems.slope_stability_3d.support.mesh."
                        "ensure_same_mesh_case_hdf5"
                    ),
                    "tracked_sources": [
                        path.as_posix()
                        for path in finalizer.MANIFESTED_MESH_GENERATOR_SOURCES
                    ],
                },
                "schema_id": (
                    "fenics-nonlinear-energies.manifested-generated-meshes"
                ),
                "schema_version": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "manifest"], check=True
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    relative = next(iter(finalizer.MANIFESTED_MESH_PATHS))

    actual, bindings = finalizer._manifested_repo_input_hashes(
        relative,
        finalizer.MANIFESTED_MESH_MANIFEST,
        repo_root=repo,
        experiment_commit=commit,
    )
    assert actual == records[relative.as_posix()]["sha256"]
    assert finalizer.MANIFESTED_MESH_MANIFEST.as_posix() in bindings
    assert {
        path.as_posix() for path in finalizer.MANIFESTED_MESH_GENERATOR_SOURCES
    } <= set(bindings)

    (repo / relative).write_bytes(b"tampered\n")
    with pytest.raises(finalizer.FinalizationError, match="missing or stale"):
        finalizer._manifested_repo_input_hashes(
            relative,
            finalizer.MANIFESTED_MESH_MANIFEST,
            repo_root=repo,
            experiment_commit=commit,
        )


@pytest.mark.parametrize(
    ("terminal", "comparative"),
    [
        ("tier_b_comparative_ranking_admissible", False),
        ("tier_b_descriptive_timing_only", True),
    ],
)
def test_finalizer_rejects_endpoint_terminal_boolean_mismatch(
    tmp_path: Path, terminal: str, comparative: bool
) -> None:
    relative = Path("route/endpoint.json")
    path = tmp_path / finalizer.STAGING_DIRECTORY / relative
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": {
                    "id": "fenics-nonlinear-energies.exp-route-001.tier-b-endpoints",
                    "version": 1,
                },
                "experiment_id": "EXP-ROUTE-001",
                "terminal_decision": terminal,
                "comparative_ranking_admissible": comparative,
                "endpoint_correct_timing_admissible": True,
                "matrix_policy_violations": [],
                "coverage_and_campaign_failure_reasons": [],
                "blocks": [{"status": "timing_admitted"} for _ in range(30)],
                "structural_censors": [
                    {
                        "status": "censored",
                        "reason": "prespecified_not_attempted_memory_risk_no_threshold_claim",
                        "route": "colored_sfd",
                        "timing_exposed": False,
                        "admitted_collective_max_wall_time_s": None,
                    }
                    for _ in range(2)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(finalizer.FinalizationError, match="not publication-admissible"):
        finalizer._route_endpoint_summary(
            command={"route_endpoint_analysis": relative.as_posix()},
            evidence_root=tmp_path,
        )


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _initialize_fake_repo(path: Path) -> str:
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test User"], check=True)
    (path / ".gitignore").write_text("/artifacts/\n", encoding="utf-8")
    for spec in finalizer.SOURCE_SPECS:
        producer = path / spec.producer_path
        producer.parent.mkdir(parents=True, exist_ok=True)
        producer.write_text(f"# producer for {spec.key}\n", encoding="utf-8")
    finalizer_copy = path / finalizer.FINALIZER_PATH
    finalizer_copy.parent.mkdir(parents=True, exist_ok=True)
    finalizer_copy.write_bytes(Path(finalizer.__file__).read_bytes())
    contract = path / "paper/protocols/EXP-ROUTE-001-analysis-contract.json"
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_bytes(
        (finalizer.REPO_ROOT / "paper/protocols/EXP-ROUTE-001-analysis-contract.json").read_bytes()
    )
    generator = path / "paper/scripts/generate_revision_evidence_tables.py"
    generator.parent.mkdir(parents=True, exist_ok=True)
    generator.write_bytes(
        (finalizer.REPO_ROOT / "paper/scripts/generate_revision_evidence_tables.py").read_bytes()
    )
    admission_tool = path / "paper/scripts/admit_revision_publication_evidence.py"
    admission_tool.write_bytes(
        (finalizer.REPO_ROOT / "paper/scripts/admit_revision_publication_evidence.py").read_bytes()
    )
    checker = path / "paper/scripts/check_revision_evidence_manifest.py"
    checker.write_bytes(
        (finalizer.REPO_ROOT / "paper/scripts/check_revision_evidence_manifest.py").read_bytes()
    )
    run_record = path / "src/core/benchmark/run_record.py"
    run_record.parent.mkdir(parents=True, exist_ok=True)
    run_record.write_bytes(
        (finalizer.REPO_ROOT / "src/core/benchmark/run_record.py").read_bytes()
    )
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "experiment"], check=True, capture_output=True)
    return _git(path, "rev-parse", "HEAD")


def _minimal_plan(commit: str) -> dict:
    commands = []
    for spec in finalizer.SOURCE_SPECS:
        command = {
            "id": f"produce_{spec.key}",
            "source_keys": [spec.key],
            "producer": spec.producer_path.as_posix(),
            "argv": ["python", spec.producer_path.as_posix()],
            "environment": {},
            "configuration_files": ["paper/protocols/EXP-ROUTE-001-analysis-contract.json"],
            "input_files": [],
            "expected_artifacts": [],
        }
        if spec.key in finalizer.QUADRATURE_DEGREES:
            command["expected_artifacts"] = [
                path.as_posix()
                for path in finalizer._quadrature_expected_artifacts(
                    finalizer.QUADRATURE_DEGREES[spec.key]
                )
            ]
        if spec.key == "route_analysis":
            command["route_endpoint_analysis"] = "route/endpoint.json"
            command["input_files"] = [{"scope": "staging", "path": "route/endpoint.json"}]
        commands.append(command)
    return {
        "schema_id": finalizer.PLAN_SCHEMA_ID,
        "schema_version": finalizer.PLAN_SCHEMA_VERSION,
        "plan_kind": "source_campaign",
        "campaign_id": "test-clean-campaign",
        "experiment_commit": commit,
        "commands": commands,
    }


def test_execute_captures_exact_argv_environment_and_rejects_dirty_tree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _initialize_fake_repo(repo)
    spec = finalizer.SOURCE_BY_KEY["plaplace"]
    producer = repo / spec.producer_path
    producer.write_text(
        "import json, pathlib, sys\n"
        "p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)\n"
        "p.write_text(json.dumps({'experiment_id':'EXP-VAL-001-PLAPLACE-MANUFACTURED','schema_version':1,'status':'passed','publication_evidence':False})+'\\n')\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(repo), "add", spec.producer_path.as_posix()], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "runnable producer"], check=True, capture_output=True)
    commit = _git(repo, "rev-parse", "HEAD")
    plan = _minimal_plan(commit)
    command = next(row for row in plan["commands"] if row["source_keys"] == ["plaplace"])
    command["argv"] = [
        sys.executable,
        spec.producer_path.as_posix(),
        f"{{staging_root}}/{spec.relative_path.as_posix()}",
    ]
    command["environment"] = {"OMP_NUM_THREADS": "1", "JAX_PLATFORMS": "cpu"}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
    evidence = repo / "artifacts/evidence"

    receipt_path = finalizer.execute_plan_command(
        plan_path=plan_path,
        command_id=command["id"],
        evidence_root=evidence,
        repo_root=repo,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"
    assert receipt["command"]["argv_template"] == command["argv"]
    assert receipt["command"]["return_code"] == 0
    assert receipt["environment"]["selected_variables"]["OMP_NUM_THREADS"] == "1"
    assert receipt["environment"]["selected_variables"]["JAX_PLATFORMS"] == "cpu"
    assert receipt["producer"]["sha256"] == finalizer.sha256_file(producer)

    (repo / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")
    with pytest.raises(Exception, match="Publication experiments require an empty git status"):
        finalizer.execute_plan_command(
            plan_path=plan_path,
            command_id="produce_ginzburg_landau",
            evidence_root=evidence,
            repo_root=repo,
        )


def _run_record(
    *,
    spec: finalizer.SourceSpec,
    relative: Path,
    commit: str,
    producer: Path,
    producer_hash: str,
    record_id: str,
) -> dict:
    if spec.key == "material_point":
        experiment = "EXP-MC-001"
        case = "dimensionless-five-branch-material-point-matrix"
        method = "jax-scalar-autodiff"
        route = "production-mohr-coulomb-scalar"
        ranks = 1
        parameters = {}
    else:
        ranks = int(relative.stem.removeprefix("run_record_np"))
        experiment = "EXP-DIST-001"
        case = f"hyperelasticity-p1-l1-np{ranks}"
        method = "fixed-state-distributed-equivalence"
        route = "rank-local-procedural-p2p-local-coo"
        parameters = {
            "problem": "hyperelasticity",
            "mesh_source": "procedural",
            "problem_build_mode": "rank_local",
            "distribution_strategy": "overlap_p2p",
            "assembly_backend": "coo_local",
            "local_hessian_mode": "element",
            "element_reorder_mode": "block_xyz",
            "element_degree": 1,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "factor_solver_type": "mumps",
            "use_near_nullspace": False,
            "mesh_level": 1,
            "canonical_twist_angle_rad": 0.15,
            "repetitions": 3,
            "ksp_rtol": 1.0e-12,
            "linear_residual_tolerance": 1.0e-10,
            "residual_scale_floor": 1.0,
        }
    return {
        "schema": {"id": RUN_RECORD_SCHEMA_ID, "version": RUN_RECORD_SCHEMA_VERSION},
        "record_id": record_id,
        "run_kind": "publication",
        "identifiers": {
            "campaign": "clean",
            "experiment": experiment,
            "case": case,
            "method": method,
            "route": route,
            "repetition": 1,
        },
        "problem": {
            "name": "test",
            "mesh": "unit",
            "degree": 1,
            "quadrature": "exact",
            "total_degrees_of_freedom": 1,
            "free_degrees_of_freedom": 1,
            "notes": "contract fixture",
        },
        "solver": {
            "algorithm": "direct",
            "implementation": "fixture",
            "parameters": parameters,
            "preconditioner": {},
            "stopping_contract": "fixture-v1",
        },
        "termination": {
            "status": "success",
            "reason": "gates passed",
            "exit_code": 0,
            "started_at_utc": "2026-07-10T08:00:00Z",
            "finished_at_utc": "2026-07-10T08:00:01Z",
            "limit_kind": None,
            "limit_value": None,
            "censored": False,
        },
        "accuracy": {
            "contract_id": "fixture-v1",
            "gate_passed": True,
            "absolute_residual": 0.0,
            "relative_residual": 0.0,
            "scaled_residual": 0.0,
            "relative_correction": 0.0,
            "energy_change": 0.0,
            "custom_metrics": {},
            "notes": "exact fixture",
        },
        "counts": {
            "nonlinear_iterations": 1,
            "krylov_iterations": 0,
            "function_evaluations": 1,
            "gradient_evaluations": 1,
            "hessian_evaluations": 0,
            "hvp_evaluations": 0,
            "preconditioner_setups": 0,
            "notes": "fixture counts",
        },
        "timing": {
            "aggregation": "rank-maximum",
            "cold_process": False,
            "barrier_policy": "single rank",
            "synchronization_policy": "synchronous",
            "phases_overlap": False,
            "relation_to_total": "all phases contained",
            "process_startup_s": 0.0,
            "jit_compilation_s": 0.0,
            "coloring_s": 0.0,
            "derivative_evaluation_s": 0.1,
            "constitutive_contraction_s": 0.0,
            "assembly_s": 0.0,
            "communication_s": 0.0,
            "preconditioner_setup_s": 0.0,
            "krylov_solve_s": 0.0,
            "globalization_s": 0.0,
            "state_output_s": 0.0,
            "total_s": 0.1,
            "notes": "fixture timing",
        },
        "resources": {
            "nodes": 1,
            "ranks": ranks,
            "threads_per_rank": 1,
            "peak_memory_per_rank_bytes": 1,
            "peak_memory_per_node_bytes": 1,
            "tracked_allocations_bytes": 1,
            "measurement_method": "fixture",
            "notes": "fixture resources",
        },
        "diagnostics": {"state": {}, "branch": {}, "feasibility": {}, "kkt": {}},
        "environment": {
            "python": "3.12",
            "packages": {"numpy": "test"},
            "platform": "Linux",
            "jax": "not-applicable",
            "xla": "not-applicable",
            "jax_enable_x64": True,
            "petsc": "not-applicable",
            "mpi": "not-applicable",
            "compiler": "test",
            "blas": "test",
            "cpu_model": "test",
            "node_model": "test",
            "memory_model": "test",
            "scheduler": "local",
            "scheduler_job_id": None,
            "affinity": "one thread",
        },
        "provenance": {
            "git_commit": commit,
            "git_clean": True,
            "git_status_porcelain": [],
            "pilot_override": False,
            "pilot_override_reason": None,
            "command_argv": ["python", producer.as_posix()],
            "working_directory": ".",
            "code_hashes": {producer.as_posix(): producer_hash},
            "configuration_hashes": {},
            "input_hashes": {},
            "dirty_patch_sha256": None,
            "seed": 1,
            "deterministic_policy": "fixed fixture",
            "recorded_at_utc": "2026-07-10T08:00:02Z",
        },
        "artifacts": {
            "raw_outputs": [spec.relative_path.as_posix()],
            "states": [],
            "logs": [],
            "tables": [],
            "figures": [],
            "reports": [],
        },
    }


def _write_receipt(
    *,
    path: Path,
    command: dict,
    plan: dict,
    plan_path: Path,
    repo: Path,
    evidence: Path,
    commit: str,
) -> None:
    producer = repo / command["producer"]
    required = finalizer._required_raw_paths(command)
    raw_hashes = {
        (Path(finalizer.STAGING_DIRECTORY) / relative).as_posix(): finalizer.sha256_file(
            evidence / finalizer.STAGING_DIRECTORY / relative
        )
        for relative in required
    }
    receipt = {
        "schema_id": finalizer.RECEIPT_SCHEMA_ID,
        "schema_version": finalizer.RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "campaign_id": plan["campaign_id"],
        "command_id": command["id"],
        "source_keys": command["source_keys"],
        "experiment_commit": commit,
        "preflight": {
            "git_commit": commit,
            "git_clean": True,
            "git_status_porcelain": [],
            "pilot_override": False,
            "checked_at_utc": "2026-07-10T08:00:00Z",
        },
        "postflight": {"git_commit": commit, "git_clean": True},
        "command": {
            "argv_template": command["argv"],
            "argv": command["argv"],
            "working_directory": ".",
            "return_code": 0,
            "execution_error": None,
            "started_at_utc": "2026-07-10T08:00:00Z",
            "finished_at_utc": "2026-07-10T08:00:01Z",
        },
        "environment": {"python": "3.12", "packages": {"numpy": "test"}},
        "plan": {"path": plan_path.as_posix(), "sha256": finalizer.sha256_file(plan_path)},
        "producer": {
            "path": command["producer"],
            "sha256": finalizer.sha256_file(producer),
        },
        "configuration_hashes": {
            relative: finalizer.sha256_file(repo / relative)
            for relative in command["configuration_files"]
        },
        "input_hashes": {
            (Path(finalizer.STAGING_DIRECTORY) / item["path"]).as_posix(): finalizer.sha256_file(
                evidence / finalizer.STAGING_DIRECTORY / item["path"]
            )
            for item in command.get("input_files", [])
        },
        "raw_output_hashes": raw_hashes,
        "referenced_artifact_hashes": {
            relative: digest
            for relative, digest in raw_hashes.items()
            if relative.endswith((
                "_hessian_action.npy",
                "_residual.npy",
                "_branch_map.npy",
            ))
        },
        "artifact_validation_errors": [],
        "logs": {},
        "missing_outputs": [],
    }
    receipt["receipt_fingerprint_sha256"] = finalizer._json_sha256(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "route_terminal",
    ["predictive_selector_admissible", "finite_empirical_map_only"],
)
def test_finalizer_supports_clean_descendant_and_rejects_raw_and_final_tampering(
    tmp_path: Path, route_terminal: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    experiment_commit = _initialize_fake_repo(repo)
    evidence = repo / "artifacts/evidence"
    staging = evidence / finalizer.STAGING_DIRECTORY

    route_evidence_roles = (
        "route_campaign_master",
        "route_tranche_manifest",
        "route_submission_ledger",
        "route_release_authorization",
        "reviewed_release_artifact",
    )
    route_evidence_entries = []
    for role in route_evidence_roles:
        path = staging / "EXP-ROUTE-001/source_archives/karolina" / f"{role}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"role": role}) + "\n", encoding="utf-8")
        route_evidence_entries.append(
            {"role": role, "path": path.as_posix(), "sha256": finalizer.sha256_file(path)}
        )
    endpoint_relative = Path("EXP-ROUTE-001/reviewed_inputs/tier_b_endpoint_analysis.json")
    endpoint_path = staging / endpoint_relative
    endpoint_path.parent.mkdir(parents=True, exist_ok=True)
    endpoint_path.write_text(
        json.dumps(
            {
                "schema": {
                    "id": "fenics-nonlinear-energies.exp-route-001.tier-b-endpoints",
                    "version": 1,
                },
                "experiment_id": "EXP-ROUTE-001",
                "terminal_decision": "tier_b_comparative_ranking_admissible",
                "comparative_ranking_admissible": True,
                "endpoint_correct_timing_admissible": True,
                "matrix_policy_violations": [],
                "coverage_and_campaign_failure_reasons": [],
                "blocks": [{"status": "timing_admitted"} for _ in range(30)],
                "structural_censors": [
                    {
                        "status": "censored",
                        "reason": "prespecified_not_attempted_memory_risk_no_threshold_claim",
                        "route": "colored_sfd",
                        "timing_exposed": False,
                        "admitted_collective_max_wall_time_s": None,
                    }
                    for _ in range(2)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    commands = []
    route_contract = repo / "paper/protocols/EXP-ROUTE-001-analysis-contract.json"
    for spec in finalizer.SOURCE_SPECS:
        command = {
            "id": f"produce_{spec.key}",
            "source_keys": [spec.key],
            "producer": spec.producer_path.as_posix(),
            "argv": ["python", spec.producer_path.as_posix()],
            "environment": {},
            "configuration_files": ["paper/protocols/EXP-ROUTE-001-analysis-contract.json"],
            "input_files": [],
            "expected_artifacts": [],
        }
        raw = _scientific_raw_payload(spec, commit=experiment_commit)
        if spec.key in finalizer.QUADRATURE_DEGREES:
            command["expected_artifacts"] = [
                path.as_posix()
                for path in finalizer._quadrature_expected_artifacts(
                    finalizer.QUADRATURE_DEGREES[spec.key]
                )
            ]
            _write_quadrature_artifact_fixture(
                spec,
                raw,
                staging_root=staging,
            )
        if spec.key == "route_analysis":
            raw.pop("status", None)
            from paper.scripts import admit_revision_publication_evidence as admission

            contract_payload = json.loads(route_contract.read_text(encoding="utf-8"))
            slots, censor_slots = admission._route_expected_slots(
                contract_payload, {"workstation_local", "karolina_cpu"}
            )
            rows = []
            for slot in sorted(slots):
                hardware, configuration, state, ranks, route_name = slot
                split = (
                    "training"
                    if hardware == "workstation_local"
                    or ranks in contract_payload["hardware"]["karolina_cpu"]["training_ranks"]
                    else "holdout"
                )
                if slot in censor_slots:
                    reason = "prespecified_not_attempted_memory_risk_no_threshold_claim"
                    row = {
                        "hardware_id": hardware,
                        "configuration_id": configuration,
                        "state_id": state,
                        "rank_count": ranks,
                        "route": route_name,
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
                        "route": route_name,
                        "split": split,
                        "status": "admitted",
                        "reason": "",
                        "publication_model_eligible": True,
                        "model_exclusion_reason": "",
                        "admitted_wall_time_median_s": 1.0,
                        "action_relative_l2_error": 0.0,
                        "action_relative_l2_errors": [0.0, 0.0, 0.0, 0.0],
                        "action_max_absolute_error": 0.0,
                        "gradient_residual_relative_error": 0.0,
                        "state_sha256": "1" * 64,
                        "action_sha256": "2" * 64,
                        "source_commit": experiment_commit,
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
            features = contract_payload["cost_model"]["features_in_order"]
            if route_terminal == "predictive_selector_admissible":
                cost_model = {
                    "status": "selection_rule_passed",
                    "selector_claim_admissible": True,
                    "training_rows": training,
                    "holdout_rows": holdout,
                    "feature_order": features,
                    "gate_results": {
                        "median_absolute_percentage_error": True,
                        "p90_absolute_percentage_error": True,
                        "minimum_resolved_holdout_groups": True,
                        "resolved_ordering_accuracy": True,
                        "distinct_observed_holdout_winners": True,
                    },
                    "holdout_median_absolute_percentage_error": 0.0,
                    "holdout_p90_absolute_percentage_error": 0.0,
                    "resolved_holdout_groups": 8,
                    "resolved_ordering_accuracy": 1.0,
                    "distinct_observed_holdout_winners": [
                        "element_ad",
                        "constitutive_ad",
                    ],
                    "coefficients": {feature: 0.0 for feature in features},
                }
                factor_gate = {
                    "passed": True,
                    "failures": [],
                    "calibration_model": {"status": "passed"},
                }
            else:
                cost_model = {
                    "status": "fit_gate_failed",
                    "selector_claim_admissible": False,
                    "training_rows": training,
                    "holdout_rows": holdout,
                    "feature_order": features,
                    "preflight_failures": [],
                    "failed_gates": ["median_absolute_percentage_error"],
                }
                factor_gate = {
                    "passed": False,
                    "failures": ["factorized calibration holdout gates failed"],
                    "calibration_model": None,
                }
            factor_gate.update(
                {
                    "calibration_integrated": False,
                    "selector_use": "descriptive_replicated_synthetic_non_route_faithful_proxy",
                    "selector_blockers": [],
                    "required_ranks": [1, 8, 32],
                    "independent_blocks_per_rank": 3,
                }
            )
            raw.update(
                {
                    "analysis_schema_version": 1,
                    "terminal_decision": route_terminal,
                    "contract_path": route_contract.as_posix(),
                    "contract_sha256": finalizer.sha256_file(route_contract),
                    "sources": [
                        {
                            "hardware_id": hardware,
                            "publication_provenance_gate": {
                                "eligible": True,
                                "source_commit": experiment_commit,
                            },
                        }
                        for hardware in ("workstation_local", "karolina_cpu")
                    ],
                    "empirical_map": rows,
                    "cost_model": cost_model,
                    "factorized_microbenchmark_gate": factor_gate,
                    "invalid_records": [],
                    "provenance": {
                        "git": {"commit": experiment_commit, "dirty": False},
                        "input_files": route_evidence_entries,
                    },
                }
            )
            command["route_endpoint_analysis"] = endpoint_relative.as_posix()
            command["input_files"] = [
                {"scope": "staging", "path": endpoint_relative.as_posix()}
            ]
            command["expected_artifacts"] = [endpoint_relative.as_posix()]
        raw_path = staging / spec.relative_path
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
        producer_hash = finalizer.sha256_file(repo / spec.producer_path)
        for index, record_path in enumerate(spec.run_records):
            record = _run_record(
                spec=spec,
                relative=record_path,
                commit=experiment_commit,
                producer=spec.producer_path,
                producer_hash=producer_hash,
                record_id=f"{spec.key}-{index}",
            )
            destination = staging / record_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(record) + "\n", encoding="utf-8")
        commands.append(command)

    plan = {
        "schema_id": finalizer.PLAN_SCHEMA_ID,
        "schema_version": finalizer.PLAN_SCHEMA_VERSION,
        "plan_kind": "source_campaign",
        "campaign_id": "test-clean-campaign",
        "experiment_commit": experiment_commit,
        "commands": commands,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    finalizer._plan_command_map(plan)
    for command in commands:
        _write_receipt(
            path=evidence / finalizer.RECEIPT_DIRECTORY / f"{command['id']}.json",
            command=command,
            plan=plan,
            plan_path=plan_path,
            repo=repo,
            evidence=evidence,
            commit=experiment_commit,
        )

    raw_plaplace = staging / finalizer.SOURCE_BY_KEY["plaplace"].relative_path
    pristine_raw = raw_plaplace.read_bytes()
    raw_plaplace.write_bytes(pristine_raw + b" ")
    with pytest.raises(finalizer.FinalizationError, match="raw output was tampered"):
        finalizer.finalize_campaign(
            plan_path=plan_path, evidence_root=evidence, repo_root=repo
        )
    assert not (evidence / finalizer.SOURCE_BY_KEY["plaplace"].relative_path).exists()
    raw_plaplace.write_bytes(pristine_raw)

    # Release/documentation commits may follow the experiment as long as every
    # bound producer is byte-identical and ancestry is preserved.
    (repo / "release-note.txt").write_text("release descendant\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "release-note.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "release"], check=True, capture_output=True)

    manifest = finalizer.finalize_campaign(
        plan_path=plan_path, evidence_root=evidence, repo_root=repo
    )
    verified = finalizer.verify_finalized_campaign(
        manifest_path=manifest, evidence_root=evidence, repo_root=repo
    )
    assert verified["experiment_commit"] == experiment_commit
    assert verified["release_commit"] == _git(repo, "rev-parse", "HEAD")
    route = json.loads(
        (evidence / finalizer.SOURCE_BY_KEY["route_analysis"].relative_path).read_text()
    )
    assert route["endpoint_analysis"]["publication_admissible"] is True
    assert route["terminal_decision"] == route_terminal
    if route_terminal == "finite_empirical_map_only":
        assert route["cost_model"]["selector_claim_admissible"] is False
        assert not {
            "coefficients",
            "holdout_ordering",
            "coefficient_bootstrap_confidence_intervals",
        } & set(route["cost_model"])
    companion = json.loads(
        (evidence / finalizer.SOURCE_BY_KEY["route_analysis"].companion_manifest).read_text()
    )
    assert set(route["provenance"]["input_files"][0]) >= {"role", "path", "sha256"}
    assert not Path(route["provenance"]["input_files"][0]["path"]).is_absolute()
    assert len(companion["artifacts"]) == 6

    moved = repo / "artifacts/moved-evidence"
    shutil.move(evidence, moved)
    manifest = moved / finalizer.FINALIZATION_MANIFEST
    finalizer.verify_finalized_campaign(
        manifest_path=manifest, evidence_root=moved, repo_root=repo
    )
    from paper.scripts import admit_revision_publication_evidence as admission

    audit = admission.audit_evidence(
        moved,
        repo_root=repo,
        git_metadata={"commit": _git(repo, "rev-parse", "HEAD"), "worktree_clean": True},
    )
    blockers = {
        key: row["blockers"] for key, row in audit["inputs"].items() if row["blockers"]
    }
    assert audit["eligible"] is True, blockers
    assert audit["admitted_input_count"] == 14

    if route_terminal == "finite_empirical_map_only":
        source_manifest_path = moved / "publication_evidence_manifest.json"
        clean_python_environment = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        subprocess.run(
            [
                sys.executable,
                str(repo / "paper/scripts/admit_revision_publication_evidence.py"),
                "admit",
                "--evidence-root",
                str(moved),
                "--manifest-out",
                str(source_manifest_path),
            ],
            cwd=repo,
            env=clean_python_environment,
            check=True,
            capture_output=True,
            text=True,
        )
        table_dir = repo / "paper/tables/generated"
        subprocess.run(
            [
                sys.executable,
                str(repo / "paper/scripts/generate_revision_evidence_tables.py"),
                "--out-dir",
                str(table_dir),
                "--evidence-root",
                str(moved),
                "--evidence-class",
                "publication",
                "--evidence-manifest",
                str(source_manifest_path),
            ],
            cwd=repo,
            env=clean_python_environment,
            check=True,
            capture_output=True,
            text=True,
        )
        status_table = (table_dir / "revision_evidence_status.tex").read_text(
            encoding="utf-8"
        )
        assert "96/102" in status_table
        assert "admitted finite map" in status_table
        assert "Predictive cost selector & 74 train; 22 holdout & not admitted" in status_table
        assert "0 confirmation rows & not evaluated" in status_table
        manuscript_inputs = {
            Path("paper/sections/validation.tex"): [
                r"\input{tables/generated/revision_verification_summary.tex}",
                r"\input{tables/generated/revision_derivative_checks.tex}",
            ],
            Path("paper/sections/results.tex"): [
                r"\input{tables/generated/revision_quadrature_sensitivity.tex}",
                r"\input{tables/generated/revision_evidence_status.tex}",
            ],
        }
        for relative, literals in manuscript_inputs.items():
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(literals) + "\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "paper"], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", "publication tables"],
            check=True,
            capture_output=True,
        )
        checked = subprocess.run(
            [
                sys.executable,
                str(repo / "paper/scripts/check_revision_evidence_manifest.py"),
                "--manifest",
                str(table_dir / "revision_evidence_manifest.json"),
                "--repo-root",
                str(repo),
            ],
            cwd=repo,
            env=clean_python_environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr

    canonical = moved / finalizer.SOURCE_BY_KEY["plaplace"].relative_path
    original = canonical.read_text(encoding="utf-8")
    canonical.write_text(original + " ", encoding="utf-8")
    with pytest.raises(finalizer.FinalizationError, match="finalized output hash mismatch"):
        finalizer.verify_finalized_campaign(
            manifest_path=manifest, evidence_root=moved, repo_root=repo
        )

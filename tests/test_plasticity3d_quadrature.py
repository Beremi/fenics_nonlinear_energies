from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
from mpi4py import MPI
import numpy as np
import pytest

from src.core.benchmark.run_record import ExperimentPreflight
from src.problems.slope_stability_3d.support.fixed_state import (
    evaluate_fixed_state_overintegrated,
    evaluate_fixed_state_quadrature_diagnostics,
    evaluate_fixed_state_with_quadrature,
    prescribed_analytic_displacement,
)
from experiments.runners import prepare_plasticity3d_fixed_state as state_preparer
from experiments.runners import run_plasticity3d_fixed_state_quadrature as fixed_runner
from src.problems.slope_stability_3d.support import mesh as mesh_tools
from src.problems.slope_stability_3d.support.mesh import (
    LEGACY_SAME_MESH_HDF5_SCHEMA_VERSION,
    SAME_MESH_HDF5_SCHEMA_VERSION,
    SlopeStability3DCaseData,
    TETRA_QUADRATURE_1POINT,
    TETRA_QUADRATURE_11POINT,
    TETRA_QUADRATURE_24POINT,
    TETRA_QUADRATURE_DUFFY_125POINT,
    _assemble_local_tet_ops,
    default_tetra_quadrature_rule_id,
    expand_tetra_connectivity_to_dofs,
    load_case_hdf5,
    load_case_hdf5_fields,
    load_same_mesh_case_hdf5_rank_local,
    same_mesh_case_hdf5_path,
    tetra_quadrature_rule,
    write_case_hdf5,
)


def _one_tetra_case(rule_id: str = TETRA_QUADRATURE_24POINT) -> SlopeStability3DCaseData:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    elems_scalar = np.array([[0, 1, 2, 3]], dtype=np.int64)
    elems = expand_tetra_connectivity_to_dofs(elems_scalar)
    dphix, dphiy, dphiz, quad_weight, _hatp = _assemble_local_tet_ops(
        nodes,
        elems_scalar,
        degree=1,
        quadrature_rule_id=rule_id,
    )
    n_q = int(quad_weight.shape[1])
    constant = lambda value: np.full((1, n_q), float(value), dtype=np.float64)
    return SlopeStability3DCaseData(
        case_name=f"one_tetra_quad_{rule_id}",
        mesh_name="one_tetra",
        degree=1,
        quadrature_rule_id=str(rule_id),
        raw_mesh_filename="one_tetra.msh",
        constraint_variant="glued_bottom",
        nodes=nodes,
        elems_scalar=elems_scalar,
        elems=elems,
        surf=np.empty((0, 3), dtype=np.int64),
        boundary_label=np.empty(0, dtype=np.int64),
        q_mask=np.ones((4, 3), dtype=bool),
        freedofs=np.arange(12, dtype=np.int64),
        dphix=dphix,
        dphiy=dphiy,
        dphiz=dphiz,
        quad_weight=quad_weight,
        force=np.zeros(12, dtype=np.float64),
        u_0=np.zeros(12, dtype=np.float64),
        material_id=np.zeros(1, dtype=np.int64),
        c0_q=constant(15.0),
        phi_q=constant(np.deg2rad(30.0)),
        psi_q=constant(0.0),
        shear_q=constant(3846.153846153846),
        bulk_q=constant(8333.333333333334),
        lame_q=constant(5769.2307692307695),
        gamma_q=constant(19.0),
        eps_p_old=np.zeros((1, n_q, 6), dtype=np.float64),
        adjacency=None,
        elastic_kernel=np.zeros((12, 6), dtype=np.float64),
        macro_parent=np.zeros(1, dtype=np.int64),
        macro_parent_mesh_name="one_tetra",
    )


def test_degree_defaults_and_nondefault_paths_preserve_legacy_names() -> None:
    assert default_tetra_quadrature_rule_id(1) == TETRA_QUADRATURE_1POINT
    assert default_tetra_quadrature_rule_id(2) == TETRA_QUADRATURE_11POINT
    assert default_tetra_quadrature_rule_id(4) == TETRA_QUADRATURE_24POINT

    implicit = same_mesh_case_hdf5_path("hetero_ssr_L1", 2, "glued_bottom")
    explicit = same_mesh_case_hdf5_path(
        "hetero_ssr_L1",
        2,
        "glued_bottom",
        quadrature_rule_id=TETRA_QUADRATURE_11POINT,
    )
    enriched = same_mesh_case_hdf5_path(
        "hetero_ssr_L1",
        2,
        "glued_bottom",
        quadrature_rule_id=TETRA_QUADRATURE_24POINT,
    )
    assert implicit == explicit
    assert enriched != implicit
    assert enriched.name.endswith("_quad_tetra_24point.h5")


def test_duffy_125point_rule_integrates_total_degree_seven_monomials() -> None:
    points, weights = tetra_quadrature_rule(TETRA_QUADRATURE_DUFFY_125POINT)
    assert points.shape == (3, 125)
    assert weights.shape == (125,)
    assert np.all(weights > 0.0)
    np.testing.assert_allclose(np.sum(weights), 1.0 / 6.0, rtol=0.0, atol=2.0e-15)
    assert np.all(points >= 0.0)
    assert np.all(np.sum(points, axis=0) <= 1.0 + 1.0e-15)

    x, y, z = points
    for i in range(8):
        for j in range(8 - i):
            for k in range(8 - i - j):
                exact = (
                    math.factorial(i)
                    * math.factorial(j)
                    * math.factorial(k)
                    / math.factorial(i + j + k + 3)
                )
                actual = float(np.dot(weights, x**i * y**j * z**k))
                np.testing.assert_allclose(actual, exact, rtol=2.0e-13, atol=2.0e-15)


def test_24point_rule_integrates_reference_elastic_p4_degree_six_products() -> None:
    points, weights = tetra_quadrature_rule(TETRA_QUADRATURE_24POINT)
    x, y, z = points
    for i in range(7):
        for j in range(7 - i):
            for k in range(7 - i - j):
                exact = (
                    math.factorial(i)
                    * math.factorial(j)
                    * math.factorial(k)
                    / math.factorial(i + j + k + 3)
                )
                actual = float(np.dot(weights, x**i * y**j * z**k))
                np.testing.assert_allclose(
                    actual, exact, rtol=2.0e-13, atol=2.0e-15
                )


def test_hdf5_round_trip_records_rule_and_legacy_file_infers_default(tmp_path: Path) -> None:
    case = _one_tetra_case(TETRA_QUADRATURE_24POINT)
    path = tmp_path / "case.h5"
    write_case_hdf5(path, case)
    with h5py.File(path, "r") as handle:
        assert int(handle["schema_version"][()]) == SAME_MESH_HDF5_SCHEMA_VERSION
        assert handle["quadrature_rule_id"][()].decode() == TETRA_QUADRATURE_24POINT
    loaded = load_case_hdf5(path)
    assert loaded.quadrature_rule_id == TETRA_QUADRATURE_24POINT

    legacy_path = tmp_path / "legacy.h5"
    write_case_hdf5(legacy_path, _one_tetra_case(TETRA_QUADRATURE_1POINT))
    with h5py.File(legacy_path, "r+") as handle:
        del handle["quadrature_rule_id"]
        del handle["schema_version"]
        handle.create_dataset("schema_version", data=LEGACY_SAME_MESH_HDF5_SCHEMA_VERSION)
    legacy = load_case_hdf5(legacy_path)
    assert legacy.quadrature_rule_id == TETRA_QUADRATURE_1POINT
    fields, _adjacency = load_case_hdf5_fields(
        legacy_path,
        fields=("quadrature_rule_id",),
    )
    assert fields == {"quadrature_rule_id": TETRA_QUADRATURE_1POINT}


def test_rank_local_heavy_loader_carries_custom_rule_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "case.h5"
    write_case_hdf5(path, _one_tetra_case(TETRA_QUADRATURE_24POINT))
    mesh_tools.clear_same_mesh_case_hdf5_caches()
    monkeypatch.setattr(mesh_tools, "ensure_same_mesh_case_hdf5", lambda *args, **kwargs: path)
    payload = load_same_mesh_case_hdf5_rank_local(
        "one_tetra",
        1,
        constraint_variant="glued_bottom",
        quadrature_rule_id=TETRA_QUADRATURE_24POINT,
        reorder_mode="block_xyz",
        comm=MPI.COMM_SELF,
    )
    assert payload["quadrature_rule_id"] == TETRA_QUADRATURE_24POINT
    assert payload["_distributed_quad_weight"].shape == (1, 24)
    mesh_tools.clear_same_mesh_case_hdf5_caches()


def test_fixed_state_reference_rebuilds_quadrature_independently() -> None:
    case = _one_tetra_case(TETRA_QUADRATURE_1POINT)
    displacement = 1.0e-6 * np.asarray(case.nodes, dtype=np.float64).reshape(-1)
    one_point = evaluate_fixed_state_with_quadrature(
        case,
        displacement,
        lambda_target=1.5,
        quadrature_rule_id=TETRA_QUADRATURE_1POINT,
        element_chunk_size=1,
    )
    enriched = evaluate_fixed_state_overintegrated(
        case,
        displacement,
        lambda_target=1.5,
        element_chunk_size=1,
    )
    lightweight_case = {
        key: getattr(case, key)
        for key in ("degree", "nodes", "elems_scalar", "elems", "material_id")
    }
    lightweight = evaluate_fixed_state_with_quadrature(
        lightweight_case,
        displacement,
        lambda_target=1.5,
        quadrature_rule_id=TETRA_QUADRATURE_24POINT,
        element_chunk_size=1,
    )
    assert one_point["quadrature_points"] == 1
    assert enriched["quadrature_points"] == 125
    assert lightweight["quadrature_points"] == 24
    np.testing.assert_allclose(
        enriched["total_potential_energy"],
        one_point["total_potential_energy"],
        rtol=1.0e-11,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        lightweight["total_potential_energy"],
        one_point["total_potential_energy"],
        rtol=1.0e-11,
        atol=1.0e-14,
    )


def test_fixed_state_diagnostics_rebuild_residual_action_and_branch_data() -> None:
    case = _one_tetra_case(TETRA_QUADRATURE_24POINT)
    displacement = 1.0e-5 * np.asarray(case.nodes, dtype=np.float64).reshape(-1)
    center = evaluate_fixed_state_quadrature_diagnostics(
        case,
        displacement,
        lambda_target=1.5,
        quadrature_rule_id=TETRA_QUADRATURE_24POINT,
        element_chunk_size=1,
    )
    direction = center.deterministic_direction
    step = 1.0e-7
    plus = evaluate_fixed_state_quadrature_diagnostics(
        case,
        displacement + step * direction,
        lambda_target=1.5,
        quadrature_rule_id=TETRA_QUADRATURE_24POINT,
        element_chunk_size=1,
    )
    minus = evaluate_fixed_state_quadrature_diagnostics(
        case,
        displacement - step * direction,
        lambda_target=1.5,
        quadrature_rule_id=TETRA_QUADRATURE_24POINT,
        element_chunk_size=1,
    )

    energy_directional_fd = (
        float(plus.summary["total_potential_energy"])
        - float(minus.summary["total_potential_energy"])
    ) / (2.0 * step)
    residual_directional = float(np.dot(center.full_residual, direction))
    np.testing.assert_allclose(
        energy_directional_fd,
        residual_directional,
        rtol=2.0e-9,
        atol=2.0e-10,
    )
    residual_fd = (plus.full_residual - minus.full_residual) / (2.0 * step)
    np.testing.assert_allclose(
        residual_fd,
        center.hessian_action,
        rtol=2.0e-9,
        atol=2.0e-7,
    )
    assert center.summary["free_residual_l2_norm"] == center.summary[
        "full_residual_l2_norm"
    ]
    assert center.summary["free_hessian_action_l2_norm"] == center.summary[
        "full_hessian_action_l2_norm"
    ]
    assert sum(center.summary["branch_point_counts"].values()) == 24
    assert center.summary["branch_point_counts"]["elastic"] == 24
    assert center.summary["quadrature_points_at_or_below_margin_gate"] == 0
    assert center.summary["minimum_normalized_constitutive_denominator"] > 0.0
    json.dumps(center.summary, allow_nan=False)


def test_fixed_state_runner_saves_hashable_actions_and_strict_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case_path = tmp_path / "case.h5"
    case = _one_tetra_case(TETRA_QUADRATURE_24POINT)
    write_case_hdf5(case_path, case)
    experiment_root = tmp_path / "EXP-DISC-001"
    state_path = experiment_root / "clean_inputs" / "state.npz"
    state_path.parent.mkdir(parents=True)
    displacement = 1.0e-5 * np.asarray(case.nodes, dtype=np.float64)
    np.savez(
        state_path,
        displacement=displacement,
        coords_ref=np.asarray(case.nodes, dtype=np.float64),
        mesh_name=np.asarray(case.mesh_name),
        element_degree=np.asarray(case.degree, dtype=np.int64),
        lambda_target=np.asarray(1.5, dtype=np.float64),
        quadrature_rule_id=np.asarray(TETRA_QUADRATURE_24POINT),
        constraint_variant=np.asarray(case.constraint_variant),
    )
    monkeypatch.setattr(
        fixed_runner,
        "ensure_same_mesh_case_hdf5",
        lambda *args, **kwargs: case_path,
    )
    action_dir = experiment_root / "actions" / "p1_l1"
    output_path = experiment_root / "p1_l1_fixed_state_quadrature_v2.json"
    payload = fixed_runner.run(
        argparse.Namespace(
            state=state_path,
            output=output_path,
            constraint_variant=None,
            quadrature_rules=(
                f"{TETRA_QUADRATURE_1POINT},{TETRA_QUADRATURE_DUFFY_125POINT}"
            ),
            element_chunk_size=1,
            coordinate_atol=1.0e-12,
            action_output_dir=action_dir,
        )
    )

    assert payload["common_free_dof_set"] is True
    assert payload["state_path"] == "clean_inputs/state.npz"
    assert len(payload["common_direction_content_sha256"]) == 64
    assert payload["reference_rule_id"] == TETRA_QUADRATURE_DUFFY_125POINT
    assert len(payload["evaluations"]) == 2
    for row in payload["evaluations"]:
        artifact = row["hessian_action_artifact"]
        assert artifact["path"].startswith("actions/p1_l1/")
        action_path = experiment_root / artifact["path"]
        assert action_path.is_file()
        action = np.load(action_path, allow_pickle=False)
        assert action.shape == (12,)
        assert np.all(np.isfinite(action))
        assert hashlib.sha256(action_path.read_bytes()).hexdigest() == artifact["sha256"]
        residual_artifact = row["residual_artifact"]
        assert residual_artifact["path"].startswith("actions/p1_l1/")
        residual_path = experiment_root / residual_artifact["path"]
        residual = np.load(residual_path, allow_pickle=False)
        assert residual.shape == (12,)
        assert np.all(np.isfinite(residual))
        assert hashlib.sha256(residual_path.read_bytes()).hexdigest() == residual_artifact[
            "sha256"
        ]
        branch_artifact = row["branch_map_artifact"]
        assert branch_artifact["path"].startswith("actions/p1_l1/")
        branch_path = experiment_root / branch_artifact["path"]
        branch_map = np.load(branch_path, allow_pickle=False)
        assert branch_map.dtype == np.int8
        assert branch_map.size == row["branch_sample_points"]
        assert hashlib.sha256(branch_path.read_bytes()).hexdigest() == branch_artifact[
            "sha256"
        ]
        assert sum(row["branch_point_counts"].values()) == row["branch_sample_points"]
        assert "hessian_action_vector_comparison_to_last_rule" in row
        assert "free_hessian_action_vector_comparison_to_last_rule" in row
        assert "full_residual_vector_comparison_to_last_rule" in row
        assert "free_residual_vector_comparison_to_last_rule" in row
        assert "branch_comparison_to_last_rule" in row
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("escaping_path", ["state", "actions"])
def test_fixed_state_runner_rejects_experiment_root_escape_before_evaluation(
    tmp_path: Path,
    monkeypatch,
    escaping_path: str,
) -> None:
    experiment_root = tmp_path / "EXP-DISC-001"
    contained_state = experiment_root / "clean_inputs" / "state.npz"
    contained_state.parent.mkdir(parents=True)
    np.savez(contained_state, placeholder=np.asarray(1, dtype=np.int64))
    outside_state = tmp_path / "outside_state.npz"
    np.savez(outside_state, placeholder=np.asarray(1, dtype=np.int64))
    state_path = outside_state if escaping_path == "state" else contained_state
    action_dir = (
        tmp_path / "outside_actions"
        if escaping_path == "actions"
        else experiment_root / "actions" / "p1_l1"
    )

    def fail_if_evaluated(*_args, **_kwargs):
        raise AssertionError("path validation must precede fixed-state evaluation")

    monkeypatch.setattr(
        fixed_runner,
        "evaluate_fixed_state_quadrature_diagnostics",
        fail_if_evaluated,
    )
    with pytest.raises(RuntimeError, match="contained in the output experiment directory"):
        fixed_runner.run(
            argparse.Namespace(
                state=state_path,
                output=experiment_root / "p1_l1_fixed_state_quadrature_v2.json",
                constraint_variant=None,
                quadrature_rules=TETRA_QUADRATURE_1POINT,
                element_chunk_size=1,
                coordinate_atol=1.0e-12,
                action_output_dir=action_dir,
            )
        )
    assert not action_dir.exists()


def test_prescribed_state_preparer_exports_constraint_aware_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = _one_tetra_case(TETRA_QUADRATURE_1POINT)
    nodes = np.vstack(
        [np.asarray(case.nodes, dtype=np.float64), np.asarray([[0.25, 0.25, 0.25]])]
    )
    freedofs = np.arange(3, 15, dtype=np.int64)
    case_mapping = {
        "nodes": nodes,
        "elems_scalar": np.asarray(case.elems_scalar, dtype=np.int64),
        "surf": np.asarray(case.surf, dtype=np.int64),
        "boundary_label": np.asarray(case.boundary_label, dtype=np.int64),
        "freedofs": freedofs,
        "u_0": np.zeros(15, dtype=np.float64),
    }
    monkeypatch.setattr(
        state_preparer,
        "load_same_mesh_case_hdf5_light",
        lambda *args, **kwargs: case_mapping,
    )
    monkeypatch.setattr(
        state_preparer,
        "check_experiment_preflight",
        lambda *args, **kwargs: ExperimentPreflight(
            run_kind="publication",
            git_commit="a" * 40,
            git_clean=True,
            git_status_porcelain=(),
            pilot_override=False,
            pilot_override_reason=None,
            checked_at_utc="2026-07-10T00:00:00Z",
        ),
    )
    state_path = tmp_path / "p1_state.npz"
    manifest_path = tmp_path / "p1_state_manifest.json"
    payload = state_preparer.prepare(
        argparse.Namespace(
            output=state_path,
            manifest=manifest_path,
            run_kind="publication",
            pilot_dirty_override=False,
            pilot_override_reason=None,
            degree=1,
            mesh_name="one_tetra",
            constraint_variant="glued_bottom",
            lambda_target=1.55,
            state_label="mixed",
            amplitude=0.02,
        )
    )

    assert payload["status"] == "completed"
    assert payload["identifiers"]["state_kind"] == "analytic_not_solved"
    assert payload["state"]["constrained_coefficients_match_reference"] is True
    assert manifest_path.is_file()
    with np.load(state_path, allow_pickle=False) as state:
        displacement = np.asarray(state["displacement"], dtype=np.float64).reshape(-1)
        assert np.array_equal(displacement[:3], np.zeros(3))
        assert np.linalg.norm(displacement[freedofs]) > 0.0
        assert state["solver_family"].item() == "prescribed_fixed_state"
        assert state["state_kind"].item() == "analytic_not_solved"

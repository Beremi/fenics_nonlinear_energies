from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.runners import run_exp_stop_001_local_calibration as stop_campaign
from src.core.benchmark.state_export import export_hyperelasticity_state_npz


def _mesh() -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return coordinates, np.asarray([[0, 1, 2, 3]], dtype=np.int32)


def _write_state(
    path: Path,
    *,
    free: np.ndarray,
    stiffness: np.ndarray,
) -> None:
    coordinates, tetrahedra = _mesh()
    export_hyperelasticity_state_npz(
        path,
        coords_ref=coordinates,
        x_final=coordinates + 0.01,
        tetrahedra=tetrahedra,
        mesh_level=1,
        total_steps=1,
        free_deformation=free,
        reference_elastic_action=stiffness @ free,
    )


def test_hyperelasticity_export_binds_reference_operator_action(tmp_path: Path) -> None:
    path = tmp_path / "state.npz"
    free = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    stiffness = np.diag(np.asarray([2.0, 3.0, 4.0], dtype=np.float64))
    _write_state(path, free=free, stiffness=stiffness)

    with np.load(path, allow_pickle=False) as state:
        np.testing.assert_array_equal(state["free_deformation_original"], free)
        np.testing.assert_array_equal(
            state["reference_elastic_action"], stiffness @ free
        )
        assert float(state["reference_elastic_state_quadratic"]) == pytest.approx(
            float(free @ stiffness @ free)
        )

    coordinates, tetrahedra = _mesh()
    with pytest.raises(ValueError, match="must be supplied together"):
        export_hyperelasticity_state_npz(
            tmp_path / "incomplete.npz",
            coords_ref=coordinates,
            x_final=coordinates,
            tetrahedra=tetrahedra,
            mesh_level=1,
            total_steps=1,
            free_deformation=free,
        )


def test_stop_comparison_uses_reference_elastic_state_difference(tmp_path: Path) -> None:
    stiffness = np.diag(np.asarray([2.0, 3.0, 4.0], dtype=np.float64))
    reference_free = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    candidate_free = reference_free + np.asarray([1.0e-7, -2.0e-7, 1.0e-7])
    reference_path = tmp_path / "reference.npz"
    candidate_path = tmp_path / "candidate.npz"
    _write_state(reference_path, free=reference_free, stiffness=stiffness)
    _write_state(candidate_path, free=candidate_free, stiffness=stiffness)

    candidate_row = {
        "row_id": "candidate",
        "expected_outputs": [str(candidate_path)],
    }
    reference_row = {
        "row_id": "reference",
        "expected_outputs": [str(reference_path)],
    }
    contract = {
        "he_nonlinear_displacement_relative_difference_max": 1.0e-5,
        "he_nonlinear_reference_elastic_relative_state_difference_max": 1.0e-5,
        "he_nonlinear_energy_absolute_difference_max": 1.0e-8,
    }
    comparison = stop_campaign._compare_he_nonlinear(
        candidate_row,
        {"status": "endpoint_admitted", "energy": 1.0},
        reference_row,
        {"status": "endpoint_admitted", "energy": 1.0},
        contract,
    )

    difference = candidate_free - reference_free
    expected = np.sqrt(float(difference @ stiffness @ difference))
    reference_norm = np.sqrt(float(reference_free @ stiffness @ reference_free))
    assert comparison["status"] == "accepted"
    assert comparison["riesz_state_difference_available"] is True
    assert comparison["reference_elastic_state_difference"] == pytest.approx(expected)
    assert comparison["reference_elastic_relative_state_difference"] == pytest.approx(
        expected / reference_norm
    )


from __future__ import annotations

from pathlib import Path

import numpy as np
from mpi4py import MPI

from experiments.runners import run_hyperelasticity_distribution_equivalence as runner
from src.problems.hyperelasticity.support.mesh import load_rank_local_hyperelasticity


def _worker(*, energy: float = 2.0) -> dict[str, object]:
    return {
        "status": "passed",
        "mesh_semantics": {
            "topology_hashes": {
                "coordinates": "a",
                "connectivity": "b",
                "freedofs": "c",
                "right_boundary_nodes": "d",
                "affine_lift": "e",
            }
        },
        "algebraic_objects": {"energy": energy},
        "linear_solve": {"gate_passed": True},
    }


def _arrays() -> dict[str, np.ndarray]:
    return {
        "state": np.array([1.0, 2.0]),
        "direction": np.array([0.5, -0.25]),
        "residual": np.array([3.0, 4.0]),
        "matrix_indptr": np.array([0, 2, 4], dtype=np.int64),
        "matrix_indices": np.array([0, 1, 0, 1], dtype=np.int64),
        "matrix_data": np.array([2.0, -1.0, -1.0, 2.0]),
        "matrix_action": np.array([1.25, -1.0]),
        "correction": np.array([0.1, 0.2]),
    }


def test_worker_commands_hold_all_scientific_factors_fixed(tmp_path: Path):
    commands = {
        ranks: runner.build_worker_command(
            ranks=ranks,
            output_json=tmp_path / f"np{ranks}.json",
            output_npz=tmp_path / f"np{ranks}.npz",
            level=1,
            angle=0.15,
            repetitions=3,
            ksp_rtol=1.0e-12,
            linear_residual_tolerance=1.0e-10,
        )
        for ranks in (1, 2)
    }

    for ranks, command in commands.items():
        assert command[command.index("-n") + 1] == str(ranks)
        assert command[command.index("--level") + 1] == "1"
        assert command[command.index("--angle") + 1] == "0.15"
        assert command[command.index("--repetitions") + 1] == "3"
        assert command[command.index("--ksp-rtol") + 1] == "1e-12"
        assert command[command.index("--linear-residual-tolerance") + 1] == "1e-10"

    assert runner.FACTOR_CONFIGURATION == {
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
    }


def test_comparison_requires_exact_objects_and_scaled_derivative_agreement():
    reference = _arrays()
    candidate = {key: value.copy() for key, value in reference.items()}
    passed = runner.compare_worker_outputs(
        _worker(),
        _worker(energy=2.0 + 1.0e-12),
        reference,
        candidate,
        derivative_tolerance=1.0e-8,
        solve_tolerance=1.0e-8,
    )
    assert passed["algebraic_gate_passed"] is True
    assert all(passed["exact_topology_gates"].values())
    assert all(passed["exact_object_gates"].values())

    candidate["residual"][0] += 1.0e-3
    failed = runner.compare_worker_outputs(
        _worker(),
        _worker(),
        reference,
        candidate,
        derivative_tolerance=1.0e-8,
        solve_tolerance=1.0e-8,
    )
    assert failed["algebraic_gate_passed"] is False
    assert failed["derivative_gates"]["residual_relative"] is False


def test_rank_local_mesh_validation_checks_closed_form_level_one_data():
    params, _, _ = load_rank_local_hyperelasticity(
        1,
        comm=MPI.COMM_SELF,
        reorder_mode="block_xyz",
        mesh_source="procedural",
        element_degree=1,
    )
    validation = runner._validate_local_mesh(params)
    assert validation["passed"] is True
    assert all(validation["checks"].values())


def test_report_forbids_performance_claim_and_marks_endpoint_outstanding():
    payload = {
        "status": "passed",
        "comparison": {
            "algebraic_gate_passed": True,
            "derivative_tolerance": 1.0e-8,
            "solve_tolerance": 1.0e-8,
            "relative_errors": {
                "energy_relative": 0.0,
                "residual_relative": 0.0,
                "matrix_relative": 0.0,
                "matrix_action_relative": 0.0,
                "linear_correction_relative": 0.0,
            },
            "derivative_gates": {
                "energy_relative": True,
                "residual_relative": True,
                "matrix_relative": True,
                "matrix_action_relative": True,
            },
            "linear_solve_gates": {"linear_correction": True},
        },
        "descriptive_phase_timings": {
            "assembly": {"np1_s": 1.0, "np2_s": 0.75, "np2_over_np1": 0.75}
        },
    }
    report = runner._render_report(payload)
    assert "not publication performance evidence" in report
    assert "not a converged nonlinear endpoint" in report
    assert "Only after those gates pass" in report

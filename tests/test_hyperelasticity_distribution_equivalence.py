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
            residual_scale_floor=1.0,
            run_kind="publication",
        )
        for ranks in runner.RANK_COUNTS
    }

    for ranks, command in commands.items():
        assert command[command.index("-n") + 1] == str(ranks)
        assert command[command.index("--level") + 1] == "1"
        assert command[command.index("--angle") + 1] == "0.15"
        assert command[command.index("--repetitions") + 1] == "3"
        assert command[command.index("--ksp-rtol") + 1] == "1e-12"
        assert command[command.index("--linear-residual-tolerance") + 1] == "1e-10"
        assert command[command.index("--residual-scale-floor") + 1] == "1.0"
        assert command[command.index("--run-kind") + 1] == "publication"

    assert runner.RANK_COUNTS == (1, 2, 4)

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


def test_rank_comparison_aggregation_requires_and_conjoins_np2_and_np4():
    reference = _arrays()
    pair = runner.compare_worker_outputs(
        _worker(),
        _worker(),
        reference,
        {key: value.copy() for key, value in reference.items()},
        derivative_tolerance=1.0e-8,
        solve_tolerance=1.0e-8,
    )
    aggregate = runner.aggregate_rank_comparisons({"np2": pair, "np4": pair})
    assert aggregate["algebraic_gate_passed"] is True
    assert aggregate["relative_errors"]["matrix_action_relative"] == 0.0

    failed_np4 = dict(pair)
    failed_np4["algebraic_gate_passed"] = False
    failed_np4["relative_errors"] = {
        **pair["relative_errors"],
        "matrix_action_relative": 2.0e-8,
    }
    failed_np4["derivative_gates"] = {
        **pair["derivative_gates"],
        "matrix_action_relative": False,
    }
    aggregate = runner.aggregate_rank_comparisons({"np2": pair, "np4": failed_np4})
    assert aggregate["algebraic_gate_passed"] is False
    assert aggregate["derivative_gates"]["matrix_action_relative"] is False
    assert aggregate["relative_errors"]["matrix_action_relative"] == 2.0e-8

    try:
        runner.aggregate_rank_comparisons({"np2": pair})
    except ValueError as exc:
        assert "np4" in str(exc)
    else:
        raise AssertionError("aggregation accepted an omitted four-rank comparison")


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
    rank_comparison = {
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
        "linear_solve_gates": {
            "linear_correction": True,
            "reference_true_residual": True,
            "candidate_true_residual": True,
        },
    }
    payload = {
        "run_kind": "publication",
        "status": "passed",
        "comparison": {
            **rank_comparison,
        },
        "rank_comparisons": {"np2": rank_comparison, "np4": rank_comparison},
        "descriptive_phase_timings": {
            "assembly": {
                "np1_s": 1.0,
                "np2_s": 0.75,
                "np2_over_np1": 0.75,
                "np4_s": 0.5,
                "np4_over_np1": 0.5,
            }
        },
    }
    report = runner._render_report(payload)
    assert "not publication performance evidence" in report
    assert "not a converged nonlinear endpoint" in report
    assert "np1 vs np4" in report
    assert "Only after those gates pass" in report


def test_publication_configuration_is_frozen_and_main_dispatches(monkeypatch, tmp_path: Path):
    args = runner._parser().parse_args(
        ["--run-kind", "publication", "--output-dir", str(tmp_path)]
    )
    runner._validate_publication_configuration(args)

    drifted = runner._parser().parse_args(
        [
            "--run-kind",
            "publication",
            "--output-dir",
            str(tmp_path),
            "--level",
            "2",
        ]
    )
    try:
        runner._validate_publication_configuration(drifted)
    except ValueError as exc:
        assert "frozen EXP-DIST-001 configuration" in str(exc)
    else:
        raise AssertionError("publication mode accepted a changed mesh level")

    called: dict[str, object] = {}
    monkeypatch.setattr(
        runner,
        "_controller",
        lambda parsed: called.update(run_kind=parsed.run_kind),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_hyperelasticity_distribution_equivalence.py",
            "--run-kind",
            "publication",
            "--output-dir",
            str(tmp_path),
        ],
    )
    runner.main()
    assert called == {"run_kind": "publication"}


def test_publication_controller_refuses_nonempty_output_before_workers(
    monkeypatch, tmp_path: Path
):
    (tmp_path / "preexisting.txt").write_text("do not overwrite\n", encoding="utf-8")
    args = runner._parser().parse_args(
        ["--run-kind", "publication", "--output-dir", str(tmp_path)]
    )
    monkeypatch.setattr(runner, "check_experiment_preflight", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner, "_dirty_snapshot_sha256", lambda: None)
    monkeypatch.setattr(
        runner,
        "_run_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("worker launched before freshness check")
        ),
    )
    try:
        runner._controller(args)
    except FileExistsError as exc:
        assert "fresh and empty" in str(exc)
    else:
        raise AssertionError("publication controller accepted a nonempty output directory")

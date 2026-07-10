from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI

from experiments.runners import generate_scalar_uniform_l10_meshes as scalar_meshes
from experiments.runners import run_globalization_method_compare as campaign
from experiments.runners import run_trust_region_case as case_runner
from src.core.benchmark.run_record import (
    ExperimentPreflight,
    ExperimentPreflightError,
    validate_run_record,
)
from src.core.benchmark.state_export import (
    export_hyperelasticity_state_npz,
    export_scalar_mesh_state_npz,
)
from src.core.petsc import scalar_problem_driver
from src.problems.hyperelasticity.jax_petsc import solver as he_solver


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-standard JSON constant: {token}")


def test_full_case_matrix_matches_requested_campaign_shape(tmp_path: Path):
    cases = campaign.build_case_matrix("full")

    assert len(cases) == 12
    assert {(case.benchmark.problem, case.benchmark.level, case.benchmark.nprocs) for case in cases} == {
        ("plaplace", 10, 32),
        ("gl", 10, 16),
        ("he", 4, 32),
        ("plasticity3d", 1, 32),
    }
    assert {case.benchmark.wall_cap_s for case in cases if case.benchmark.problem == "plaplace"} == {30.0}
    assert {case.benchmark.wall_cap_s for case in cases if case.benchmark.problem == "gl"} == {120.0}
    assert {case.benchmark.wall_cap_s for case in cases if case.benchmark.problem == "he"} == {270.0}
    assert {case.benchmark.wall_cap_s for case in cases if case.benchmark.problem == "plasticity3d"} == {180.0}
    assert {case.method.key for case in cases} == {
        "newton_linesearch",
        "steihaug_trust",
        "hybrid_trust_linesearch",
    }


def test_case_runner_serializes_optional_nonfinite_diagnostics_as_null(tmp_path: Path):
    path = tmp_path / "case.json"
    case_runner._write_payload(
        str(path),
        {"result": {"history": [{"trust_ratio": float("nan"), "value": 1.0}]}},
    )

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    assert payload["result"]["history"][0]["trust_ratio"] is None


def test_case_runner_accepts_canonical_state_input_and_final_state_output():
    args = case_runner._build_parser().parse_args(
        [
            "--problem",
            "gl",
            "--backend",
            "element",
            "--level",
            "5",
            "--out",
            "output.json",
            "--state-in",
            "canonical_start.npz",
            "--state-out",
            "final_state.npz",
        ]
    )
    assert args.state_in == "canonical_start.npz"
    assert args.state_out == "final_state.npz"


def test_controlled_matrix_excludes_unprescribed_cases_and_holds_ksp_fixed(
    tmp_path: Path,
):
    cases = campaign.build_case_matrix("full", "controlled")

    assert len(cases) == 12
    assert {case.benchmark.problem for case in cases} == {"gl", "he"}
    assert all(case.benchmark.steps == 1 for case in cases)
    assert {case.robustness_instance.key for case in cases} == {
        "nominal",
        "mode_plus",
        "mode_minus",
    }
    assert {case.method.key for case in cases} == {
        "newton_armijo",
        "reduced_trust_armijo",
    }
    assert {case.comparison_tier for case in cases} == {"controlled"}

    for problem in ("gl", "he"):
        problem_cases = [
            case
            for case in cases
            if case.benchmark.problem == problem
            and case.robustness_instance.key == "nominal"
        ]
        commands = [
            campaign.build_command(
                case,
                tmp_path / f"{case.key}.json",
                state_in=tmp_path / f"{problem}_start.npz",
                state_out=tmp_path / f"{case.key}_final.npz",
            )
            for case in problem_cases
        ]
        ksp_types = {
            command[command.index("--ksp-type") + 1] for command in commands
        }
        assert ksp_types == {problem_cases[0].benchmark.line_ksp_type}
        assert all(command[command.index("--line-search") + 1] == "armijo" for command in commands)

    trust_case = next(case for case in cases if case.method.key == "reduced_trust_armijo")
    trust_command = campaign.build_command(
        trust_case,
        tmp_path / "trust.json",
        state_in=tmp_path / "common_start.npz",
        state_out=tmp_path / "trust_final.npz",
    )
    assert "--use-trust-region" in trust_command
    assert "--no-trust-subproblem-line-search" in trust_command
    assert trust_command[trust_command.index("--state-in") + 1].endswith(
        "common_start.npz"
    )
    assert trust_command[trust_command.index("--state-out") + 1].endswith(
        "trust_final.npz"
    )


def test_controlled_common_start_artifacts_are_one_per_retained_benchmark(
    tmp_path: Path,
):
    cases = campaign.build_case_matrix("smoke", "controlled")
    starts = campaign.prepare_controlled_starts(cases, tmp_path)

    assert len(cases) == 12
    assert {case.benchmark.problem for case in cases} == {"gl", "he"}
    assert len(starts) == 6
    assert set(starts) == {case.start_key for case in cases}
    assert all(len(identity["file_sha256"]) == 64 for identity in starts.values())
    assert all(len(identity["state_sha256"]) == 64 for identity in starts.values())
    for start_key, identity in starts.items():
        path = Path(identity["path"])
        assert path.is_file(), start_key
        case = next(case for case in cases if case.start_key == start_key)
        assert path == campaign.canonical_start_path(
            tmp_path,
            case.benchmark,
            case.robustness_instance,
        ).resolve()

    manifest = json.loads(
        (tmp_path / "_canonical_starts" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "prepared"
    assert manifest["schema_version"] == 2
    assert set(manifest["instances"]) == set(starts)
    for problem in ("gl", "he"):
        assert len(
            {
                identity["state_sha256"]
                for identity in starts.values()
                if identity["problem"] == problem
            }
        ) == 3


def test_controlled_publication_grid_separates_instances_and_timing_repetitions():
    cases = campaign.build_case_matrix(
        "smoke", "controlled", timing_repetitions=5
    )

    assert len(cases) == 2 * 3 * 5 * 2
    assert {case.timing_repetition for case in cases} == {1, 2, 3, 4, 5}
    assert len({case.key for case in cases}) == len(cases)


def test_controlled_audit_admits_repeated_timing_but_refuses_generalization():
    rows = []
    starts = {}
    for benchmark in ("gl_l5_np2", "he_l2_np2_step1"):
        for instance_index, instance in enumerate(campaign.ROBUSTNESS_INSTANCES, 1):
            file_hash = f"{instance_index}" * 64
            state_hash = f"{instance_index + 3}" * 64
            starts[f"{benchmark}::{instance.key}"] = {
                "file_sha256": file_hash,
                "state_sha256": state_hash,
            }
            for repetition in range(1, 6):
                endpoint_hash = hashlib.sha256(
                    f"{benchmark}:{instance.key}:{repetition}".encode()
                ).hexdigest()
                for method in ("newton_armijo", "reduced_trust_armijo"):
                    rows.append(
                        {
                            "benchmark": benchmark,
                            "robustness_instance": instance.key,
                            "timing_repetition": repetition,
                            "method": method,
                            "result": "completed",
                            "initial_state_file_sha256": file_hash,
                            "initial_state_content_sha256": state_hash,
                            "final_state_file_sha256": "a" * 64,
                            "final_state_content_sha256": endpoint_hash,
                            "endpoint_state_sha256": endpoint_hash,
                            "independent_dual_residual": 1.0e-8,
                            "independent_coefficient_residual": 2.0e-8,
                            "independent_residual_sha256": "b" * 64,
                        }
                    )

    audit = campaign.controlled_identity_audit(
        rows,
        starts,
        expected_repetitions=5,
        expected_instances=[instance.key for instance in campaign.ROBUSTNESS_INSTANCES],
    )

    assert audit["status"] == "passed"
    assert audit["timing_claim_admissible"] is True
    assert audit["tested_instance_comparison_admissible"] is True
    assert audit["robustness_generalization_claim_admissible"] is False


def test_publication_preflight_requires_exact_sha1_commit(monkeypatch):
    monkeypatch.setattr(
        campaign,
        "check_experiment_preflight",
        lambda *_args, **_kwargs: ExperimentPreflight(
            run_kind="publication",
            git_commit="a" * 64,
            git_clean=True,
            git_status_porcelain=(),
            pilot_override=False,
            pilot_override_reason=None,
            checked_at_utc="2026-07-10T00:00:00Z",
        ),
    )

    with pytest.raises(ExperimentPreflightError, match="40-character"):
        campaign.require_publication_preflight()


def test_publication_run_record_is_strictly_validated(tmp_path: Path):
    case = campaign.build_case_matrix(
        "smoke",
        "controlled",
        robustness_instances=(campaign.ROBUSTNESS_BY_KEY["nominal"],),
    )[0]
    start = tmp_path / "start.npz"
    start.write_bytes(b"immutable-start")
    row = {
        "result": "completed",
        "failure_mode": "",
        "returncode": 0,
        "wall_time_s": 1.5,
        "solve_time_s": 1.0,
        "setup_time_s": 0.25,
        "line_search_time_s": 0.1,
        "newton_iters": 2,
        "krylov_iters": 4,
        "robustness_instance": "nominal",
        "robustness_parameters_json": json.dumps(
            campaign._instance_parameters(case.benchmark, case.robustness_instance)
        ),
        "final_state_file_sha256": "1" * 64,
        "final_state_content_sha256": "2" * 64,
        "endpoint_state_sha256": "3" * 64,
        "independent_residual_sha256": "4" * 64,
        "independent_dual_residual": 1.0e-8,
        "independent_coefficient_residual": 2.0e-8,
        "initial_state_file_sha256": "5" * 64,
        "initial_state_content_sha256": "6" * 64,
        "started_at_utc": "2026-07-10T00:00:00Z",
        "finished_at_utc": "2026-07-10T00:00:02Z",
    }
    environment = {
        "python": "3.11",
        "packages": {"numpy": "2.0"},
        "platform": "test-platform",
        "jax": "test-jax",
        "xla": "test-xla",
        "jax_enable_x64": True,
        "petsc": "test-petsc",
        "mpi": "test-mpi",
        "compiler": "test-compiler",
        "blas": "test-blas",
        "cpu_model": "test-cpu",
        "node_model": "test-node",
        "memory_model": "test-memory",
        "scheduler": "local",
        "scheduler_job_id": None,
        "affinity": "0",
    }
    preflight = ExperimentPreflight(
        run_kind="publication",
        git_commit="a" * 40,
        git_clean=True,
        git_status_porcelain=(),
        pilot_override=False,
        pilot_override_reason=None,
        checked_at_utc="2026-07-10T00:00:00Z",
    )
    canonical_start = {
        "path": str(start),
        "file_sha256": hashlib.sha256(start.read_bytes()).hexdigest(),
    }
    command = campaign.build_command(
        case,
        tmp_path / "raw" / case.key / "output.json",
        state_in=start,
        state_out=tmp_path / "raw" / case.key / "final_state.npz",
    )

    record = campaign.build_publication_run_record(
        case=case,
        mode="smoke",
        row=row,
        command=command,
        preflight=preflight,
        environment=environment,
        source_hashes={"runner": "b" * 64},
        campaign_configuration_sha256="c" * 64,
        canonical_start=canonical_start,
        raw_dir=tmp_path / "raw",
    )

    validate_run_record(record, require_publication_ready=True)


def test_controlled_identity_audit_requires_common_start_and_terminal_identity():
    rows = []
    for method in ("newton_armijo", "reduced_trust_armijo"):
        rows.append(
            {
                "benchmark": "gl_l5_np2",
                "method": method,
                "result": "completed",
                "initial_state_file_sha256": "1" * 64,
                "initial_state_content_sha256": "2" * 64,
                "final_state_file_sha256": "3" * 64,
                "final_state_content_sha256": "4" * 64,
                "endpoint_state_sha256": "5" * 64,
                "independent_dual_residual": 1.0e-8,
                "independent_coefficient_residual": 2.0e-8,
                "independent_residual_sha256": "6" * 64,
            }
        )
    audit = campaign.controlled_identity_audit(
        rows,
        {
            "gl_l5_np2": {
                "file_sha256": "1" * 64,
                "state_sha256": "2" * 64,
            }
        },
    )
    assert audit["status"] == "passed"
    assert audit["benchmarks"][0]["endpoint_content_identity_equal"] is True

    rows[1]["initial_state_content_sha256"] = "7" * 64
    failed = campaign.controlled_identity_audit(rows)
    assert failed["status"] == "failed"
    assert any("common hashed start" in error for error in failed["errors"])


def test_solver_state_loaders_validate_mesh_and_convert_to_solver_order(tmp_path: Path):
    scalar_path = tmp_path / "scalar_start.npz"
    scalar_params = {
        "nodes": np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "elems": np.asarray([[0, 1, 2]], dtype=np.int32),
        "u_0": np.zeros(3),
        "freedofs": np.asarray([0, 2], dtype=np.int64),
    }
    scalar_state = np.asarray([1.0, 0.0, 3.0])
    export_scalar_mesh_state_npz(
        scalar_path,
        coords=scalar_params["nodes"],
        triangles=scalar_params["elems"],
        u=scalar_state,
        mesh_level=5,
        problem_name="GinzburgLandau2D",
    )
    scalar_assembler = SimpleNamespace(
        part=SimpleNamespace(perm=np.asarray([1, 0], dtype=np.int64))
    )
    reordered, scalar_identity = scalar_problem_driver._load_scalar_initial_state(
        path=str(scalar_path),
        params=scalar_params,
        assembler=scalar_assembler,
        comm=MPI.COMM_SELF,
        mesh_level=5,
        problem_name="GinzburgLandau2D",
    )
    np.testing.assert_array_equal(reordered, np.asarray([3.0, 1.0]))
    assert scalar_identity["mesh_identity_verified"] is True
    assert len(scalar_identity["file_sha256"]) == 64

    he_path = tmp_path / "he_start.npz"
    coords = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    deformed = coords + np.asarray([0.1, 0.2, 0.3])
    tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    export_hyperelasticity_state_npz(
        he_path,
        coords_ref=coords,
        x_final=deformed,
        tetrahedra=tetrahedra,
        mesh_level=1,
        total_steps=24,
    )
    freedofs = np.asarray([3, 4, 5, 9, 10, 11], dtype=np.int64)
    he_params = {
        "nodes2coord": coords,
        "elems_scalar": tetrahedra,
        "freedofs": freedofs,
    }
    he_assembler = SimpleNamespace(
        part=SimpleNamespace(perm=np.asarray([3, 4, 5, 0, 1, 2], dtype=np.int64))
    )
    he_values, he_identity = he_solver._load_hyperelasticity_initial_state(
        path=str(he_path),
        args=SimpleNamespace(level=1),
        params=he_params,
        assembler=he_assembler,
        comm=MPI.COMM_SELF,
    )
    expected_free = deformed.reshape(-1)[freedofs]
    np.testing.assert_array_equal(he_values, expected_free[he_assembler.part.perm])
    assert he_identity["mesh_identity_verified"] is True
    assert len(he_identity["state_sha256"]) == 64


def test_full_mode_reports_generated_l10_mesh_prerequisite(tmp_path: Path, monkeypatch):
    missing_path = tmp_path / "pLaplace_level10.h5"
    monkeypatch.setattr(campaign, "GENERATED_FULL_MODE_INPUTS", (missing_path,))

    with pytest.raises(SystemExit) as excinfo:
        campaign.require_generated_inputs("full")

    message = str(excinfo.value)
    assert "requires generated scalar level-10 meshes" in message
    assert campaign.GENERATE_L10_COMMAND in message
    campaign.require_generated_inputs("smoke")


def test_scalar_uniform_refinement_splits_each_triangle_and_free_dofs():
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    elems = np.array([[0, 1, 2]], dtype=np.int32)

    refined_nodes, refined_elems = scalar_meshes._uniform_refine_triangles(nodes, elems)
    dvx, dvy, vol = scalar_meshes._triangle_derivatives(refined_nodes, refined_elems)

    assert refined_nodes.shape == (6, 2)
    assert refined_elems.shape == (4, 3)
    np.testing.assert_allclose(vol, np.full(4, 0.125))
    assert np.all(np.isfinite(dvx))
    assert np.all(np.isfinite(dvy))


def test_method_commands_encode_globalization_flags(tmp_path: Path):
    cases = {case.method.key: case for case in campaign.build_case_matrix("full") if case.benchmark.problem == "plaplace"}

    line_cmd = campaign.build_command(cases["newton_linesearch"], tmp_path / "line.json")
    steihaug_cmd = campaign.build_command(cases["steihaug_trust"], tmp_path / "steihaug.json")
    hybrid_cmd = campaign.build_command(cases["hybrid_trust_linesearch"], tmp_path / "hybrid.json")

    assert "--no-use-trust-region" in line_cmd
    assert line_cmd[line_cmd.index("--ksp-type") + 1] == "cg"
    assert "--line-search" in line_cmd
    assert line_cmd[line_cmd.index("--line-search") + 1] == "armijo"

    assert "--use-trust-region" in steihaug_cmd
    assert "--no-trust-subproblem-line-search" in steihaug_cmd
    assert steihaug_cmd[steihaug_cmd.index("--ksp-type") + 1] == "stcg"

    assert "--use-trust-region" in hybrid_cmd
    assert "--trust-subproblem-line-search" in hybrid_cmd
    assert hybrid_cmd[hybrid_cmd.index("--ksp-type") + 1] == "stcg"
    assert hybrid_cmd[hybrid_cmd.index("--line-search") + 1] == "armijo"


def test_gl_line_search_uses_gmres_while_trust_rows_use_stcg(tmp_path: Path):
    cases = {case.method.key: case for case in campaign.build_case_matrix("full") if case.benchmark.problem == "gl"}

    line_cmd = campaign.build_command(cases["newton_linesearch"], tmp_path / "line.json")
    hybrid_cmd = campaign.build_command(cases["hybrid_trust_linesearch"], tmp_path / "hybrid.json")

    assert line_cmd[line_cmd.index("--ksp-type") + 1] == "gmres"
    assert hybrid_cmd[hybrid_cmd.index("--ksp-type") + 1] == "stcg"


def test_hyperelasticity_line_search_does_not_use_trust_ksp(tmp_path: Path):
    cases = {case.method.key: case for case in campaign.build_case_matrix("full") if case.benchmark.problem == "he"}

    line_cmd = campaign.build_command(cases["newton_linesearch"], tmp_path / "line.json")
    hybrid_cmd = campaign.build_command(cases["hybrid_trust_linesearch"], tmp_path / "hybrid.json")

    assert "--no-use-trust-region" in line_cmd
    assert line_cmd[line_cmd.index("--ksp-type") + 1] == "gmres"
    assert hybrid_cmd[hybrid_cmd.index("--ksp-type") + 1] == "stcg"


def test_plasticity3d_rows_use_backend_mix_runner_and_method_flags(tmp_path: Path):
    cases = {
        case.method.key: case
        for case in campaign.build_case_matrix("full")
        if case.benchmark.problem == "plasticity3d"
    }

    line_cmd = campaign.build_command(cases["newton_linesearch"], tmp_path / "line.json")
    steihaug_cmd = campaign.build_command(cases["steihaug_trust"], tmp_path / "steihaug.json")
    hybrid_cmd = campaign.build_command(cases["hybrid_trust_linesearch"], tmp_path / "hybrid.json")

    assert "run_plasticity3d_backend_mix_case.py" in " ".join(line_cmd)
    assert "--mesh-name" in line_cmd
    assert line_cmd[line_cmd.index("--mesh-name") + 1] == "hetero_ssr_L1"
    assert line_cmd[line_cmd.index("--elem-degree") + 1] == "2"
    assert line_cmd[line_cmd.index("--lambda-target") + 1] == "1.55"
    assert line_cmd[line_cmd.index("--grad-stop-tol") + 1] == "1e-4"
    assert "--no-use-trust-region" in line_cmd
    assert "--use-trust-region" in steihaug_cmd
    assert "--no-trust-subproblem-line-search" in steihaug_cmd
    assert "--trust-subproblem-line-search" in hybrid_cmd


def test_summary_parses_completed_payload(tmp_path: Path):
    case = campaign.build_case_matrix("smoke")[0]
    payload = {
        "case": {"backend": "element"},
        "result": {
            "setup_time": 0.25,
            "solve_time_total": 1.5,
            "total_time": 2.0,
            "steps": [
                {
                    "step": 1,
                    "nit": 3,
                    "linear_iters": 7,
                    "energy": -1.25,
                    "message": "Converged (energy, step, gradient)",
                    "history": [{"ls_evals": 2, "t_ls": 0.1, "trust_rejects": 1}],
                }
            ],
        },
    }

    row = campaign.summarize_payload(
        mode="smoke",
        case=case,
        payload=payload,
        json_path=tmp_path / "out.json",
        log_path=tmp_path / "run.log",
        command=["mpiexec", "-n", "2"],
        returncode=0,
        wall_time_s=2.5,
    )

    assert row["result"] == "completed"
    assert row["completed_steps"] == 1
    assert row["newton_iters"] == 3
    assert row["krylov_iters"] == 7
    assert row["line_search_evals"] == 2
    assert row["trust_rejects"] == 1
    assert row["final_energy"] == -1.25


def test_summary_parses_plasticity3d_payload(tmp_path: Path):
    case = next(case for case in campaign.build_case_matrix("full") if case.benchmark.problem == "plasticity3d")
    payload = {
        "status": "completed",
        "message": "Converged",
        "nit": 4,
        "solve_time": 12.5,
        "total_time": 14.0,
        "linear_iterations_total": 19,
        "energy": -3.25,
        "history": [
            {"ls_evals": 1, "t_ls": 0.1, "trust_rejects": 0},
            {"ls_evals": 2, "t_ls": 0.2, "trust_rejects": 1},
        ],
    }

    row = campaign.summarize_payload(
        mode="full",
        case=case,
        payload=payload,
        json_path=tmp_path / "out.json",
        log_path=tmp_path / "run.log",
        command=["mpiexec", "-n", "32"],
        returncode=0,
        wall_time_s=15.0,
    )

    assert row["result"] == "completed"
    assert row["completed_steps"] == 1
    assert row["newton_iters"] == 4
    assert row["krylov_iters"] == 19
    assert row["line_search_evals"] == 3
    assert row["trust_rejects"] == 1
    assert row["final_energy"] == -3.25


def test_summary_preserves_timeout_as_reportable_row(tmp_path: Path):
    case = campaign.build_case_matrix("smoke")[0]
    payload = {
        "case": {},
        "result": {
            "status": "timeout",
            "failure_mode": "timeout",
            "steps": [],
        },
    }

    row = campaign.summarize_payload(
        mode="smoke",
        case=case,
        payload=payload,
        json_path=tmp_path / "out.json",
        log_path=tmp_path / "run.log",
        command=["mpiexec", "-n", "2"],
        returncode=-15,
        wall_time_s=20.0,
        launcher_failure="timeout",
    )

    assert row["result"] == "timeout"
    assert row["failure_mode"] == "timeout"
    assert row["completed_steps"] == 0
    assert row["newton_iters"] == 0


def test_paper_table_rows_sort_by_benchmark_and_method(tmp_path: Path):
    csv_path = tmp_path / "full_summary.csv"
    fields = [
        "benchmark",
        "benchmark_label",
        "method",
        "method_label",
        "result",
        "completed_steps",
        "steps_requested",
        "newton_iters",
        "krylov_iters",
        "line_search_evals",
        "trust_rejects",
        "solve_time_s",
        "wall_time_s",
        "final_energy",
    ]
    rows = [
        {
            "benchmark": "he_l4_np32_steps8",
            "benchmark_label": "HyperElasticity L4",
            "method": "hybrid_trust_linesearch",
            "method_label": "Hybrid",
            "result": "completed",
            "completed_steps": "8",
            "steps_requested": "8",
            "newton_iters": "20",
            "krylov_iters": "40",
            "line_search_evals": "8",
            "trust_rejects": "0",
            "solve_time_s": "12.0",
            "wall_time_s": "13.0",
            "final_energy": "1.0",
        },
        {
            "benchmark": "plaplace_l10_np32",
            "benchmark_label": "p-Laplace L10",
            "method": "newton_linesearch",
            "method_label": "Newton",
            "result": "completed",
            "completed_steps": "1",
            "steps_requested": "1",
            "newton_iters": "6",
            "krylov_iters": "11",
            "line_search_evals": "6",
            "trust_rejects": "0",
            "solve_time_s": "1.0",
            "wall_time_s": "2.0",
            "final_energy": "-7.0",
        },
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    scripts_dir = Path("paper/scripts").resolve()
    sys.path.insert(0, str(scripts_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "generate_paper_tables_for_test",
            scripts_dir / "generate_paper_tables.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts_dir))

    sorted_rows = module.globalization_method_rows(csv_path)

    assert [row["benchmark"] for row in sorted_rows] == ["plaplace_l10_np32", "he_l4_np32_steps8"]

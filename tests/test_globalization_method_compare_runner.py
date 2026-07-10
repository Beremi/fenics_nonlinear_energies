from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.runners import generate_scalar_uniform_l10_meshes as scalar_meshes
from experiments.runners import run_globalization_method_compare as campaign
from experiments.runners import run_trust_region_case as case_runner


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


def test_controlled_matrix_excludes_unfrozen_plasticity_and_holds_ksp_fixed(
    tmp_path: Path,
):
    cases = campaign.build_case_matrix("full", "controlled")

    assert len(cases) == 6
    assert {case.benchmark.problem for case in cases} == {"plaplace", "gl", "he"}
    assert {case.method.key for case in cases} == {
        "newton_armijo",
        "reduced_trust_armijo",
    }
    assert {case.comparison_tier for case in cases} == {"controlled"}

    for problem in ("plaplace", "gl", "he"):
        problem_cases = [case for case in cases if case.benchmark.problem == problem]
        commands = [
            campaign.build_command(case, tmp_path / f"{case.key}.json")
            for case in problem_cases
        ]
        ksp_types = {
            command[command.index("--ksp-type") + 1] for command in commands
        }
        assert ksp_types == {problem_cases[0].benchmark.line_ksp_type}
        assert all(command[command.index("--line-search") + 1] == "armijo" for command in commands)

    trust_case = next(case for case in cases if case.method.key == "reduced_trust_armijo")
    trust_command = campaign.build_command(trust_case, tmp_path / "trust.json")
    assert "--use-trust-region" in trust_command
    assert "--no-trust-subproblem-line-search" in trust_command


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

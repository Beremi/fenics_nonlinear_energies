from __future__ import annotations

import csv
from pathlib import Path

from experiments.runners import run_derivative_route_compare as campaign


def test_full_matrix_matches_requested_derivative_routes():
    cases = campaign.build_case_matrix("full")

    assert len(cases) == 7
    assert {(case.problem, case.route, case.nprocs) for case in cases} == {
        ("plaplace", "element_ad", 32),
        ("plaplace", "colored_sfd", 32),
        ("he", "element_ad", 32),
        ("he", "colored_sfd", 32),
        ("plasticity3d", "element_ad", 32),
        ("plasticity3d", "colored_sfd", 32),
        ("plasticity3d", "constitutive_ad", 32),
    }
    assert {case.level for case in cases if case.problem == "he"} == {4}
    assert {case.level for case in cases if case.problem == "plasticity3d"} == {1}


def test_smoke_matrix_keeps_plaplace_mined_and_uses_np2_for_new_runs():
    cases = campaign.build_case_matrix("smoke")

    assert [case.runner for case in cases[:2]] == ["plaplace_docs", "plaplace_docs"]
    assert {case.nprocs for case in cases if case.runner != "plaplace_docs"} == {2}
    assert {case.level for case in cases if case.problem == "he"} == {1}


def test_hyperelasticity_commands_use_replicated_element_and_sfd_paths(tmp_path: Path):
    cases = {
        case.route: case
        for case in campaign.build_case_matrix("full")
        if case.problem == "he"
    }

    element_cmd = campaign.build_command(cases["element_ad"], tmp_path / "element.json")
    sfd_cmd = campaign.build_command(cases["colored_sfd"], tmp_path / "sfd.json")

    for cmd in (element_cmd, sfd_cmd):
        assert "--problem-build-mode" in cmd
        assert cmd[cmd.index("--problem-build-mode") + 1] == "replicated"
        assert cmd[cmd.index("--distribution-strategy") + 1] == "overlap_allgather"
        assert cmd[cmd.index("--assembly-backend") + 1] == "coo"
        assert cmd[cmd.index("--steps") + 1] == "1"
        assert cmd[cmd.index("--ksp-type") + 1] == "stcg"

    assert element_cmd[element_cmd.index("--local-hessian-mode") + 1] == "element"
    assert sfd_cmd[sfd_cmd.index("--local-hessian-mode") + 1] == "sfd_local"


def test_plasticity3d_commands_use_p2_lambda155_and_requested_backends(tmp_path: Path):
    cases = {
        case.route: case
        for case in campaign.build_case_matrix("full")
        if case.problem == "plasticity3d"
    }

    expected_backends = {
        "element_ad": "local",
        "colored_sfd": "local_sfd",
        "constitutive_ad": "local_constitutiveAD",
    }
    for route, backend in expected_backends.items():
        cmd = campaign.build_command(cases[route], tmp_path / route / "output.json")
        assert "run_plasticity3d_backend_mix_case.py" in " ".join(cmd)
        assert cmd[cmd.index("--assembly-backend") + 1] == backend
        assert cmd[cmd.index("--solver-backend") + 1] == "local_pmg_mumps"
        assert cmd[cmd.index("--elem-degree") + 1] == "2"
        assert cmd[cmd.index("--lambda-target") + 1] == "1.55"
        assert cmd[cmd.index("--grad-stop-tol") + 1] == "1e-4"
        assert "--use-trust-region" in cmd
        assert "--trust-subproblem-line-search" in cmd


def test_plaplace_summary_mines_tracked_docs_scaling(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "strong_scaling.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "solver",
                "nprocs",
                "total_time_s",
                "final_energy",
                "total_newton_iters",
                "total_linear_iters",
                "result",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "solver": "jax_petsc_local_sfd",
                "nprocs": "32",
                "total_time_s": "1.5",
                "final_energy": "-7.96",
                "total_newton_iters": "6",
                "total_linear_iters": "11",
                "result": "completed",
            }
        )
    monkeypatch.setattr(campaign, "PLAPLACE_SCALING", csv_path)
    case = next(case for case in campaign.build_case_matrix("full") if case.key.endswith("colored_sfd"))

    row = campaign.summarize_plaplace_case(case, mode="full")

    assert row["result"] == "completed"
    assert row["route"] == "colored_sfd"
    assert row["nprocs"] == 32
    assert row["total_time_s"] == 1.5
    assert row["krylov_iters"] == 11


def test_plasticity3d_summary_extracts_hessian_and_sfd_color_metadata(tmp_path: Path):
    case = next(
        case
        for case in campaign.build_case_matrix("full")
        if case.problem == "plasticity3d" and case.route == "colored_sfd"
    )
    payload = {
        "status": "completed",
        "message": "Converged",
        "solve_time": 12.0,
        "total_time": 13.0,
        "nit": 3,
        "linear_iterations_total": 21,
        "history": [{"ls_evals": 1, "trust_rejects": 0}],
        "energy": -1.0,
        "omega": 2.0,
        "u_max": 3.0,
        "assembly_callbacks": {
            "hessian": {
                "hvp_compute": 4.5,
                "total": 5.5,
            }
        },
        "assembler_rank_diagnostics": {
            "sfd_coloring": {
                "colors_min": 12,
                "colors_max": 18,
                "colors_unique": [12, 15, 18],
            }
        },
    }

    row = campaign.summarize_plasticity3d_payload(
        mode="full",
        case=case,
        payload=payload,
        json_path=tmp_path / "output.json",
        log_path=tmp_path / "run.log",
        command=["true"],
        returncode=0,
        wall_time_s=14.0,
    )

    assert row["result"] == "completed"
    assert row["hessian_hvp_time_s"] == 4.5
    assert row["hessian_time_s"] == 5.5
    assert row["sfd_colors_min"] == 12
    assert row["sfd_colors_max"] == 18
    assert row["sfd_colors_unique"] == "12 15 18"

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiments.runners import run_paper_reviewer_gap_experiments as campaign


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_case_matrices_match_requested_shapes():
    smoke = campaign.build_case_matrix("smoke")
    full = campaign.build_case_matrix("full")

    assert len(smoke) == 14
    assert len(full) == 32
    assert {case.section for case in full} == set(campaign.SECTIONS)
    assert sum(case.section == "he_distribution" for case in full) == 8
    assert sum(case.section == "he_pmg" for case in full) == 4
    assert sum(case.section == "topology_consistency" for case in full) == 6
    assert sum(case.section == "gl_globalization" for case in full) == 3
    assert sum(case.section == "p3d_derivative_degree" for case in full) == 11


def test_he_distribution_commands_encode_replicated_and_rank_local_modes():
    cases = {
        case.metadata["build_mode"]: case
        for case in campaign.build_case_matrix("full")
        if case.section == "he_distribution" and case.metadata["probe"] == "correctness"
    }

    replicated = campaign.build_command("full", cases["replicated"])
    rank_local = campaign.build_command("full", cases["rank_local"])

    assert replicated[replicated.index("--problem-build-mode") + 1] == "replicated"
    assert replicated[replicated.index("--he-mesh-source") + 1] == "hdf5"
    assert replicated[replicated.index("--distribution-strategy") + 1] == "overlap_allgather"
    assert replicated[replicated.index("--assembly-backend") + 1] == "coo"

    assert rank_local[rank_local.index("--problem-build-mode") + 1] == "rank_local"
    assert rank_local[rank_local.index("--he-mesh-source") + 1] == "procedural"
    assert rank_local[rank_local.index("--distribution-strategy") + 1] == "overlap_p2p"
    assert rank_local[rank_local.index("--assembly-backend") + 1] == "coo_local"


def test_he_pmg_candidates_encode_coarse_solver_choices():
    cases = {
        case.metadata["candidate"]: campaign.build_command("full", case)
        for case in campaign.build_case_matrix("full")
        if case.section == "he_pmg"
    }

    assert cases["gamg"][cases["gamg"].index("--pc-type") + 1] == "gamg"
    assert cases["pmg_l2_hypre"][cases["pmg_l2_hypre"].index("--pc-type") + 1] == "mg"
    assert cases["pmg_l2_hypre"][cases["pmg_l2_hypre"].index("--he-pmg-coarsest-level") + 1] == "2"
    assert cases["pmg_l2_hypre"][cases["pmg_l2_hypre"].index("--he-pmg-coarse-pc-type") + 1] == "hypre"
    assert cases["pmg_l2_redundant_mumps"][
        cases["pmg_l2_redundant_mumps"].index("--he-pmg-coarse-pc-type") + 1
    ] == "redundant"
    assert cases["pmg_l2_redundant_mumps"][
        cases["pmg_l2_redundant_mumps"].index("--he-pmg-coarse-factor-solver-type") + 1
    ] == "mumps"
    assert cases["pmg_l3_redundant_mumps"][
        cases["pmg_l3_redundant_mumps"].index("--he-pmg-coarsest-level") + 1
    ] == "3"


def test_topology_and_gl_commands_encode_fixed_schedule_and_l10_cleanup():
    topology = next(case for case in campaign.build_case_matrix("full") if case.section == "topology_consistency")
    topo_cmd = campaign.build_command("full", topology)
    assert "--fixed_outer_schedule" in topo_cmd
    assert topo_cmd[topo_cmd.index("--outer_maxit") + 1] == "40"
    assert "--state_out" in topo_cmd

    gl_cases = {
        case.metadata["method"]: campaign.build_command("full", case)
        for case in campaign.build_case_matrix("full")
        if case.section == "gl_globalization"
    }
    line = gl_cases["newton_linesearch"]
    hybrid = gl_cases["hybrid_trust_linesearch"]
    assert line[line.index("--level") + 1] == "10"
    assert line[line.index("-n") + 1] == "8"
    assert line[line.index("--ksp-type") + 1] == "gmres"
    assert "--no-use-trust-region" in line
    assert hybrid[hybrid.index("--ksp-type") + 1] == "stcg"
    assert "--trust-subproblem-line-search" in hybrid


def test_p3d_full_matrix_uses_degree_specific_routes_and_pmg():
    cases = [
        case
        for case in campaign.build_case_matrix("full")
        if case.section == "p3d_derivative_degree"
    ]

    assert {(case.metadata["mesh_case"], case.metadata["route"]) for case in cases} == {
        ("p1_l1", "element_ad"),
        ("p1_l1", "colored_sfd"),
        ("p1_l1", "constitutive_ad"),
        ("p1_l1_2", "element_ad"),
        ("p1_l1_2", "colored_sfd"),
        ("p1_l1_2", "constitutive_ad"),
        ("p2_l1", "element_ad"),
        ("p2_l1", "colored_sfd"),
        ("p2_l1", "constitutive_ad"),
        ("p4_l1", "element_ad"),
        ("p4_l1", "constitutive_ad"),
    }
    expected = {
        "p1_l1": ("hetero_ssr_L1", "uniform_refined_p1_chain"),
        "p1_l1_2": ("hetero_ssr_L1_2", "uniform_refined_p1_chain"),
        "p2_l1": ("hetero_ssr_L1", "same_mesh_p2_p1"),
        "p4_l1": ("hetero_ssr_L1", "same_mesh_p4_p2_p1"),
    }
    for case in cases:
        cmd = campaign.build_command("full", case)
        mesh_name, pmg_strategy = expected[str(case.metadata["mesh_case"])]
        assert cmd[cmd.index("--lambda-target") + 1] == "1.55"
        assert cmd[cmd.index("--mesh-name") + 1] == mesh_name
        assert cmd[cmd.index("--pmg-strategy") + 1] == pmg_strategy
        assert cmd[cmd.index("--maxit") + 1] == "1"
    assert not any(
        case.metadata["mesh_case"] == "p4_l1" and case.metadata["route"] == "colored_sfd"
        for case in cases
    )


def test_he_summary_preserves_timeout_and_extracts_memory(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(campaign, "RAW_ROOT", tmp_path / "raw")
    case = next(
        case
        for case in campaign.build_case_matrix("full")
        if case.section == "he_distribution"
        and case.metadata["probe"] == "memory"
        and case.metadata["build_mode"] == "rank_local"
        and case.nprocs == 8
    )
    case_dir, output_json, _log = campaign._case_paths("full", case)
    _write_json(case_dir / "case_metadata.json", {"command": "mpiexec -n 8 ..."})
    _write_json(case_dir / "run_info.json", {"returncode": 0, "wall_time_s": 12.0})
    _write_json(
        output_json,
        {
            "result": {
                "free_dofs": 1000,
                "metadata": {
                    "linear_solver": {
                        "resource_usage": {
                            "ru_maxrss_mib_max": 256.0,
                            "ru_maxrss_mib_total": 1024.0,
                        },
                        "assembler_memory_by_rank": {
                            "tracked_total_gib_total": 1.5,
                            "local_elements_max": 33,
                            "local_overlap_dofs_max": 120,
                            "local_overlap_dofs_total": 1800,
                        },
                        "problem_build_mode": "rank_local",
                        "mesh_source": "procedural",
                        "assembly_backend": "coo_local",
                    }
                },
                "steps": [
                    {
                        "nit": 1,
                        "linear_iters": 7,
                        "energy": 0.25,
                        "message": "Maximum nonlinear iterations reached",
                    }
                ],
            }
        },
    )

    row = campaign.summarize_he_distribution("full", [case])[0]

    assert row["result"] == "fixed_work"
    assert row["ru_maxrss_mib_total"] == 1024.0
    assert row["tracked_total_gib_total"] == 1.5
    assert row["local_elements_max"] == 33
    assert row["overlap_owned_ratio"] == 1.8


def test_topology_summary_computes_rank_consistency_against_np1(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(campaign, "RAW_ROOT", tmp_path / "raw")
    cases = [
        case
        for case in campaign.build_case_matrix("smoke")
        if case.section == "topology_consistency"
    ]
    for case, compliance, theta in (
        (cases[0], 10.0, np.ones((2, 2))),
        (cases[1], 10.4, np.array([[1.0, 0.9], [1.0, 0.9]])),
    ):
        case_dir, output_json, _log = campaign._case_paths("smoke", case)
        _write_json(case_dir / "run_info.json", {"returncode": 0, "wall_time_s": 1.0})
        _write_json(
            output_json,
            {
                "result": "fixed_work_completed",
                "time": 2.0,
                "setup_time": 0.5,
                "final_metrics": {
                    "outer_iterations": 3,
                    "final_compliance": compliance,
                    "final_volume_fraction": 0.4,
                    "final_p_penal": 1.6,
                },
            },
        )
        np.savez(case_dir / "state.npz", theta_grid=theta)

    rows = campaign.summarize_topology("smoke", cases)

    assert rows[0]["density_rel_l2_vs_np1"] == 0.0
    assert np.isclose(rows[1]["compliance_rel_diff_vs_np1"], 0.04)
    assert rows[1]["density_rel_l2_vs_np1"] > 0.0


def test_p3d_summary_extracts_degree_color_and_resource_metadata(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(campaign, "RAW_ROOT", tmp_path / "raw")
    case = next(
        case
        for case in campaign.build_case_matrix("full")
        if case.section == "p3d_derivative_degree"
        and case.metadata["mesh_case"] == "p1_l1_2"
        and case.metadata["route"] == "colored_sfd"
    )
    case_dir, output_json, _log = campaign._case_paths("full", case)
    _write_json(case_dir / "run_info.json", {"returncode": 0, "wall_time_s": 11.0})
    _write_json(
        output_json,
        {
            "status": "completed",
            "nit": 1,
            "linear_iterations_total": 9,
            "solve_time": 7.0,
            "total_time": 8.0,
            "energy": -12.5,
            "final_grad_norm": 1e-3,
            "assembly_callbacks": {"hessian": {"hvp_compute": 2.0, "total": 2.5}},
            "assembler_rank_diagnostics": {
                "sfd_coloring": {"colors_min": 120, "colors_max": 240},
                "resource_usage": {"ru_maxrss_mib_max": 2048.0, "ru_maxrss_mib_total": 4096.0},
            },
            "assembler_memory": {"local_elements": 10, "local_overlap_dofs": 400},
        },
    )

    row = campaign.summarize_p3d("full", [case])[0]

    assert row["discretization"] == "P1(L2)"
    assert row["mesh_name"] == "hetero_ssr_L1_2"
    assert row["free_dofs"] == 79024
    assert row["local_element_dofs"] == 12
    assert row["sfd_colors_min"] == 120
    assert row["sfd_colors_max"] == 240
    assert row["ru_maxrss_mib_max"] == 2048.0
    assert row["local_overlap_dofs"] == 400
    assert row["finite_metrics"] is True


def test_summarize_writes_section_csvs(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(campaign, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(campaign, "REPORT_ROOT", tmp_path / "reports")
    selected = {"he_distribution"}
    case = next(case for case in campaign.build_case_matrix("smoke") if case.section == "he_distribution")
    case_dir, _output_json, _log = campaign._case_paths("smoke", case)
    _write_json(case_dir / "run_info.json", {"returncode": 1, "timed_out": True, "wall_time_s": 2.0})

    campaign.summarize("smoke", selected)

    csv_path = tmp_path / "reports" / "smoke_he_distribution.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["result"] == "timeout"


def test_run_cases_launches_full_p3d_matrix_without_p4_sfd_guard(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(campaign, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(campaign, "REPORT_ROOT", tmp_path / "reports")
    cases = [
        case
        for case in campaign.build_case_matrix("full")
        if case.section == "p3d_derivative_degree"
    ]
    monkeypatch.setattr(campaign, "build_case_matrix", lambda mode: cases)
    launched = []

    def fake_run(*args, **kwargs):
        launched.append((args, kwargs))
        return {"returncode": 0, "timed_out": False, "wall_time_s": 0.0}

    monkeypatch.setattr(campaign, "_run_subprocess", fake_run)
    campaign.run_cases(
        "full",
        {"p3d_derivative_degree"},
        resume=True,
        campaign_wall_s=None,
        allow_oom_risk=False,
    )

    assert len(launched) == 11
    assert not any("p3d_p4_l1_colored_sfd" in str(call) for call in launched)

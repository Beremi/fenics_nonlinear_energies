from __future__ import annotations

import pytest

from src.core.benchmark.results import (
    cumulative_linear_timing,
    summarize_load_step_case,
    summarize_pure_jax_load_step_case,
    summarize_single_step_case,
    sum_step_history,
    sum_step_linear,
    sum_step_linear_time,
)


def test_step_aggregation_helpers_cover_old_and_new_shapes():
    step = {
        "history": [{"t_ls": 0.1}, {"t_ls": 0.2}],
        "linear_timing": [
            {"ksp_its": 3, "assemble_time": 0.4, "pc_setup_time": 0.5, "solve_time": 0.6},
            {
                "ksp_its": 4,
                "assemble_total_time": 0.7,
                "pc_setup_time": 0.8,
                "solve_time": 0.9,
            },
        ],
    }

    assert sum_step_linear(step) == 7
    assert sum_step_linear_time(step, "assemble_time") == 0.4
    assert sum_step_linear_time(step, "assemble_total_time") == 0.7
    assert sum_step_history(step, "t_ls") == pytest.approx(0.3)
    totals = cumulative_linear_timing(step["linear_timing"])
    assert totals["asm_time_cumulative"] == pytest.approx(1.1)
    assert totals["pc_setup_time_cumulative"] == pytest.approx(1.3)
    assert totals["linear_solve_time_cumulative"] == pytest.approx(1.5)
    assert totals["ksp_time_cumulative"] == pytest.approx(2.8)


def test_summarize_single_step_case_matches_authoritative_row_shape():
    payload = {
        "case": {"backend": "fenics"},
        "result": {
            "solve_time_total": 3.5,
            "steps": [
                {
                    "step": 1,
                    "nit": 5,
                    "time": 3.5,
                    "energy": -7.25,
                    "message": "Converged (energy, gradient)",
                    "history": [{"t_ls": 0.05}, {"t_ls": 0.07}],
                    "linear_timing": [
                        {"ksp_its": 4, "assemble_total_time": 0.3, "pc_setup_time": 0.2, "solve_time": 0.4},
                        {"ksp_its": 5, "assemble_total_time": 0.1, "pc_setup_time": 0.2, "solve_time": 0.3},
                    ],
                }
            ],
        },
    }

    row = summarize_single_step_case("fenics_custom", 5, 2, payload)

    assert row["solver"] == "fenics_custom"
    assert row["backend"] == "fenics"
    assert row["level"] == 5
    assert row["nprocs"] == 2
    assert row["completed_steps"] == 1
    assert row["first_failed_step"] is None
    assert row["failure_mode"] is None
    assert row["failure_time_s"] is None
    assert row["total_newton_iters"] == 5
    assert row["total_linear_iters"] == 9
    assert row["total_time_s"] == pytest.approx(3.5)
    assert row["mean_step_time_s"] == pytest.approx(3.5)
    assert row["max_step_time_s"] == pytest.approx(3.5)
    assert row["assembly_time_s"] == pytest.approx(0.4)
    assert row["pc_init_time_s"] == pytest.approx(0.4)
    assert row["ksp_solve_time_s"] == pytest.approx(0.7)
    assert row["line_search_time_s"] == pytest.approx(0.12)
    assert row["final_energy"] == pytest.approx(-7.25)
    assert row["result"] == "completed"


def test_summarize_load_step_case_handles_kill_switch_and_trust_rejects():
    payload = {
        "case": {"backend": "element"},
        "result": {
            "solve_time_total": 4.0,
            "steps": [
                {
                    "step": 1,
                    "nit": 4,
                    "time": 1.5,
                    "energy": 1.2,
                    "message": "Converged",
                    "history": [{"t_ls": 0.1, "trust_rejects": 1}],
                    "linear_timing": [{"ksp_its": 6, "assemble_time": 0.2, "pc_setup_time": 0.1, "solve_time": 0.3}],
                },
                {
                    "step": 2,
                    "nit": 3,
                    "time": 2.5,
                    "energy": 1.1,
                    "message": "Stopped by kill switch",
                    "kill_switch_exceeded": True,
                    "history": [{"t_ls": 0.2, "trust_rejects": 2}],
                    "linear_timing": [{"ksp_its": 7, "assemble_time": 0.4, "pc_setup_time": 0.2, "solve_time": 0.5}],
                },
            ],
        },
    }

    row = summarize_load_step_case("jax_petsc_element", 24, 1, 2, payload)

    assert row["solver"] == "jax_petsc_element"
    assert row["backend"] == "element"
    assert row["total_steps"] == 24
    assert row["level"] == 1
    assert row["nprocs"] == 2
    assert row["completed_steps"] == 1
    assert row["first_failed_step"] == 2
    assert row["failure_mode"] == "kill-switch"
    assert row["failure_time_s"] == pytest.approx(2.5)
    assert row["total_newton_iters"] == 7
    assert row["total_linear_iters"] == 13
    assert row["total_time_s"] == pytest.approx(4.0)
    assert row["mean_step_time_s"] == pytest.approx(2.0)
    assert row["max_step_time_s"] == pytest.approx(2.5)
    assert row["assembly_time_s"] == pytest.approx(0.6)
    assert row["pc_init_time_s"] == pytest.approx(0.3)
    assert row["ksp_solve_time_s"] == pytest.approx(0.8)
    assert row["line_search_time_s"] == pytest.approx(0.3)
    assert row["trust_rejects"] == 3
    assert row["final_energy"] == pytest.approx(1.1)
    assert row["result"] == "kill-switch"


def test_summarize_pure_jax_load_step_case_matches_runner_schema():
    payload = {
        "solver": "pure_jax",
        "level": 1,
        "total_steps": 24,
        "total_dofs": 2187,
        "free_dofs": 2133,
        "time": 41.5,
        "total_newton_iters": 559,
        "total_linear_iters": 2284,
        "result": "completed",
        "steps": [
            {"step": 1, "time": 1.5},
            {"step": 2, "time": 2.5},
        ],
    }

    assert summarize_pure_jax_load_step_case(payload) == {
        "solver": "pure_jax",
        "level": 1,
        "total_steps": 24,
        "total_dofs": 2187,
        "free_dofs": 2133,
        "time": 41.5,
        "total_newton_iters": 559,
        "total_linear_iters": 2284,
        "max_step_time": 2.5,
        "result": "completed",
    }

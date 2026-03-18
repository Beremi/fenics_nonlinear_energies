from __future__ import annotations

from pathlib import Path

from experiments.analysis import (
    generate_parallel_full_report,
    generate_parallel_scaling_stallstop_report,
    generate_report_assets,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_serial_topology_report_generator_uses_canonical_paths():
    assert generate_report_assets.REPO_ROOT == REPO_ROOT
    assert generate_report_assets.DEFAULT_ASSET_DIR == (
        REPO_ROOT / "artifacts" / "raw_results" / "topology_reports" / "serial_reference"
    )
    assert generate_report_assets.DEFAULT_REPORT_PATH == (
        generate_report_assets.DEFAULT_ASSET_DIR / "report.md"
    )
    assert generate_report_assets.REPORT_GENERATOR_CMD == (
        "./.venv/bin/python experiments/analysis/generate_report_assets.py"
    )

    solver_cmd = generate_report_assets._solver_command(generate_report_assets.CASE_PARAMS)
    assert "src/problems/topology/jax/solve_topopt_jax.py" in solver_cmd
    assert "topological_optimisation_jax" not in solver_cmd


def test_parallel_topology_generators_use_canonical_paths():
    expected_root = REPO_ROOT / "artifacts" / "raw_results" / "topology_reports"

    assert generate_parallel_full_report.REPO_ROOT == REPO_ROOT
    assert generate_parallel_full_report.DEFAULT_ASSET_DIR == expected_root / "parallel_final"
    assert generate_parallel_scaling_stallstop_report.REPO_ROOT == REPO_ROOT
    assert generate_parallel_scaling_stallstop_report.DEFAULT_ASSET_DIR == expected_root / "scaling"
    assert generate_parallel_scaling_stallstop_report.SOLVER == (
        REPO_ROOT / "src" / "problems" / "topology" / "jax" / "solve_topopt_parallel.py"
    )

    solver_cmd = generate_parallel_full_report._solver_command(
        {
            "nprocs": 2,
            "parameters": {
                "nx": 768,
                "ny": 384,
                "length": 2.0,
                "height": 1.0,
                "traction": 1.0,
                "load_fraction": 0.2,
                "fixed_pad_cells": 32,
                "load_pad_cells": 32,
                "volume_fraction_target": 0.4,
                "theta_min": 1e-6,
                "solid_latent": 10.0,
                "young": 1.0,
                "poisson": 0.3,
                "alpha_reg": 0.005,
                "ell_pf": 0.08,
                "mu_move": 0.01,
                "beta_lambda": 12.0,
                "volume_penalty": 10.0,
                "p_start": 1.0,
                "p_max": 10.0,
                "p_increment": 0.2,
                "continuation_interval": 1,
                "outer_maxit": 2000,
                "outer_tol": 0.02,
                "volume_tol": 0.001,
                "design_maxit": 20,
                "tolf": 1e-6,
                "tolg": 1e-3,
                "linesearch_tol": 0.1,
                "mechanics_ksp_rtol": 1e-4,
                "mechanics_ksp_max_it": 100,
            },
            "mesh": {},
            "solver_options": {
                "design_gd_line_search": "golden_adaptive",
                "design_gd_adaptive_window_scale": 2.0,
                "linesearch_relative_to_bound": True,
            },
        },
        REPO_ROOT / "artifacts" / "reproduction" / "campaign" / "frames",
        REPO_ROOT / "artifacts" / "reproduction" / "campaign" / "parallel_full_run.json",
        REPO_ROOT / "artifacts" / "reproduction" / "campaign" / "parallel_full_state.npz",
    )
    assert "src/problems/topology/jax/solve_topopt_parallel.py" in solver_cmd
    assert "topological_optimisation_jax" not in solver_cmd

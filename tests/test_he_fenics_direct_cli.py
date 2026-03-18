from __future__ import annotations

from src.problems.hyperelasticity.fenics import solve_HE_custom_jaxversion


def test_he_direct_cli_defaults_match_maintained_profile():
    parser = solve_HE_custom_jaxversion._build_parser()
    args = parser.parse_args([])

    assert args.ksp_type == "stcg"
    assert args.pc_type == "gamg"
    assert args.ksp_rtol == 1e-1
    assert args.ksp_max_it == 30
    assert args.linesearch_tol == 1e-1
    assert args.require_all_convergence is True
    assert args.use_trust_region is True
    assert args.trust_radius_init == 0.5
    assert args.trust_subproblem_line_search is True
    assert args.pc_setup_on_ksp_cap is False
    assert args.gamg_set_coordinates is True
    assert args.use_near_nullspace is True


def test_he_direct_cli_accepts_canonical_and_legacy_aliases():
    parser = solve_HE_custom_jaxversion._build_parser()

    args = parser.parse_args(
        [
            "--start-step",
            "3",
            "--total_steps",
            "96",
            "--linesearch_a",
            "-0.25",
            "--linesearch-b",
            "1.5",
            "--trust_radius_init",
            "1.0",
            "--no-trust-subproblem-line-search",
            "--no_retry_on_maxit",
            "--no_gamg_coordinates",
            "--no_near_nullspace",
        ]
    )

    assert args.start_step == 3
    assert args.total_steps == 96
    assert args.linesearch_a == -0.25
    assert args.linesearch_b == 1.5
    assert args.trust_radius_init == 1.0
    assert args.trust_subproblem_line_search is False
    assert args.retry_on_maxit is False
    assert args.gamg_set_coordinates is False
    assert args.use_near_nullspace is False

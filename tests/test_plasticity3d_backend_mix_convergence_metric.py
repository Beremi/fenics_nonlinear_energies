from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from experiments.runners import run_plasticity3d_backend_mix_case as case_runner


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
RUNNER = REPO_ROOT / "experiments/runners/run_plasticity3d_backend_mix_case.py"


def test_backend_mix_parser_preserves_legacy_default_and_accepts_riesz_controls(
    tmp_path: Path,
) -> None:
    required = [
        "--assembly-backend",
        "local",
        "--solver-backend",
        "local",
        "--out-dir",
        str(tmp_path),
        "--output-json",
        str(tmp_path / "output.json"),
    ]
    legacy = case_runner._build_parser().parse_args(required)
    configured = case_runner._build_parser().parse_args(
        [
            *required,
            "--convergence-metric",
            "reference_elastic_energy",
            "--convergence-state-scale",
            "2.5",
            "--riesz-ksp-type",
            "gmres",
            "--riesz-pc-type",
            "hypre",
            "--riesz-ksp-rtol",
            "1e-11",
            "--riesz-ksp-atol",
            "1e-15",
            "--riesz-ksp-max-it",
            "700",
            "--riesz-true-residual-rtol",
            "1e-9",
            "--riesz-spd-factor-solver-type",
            "mumps",
            "--riesz-symmetry-tol",
            "1e-13",
        ]
    )

    assert legacy.convergence_metric == "coefficient_l2"
    assert legacy.convergence_state_scale is None
    assert configured.convergence_metric == "reference_elastic_energy"
    assert configured.convergence_state_scale == pytest.approx(2.5)
    assert configured.riesz_ksp_type == "gmres"
    assert configured.riesz_pc_type == "hypre"
    assert configured.riesz_ksp_rtol == pytest.approx(1.0e-11)
    assert configured.riesz_ksp_atol == pytest.approx(1.0e-15)
    assert configured.riesz_ksp_max_it == 700
    assert configured.riesz_true_residual_rtol == pytest.approx(1.0e-9)
    assert configured.riesz_spd_factor_solver_type == "mumps"
    assert configured.riesz_symmetry_tol == pytest.approx(1.0e-13)


def test_backend_mix_reference_status_requires_fresh_residual_gate() -> None:
    converged = {"message": "Converged (energy, step, gradient)"}
    failed_gate = {"residual_gate": {"passed": False}}
    passed_gate = {"residual_gate": {"passed": True}}

    assert case_runner._backend_result_status(
        converged,
        metric_selection="coefficient_l2",
        convergence_payload=failed_gate,
    ) == "completed"
    assert case_runner._backend_result_status(
        converged,
        metric_selection="reference_elastic_energy",
        convergence_payload=failed_gate,
    ) == "failed"
    assert case_runner._backend_result_status(
        converged,
        metric_selection="reference_elastic_energy",
        convergence_payload=passed_gate,
    ) == "completed"
    assert case_runner._backend_result_status(
        {"message": "Maximum number of iterations reached"},
        metric_selection="reference_elastic_energy",
        convergence_payload=passed_gate,
    ) == "failed"


def test_gradient_only_policy_preserves_explicit_zero_absolute_tolerance() -> None:
    policy = case_runner._gradient_stopping_policy(
        convergence_mode="gradient_only",
        grad_stop_tol=0.0,
        grad_stop_rtol=1.0e-8,
        stop_tol=1.0e-8,
    )
    assert policy["configured_absolute"] == 0.0
    assert policy["gradient_target"] == 0.0
    assert policy["relative_gradient_target"] == pytest.approx(1.0e-8)
    assert policy["require_all_convergence"] is False

    relative_only = case_runner._gradient_stopping_policy(
        convergence_mode="gradient_only",
        grad_stop_tol=None,
        grad_stop_rtol=1.0e-8,
        stop_tol=1.0e-8,
    )
    assert relative_only["gradient_target"] == 0.0
    assert relative_only["relative_gradient_target"] == pytest.approx(1.0e-8)


def test_gradient_stopping_policy_rejects_invalid_tolerances() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        case_runner._gradient_stopping_policy(
            convergence_mode="gradient_only",
            grad_stop_tol=-1.0,
            grad_stop_rtol=1.0e-8,
            stop_tol=1.0e-8,
        )
    with pytest.raises(ValueError, match="finite"):
        case_runner._gradient_stopping_policy(
            convergence_mode="gradient_only",
            grad_stop_tol=0.0,
            grad_stop_rtol=float("nan"),
            stop_tol=1.0e-8,
        )


@pytest.mark.parametrize("ranks", [1, 2])
def test_backend_mix_reference_elastic_metric_mpi_smoke(
    tmp_path: Path,
    ranks: int,
) -> None:
    run_dir = tmp_path / f"np{ranks}"
    output_path = run_dir / "output.json"
    command = [
        "mpiexec",
        "-n",
        str(ranks),
        str(PYTHON),
        "-u",
        str(RUNNER),
        "--assembly-backend",
        "local",
        "--solver-backend",
        "local",
        "--out-dir",
        str(run_dir),
        "--output-json",
        str(output_path),
        "--mesh-name",
        "hetero_ssr_L1",
        "--elem-degree",
        "1",
        "--quadrature-rule",
        "tetra_1point",
        "--constraint-variant",
        "glued_bottom",
        "--lambda-target",
        "1.0",
        "--ksp-rtol",
        "1e-4",
        "--ksp-max-it",
        "200",
        "--convergence-mode",
        "all",
        "--grad-stop-tol",
        "1e-3",
        "--grad-stop-rtol",
        "1e-3",
        "--stop-tol",
        "1e-3",
        "--maxit",
        "1",
        "--line-search",
        "armijo",
        "--convergence-metric",
        "reference_elastic_energy",
        "--riesz-ksp-type",
        "cg",
        "--riesz-pc-type",
        "jacobi",
        "--riesz-ksp-rtol",
        "1e-10",
        "--riesz-true-residual-rtol",
        "1e-8",
    ]
    environment = dict(os.environ)
    environment["FNE_SKIP_REORDERED_WARMUP"] = "1"
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    convergence = payload["nonlinear_convergence"]
    configuration = convergence["configuration"]
    metric = convergence["metric"]
    provenance = metric["provenance"]
    certificate = provenance["spd_certificate"]
    riesz_solve = convergence["last_riesz_solve"]

    assert payload["ranks"] == ranks
    assert len(payload["assembler_rank_diagnostics"]["rank_summaries"]) == ranks
    assert payload["convergence_metric_requested"] == "reference_elastic_energy"
    assert payload["convergence_metric"] == "reference_elastic_energy"
    assert configuration["selection"] == "reference_elastic_energy"
    assert configuration["correction_normalization"] == "metric_current_state"
    assert configuration["state_scale_source"] == (
        "initial_nonlinear_iterate_primal_norm"
    )
    assert configuration["state_scale"] > 0.0

    expected_dofs = payload["parallel_setup"]["owned_free_dofs_sum"]
    assert provenance["constraint_variant"] == "glued_bottom"
    assert provenance["free_dofs"] == expected_dofs
    assert provenance["operator_source"] == "elastic_tangent_at_zero_displacement"
    assert provenance["reference_operator_tangent_mode"] == (
        "element_ad_for_all_routes"
    )
    assert provenance["input_identity"]["tangent_route"][
        "autodiff_tangent_mode"
    ] == "element"
    assert provenance["input_identity"]["tangent_route"][
        "reference_operator_forced_common"
    ] is True
    assert provenance["input_identity"]["tangent_route"]["constitutive_mode"] == (
        "elastic"
    )
    assert certificate["certified_spd"] is True
    assert certificate["inertia"] == {
        "negative": 0,
        "zero": 0,
        "positive": expected_dofs,
    }

    assert convergence["absolute_dual_residual"]["value"] is not None
    assert convergence["initial_absolute_dual_residual"]["value"] is not None
    assert convergence["initial_relative_dual_residual"]["value"] is not None
    assert convergence["state_norm"]["value"] is not None
    assert convergence["relative_correction"]["value"] is not None
    assert convergence["coefficient_gradient_l2"] == pytest.approx(
        payload["final_grad_norm"]
    )
    assert convergence["initial_absolute_dual_residual"]["value"] == pytest.approx(
        payload["history"][0]["dual_residual_norm"]
    )
    assert convergence["absolute_dual_residual"]["value"] == pytest.approx(
        payload["history"][-1]["grad_norm_post"]
    )
    assert convergence["initial_relative_dual_residual"]["value"] == pytest.approx(
        convergence["absolute_dual_residual"]["value"]
        / convergence["initial_absolute_dual_residual"]["value"]
    )
    assert riesz_solve["riesz_solve"] == "iterative"
    assert riesz_solve["reason"] > 0
    assert riesz_solve["rhs_norm"] == pytest.approx(payload["final_grad_norm"])
    assert riesz_solve["relative_true_residual"] <= riesz_solve[
        "true_residual_rtol_gate"
    ]

    # A one-iteration smoke is expected to stop at the iteration cap.  The
    # route must remain failed unless the certified endpoint residual gate is
    # actually satisfied; a coefficient-norm success cannot leak through.
    assert payload["status"] == "failed"
    assert payload["solver_success"] is False
    assert convergence["residual_gate"]["passed"] is False
    assert len(payload["branch_diagnostics"]["canonical_map_sha256"]) == 64

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
RUNNER = REPO_ROOT / "experiments" / "runners" / "run_paper_derivative_verification.py"


def test_p3d_fixed_element_derivative_verification_smoke(tmp_path: Path) -> None:
    output = tmp_path / "derivative_verification.json"
    env = os.environ.copy()
    env.update(
        {
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
        }
    )
    subprocess.run(
        [
            str(PYTHON),
            str(RUNNER),
            "--degree",
            "1",
            "--states",
            "1",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "EXP-DERIV-001-P3D-FIXED-ELEMENT"
    assert payload["status"] == "passed"
    assert payload["case"]["degree"] == 1
    assert payload["summary"]["maximum_residual_relative_error"] <= 1.0e-9
    assert payload["summary"]["maximum_hessian_relative_error"] <= 1.0e-9
    assert payload["summary"]["maximum_hessian_symmetry_defect"] <= 1.0e-10
    assert payload["summary"]["maximum_centered_fd_energy_error_at_gate"] <= 1.0e-7
    assert payload["summary"]["maximum_centered_fd_hvp_error_at_gate"] <= 1.0e-7
    assert payload["summary"]["all_states_branch_stable_at_fd_gate"] is True
    record = payload["records"][0]
    assert record["branch_stable_across_fd_gate"] is True
    assert record["branch_diagnostics"]["minimum_normalized_active_branch_margin"] > 0.0
    assert record["branch_diagnostics"]["interpretation"].startswith(
        "production predicate replay"
    )
    assert payload["provenance"]["jax_enable_x64"] is True


def test_p3d_assembled_derivative_routes_agree_without_a_solver(tmp_path: Path) -> None:
    output = tmp_path / "assembled_route_equivalence.json"
    env = os.environ.copy()
    env.update(
        {
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
        }
    )
    subprocess.run(
        [
            str(PYTHON),
            str(RUNNER),
            "--degree",
            "1",
            "--states",
            "1",
            "--assembled-route-equivalence",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["summary"]["assembled_route_equivalence_status"] == "passed"
    assembled = payload["assembled_route_equivalence"]
    assert assembled["case"]["degree"] == 1
    assert assembled["case"]["state_norm"] > 0.0
    assert assembled["branch_diagnostics"][
        "all_quadrature_points_strictly_elastic"
    ] is True
    assert assembled["branch_diagnostics"]["plastic_quadrature_points"] == 0
    assert set(assembled["routes"]) == {
        "element_ad",
        "local_sfd",
        "constitutive_ad",
    }
    assert assembled["routes"]["local_sfd"]["assembly_mode"] == "sfd_overlap_local"
    assert assembled["algebraic_scope"]["linear_solver_called"] is False
    assert assembled["algebraic_scope"]["nonlinear_solver_called"] is False
    assert assembled["algebraic_scope"]["ksp_tolerance_used_for_comparison"] is None
    assert len(assembled["pairwise_comparisons"]) == 3
    for comparison in assembled["pairwise_comparisons"]:
        assert comparison["passed"] is True
        assert comparison["energy_relative_error"] <= 1.0e-12
        assert comparison["gradient_relative_error"] <= 1.0e-12
        assert comparison["hessian_csr_structure_equal"] is True
        assert comparison["hessian_relative_error"] <= 1.0e-12
        assert comparison["hessian_maximum_entry_error"] <= 1.0e-8

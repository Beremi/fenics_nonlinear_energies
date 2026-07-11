from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from experiments.runners.run_paper_derivative_verification import (
    _chunked_array_equal,
    _chunked_hessian_errors,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
RUNNER = REPO_ROOT / "experiments" / "runners" / "run_paper_derivative_verification.py"


def test_chunked_csr_comparison_matches_dense_reference(tmp_path: Path) -> None:
    left = np.asarray([1.0, -2.0, 3.0, 4.0, -5.0], dtype=np.float64)
    right = np.asarray([1.0, -2.5, 3.0, 4.25, -5.0], dtype=np.float64)
    left_path = tmp_path / "left.npy"
    right_path = tmp_path / "right.npy"
    np.save(left_path, left, allow_pickle=False)
    np.save(right_path, right, allow_pickle=False)

    absolute, relative, maximum = _chunked_hessian_errors(
        left_path,
        right_path,
        chunk_entries=2,
    )
    difference = left - right
    expected_scale = max(np.linalg.norm(left), np.linalg.norm(right))
    np.testing.assert_allclose(absolute, np.linalg.norm(difference), rtol=1.0e-15)
    np.testing.assert_allclose(
        relative,
        np.linalg.norm(difference) / expected_scale,
        rtol=1.0e-15,
    )
    assert maximum == np.max(np.abs(difference))
    assert _chunked_array_equal(left_path, left_path, chunk_entries=2) is True
    assert _chunked_array_equal(left_path, right_path, chunk_entries=2) is False


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
            "5",
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
    assert list(tmp_path.glob(".paper_derivative_csr_*")) == []
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
    resources = assembled["execution_resources"]
    assert resources["routes"]["local_sfd"]["sfd_recovery"]["backend"] == (
        "petsc_scalar_distance2_component_lift"
    )
    assert resources["routes"]["local_sfd"]["sfd_recovery"]["validated"] is True
    assert resources["routes"]["local_sfd"]["sfd_recovery"]["hvp_batch_size"] == 4
    assert resources["routes"]["local_sfd"]["process_rss_hwm_gib"] > 0.0
    assert resources["requested_sfd_hvp_batch_size"] == 4
    assert resources["memory_guard_total_gib"] == 48.0
    assert resources["process_address_space_limit_gib"] == 64.0
    assert resources["csr_comparison_mode"] == "temporary_disk_backed_chunked"
    assert payload["provenance"]["process_address_space_limit_gib"] == 64.0
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

    from paper.scripts import admit_revision_publication_evidence as admission

    spec = next(
        spec for spec in admission.EVIDENCE_SPECS if spec.key == "p1_derivatives"
    )
    assert admission._derivative_errors(spec, payload) == []

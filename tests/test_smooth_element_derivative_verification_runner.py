from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
RUNNER = REPO_ROOT / "experiments" / "runners" / "run_smooth_element_derivative_verification.py"


def test_smooth_fixed_element_derivative_verification(tmp_path: Path) -> None:
    output = tmp_path / "smooth_derivatives.json"
    environment = os.environ.copy()
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
        }
    )
    subprocess.run(
        [str(PYTHON), str(RUNNER), "--output", str(output)],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "EXP-DERIV-001-SMOOTH-FIXED-ELEMENT"
    assert payload["status"] == "passed"
    assert payload["summary"]["cases"] == 5
    assert payload["summary"]["maximum_gradient_relative_error"] <= 1.0e-10
    assert payload["summary"]["maximum_hessian_relative_error"] <= 1.0e-10
    assert payload["summary"]["maximum_hessian_symmetry_defect"] <= 1.0e-12
    assert payload["summary"]["maximum_fd_gradient_error_at_gate"] <= 1.0e-7
    assert payload["summary"]["maximum_fd_hvp_error_at_gate"] <= 1.0e-7
    gl_indefinite = next(
        row for row in payload["records"] if row["case"] == "ginzburg_landau_indefinite"
    )
    assert gl_indefinite["independent_diagnostics"]["minimum_hessian_eigenvalue"] < 0.0
    for row in payload["records"]:
        if row["problem"] == "hyperelasticity":
            assert row["independent_diagnostics"]["determinant"] > 0.0
            assert (
                row["independent_diagnostics"]["analytic_vs_constitutive_ad_stress_error"]
                <= 1.0e-10
            )
    assert payload["provenance"]["jax_enable_x64"] is True

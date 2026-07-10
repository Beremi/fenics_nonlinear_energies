from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv/bin/python"
RUNNER = REPO_ROOT / "experiments/runners/run_manufactured_hyperelastic_verification.py"


def test_analytic_body_force_matches_piola_divergence() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("manufactured_he", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    points = np.array([[0.23, 0.37, 0.41], [0.61, 0.29, 0.73]], dtype=np.float64)
    step = 2.0e-6
    numerical_divergence = np.zeros_like(points)
    for axis in range(3):
        plus = points.copy()
        minus = points.copy()
        plus[:, axis] += step
        minus[:, axis] -= step
        plus_piola = module._density_piola_tangent(module._exact_gradient(plus))[1]
        minus_piola = module._density_piola_tangent(module._exact_gradient(minus))[1]
        numerical_divergence += (plus_piola[:, :, axis] - minus_piola[:, :, axis]) / (
            2.0 * step
        )
    np.testing.assert_allclose(
        module._body_force(points),
        -numerical_divergence,
        rtol=2.0e-8,
        atol=2.0e-9,
    )


def test_nonaffine_manufactured_hyperelasticity_converges(tmp_path: Path) -> None:
    output = tmp_path / "manufactured_he.json"
    subprocess.run(
        [str(PYTHON), str(RUNNER), "--output", str(output), "--subdivisions", "4", "8", "16"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["status"] == "passed"
    assert all(payload["gates"].values())
    assert payload["contract"]["load_quadrature_orders"] == [4, 6, 8]
    assert all(level["minimum_discrete_determinant"] > 0.5 for level in payload["levels"])
    for level in payload["levels"]:
        quadrature = level["load_quadrature_check"]
        assert quadrature["reference_load_stable"]
        assert quadrature["refined_solution_status"] == "converged"
        assert quadrature["refined_solution_resolves_load_change"]
        assert quadrature["below_fe_error"]
        assert quadrature["maximum_fraction_of_fe_error"] < 1.0e-4
        assert (
            quadrature[
                "free_load_primary_refinement_fraction_of_interpolant_consistency_residual"
            ]
            < 1.0e-4
        )
        assert (
            quadrature["free_load_refinement_confirmation_absolute_difference"]
            < quadrature["free_load_primary_refinement_absolute_difference"]
        )
    assert payload["rates"][-1]["l2_displacement_error"] >= 1.75
    assert payload["rates"][-1]["h1_deformation_error"] >= 0.75
    assert payload["rates"][-1]["first_piola_l2_error"] >= 0.75

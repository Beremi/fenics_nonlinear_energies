from __future__ import annotations

import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv/bin/python"
RUNNER = REPO_ROOT / "experiments/runners/run_hyperelastic_affine_patch_verification.py"


def test_hyperelastic_affine_patch_matches_analytic_reference(tmp_path: Path) -> None:
    output = tmp_path / "hyperelastic_patch.json"
    subprocess.run(
        [str(PYTHON), str(RUNNER), "--output", str(output)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["case"]["determinant"] > 0.0
    assert abs(payload["case"]["volume"] - 1.0) <= 1.0e-14
    assert all(
        payload["metrics"][field] <= payload["contract"]["relative_tolerance"]
        for field in (
            "energy_relative_error",
            "residual_relative_error",
            "hessian_relative_error",
            "hessian_symmetry_defect",
            "traction_balance_relative_error",
            "objectivity_energy_relative_error",
            "piola_rotation_covariance_relative_error",
        )
    )
    assert max(payload["metrics"]["translation_mode_hessian_action_norms"]) <= 2.0e-11

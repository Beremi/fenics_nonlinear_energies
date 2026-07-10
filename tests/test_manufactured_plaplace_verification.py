from __future__ import annotations

import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv/bin/python"
RUNNER = REPO_ROOT / "experiments/runners/run_manufactured_plaplace_verification.py"


def test_manufactured_plaplace_has_expected_spatial_rates(tmp_path: Path) -> None:
    output = tmp_path / "manufactured_plaplace.json"
    subprocess.run(
        [
            str(PYTHON),
            str(RUNNER),
            "--subdivisions",
            "6",
            "12",
            "24",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["publication_evidence"] is False
    assert len(payload["levels"]) == 3
    assert all(level["status"] == "converged" for level in payload["levels"])
    assert all(level["final_relative_residual"] <= 1.0e-8 for level in payload["levels"])
    assert all(level["minimum_element_gradient_norm"] > 0.5 for level in payload["levels"])
    assert all(level["tangent_symmetry_defect"] <= 1.0e-12 for level in payload["levels"])
    assert payload["rates"][-1]["l2_rate"] >= 1.75
    assert payload["rates"][-1]["h1_seminorm_rate"] >= 0.85
    assert payload["levels"][-1]["l2_error"] < payload["levels"][0]["l2_error"]
    assert payload["levels"][-1]["h1_seminorm_error"] < payload["levels"][0]["h1_seminorm_error"]

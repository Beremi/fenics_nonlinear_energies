from __future__ import annotations

import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv/bin/python"
RUNNER = REPO_ROOT / "experiments/runners/run_manufactured_ginzburg_landau_verification.py"


def test_manufactured_ginzburg_landau_has_expected_spatial_rates(tmp_path: Path) -> None:
    output = tmp_path / "manufactured_gl.json"
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
    assert all(level["status"] == "converged" for level in payload["levels"])
    assert all(level["minimum_nodal_value"] > 0.577 for level in payload["levels"])
    assert payload["rates"][-1]["l2_rate"] >= 1.75
    assert payload["rates"][-1]["h1_seminorm_rate"] >= 0.85

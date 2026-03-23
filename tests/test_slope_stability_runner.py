from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_slope_stability_runner_writes_summary(tmp_path: Path) -> None:
    out_dir = tmp_path / "runner_out"
    subprocess.run(
        [
            sys.executable,
            "experiments/runners/run_slope_stability_bringup.py",
            "--out-dir",
            str(out_dir),
            "--no-resume",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary_path = out_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert payload["runner"] == "slope_stability_bringup"
    assert set(row) == {
        "case",
        "lambda_target",
        "nodes",
        "elements",
        "free_dofs",
        "final_energy",
        "u_max",
        "newton_iters",
        "linear_iters",
        "total_time_s",
        "result",
        "json_path",
    }

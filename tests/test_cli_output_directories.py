from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("script_path", "output_flag", "extra_args", "check_keys"),
    [
        (
            "src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py",
            "--json",
            ["--levels", "5", "--quiet"],
            ["results"],
        ),
        (
            "src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py",
            "--json",
            ["--levels", "5", "--quiet"],
            ["results"],
        ),
        (
            "src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py",
            "--out",
            ["--level", "1", "--steps", "1", "--total-steps", "24", "--quiet"],
            ["steps"],
        ),
        (
            "src/problems/topology/jax/solve_topopt_jax.py",
            "--json_out",
            [
                "--nx",
                "32",
                "--ny",
                "16",
                "--fixed_pad_cells",
                "2",
                "--load_pad_cells",
                "2",
                "--outer_maxit",
                "1",
                "--mechanics_maxit",
                "10",
                "--design_maxit",
                "10",
                "--quiet",
            ],
            ["result"],
        ),
    ],
)
def test_cli_creates_missing_output_directories(
    script_path: str,
    output_flag: str,
    extra_args: list[str],
    check_keys: list[str],
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "nested" / "results" / "run.json"
    cmd = [sys.executable, script_path, *extra_args, output_flag, str(out_path)]
    subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    for key in check_keys:
        assert key in payload

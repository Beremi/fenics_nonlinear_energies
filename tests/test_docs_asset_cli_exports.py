from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def _help_text(script: str) -> str:
    result = subprocess.run(
        [str(PYTHON), script, "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_scalar_publication_export_flags_are_available() -> None:
    assert "--state-out" in _help_text("src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py")
    assert "--state-out" in _help_text("src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py")


def test_he_publication_export_flag_is_available() -> None:
    assert "--state-out" in _help_text("src/problems/hyperelasticity/jax/solve_HE_jax_newton.py")

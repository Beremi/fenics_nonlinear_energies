from __future__ import annotations

from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv/bin/python"
VALIDATOR = REPO_ROOT / "paper/scripts/validate_paper_assets.py"


def test_focused_manuscript_assets_are_covered_by_their_manifests() -> None:
    completed = subprocess.run(
        [str(PYTHON), str(VALIDATOR), "--archive-neutral"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "3 tables" in completed.stdout

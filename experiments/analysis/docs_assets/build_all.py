#!/usr/bin/env python3
"""Rebuild all curated figures under ``docs/assets`` from tracked source data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SCRIPTS = (
    "experiments/analysis/docs_assets/build_plaplace_figures.py",
    "experiments/analysis/docs_assets/build_ginzburg_landau_figures.py",
    "experiments/analysis/docs_assets/build_hyperelasticity_figures.py",
    "experiments/analysis/docs_assets/build_topology_figures.py",
)


def main() -> int:
    for script in SCRIPTS:
        subprocess.run([str(PYTHON), script], cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

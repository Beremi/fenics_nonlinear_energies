#!/usr/bin/env python3
"""Compatibility wrapper for the thesis single-case CLI.

The actual script lives in ``src/problems/plaplace_u3/thesis/scripts/solve_case.py``.
Keeping this thin wrapper avoids breaking old commands while the parser/output
logic stays clearly separated from the reusable solver modules.
"""

from __future__ import annotations

from src.problems.plaplace_u3.thesis.scripts.solve_case import main


if __name__ == "__main__":
    main()

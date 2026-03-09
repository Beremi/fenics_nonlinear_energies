#!/usr/bin/env python3
"""Run the remaining level-4 HE final-suite cases with a higher step cap."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CASES = (
    ("fenics_custom", 24, 4, 16),
    ("fenics_custom", 24, 4, 8),
    ("jax_petsc_element", 24, 4, 32),
    ("jax_petsc_element", 24, 4, 16),
    ("jax_petsc_element", 24, 4, 8),
    ("fenics_custom", 96, 4, 32),
    ("fenics_custom", 96, 4, 16),
    ("fenics_custom", 96, 4, 8),
    ("jax_petsc_element", 96, 4, 32),
    ("jax_petsc_element", 96, 4, 16),
    ("jax_petsc_element", 96, 4, 8),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-time-limit-s", type=float, default=300.0)
    parser.add_argument("--trust-radius-init", type=float, default=2.0)
    parser.add_argument("--max-case-wall-s", type=float, default=21600.0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiment_results_cache/he_final_suite_r2_0",
    )
    return parser


def _load_rows(summary_json: Path) -> list[dict]:
    if not summary_json.exists():
        return []
    return json.loads(summary_json.read_text())["rows"]


def _existing_result(rows: list[dict], solver: str, total_steps: int, level: int, nprocs: int) -> str | None:
    for row in rows:
        if (
            row["solver"] == solver
            and int(row["total_steps"]) == int(total_steps)
            and int(row["level"]) == int(level)
            and int(row["nprocs"]) == int(nprocs)
        ):
            return str(row["result"])
    return None


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out_dir
    summary_json = out_dir / "summary.json"

    for solver, total_steps, level, nprocs in CASES:
        rows = _load_rows(summary_json)
        result = _existing_result(rows, solver, total_steps, level, nprocs)
        if result == "completed":
            print(f"skip completed: {solver} steps={total_steps} level={level} np={nprocs}", flush=True)
            continue

        cmd = [
            sys.executable,
            "experiment_scripts/run_he_final_case.py",
            "--solver",
            solver,
            "--level",
            str(level),
            "--total-steps",
            str(total_steps),
            "--nprocs",
            str(nprocs),
            "--step-time-limit-s",
            str(args.step_time_limit_s),
            "--trust-radius-init",
            str(args.trust_radius_init),
            "--max-case-wall-s",
            str(args.max_case_wall_s),
            "--out-dir",
            args.out_dir,
        ]
        print(f"run: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()

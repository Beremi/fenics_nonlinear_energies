#!/usr/bin/env python3
"""Run one final-suite HE case and upsert it into the suite summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from experiment_scripts.run_he_final_suite import (
    _run_case,
    _solver_config,
    _write_summary_markdown,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solver",
        choices=("fenics_custom", "jax_petsc_element"),
        required=True,
    )
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--total-steps", type=int, choices=(24, 96), required=True)
    parser.add_argument("--nprocs", type=int, required=True)
    parser.add_argument("--step-time-limit-s", type=float, default=None)
    parser.add_argument("--trust-radius-init", type=float, default=2.0)
    parser.add_argument("--max-case-wall-s", type=float, default=14400.0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiment_results_cache/he_final_suite_r2_0",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    solver = _solver_config(args.solver)
    suite_args = SimpleNamespace(
        trust_radius_init=float(args.trust_radius_init),
        step_time_limit_s=(
            None if args.step_time_limit_s is None else float(args.step_time_limit_s)
        ),
        max_case_wall_s=float(args.max_case_wall_s),
    )

    row = _run_case(
        repo_root=repo_root,
        out_dir=out_dir,
        solver=solver,
        total_steps=int(args.total_steps),
        level=int(args.level),
        nprocs=int(args.nprocs),
        args=suite_args,
    )

    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    rows = json.loads(summary_json.read_text())["rows"] if summary_json.exists() else []
    key = (args.solver, int(args.total_steps), int(args.level), int(args.nprocs))
    rows = [
        existing
        for existing in rows
        if (
            existing["solver"],
            int(existing["total_steps"]),
            int(existing["level"]),
            int(existing["nprocs"]),
        )
        != key
    ]
    rows.append(row)
    rows.sort(key=lambda r: (r["solver"], int(r["total_steps"]), int(r["level"]), int(r["nprocs"])))

    summary_json.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    _write_summary_markdown(summary_md, rows)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the experimental slope-stability pure-JAX bring-up case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.benchmark.replication import read_json, run_logged_command, write_json
from src.problems.slope_stability.support import DEFAULT_CASE


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/raw_results/slope_stability_bringup",
    )
    parser.add_argument("--case", type=str, default=DEFAULT_CASE)
    parser.add_argument("--lambda-target", type=float, default=1.21)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    leaf_dir = out_dir / f"{args.case}_lambda_{str(args.lambda_target).replace('.', 'p')}"
    json_out = leaf_dir / "output.json"
    state_out = leaf_dir / "state.npz"
    command = [
        str(PYTHON),
        "-u",
        "src/problems/slope_stability/jax/solve_slope_stability_jax.py",
        "--case",
        str(args.case),
        "--lambda-target",
        str(args.lambda_target),
        "--quiet",
        "--json",
        str(json_out),
        "--state-out",
        str(state_out),
    ]

    run_logged_command(
        command=command,
        cwd=REPO_ROOT,
        leaf_dir=leaf_dir,
        expected_outputs=[json_out, state_out],
        resume=bool(args.resume),
        notes="Experimental slope-stability pure-JAX zero-history endpoint bring-up.",
    )

    payload = read_json(json_out)
    row = {
        "case": str(payload["case"]["name"]),
        "lambda_target": float(payload["case"]["lambda_target"]),
        "nodes": int(payload["mesh"]["nodes"]),
        "elements": int(payload["mesh"]["elements"]),
        "free_dofs": int(payload["mesh"]["free_dofs"]),
        "final_energy": float(payload["result"]["final_energy"]),
        "u_max": float(payload["result"]["u_max"]),
        "newton_iters": int(payload["result"]["newton_iters"]),
        "linear_iters": int(payload["result"]["linear_iters"]),
        "total_time_s": float(payload["timings"]["total_time"]),
        "result": str(payload["result"]["status"]),
        "json_path": _display_path(json_out),
    }
    summary = {
        "runner": "slope_stability_bringup",
        "rows": [row],
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

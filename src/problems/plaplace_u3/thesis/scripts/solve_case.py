#!/usr/bin/env python3
"""CLI entrypoint for one thesis-faithful ``plaplace_u3`` solve.

This module intentionally owns argument parsing, filesystem output, and other
script-oriented glue so the solver modules can stay focused on reusable
numerical logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.problems.plaplace_u3.support.mesh import SUPPORTED_GEOMETRIES, SUPPORTED_INIT_MODES
from src.problems.plaplace_u3.thesis.mesh1d import GEOMETRY_INTERVAL_PI, SUPPORTED_1D_INIT_MODES
from src.problems.plaplace_u3.thesis.presets import (
    THESIS_DIRECTIONS,
    THESIS_MAXIT_MPA,
    THESIS_MAXIT_RMPA_OA,
    THESIS_METHODS,
    THESIS_MPA_NUM_NODES,
    THESIS_MPA_RHO,
    THESIS_MPA_SEGMENT_TOL_FACTOR,
    THESIS_OA_DELTA_HAT,
    THESIS_OA_GOLDEN_TOL,
    THESIS_RMPA_DELTA0,
    THESIS_TOL_MAIN,
)
from src.problems.plaplace_u3.thesis.solver_common import build_problem
from src.problems.plaplace_u3.thesis.solver_mpa import run_mpa
from src.problems.plaplace_u3.thesis.solver_oa import run_oa
from src.problems.plaplace_u3.thesis.solver_rmpa import run_rmpa


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser used by the thesis single-case solver script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=THESIS_METHODS, default="rmpa")
    parser.add_argument("--direction", choices=THESIS_DIRECTIONS, default="d_vh")
    parser.add_argument("--dimension", type=int, choices=(1, 2), default=2)
    parser.add_argument("--level", type=int, default=6)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--geometry", type=str, default="")
    parser.add_argument("--init-mode", type=str, default="sine")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=THESIS_TOL_MAIN)
    parser.add_argument("--maxit", type=int, default=0)
    parser.add_argument("--delta0", type=float, default=THESIS_RMPA_DELTA0)
    parser.add_argument("--rmpa-step-search", choices=("halving", "golden"), default="halving")
    parser.add_argument("--delta-hat", type=float, default=THESIS_OA_DELTA_HAT)
    parser.add_argument("--golden-tol", type=float, default=THESIS_OA_GOLDEN_TOL)
    parser.add_argument("--num-nodes", type=int, default=THESIS_MPA_NUM_NODES)
    parser.add_argument("--rho", type=float, default=THESIS_MPA_RHO)
    parser.add_argument("--segment-tol-factor", type=float, default=THESIS_MPA_SEGMENT_TOL_FACTOR)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--json", type=str, default="")
    parser.add_argument("--state-out", type=str, default="")
    return parser


def _resolve_output_path(args: argparse.Namespace) -> str:
    if args.out and args.json and Path(args.out) != Path(args.json):
        raise ValueError("--out and --json must match when both are provided")
    return args.out or args.json


def _resolve_geometry(args: argparse.Namespace) -> str:
    if args.geometry:
        return str(args.geometry)
    return GEOMETRY_INTERVAL_PI if int(args.dimension) == 1 else "square_pi"


def _validate_case_args(args: argparse.Namespace, *, geometry: str) -> None:
    """Check that the chosen seed and geometry belong to the selected dimension."""
    if int(args.dimension) == 1:
        if geometry != GEOMETRY_INTERVAL_PI:
            raise ValueError("1D thesis runs only support geometry=interval_pi")
        if str(args.init_mode) not in SUPPORTED_1D_INIT_MODES:
            raise ValueError(
                f"Unsupported 1D init_mode={args.init_mode!r}; expected one of {SUPPORTED_1D_INIT_MODES}"
            )
        return

    if geometry not in SUPPORTED_GEOMETRIES:
        raise ValueError(
            f"Unsupported 2D geometry={geometry!r}; expected one of {SUPPORTED_GEOMETRIES}"
        )
    if str(args.init_mode) not in SUPPORTED_INIT_MODES:
        raise ValueError(
            f"Unsupported 2D init_mode={args.init_mode!r}; expected one of {SUPPORTED_INIT_MODES}"
        )


def run_case_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Dispatch one parsed CLI case to the appropriate thesis solver."""
    geometry = _resolve_geometry(args)
    _validate_case_args(args, geometry=geometry)

    problem = build_problem(
        dimension=int(args.dimension),
        level=int(args.level),
        p=float(args.p),
        geometry=geometry,
        init_mode=str(args.init_mode),
        seed=int(args.seed),
    )

    default_maxit = THESIS_MAXIT_MPA if str(args.method) == "mpa" else THESIS_MAXIT_RMPA_OA
    maxit = int(args.maxit) if int(args.maxit) > 0 else int(default_maxit)

    if str(args.method) == "rmpa":
        return run_rmpa(
            problem,
            direction=str(args.direction),
            epsilon=float(args.epsilon),
            maxit=maxit,
            delta0=float(args.delta0),
            step_search=str(args.rmpa_step_search),
            state_out=str(args.state_out),
        )
    if str(args.method) == "mpa":
        return run_mpa(
            problem,
            direction=str(args.direction),
            epsilon=float(args.epsilon),
            maxit=maxit,
            num_nodes=int(args.num_nodes),
            rho=float(args.rho),
            segment_tol_factor=float(args.segment_tol_factor),
            state_out=str(args.state_out),
        )
    if str(args.method) in {"oa1", "oa2"}:
        return run_oa(
            problem,
            variant=str(args.method),
            direction=str(args.direction),
            epsilon=float(args.epsilon),
            maxit=maxit,
            delta_hat=float(args.delta_hat),
            golden_tol=float(args.golden_tol),
            state_out=str(args.state_out),
        )
    raise ValueError(f"Unsupported method={args.method!r}")


def main() -> None:
    """Run the thesis single-case CLI and optionally write the JSON payload."""
    args = build_parser().parse_args()
    out_path = _resolve_output_path(args)
    result = run_case_from_args(args)

    text = json.dumps(result, indent=2)
    print(text)
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

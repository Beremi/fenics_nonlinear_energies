#!/usr/bin/env python3
"""Experimental JAX+PETSc solver for refined-P1 slope-stability cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from src.core.cli.threading import configure_jax_cpu_threading
from src.problems.slope_stability.support import (
    build_elastic_initial_guess,
    build_near_nullspace_modes,
    build_refined_p1_case_data,
    case_name_for_level,
)


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--nproc", type=int, default=1)
    pre_args, _ = pre_parser.parse_known_args()
    configure_jax_cpu_threading(pre_args.nproc)

    from src.problems.slope_stability.jax_petsc.solve_slope_stability_dof import (
        _build_parser,
    )
    from src.problems.slope_stability.jax_petsc.solver import PROFILE_DEFAULTS, run

    parser = _build_parser(PROFILE_DEFAULTS)
    args = parser.parse_args()

    case_data = build_refined_p1_case_data(case_name_for_level(int(args.level)))
    params = dict(case_data.__dict__)
    params["elastic_kernel"] = build_near_nullspace_modes(
        np.asarray(case_data.nodes, dtype=np.float64),
        np.asarray(case_data.freedofs, dtype=np.int64),
    )
    params["elem_type"] = "P1_refined_same_nodes"
    u_init = build_elastic_initial_guess(
        elems=np.asarray(case_data.elems, dtype=np.int64),
        elem_B=np.asarray(case_data.elem_B, dtype=np.float64),
        quad_weight=np.asarray(case_data.quad_weight, dtype=np.float64),
        force=np.asarray(case_data.force, dtype=np.float64),
        freedofs=np.asarray(case_data.freedofs, dtype=np.int64),
        E=float(case_data.E),
        nu=float(case_data.nu),
    )
    result = run(args, problem_data=(params, case_data.adjacency, u_init))

    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(result, indent=2))
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark one HE reordered-element local Hessian mode.

Measures:
- isolated Hessian assembly at the step-1 state, reduced as MPI max per repeat
- full Newton solve on the same problem, reduced as MPI max per linear iteration
"""

from __future__ import annotations

import argparse
import json
import math
import os
from types import SimpleNamespace


def _configure_thread_env(nproc: int) -> None:
    threads = max(1, int(nproc))
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false "
        "--xla_force_host_platform_device_count=1"
    )
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--start_step", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=96)
    parser.add_argument("--profile", type=str, default="performance")
    parser.add_argument("--ksp_type", type=str, default="gmres")
    parser.add_argument("--pc_type", type=str, default="gamg")
    parser.add_argument("--ksp_rtol", type=float, default=1e-1)
    parser.add_argument("--ksp_max_it", type=int, default=30)
    parser.add_argument("--gamg_threshold", type=float, default=0.05)
    parser.add_argument("--gamg_agg_nsmooths", type=int, default=1)
    parser.add_argument("--use_near_nullspace", action="store_true", default=True)
    parser.add_argument("--gamg_set_coordinates", action="store_true", default=True)
    parser.add_argument("--pc_setup_on_ksp_cap", action="store_true", default=True)
    parser.add_argument("--element_reorder_mode", type=str, default="block_xyz")
    parser.add_argument(
        "--local_hessian_mode",
        choices=("element", "sfd_local", "sfd_local_vmap"),
        required=True,
    )
    parser.add_argument(
        "--phase",
        choices=("isolated", "full", "both"),
        default="both",
    )
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--assembly_repeats", type=int, default=5)
    parser.add_argument("--out", type=str, default="")
    return parser


def _max_float(comm, value: float) -> float:
    from mpi4py import MPI

    return float(comm.allreduce(float(value), op=MPI.MAX))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sum_linear_timing_max(comm, linear_timing: list[dict]) -> dict[str, float]:
    from mpi4py import MPI

    n_local = len(linear_timing)
    n_min = comm.allreduce(n_local, op=MPI.MIN)
    n_max = comm.allreduce(n_local, op=MPI.MAX)
    if n_min != n_max:
        raise RuntimeError(
            f"Linear timing record length mismatch across ranks: min={n_min}, max={n_max}"
        )

    fields = (
        "assemble_total_time",
        "assemble_p2p_exchange",
        "assemble_hvp_compute",
        "assemble_extraction",
        "assemble_coo_assembly",
        "assemble_n_hvps",
        "setop_time",
        "set_tolerances_time",
        "pc_setup_time",
        "solve_time",
        "linear_total_time",
    )
    totals = {field: 0.0 for field in fields}
    ksp_total = 0
    for rec in linear_timing:
        ksp_total += int(rec["ksp_its"])
    ksp_total = int(comm.allreduce(ksp_total, op=MPI.MAX))

    for idx in range(n_local):
        rec = linear_timing[idx]
        for field in fields:
            value = rec[field]
            if field == "assemble_n_hvps":
                totals[field] += int(comm.allreduce(int(value), op=MPI.MAX))
            else:
                totals[field] += float(comm.allreduce(float(value), op=MPI.MAX))
    totals["linear_iters"] = ksp_total
    totals["calls"] = int(n_local)
    return totals


def _isolated_assembly_benchmark(args, params, adjacency, u_init):
    import numpy as np
    from mpi4py import MPI

    from HyperElasticity3D_jax_petsc.reordered_element_assembler import (
        HEReorderedElementAssembler,
    )
    from HyperElasticity3D_petsc_support.rotate_boundary import (
        rotate_right_face_from_reference,
    )

    comm = MPI.COMM_WORLD
    assembler = HEReorderedElementAssembler(
        params=params,
        comm=comm,
        adjacency=adjacency,
        ksp_rtol=float(args.ksp_rtol),
        ksp_type=str(args.ksp_type),
        pc_type=str(args.pc_type),
        ksp_max_it=int(args.ksp_max_it),
        use_near_nullspace=bool(args.use_near_nullspace),
        pc_options={
            "pc_gamg_threshold": float(args.gamg_threshold),
            "pc_gamg_agg_nsmooths": int(args.gamg_agg_nsmooths),
        }
        if args.pc_type == "gamg"
        else None,
        reorder_mode=str(args.element_reorder_mode),
        local_hessian_mode=str(args.local_hessian_mode),
    )
    try:
        angle = args.start_step * 4.0 * 2.0 * math.pi / float(args.total_steps)
        u0_step = rotate_right_face_from_reference(
            params["u_0_ref"], params["nodes2coord"], angle, params["right_nodes"]
        )
        assembler.update_dirichlet(u0_step)

        u_init_reordered = np.asarray(u_init, dtype=np.float64)[assembler.part.perm]
        u_owned = np.asarray(
            u_init_reordered[assembler.part.lo : assembler.part.hi], dtype=np.float64
        )

        assembler.assemble_hessian(u_owned)
        repeats = []
        for _ in range(int(args.assembly_repeats)):
            timings = assembler.assemble_hessian(u_owned)
            reduced = {
                key: _max_float(comm, timings[key])
                for key in (
                    "total",
                    "p2p_exchange",
                    "hvp_compute",
                    "extraction",
                    "coo_assembly",
                    "allgatherv",
                    "build_v_local",
                )
            }
            reduced["n_hvps"] = int(comm.allreduce(int(timings["n_hvps"]), op=MPI.MAX))
            repeats.append(reduced)

        summary = {
            field: _mean([rep[field] for rep in repeats])
            for field in (
                "total",
                "p2p_exchange",
                "hvp_compute",
                "extraction",
                "coo_assembly",
                "allgatherv",
                "build_v_local",
                "n_hvps",
            )
        }
        return {"repeats": repeats, "summary": summary}
    finally:
        assembler.cleanup()


def _full_step_benchmark(args):
    from mpi4py import MPI

    from HyperElasticity3D_jax_petsc.solver import run

    comm = MPI.COMM_WORLD
    run_args = SimpleNamespace(
        level=int(args.level),
        steps=int(args.steps),
        start_step=int(args.start_step),
        total_steps=int(args.total_steps),
        profile=str(args.profile),
        ksp_type=str(args.ksp_type),
        pc_type=str(args.pc_type),
        ksp_rtol=float(args.ksp_rtol),
        ksp_max_it=int(args.ksp_max_it),
        pc_setup_on_ksp_cap=bool(args.pc_setup_on_ksp_cap),
        gamg_threshold=float(args.gamg_threshold),
        gamg_agg_nsmooths=int(args.gamg_agg_nsmooths),
        gamg_set_coordinates=bool(args.gamg_set_coordinates),
        use_near_nullspace=bool(args.use_near_nullspace),
        reorder=None,
        local_coloring=True,
        hvp_eval_mode="sequential",
        coloring_trials=10,
        assembly_mode="element",
        element_reorder_mode=str(args.element_reorder_mode),
        local_hessian_mode=str(args.local_hessian_mode),
        tolf=1e-4,
        tolg=1e-3,
        tolg_rel=1e-3,
        tolx_rel=1e-3,
        tolx_abs=1e-10,
        maxit=100,
        linesearch_a=-0.5,
        linesearch_b=2.0,
        linesearch_tol=1e-3,
        retry_on_failure=True,
        stop_on_fail=False,
        use_abs_det=False,
        nproc=int(args.nproc),
        save_history=False,
        save_linear_timing=True,
        quiet=True,
        out="",
        use_trust_region=False,
        trust_radius_init=1.0,
        trust_radius_min=1e-8,
        trust_radius_max=1e6,
        trust_shrink=0.5,
        trust_expand=1.5,
        trust_eta_shrink=0.05,
        trust_eta_expand=0.75,
        trust_max_reject=6,
    )
    result = run(run_args)
    if len(result["steps"]) != 1:
        raise RuntimeError("Expected exactly one load step in benchmark")
    step = result["steps"][0]
    linear_summary = _sum_linear_timing_max(comm, step["linear_timing"])
    return {
        "mesh_level": int(result["mesh_level"]),
        "setup_time_max": _max_float(comm, result["setup_time"]),
        "step_time_max": _max_float(comm, step["time"]),
        "total_time_max": _max_float(comm, result["total_time"]),
        "nit": int(comm.allreduce(int(step["nit"]), op=MPI.MAX)),
        "linear_iters": int(comm.allreduce(int(step["linear_iters"]), op=MPI.MAX)),
        "energy": _max_float(comm, step["energy"]),
        "message": step["message"],
        "linear_timing_sum_max": linear_summary,
        "metadata": result["metadata"],
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_thread_env(args.nproc)

    import numpy as np
    from mpi4py import MPI

    from HyperElasticity3D_petsc_support.mesh import MeshHyperElasticity3D

    comm = MPI.COMM_WORLD
    mesh_obj = MeshHyperElasticity3D(args.level)
    params, adjacency, u_init = mesh_obj.get_data()

    isolated = None
    full_step = None
    if args.phase in ("isolated", "both"):
        isolated = _isolated_assembly_benchmark(args, params, adjacency, u_init)
    if args.phase in ("full", "both"):
        full_step = _full_step_benchmark(args)

    result = {
        "mode": str(args.local_hessian_mode),
        "phase": str(args.phase),
        "nprocs": int(comm.size),
        "nproc_threads": int(args.nproc),
        "isolated_assembly": isolated,
        "full_step": full_step,
    }

    if comm.rank == 0:
        print(json.dumps(result, indent=2))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

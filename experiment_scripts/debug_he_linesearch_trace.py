#!/usr/bin/env python3
"""Reproduce the fine HE JAX+PETSc step-2 line search and log sampled points."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


def _configure_thread_env(nproc_threads: int) -> None:
    threads = max(1, int(nproc_threads))
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false "
        "--xla_force_host_platform_device_count=1"
    )
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", type=int, default=4)
    p.add_argument("--step", type=int, default=2, help="1-based step to debug")
    p.add_argument("--total-steps", type=int, default=24)
    p.add_argument("--target-iter", type=int, default=3)
    p.add_argument("--linesearch-a", type=float, default=-0.5)
    p.add_argument("--linesearch-b", type=float, default=2.0)
    p.add_argument("--linesearch-tol", type=float, default=1e-3)
    p.add_argument("--maxit", type=int, default=3)
    p.add_argument("--ksp-rtol", type=float, default=1e-1)
    p.add_argument("--ksp-max-it", type=int, default=100)
    p.add_argument("--nproc-threads", type=int, default=1)
    p.add_argument("--out", required=True)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    _configure_thread_env(args.nproc_threads)

    from HyperElasticity3D_petsc_support.mesh import MeshHyperElasticity3D
    from HyperElasticity3D_jax_petsc.reordered_element_assembler import (
        HEReorderedElementAssembler,
    )
    from HyperElasticity3D_petsc_support.rotate_boundary import (
        rotate_right_face_from_reference,
    )
    from HyperElasticity3D_jax_petsc.solver import _build_gamg_coordinates
    import tools_petsc4py.minimizers as minimizers

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    mesh_obj = MeshHyperElasticity3D(args.level)
    params, adjacency, u_init = mesh_obj.get_data()

    assembler = HEReorderedElementAssembler(
        params=params,
        comm=comm,
        adjacency=adjacency,
        ksp_rtol=float(args.ksp_rtol),
        ksp_type="gmres",
        pc_type="gamg",
        ksp_max_it=int(args.ksp_max_it),
        use_near_nullspace=True,
        pc_options={
            "pc_gamg_threshold": 0.05,
            "pc_gamg_agg_nsmooths": 1,
        },
        reorder_mode="block_xyz",
        local_hessian_mode="element",
    )

    ksp = assembler.ksp
    A = assembler.A
    pc = ksp.getPC()
    gamg_coords = _build_gamg_coordinates(
        assembler.part, params["freedofs"], params["nodes2coord"]
    )

    x = assembler.create_vec(np.asarray(u_init, dtype=np.float64)[assembler.part.perm])
    x_after_prev = x.duplicate()

    force_pc_setup_next = True
    newton_direction_norms = []

    def hessian_solve_fn(vec, rhs, sol):
        nonlocal force_pc_setup_next
        u_owned = np.array(vec.array[:], dtype=np.float64)
        assembler.assemble_hessian(u_owned)
        ksp.setOperators(A)
        nonlocal gamg_coords
        if gamg_coords is not None:
            pc.setCoordinates(gamg_coords)
            gamg_coords = None
        ksp.setTolerances(rtol=float(args.ksp_rtol), max_it=int(args.ksp_max_it))
        if force_pc_setup_next:
            ksp.setUp()
            force_pc_setup_next = False
        ksp.solve(rhs, sol)
        ksp_its = int(ksp.getIterationNumber())
        newton_direction_norms.append(float(sol.norm(PETSc.NormType.NORM_2)))
        if ksp_its >= int(args.ksp_max_it):
            force_pc_setup_next = True
        return ksp_its

    rot = 4.0 * 2.0 * np.pi / float(args.total_steps)

    try:
        for step in range(1, int(args.step)):
            angle = step * rot
            u0_step = rotate_right_face_from_reference(
                params["u_0_ref"], params["nodes2coord"], angle, params["right_nodes"]
            )
            assembler.update_dirichlet(u0_step)
            force_pc_setup_next = True
            result = minimizers.newton(
                energy_fn=assembler.energy_fn,
                gradient_fn=assembler.gradient_fn,
                hessian_solve_fn=hessian_solve_fn,
                x=x,
                tolf=1e-4,
                tolg=1e-3,
                tolg_rel=1e-3,
                linesearch_tol=float(args.linesearch_tol),
                linesearch_interval=(float(args.linesearch_a), float(args.linesearch_b)),
                maxit=300,
                tolx_rel=1e-3,
                tolx_abs=1e-10,
                require_all_convergence=True,
                fail_on_nonfinite=True,
                verbose=False,
                comm=comm,
                ghost_update_fn=None,
                hessian_matvec_fn=lambda _x, vin, vout: assembler.A.mult(vin, vout),
                save_history=False,
                trust_region=False,
                step_time_limit_s=None,
            )
            if rank == 0:
                print(
                    f"prepared step {step}: nit={result['nit']} msg={result['message']}",
                    flush=True,
                )

        x.copy(x_after_prev)

        angle = int(args.step) * rot
        u0_step = rotate_right_face_from_reference(
            params["u_0_ref"], params["nodes2coord"], angle, params["right_nodes"]
        )
        assembler.update_dirichlet(u0_step)

        trace_calls = []
        call_counter = 0
        orig_gss = minimizers.golden_section_search

        def traced_gss(f, a, b, tol):
            nonlocal call_counter
            call_counter += 1
            samples = []

            def wrapped(alpha):
                val = f(alpha)
                samples.append(
                    {
                        "alpha": float(alpha),
                        "energy": None if not np.isfinite(val) else float(val),
                        "nonfinite": bool(not np.isfinite(val)),
                    }
                )
                return val

            alpha_star, n_evals = orig_gss(wrapped, a, b, tol)
            final_energy = wrapped(alpha_star)
            trace_calls.append(
                {
                    "newton_iter": int(call_counter),
                    "interval": [float(a), float(b)],
                    "tol": float(tol),
                    "returned_alpha": float(alpha_star),
                    "reported_ls_evals": int(n_evals),
                    "final_energy_at_returned_alpha": (
                        None if not np.isfinite(final_energy) else float(final_energy)
                    ),
                    "samples": samples,
                }
            )
            return alpha_star, n_evals

        minimizers.golden_section_search = traced_gss
        try:
            force_pc_setup_next = True
            result = minimizers.newton(
                energy_fn=assembler.energy_fn,
                gradient_fn=assembler.gradient_fn,
                hessian_solve_fn=hessian_solve_fn,
                x=x,
                tolf=1e-4,
                tolg=1e-3,
                tolg_rel=1e-3,
                linesearch_tol=float(args.linesearch_tol),
                linesearch_interval=(float(args.linesearch_a), float(args.linesearch_b)),
                maxit=int(args.maxit),
                tolx_rel=1e-3,
                tolx_abs=1e-10,
                require_all_convergence=True,
                fail_on_nonfinite=True,
                verbose=False,
                comm=comm,
                ghost_update_fn=None,
                hessian_matvec_fn=lambda _x, vin, vout: assembler.A.mult(vin, vout),
                save_history=True,
                trust_region=False,
                step_time_limit_s=None,
            )
        finally:
            minimizers.golden_section_search = orig_gss

        out = {
            "case": {
                "solver": "jax_petsc_element",
                "level": int(args.level),
                "nprocs": int(nprocs),
                "step": int(args.step),
                "total_steps": int(args.total_steps),
                "target_iter": int(args.target_iter),
                "linesearch_interval": [float(args.linesearch_a), float(args.linesearch_b)],
                "linesearch_tol": float(args.linesearch_tol),
                "ksp_rtol": float(args.ksp_rtol),
                "ksp_max_it": int(args.ksp_max_it),
                "maxit": int(args.maxit),
            },
            "result": {
                "message": str(result["message"]),
                "nit": int(result["nit"]),
                "fun": float(result["fun"]),
                "history": result.get("history", []),
                "newton_direction_norms": newton_direction_norms,
                "linesearch_calls": trace_calls,
            },
        }
        if rank == 0:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2)
    finally:
        x_after_prev.destroy()
        x.destroy()
        assembler.cleanup()


if __name__ == "__main__":
    main()

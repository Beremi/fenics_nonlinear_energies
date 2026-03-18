#!/usr/bin/env python3
"""Trace the origin of non-finite HE line-search energies on a target step/iter."""

from __future__ import annotations

import argparse
import json
import os

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
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--total-steps", type=int, default=24)
    p.add_argument("--target-iter", type=int, default=3)
    p.add_argument("--linesearch-a", type=float, default=-0.5)
    p.add_argument("--linesearch-b", type=float, default=2.0)
    p.add_argument("--linesearch-tol", type=float, default=1e-3)
    p.add_argument("--ksp-rtol", type=float, default=1e-1)
    p.add_argument("--ksp-max-it", type=int, default=100)
    p.add_argument("--nproc-threads", type=int, default=1)
    p.add_argument("--out", required=True)
    return p


def _compute_owned_det_stats(assembler, vec) -> dict[str, float | int | None]:
    full, _ = assembler._allgather_full_owned(np.asarray(vec.array[:], dtype=np.float64))
    v_local, _ = assembler._build_v_local(full)
    elems = assembler.local_data.elems_local_np
    dphix = assembler.local_data.local_elem_data["dphix"]
    dphiy = assembler.local_data.local_elem_data["dphiy"]
    dphiz = assembler.local_data.local_elem_data["dphiz"]
    owned_mask = assembler.local_data.energy_weights > 0.5

    v_e = v_local[elems]
    vx = v_e[:, 0::3]
    vy = v_e[:, 1::3]
    vz = v_e[:, 2::3]

    F11 = np.sum(vx * dphix, axis=1)
    F12 = np.sum(vx * dphiy, axis=1)
    F13 = np.sum(vx * dphiz, axis=1)
    F21 = np.sum(vy * dphix, axis=1)
    F22 = np.sum(vy * dphiy, axis=1)
    F23 = np.sum(vy * dphiz, axis=1)
    F31 = np.sum(vz * dphix, axis=1)
    F32 = np.sum(vz * dphiy, axis=1)
    F33 = np.sum(vz * dphiz, axis=1)

    det = (
        F11 * (F22 * F33 - F23 * F32)
        - F12 * (F21 * F33 - F23 * F31)
        + F13 * (F21 * F32 - F22 * F31)
    )

    det_owned = det[owned_mask]
    if det_owned.size == 0:
        local_min = np.inf
        local_nonpos = 0
        local_nonfinite = 0
    else:
        local_min = float(np.min(det_owned))
        local_nonpos = int(np.sum(det_owned <= 0.0))
        local_nonfinite = int(np.sum(~np.isfinite(det_owned)))

    comm = assembler.comm
    global_min = comm.allreduce(local_min, op=MPI.MIN)
    global_nonpos = comm.allreduce(local_nonpos, op=MPI.SUM)
    global_nonfinite = comm.allreduce(local_nonfinite, op=MPI.SUM)
    return {
        "min_det_owned": None if not np.isfinite(global_min) else float(global_min),
        "nonpositive_det_owned": int(global_nonpos),
        "nonfinite_det_owned": int(global_nonfinite),
    }


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
    g = x.duplicate()
    h = x.duplicate()
    x_trial = x.duplicate()
    x_prev = x.duplicate()
    p = x.duplicate()

    force_pc_setup_next = True

    def hessian_solve_fn(vec, rhs, sol):
        nonlocal force_pc_setup_next, gamg_coords
        u_owned = np.array(vec.array[:], dtype=np.float64)
        assembler.assemble_hessian(u_owned)
        ksp.setOperators(A)
        if gamg_coords is not None:
            pc.setCoordinates(gamg_coords)
            gamg_coords = None
        ksp.setTolerances(rtol=float(args.ksp_rtol), max_it=int(args.ksp_max_it))
        if force_pc_setup_next:
            ksp.setUp()
            force_pc_setup_next = False
        ksp.solve(rhs, sol)
        ksp_its = int(ksp.getIterationNumber())
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
            minimizers.newton(
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

        angle = int(args.step) * rot
        u0_step = rotate_right_face_from_reference(
            params["u_0_ref"], params["nodes2coord"], angle, params["right_nodes"]
        )
        assembler.update_dirichlet(u0_step)

        iter_records = []
        for it in range(1, int(args.target_iter) + 1):
            fx = float(assembler.energy_fn(x))
            assembler.gradient_fn(x, g)
            grad_norm = float(g.norm(PETSc.NormType.NORM_2))

            g.copy(p)
            p.scale(-1.0)
            ksp_its = hessian_solve_fn(x, p, h)
            pnorm = float(h.norm(PETSc.NormType.NORM_2))
            h.copy(p)
            x.copy(x_prev)

            samples = []

            def energy_at_alpha(alpha_local):
                x_trial.waxpy(alpha_local, p, x_prev)
                trial_energy = float(assembler.energy_fn(x_trial))
                det_stats = _compute_owned_det_stats(assembler, x_trial)
                rec = {
                    "alpha": float(alpha_local),
                    "energy": None if not np.isfinite(trial_energy) else trial_energy,
                    "nonfinite_energy": bool(not np.isfinite(trial_energy)),
                    **det_stats,
                }
                samples.append(rec)
                if not np.isfinite(trial_energy):
                    return np.inf
                return trial_energy

            alpha_raw, ls_evals = minimizers.golden_section_search(
                energy_at_alpha,
                float(args.linesearch_a),
                float(args.linesearch_b),
                float(args.linesearch_tol),
            )
            final_trial = energy_at_alpha(alpha_raw)
            accepted = bool(np.isfinite(final_trial) and final_trial < fx)
            if accepted:
                x_trial.copy(x)
            else:
                x_prev.copy(x)

            probe_alphas = [
                -0.5,
                -0.1,
                -0.01,
                -1e-3,
                0.0,
                1e-6,
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                5e-2,
                1e-1,
                2e-1,
                3e-1,
                4e-1,
                4.5491502812526297e-1,
                5e-1,
                1.0,
                2.0,
            ]
            probe_records = []
            if it == int(args.target_iter):
                for alpha_probe in probe_alphas:
                    x_trial.waxpy(float(alpha_probe), p, x_prev)
                    trial_energy = float(assembler.energy_fn(x_trial))
                    probe_records.append(
                        {
                            "alpha": float(alpha_probe),
                            "energy": None if not np.isfinite(trial_energy) else trial_energy,
                            "nonfinite_energy": bool(not np.isfinite(trial_energy)),
                            **_compute_owned_det_stats(assembler, x_trial),
                        }
                    )

            iter_records.append(
                {
                    "it": int(it),
                    "energy_before": float(fx),
                    "grad_norm": float(grad_norm),
                    "newton_direction_norm": float(pnorm),
                    "ksp_its": int(ksp_its),
                    "raw_alpha": float(alpha_raw),
                    "accepted": bool(accepted),
                    "ls_evals": int(ls_evals),
                    "samples": samples,
                    "probe_records": probe_records,
                }
            )

        if rank == 0:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "case": {
                            "level": int(args.level),
                            "step": int(args.step),
                            "target_iter": int(args.target_iter),
                            "linesearch_interval": [
                                float(args.linesearch_a),
                                float(args.linesearch_b),
                            ],
                            "linesearch_tol": float(args.linesearch_tol),
                            "ksp_max_it": int(args.ksp_max_it),
                        },
                        "iterations": iter_records,
                    },
                    fh,
                    indent=2,
                )
    finally:
        p.destroy()
        x_prev.destroy()
        x_trial.destroy()
        h.destroy()
        g.destroy()
        x.destroy()
        assembler.cleanup()


if __name__ == "__main__":
    main()

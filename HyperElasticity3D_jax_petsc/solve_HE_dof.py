#!/usr/bin/env python3
"""HyperElasticity 3D solver — DOF-partitioned JAX + PETSc (MPI)."""

import argparse
import json
import os
import time

import numpy as np
from mpi4py import MPI


PROFILE_DEFAULTS = {
    "reference": {
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "pc_setup_on_ksp_cap": True,
        "gamg_threshold": 0.05,
        "gamg_agg_nsmooths": 1,
        "use_near_nullspace": True,
        "gamg_set_coordinates": True,
        "reorder": False,
    },
    "performance": {
        "ksp_type": "gmres",
        "pc_type": "gamg",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "pc_setup_on_ksp_cap": True,
        "gamg_threshold": 0.05,
        "gamg_agg_nsmooths": 1,
        "use_near_nullspace": True,
        "gamg_set_coordinates": True,
        "reorder": False,
    },
}


def _build_parser():
    parser = argparse.ArgumentParser(
        description="HyperElasticity3D DOF-partitioned JAX + PETSc solver"
    )

    parser.add_argument("--level", type=int, default=1, help="Mesh level (1-4)")
    parser.add_argument("--steps", type=int, default=1, help="Number of load steps")
    parser.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help="Total steps corresponding to full 4*2*pi rotation (default: --steps)",
    )
    parser.add_argument("--start_step", type=int, default=1, help="Starting load-step index")

    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_DEFAULTS.keys()),
        default="reference",
        help="Linear solver profile",
    )

    parser.add_argument("--ksp_type", type=str, default=None, help="PETSc KSP type")
    parser.add_argument("--pc_type", type=str, default=None, help="PETSc PC type")
    parser.add_argument("--ksp_rtol", type=float, default=None, help="KSP relative tolerance")
    parser.add_argument("--ksp_max_it", type=int, default=None, help="KSP maximum iterations")
    parser.add_argument(
        "--pc_setup_on_ksp_cap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reuse PC setup and refresh only when previous KSP hit ksp_max_it",
    )
    parser.add_argument(
        "--gamg_threshold",
        type=float,
        default=None,
        help="GAMG threshold (critical for HE; performance profile uses 0.05)",
    )
    parser.add_argument(
        "--gamg_agg_nsmooths",
        type=int,
        default=None,
        help="GAMG agg_nsmooths option",
    )
    parser.add_argument(
        "--gamg_set_coordinates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Set PC coordinates (after setOperators) when using GAMG",
    )
    parser.add_argument(
        "--use_near_nullspace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Attach 6 rigid-body near-nullspace vectors",
    )
    parser.add_argument(
        "--reorder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable RCM DOF reordering in partition",
    )

    parser.add_argument(
        "--local_coloring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use per-rank local coloring assembler",
    )
    parser.add_argument(
        "--hvp_eval_mode",
        choices=("batched", "sequential"),
        default="sequential",
        help="Local-coloring HVP evaluation mode",
    )
    parser.add_argument(
        "--coloring_trials",
        type=int,
        default=10,
        help="Coloring trials per rank (global-coloring mode)",
    )
    parser.add_argument(
        "--assembly_mode",
        choices=("sfd", "element"),
        default="sfd",
        help="Hessian assembly mode: 'sfd' (graph coloring + HVP) or "
             "'element' (analytical element Hessians via jax.hessian)",
    )

    parser.add_argument("--tolf", type=float, default=1e-4, help="Energy-change tolerance")
    parser.add_argument("--tolg", type=float, default=1e-3, help="Gradient-norm tolerance")
    parser.add_argument(
        "--tolg_rel",
        type=float,
        default=1e-3,
        help="Relative gradient tolerance (scaled by initial norm)",
    )
    parser.add_argument("--tolx_rel", type=float, default=1e-3, help="Relative step tolerance")
    parser.add_argument("--tolx_abs", type=float, default=1e-10, help="Absolute step tolerance")
    parser.add_argument("--maxit", type=int, default=100, help="Maximum Newton iterations")

    parser.add_argument("--linesearch_a", type=float, default=-0.5, help="Line-search lower bound")
    parser.add_argument("--linesearch_b", type=float, default=2.0, help="Line-search upper bound")
    parser.add_argument(
        "--linesearch_tol",
        type=float,
        default=1e-3,
        help="Golden-section line-search tolerance",
    )

    parser.add_argument(
        "--retry_on_failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry once on non-finite/maxit with tighter linear settings",
    )
    parser.add_argument("--stop_on_fail", action="store_true", help="Stop load stepping on first failure")

    parser.add_argument(
        "--use_abs_det",
        action="store_true",
        help="Use abs(det(F)) in energy (debug compatibility option)",
    )
    parser.add_argument("--nproc", type=int, default=1, help="OMP thread count per rank")
    parser.add_argument("--save_history", action="store_true", help="Save per-iteration Newton history")
    parser.add_argument(
        "--save_linear_timing",
        action="store_true",
        help="Save per-Newton linear timing breakdown",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-iteration output")
    parser.add_argument("--out", type=str, default="", help="Write JSON output to this file")

    return parser


def _configure_thread_env(nproc):
    threads = max(1, int(nproc))
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false "
        "--xla_force_host_platform_device_count=1"
    )
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)


def _resolve_linear_settings(args):
    settings = dict(PROFILE_DEFAULTS[args.profile])

    overrides = {
        "ksp_type": args.ksp_type,
        "pc_type": args.pc_type,
        "ksp_rtol": args.ksp_rtol,
        "ksp_max_it": args.ksp_max_it,
        "pc_setup_on_ksp_cap": args.pc_setup_on_ksp_cap,
        "gamg_threshold": args.gamg_threshold,
        "gamg_agg_nsmooths": args.gamg_agg_nsmooths,
        "use_near_nullspace": args.use_near_nullspace,
        "gamg_set_coordinates": args.gamg_set_coordinates,
        "reorder": args.reorder,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value

    return settings


def _pc_options(settings):
    opts = {}
    if settings["pc_type"] == "gamg":
        opts["pc_gamg_threshold"] = float(settings["gamg_threshold"])
        opts["pc_gamg_agg_nsmooths"] = int(settings["gamg_agg_nsmooths"])
    return opts


def _needs_repair(result):
    msg = str(result.get("message", "")).lower()
    if not np.isfinite(float(result.get("fun", np.nan))):
        return True
    return (
        "non-finite" in msg
        or "nonfinite" in msg
        or "nan" in msg
        or "maximum number of iterations reached" in msg
    )


def _build_gamg_coordinates(part, freedofs, nodes2coord):
    """Build local coordinates for block rows (bs=3)."""
    freedofs = np.asarray(freedofs, dtype=np.int64)
    owned_orig_free = part.perm[part.lo:part.hi]
    owned_total_dofs = freedofs[owned_orig_free]

    if owned_total_dofs.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if owned_total_dofs.size % 3 != 0:
        raise RuntimeError("Owned DOFs are not divisible by 3; cannot build block coordinates")

    blocks = owned_total_dofs.reshape(-1, 3)
    contiguous = (
        np.all(blocks[:, 1] == blocks[:, 0] + 1)
        and np.all(blocks[:, 2] == blocks[:, 0] + 2)
    )
    same_node = (
        np.all(blocks[:, 0] // 3 == blocks[:, 1] // 3)
        and np.all(blocks[:, 0] // 3 == blocks[:, 2] // 3)
    )
    if not (contiguous and same_node):
        raise RuntimeError(
            "DOF ordering does not preserve xyz triplets. "
            "Disable reordering for GAMG coordinates / block size 3."
        )

    node_ids = blocks[:, 0] // 3
    return np.asarray(nodes2coord[node_ids], dtype=np.float64)


def run(args):

    from HyperElasticity3D_jax_petsc.mesh import MeshHyperElasticity3D
    from HyperElasticity3D_jax_petsc.parallel_hessian_dof import (
        LocalColoringAssembler,
        ParallelDOFHessianAssembler,
    )
    from HyperElasticity3D_jax_petsc.rotate_boundary import rotate_right_face_from_reference
    from tools_petsc4py.minimizers import newton

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    settings = _resolve_linear_settings(args)
    pc_options = _pc_options(settings)

    mesh_obj = MeshHyperElasticity3D(args.level)
    params, adjacency, u_init = mesh_obj.get_data()

    assembler_cls = LocalColoringAssembler if args.local_coloring else ParallelDOFHessianAssembler
    assembler_kwargs = dict(
        params=params,
        comm=comm,
        adjacency=adjacency,
        coloring_trials_per_rank=args.coloring_trials,
        ksp_rtol=float(settings["ksp_rtol"]),
        ksp_type=str(settings["ksp_type"]),
        pc_type=str(settings["pc_type"]),
        ksp_max_it=int(settings["ksp_max_it"]),
        use_near_nullspace=bool(settings["use_near_nullspace"]),
        pc_options=pc_options,
        reorder=bool(settings["reorder"]),
        use_abs_det=bool(args.use_abs_det),
    )
    if args.local_coloring:
        assembler_kwargs["hvp_eval_mode"] = str(args.hvp_eval_mode)
    setup_start = time.perf_counter()
    assembler = assembler_cls(**assembler_kwargs)
    assembler.A.setBlockSize(3)

    use_element_assembly = (args.assembly_mode == "element")
    if use_element_assembly:
        assembler.setup_element_hessian()

    setup_time = time.perf_counter() - setup_start

    u_init_reordered = np.asarray(u_init, dtype=np.float64)[assembler.part.perm]
    x = assembler.create_vec(u_init_reordered)
    x_step_start = x.duplicate()

    ksp = assembler.ksp
    A = assembler.A
    pc = ksp.getPC()

    gamg_coords = None
    if settings["pc_type"] == "gamg" and settings["gamg_set_coordinates"]:
        gamg_coords = _build_gamg_coordinates(
            assembler.part, params["freedofs"], params["nodes2coord"]
        )

    rotation_per_iter = 4.0 * 2.0 * np.pi / float(args.total_steps)
    ls_primary = (float(args.linesearch_a), float(args.linesearch_b))

    if rank == 0 and not args.quiet:
        print(
            f"HE 3D DOF solver | level={args.level} np={nprocs} profile={args.profile} "
            f"ksp={settings['ksp_type']} pc={settings['pc_type']} setup={setup_time:.3f}s",
            flush=True,
        )

    step_records = []

    try:
        for step in range(args.start_step, args.start_step + args.steps):
            angle = step * rotation_per_iter

            # BC update from original coordinates (no cumulative drift).
            u0_step = rotate_right_face_from_reference(
                params["u_0_ref"], params["nodes2coord"], angle, params["right_nodes"]
            )
            assembler.update_dirichlet(u0_step)

            # Save current iterate as the load-step start point.
            x.copy(x_step_start)

            attempt_specs = [
                (
                    "primary",
                    ls_primary,
                    float(settings["ksp_rtol"]),
                    int(settings["ksp_max_it"]),
                )
            ]
            if args.retry_on_failure:
                attempt_specs.append(
                    (
                        "repair",
                        (ls_primary[0], min(ls_primary[1], 1.0)),
                        max(1e-12, float(settings["ksp_rtol"]) * 0.1),
                        max(int(settings["ksp_max_it"]) + 1, int(2 * settings["ksp_max_it"])),
                    )
                )

            result = None
            used_attempt = "primary"
            used_ksp_rtol = float(settings["ksp_rtol"])
            used_ksp_max_it = int(settings["ksp_max_it"])
            used_linesearch = ls_primary
            step_time = 0.0
            linear_iters_this_attempt = []
            linear_timing_records = []

            for idx, (attempt_name, ls_interval, ksp_rtol_attempt, ksp_max_it_attempt) in enumerate(
                attempt_specs
            ):
                # Restore step-start state for each attempt.
                x_step_start.copy(x)

                force_pc_setup_next = True
                linear_iters_this_attempt = []
                if args.save_linear_timing:
                    linear_timing_records = []

                def hessian_solve_fn(vec, rhs, sol):
                    nonlocal force_pc_setup_next

                    t_asm0 = time.perf_counter()
                    u_owned = np.array(vec.array[:], dtype=np.float64)
                    if use_element_assembly:
                        assembler.assemble_hessian_element(u_owned)
                    else:
                        assembler.assemble_hessian(u_owned, variant=2)
                    asm_total_time = time.perf_counter() - t_asm0
                    asm_details = {}
                    if assembler.iter_timings:
                        asm_details = dict(assembler.iter_timings[-1])
                    asm_details["assembly_total_time"] = float(asm_total_time)

                    t_setop0 = time.perf_counter()
                    ksp.setOperators(A)
                    nonlocal gamg_coords
                    if gamg_coords is not None:
                        pc.setCoordinates(gamg_coords)
                        gamg_coords = None
                    t_setop = time.perf_counter() - t_setop0

                    t_tol0 = time.perf_counter()
                    ksp.setTolerances(
                        rtol=float(ksp_rtol_attempt), max_it=int(ksp_max_it_attempt)
                    )
                    t_tol = time.perf_counter() - t_tol0

                    t_setup0 = time.perf_counter()
                    if settings["pc_setup_on_ksp_cap"]:
                        if force_pc_setup_next:
                            ksp.setUp()
                            force_pc_setup_next = False
                    else:
                        ksp.setUp()
                    t_setup = time.perf_counter() - t_setup0

                    t_solve0 = time.perf_counter()
                    ksp.solve(rhs, sol)
                    t_solve = time.perf_counter() - t_solve0
                    ksp_its = int(ksp.getIterationNumber())
                    linear_iters_this_attempt.append(ksp_its)

                    if settings["pc_setup_on_ksp_cap"] and ksp_its >= int(ksp_max_it_attempt):
                        force_pc_setup_next = True

                    if args.save_linear_timing:
                        record = {
                            "assemble_total_time": float(asm_total_time),
                            "assemble_p2p_exchange": float(asm_details.get("p2p_exchange", 0.0)),
                            "assemble_hvp_compute": float(asm_details.get("hvp_compute", 0.0)),
                            "assemble_extraction": float(asm_details.get("extraction", 0.0)),
                            "assemble_coo_assembly": float(asm_details.get("coo_assembly", 0.0)),
                            "setop_time": float(t_setop),
                            "set_tolerances_time": float(t_tol),
                            "pc_setup_time": float(t_setup),
                            "solve_time": float(t_solve),
                            "linear_total_time": float(
                                asm_total_time + t_setop + t_tol + t_setup + t_solve
                            ),
                            "ksp_its": int(ksp_its),
                            "attempt": attempt_name,
                        }
                        linear_timing_records.append(record)

                    return ksp_its

                t0 = time.perf_counter()
                result = newton(
                    energy_fn=assembler.energy_fn,
                    gradient_fn=assembler.gradient_fn,
                    hessian_solve_fn=hessian_solve_fn,
                    x=x,
                    tolf=float(args.tolf),
                    tolg=float(args.tolg),
                    tolg_rel=float(args.tolg_rel),
                    linesearch_tol=float(args.linesearch_tol),
                    linesearch_interval=ls_interval,
                    maxit=int(args.maxit),
                    tolx_rel=float(args.tolx_rel),
                    tolx_abs=float(args.tolx_abs),
                    require_all_convergence=True,
                    fail_on_nonfinite=True,
                    verbose=(not args.quiet),
                    comm=comm,
                    ghost_update_fn=None,
                    save_history=bool(args.save_history),
                )
                step_time = time.perf_counter() - t0

                used_attempt = attempt_name
                used_ksp_rtol = float(ksp_rtol_attempt)
                used_ksp_max_it = int(ksp_max_it_attempt)
                used_linesearch = ls_interval

                needs_repair = _needs_repair(result)
                if needs_repair and idx + 1 < len(attempt_specs):
                    if rank == 0 and not args.quiet:
                        print(
                            f"Step {step}: retrying with repair settings "
                            f"(rtol={used_ksp_rtol:.3e}, ksp_max_it={used_ksp_max_it}, "
                            f"ls=[{used_linesearch[0]:.3g},{used_linesearch[1]:.3g}])",
                            flush=True,
                        )
                    continue
                break

            if result is None:
                raise RuntimeError("Newton solver did not return a result")

            step_record = {
                "step": int(step),
                "angle": float(angle),
                "time": float(round(step_time, 6)),
                "nit": int(result["nit"]),
                "linear_iters": int(sum(linear_iters_this_attempt)),
                "energy": float(result["fun"]),
                "message": str(result["message"]),
                "attempt": used_attempt,
                "ksp_rtol_used": float(used_ksp_rtol),
                "ksp_max_it_used": int(used_ksp_max_it),
                "linesearch_interval_used": [
                    float(used_linesearch[0]),
                    float(used_linesearch[1]),
                ],
            }
            if args.save_history:
                step_record["history"] = result.get("history", [])
            if args.save_linear_timing:
                step_record["linear_timing"] = list(linear_timing_records)
            step_records.append(step_record)

            if rank == 0 and not args.quiet:
                print(
                    f"step={step_record['step']:3d} angle={step_record['angle']:.6f} "
                    f"time={step_record['time']:.3f}s nit={step_record['nit']:3d} "
                    f"ksp={step_record['linear_iters']:5d} "
                    f"energy={step_record['energy']:.6e} "
                    f"[{step_record['message']}]",
                    flush=True,
                )

            if args.stop_on_fail and "converged" not in step_record["message"].lower():
                if rank == 0 and not args.quiet:
                    print(f"Stopping at step {step} due to failure message.", flush=True)
                break

    finally:
        x_step_start.destroy()
        x.destroy()
        assembler.cleanup()

    return {
        "mesh_level": int(args.level),
        "total_dofs": int(len(params["u_0"])),
        "free_dofs": int(assembler.part.n_free),
        "steps": step_records,
        "metadata": {
            "profile": args.profile,
            "nprocs": nprocs,
            "nproc_threads": max(1, int(args.nproc)),
            "linear_solver": {
                "ksp_type": str(settings["ksp_type"]),
                "pc_type": str(settings["pc_type"]),
                "ksp_rtol": float(settings["ksp_rtol"]),
                "ksp_max_it": int(settings["ksp_max_it"]),
                "pc_setup_on_ksp_cap": bool(settings["pc_setup_on_ksp_cap"]),
                "gamg_threshold": float(settings["gamg_threshold"]),
                "gamg_agg_nsmooths": int(settings["gamg_agg_nsmooths"]),
                "gamg_set_coordinates": bool(settings["gamg_set_coordinates"]),
                "use_near_nullspace": bool(settings["use_near_nullspace"]),
                "matrix_block_size": 3,
                "reorder": bool(settings["reorder"]),
                "hvp_eval_mode": str(getattr(assembler, "_hvp_eval_mode", "batched")),
                "assembly_mode": str(args.assembly_mode),
            },
            "newton": {
                "tolf": float(args.tolf),
                "tolg": float(args.tolg),
                "tolg_rel": float(args.tolg_rel),
                "tolx_rel": float(args.tolx_rel),
                "tolx_abs": float(args.tolx_abs),
                "maxit": int(args.maxit),
                "require_all_convergence": True,
                "fail_on_nonfinite": True,
                "linesearch_interval": [float(args.linesearch_a), float(args.linesearch_b)],
                "linesearch_tol": float(args.linesearch_tol),
            },
            "load_stepping": {
                "start_step": int(args.start_step),
                "steps": int(args.steps),
                "total_steps": int(args.total_steps),
                "rotation_per_iter": float(rotation_per_iter),
                "retry_on_failure": bool(args.retry_on_failure),
            },
        },
    }


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.total_steps is None:
        args.total_steps = args.steps

    _configure_thread_env(args.nproc)

    result = run(args)

    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(result, indent=2))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

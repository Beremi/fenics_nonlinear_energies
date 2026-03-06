"""
HyperElasticity 3D — solver logic (DOF-partitioned JAX + PETSc).

Provides ``PROFILE_DEFAULTS`` and ``run(args)`` which runs all load steps.
CLI entry point (argparse) is in ``solve_HE_dof.py``.
"""

import time

import numpy as np
from mpi4py import MPI

from HyperElasticity3D_petsc_support.mesh import MeshHyperElasticity3D
from HyperElasticity3D_jax_petsc.parallel_hessian_dof import (
    LocalColoringAssembler,
    ParallelDOFHessianAssembler,
)
from HyperElasticity3D_jax_petsc.reordered_element_assembler import (
    HEReorderedElementAssembler,
)
from HyperElasticity3D_petsc_support.rotate_boundary import rotate_right_face_from_reference
from tools_petsc4py.minimizers import newton


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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    total_runtime_start = time.perf_counter()

    settings = _resolve_linear_settings(args)
    pc_options = _pc_options(settings)
    use_element_assembly = args.assembly_mode == "element"
    element_reorder_mode = str(
        getattr(args, "element_reorder_mode", None) or "block_xyz"
    )

    mesh_obj = MeshHyperElasticity3D(args.level)
    params, adjacency, u_init = mesh_obj.get_data()

    setup_start = time.perf_counter()
    if use_element_assembly:
        if not args.local_coloring:
            raise ValueError("--assembly_mode element requires --local_coloring")
        assembler = HEReorderedElementAssembler(
            params=params,
            comm=comm,
            adjacency=adjacency,
            ksp_rtol=float(settings["ksp_rtol"]),
            ksp_type=str(settings["ksp_type"]),
            pc_type=str(settings["pc_type"]),
            ksp_max_it=int(settings["ksp_max_it"]),
            use_near_nullspace=bool(settings["use_near_nullspace"]),
            pc_options=pc_options,
            reorder_mode=element_reorder_mode,
            use_abs_det=bool(args.use_abs_det),
        )
    else:
        assembler_cls = (
            LocalColoringAssembler if args.local_coloring else ParallelDOFHessianAssembler
        )
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
        assembler = assembler_cls(**assembler_kwargs)
        assembler.A.setBlockSize(3)

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
    use_trust_region = bool(getattr(args, "use_trust_region", False))
    trust_radius_init = float(getattr(args, "trust_radius_init", 1.0))
    trust_radius_min = float(getattr(args, "trust_radius_min", 1e-8))
    trust_radius_max = float(getattr(args, "trust_radius_max", 1e6))
    trust_shrink = float(getattr(args, "trust_shrink", 0.5))
    trust_expand = float(getattr(args, "trust_expand", 1.5))
    trust_eta_shrink = float(getattr(args, "trust_eta_shrink", 0.05))
    trust_eta_expand = float(getattr(args, "trust_eta_expand", 0.75))
    trust_max_reject = int(getattr(args, "trust_max_reject", 6))

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

            u0_step = rotate_right_face_from_reference(
                params["u_0_ref"], params["nodes2coord"], angle, params["right_nodes"]
            )
            assembler.update_dirichlet(u0_step)

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
                    hessian_matvec_fn=lambda _x, vin, vout: assembler.A.mult(vin, vout),
                    save_history=bool(args.save_history),
                    trust_region=use_trust_region,
                    trust_radius_init=trust_radius_init,
                    trust_radius_min=trust_radius_min,
                    trust_radius_max=trust_radius_max,
                    trust_shrink=trust_shrink,
                    trust_expand=trust_expand,
                    trust_eta_shrink=trust_eta_shrink,
                    trust_eta_expand=trust_eta_expand,
                    trust_max_reject=trust_max_reject,
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
        "setup_time": float(round(setup_time, 6)),
        "solve_time_total": float(round(sum(step["time"] for step in step_records), 6)),
        "total_time": float(round(time.perf_counter() - total_runtime_start, 6)),
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
                "element_reorder_mode": (
                    element_reorder_mode if use_element_assembly else None
                ),
                "distribution_strategy": str(
                    getattr(assembler, "distribution_strategy", "reduced_free_dofs")
                ),
                "assembler": assembler.__class__.__name__,
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
                "trust_region": bool(use_trust_region),
                "trust_radius_init": float(trust_radius_init),
                "trust_radius_min": float(trust_radius_min),
                "trust_radius_max": float(trust_radius_max),
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

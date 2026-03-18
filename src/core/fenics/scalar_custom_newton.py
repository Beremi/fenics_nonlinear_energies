"""Shared scalar FEniCS custom-Newton driver."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_matrix,
    set_bc,
)
from petsc4py import PETSc

from src.core.benchmark.repair import build_retry_attempts, needs_solver_repair
from src.core.petsc.fenics_tools import ghost_update as _ghost_update
from src.core.petsc.minimizers import newton
from src.core.petsc.trust_ksp import ksp_cg_set_radius


LinearRecordExtras = Callable[[PETSc.KSP, PETSc.Vec], dict[str, object]]
StateExporter = Callable[[str, dict[str, object], dict[str, object], dict[str, object], Any], None]


@dataclass(frozen=True)
class ScalarCustomNewtonProblemSpec:
    problem_name: str
    build_mesh: Callable[[int, MPI.Comm], Any]
    build_function_space: Callable[[Any], Any]
    build_boundary_conditions: Callable[[Any, Any, Any], tuple[list, dict[str, object]]]
    build_forms: Callable[[Any, dict[str, object], Any], dict[str, object]]
    make_initial_guess: Callable[[Any, dict[str, object], Any], object]
    default_linear_options: Mapping[str, object]
    result_metadata: Mapping[str, object] | Callable[[Any], Mapping[str, object]]
    linear_record_extras: LinearRecordExtras | None = None
    newton_defaults: Mapping[str, object] = field(default_factory=dict)
    export_state: StateExporter | None = None


def run_scalar_custom_newton(spec: ScalarCustomNewtonProblemSpec, args) -> dict:
    """Run a scalar FEniCS problem with the shared custom-Newton scaffold."""
    comm = MPI.COMM_WORLD
    total_runtime_start = time.perf_counter()
    setup_start = time.perf_counter()

    msh = spec.build_mesh(int(args.level), comm)
    V = spec.build_function_space(msh)
    total_dofs = int(V.dofmap.index_map.size_global)

    bcs, raw_problem_data = spec.build_boundary_conditions(V, msh, None)
    raw_problem_data = dict(raw_problem_data)
    raw_problem_data.setdefault("bcs", bcs)

    u = fem.Function(V)
    v = ufl.TestFunction(V)
    w = ufl.TrialFunction(V)
    state = {
        "mesh": msh,
        "u": u,
        "v": v,
        "w": w,
    }
    form_payload = spec.build_forms(V, state, None)
    J_energy = form_payload["energy"]
    dJ = ufl.derivative(J_energy, u, v)
    ddJ = ufl.derivative(dJ, u, w)

    grad_form = fem.form(dJ)
    hessian_form = fem.form(ddJ)
    u_ls = fem.Function(V)
    energy_ls_form = fem.form(ufl.replace(J_energy, {u: u_ls}))

    initial_guess = spec.make_initial_guess(V, raw_problem_data, args)
    if callable(initial_guess):
        initial_guess(u)
    elif initial_guess is not None:
        x_init = u.x.petsc_vec
        x_init.array[:] = initial_guess
    x = u.x.petsc_vec
    _ghost_update(x)
    x.assemble()
    set_bc(x, bcs)

    A = create_matrix(hessian_form)
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setType(str(args.ksp_type))
    pc = ksp.getPC()
    pc.setType(str(args.pc_type))
    if str(args.pc_type) == "hypre":
        pc.setHYPREType("boomeramg")
    elif str(args.pc_type) == "gamg":
        opts = PETSc.Options()
        opts["pc_gamg_threshold"] = float(args.gamg_threshold)
        opts["pc_gamg_agg_nsmooths"] = int(args.gamg_agg_nsmooths)
    ksp.setTolerances(rtol=float(args.ksp_rtol), max_it=int(args.ksp_max_it))
    ksp.setFromOptions()

    gamg_coords = None
    if str(args.pc_type) == "gamg" and bool(args.gamg_set_coordinates):
        index_map = V.dofmap.index_map
        gamg_coords = V.tabulate_dof_coordinates()[: index_map.size_local, :]

    trust_ksp_subproblem = bool(
        args.use_trust_region and str(args.ksp_type).lower() in {"stcg", "nash", "gltr"}
    )
    step_time_limit_s = getattr(args, "step_time_limit_s", None)
    setup_time = time.perf_counter() - setup_start

    linear_timing_records: list[dict[str, object]] = []
    force_pc_setup_next = True
    used_ksp_rtol = float(args.ksp_rtol)
    used_ksp_max_it = int(args.ksp_max_it)

    def _sync_primal(vec):
        vec.copy(u.x.petsc_vec)
        _ghost_update(u.x.petsc_vec)
        u.x.petsc_vec.assemble()

    def energy_fn(vec):
        vec.copy(u_ls.x.petsc_vec)
        u_ls.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD,
        )
        local_val = fem.assemble_scalar(energy_ls_form)
        return comm.allreduce(local_val, op=MPI.SUM)

    def gradient_fn(vec, g):
        _sync_primal(vec)
        with g.localForm() as g_loc:
            g_loc.set(0.0)
        assemble_vector(g, grad_form)
        apply_lifting(g, [hessian_form], [bcs], x0=[vec])
        g.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(g, bcs, vec)

    def _assemble_and_solve(
        vec,
        rhs,
        sol,
        ksp_rtol_attempt,
        ksp_max_it_attempt,
        trust_radius=None,
    ):
        nonlocal force_pc_setup_next, gamg_coords

        _sync_primal(vec)

        t_asm0 = time.perf_counter()
        A.zeroEntries()
        assemble_matrix(A, hessian_form, bcs=bcs)
        A.assemble()
        asm_total_time = time.perf_counter() - t_asm0

        if trust_radius is not None:
            ksp_cg_set_radius(ksp, float(trust_radius))

        t_setop0 = time.perf_counter()
        ksp.setOperators(A)
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
        if args.pc_setup_on_ksp_cap:
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

        if args.pc_setup_on_ksp_cap and ksp_its >= int(ksp_max_it_attempt):
            force_pc_setup_next = True

        if args.save_linear_timing:
            record: dict[str, object] = {
                "assemble_total_time": float(asm_total_time),
                "setop_time": float(t_setop),
                "set_tolerances_time": float(t_tol),
                "pc_setup_time": float(t_setup),
                "solve_time": float(t_solve),
                "linear_total_time": float(
                    asm_total_time + t_setop + t_tol + t_setup + t_solve
                ),
                "ksp_its": int(ksp_its),
            }
            if spec.linear_record_extras is not None:
                record.update(spec.linear_record_extras(ksp, rhs))
            if trust_radius is not None:
                record["trust_radius"] = float(trust_radius)
            linear_timing_records.append(record)

        return ksp_its

    def hessian_solve_fn(vec, rhs, sol):
        return _assemble_and_solve(vec, rhs, sol, used_ksp_rtol, used_ksp_max_it)

    def trust_subproblem_solve_fn(vec, rhs, sol, trust_radius):
        return _assemble_and_solve(
            vec,
            rhs,
            sol,
            used_ksp_rtol,
            used_ksp_max_it,
            trust_radius=float(trust_radius),
        )

    x_initial = x.duplicate()
    x.copy(x_initial)

    linesearch_interval = (float(args.linesearch_a), float(args.linesearch_b))
    attempt_specs = build_retry_attempts(
        retry_on_failure=bool(args.retry_on_failure),
        linesearch_interval=linesearch_interval,
        ksp_rtol=float(args.ksp_rtol),
        ksp_max_it=int(args.ksp_max_it),
    )

    result = None
    solve_time = 0.0
    used_attempt = "primary"
    used_linesearch = linesearch_interval
    steps: list[dict[str, object]] = []

    try:
        for idx, (attempt_name, ls_interval, ksp_rtol_attempt, ksp_max_it_attempt) in enumerate(
            attempt_specs
        ):
            x_initial.copy(x)
            force_pc_setup_next = True
            if args.save_linear_timing:
                linear_timing_records = []

            used_attempt = attempt_name
            used_linesearch = ls_interval
            used_ksp_rtol = float(ksp_rtol_attempt)
            used_ksp_max_it = int(ksp_max_it_attempt)

            t0 = time.perf_counter()
            result = newton(
                energy_fn=energy_fn,
                gradient_fn=gradient_fn,
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
                ghost_update_fn=_ghost_update,
                project_fn=lambda vec: set_bc(vec, bcs),
                hessian_matvec_fn=lambda _x, vin, vout: A.mult(vin, vout),
                trust_subproblem_solve_fn=(
                    trust_subproblem_solve_fn if trust_ksp_subproblem else None
                ),
                trust_subproblem_line_search=bool(args.trust_subproblem_line_search),
                save_history=bool(args.save_history),
                trust_region=bool(args.use_trust_region),
                trust_radius_init=float(args.trust_radius_init),
                trust_radius_min=float(args.trust_radius_min),
                trust_radius_max=float(args.trust_radius_max),
                trust_shrink=float(args.trust_shrink),
                trust_expand=float(args.trust_expand),
                trust_eta_shrink=float(args.trust_eta_shrink),
                trust_eta_expand=float(args.trust_eta_expand),
                trust_max_reject=int(args.trust_max_reject),
                step_time_limit_s=step_time_limit_s,
            )
            solve_time = time.perf_counter() - t0

            if needs_solver_repair(result) and idx + 1 < len(attempt_specs):
                continue
            break

        if result is None:
            raise RuntimeError("Newton solver did not return a result")

        step_record: dict[str, object] = {
            "step": 1,
            "time": float(round(solve_time, 6)),
            "nit": int(result["nit"]),
            "linear_iters": int(
                sum(int(rec.get("ksp_its", 0)) for rec in linear_timing_records)
            ),
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
        if step_time_limit_s is not None:
            step_record["step_time_limit_s"] = float(step_time_limit_s)
            step_record["kill_switch_exceeded"] = bool(
                solve_time > float(step_time_limit_s)
            )
        if args.save_history:
            step_record["history"] = result.get("history", [])
        if args.save_linear_timing:
            step_record["linear_timing"] = list(linear_timing_records)
        steps = [step_record]

    finally:
        state_out = str(getattr(args, "state_out", "") or "")
        if state_out and spec.export_state is not None:
            spec.export_state(state_out, state, raw_problem_data, result or {}, args)
        x_initial.destroy()
        ksp.destroy()
        A.destroy()

    extra_metadata = (
        spec.result_metadata(args)
        if callable(spec.result_metadata)
        else dict(spec.result_metadata)
    )
    return {
        "mesh_level": int(args.level),
        "total_dofs": int(total_dofs),
        "free_dofs": int(total_dofs - len(raw_problem_data.get("bc_dofs", []))),
        "setup_time": float(round(setup_time, 6)),
        "solve_time_total": float(round(sum(step["time"] for step in steps), 6)),
        "total_time": float(round(time.perf_counter() - total_runtime_start, 6)),
        "steps": steps,
        "metadata": {
            "nprocs": int(comm.size),
            "problem": {
                "name": spec.problem_name,
                **extra_metadata,
            },
            "linear_solver": {
                "ksp_type": str(args.ksp_type),
                "pc_type": str(args.pc_type),
                "ksp_rtol": float(args.ksp_rtol),
                "ksp_max_it": int(args.ksp_max_it),
                "pc_setup_on_ksp_cap": bool(args.pc_setup_on_ksp_cap),
                "gamg_threshold": float(args.gamg_threshold),
                "gamg_agg_nsmooths": int(args.gamg_agg_nsmooths),
                "gamg_set_coordinates": bool(args.gamg_set_coordinates),
                "assembly_mode": "fenics",
                "assembler": "FEniCSFormAssembly",
                "distribution_strategy": "dolfinx_mesh_partition",
                "trust_subproblem_solver": (
                    "petsc_ksp" if trust_ksp_subproblem else "direct_linear_solve"
                ),
                "trust_subproblem_line_search": bool(args.trust_subproblem_line_search),
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
                "trust_region": bool(args.use_trust_region),
                "trust_radius_init": float(args.trust_radius_init),
                "trust_radius_min": float(args.trust_radius_min),
                "trust_radius_max": float(args.trust_radius_max),
                "trust_subproblem_line_search": bool(args.trust_subproblem_line_search),
                "step_time_limit_s": (
                    None if step_time_limit_s is None else float(step_time_limit_s)
                ),
            },
        },
    }

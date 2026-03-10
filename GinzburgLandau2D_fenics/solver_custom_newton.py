"""Ginzburg-Landau 2D FEniCS custom-Newton solver logic."""

from __future__ import annotations

import time
from types import SimpleNamespace

import basix.ufl
import h5py
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_matrix,
    set_bc,
)
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from tools_petsc4py.fenics_tools import ghost_update as _ghost_update
from tools_petsc4py.minimizers import newton
from tools_petsc4py.reasons import ksp_reason_name
from tools_petsc4py.trust_ksp import ksp_cg_set_radius


EPS = 0.01


def _load_mesh(mesh_level: int, comm: MPI.Comm):
    rank = comm.rank
    if rank == 0:
        with h5py.File(
            f"mesh_data/GinzburgLandau/GL_level{mesh_level}.h5",
            "r",
            driver="core",
            backing_store=False,
        ) as f:
            points = f["nodes"][:]
            triangles = f["elems"][:].astype(np.int64)
    else:
        points = np.empty((0, 2), dtype=np.float64)
        triangles = np.empty((0, 3), dtype=np.int64)
    c_el = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    return mesh.create_mesh(comm, triangles, c_el, points)


def _sum_step_linear(step: dict) -> int:
    return int(sum(int(rec.get("ksp_its", 0)) for rec in step.get("linear_timing", [])))


def _assemble_time(rec: dict) -> float:
    return float(rec.get("assemble_total_time", rec.get("assemble_time", 0.0)))


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


def run(args):
    comm = MPI.COMM_WORLD
    total_runtime_start = time.perf_counter()
    setup_start = time.perf_counter()

    msh = _load_mesh(int(args.level), comm)
    V = fem.functionspace(msh, ("Lagrange", 1))
    total_dofs = int(V.dofmap.index_map.size_global)

    msh.topology.create_connectivity(1, 2)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    bc_dofs = fem.locate_dofs_topological(V, 1, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0), bc_dofs, V)
    bcs = [bc]

    u = fem.Function(V)
    v = ufl.TestFunction(V)
    w = ufl.TrialFunction(V)

    eps = fem.Constant(msh, ScalarType(EPS))
    J_energy = (
        (eps / 2.0) * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx
        + 0.25 * (u**2 - 1.0) ** 2 * ufl.dx
    )
    dJ = ufl.derivative(J_energy, u, v)
    ddJ = ufl.derivative(dJ, u, w)

    grad_form = fem.form(dJ)
    hessian_form = fem.form(ddJ)
    u_ls = fem.Function(V)
    energy_ls_form = fem.form(ufl.replace(J_energy, {u: u_ls}))

    def initial_guess(x_coords):
        return np.sin(np.pi * (x_coords[0] - 1.0) / 2.0) * np.sin(
            np.pi * (x_coords[1] - 1.0) / 2.0
        )

    u.interpolate(initial_guess)
    x = u.x.petsc_vec
    set_bc(x, bcs)
    _ghost_update(x)
    x.assemble()

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

    linear_timing_records = []
    force_pc_setup_next = True

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

    def _assemble_and_solve(vec, rhs, sol, ksp_rtol_attempt, ksp_max_it_attempt, trust_radius=None):
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

        rhs_norm = float(rhs.norm(PETSc.NormType.NORM_2))
        t_solve0 = time.perf_counter()
        ksp.solve(rhs, sol)
        t_solve = time.perf_counter() - t_solve0
        ksp_its = int(ksp.getIterationNumber())
        ksp_reason_code = int(ksp.getConvergedReason())
        ksp_residual_norm = float(ksp.getResidualNorm())

        if args.pc_setup_on_ksp_cap and ksp_its >= int(ksp_max_it_attempt):
            force_pc_setup_next = True

        if args.save_linear_timing:
            record = {
                "assemble_total_time": float(asm_total_time),
                "setop_time": float(t_setop),
                "set_tolerances_time": float(t_tol),
                "pc_setup_time": float(t_setup),
                "solve_time": float(t_solve),
                "linear_total_time": float(
                    asm_total_time + t_setop + t_tol + t_setup + t_solve
                ),
                "ksp_its": int(ksp_its),
                "ksp_reason_code": int(ksp_reason_code),
                "ksp_reason_name": ksp_reason_name(ksp_reason_code),
                "ksp_residual_norm": float(ksp_residual_norm),
                "rhs_norm": float(rhs_norm),
            }
            if trust_radius is not None:
                record["trust_radius"] = float(trust_radius)
            linear_timing_records.append(record)

        return ksp_its

    used_ksp_rtol = float(args.ksp_rtol)
    used_ksp_max_it = int(args.ksp_max_it)

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

    attempt_specs = [
        (
            "primary",
            (float(args.linesearch_a), float(args.linesearch_b)),
            float(args.ksp_rtol),
            int(args.ksp_max_it),
        )
    ]
    if args.retry_on_failure:
        attempt_specs.append(
            (
                "repair",
                (float(args.linesearch_a), min(float(args.linesearch_b), 1.0)),
                max(1e-12, float(args.ksp_rtol) * 0.1),
                max(int(args.ksp_max_it) + 1, int(2 * args.ksp_max_it)),
            )
        )

    result = None
    solve_time = 0.0
    used_attempt = "primary"
    used_linesearch = (float(args.linesearch_a), float(args.linesearch_b))

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

            needs_repair = _needs_repair(result)
            if needs_repair and idx + 1 < len(attempt_specs):
                continue
            break

        if result is None:
            raise RuntimeError("Newton solver did not return a result")

        step_record = {
            "step": 1,
            "time": float(round(solve_time, 6)),
            "nit": int(result["nit"]),
            "linear_iters": int(sum(int(rec.get("ksp_its", 0)) for rec in linear_timing_records)),
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
        x_initial.destroy()
        ksp.destroy()
        A.destroy()

    return {
        "mesh_level": int(args.level),
        "total_dofs": int(total_dofs),
        "free_dofs": int(total_dofs - len(bc_dofs)),
        "setup_time": float(round(setup_time, 6)),
        "solve_time_total": float(round(sum(step["time"] for step in steps), 6)),
        "total_time": float(round(time.perf_counter() - total_runtime_start, 6)),
        "steps": steps,
        "metadata": {
            "nprocs": int(comm.size),
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


def _flatten_result(result: dict) -> dict:
    steps = list(result.get("steps", []))
    step = steps[0] if steps else {}
    linear_timing = step.get("linear_timing", [])
    asm_cumulative = sum(_assemble_time(rec) for rec in linear_timing)
    pc_setup_cumulative = sum(float(rec.get("pc_setup_time", 0.0)) for rec in linear_timing)
    linear_solve_cumulative = sum(float(rec.get("solve_time", 0.0)) for rec in linear_timing)
    ksp_cumulative = sum(
        float(rec.get("pc_setup_time", 0.0)) + float(rec.get("solve_time", 0.0))
        for rec in linear_timing
    )

    return {
        "mesh_level": int(result["mesh_level"]),
        "total_dofs": int(result["total_dofs"]),
        "dofs": int(result["total_dofs"]),
        "nprocs": int(result["metadata"]["nprocs"]),
        "pc_type": str(result["metadata"]["linear_solver"]["pc_type"]),
        "ksp_rtol": float(result["metadata"]["linear_solver"]["ksp_rtol"]),
        "assembly_mode": "fenics",
        "setup_time": round(float(result["setup_time"]), 4),
        "solve_time": round(float(result.get("solve_time_total", 0.0)), 4),
        "total_time": round(float(result["total_time"]), 4),
        "iters": int(step.get("nit", 0)),
        "energy": round(float(step.get("energy", np.nan)), 10),
        "message": str(step.get("message", "")),
        "total_ksp_its": int(_sum_step_linear(step)),
        "asm_time_cumulative": round(float(asm_cumulative), 4),
        "pc_setup_time_cumulative": round(float(pc_setup_cumulative), 4),
        "linear_solve_time_cumulative": round(float(linear_solve_cumulative), 4),
        "ksp_time_cumulative": round(float(ksp_cumulative), 4),
        "hess_timings": list(linear_timing),
    }


def run_level(
    mesh_level,
    verbose=True,
    pc_type="hypre",
    ksp_rtol=1e-3,
    linesearch_interval=(-0.5, 2.0),
    linesearch_tol=1e-3,
    maxit=100,
    use_trust_region=False,
    trust_radius_init=1.0,
    trust_radius_min=1e-8,
    trust_radius_max=1e6,
    trust_shrink=0.5,
    trust_expand=1.5,
    trust_eta_shrink=0.05,
    trust_eta_expand=0.75,
    trust_max_reject=6,
    ksp_type="gmres",
    ksp_max_it=200,
    tolf=1e-6,
    tolg=1e-5,
    tolg_rel=1e-3,
    tolx_rel=1e-3,
    tolx_abs=1e-10,
    save_history=True,
    save_linear_timing=True,
    quiet=False,
    retry_on_failure=False,
    pc_setup_on_ksp_cap=False,
    gamg_threshold=0.05,
    gamg_agg_nsmooths=1,
    gamg_set_coordinates=True,
    step_time_limit_s=None,
    trust_subproblem_line_search=False,
):
    ns = SimpleNamespace(
        level=int(mesh_level),
        quiet=bool(quiet or (not verbose)),
        save_history=bool(save_history),
        save_linear_timing=bool(save_linear_timing),
        ksp_type=str(ksp_type),
        pc_type=str(pc_type),
        ksp_rtol=float(ksp_rtol),
        ksp_max_it=int(ksp_max_it),
        tolf=float(tolf),
        tolg=float(tolg),
        tolg_rel=float(tolg_rel),
        tolx_rel=float(tolx_rel),
        tolx_abs=float(tolx_abs),
        maxit=int(maxit),
        linesearch_a=float(linesearch_interval[0]),
        linesearch_b=float(linesearch_interval[1]),
        linesearch_tol=float(linesearch_tol),
        use_trust_region=bool(use_trust_region),
        trust_radius_init=float(trust_radius_init),
        trust_radius_min=float(trust_radius_min),
        trust_radius_max=float(trust_radius_max),
        trust_shrink=float(trust_shrink),
        trust_expand=float(trust_expand),
        trust_eta_shrink=float(trust_eta_shrink),
        trust_eta_expand=float(trust_eta_expand),
        trust_max_reject=int(trust_max_reject),
        retry_on_failure=bool(retry_on_failure),
        pc_setup_on_ksp_cap=bool(pc_setup_on_ksp_cap),
        gamg_threshold=float(gamg_threshold),
        gamg_agg_nsmooths=int(gamg_agg_nsmooths),
        gamg_set_coordinates=bool(gamg_set_coordinates),
        step_time_limit_s=step_time_limit_s,
        trust_subproblem_line_search=bool(trust_subproblem_line_search),
    )
    return _flatten_result(run(ns))

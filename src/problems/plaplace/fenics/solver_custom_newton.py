"""pLaplace 2D FEniCS custom-Newton solver logic."""

from __future__ import annotations

from types import SimpleNamespace

import basix.ufl
import h5py
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from petsc4py.PETSc import ScalarType

from src.core.benchmark.results import (
    cumulative_linear_timing,
    sum_step_linear,
)
from src.core.benchmark.state_export import export_scalar_function_state_npz
from src.core.fenics.scalar_custom_newton import (
    ScalarCustomNewtonProblemSpec,
    run_scalar_custom_newton,
)
from src.core.problem_data.hdf5 import mesh_data_path


def _load_mesh(mesh_level: int, comm: MPI.Comm):
    rank = comm.rank
    if rank == 0:
        with h5py.File(
            mesh_data_path("pLaplace", f"pLaplace_level{mesh_level}.h5"),
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


def _build_function_space(msh):
    return fem.functionspace(msh, ("Lagrange", 1))


def _build_boundary_conditions(V, msh, step_ctx):
    del step_ctx
    msh.topology.create_connectivity(1, 2)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    bc_dofs = fem.locate_dofs_topological(V, 1, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0), bc_dofs, V)
    return [bc], {"bc_dofs": bc_dofs}


def _build_forms(V, state, step_ctx):
    del V, step_ctx
    msh = state["mesh"]
    u = state["u"]
    p = 3.0
    f_rhs = fem.Constant(msh, ScalarType(-10.0))
    return {
        "energy": (
            (1.0 / p) * ufl.inner(ufl.grad(u), ufl.grad(u)) ** (p / 2) * ufl.dx
            - f_rhs * u * ufl.dx
        )
    }


def _make_initial_guess(V, raw_problem_data, args):
    del V, raw_problem_data, args

    def _initialise(u):
        np.random.seed(42)
        x = u.x.petsc_vec
        lo, hi = x.getOwnershipRange()
        x.setValues(range(lo, hi), 1e-2 * np.random.rand(hi - lo))

    return _initialise


def _export_state(path, state, raw_problem_data, result, args):
    del raw_problem_data
    steps = list(result.get("steps", []))
    step = steps[-1] if steps else {}
    export_scalar_function_state_npz(
        path,
        state["u"].function_space,
        state["u"],
        mesh_level=int(args.level),
        problem_name="pLaplace2D",
        energy=(None if "energy" not in step else float(step["energy"])),
        metadata={"solver_family": "fenics_custom"},
    )


SPEC = ScalarCustomNewtonProblemSpec(
    problem_name="pLaplace2D",
    build_mesh=_load_mesh,
    build_function_space=_build_function_space,
    build_boundary_conditions=_build_boundary_conditions,
    build_forms=_build_forms,
    make_initial_guess=_make_initial_guess,
    default_linear_options={
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-3,
        "ksp_max_it": 10000,
    },
    result_metadata={"p": 3.0, "rhs_f": -10.0},
    export_state=_export_state,
)


def run(args):
    return run_scalar_custom_newton(SPEC, args)


def _flatten_result(result: dict) -> dict:
    steps = list(result.get("steps", []))
    step = steps[0] if steps else {}
    linear_timing = step.get("linear_timing", [])
    timing_totals = cumulative_linear_timing(linear_timing)

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
        "total_ksp_its": int(sum_step_linear(step)),
        "asm_time_cumulative": round(timing_totals["asm_time_cumulative"], 4),
        "pc_setup_time_cumulative": round(
            timing_totals["pc_setup_time_cumulative"], 4
        ),
        "linear_solve_time_cumulative": round(
            timing_totals["linear_solve_time_cumulative"], 4
        ),
        "ksp_time_cumulative": round(timing_totals["ksp_time_cumulative"], 4),
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
    ksp_type="cg",
    ksp_max_it=10000,
    tolf=1e-5,
    tolg=1e-3,
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
    state_out="",
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
        state_out=str(state_out),
    )
    return _flatten_result(run(ns))

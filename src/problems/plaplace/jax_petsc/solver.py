"""pLaplace 2D solver logic for JAX + PETSc backends."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.core.benchmark.results import (
    cumulative_linear_timing,
    sum_step_linear,
)
from src.core.petsc.scalar_problem_driver import (
    ScalarProblemDriverSpec,
    run_scalar_problem,
)
from src.problems.plaplace.jax_petsc.parallel_hessian_dof import (
    LocalColoringAssembler,
    ParallelDOFHessianAssembler,
)
from src.problems.plaplace.jax_petsc.reordered_element_assembler import (
    PLaplaceReorderedElementAssembler,
)
from src.problems.plaplace.support.mesh import MeshpLaplace2D


PROFILE_DEFAULTS = {
    "reference": {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-3,
        "ksp_max_it": 10000,
        "pc_setup_on_ksp_cap": False,
        "gamg_threshold": 0.05,
        "gamg_agg_nsmooths": 1,
        "gamg_set_coordinates": True,
        "reorder": True,
    },
    "performance": {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-1,
        "ksp_max_it": 30,
        "pc_setup_on_ksp_cap": False,
        "gamg_threshold": 0.05,
        "gamg_agg_nsmooths": 1,
        "gamg_set_coordinates": True,
        "reorder": True,
    },
}


def _mesh_loader(level: int):
    mesh_obj = MeshpLaplace2D(level)
    return mesh_obj.get_data_jax()


def _flatten_result(result: dict) -> dict:
    steps = list(result.get("steps", []))
    step = steps[0] if steps else {}
    linear_timing = step.get("linear_timing", [])
    timing_totals = cumulative_linear_timing(linear_timing)

    return {
        "mesh_level": int(result["mesh_level"]),
        "dofs": int(result["total_dofs"]),
        "nprocs": int(result["metadata"]["nprocs"]),
        "nproc_threads": int(result["metadata"]["nproc_threads"]),
        "pc_type": str(result["metadata"]["linear_solver"]["pc_type"]),
        "ksp_rtol": float(result["metadata"]["linear_solver"]["ksp_rtol"]),
        "assembly_mode": str(result["metadata"]["linear_solver"]["assembly_mode"]),
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
        "history": list(step.get("history", [])),
        "assembly_details": list(linear_timing),
    }


SPEC = ScalarProblemDriverSpec(
    problem_name="pLaplace2D",
    mesh_loader=_mesh_loader,
    assembler_factories={
        "element": PLaplaceReorderedElementAssembler,
        "local_coloring": LocalColoringAssembler,
        "parallel_dof": ParallelDOFHessianAssembler,
    },
    default_profile_defaults=PROFILE_DEFAULTS,
    line_search_defaults={"linesearch_interval": (-0.5, 2.0), "linesearch_tol": 1e-3},
    trust_region_defaults={
        "trust_radius_init": 1.0,
        "trust_radius_min": 1e-8,
        "trust_radius_max": 1e6,
        "trust_shrink": 0.5,
        "trust_expand": 1.5,
        "trust_eta_shrink": 0.05,
        "trust_eta_expand": 0.75,
        "trust_max_reject": 6,
        "trust_subproblem_line_search": False,
    },
    repair_policy=None,
    result_formatter=_flatten_result,
)


def run(args):
    return run_scalar_problem(SPEC, args)


def run_level(
    mesh_level,
    comm,
    verbose=True,
    coloring_trials=10,
    ksp_rtol=1e-3,
    pc_type="hypre",
    tolf=1e-5,
    tolg=1e-3,
    local_coloring=False,
    assembly_mode="sfd",
    nproc_threads=1,
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
    profile="reference",
    pc_setup_on_ksp_cap=False,
    gamg_threshold=0.05,
    gamg_agg_nsmooths=1,
    gamg_set_coordinates=True,
    tolg_rel=1e-3,
    tolx_rel=1e-3,
    tolx_abs=1e-10,
    save_history=True,
    save_linear_timing=True,
    quiet=False,
    retry_on_failure=False,
    local_hessian_mode="element",
    element_reorder_mode="block_xyz",
    hvp_eval_mode="sequential",
    step_time_limit_s=None,
    trust_subproblem_line_search=False,
):
    del comm
    ns = SimpleNamespace(
        level=int(mesh_level),
        profile=str(profile),
        ksp_type=str(ksp_type),
        pc_type=str(pc_type),
        ksp_rtol=float(ksp_rtol),
        ksp_max_it=int(ksp_max_it),
        pc_setup_on_ksp_cap=bool(pc_setup_on_ksp_cap),
        gamg_threshold=float(gamg_threshold),
        gamg_agg_nsmooths=int(gamg_agg_nsmooths),
        gamg_set_coordinates=bool(gamg_set_coordinates),
        reorder=True,
        local_coloring=bool(local_coloring),
        hvp_eval_mode=str(hvp_eval_mode),
        coloring_trials=int(coloring_trials),
        assembly_mode=str(assembly_mode),
        element_reorder_mode=str(element_reorder_mode),
        local_hessian_mode=str(local_hessian_mode),
        tolf=float(tolf),
        tolg=float(tolg),
        tolg_rel=float(tolg_rel),
        tolx_rel=float(tolx_rel),
        tolx_abs=float(tolx_abs),
        maxit=int(maxit),
        linesearch_a=float(linesearch_interval[0]),
        linesearch_b=float(linesearch_interval[1]),
        linesearch_tol=float(linesearch_tol),
        retry_on_failure=bool(retry_on_failure),
        nproc=int(nproc_threads),
        save_history=bool(save_history),
        save_linear_timing=bool(save_linear_timing),
        quiet=bool(quiet or (not verbose)),
        out="",
        use_trust_region=bool(use_trust_region),
        trust_radius_init=float(trust_radius_init),
        trust_radius_min=float(trust_radius_min),
        trust_radius_max=float(trust_radius_max),
        trust_shrink=float(trust_shrink),
        trust_expand=float(trust_expand),
        trust_eta_shrink=float(trust_eta_shrink),
        trust_eta_expand=float(trust_eta_expand),
        trust_max_reject=int(trust_max_reject),
        trust_subproblem_line_search=bool(trust_subproblem_line_search),
        step_time_limit_s=step_time_limit_s,
    )
    return _flatten_result(run(ns))

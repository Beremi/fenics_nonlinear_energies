"""JAX + PETSc certified stationary solve for the arctan-resonance problem."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from mpi4py import MPI
from petsc4py import PETSc

from src.core.benchmark.state_export import export_scalar_mesh_state_npz
from src.core.petsc.reasons import ksp_reason_name
from src.problems.plaplace_up_arctan.functionals import compute_state_stats_free
from src.problems.plaplace_up_arctan.jax_petsc.multigrid import (
    ScalarStructuredMGHierarchy,
    build_structured_scalar_pmg_hierarchy,
    configure_scalar_pmg,
    refresh_scalar_pmg_operators,
)
from src.problems.plaplace_up_arctan.jax_petsc.reordered_element_assembler import (
    PLaplaceUPArctanReorderedElementAssembler,
)
from src.problems.plaplace_up_arctan.solver_common import ProblemInstance
from src.problems.plaplace_up_arctan.transfer import prolong_free_to_problem


CERTIFIED_DIRECTION_MODEL_PETSC = "jax_petsc_stationary_newton_pmg"


@dataclass
class PetscMeritState:
    energy: float
    gradient_residual_norm: float
    dual_residual_norm: float
    merit: float
    merit_ksp_its: int
    merit_reason: str
    merit_solve_time: float
    gradient: PETSc.Vec


def _string_value(value: object, default: str = "") -> str:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.ndim == 0:
        return str(arr.item())
    return default


def _build_reordered_petsc_matrix(
    matrix_orig: sp.spmatrix,
    *,
    iperm: np.ndarray,
    lo: int,
    hi: int,
    n_free: int,
    comm: MPI.Comm,
) -> PETSc.Mat:
    matrix_coo = matrix_orig.tocoo()
    rows = np.asarray(iperm[np.asarray(matrix_coo.row, dtype=np.int64)], dtype=np.int64)
    cols = np.asarray(iperm[np.asarray(matrix_coo.col, dtype=np.int64)], dtype=np.int64)
    vals = np.asarray(matrix_coo.data, dtype=np.float64)
    owned_mask = (rows >= int(lo)) & (rows < int(hi))
    mat = PETSc.Mat().create(comm=comm)
    mat.setType(PETSc.Mat.Type.MPIAIJ)
    mat.setSizes(((int(hi) - int(lo), int(n_free)), (int(hi) - int(lo), int(n_free))))
    mat.setPreallocationCOO(
        rows[owned_mask].astype(PETSc.IntType),
        cols[owned_mask].astype(PETSc.IntType),
    )
    mat.setBlockSize(1)
    mat.setValuesCOO(vals[owned_mask].astype(PETSc.ScalarType), addv=PETSc.InsertMode.INSERT_VALUES)
    mat.assemble()
    return mat


def _allgather_reordered(assembler: PLaplaceUPArctanReorderedElementAssembler, vec: PETSc.Vec) -> np.ndarray:
    full_reordered, _ = assembler._allgather_full_owned(np.asarray(vec.array[:], dtype=np.float64))
    return np.asarray(full_reordered, dtype=np.float64)


def reordered_to_original_free(
    assembler: PLaplaceUPArctanReorderedElementAssembler,
    full_reordered: np.ndarray,
) -> np.ndarray:
    return np.asarray(full_reordered[np.asarray(assembler.part.iperm, dtype=np.int64)], dtype=np.float64)


def original_free_to_reordered(
    assembler: PLaplaceUPArctanReorderedElementAssembler,
    u_free_orig: np.ndarray,
) -> np.ndarray:
    return np.asarray(np.asarray(u_free_orig, dtype=np.float64)[np.asarray(assembler.part.perm, dtype=np.int64)], dtype=np.float64)


def _load_init_free_from_state(problem: ProblemInstance, state_path: str) -> np.ndarray:
    data = np.load(Path(state_path))
    source_level = int(np.asarray(data["mesh_level"]).item())
    source_full = np.asarray(data["u"], dtype=np.float64)
    if source_level == int(problem.mesh_level):
        return np.asarray(source_full[np.asarray(problem.params["freedofs"], dtype=np.int64)], dtype=np.float64)

    from src.problems.plaplace_up_arctan.solver_common import build_problem

    source_problem = build_problem(
        level=int(source_level),
        p=float(problem.p),
        geometry=str(problem.geometry),
        init_mode="sine",
        lambda1=float(problem.lambda1),
        lambda_level=int(problem.lambda_level),
        seed=0,
    )
    source_free = np.asarray(source_full[np.asarray(source_problem.params["freedofs"], dtype=np.int64)], dtype=np.float64)
    return prolong_free_to_problem(source_problem.params, source_free, problem.params)


def _export_state(
    problem: ProblemInstance,
    u_free: np.ndarray,
    *,
    energy: float,
    path: str,
    metadata: dict[str, object],
) -> str | None:
    if not path:
        return None
    u_full = problem.expand_free(np.asarray(u_free, dtype=np.float64))
    export_scalar_mesh_state_npz(
        path,
        coords=np.asarray(problem.plot_coords, dtype=np.float64),
        triangles=np.asarray(problem.plot_cells, dtype=np.int32),
        u=u_full,
        mesh_level=int(problem.mesh_level),
        problem_name="pLaplaceUPArctan",
        energy=float(energy),
        metadata=metadata,
    )
    return str(Path(path))


def _create_ksp(
    comm: MPI.Comm,
    *,
    ksp_type: str,
    pc_type: str,
    ksp_rtol: float,
    ksp_max_it: int,
) -> PETSc.KSP:
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setType(str(ksp_type))
    ksp.getPC().setType(str(pc_type))
    ksp.setTolerances(rtol=float(ksp_rtol), max_it=int(ksp_max_it))
    ksp.setFromOptions()
    return ksp


def _maybe_build_hierarchy(
    *,
    problem: ProblemInstance,
    comm: MPI.Comm,
    pc_type: str,
    coarsest_level: int,
    reorder_mode: str,
) -> ScalarStructuredMGHierarchy | None:
    if str(pc_type) != "mg":
        return None
    finest_level = int(problem.mesh_level)
    if int(coarsest_level) >= finest_level:
        return None
    return build_structured_scalar_pmg_hierarchy(
        finest_level=finest_level,
        coarsest_level=int(coarsest_level),
        p=float(problem.p),
        geometry=str(problem.geometry),
        comm=comm,
        reorder_mode=str(reorder_mode),
    )


def _configure_mg(
    ksp: PETSc.KSP,
    hierarchy: ScalarStructuredMGHierarchy | None,
    *,
    coarse_ksp_type: str,
    coarse_pc_type: str,
    smoother_ksp_type: str,
    smoother_pc_type: str,
    smoother_steps: int,
) -> dict[str, object] | None:
    if hierarchy is None:
        return None
    return configure_scalar_pmg(
        ksp,
        hierarchy,
        coarse_ksp_type=str(coarse_ksp_type),
        coarse_pc_type=str(coarse_pc_type),
        smoother_ksp_type=str(smoother_ksp_type),
        smoother_pc_type=str(smoother_pc_type),
        smoother_steps=int(smoother_steps),
    )


def _refresh_mg(
    ksp: PETSc.KSP,
    hierarchy: ScalarStructuredMGHierarchy | None,
    cache: dict[str, object] | None,
    *,
    fine_operator: PETSc.Mat,
) -> float:
    if hierarchy is None or cache is None:
        return 0.0
    t0 = time.perf_counter()
    refresh_scalar_pmg_operators(ksp, hierarchy, fine_operator=fine_operator, cache=cache)
    return float(time.perf_counter() - t0)


def _compute_merit_state(
    assembler: PLaplaceUPArctanReorderedElementAssembler,
    merit_ksp: PETSc.KSP,
    x: PETSc.Vec,
    gradient: PETSc.Vec,
    aux: PETSc.Vec,
) -> PetscMeritState:
    energy = float(assembler.energy_fn(x))
    assembler.gradient_fn(x, gradient)
    grad_norm = float(gradient.norm(PETSc.NormType.NORM_2))
    if grad_norm <= 0.0 or gradient.getSize() == 0:
        return PetscMeritState(
            energy=energy,
            gradient_residual_norm=grad_norm,
            dual_residual_norm=grad_norm,
            merit=0.5 * grad_norm * grad_norm,
            merit_ksp_its=0,
            merit_reason="ZERO_GRADIENT",
            merit_solve_time=0.0,
            gradient=gradient,
        )
    t0 = time.perf_counter()
    merit_ksp.solve(gradient, aux)
    merit_solve_time = float(time.perf_counter() - t0)
    dual_sq = float(gradient.dot(aux))
    if not np.isfinite(dual_sq):
        dual_sq = grad_norm * grad_norm
    dual_norm = float(math.sqrt(max(dual_sq, 0.0)))
    return PetscMeritState(
        energy=energy,
        gradient_residual_norm=grad_norm,
        dual_residual_norm=dual_norm,
        merit=0.5 * max(dual_sq, 0.0),
        merit_ksp_its=int(merit_ksp.getIterationNumber()),
        merit_reason=str(ksp_reason_name(int(merit_ksp.getConvergedReason()))),
        merit_solve_time=merit_solve_time,
        gradient=gradient,
    )


def solve_certified_stationary_petsc(
    problem: ProblemInstance,
    *,
    init_free: np.ndarray,
    epsilon: float,
    maxit: int,
    state_out: str = "",
    handoff_source: str = "init_state",
    ksp_type: str = "fgmres",
    ksp_rtol: float = 1.0e-8,
    ksp_max_it: int = 400,
    merit_ksp_type: str = "cg",
    merit_ksp_rtol: float = 1.0e-10,
    merit_ksp_max_it: int = 400,
    pc_type: str = "mg",
    reorder_mode: str = "block_xyz",
    local_hessian_mode: str = "element",
    distribution_strategy: str = "overlap_p2p",
    mg_coarsest_level: int = 2,
    mg_smoother_ksp_type: str = "richardson",
    mg_smoother_pc_type: str = "sor",
    mg_smoother_steps: int = 2,
    mg_coarse_ksp_type: str = "cg",
    mg_coarse_pc_type: str = "hypre",
    initial_mu: float = 1.0e-8,
) -> dict[str, object]:
    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    assembler = None
    hierarchy = None
    step_ksp = None
    step_mg_cache = None
    stiffness_mat = None
    merit_ksp = None
    merit_mg_cache = None
    x = None
    gradient = None
    aux = None
    rhs = None
    step = None
    candidate = None

    total_start = time.perf_counter()
    setup_start = time.perf_counter()

    assembler = PLaplaceUPArctanReorderedElementAssembler(
        params=problem.params,
        comm=comm,
        adjacency=problem.adjacency,
        ksp_rtol=float(ksp_rtol),
        ksp_type=str(ksp_type),
        pc_type=str(pc_type),
        ksp_max_it=int(ksp_max_it),
        reorder_mode=str(reorder_mode),
        local_hessian_mode=str(local_hessian_mode),
        distribution_strategy=str(distribution_strategy),
    )
    hierarchy = _maybe_build_hierarchy(
        problem=problem,
        comm=comm,
        pc_type=str(pc_type),
        coarsest_level=int(mg_coarsest_level),
        reorder_mode=str(reorder_mode),
    )
    step_ksp = assembler.ksp
    step_mg_cache = _configure_mg(
        step_ksp,
        hierarchy,
        coarse_ksp_type=str(mg_coarse_ksp_type),
        coarse_pc_type=str(mg_coarse_pc_type),
        smoother_ksp_type=str(mg_smoother_ksp_type),
        smoother_pc_type=str(mg_smoother_pc_type),
        smoother_steps=int(mg_smoother_steps),
    )

    stiffness_mat = _build_reordered_petsc_matrix(
        problem.stiffness,
        iperm=np.asarray(assembler.part.iperm, dtype=np.int64),
        lo=int(assembler.part.lo),
        hi=int(assembler.part.hi),
        n_free=int(assembler.part.n_free),
        comm=comm,
    )
    merit_ksp = _create_ksp(
        comm,
        ksp_type=str(merit_ksp_type),
        pc_type=str(pc_type),
        ksp_rtol=float(merit_ksp_rtol),
        ksp_max_it=int(merit_ksp_max_it),
    )
    merit_mg_cache = _configure_mg(
        merit_ksp,
        hierarchy,
        coarse_ksp_type=str(mg_coarse_ksp_type),
        coarse_pc_type=str(mg_coarse_pc_type),
        smoother_ksp_type=str(mg_smoother_ksp_type),
        smoother_pc_type=str(mg_smoother_pc_type),
        smoother_steps=int(mg_smoother_steps),
    )
    if str(pc_type) == "mg":
        step_ksp.setOperators(stiffness_mat, stiffness_mat)
        _refresh_mg(step_ksp, hierarchy, step_mg_cache, fine_operator=stiffness_mat)
    else:
        step_ksp.setOperators(stiffness_mat)
    merit_ksp.setOperators(stiffness_mat, stiffness_mat)
    merit_refresh_time = _refresh_mg(merit_ksp, hierarchy, merit_mg_cache, fine_operator=stiffness_mat)
    setup_time = float(time.perf_counter() - setup_start)

    x = assembler.create_vec(original_free_to_reordered(assembler, np.asarray(init_free, dtype=np.float64)))
    gradient = x.duplicate()
    aux = x.duplicate()
    rhs = x.duplicate()
    step = x.duplicate()
    candidate = x.duplicate()
    best_reordered = _allgather_reordered(assembler, x)
    best_merit_state = _compute_merit_state(assembler, merit_ksp, x, gradient, aux)
    best_residual_outer_it = 0
    history: list[dict[str, object]] = []
    step_records: list[dict[str, object]] = []
    shift_mats: list[PETSc.Mat] = []
    direction_solves = 0
    status = "maxit"
    message = "Maximum number of certification iterations reached"
    mu = max(float(initial_mu), 1.0e-12)
    solve_start = time.perf_counter()

    try:
        for outer_it in range(1, int(maxit) + 1):
            current_state = _compute_merit_state(assembler, merit_ksp, x, gradient, aux)
            if current_state.dual_residual_norm < best_merit_state.dual_residual_norm:
                best_merit_state = current_state
                best_reordered = _allgather_reordered(assembler, x)
                best_residual_outer_it = int(outer_it)
            if current_state.dual_residual_norm <= float(epsilon):
                status = "completed"
                message = "Stationarity merit converged"
                break

            assemble_start = time.perf_counter()
            assembler.assemble_hessian(np.asarray(x.array[:], dtype=np.float64))
            hessian_assemble_time = float(time.perf_counter() - assemble_start)
            last_assembly = dict(assembler.iter_timings[-1]) if assembler.iter_timings else {}

            accepted = False
            chosen_alpha = 0.0
            chosen_mu = mu
            chosen_step_norm = 0.0
            chosen_halves = 0
            chosen_reg_attempt = 0
            chosen_trial_state = current_state
            chosen_step_reason = "NOT_RUN"
            chosen_step_its = 0
            chosen_step_solve_time = 0.0
            chosen_shift_build_time = 0.0
            chosen_mg_refresh_time = 0.0
            chosen_candidate_energy = current_state.energy

            attempt_mu = max(float(mu), 1.0e-12)
            for reg_attempt in range(1, 9):
                chosen_reg_attempt = int(reg_attempt)
                shift_start = time.perf_counter()
                current_shift = assembler.A.copy()
                current_shift.axpy(float(attempt_mu), stiffness_mat, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
                current_shift.assemble()
                chosen_shift_build_time = float(time.perf_counter() - shift_start)
                shift_mats.append(current_shift)

                if str(pc_type) == "mg":
                    chosen_mg_refresh_time = 0.0
                    step_ksp.setOperators(current_shift, stiffness_mat)
                else:
                    chosen_mg_refresh_time = _refresh_mg(step_ksp, hierarchy, step_mg_cache, fine_operator=current_shift)
                    step_ksp.setOperators(current_shift)

                rhs.array[:] = gradient.array
                rhs.scale(-1.0)
                solve_start_iter = time.perf_counter()
                step_ksp.solve(rhs, step)
                chosen_step_solve_time = float(time.perf_counter() - solve_start_iter)
                chosen_step_its = int(step_ksp.getIterationNumber())
                chosen_step_reason = str(ksp_reason_name(int(step_ksp.getConvergedReason())))
                direction_solves += 1

                step_norm = float(step.norm(PETSc.NormType.NORM_2))
                if int(step_ksp.getConvergedReason()) < 0 or not np.isfinite(step_norm):
                    attempt_mu *= 10.0
                    continue

                alpha = 1.0
                for halves in range(0, 21):
                    candidate.waxpy(float(alpha), step, x)
                    candidate_energy = float(assembler.energy_fn(candidate))
                    if not np.isfinite(candidate_energy):
                        alpha *= 0.5
                        continue
                    trial_state = _compute_merit_state(assembler, merit_ksp, candidate, gradient, aux)
                    if np.isfinite(trial_state.merit) and trial_state.merit < current_state.merit:
                        accepted = True
                        chosen_alpha = float(alpha)
                        chosen_mu = float(attempt_mu)
                        chosen_step_norm = float(step_norm)
                        chosen_halves = int(halves)
                        chosen_trial_state = trial_state
                        chosen_candidate_energy = candidate_energy
                        candidate.copy(x)
                        break
                    alpha *= 0.5
                if accepted:
                    break
                attempt_mu *= 10.0

            history.append(
                {
                    "outer_it": int(outer_it),
                    "J": float(current_state.energy),
                    "stop_measure": float(current_state.dual_residual_norm),
                    "stop_name": "stationary_merit",
                    "gradient_residual_norm": float(current_state.gradient_residual_norm),
                    "dual_residual_norm": float(current_state.dual_residual_norm),
                    "merit": float(current_state.merit),
                    "alpha": float(chosen_alpha),
                    "mu": float(chosen_mu),
                    "accepted": bool(accepted),
                    "halves": int(chosen_halves),
                    "regularization_attempts": int(chosen_reg_attempt),
                    "step_norm": float(chosen_step_norm),
                    "trial_dual_residual_norm": float(chosen_trial_state.dual_residual_norm),
                    "direction_model": CERTIFIED_DIRECTION_MODEL_PETSC,
                    "backend": "jax_petsc",
                    "merit_ksp_its": int(current_state.merit_ksp_its),
                    "merit_reason": str(current_state.merit_reason),
                    "merit_solve_time": float(current_state.merit_solve_time),
                    "hessian_assemble_time": float(hessian_assemble_time),
                    "shift_build_time": float(chosen_shift_build_time),
                    "pmg_refresh_time": float(chosen_mg_refresh_time),
                    "linear_solve_time": float(chosen_step_solve_time),
                    "step_ksp_its": int(chosen_step_its),
                    "step_reason": str(chosen_step_reason),
                    "assembly_callback_total": float(last_assembly.get("total", 0.0)),
                }
            )
            step_records.append(
                {
                    "outer_it": int(outer_it),
                    "accepted": bool(accepted),
                    "mu": float(chosen_mu),
                    "alpha": float(chosen_alpha),
                    "ksp_its": int(chosen_step_its),
                    "ksp_reason": str(chosen_step_reason),
                    "assemble_time": float(hessian_assemble_time),
                    "pmg_refresh_time": float(chosen_mg_refresh_time),
                    "shift_build_time": float(chosen_shift_build_time),
                    "solve_time": float(chosen_step_solve_time),
                    "merit_solve_time": float(current_state.merit_solve_time),
                    "candidate_energy": float(chosen_candidate_energy),
                    "residual_norm": float(chosen_trial_state.dual_residual_norm),
                }
            )
            if not accepted:
                near_tolerance_stagnation = (
                    current_state.dual_residual_norm <= 3.0 * float(epsilon)
                    and chosen_trial_state.dual_residual_norm
                    <= current_state.dual_residual_norm * (1.0 + 1.0e-3)
                )
                if near_tolerance_stagnation:
                    status = "completed"
                    message = "Stationarity merit reached near-tolerance stagnation"
                    break
                status = "failed"
                message = "PETSc certification Newton step failed to reduce the stationarity merit"
                break
            if chosen_trial_state.dual_residual_norm < best_merit_state.dual_residual_norm:
                best_merit_state = chosen_trial_state
                best_reordered = _allgather_reordered(assembler, x)
                best_residual_outer_it = int(outer_it)
            mu = max(1.0e-12, chosen_mu * (0.25 if chosen_alpha >= 0.5 else 1.0))
        else:
            current_state = _compute_merit_state(assembler, merit_ksp, x, gradient, aux)
            if current_state.dual_residual_norm < best_merit_state.dual_residual_norm:
                best_merit_state = current_state
                best_reordered = _allgather_reordered(assembler, x)
                best_residual_outer_it = int(maxit)

        if status == "completed":
            reported_reordered = _allgather_reordered(assembler, x)
            reported_iterate_source = "final"
        else:
            reported_reordered = np.asarray(best_reordered, dtype=np.float64)
            reported_iterate_source = "best_dual_residual"

        reported_free = reordered_to_original_free(assembler, reported_reordered)
        state_stats = compute_state_stats_free(problem.params, reported_free)
        solve_time = float(time.perf_counter() - solve_start)
        total_time = float(time.perf_counter() - total_start)
        state_path = None
        if rank == 0:
            state_path = _export_state(
                problem,
                reported_free,
                energy=float(state_stats.J),
                path=str(state_out),
                metadata={
                    "geometry": str(problem.geometry),
                    "p": float(problem.p),
                    "lambda1": float(problem.lambda1),
                    "lambda_level": int(problem.lambda_level),
                    "method": "certified_newton_petsc",
                    "seed_name": "petsc_init",
                    "backend": "jax_petsc",
                    "handoff_source": str(handoff_source),
                },
            )

        comm.Barrier()
        state_path = comm.bcast(state_path, root=0)

        if reported_iterate_source == "final":
            reported_merit_state = _compute_merit_state(assembler, merit_ksp, x, gradient, aux)
        else:
            reported_merit_state = best_merit_state

        return {
            "method": "certified_newton",
            "status": str(status),
            "message": str(message),
            "geometry": str(problem.geometry),
            "level": int(problem.mesh_level),
            "h": float(problem.h),
            "p": float(problem.p),
            "lambda1": float(problem.lambda1),
            "lambda_level": int(problem.lambda_level),
            "epsilon": float(epsilon),
            "J": float(state_stats.J),
            "residual_norm": float(reported_merit_state.dual_residual_norm),
            "gradient_residual_norm": float(reported_merit_state.gradient_residual_norm),
            "outer_iterations": int(len(history)),
            "accepted_step_count": int(sum(1 for item in history if bool(item.get("accepted")))),
            "direction_solves": int(direction_solves),
            "history": history,
            "iterate_free": reported_free.tolist(),
            "state_out": state_path,
            "objective_name": "J",
            "direction_model": CERTIFIED_DIRECTION_MODEL_PETSC,
            "handoff_source": str(handoff_source),
            "certified_newton_iters": int(len(history)),
            "reported_iterate_source": str(reported_iterate_source),
            "best_residual_norm": float(best_merit_state.dual_residual_norm),
            "best_gradient_residual_norm": float(best_merit_state.gradient_residual_norm),
            "best_residual_outer_it": int(best_residual_outer_it),
            "backend": "jax_petsc",
            "linear_solver": {
                "ksp_type": str(ksp_type),
                "ksp_rtol": float(ksp_rtol),
                "ksp_max_it": int(ksp_max_it),
                "pc_type": str(pc_type),
                "merit_ksp_type": str(merit_ksp_type),
                "merit_ksp_rtol": float(merit_ksp_rtol),
                "merit_ksp_max_it": int(merit_ksp_max_it),
                "distribution_strategy": str(distribution_strategy),
                "element_reorder_mode": str(reorder_mode),
                "local_hessian_mode": str(local_hessian_mode),
                "step_preconditioner_operator": "stiffness_galerkin" if str(pc_type) == "mg" else "operator",
            },
            "timings": {
                "setup_time": float(setup_time),
                "merit_setup_refresh_time": float(merit_refresh_time),
                "solve_time": float(solve_time),
                "total_time": float(total_time),
                "assembler_setup_breakdown": assembler.setup_summary(),
                "callback_summary": assembler.callback_summary(),
                "linear_iterations_total": int(sum(int(item["ksp_its"]) for item in step_records)),
                "linear_timing": step_records,
            },
        }
    finally:
        for mat in shift_mats:
            try:
                mat.destroy()
            except Exception:
                pass
        for vec in (gradient, aux, rhs, step, candidate, x):
            if vec is None:
                continue
            try:
                vec.destroy()
            except Exception:
                pass
        if merit_ksp is not None:
            try:
                merit_ksp.destroy()
            except Exception:
                pass
        if stiffness_mat is not None:
            try:
                stiffness_mat.destroy()
            except Exception:
                pass
        if assembler is not None:
            try:
                assembler.cleanup()
            except Exception:
                pass
        if hierarchy is not None:
            try:
                hierarchy.cleanup()
            except Exception:
                pass


def solve_from_state_path(
    problem: ProblemInstance,
    *,
    init_state_path: str,
    init_scale: float = 1.0,
    **kwargs: Any,
) -> dict[str, object]:
    init_free = _load_init_free_from_state(problem, init_state_path)
    init_free = float(init_scale) * np.asarray(init_free, dtype=np.float64)
    return solve_certified_stationary_petsc(
        problem,
        init_free=init_free,
        **kwargs,
    )

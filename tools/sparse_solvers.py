import time
from typing import Literal

import numpy as np
import pyamg
import scipy.sparse.linalg as spla


def _tau_to_boundary(step, direction, radius):
    a = float(np.dot(direction, direction))
    if a <= 0.0:
        return 0.0
    b = 2.0 * float(np.dot(step, direction))
    c = float(np.dot(step, step) - radius * radius)
    disc = max(0.0, b * b - 4.0 * a * c)
    sqrt_disc = np.sqrt(disc)
    tau1 = (-b - sqrt_disc) / (2.0 * a)
    tau2 = (-b + sqrt_disc) / (2.0 * a)
    candidates = [tau for tau in (tau1, tau2) if tau >= 0.0]
    if not candidates:
        return 0.0
    return float(max(candidates))


def _apply_preconditioner(preconditioner, vector):
    if preconditioner is None:
        return np.asarray(vector, dtype=np.float64).copy()
    if hasattr(preconditioner, "matvec"):
        return np.asarray(preconditioner.matvec(vector), dtype=np.float64)
    return np.asarray(preconditioner @ vector, dtype=np.float64)


def _steihaug_toint(H, gradient, radius, preconditioner=None, rtol=1e-1, maxiter=30):
    gradient = np.asarray(gradient, dtype=np.float64)
    radius = float(max(0.0, radius))
    maxiter = max(1, int(maxiter))
    if radius <= 0.0:
        return np.zeros_like(gradient), 0, False, "zero_radius"

    step = np.zeros_like(gradient)
    residual = -gradient.copy()
    residual_norm0 = float(np.linalg.norm(residual))
    if residual_norm0 <= 0.0 or not np.isfinite(residual_norm0):
        return step, 0, False, "zero_rhs"

    z = _apply_preconditioner(preconditioner, residual)
    rz = float(np.dot(residual, z))
    if not np.isfinite(rz) or rz <= 0.0:
        z = residual.copy()
        rz = float(np.dot(residual, z))
    direction = z.copy()
    target = max(float(rtol) * residual_norm0, 1e-14)

    for iteration in range(1, maxiter + 1):
        Hd = np.asarray(H @ direction, dtype=np.float64)
        curvature = float(np.dot(direction, Hd))

        if not np.isfinite(curvature) or curvature <= 1e-14:
            tau = _tau_to_boundary(step, direction, radius)
            return step + tau * direction, iteration, False, "negative_curvature"

        alpha = rz / curvature
        candidate = step + alpha * direction
        if np.linalg.norm(candidate) >= radius:
            tau = _tau_to_boundary(step, direction, radius)
            return step + tau * direction, iteration, False, "boundary"

        step = candidate
        residual = residual - alpha * Hd
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= target:
            return step, iteration, False, "converged"

        z_new = _apply_preconditioner(preconditioner, residual)
        rz_new = float(np.dot(residual, z_new))
        if not np.isfinite(rz_new) or rz_new <= 0.0:
            return step, iteration, False, "preconditioner_breakdown"

        beta = rz_new / rz
        direction = z_new + beta * direction
        z = z_new
        rz = rz_new

    return step, maxiter, True, "maxit"


class SolverAMGElasticity:
    def __init__(self, H, elastic_kernel=None, verbose=False, tol=1e-3, maxiter=1000, assemble_time=0.0):
        self.H = H
        self.verbose = verbose
        self.tol = tol
        self.maxiter = maxiter
        self.U = elastic_kernel
        self.assemble_time = float(assemble_time)
        t0 = time.perf_counter()
        ml = pyamg.smoothed_aggregation_solver(self.H.tocsr(), B=self.U, smooth='energy')
        self.M_lin = ml.aspreconditioner()
        self.pc_setup_time = time.perf_counter() - t0
        self.last_solve_info = {}

    def solve(self, x):
        iteration_count = [0]

        def callback(xk):
            iteration_count[0] += 1

        t0 = time.perf_counter()
        sol, info = spla.cg(
            self.H,
            x.copy(),
            rtol=self.tol,
            M=self.M_lin,
            callback=callback,
            maxiter=self.maxiter,
        )
        solve_time = time.perf_counter() - t0
        hit_maxit = info > 0
        if info < 0:
            reason = "breakdown"
        elif hit_maxit:
            reason = "maxit"
        else:
            reason = "converged"
        self.last_solve_info = {
            "ksp_its": int(iteration_count[0]),
            "solve_time": float(solve_time),
            "hit_maxit": bool(hit_maxit),
            "reason": reason,
        }
        if self.verbose:
            print(f"Iterations in AMG solver: {iteration_count[0]}.")

        return sol

    def trust_subproblem_solve(self, gradient, radius):
        t0 = time.perf_counter()
        step, iterations, hit_maxit, reason = _steihaug_toint(
            self.H,
            gradient,
            radius,
            preconditioner=self.M_lin,
            rtol=self.tol,
            maxiter=self.maxiter,
        )
        solve_time = time.perf_counter() - t0
        self.last_solve_info = {
            "ksp_its": int(iterations),
            "solve_time": float(solve_time),
            "hit_maxit": bool(hit_maxit),
            "reason": reason,
        }
        return step


class HessSolveSparse:
    def __init__(self, H, assemble_time=0.0):
        self.H = H
        self.assemble_time = float(assemble_time)
        self.pc_setup_time = 0.0
        self.last_solve_info = {}

    def solve(self, x):
        t0 = time.perf_counter()
        x = spla.spsolve(self.H, x.copy())
        solve_time = time.perf_counter() - t0
        self.last_solve_info = {
            "ksp_its": 1,
            "solve_time": float(solve_time),
            "hit_maxit": False,
            "reason": "direct",
        }
        return x

    def trust_subproblem_solve(self, gradient, radius):
        t0 = time.perf_counter()
        step, iterations, hit_maxit, reason = _steihaug_toint(
            self.H,
            gradient,
            radius,
            preconditioner=None,
            rtol=self.tol if hasattr(self, "tol") else 1e-10,
            maxiter=self.maxiter if hasattr(self, "maxiter") else max(1, self.H.shape[0]),
        )
        solve_time = time.perf_counter() - t0
        self.last_solve_info = {
            "ksp_its": int(iterations),
            "solve_time": float(solve_time),
            "hit_maxit": bool(hit_maxit),
            "reason": reason,
        }
        return step


class HessSolverGenerator:
    def __init__(self, ddf, solver_type: Literal["direct", "amg"] = "direct",
                 elastic_kernel=None, verbose=False, tol=1e-3, maxiter=100):
        self.ddf = ddf
        self.verbose = verbose
        self.tol = tol
        self.maxiter = maxiter
        self.elastic_kernel = elastic_kernel
        self.solver_type = solver_type

    def __call__(self, x):
        assemble_start = time.perf_counter()
        sparse_matrix_scipy = self.ddf(x)
        assemble_time = time.perf_counter() - assemble_start
        if self.solver_type == "direct":
            solver = HessSolveSparse(sparse_matrix_scipy, assemble_time=assemble_time)
            solver.tol = self.tol
            solver.maxiter = self.maxiter
            return solver
        if self.solver_type == "amg":
            return SolverAMGElasticity(
                sparse_matrix_scipy,
                elastic_kernel=self.elastic_kernel,
                verbose=self.verbose,
                tol=self.tol,
                maxiter=self.maxiter,
                assemble_time=assemble_time,
            )
        raise ValueError(f"Unknown type {self.solver_type} for the solver.")

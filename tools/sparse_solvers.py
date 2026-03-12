import time
from typing import Literal

import numpy as np
import pyamg
import scipy.sparse.linalg as spla

try:
    from petsc4py import PETSc
except ImportError:  # pragma: no cover - optional dependency
    PETSc = None

try:
    from tools_petsc4py.reasons import ksp_reason_name
except ImportError:  # pragma: no cover - optional dependency
    def ksp_reason_name(reason_code):
        return f"UNKNOWN_{reason_code}"


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


class SolverPETScGAMGElasticity:
    def __init__(
        self,
        H,
        *,
        preconditioner_matrix=None,
        elastic_kernel=None,
        coordinates=None,
        block_size=1,
        verbose=False,
        tol=1e-3,
        maxiter=1000,
        assemble_time=0.0,
        ksp_type="cg",
        pc_type="gamg",
        gamg_threshold=0.05,
        gamg_agg_nsmooths=1,
        gamg_repartition=False,
        gamg_reuse_interpolation=False,
        gamg_aggressive_coarsening=0,
        gamg_set_coordinates=False,
        use_near_nullspace=True,
        norm_type="default",
        mat_symmetric=False,
        pmat_symmetric=False,
        petsc_options=None,
    ):
        if PETSc is None:
            raise ImportError("petsc4py is required for SolverPETScGAMGElasticity.")

        self.H = H.tocsr()
        self.P = self.H if preconditioner_matrix is None else preconditioner_matrix.tocsr()
        self.verbose = verbose
        self.tol = tol
        self.maxiter = maxiter
        self.elastic_kernel = elastic_kernel
        self.coordinates = coordinates
        self.block_size = int(block_size)
        self.assemble_time = float(assemble_time)
        self.pc_setup_time = 0.0
        self.last_solve_info = {}
        self._mat = None
        self._pmat = None
        self._ksp = None
        self._nullspace = None
        self.norm_type = str(norm_type)
        self.pc_type = str(pc_type)
        self.gamg_repartition = bool(gamg_repartition)
        self.gamg_reuse_interpolation = bool(gamg_reuse_interpolation)
        self.gamg_aggressive_coarsening = int(gamg_aggressive_coarsening)
        self.mat_symmetric = bool(mat_symmetric)
        self.pmat_symmetric = bool(pmat_symmetric)
        self.petsc_options = {} if petsc_options is None else dict(petsc_options)

        setup_start = time.perf_counter()
        self._mat = self._create_petsc_mat(self.H, self.block_size)
        self._pmat = self._create_petsc_mat(self.P, self.block_size)
        if self.mat_symmetric:
            self._mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            self._mat.setOption(PETSc.Mat.Option.STRUCTURALLY_SYMMETRIC, True)
        if self.pmat_symmetric:
            self._pmat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            self._pmat.setOption(PETSc.Mat.Option.STRUCTURALLY_SYMMETRIC, True)
        if bool(use_near_nullspace) and elastic_kernel is not None:
            self._nullspace = self._create_near_nullspace(np.asarray(elastic_kernel, dtype=np.float64))
            self._pmat.setNearNullSpace(self._nullspace)

        self._ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        self._ksp.setType(str(ksp_type))
        if self.norm_type == "unpreconditioned":
            self._ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        elif self.norm_type != "default":
            raise ValueError(f"Unknown PETSc norm_type {self.norm_type!r}.")
        pc = self._ksp.getPC()
        pc.setType(self.pc_type)
        if self.pc_type == "gamg" and bool(gamg_set_coordinates) and coordinates is not None:
            pc.setCoordinates(np.asarray(coordinates, dtype=np.float64))

        prefix = f"topopt_petsc_{id(self)}_"
        self._ksp.setOptionsPrefix(prefix)
        opts = PETSc.Options()
        if self.pc_type == "gamg":
            opts[f"{prefix}pc_gamg_threshold"] = float(gamg_threshold)
            opts[f"{prefix}pc_gamg_agg_nsmooths"] = int(gamg_agg_nsmooths)
            opts[f"{prefix}pc_gamg_repartition"] = str(bool(self.gamg_repartition)).lower()
            opts[f"{prefix}pc_gamg_reuse_interpolation"] = str(bool(self.gamg_reuse_interpolation)).lower()
            opts[f"{prefix}pc_gamg_aggressive_coarsening"] = int(self.gamg_aggressive_coarsening)
        for key, value in self.petsc_options.items():
            opts[f"{prefix}{key}"] = value
        self._ksp.setTolerances(rtol=float(tol), max_it=int(maxiter))
        self._ksp.setOperators(self._mat, self._pmat)
        self._ksp.setFromOptions()
        self.pc_setup_time = time.perf_counter() - setup_start

    @staticmethod
    def _create_petsc_mat(H, block_size):
        H = H.tocsr()
        indptr = np.asarray(H.indptr, dtype=PETSc.IntType)
        indices = np.asarray(H.indices, dtype=PETSc.IntType)
        data = np.asarray(H.data, dtype=np.float64)
        A = PETSc.Mat().createAIJ(size=H.shape, csr=(indptr, indices, data), comm=PETSc.COMM_SELF)
        if int(block_size) > 1:
            A.setBlockSize(int(block_size))
        A.assemble()
        return A

    @staticmethod
    def _create_near_nullspace(kernel):
        vecs = []
        n = int(kernel.shape[0])
        for i in range(kernel.shape[1]):
            v = PETSc.Vec().createSeq(n)
            v.array[:] = kernel[:, i]
            v.assemble()
            vecs.append(v)
        return PETSc.NullSpace().create(vectors=vecs, comm=PETSc.COMM_SELF)

    def solve(self, x):
        rhs = PETSc.Vec().createSeq(self.H.shape[0])
        rhs.array[:] = np.asarray(x, dtype=np.float64)
        rhs.assemble()

        sol = rhs.duplicate()
        solve_start = time.perf_counter()
        self._ksp.solve(rhs, sol)
        solve_time = time.perf_counter() - solve_start

        its = int(self._ksp.getIterationNumber())
        reason_code = int(self._ksp.getConvergedReason())
        hit_maxit = reason_code <= 0
        reason = ksp_reason_name(reason_code)
        self.last_solve_info = {
            "ksp_its": its,
            "solve_time": float(solve_time),
            "hit_maxit": bool(hit_maxit),
            "reason": reason,
            "residual_norm": float(self._ksp.getResidualNorm()),
        }
        if self.verbose:
            print(f"PETSc GAMG iterations: {its} ({reason}).")
        return np.asarray(sol.array, dtype=np.float64)

    def _apply_pc(self, x):
        rhs = PETSc.Vec().createSeq(self.H.shape[0])
        rhs.array[:] = np.asarray(x, dtype=np.float64)
        rhs.assemble()
        out = rhs.duplicate()
        self._ksp.setUp()
        self._ksp.getPC().apply(rhs, out)
        return np.asarray(out.array, dtype=np.float64).copy()

    def trust_subproblem_solve(self, gradient, radius):
        class _PETScPCAdapter:
            def __init__(self, parent):
                self.parent = parent

            def matvec(self, vector):
                return self.parent._apply_pc(vector)

        t0 = time.perf_counter()
        step, iterations, hit_maxit, reason = _steihaug_toint(
            self.H,
            gradient,
            radius,
            preconditioner=_PETScPCAdapter(self),
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


class HessSolverGenerator:
    def __init__(self, ddf, solver_type: Literal["direct", "amg", "petsc_gamg"] = "direct",
                 elastic_kernel=None, verbose=False, tol=1e-3, maxiter=100,
                 coordinates=None, block_size=1, ksp_type="cg",
                 gamg_threshold=0.05, gamg_agg_nsmooths=1,
                 gamg_repartition=False, gamg_reuse_interpolation=False,
                 gamg_aggressive_coarsening=0,
                 gamg_set_coordinates=False, use_near_nullspace=True,
                 norm_type="default", preconditioner_matrix=None,
                 mat_symmetric=False, pmat_symmetric=False, pc_type="gamg",
                 petsc_options=None):
        self.ddf = ddf
        self.verbose = verbose
        self.tol = tol
        self.maxiter = maxiter
        self.elastic_kernel = elastic_kernel
        self.solver_type = solver_type
        self.coordinates = coordinates
        self.block_size = int(block_size)
        self.ksp_type = str(ksp_type)
        self.gamg_threshold = float(gamg_threshold)
        self.gamg_agg_nsmooths = int(gamg_agg_nsmooths)
        self.gamg_repartition = bool(gamg_repartition)
        self.gamg_reuse_interpolation = bool(gamg_reuse_interpolation)
        self.gamg_aggressive_coarsening = int(gamg_aggressive_coarsening)
        self.gamg_set_coordinates = bool(gamg_set_coordinates)
        self.use_near_nullspace = bool(use_near_nullspace)
        self.norm_type = str(norm_type)
        self.preconditioner_matrix = preconditioner_matrix
        self.mat_symmetric = bool(mat_symmetric)
        self.pmat_symmetric = bool(pmat_symmetric)
        self.pc_type = str(pc_type)
        self.petsc_options = {} if petsc_options is None else dict(petsc_options)

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
        if self.solver_type == "petsc_gamg":
            return SolverPETScGAMGElasticity(
                sparse_matrix_scipy,
                preconditioner_matrix=self.preconditioner_matrix,
                elastic_kernel=self.elastic_kernel,
                coordinates=self.coordinates,
                block_size=self.block_size,
                verbose=self.verbose,
                tol=self.tol,
                maxiter=self.maxiter,
                assemble_time=assemble_time,
                ksp_type=self.ksp_type,
                pc_type=self.pc_type,
                gamg_threshold=self.gamg_threshold,
                gamg_agg_nsmooths=self.gamg_agg_nsmooths,
                gamg_repartition=self.gamg_repartition,
                gamg_reuse_interpolation=self.gamg_reuse_interpolation,
                gamg_aggressive_coarsening=self.gamg_aggressive_coarsening,
                gamg_set_coordinates=self.gamg_set_coordinates,
                use_near_nullspace=self.use_near_nullspace,
                norm_type=self.norm_type,
                mat_symmetric=self.mat_symmetric,
                pmat_symmetric=self.pmat_symmetric,
                petsc_options=self.petsc_options,
            )
        raise ValueError(f"Unknown type {self.solver_type} for the solver.")

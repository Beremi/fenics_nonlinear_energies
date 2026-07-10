"""Mesh-aware primal and dual norms for PETSc coefficient vectors."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


def certify_spd_by_cholesky(
    operator: PETSc.Mat,
    *,
    factor_solver_type: str = "mumps",
    symmetry_tol: float = 1.0e-12,
    options_prefix: str = "riesz_spd_certificate_",
) -> dict[str, Any]:
    """Return a numerical SPD certificate for a constrained PETSc operator.

    Symmetry is checked explicitly using ``symmetry_tol`` relative to the
    matrix infinity norm. Positive definiteness is then certified by a
    symmetric direct factorization and its inertia: the factor must report no
    negative or zero pivots and one positive pivot per free-space row. The
    temporary factorization and any PETSc options used to request MUMPS inertia
    are released/restored even when setup fails.
    """

    rows, columns = (int(value) for value in operator.getSize())
    if rows <= 0 or columns != rows:
        raise ValueError("The Riesz operator must be a nonempty square free-space matrix.")
    symmetry_tol = float(symmetry_tol)
    if not np.isfinite(symmetry_tol) or symmetry_tol <= 0.0:
        raise ValueError("symmetry_tol must be finite and strictly positive")
    matrix_infinity_norm = float(operator.norm(PETSc.NormType.NORM_INFINITY))
    symmetry_absolute_tolerance = float(
        symmetry_tol * max(1.0, matrix_infinity_norm)
    )
    if not bool(operator.isSymmetric(tol=symmetry_absolute_tolerance)):
        raise ValueError(
            "The Riesz operator is not symmetric on the constrained free space "
            "at the requested scale-aware tolerance: "
            f"absolute={symmetry_absolute_tolerance:.3e}, "
            f"relative-to-infinity-norm={symmetry_tol:.3e}."
        )

    solver_type = str(factor_solver_type).strip()
    if not solver_type:
        raise ValueError("factor_solver_type must name a PETSc factorization backend")
    prefix = str(options_prefix).strip() or "riesz_spd_certificate_"
    if not prefix.endswith("_"):
        prefix += "_"

    # MUMPS requires null-pivot detection for MatGetInertia. Keep the option
    # private to this KSP and restore a pre-existing value exactly.
    option_updates: dict[str, object] = {}
    if solver_type.lower() == "mumps":
        option_updates[f"{prefix}mat_mumps_icntl_24"] = 1

    options = PETSc.Options()
    previous_options: dict[str, tuple[bool, object | None]] = {}
    certificate_ksp: PETSc.KSP | None = None
    try:
        for key, value in option_updates.items():
            present = key in options
            previous_options[key] = (present, options[key] if present else None)
            options[key] = value

        # PCCHOLESKY uses an LDL^T factorization for factor packages such as
        # MUMPS; inertia is the decisive SPD check, rather than setup success.
        operator.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        certificate_ksp = PETSc.KSP().create(comm=operator.getComm())
        certificate_ksp.setOptionsPrefix(prefix)
        certificate_ksp.setType("preonly")
        pc = certificate_ksp.getPC()
        pc.setType("cholesky")
        pc.setFactorSolverType(solver_type)
        certificate_ksp.setOperators(operator)
        certificate_ksp.setUp()
        factor = pc.getFactorMatrix()
        negative, zero, positive = (int(value) for value in factor.getInertia())
        if negative != 0 or zero != 0 or positive != rows:
            raise ValueError(
                "The constrained Riesz operator is not positive definite: "
                f"inertia=(negative={negative}, zero={zero}, positive={positive}), "
                f"expected (0, 0, {rows})."
            )
        return {
            "method": "symmetric_direct_factorization_inertia",
            "factor_solver_type": str(pc.getFactorSolverType() or solver_type),
            "symmetry_checked": True,
            "matrix_infinity_norm": float(matrix_infinity_norm),
            "symmetry_relative_tolerance": float(symmetry_tol),
            "symmetry_absolute_tolerance": float(symmetry_absolute_tolerance),
            "matrix_rows": int(rows),
            "matrix_columns": int(columns),
            "inertia": {
                "negative": int(negative),
                "zero": int(zero),
                "positive": int(positive),
            },
            "certified_spd": True,
        }
    except PETSc.Error as exc:
        raise RuntimeError(
            "PETSc could not certify the constrained Riesz operator by "
            f"{solver_type} Cholesky/inertia: {exc}"
        ) from exc
    finally:
        if certificate_ksp is not None:
            certificate_ksp.destroy()
        for key, (present, value) in previous_options.items():
            if present:
                options[key] = value
            elif key in options:
                del options[key]


@dataclass(frozen=True)
class MetricEvaluation:
    """A norm value and the diagnostics needed to audit its computation."""

    value: float
    metadata: dict[str, Any]


class EuclideanMetric:
    """The coefficient-vector Euclidean metric used by legacy solver paths."""

    name = "coefficient_l2"

    def __init__(self) -> None:
        self.first_dual_evaluation: MetricEvaluation | None = None
        self.last_dual_evaluation: MetricEvaluation | None = None

    def primal_norm(self, vec: PETSc.Vec) -> MetricEvaluation:
        value = float(vec.norm(PETSc.NormType.NORM_2))
        return MetricEvaluation(value, {"metric": self.name, "kind": "primal"})

    def dual_norm(self, vec: PETSc.Vec) -> MetricEvaluation:
        value = float(vec.norm(PETSc.NormType.NORM_2))
        evaluation = MetricEvaluation(value, {"metric": self.name, "kind": "dual"})
        if self.first_dual_evaluation is None:
            self.first_dual_evaluation = evaluation
        self.last_dual_evaluation = evaluation
        return evaluation

    def distance(self, left: PETSc.Vec, right: PETSc.Vec) -> MetricEvaluation:
        work = left.duplicate()
        try:
            left.copy(work)
            work.axpy(-1.0, right)
            evaluation = self.primal_norm(work)
            return MetricEvaluation(
                evaluation.value,
                {**evaluation.metadata, "kind": "primal_distance"},
            )
        finally:
            work.destroy()

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "riesz_operator": "identity"}


class DiagonalRieszMetric:
    """An SPD diagonal Riesz map, suitable for lumped mass/energy metrics."""

    def __init__(
        self,
        weights: PETSc.Vec,
        *,
        name: str,
        provenance: dict[str, Any] | None = None,
        positivity_floor: float = 0.0,
    ) -> None:
        self.name = str(name)
        self.weights = weights.copy()
        self.provenance = dict(provenance or {})
        self.first_dual_evaluation: MetricEvaluation | None = None
        self.last_dual_evaluation: MetricEvaluation | None = None
        local = np.asarray(self.weights.getArray(readonly=True), dtype=np.float64)
        local_all_finite = bool(np.all(np.isfinite(local)))
        global_all_finite = bool(
            self.weights.getComm().tompi4py().allreduce(local_all_finite, op=MPI.LAND)
        )
        local_min = float(np.min(local)) if local.size else math.inf
        global_min = float(
            self.weights.getComm().tompi4py().allreduce(local_min, op=MPI.MIN)
        )
        if (
            not global_all_finite
            or not np.isfinite(global_min)
            or global_min <= float(positivity_floor)
        ):
            self.weights.destroy()
            raise ValueError(
                "Diagonal Riesz weights must be finite and strictly positive on the free space."
            )
        self._work = self.weights.duplicate()

    def _weighted_square_sum(self, vec: PETSc.Vec, *, inverse: bool) -> float:
        values = np.asarray(vec.getArray(readonly=True), dtype=np.float64)
        weights = np.asarray(self.weights.getArray(readonly=True), dtype=np.float64)
        local = float(np.dot(values, values / weights if inverse else weights * values))
        return float(vec.getComm().tompi4py().allreduce(local, op=MPI.SUM))

    def primal_norm(self, vec: PETSc.Vec) -> MetricEvaluation:
        squared = self._weighted_square_sum(vec, inverse=False)
        if not np.isfinite(squared):
            raise ValueError("The diagonal Riesz primal quadratic form is nonfinite.")
        value = float(np.sqrt(max(0.0, squared)))
        return MetricEvaluation(
            value,
            {"metric": self.name, "kind": "primal", "riesz_solve": "diagonal_exact"},
        )

    def dual_norm(self, vec: PETSc.Vec) -> MetricEvaluation:
        squared = self._weighted_square_sum(vec, inverse=True)
        if not np.isfinite(squared):
            raise ValueError("The diagonal inverse-Riesz quadratic form is nonfinite.")
        value = float(np.sqrt(max(0.0, squared)))
        evaluation = MetricEvaluation(
            value,
            {"metric": self.name, "kind": "dual", "riesz_solve": "diagonal_exact"},
        )
        if self.first_dual_evaluation is None:
            self.first_dual_evaluation = evaluation
        self.last_dual_evaluation = evaluation
        return evaluation

    def distance(self, left: PETSc.Vec, right: PETSc.Vec) -> MetricEvaluation:
        left.copy(self._work)
        self._work.axpy(-1.0, right)
        evaluation = self.primal_norm(self._work)
        return MetricEvaluation(
            evaluation.value,
            {**evaluation.metadata, "kind": "primal_distance"},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "riesz_operator": "diagonal",
            "provenance": self.provenance,
        }

    def destroy(self) -> None:
        self._work.destroy()
        self.weights.destroy()


class MatrixRieszMetric:
    """Primal/dual norms induced by an SPD PETSc matrix on the free space."""

    def __init__(
        self,
        operator: PETSc.Mat,
        *,
        name: str,
        provenance: dict[str, Any] | None = None,
        ksp_type: str = "cg",
        pc_type: str = "jacobi",
        rtol: float = 1.0e-10,
        atol: float = 1.0e-14,
        max_it: int = 1000,
        require_symmetric: bool = True,
        true_residual_rtol: float | None = None,
        set_from_options: bool = False,
        options_prefix: str | None = None,
    ) -> None:
        self.name = str(name)
        self.operator = operator
        self.provenance = dict(provenance or {})
        self.solution: PETSc.Vec | None = None
        self.difference: PETSc.Vec | None = None
        self.work: PETSc.Vec | None = None
        self.residual: PETSc.Vec | None = None
        self.ksp: PETSc.KSP | None = None
        self._destroyed = False
        self.first_dual_evaluation: MetricEvaluation | None = None
        self.last_dual_evaluation: MetricEvaluation | None = None
        self.requested_rtol = float(rtol)
        self.requested_atol = float(atol)
        self.requested_max_it = int(max_it)
        self.true_residual_rtol = (
            None if true_residual_rtol is None else float(true_residual_rtol)
        )
        self.set_from_options = bool(set_from_options)
        prefix = str(options_prefix or "").strip()
        if self.set_from_options and not prefix:
            safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", self.name).strip("_").lower()
            prefix = f"riesz_{safe_name or 'matrix'}_"
        if prefix and not prefix.endswith("_"):
            prefix += "_"
        self.options_prefix = prefix
        self.effective_rtol = self.requested_rtol
        self.effective_atol = self.requested_atol
        self.effective_dtol = PETSc.DECIDE
        self.effective_max_it = self.requested_max_it
        if not np.isfinite(self.requested_rtol) or self.requested_rtol < 0.0:
            raise ValueError("Riesz KSP rtol must be finite and nonnegative")
        if not np.isfinite(self.requested_atol) or self.requested_atol < 0.0:
            raise ValueError("Riesz KSP atol must be finite and nonnegative")
        if self.requested_max_it <= 0:
            raise ValueError("Riesz KSP max_it must be strictly positive")
        if self.true_residual_rtol is not None and (
            not np.isfinite(self.true_residual_rtol)
            or self.true_residual_rtol <= 0.0
        ):
            raise ValueError("true_residual_rtol must be finite and strictly positive")
        if require_symmetric and not bool(operator.isSymmetric(tol=1.0e-12)):
            raise ValueError("The Riesz operator must be symmetric on the constrained free space.")
        try:
            self.solution = operator.createVecRight()
            self.difference = operator.createVecRight()
            self.work = operator.createVecLeft()
            self.residual = operator.createVecLeft()
            self.ksp = PETSc.KSP().create(comm=operator.getComm())
            if self.options_prefix:
                self.ksp.setOptionsPrefix(self.options_prefix)
            self.ksp.setOperators(operator)
            self.ksp.setType(str(ksp_type))
            self.ksp.getPC().setType(str(pc_type))
            self.ksp.setTolerances(
                rtol=self.requested_rtol,
                atol=self.requested_atol,
                max_it=self.requested_max_it,
            )
            if self.set_from_options:
                self.ksp.setFromOptions()
            self.ksp.setUp()
            (
                self.effective_rtol,
                self.effective_atol,
                self.effective_dtol,
                self.effective_max_it,
            ) = self.ksp.getTolerances()
        except Exception:
            self.destroy()
            raise

    def primal_norm(self, vec: PETSc.Vec) -> MetricEvaluation:
        if self.work is None:
            raise RuntimeError("The Riesz metric has been destroyed")
        self.operator.mult(vec, self.work)
        squared = float(vec.dot(self.work))
        if not np.isfinite(squared):
            raise ValueError("The Riesz primal quadratic form is nonfinite.")
        tolerance = 256.0 * np.finfo(np.float64).eps * max(1.0, abs(squared))
        if squared < -tolerance:
            raise ValueError("The Riesz operator produced a negative quadratic form.")
        value = float(np.sqrt(max(0.0, squared)))
        return MetricEvaluation(
            value,
            {"metric": self.name, "kind": "primal", "riesz_solve": "not_required"},
        )

    def dual_norm(self, vec: PETSc.Vec) -> MetricEvaluation:
        if self.solution is None or self.residual is None or self.ksp is None:
            raise RuntimeError("The Riesz metric has been destroyed")
        self.solution.set(0.0)
        self.ksp.solve(vec, self.solution)
        reason = int(self.ksp.getConvergedReason())
        self.operator.mult(self.solution, self.residual)
        self.residual.axpy(-1.0, vec)
        true_residual = float(self.residual.norm(PETSc.NormType.NORM_2))
        rhs_norm = float(vec.norm(PETSc.NormType.NORM_2))
        relative_true_residual = true_residual / max(rhs_norm, np.finfo(np.float64).tiny)
        squared = float(vec.dot(self.solution))
        if not all(
            np.isfinite(value)
            for value in (true_residual, rhs_norm, relative_true_residual, squared)
        ):
            raise ValueError("The inverse-Riesz solve or quadratic form is nonfinite.")
        tolerance = 256.0 * np.finfo(np.float64).eps * max(1.0, abs(squared))
        if reason <= 0:
            raise RuntimeError(
                f"Riesz solve failed with PETSc reason {reason}; dual norm is not certified."
            )
        if (
            self.true_residual_rtol is not None
            and relative_true_residual > self.true_residual_rtol
        ):
            raise RuntimeError(
                "Riesz solve reported convergence but failed the independently "
                "recomputed true-residual gate: "
                f"{relative_true_residual:.3e} > {self.true_residual_rtol:.3e}."
            )
        if squared < -tolerance:
            raise ValueError("The inverse Riesz quadratic form is negative.")
        value = float(np.sqrt(max(0.0, squared)))
        evaluation = MetricEvaluation(
            value,
            {
                "metric": self.name,
                "kind": "dual",
                "riesz_solve": "iterative",
                "ksp_type": self.ksp.getType(),
                "pc_type": self.ksp.getPC().getType(),
                "iterations": int(self.ksp.getIterationNumber()),
                "reason": reason,
                "reported_residual": float(self.ksp.getResidualNorm()),
                "true_residual": true_residual,
                "relative_true_residual": relative_true_residual,
                "rhs_norm": rhs_norm,
                "requested_rtol": float(self.requested_rtol),
                "requested_atol": float(self.requested_atol),
                "requested_max_it": int(self.requested_max_it),
                "effective_rtol": float(self.effective_rtol),
                "effective_atol": float(self.effective_atol),
                "effective_dtol": float(self.effective_dtol),
                "effective_max_it": int(self.effective_max_it),
                "true_residual_rtol_gate": self.true_residual_rtol,
            },
        )
        if self.first_dual_evaluation is None:
            self.first_dual_evaluation = evaluation
        self.last_dual_evaluation = evaluation
        return evaluation

    def distance(self, left: PETSc.Vec, right: PETSc.Vec) -> MetricEvaluation:
        if self.difference is None:
            raise RuntimeError("The Riesz metric has been destroyed")
        left.copy(self.difference)
        self.difference.axpy(-1.0, right)
        evaluation = self.primal_norm(self.difference)
        return MetricEvaluation(
            evaluation.value,
            {**evaluation.metadata, "kind": "primal_distance"},
        )

    def describe(self) -> dict[str, Any]:
        if self.ksp is None:
            raise RuntimeError("The Riesz metric has been destroyed")
        return {
            "name": self.name,
            "riesz_operator": "petsc_matrix",
            "provenance": self.provenance,
            "ksp_type": self.ksp.getType(),
            "pc_type": self.ksp.getPC().getType(),
            "requested_rtol": float(self.requested_rtol),
            "requested_atol": float(self.requested_atol),
            "requested_max_it": int(self.requested_max_it),
            "effective_rtol": float(self.effective_rtol),
            "effective_atol": float(self.effective_atol),
            "effective_dtol": float(self.effective_dtol),
            "effective_max_it": int(self.effective_max_it),
            "true_residual_rtol_gate": self.true_residual_rtol,
            "set_from_petsc_options": bool(self.set_from_options),
            "petsc_options_prefix": str(self.options_prefix),
        }

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        for name in ("ksp", "solution", "difference", "work", "residual"):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.destroy()
                setattr(self, name, None)

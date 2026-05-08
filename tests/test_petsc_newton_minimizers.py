from __future__ import annotations

import numpy as np
from petsc4py import PETSc

from src.core.petsc.minimizers import newton


def _vec_from_array(x0: np.ndarray) -> PETSc.Vec:
    vec = PETSc.Vec().createSeq(len(x0), comm=PETSc.COMM_SELF)
    arr = vec.getArray(readonly=False)
    arr[:] = x0
    del arr
    return vec


def test_hybrid_trust_accepts_roundoff_limited_converged_step():
    x = _vec_from_array(np.array([1.0e-8], dtype=np.float64))

    def energy_fn(v: PETSc.Vec) -> float:
        xa = np.asarray(v.getArray(readonly=True), dtype=np.float64)
        return -2.0e-20 if abs(float(xa[0])) < 0.5e-8 else 0.0

    def gradient_fn(v: PETSc.Vec, g: PETSc.Vec) -> None:
        xa = np.asarray(v.getArray(readonly=True), dtype=np.float64)
        ga = g.getArray(readonly=False)
        ga[:] = xa
        del ga

    def hessian_matvec_fn(_x: PETSc.Vec, vin: PETSc.Vec, vout: PETSc.Vec) -> None:
        vin.copy(vout)

    def trust_solve_fn(_x: PETSc.Vec, rhs: PETSc.Vec, p: PETSc.Vec, _radius: float) -> int:
        rhs.copy(p)
        return 1

    res = newton(
        energy_fn,
        gradient_fn,
        lambda _x, _p, _h: 0,
        x,
        tolf=1e-12,
        tolg=1e-10,
        line_search="armijo",
        maxit=4,
        tolx_rel=1e-6,
        tolx_abs=1e-14,
        require_all_convergence=True,
        hessian_matvec_fn=hessian_matvec_fn,
        trust_subproblem_solve_fn=trust_solve_fn,
        trust_subproblem_line_search=True,
        trust_region=True,
        trust_eta_shrink=0.05,
        save_history=True,
        comm=PETSc.COMM_SELF.tompi4py(),
    )

    assert res["success"]
    assert res["message"] == "Converged (energy, step, gradient)"
    assert res["history"][-1]["used_roundoff_acceptance"] is True
    assert res["history"][-1]["accepted_step"] is True
    assert res["history"][-1]["trust_ratio"] < 0.05
    x.destroy()

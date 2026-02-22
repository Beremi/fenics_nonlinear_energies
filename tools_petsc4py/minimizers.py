"""
Newton minimizer with golden-section line search for PETSc Vec objects.

This module mirrors the JAX minimizer in ``tools/minimizers.py`` but operates
on PETSc vectors, enabling MPI-parallel execution.  It depends **only** on
petsc4py and numpy — all problem-specific assembly (DOLFINx / UFL / …) is
done through user-provided callbacks.

Typical usage
-------------
>>> from tools_petsc4py.minimizers import newton
>>> result = newton(energy_fn, gradient_fn, hessian_solve_fn, x,
...                 comm=MPI.COMM_WORLD, ghost_update_fn=ghost_update)
"""
import time
import numpy as np
from petsc4py import PETSc


# ---------------------------------------------------------------------------
# Golden-section search (mirrors tools.minimizers.zlatyrez)
# ---------------------------------------------------------------------------

def golden_section_search(f, a, b, tol):
    """Minimise a univariate function on *[a, b]* via golden-section search.

    Parameters
    ----------
    f : callable(float) -> float
        Scalar function to minimise.
    a, b : float
        Search interval end-points.
    tol : float
        Stop when the interval width drops below *tol*.

    Returns
    -------
    alpha : float
        Approximate minimiser (midpoint of the final interval).
    n_evals : int
        Number of *f* evaluations beyond the two initial probes.
    """
    gamma = 0.5 + np.sqrt(5) / 2          # golden ratio ≈ 1.618

    an, bn = float(a), float(b)
    dn = (bn - an) / gamma + an
    cn = an + bn - dn

    fcn = f(cn)
    fdn = f(dn)
    n_evals = 0

    while bn - an > tol:
        if fcn < fdn:
            bn = dn
            dn, cn = cn, an + bn - cn
            fdn, fcn = fcn, f(cn)
        else:
            an = cn
            cn, dn = dn, an + bn - dn
            fcn, fdn = fdn, f(dn)
        n_evals += 1

    return (an + bn) / 2.0, n_evals


# ---------------------------------------------------------------------------
# Newton minimiser (mirrors tools.minimizers.newton)
# ---------------------------------------------------------------------------

def newton(
    energy_fn,
    gradient_fn,
    hessian_solve_fn,
    x,
    tolf=1e-5,
    tolg=1e-3,
    linesearch_tol=1e-3,
    linesearch_interval=(-0.5, 2.0),
    maxit=100,
    verbose=False,
    comm=None,
    ghost_update_fn=None,
    save_history=False,
):
    """Newton's method for energy minimisation on PETSc vectors.

    The algorithm is identical to :func:`tools.minimizers.newton` — the only
    difference is that all vector operations use PETSc instead of NumPy, so
    the solver can run under MPI.

    Parameters
    ----------
    energy_fn : callable(PETSc.Vec) -> float
        Evaluate the energy functional at *u*.  Must return a **globally
        reduced** scalar (i.e. the caller is responsible for ``allreduce``).
    gradient_fn : callable(x: PETSc.Vec, g: PETSc.Vec) -> None
        Assemble the gradient of the energy into *g*.  The callback must
        apply boundary conditions and any required ghost updates on *g*.
    hessian_solve_fn : callable(x: PETSc.Vec, rhs: PETSc.Vec, sol: PETSc.Vec) -> int
        Assemble the Hessian at *x*, solve  ``H · sol = rhs``, and return
        the number of KSP iterations.  The callback must apply BCs, update
        the KSP operator, and ghost-update *sol*.
    x : PETSc.Vec
        Initial guess — **modified in-place** to the solution.
    tolf : float
        Stop when  ``|J(x_new) − J(x_old)| < tolf``.
    tolg : float
        Stop when  ``‖∇J‖₂ < tolg``  (checked *before* the linear solve).
    linesearch_tol : float
        Golden-section interval tolerance.
    linesearch_interval : (float, float)
        Search interval for the step-size α.
    maxit : int
        Maximum Newton iterations.
    verbose : bool
        Print one line per iteration (rank 0 only).
    comm : MPI communicator (mpi4py) or None
        Used only for rank-aware printing.  If *None*, every rank prints.
    ghost_update_fn : callable(PETSc.Vec) -> None, optional
        Called after every vector arithmetic operation (``axpy``, ``waxpy``)
        to synchronise ghost values.  Required for distributed DOLFINx
        vectors; can be omitted for purely local vectors.

    Returns
    -------
    dict
        ``x`` – PETSc.Vec solution (same object as the input),
        ``fun`` – final energy value,
        ``nit`` – number of Newton iterations,
        ``time`` – wall-clock time (seconds),
        ``message`` – termination reason.
    """
    if ghost_update_fn is None:
        def ghost_update_fn(_v): return None          # noqa: E731

    rank = 0
    if comm is not None:
        rank = comm.Get_rank()

    # Work vectors — allocated once and reused every iteration.
    g = x.duplicate()           # gradient
    h = x.duplicate()           # Newton direction  (= −H⁻¹ g)
    x_trial = x.duplicate()    # scratch for line search

    start = time.perf_counter()
    fx = energy_fn(x)
    nit = 0
    message = "Maximum number of iterations reached"
    history = []

    for _ in range(maxit):
        # ---- gradient ----
        gradient_fn(x, g)
        normg = g.norm(PETSc.NormType.NORM_2)

        if normg < tolg:
            message = "Gradient norm converged"
            break

        nit += 1

        # ---- Hessian solve: H h = −g  (h is the descent direction) ----
        g.scale(-1.0)
        ksp_its = hessian_solve_fn(x, g, h)
        ghost_update_fn(h)

        # ---- line search: minimise  J(x + α h) ----
        ls_a, ls_b = linesearch_interval

        def _energy_at_alpha(alpha):
            x_trial.waxpy(alpha, h, x)
            ghost_update_fn(x_trial)
            val = energy_fn(x_trial)
            if not np.isfinite(val):
                return np.inf
            return val

        alpha, ls_evals = golden_section_search(
            _energy_at_alpha, ls_a, ls_b, linesearch_tol
        )

        # Guard against non-finite trial energies (e.g. det(F) <= 0 in hyperelasticity).
        # If golden-section ends in a non-finite region, backtrack toward zero.
        trial_val = _energy_at_alpha(alpha)
        if not np.isfinite(trial_val):
            alpha_bt = min(1.0, max(0.0, ls_b))
            if alpha_bt <= 0.0:
                alpha_bt = 1.0
            while alpha_bt > 1e-12:
                trial_val = _energy_at_alpha(alpha_bt)
                if np.isfinite(trial_val):
                    alpha = alpha_bt
                    break
                alpha_bt *= 0.5

        # ---- update ----
        x.axpy(alpha, h)
        ghost_update_fn(x)

        fx_old = fx
        fx = energy_fn(x)

        if verbose and rank == 0:
            print(
                f"it={nit}, J={fx:.5f}, dJ={fx_old - fx:.5e}, "
                f"||g||={normg:.5e}, alpha={alpha:.5e}, "
                f"ksp_its={ksp_its}, ls_evals={ls_evals}"
            )

        if save_history:
            history.append({
                "it": int(nit),
                "energy": float(fx),
                "dE": float(fx_old - fx),
                "grad_norm": float(normg),
                "alpha": float(alpha),
                "ksp_its": int(ksp_its),
                "ls_evals": int(ls_evals),
            })

        if abs(fx - fx_old) < tolf:
            message = "Energy change converged"
            break

    runtime = time.perf_counter() - start

    # Clean up work vectors.
    g.destroy()
    h.destroy()
    x_trial.destroy()

    return {
        "x": x,
        "fun": fx,
        "nit": nit,
        "time": runtime,
        "message": message,
        "history": history,
    }

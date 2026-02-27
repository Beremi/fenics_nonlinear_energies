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
    tolg_rel=0.0,
    linesearch_tol=1e-3,
    linesearch_interval=(-0.5, 2.0),
    maxit=100,
    tolx_rel=1e-6,
    tolx_abs=1e-10,
    require_all_convergence=False,
    fail_on_nonfinite=True,
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
    tolg_rel : float
        Optional relative gradient target for all-criteria convergence:
        ``‖∇J‖₂ < max(tolg, tolg_rel * ‖∇J(x0)‖₂)``.
    linesearch_tol : float
        Golden-section interval tolerance.
    linesearch_interval : (float, float)
        Search interval for the step-size α.
    maxit : int
        Maximum Newton iterations.
    require_all_convergence : bool
        If true, stop only when energy-change, step-size, and gradient
        criteria are all satisfied.
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
    g = x.duplicate()          # gradient
    h = x.duplicate()          # Newton direction  (= −H⁻¹ g)
    x_trial = x.duplicate()    # scratch for line search
    x_prev = x.duplicate()     # rollback buffer for non-finite updates

    start = time.perf_counter()
    fx = energy_fn(x)
    nit = 0
    message = "Maximum number of iterations reached"
    history = []
    initial_grad_norm = None

    for _ in range(maxit):
        t_iter_start = time.perf_counter()

        if fail_on_nonfinite and not np.isfinite(fx):
            message = f"Non-finite energy before Newton iteration {nit + 1}"
            break

        # ---- gradient ----
        t0 = time.perf_counter()
        gradient_fn(x, g)
        normg = g.norm(PETSc.NormType.NORM_2)
        t_grad = time.perf_counter() - t0

        if fail_on_nonfinite and not np.isfinite(normg):
            message = f"Non-finite gradient norm at Newton iteration {nit + 1}"
            break

        if initial_grad_norm is None:
            initial_grad_norm = normg

        grad_target = tolg
        if tolg_rel > 0.0 and np.isfinite(initial_grad_norm):
            grad_target = max(tolg, tolg_rel * initial_grad_norm)

        if (not require_all_convergence) and normg < grad_target:
            message = "Gradient norm converged"
            break

        nit += 1

        # ---- Hessian solve: H h = −g  (h is the descent direction) ----
        t0 = time.perf_counter()
        g.scale(-1.0)
        ksp_its = hessian_solve_fn(x, g, h)
        ghost_update_fn(h)
        hnorm = h.norm(PETSc.NormType.NORM_2)
        t_hess = time.perf_counter() - t0

        if fail_on_nonfinite and not np.isfinite(hnorm):
            message = f"Non-finite Newton direction norm at Newton iteration {nit}"
            break

        # ---- line search: minimise  J(x + α h) ----
        t0 = time.perf_counter()
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
        ls_repaired = False
        if not np.isfinite(trial_val):
            alpha_bt = min(1.0, max(0.0, ls_b))
            if alpha_bt <= 0.0:
                alpha_bt = 1.0
            found_finite = False
            while alpha_bt > 1e-12:
                trial_val = _energy_at_alpha(alpha_bt)
                if np.isfinite(trial_val):
                    alpha = alpha_bt
                    found_finite = True
                    ls_repaired = True
                    break
                alpha_bt *= 0.5
            if not found_finite and fail_on_nonfinite:
                message = (
                    f"Line search failed: no finite trial energy at Newton iteration {nit}"
                )
                break
        t_ls = time.perf_counter() - t0

        # ---- update ----
        t0 = time.perf_counter()
        x.copy(x_prev)
        x_prev_norm = x_prev.norm(PETSc.NormType.NORM_2)
        step_norm = abs(alpha) * hnorm
        step_rel = step_norm / max(1.0, x_prev_norm)
        x.axpy(alpha, h)
        ghost_update_fn(x)

        fx_old = fx
        fx = energy_fn(x)
        dE = fx_old - fx
        t_update = time.perf_counter() - t0
        t_iter_total = time.perf_counter() - t_iter_start

        if verbose and rank == 0:
            print(
                f"it={nit}, J={fx:.5f}, dJ={dE:.5e}, "
                f"||g||={normg:.5e}, alpha={alpha:.5e}, "
                f"ksp_its={ksp_its}, ls_evals={ls_evals}"
            )

        post_grad_norm = np.nan
        converged_energy = np.isfinite(dE) and abs(dE) < tolf
        converged_step = step_norm < tolx_abs or step_rel < tolx_rel
        if require_all_convergence and converged_energy and converged_step:
            gradient_fn(x, g)
            post_grad_norm = g.norm(PETSc.NormType.NORM_2)
            if fail_on_nonfinite and not np.isfinite(post_grad_norm):
                x_prev.copy(x)
                ghost_update_fn(x)
                message = f"Non-finite post-update gradient norm at Newton iteration {nit}"
                break

        if save_history:
            history.append({
                "it": int(nit),
                "energy": float(fx),
                "dE": float(dE),
                "grad_norm": float(normg),
                "grad_target": float(grad_target),
                "grad_norm_post": float(post_grad_norm),
                "alpha": float(alpha),
                "ksp_its": int(ksp_its),
                "ls_evals": int(ls_evals),
                "ls_repaired": bool(ls_repaired),
                "step_norm": float(step_norm),
                "step_rel": float(step_rel),
                "t_grad": float(t_grad),
                "t_hess": float(t_hess),
                "t_ls": float(t_ls),
                "t_update": float(t_update),
                "t_iter": float(t_iter_total),
            })

        if fail_on_nonfinite and not np.isfinite(fx):
            x_prev.copy(x)
            ghost_update_fn(x)
            message = f"Non-finite energy after update at Newton iteration {nit}"
            break

        if require_all_convergence:
            if converged_energy and converged_step and post_grad_norm < grad_target:
                message = "Converged (energy, step, gradient)"
                break
        elif converged_energy:
            message = "Energy change converged"
            break

    runtime = time.perf_counter() - start

    # Clean up work vectors.
    g.destroy()
    h.destroy()
    x_trial.destroy()
    x_prev.destroy()

    return {
        "x": x,
        "fun": fx,
        "nit": nit,
        "time": runtime,
        "message": message,
        "history": history,
        "success": "converged" in message.lower(),
    }

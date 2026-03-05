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


def _solve_trust_1d(rhs, curv, delta):
    """Solve min rhs*t + 0.5*curv*t^2 subject to |t| <= delta."""
    delta = max(0.0, float(delta))
    if delta <= 0.0:
        return 0.0

    if np.isfinite(curv) and curv > 1e-14:
        t = -rhs / curv
    else:
        # Negative/flat curvature: best boundary point in descent direction.
        if rhs > 0.0:
            t = -delta
        elif rhs < 0.0:
            t = delta
        else:
            t = 0.0
    return float(np.clip(t, -delta, delta))


def _quad_model_value(rhs, mm, s):
    return float(rhs.dot(s) + 0.5 * s.dot(mm.dot(s)))


def _solve_trust_2d(rhs, mm, delta):
    """Solve 2D trust-region model by unconstrained test + boundary scan."""
    delta = max(0.0, float(delta))
    s0 = np.zeros(2, dtype=np.float64)
    q0 = _quad_model_value(rhs, mm, s0)
    if delta <= 0.0:
        return s0, q0

    best_s = s0
    best_q = q0

    # Try unconstrained stationary point.
    try:
        su = np.linalg.solve(mm, -rhs)
        if np.all(np.isfinite(su)) and np.linalg.norm(su) <= delta:
            qu = _quad_model_value(rhs, mm, su)
            if qu < best_q:
                best_s, best_q = su, qu
    except np.linalg.LinAlgError:
        pass

    # Boundary search s = delta*[cos(t), sin(t)].
    n_scan = 181
    thetas = np.linspace(0.0, 2.0 * np.pi, num=n_scan, endpoint=False)
    for theta in thetas:
        st = delta * np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        qt = _quad_model_value(rhs, mm, st)
        if qt < best_q:
            best_s, best_q = st, qt

    return best_s, best_q


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
    project_fn=None,
    hessian_matvec_fn=None,
    save_history=False,
    trust_region=False,
    trust_radius_init=1.0,
    trust_radius_min=1e-8,
    trust_radius_max=1e6,
    trust_shrink=0.5,
    trust_expand=1.5,
    trust_eta_shrink=0.05,
    trust_eta_expand=0.75,
    trust_max_reject=6,
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
    if project_fn is None:
        def project_fn(_v): return None               # noqa: E731
    if hessian_matvec_fn is None:
        def hessian_matvec_fn(_x, _vin, vout):        # noqa: E731
            vout.set(0.0)

    rank = 0
    if comm is not None:
        rank = comm.Get_rank()

    # Work vectors — allocated once and reused every iteration.
    g = x.duplicate()          # gradient
    h = x.duplicate()          # Newton direction  (= −H⁻¹ g)
    x_trial = x.duplicate()    # scratch for line search
    x_prev = x.duplicate()     # rollback buffer for non-finite updates
    p = x.duplicate()          # trust-region clipped direction

    start = time.perf_counter()
    fx = energy_fn(x)
    nit = 0
    message = "Maximum number of iterations reached"
    history = []
    initial_grad_norm = None
    trust_radius = float(trust_radius_init)
    if trust_region:
        trust_radius = min(max(trust_radius, trust_radius_min), trust_radius_max)

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

        # ---- Hessian solve: H h = -g ----
        t0 = time.perf_counter()
        g.copy(p)
        p.scale(-1.0)
        ksp_its = hessian_solve_fn(x, p, h)
        ghost_update_fn(h)
        hnorm = h.norm(PETSc.NormType.NORM_2)
        t_hess = time.perf_counter() - t0

        if fail_on_nonfinite and not np.isfinite(hnorm):
            message = f"Non-finite Newton direction norm at Newton iteration {nit}"
            break

        x.copy(x_prev)
        x_prev_norm = x_prev.norm(PETSc.NormType.NORM_2)
        fx_old = fx
        alpha = 0.0
        ls_evals = 0
        ls_repaired = False
        dE = 0.0
        rho = np.nan
        step_norm = 0.0
        step_rel = 0.0
        n_reject = 0
        t_update = 0.0
        trust_qp = np.nan
        accepted_step = False
        used_gradient_fallback = False

        t_ls_start = time.perf_counter()
        ls_a, ls_b = linesearch_interval
        if ls_b <= ls_a:
            ls_a, ls_b = -0.5, 2.0

        def _energy_at_alpha(alpha_local):
            x_trial.waxpy(alpha_local, p, x_prev)
            project_fn(x_trial)
            ghost_update_fn(x_trial)
            val = energy_fn(x_trial)
            if not np.isfinite(val):
                return np.inf
            return val

        if trust_region:
            # MATLAB-style trust-region step: 2D subspace model + optional line search.
            dump = 10.0
            trust_radius = min(max(trust_radius, trust_radius_min), trust_radius_max)
            delta_old = trust_radius

            # Build orthonormal 2D basis {z1, z2} from Newton and gradient directions.
            z1 = x.duplicate()
            z2 = x.duplicate()
            hz1 = x.duplicate()
            hz2 = x.duplicate()
            sg = x.duplicate()
            has_z2 = False

            if np.isfinite(hnorm) and hnorm > 1e-20:
                h.copy(z1)
                z1.scale(1.0 / hnorm)
            elif normg > 1e-20:
                g.copy(z1)
                z1.scale(-1.0 / normg)
            else:
                z1.set(0.0)

            if normg > 1e-20:
                g.copy(z2)
                z2.scale(1.0 / normg)
                z2.axpy(-z1.dot(z2), z1)
                z2_norm = z2.norm(PETSc.NormType.NORM_2)
                if z2_norm > 1e-12:
                    z2.scale(1.0 / z2_norm)
                    has_z2 = True

            hessian_matvec_fn(x, z1, hz1)
            ghost_update_fn(hz1)

            if has_z2:
                hessian_matvec_fn(x, z2, hz2)
                ghost_update_fn(hz2)

                rhs_red = np.array([g.dot(z1), g.dot(z2)], dtype=np.float64)
                mm = np.array(
                    [
                        [z1.dot(hz1), z1.dot(hz2)],
                        [z2.dot(hz1), z2.dot(hz2)],
                    ],
                    dtype=np.float64,
                )
                st, qp1 = _solve_trust_2d(rhs_red, mm, trust_radius)
                p.set(0.0)
                p.axpy(float(st[0]), z1)
                p.axpy(float(st[1]), z2)
                snod_norm = float(np.linalg.norm(st))
            else:
                rhs1 = float(g.dot(z1))
                curv1 = float(z1.dot(hz1))
                t1 = _solve_trust_1d(rhs1, curv1, trust_radius)
                z1.copy(p)
                p.scale(t1)
                snod_norm = abs(t1)
                qp1 = float(rhs1 * t1 + 0.5 * curv1 * t1 * t1)

            # Also test gradient direction (MATLAB fallback candidate).
            if normg > 1e-20:
                g.copy(sg)
                sg.scale(1.0 / normg)
                hessian_matvec_fn(x, sg, hz2)
                ghost_update_fn(hz2)
                rhs_g = float(g.dot(sg))
                curv_g = float(sg.dot(hz2))
                tg = _solve_trust_1d(rhs_g, curv_g, trust_radius)
                sg.scale(tg)
                qp2 = float(rhs_g * tg + 0.5 * curv_g * tg * tg)
                sg_norm = abs(tg)
            else:
                qp2 = np.inf
                sg_norm = 0.0

            if qp2 <= qp1:
                sg.copy(p)
                snod_norm = sg_norm
                trust_qp = qp2
                used_gradient_fallback = True
            else:
                trust_qp = qp1

            # Optional line search on trust step (as in MATLAB code).
            alpha, ls_eval_local = golden_section_search(
                _energy_at_alpha, ls_a, ls_b, linesearch_tol
            )
            ls_evals += ls_eval_local
            newval = _energy_at_alpha(alpha)
            step_size = abs(alpha) * snod_norm

            # Trust-radius update mirrors the MATLAB line-search branch.
            if np.isfinite(newval) and newval < fx_old:
                trust_radius = trust_radius * (1.0 / dump) + step_size * (1.0 - 1.0 / dump)
            else:
                trust_radius = trust_radius / dump
            trust_radius = min(max(trust_radius, trust_radius_min), trust_radius_max)

            if np.isfinite(newval) and newval < fx_old:
                fx = newval
                dE = fx_old - fx
                step_norm = step_size
                step_rel = step_norm / max(1.0, x_prev_norm)
                t_update0 = time.perf_counter()
                x_trial.copy(x)
                project_fn(x)
                ghost_update_fn(x)
                t_update = time.perf_counter() - t_update0
                accepted_step = True
            else:
                x_prev.copy(x)
                project_fn(x)
                ghost_update_fn(x)
                alpha = 0.0
                dE = 0.0
                step_norm = 0.0
                step_rel = 0.0
                n_reject = 1

            z1.destroy()
            z2.destroy()
            hz1.destroy()
            hz2.destroy()
            sg.destroy()

        else:
            # Plain Newton + line search.
            h.copy(p)
            pnorm = p.norm(PETSc.NormType.NORM_2)
            if pnorm > 1e-20:
                alpha, ls_eval_local = golden_section_search(
                    _energy_at_alpha, ls_a, ls_b, linesearch_tol
                )
                ls_evals += ls_eval_local
                fx_trial = _energy_at_alpha(alpha)
                if np.isfinite(fx_trial) and fx_trial < fx_old:
                    fx = fx_trial
                    dE = fx_old - fx
                    step_norm = abs(alpha) * pnorm
                    step_rel = step_norm / max(1.0, x_prev_norm)
                    t_update0 = time.perf_counter()
                    x_trial.copy(x)
                    project_fn(x)
                    ghost_update_fn(x)
                    t_update = time.perf_counter() - t_update0
                    accepted_step = True
                else:
                    x_prev.copy(x)
                    project_fn(x)
                    ghost_update_fn(x)
                    alpha = 0.0
                    n_reject = 1
            else:
                x_prev.copy(x)
                project_fn(x)
                ghost_update_fn(x)
                alpha = 0.0
                n_reject = 1

        t_ls = time.perf_counter() - t_ls_start
        t_iter_total = time.perf_counter() - t_iter_start

        if verbose and rank == 0:
            print(
                f"it={nit}, J={fx:.5f}, dJ={dE:.5e}, "
                f"||g||={normg:.5e}, alpha={alpha:.5e}, "
                f"ksp_its={ksp_its}, ls_evals={ls_evals}"
            )

        post_grad_norm = np.nan
        converged_energy = accepted_step and np.isfinite(dE) and abs(dE) < tolf
        converged_step = accepted_step and (step_norm < tolx_abs or step_rel < tolx_rel)
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
                "used_gradient_fallback": bool(used_gradient_fallback),
                "accepted_step": bool(accepted_step),
                "step_norm": float(step_norm),
                "step_rel": float(step_rel),
                "t_grad": float(t_grad),
                "t_hess": float(t_hess),
                "t_ls": float(t_ls),
                "t_update": float(t_update),
                "t_iter": float(t_iter_total),
                "trust_radius": float(trust_radius),
                "trust_ratio": float(rho),
                "trust_qp": float(trust_qp),
                "trust_rejects": int(n_reject),
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
    p.destroy()
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

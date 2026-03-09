"""
Newton minimizer with golden-section line search for PETSc Vec objects.

This module mirrors the JAX minimizer in ``tools/minimizers.py`` but operates
on PETSc vectors, enabling MPI-parallel execution. It depends only on
``petsc4py`` and ``numpy``. Problem-specific assembly stays in callbacks.
"""

import time

import numpy as np
from petsc4py import PETSc


def golden_section_search(f, a, b, tol):
    """Minimise a univariate function on ``[a, b]`` via golden-section search."""
    gamma = 0.5 + np.sqrt(5) / 2

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


def _repair_linesearch_interval(
    f,
    a,
    b,
    center=0.0,
    center_value=None,
    tol=1e-6,
    max_bisect=60,
):
    """Bisect non-finite interval sides back toward a known finite center point."""
    center = float(center)
    a = float(a)
    b = float(b)
    tol = max(float(tol), 1e-12)
    n_evals = 0
    repaired = False

    if center_value is None:
        center_value = f(center)
        n_evals += 1
    if not np.isfinite(center_value):
        return a, b, n_evals, repaired

    def _repair_side(bound, side_sign):
        nonlocal n_evals, repaired
        if side_sign < 0 and not (bound < center):
            return bound
        if side_sign > 0 and not (bound > center):
            return bound

        f_bound = f(bound)
        n_evals += 1
        if np.isfinite(f_bound):
            return bound

        repaired = True
        finite = center
        nonfinite = bound
        for _ in range(max_bisect):
            mid = 0.5 * (finite + nonfinite)
            f_mid = f(mid)
            n_evals += 1
            if np.isfinite(f_mid):
                finite = mid
            else:
                nonfinite = mid
            if abs(nonfinite - finite) <= tol:
                break
        return finite

    a = _repair_side(a, -1.0)
    b = _repair_side(b, +1.0)
    return a, b, n_evals, repaired


def _solve_trust_1d(rhs, curv, delta):
    """Solve ``min rhs*t + 0.5*curv*t^2`` subject to ``|t| <= delta``."""
    delta = max(0.0, float(delta))
    if delta <= 0.0:
        return 0.0

    if np.isfinite(curv) and curv > 1e-14:
        t = -rhs / curv
    else:
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
    """Solve a 2D trust-region model by stationary-point test + boundary scan."""
    delta = max(0.0, float(delta))
    s0 = np.zeros(2, dtype=np.float64)
    q0 = _quad_model_value(rhs, mm, s0)
    if delta <= 0.0:
        return s0, q0

    best_s = s0
    best_q = q0

    try:
        su = np.linalg.solve(mm, -rhs)
        if np.all(np.isfinite(su)) and np.linalg.norm(su) <= delta:
            qu = _quad_model_value(rhs, mm, su)
            if qu < best_q:
                best_s, best_q = su, qu
    except np.linalg.LinAlgError:
        pass

    thetas = np.linspace(0.0, 2.0 * np.pi, num=181, endpoint=False)
    for theta in thetas:
        st = delta * np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        qt = _quad_model_value(rhs, mm, st)
        if qt < best_q:
            best_s, best_q = st, qt

    return best_s, best_q


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
    trust_subproblem_solve_fn=None,
    trust_subproblem_line_search=False,
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
    step_time_limit_s=None,
):
    """Newton's method for energy minimisation on PETSc vectors."""
    trust_shrink = float(trust_shrink)
    trust_expand = float(trust_expand)
    trust_eta_shrink = float(trust_eta_shrink)
    trust_eta_expand = float(trust_eta_expand)
    trust_max_reject = max(0, int(trust_max_reject))

    if not np.isfinite(trust_shrink) or trust_shrink <= 0.0 or trust_shrink >= 1.0:
        trust_shrink = 0.5
    if not np.isfinite(trust_expand) or trust_expand < 1.0:
        trust_expand = 1.5
    if not np.isfinite(trust_eta_shrink):
        trust_eta_shrink = 0.05
    if not np.isfinite(trust_eta_expand):
        trust_eta_expand = 0.75
    trust_eta_shrink = min(max(trust_eta_shrink, 0.0), 1.0)
    trust_eta_expand = min(max(trust_eta_expand, trust_eta_shrink), 1.0)

    if ghost_update_fn is None:
        def ghost_update_fn(_v):
            return None

    if project_fn is None:
        def project_fn(_v):
            return None

    if hessian_matvec_fn is None:
        def hessian_matvec_fn(_x, _vin, vout):
            vout.set(0.0)

    rank = 0
    if comm is not None:
        rank = comm.Get_rank()

    g = x.duplicate()
    h = x.duplicate()
    x_trial = x.duplicate()
    x_prev = x.duplicate()
    p = x.duplicate()

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

        trust_ksp_active = bool(trust_region and trust_subproblem_solve_fn is not None)
        ksp_its = 0
        hnorm = np.nan
        t_hess = 0.0
        if not trust_ksp_active:
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
        trust_qp = np.nan
        pred_reduction = np.nan
        actual_reduction = np.nan
        trust_rejects = 0
        accepted_step = False
        used_gradient_fallback = False
        terminate_after_iter = False
        terminate_message = None

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
            trust_radius = min(max(trust_radius, trust_radius_min), trust_radius_max)

            if trust_ksp_active:
                max_trust_attempts = trust_max_reject + 1
                for attempt_idx in range(max_trust_attempts):
                    trial_radius = float(min(max(trust_radius, trust_radius_min), trust_radius_max))

                    t_hess0 = time.perf_counter()
                    g.copy(h)
                    h.scale(-1.0)
                    ksp_its = trust_subproblem_solve_fn(x, h, p, trial_radius)
                    ghost_update_fn(p)
                    snod_norm = p.norm(PETSc.NormType.NORM_2)
                    t_hess += time.perf_counter() - t_hess0

                    if fail_on_nonfinite and not np.isfinite(snod_norm):
                        message = f"Non-finite trust-region step norm at Newton iteration {nit}"
                        terminate_after_iter = True
                        terminate_message = message
                        break

                    used_gradient_fallback = False
                    if np.isfinite(snod_norm) and snod_norm > 1e-20:
                        hessian_matvec_fn(x, p, h)
                        ghost_update_fn(h)
                        step_linear = float(g.dot(p))
                        step_curv = float(p.dot(h))
                        trust_qp = float(step_linear + 0.5 * step_curv)
                        if trust_subproblem_line_search:
                            alpha_lo = float(ls_a)
                            alpha_hi = float(ls_b)
                            alpha_cap = trial_radius / snod_norm
                            alpha_lo = max(alpha_lo, -alpha_cap)
                            alpha_hi = min(alpha_hi, alpha_cap)
                            if (
                                np.isfinite(alpha_lo)
                                and np.isfinite(alpha_hi)
                                and alpha_hi > alpha_lo + 1e-14
                            ):
                                alpha_lo, alpha_hi, ls_repair_evals, ls_repaired = _repair_linesearch_interval(
                                    _energy_at_alpha,
                                    alpha_lo,
                                    alpha_hi,
                                    center=0.0,
                                    center_value=fx_old,
                                    tol=linesearch_tol,
                                )
                                ls_evals += ls_repair_evals
                                alpha, ls_eval_local = golden_section_search(
                                    _energy_at_alpha, alpha_lo, alpha_hi, linesearch_tol
                                )
                                ls_evals += ls_eval_local
                                newval = _energy_at_alpha(alpha)
                                pred_reduction = -(
                                    alpha * step_linear + 0.5 * (alpha ** 2) * step_curv
                                )
                                actual_reduction = (
                                    fx_old - newval if np.isfinite(newval) else -np.inf
                                )
                            else:
                                alpha = 0.0
                                newval = np.inf
                                pred_reduction = 0.0
                                actual_reduction = -np.inf
                        else:
                            alpha = 1.0
                            pred_reduction = -(step_linear + 0.5 * step_curv)
                            newval = _energy_at_alpha(1.0)
                            actual_reduction = (
                                fx_old - newval if np.isfinite(newval) else -np.inf
                            )
                    else:
                        trust_qp = 0.0
                        alpha = 0.0
                        pred_reduction = 0.0
                        actual_reduction = -np.inf
                        newval = np.inf

                    rho = np.nan
                    if (
                        np.isfinite(pred_reduction)
                        and pred_reduction > 0.0
                        and np.isfinite(actual_reduction)
                    ):
                        rho = actual_reduction / pred_reduction

                    step_size = abs(alpha) * snod_norm if np.isfinite(snod_norm) else 0.0

                    if (
                        np.isfinite(newval)
                        and np.isfinite(rho)
                        and rho >= trust_eta_shrink
                        and actual_reduction > 0.0
                    ):
                        fx = newval
                        dE = actual_reduction
                        step_norm = step_size
                        step_rel = step_norm / max(1.0, x_prev_norm)
                        t_update0 = time.perf_counter()
                        x_trial.copy(x)
                        project_fn(x)
                        ghost_update_fn(x)
                        t_update = time.perf_counter() - t_update0
                        accepted_step = True

                        if rho >= trust_eta_expand and step_norm >= 0.9 * trial_radius:
                            trust_radius = min(trust_radius_max, trial_radius * trust_expand)
                        else:
                            trust_radius = trial_radius
                        break

                    trust_rejects = attempt_idx + 1
                    trust_radius = max(trust_radius_min, trial_radius * trust_shrink)

                    if attempt_idx + 1 >= max_trust_attempts:
                        x_prev.copy(x)
                        project_fn(x)
                        ghost_update_fn(x)
                        alpha = 0.0
                        dE = 0.0
                        step_norm = 0.0
                        step_rel = 0.0
                        t_update = 0.0
                        terminate_after_iter = True
                        step_tol = max(tolx_abs, tolx_rel * max(1.0, x_prev_norm))
                        small_step = np.isfinite(step_size) and step_size <= step_tol
                        small_pred = np.isfinite(pred_reduction) and abs(pred_reduction) <= max(
                            tolf, 1e-16
                        )
                        if small_step or small_pred:
                            if require_all_convergence and normg >= grad_target:
                                terminate_message = (
                                    "Trust-region radius exhausted before full convergence"
                                )
                            else:
                                terminate_message = "Trust-region step converged"
                        else:
                            terminate_message = (
                                f"Trust-region rejected all candidate steps at Newton iteration {nit}"
                            )
                        break
            else:
                z1 = x.duplicate()
                z2 = x.duplicate()
                hz1 = x.duplicate()
                hz2 = x.duplicate()
                sg = x.duplicate()
                has_z2 = False
                rhs_red = None
                mm = None
                rhs_g = np.nan
                curv_g = np.nan

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

                if normg > 1e-20:
                    g.copy(sg)
                    sg.scale(1.0 / normg)
                    hessian_matvec_fn(x, sg, hz2)
                    ghost_update_fn(hz2)
                    rhs_g = float(g.dot(sg))
                    curv_g = float(sg.dot(hz2))

                def _build_trust_step(delta_local):
                    p.set(0.0)
                    step_norm_local = 0.0
                    q_local = np.inf
                    step_linear_local = np.nan
                    step_curv_local = np.nan
                    used_grad_local = False

                    if has_z2:
                        st, q2d = _solve_trust_2d(rhs_red, mm, delta_local)
                        p.axpy(float(st[0]), z1)
                        p.axpy(float(st[1]), z2)
                        step_norm_local = float(np.linalg.norm(st))
                        step_linear_local = float(rhs_red.dot(st))
                        step_curv_local = float(st.dot(mm.dot(st)))
                        q_local = float(q2d)
                    else:
                        rhs1 = float(g.dot(z1))
                        curv1 = float(z1.dot(hz1))
                        t1 = _solve_trust_1d(rhs1, curv1, delta_local)
                        z1.copy(p)
                        p.scale(t1)
                        step_norm_local = abs(t1)
                        step_linear_local = float(rhs1 * t1)
                        step_curv_local = float(curv1 * t1 * t1)
                        q_local = float(step_linear_local + 0.5 * step_curv_local)

                    if normg > 1e-20:
                        tg = _solve_trust_1d(rhs_g, curv_g, delta_local)
                        qp_grad = float(rhs_g * tg + 0.5 * curv_g * tg * tg)
                        if qp_grad <= q_local:
                            g.copy(sg)
                            sg.scale(1.0 / normg)
                            sg.scale(tg)
                            sg.copy(p)
                            step_norm_local = abs(tg)
                            step_linear_local = float(rhs_g * tg)
                            step_curv_local = float(curv_g * tg * tg)
                            q_local = qp_grad
                            used_grad_local = True

                    return (
                        step_norm_local,
                        q_local,
                        step_linear_local,
                        step_curv_local,
                        used_grad_local,
                    )

                max_trust_attempts = trust_max_reject + 1
                for attempt_idx in range(max_trust_attempts):
                    trial_radius = float(min(max(trust_radius, trust_radius_min), trust_radius_max))
                    (
                        snod_norm,
                        trust_qp,
                        step_linear,
                        step_curv,
                        used_gradient_fallback,
                    ) = _build_trust_step(trial_radius)

                    alpha_lo = float(ls_a)
                    alpha_hi = float(ls_b)
                    if snod_norm > 1e-20:
                        alpha_cap = trial_radius / snod_norm
                        alpha_lo = max(alpha_lo, -alpha_cap)
                        alpha_hi = min(alpha_hi, alpha_cap)

                    if (
                        not np.isfinite(snod_norm)
                        or snod_norm <= 1e-20
                        or not np.isfinite(alpha_lo)
                        or not np.isfinite(alpha_hi)
                        or alpha_hi <= alpha_lo + 1e-14
                    ):
                        newval = np.inf
                        alpha = 0.0
                        pred_reduction = 0.0
                        actual_reduction = -np.inf
                    else:
                        alpha_lo, alpha_hi, ls_repair_evals, ls_repaired = _repair_linesearch_interval(
                            _energy_at_alpha,
                            alpha_lo,
                            alpha_hi,
                            center=0.0,
                            center_value=fx_old,
                            tol=linesearch_tol,
                        )
                        ls_evals += ls_repair_evals
                        alpha, ls_eval_local = golden_section_search(
                            _energy_at_alpha, alpha_lo, alpha_hi, linesearch_tol
                        )
                        ls_evals += ls_eval_local
                        newval = _energy_at_alpha(alpha)
                        pred_reduction = -(
                            alpha * step_linear + 0.5 * (alpha ** 2) * step_curv
                        )
                        actual_reduction = fx_old - newval if np.isfinite(newval) else -np.inf

                    rho = np.nan
                    if (
                        np.isfinite(pred_reduction)
                        and pred_reduction > 0.0
                        and np.isfinite(actual_reduction)
                    ):
                        rho = actual_reduction / pred_reduction

                    step_size = abs(alpha) * snod_norm if np.isfinite(snod_norm) else 0.0

                    if (
                        np.isfinite(newval)
                        and np.isfinite(rho)
                        and rho >= trust_eta_shrink
                        and actual_reduction > 0.0
                    ):
                        fx = newval
                        dE = actual_reduction
                        step_norm = step_size
                        step_rel = step_norm / max(1.0, x_prev_norm)
                        t_update0 = time.perf_counter()
                        x_trial.copy(x)
                        project_fn(x)
                        ghost_update_fn(x)
                        t_update = time.perf_counter() - t_update0
                        accepted_step = True

                        if rho >= trust_eta_expand and step_norm >= 0.9 * trial_radius:
                            trust_radius = min(trust_radius_max, trial_radius * trust_expand)
                        else:
                            trust_radius = trial_radius
                        break

                    trust_rejects = attempt_idx + 1
                    trust_radius = max(trust_radius_min, trial_radius * trust_shrink)

                    if attempt_idx + 1 >= max_trust_attempts:
                        x_prev.copy(x)
                        project_fn(x)
                        ghost_update_fn(x)
                        alpha = 0.0
                        dE = 0.0
                        step_norm = 0.0
                        step_rel = 0.0
                        t_update = 0.0
                        terminate_after_iter = True
                        step_tol = max(tolx_abs, tolx_rel * max(1.0, x_prev_norm))
                        small_step = np.isfinite(step_size) and step_size <= step_tol
                        small_pred = np.isfinite(pred_reduction) and abs(pred_reduction) <= max(
                            tolf, 1e-16
                        )
                        if small_step or small_pred:
                            if require_all_convergence and normg >= grad_target:
                                terminate_message = (
                                    "Trust-region radius exhausted before full convergence"
                                )
                            else:
                                terminate_message = "Trust-region step converged"
                        else:
                            terminate_message = (
                                f"Trust-region rejected all candidate steps at Newton iteration {nit}"
                            )
                        break

                z1.destroy()
                z2.destroy()
                hz1.destroy()
                hz2.destroy()
                sg.destroy()

        else:
            h.copy(p)
            pnorm = p.norm(PETSc.NormType.NORM_2)
            if pnorm > 1e-20:
                ls_a_eff, ls_b_eff, ls_repair_evals, ls_repaired = _repair_linesearch_interval(
                    _energy_at_alpha,
                    ls_a,
                    ls_b,
                    center=0.0,
                    center_value=fx_old,
                    tol=linesearch_tol,
                )
                ls_evals += ls_repair_evals
                alpha, ls_eval_local = golden_section_search(
                    _energy_at_alpha, ls_a_eff, ls_b_eff, linesearch_tol
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
                    t_update = 0.0
            else:
                x_prev.copy(x)
                project_fn(x)
                ghost_update_fn(x)
                alpha = 0.0
                t_update = 0.0

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
            history.append(
                {
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
                    "pred_reduction": float(pred_reduction),
                    "actual_reduction": float(actual_reduction),
                    "trust_rejects": int(trust_rejects),
                    "t_grad": float(t_grad),
                    "t_hess": float(t_hess),
                    "t_ls": float(t_ls),
                    "t_update": float(t_update),
                    "t_iter": float(t_iter_total),
                    "trust_radius": float(trust_radius),
                    "trust_ratio": float(rho),
                    "trust_qp": float(trust_qp),
                }
            )

        if (
            step_time_limit_s is not None
            and np.isfinite(step_time_limit_s)
            and (time.perf_counter() - start) > float(step_time_limit_s)
        ):
            message = (
                f"Step time limit exceeded ({time.perf_counter() - start:.3f}s > "
                f"{float(step_time_limit_s):.3f}s)"
            )
            break

        if terminate_after_iter:
            message = terminate_message
            break

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

    g.destroy()
    h.destroy()
    x_trial.destroy()
    x_prev.destroy()
    p.destroy()

    return {
        "x": x,
        "fun": fx,
        "nit": nit,
        "time": runtime,
        "message": message,
        "history": history,
        "success": "converged" in message.lower(),
    }

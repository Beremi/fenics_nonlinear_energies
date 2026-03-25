import time

import numpy as np


def _trial_value(f, x, direction, alpha):
    value = f(x + alpha * direction)
    return float(value) if np.isfinite(value) else np.inf


def _repair_interval(
    f,
    a,
    b,
    x,
    direction,
    tol,
    center=0.0,
    center_value=None,
    max_bisect=60,
):
    center = float(center)
    a = float(a)
    b = float(b)
    tol = max(float(tol), 1e-12)
    n_evals = 0
    repaired = False

    if center_value is None:
        center_value = _trial_value(f, x, direction, center)
        n_evals += 1
    if not np.isfinite(center_value):
        return a, b, n_evals, repaired

    def _repair_side(bound, side_sign):
        nonlocal n_evals, repaired
        if side_sign < 0 and not (bound < center):
            return bound
        if side_sign > 0 and not (bound > center):
            return bound

        f_bound = _trial_value(f, x, direction, bound)
        n_evals += 1
        if np.isfinite(f_bound):
            return bound

        repaired = True
        finite = center
        nonfinite = bound
        for _ in range(max_bisect):
            mid = 0.5 * (finite + nonfinite)
            f_mid = _trial_value(f, x, direction, mid)
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


def golden_section_search(f, a, b, tol):
    gamma = 0.5 + np.sqrt(5) / 2

    an, bn = float(a), float(b)
    fan = f(an)
    fbn = f(bn)
    dn = (bn - an) / gamma + an
    cn = an + bn - dn

    fcn = f(cn)
    fdn = f(dn)
    n_evals = 4
    best_alpha = an
    best_value = fan
    for alpha, value in ((bn, fbn), (cn, fcn), (dn, fdn)):
        if value < best_value:
            best_alpha = alpha
            best_value = value

    while bn - an > tol:
        if fcn < fdn:
            bn = dn
            dn, cn = cn, an + bn - cn
            fdn, fcn = fcn, f(cn)
            if fdn < best_value:
                best_alpha = dn
                best_value = fdn
            if fcn < best_value:
                best_alpha = cn
                best_value = fcn
        else:
            an = cn
            cn, dn = dn, an + bn - dn
            fcn, fdn = fdn, f(dn)
            if fcn < best_value:
                best_alpha = cn
                best_value = fcn
            if fdn < best_value:
                best_alpha = dn
                best_value = fdn
        n_evals += 1

    midpoint = 0.5 * (an + bn)
    fmid = f(midpoint)
    n_evals += 1
    if fmid < best_value:
        best_alpha = midpoint
    return float(best_alpha), n_evals


def newton(
    f,
    df,
    ddf,
    x0,
    tolf=1e-6,
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
    trust_region=False,
    trust_radius_init=1.0,
    trust_radius_min=1e-8,
    trust_radius_max=1e6,
    trust_shrink=0.5,
    trust_expand=1.5,
    trust_eta_shrink=0.05,
    trust_eta_expand=0.75,
    trust_max_reject=6,
    trust_subproblem_line_search=False,
    save_history=False,
    save_linear_timing=False,
):
    x = np.asarray(x0, dtype=np.float64).copy()
    fx = float(f(x))
    nit = 0
    message = "Maximum number of iterations reached"
    start = time.perf_counter()

    history = []
    linear_timing = []
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
        g = np.asarray(df(x), dtype=np.float64)
        normg = float(np.linalg.norm(g))
        t_grad = time.perf_counter() - t0

        if fail_on_nonfinite and not np.isfinite(normg):
            message = f"Non-finite gradient norm at Newton iteration {nit + 1}"
            break

        if initial_grad_norm is None:
            initial_grad_norm = normg

        grad_target = float(tolg)
        if tolg_rel > 0.0 and np.isfinite(initial_grad_norm):
            grad_target = max(float(tolg), float(tolg_rel) * initial_grad_norm)

        if (not require_all_convergence) and normg < grad_target:
            message = "Gradient norm converged"
            break

        nit += 1
        x_prev = x.copy()
        x_prev_norm = float(np.linalg.norm(x_prev))
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
        post_grad_norm = np.nan

        assemble_time = 0.0
        pc_setup_time = 0.0
        solve_time = 0.0
        ksp_its = 0
        hit_ksp_max_it = False
        solve_reason = ""
        hnorm = np.nan

        def _energy_at_alpha(alpha_local):
            return _trial_value(f, x_prev, p, alpha_local)

        ls_a, ls_b = linesearch_interval
        if ls_b <= ls_a:
            ls_a, ls_b = -0.5, 2.0

        t_hess_start = time.perf_counter()
        hess_solver = ddf(x_prev)
        assemble_time = float(getattr(hess_solver, "assemble_time", 0.0))
        pc_setup_time = float(getattr(hess_solver, "pc_setup_time", 0.0))
        t_hess_build = time.perf_counter() - t_hess_start
        p = np.zeros_like(x_prev)

        if trust_region and hasattr(hess_solver, "trust_subproblem_solve"):
            trust_radius = min(max(trust_radius, trust_radius_min), trust_radius_max)
            max_attempts = max(1, int(trust_max_reject) + 1)
            for attempt_idx in range(max_attempts):
                trial_radius = float(min(max(trust_radius, trust_radius_min), trust_radius_max))
                t_solve0 = time.perf_counter()
                p = np.asarray(
                    hess_solver.trust_subproblem_solve(g, trial_radius),
                    dtype=np.float64,
                )
                solve_time += time.perf_counter() - t_solve0
                solve_info = getattr(hess_solver, "last_solve_info", {})
                ksp_its = int(solve_info.get("ksp_its", 0))
                solve_time = float(solve_info.get("solve_time", solve_time))
                hit_ksp_max_it = bool(solve_info.get("hit_maxit", False))
                solve_reason = str(solve_info.get("reason", ""))

                snorm = float(np.linalg.norm(p))
                if fail_on_nonfinite and not np.isfinite(snorm):
                    message = f"Non-finite trust-region step norm at Newton iteration {nit}"
                    terminate_after_iter = True
                    terminate_message = message
                    break

                if np.isfinite(snorm) and snorm > 1e-20:
                    Hp = np.asarray(hess_solver.H @ p, dtype=np.float64)
                    step_linear = float(np.dot(g, p))
                    step_curv = float(np.dot(p, Hp))
                    trust_qp = float(step_linear + 0.5 * step_curv)
                    if trust_subproblem_line_search:
                        alpha_lo = float(ls_a)
                        alpha_hi = float(ls_b)
                        alpha_cap = trial_radius / snorm
                        alpha_lo = max(alpha_lo, -alpha_cap)
                        alpha_hi = min(alpha_hi, alpha_cap)
                        if (
                            np.isfinite(alpha_lo)
                            and np.isfinite(alpha_hi)
                            and alpha_hi > alpha_lo + 1e-14
                        ):
                            alpha_lo, alpha_hi, repair_evals, ls_repaired = _repair_interval(
                                f,
                                alpha_lo,
                                alpha_hi,
                                x_prev,
                                p,
                                linesearch_tol,
                                center=0.0,
                                center_value=fx_old,
                            )
                            ls_evals += repair_evals
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
                        newval = _energy_at_alpha(1.0)
                        pred_reduction = -(step_linear + 0.5 * step_curv)
                        actual_reduction = (
                            fx_old - newval if np.isfinite(newval) else -np.inf
                        )
                else:
                    alpha = 0.0
                    newval = np.inf
                    pred_reduction = 0.0
                    actual_reduction = -np.inf

                rho = np.nan
                if (
                    np.isfinite(pred_reduction)
                    and pred_reduction > 0.0
                    and np.isfinite(actual_reduction)
                ):
                    rho = actual_reduction / pred_reduction

                step_size = abs(alpha) * snorm if np.isfinite(snorm) else 0.0
                if (
                    np.isfinite(newval)
                    and np.isfinite(rho)
                    and rho >= trust_eta_shrink
                    and actual_reduction > 0.0
                ):
                    x = x_prev + alpha * p
                    fx = float(newval)
                    dE = float(actual_reduction)
                    step_norm = float(step_size)
                    step_rel = step_norm / max(1.0, x_prev_norm)
                    accepted_step = True
                    if rho >= trust_eta_expand and step_norm >= 0.9 * trial_radius:
                        trust_radius = min(trust_radius_max, trial_radius * trust_expand)
                    else:
                        trust_radius = trial_radius
                    break

                trust_rejects = attempt_idx + 1
                trust_radius = max(trust_radius_min, trial_radius * trust_shrink)
                if attempt_idx + 1 >= max_attempts:
                    x = x_prev.copy()
                    terminate_after_iter = True
                    alpha = 0.0
                    step_norm = 0.0
                    step_rel = 0.0
                    dE = 0.0
                    step_tol = max(tolx_abs, tolx_rel * max(1.0, x_prev_norm))
                    small_step = np.isfinite(step_size) and step_size <= step_tol
                    small_pred = np.isfinite(pred_reduction) and abs(pred_reduction) <= max(
                        tolf, 1e-16
                    )
                    if small_step or small_pred:
                        if require_all_convergence and normg >= grad_target:
                            terminate_message = "Trust-region radius exhausted before full convergence"
                        else:
                            terminate_message = "Trust-region step converged"
                    else:
                        terminate_message = (
                            f"Trust-region rejected all candidate steps at Newton iteration {nit}"
                        )
                    break

        else:
            t_solve0 = time.perf_counter()
            p = np.asarray(hess_solver.solve(-g), dtype=np.float64)
            solve_time = time.perf_counter() - t_solve0
            solve_info = getattr(hess_solver, "last_solve_info", {})
            ksp_its = int(solve_info.get("ksp_its", 0))
            solve_time = float(solve_info.get("solve_time", solve_time))
            hit_ksp_max_it = bool(solve_info.get("hit_maxit", False))
            solve_reason = str(solve_info.get("reason", ""))
            hnorm = float(np.linalg.norm(p))

            if fail_on_nonfinite and not np.isfinite(hnorm):
                message = f"Non-finite Newton direction norm at Newton iteration {nit}"
                terminate_after_iter = True
                terminate_message = message
            elif hnorm > 1e-20:
                ls_a_eff, ls_b_eff, repair_evals, ls_repaired = _repair_interval(
                    f,
                    ls_a,
                    ls_b,
                    x_prev,
                    p,
                    linesearch_tol,
                    center=0.0,
                    center_value=fx_old,
                )
                ls_evals += repair_evals
                alpha, ls_eval_local = golden_section_search(
                    _energy_at_alpha, ls_a_eff, ls_b_eff, linesearch_tol
                )
                ls_evals += ls_eval_local
                fx_trial = _energy_at_alpha(alpha)
                if np.isfinite(fx_trial) and fx_trial < fx_old:
                    x = x_prev + alpha * p
                    fx = float(fx_trial)
                    dE = float(fx_old - fx)
                    step_norm = abs(alpha) * hnorm
                    step_rel = step_norm / max(1.0, x_prev_norm)
                    accepted_step = True
                else:
                    x = x_prev.copy()
                    terminate_after_iter = True
                    terminate_message = (
                        f"Line search failed to find decreasing step at Newton iteration {nit}"
                    )
            else:
                x = x_prev.copy()
                terminate_after_iter = True
                terminate_message = f"Zero Newton direction at Newton iteration {nit}"

        t_hess = max(t_hess_build, assemble_time + pc_setup_time + solve_time)
        t_ls = time.perf_counter() - t_iter_start - t_grad - t_hess
        if t_ls < 0.0:
            t_ls = 0.0
        t_iter_total = time.perf_counter() - t_iter_start

        converged_energy = accepted_step and np.isfinite(dE) and abs(dE) < tolf
        converged_step = accepted_step and (step_norm < tolx_abs or step_rel < tolx_rel)
        if require_all_convergence and converged_energy and converged_step:
            g_post = np.asarray(df(x), dtype=np.float64)
            post_grad_norm = float(np.linalg.norm(g_post))
            if post_grad_norm < grad_target:
                message = "Converged (energy, step, gradient)"
                if save_history:
                    pass
                if save_linear_timing:
                    pass
                if verbose:
                    print(
                        f"it={nit}, f={fx:.5f}, dE={dE:.5e}, ||g||={normg:.5e}, "
                        f"alpha={alpha:.5e}, ksp_its={ksp_its}, rho={rho:.5e}"
                    )
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
                            "step_norm": float(step_norm),
                            "step_rel": float(step_rel),
                            "ksp_its": int(ksp_its),
                            "trust_ratio": float(rho),
                            "trust_radius": float(trust_radius) if trust_region else np.nan,
                            "trust_rejects": int(trust_rejects),
                            "used_gradient_fallback": bool(used_gradient_fallback),
                            "ls_evals": int(ls_evals),
                            "ls_repaired": bool(ls_repaired),
                            "accepted_step": bool(accepted_step),
                            "t_grad": float(t_grad),
                            "t_hess": float(t_hess),
                            "t_ls": float(t_ls),
                            "t_iter_total": float(t_iter_total),
                            "message": message,
                            "hit_ksp_max_it": bool(hit_ksp_max_it),
                            "solve_reason": solve_reason,
                        }
                    )
                if save_linear_timing:
                    linear_timing.append(
                        {
                            "it": int(nit),
                            "assemble_time": float(assemble_time),
                            "pc_setup_time": float(pc_setup_time),
                            "solve_time": float(solve_time),
                            "linear_total_time": float(assemble_time + pc_setup_time + solve_time),
                            "ksp_its": int(ksp_its),
                            "hit_ksp_max_it": bool(hit_ksp_max_it),
                            "solve_reason": solve_reason,
                        }
                    )
                break

        if not require_all_convergence and converged_energy:
            message = "Stopping condition for f is satisfied"
            terminate_after_iter = True
            terminate_message = message
        elif terminate_after_iter:
            message = terminate_message

        if verbose:
            print(
                f"it={nit}, f={fx:.5f}, dE={dE:.5e}, ||g||={normg:.5e}, "
                f"alpha={alpha:.5e}, ksp_its={ksp_its}, rho={rho:.5e}"
            )

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
                    "step_norm": float(step_norm),
                    "step_rel": float(step_rel),
                    "ksp_its": int(ksp_its),
                    "trust_ratio": float(rho),
                    "trust_radius": float(trust_radius) if trust_region else np.nan,
                    "trust_rejects": int(trust_rejects),
                    "used_gradient_fallback": bool(used_gradient_fallback),
                    "ls_evals": int(ls_evals),
                    "ls_repaired": bool(ls_repaired),
                    "accepted_step": bool(accepted_step),
                    "t_grad": float(t_grad),
                    "t_hess": float(t_hess),
                    "t_ls": float(t_ls),
                    "t_iter_total": float(t_iter_total),
                    "message": message if terminate_after_iter else "",
                    "hit_ksp_max_it": bool(hit_ksp_max_it),
                    "solve_reason": solve_reason,
                }
            )

        if save_linear_timing:
            linear_timing.append(
                {
                    "it": int(nit),
                    "assemble_time": float(assemble_time),
                    "pc_setup_time": float(pc_setup_time),
                    "solve_time": float(solve_time),
                    "linear_total_time": float(assemble_time + pc_setup_time + solve_time),
                    "ksp_its": int(ksp_its),
                    "hit_ksp_max_it": bool(hit_ksp_max_it),
                    "solve_reason": solve_reason,
                }
            )

        if terminate_after_iter:
            break

    runtime = time.perf_counter() - start
    result = {
        "x": x,
        "fun": float(fx),
        "nit": int(nit),
        "time": float(runtime),
        "message": message,
    }
    if save_history:
        result["history"] = history
    if save_linear_timing:
        result["linear_timing"] = linear_timing
    return result


def gradient_descent(
    f,
    df,
    x0,
    *,
    tolf=1e-6,
    tolg=1e-3,
    tolg_rel=0.0,
    linesearch_tol=1e-3,
    linesearch_interval=(0.0, 2.0),
    line_search="armijo",
    adaptive_alpha0=1.0,
    adaptive_window_scale=2.0,
    adaptive_nonnegative=False,
    armijo_alpha0=1.0,
    armijo_c1=1e-4,
    armijo_shrink=0.5,
    armijo_max_ls=40,
    maxit=200,
    tolx_rel=1e-6,
    tolx_abs=1e-10,
    require_all_convergence=False,
    fail_on_nonfinite=True,
    verbose=False,
    save_history=False,
):
    x = np.asarray(x0, dtype=np.float64).copy()
    fx = float(f(x))
    nit = 0
    message = "Maximum number of iterations reached"
    start = time.perf_counter()

    history = []
    initial_grad_norm = None
    last_alpha_abs = max(abs(float(adaptive_alpha0)), linesearch_tol)

    for _ in range(maxit):
        t_iter_start = time.perf_counter()
        if fail_on_nonfinite and not np.isfinite(fx):
            message = f"Non-finite energy before gradient iteration {nit + 1}"
            break

        t0 = time.perf_counter()
        g = np.asarray(df(x), dtype=np.float64)
        normg = float(np.linalg.norm(g))
        t_grad = time.perf_counter() - t0

        if fail_on_nonfinite and not np.isfinite(normg):
            message = f"Non-finite gradient norm at gradient iteration {nit + 1}"
            break

        if initial_grad_norm is None:
            initial_grad_norm = normg

        grad_target = float(tolg)
        if tolg_rel > 0.0 and np.isfinite(initial_grad_norm):
            grad_target = max(float(tolg), float(tolg_rel) * initial_grad_norm)

        if (not require_all_convergence) and normg < grad_target:
            message = "Gradient norm converged"
            break

        nit += 1
        x_prev = x.copy()
        x_prev_norm = float(np.linalg.norm(x_prev))
        fx_old = fx
        direction = -g
        dnorm = float(np.linalg.norm(direction))
        if fail_on_nonfinite and not np.isfinite(dnorm):
            message = f"Non-finite descent direction norm at gradient iteration {nit}"
            break
        if dnorm <= 1e-20:
            message = f"Zero descent direction at gradient iteration {nit}"
            break

        alpha = 0.0
        ls_evals = 0
        ls_repaired = False
        dE = 0.0
        step_norm = 0.0
        step_rel = 0.0
        accepted_step = False

        def _energy_at_alpha(alpha_local):
            return _trial_value(f, x_prev, direction, alpha_local)

        t_ls0 = time.perf_counter()
        if line_search == "armijo":
            alpha_trial = float(max(armijo_alpha0, 1e-12))
            directional_derivative = float(np.dot(g, direction))
            for _ls_it in range(max(1, int(armijo_max_ls))):
                trial_value = _energy_at_alpha(alpha_trial)
                ls_evals += 1
                if np.isfinite(trial_value) and trial_value <= fx_old + armijo_c1 * alpha_trial * directional_derivative:
                    alpha = float(alpha_trial)
                    fx = float(trial_value)
                    accepted_step = True
                    break
                alpha_trial *= float(armijo_shrink)
            if not accepted_step:
                message = f"Armijo line search failed at gradient iteration {nit}"
        else:
            if line_search == "golden_adaptive":
                horizon = max(float(last_alpha_abs), float(linesearch_tol))
                if adaptive_nonnegative:
                    ls_a = 0.0
                    ls_b = float(adaptive_window_scale) * horizon
                else:
                    ls_a = -float(adaptive_window_scale) * horizon
                    ls_b = float(adaptive_window_scale) * horizon
            elif line_search == "golden_fixed":
                ls_a, ls_b = linesearch_interval
            else:
                raise ValueError(f"Unknown gradient line search mode: {line_search}")

            if ls_b <= ls_a:
                ls_a, ls_b = 0.0, 2.0

            ls_a_eff, ls_b_eff, repair_evals, ls_repaired = _repair_interval(
                f,
                ls_a,
                ls_b,
                x_prev,
                direction,
                linesearch_tol,
                center=0.0,
                center_value=fx_old,
            )
            ls_evals += repair_evals
            alpha, ls_eval_local = golden_section_search(
                _energy_at_alpha, ls_a_eff, ls_b_eff, linesearch_tol
            )
            ls_evals += ls_eval_local
            fx_trial = _energy_at_alpha(alpha)
            if np.isfinite(fx_trial) and fx_trial < fx_old:
                fx = float(fx_trial)
                accepted_step = True
            else:
                message = f"Golden-section line search failed at gradient iteration {nit}"

        t_ls = time.perf_counter() - t_ls0

        if accepted_step:
            x = x_prev + alpha * direction
            dE = float(fx_old - fx)
            step_norm = float(abs(alpha) * dnorm)
            step_rel = float(step_norm / max(1.0, x_prev_norm))
            last_alpha_abs = max(abs(float(alpha)), float(linesearch_tol))
        else:
            x = x_prev.copy()

        t_iter_total = time.perf_counter() - t_iter_start

        converged_energy = accepted_step and np.isfinite(dE) and abs(dE) < tolf
        converged_step = accepted_step and (step_norm < tolx_abs or step_rel < tolx_rel)

        if require_all_convergence and accepted_step and converged_energy and converged_step:
            g_post = np.asarray(df(x), dtype=np.float64)
            if float(np.linalg.norm(g_post)) < grad_target:
                message = "Converged (energy, step, gradient)"
                if save_history:
                    history.append(
                        {
                            "it": int(nit),
                            "energy": float(fx),
                            "dE": float(dE),
                            "grad_norm": float(normg),
                            "grad_target": float(grad_target),
                            "alpha": float(alpha),
                            "step_norm": float(step_norm),
                            "step_rel": float(step_rel),
                            "ls_evals": int(ls_evals),
                            "ls_repaired": bool(ls_repaired),
                            "accepted_step": bool(accepted_step),
                            "t_grad": float(t_grad),
                            "t_ls": float(t_ls),
                            "t_iter_total": float(t_iter_total),
                            "message": message,
                            "line_search": str(line_search),
                            "adaptive_nonnegative": bool(adaptive_nonnegative),
                        }
                    )
                break

        terminate_after_iter = False
        if not require_all_convergence and converged_energy:
            message = "Stopping condition for f is satisfied"
            terminate_after_iter = True
        elif accepted_step and converged_step:
            message = "Stopping condition for step size is satisfied"
            terminate_after_iter = True
        elif not accepted_step:
            terminate_after_iter = True

        if verbose:
            print(
                f"gd it={nit}, f={fx:.5f}, dE={dE:.5e}, ||g||={normg:.5e}, "
                f"alpha={alpha:.5e}, ls={line_search}"
            )

        if save_history:
            history.append(
                {
                    "it": int(nit),
                    "energy": float(fx),
                    "dE": float(dE),
                    "grad_norm": float(normg),
                    "grad_target": float(grad_target),
                    "alpha": float(alpha),
                    "step_norm": float(step_norm),
                    "step_rel": float(step_rel),
                    "ls_evals": int(ls_evals),
                    "ls_repaired": bool(ls_repaired),
                    "accepted_step": bool(accepted_step),
                    "t_grad": float(t_grad),
                    "t_ls": float(t_ls),
                    "t_iter_total": float(t_iter_total),
                    "message": message if terminate_after_iter else "",
                    "line_search": str(line_search),
                    "adaptive_nonnegative": bool(adaptive_nonnegative),
                }
            )

        if terminate_after_iter:
            break

    runtime = time.perf_counter() - start
    result = {
        "x": x,
        "fun": float(fx),
        "nit": int(nit),
        "time": float(runtime),
        "message": message,
        "last_alpha_abs": float(last_alpha_abs),
    }
    if save_history:
        result["history"] = history
    return result

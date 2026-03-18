# Trust-Region + Line-Search Algorithm

Date: 2026-03-06

This note documents the current hybrid trust-region / line-search Newton
algorithm implemented in `src/core/petsc/minimizers.py`.

It is a Newton method with:

- an optional trust-region model step in a reduced subspace,
- optional gradient fallback inside the same trust radius,
- a golden-section line search along the chosen trust-region direction,
- standard trust-region acceptance and radius updates based on
  `rho = actual_reduction / predicted_reduction`.

## Inputs

Main nonlinear callbacks:

- `energy_fn(x)` returns the scalar objective value
- `gradient_fn(x, g)` assembles the gradient into `g`
- `hessian_solve_fn(x, rhs, sol)` approximately solves `H(x) sol = rhs`
- `hessian_matvec_fn(x, v, Hv)` applies the Hessian to a vector

Main solver controls:

- `tolf`, `tolg`, `tolg_rel`, `tolx_abs`, `tolx_rel`
- `linesearch_interval = [a, b]`
- `linesearch_tol`
- `trust_region`
- `trust_radius_init`, `trust_radius_min`, `trust_radius_max`
- `trust_shrink`, `trust_expand`
- `trust_eta_shrink`, `trust_eta_expand`
- `trust_max_reject`

## High-level idea

Without trust region:

- compute the Newton direction,
- line-search along it,
- accept only if the energy decreases.

With trust region:

- build a trust-region step inside a reduced model,
- optionally compare it with a pure gradient trust-region step,
- line-search along that step while staying inside the trust ball,
- compute predicted and actual reduction,
- accept or reject using `rho`,
- shrink or expand the radius using the exposed trust parameters.

## Pseudocode

```text
given x
fx := energy_fn(x)
Delta := clamp(trust_radius_init, trust_radius_min, trust_radius_max)

for k = 1 .. maxit:
    g := gradient(x)
    ||g|| := norm(g)

    grad_target := max(tolg, tolg_rel * ||g_0||)    if tolg_rel > 0
                   tolg                              otherwise

    if require_all_convergence is false and ||g|| < grad_target:
        stop with gradient convergence

    h := approximate Newton direction from H(x) h = -g

    save x_prev := x
    fx_prev := fx

    if trust_region is false:
        p := h
        choose alpha by golden-section search on
            phi(alpha) = energy_fn(project(x_prev + alpha p))
            over [linesearch_a, linesearch_b]

        x_trial := project(x_prev + alpha p)
        fx_trial := energy_fn(x_trial)

        if fx_trial < fx_prev:
            accept:
                x := x_trial
                fx := fx_trial
                dE := fx_prev - fx
                step_norm := |alpha| * ||p||
        else:
            reject:
                restore x := x_prev
                dE := 0
                step_norm := 0

    else:
        build a reduced trust-region model around x:

            m(s) = fx_prev + g^T s + 0.5 s^T H s

        construct subspace basis:
            z1 := normalized Newton direction h
                  or normalized steepest descent direction -g if h is unusable
            z2 := orthonormalized gradient direction if available

        precompute reduced Hessian data:
            rhs_red := [g^T z1, g^T z2]
            M_red   := [[z1^T H z1, z1^T H z2],
                        [z2^T H z1, z2^T H z2]]

        also precompute pure gradient trust-model data:
            sg := normalized gradient direction
            rhs_g := g^T sg
            curv_g := sg^T H sg

        accepted := false

        for reject_iter = 0 .. trust_max_reject:
            Delta_trial := clamp(Delta, trust_radius_min, trust_radius_max)

            solve reduced trust-region subproblem:
                p_tr := argmin g^T p + 0.5 p^T H p
                        subject to ||p|| <= Delta_trial
                        in span{z1, z2}  (or 1D if z2 unavailable)

            solve pure-gradient trust step:
                p_g := argmin g^T p + 0.5 p^T H p
                       subject to p = t sg, |t| <= Delta_trial

            choose the better model step:
                p := whichever gives smaller quadratic model value

            let
                lin  := g^T p
                curv := p^T H p
                ||p|| := norm(p)

            restrict the line-search interval so the final step stays in the
            trust ball:
                alpha_min := max(linesearch_a, -Delta_trial / ||p||)
                alpha_max := min(linesearch_b,  Delta_trial / ||p||)

            choose alpha by golden-section search on
                phi(alpha) = energy_fn(project(x_prev + alpha p))
                over [alpha_min, alpha_max]

            s := alpha p
            x_trial := project(x_prev + s)
            fx_trial := energy_fn(x_trial)

            predicted_reduction :=
                -(alpha * lin + 0.5 * alpha^2 * curv)

            actual_reduction :=
                fx_prev - fx_trial

            if predicted_reduction > 0:
                rho := actual_reduction / predicted_reduction
            else:
                rho := invalid

            if
                fx_trial is finite and
                actual_reduction > 0 and
                rho is finite and
                rho >= trust_eta_shrink
            then
                accept:
                    x := x_trial
                    fx := fx_trial
                    dE := actual_reduction
                    step_norm := ||s||
                    accepted := true

                if rho >= trust_eta_expand and step_norm >= 0.9 * Delta_trial:
                    Delta := min(trust_radius_max, trust_expand * Delta_trial)
                else:
                    Delta := Delta_trial

                break reject loop

            else
                reject:
                    Delta := max(trust_radius_min, trust_shrink * Delta_trial)

        if accepted is false:
            if the final rejected step is already tiny, or the predicted
            reduction is negligible:
                stop with trust-region step convergence
            else:
                stop with trust-region rejection failure

    evaluate convergence checks:
        energy change small?
        step small?
        gradient small?

    if require_all_convergence:
        stop only when all required checks pass
    else:
        stop when the active stopping condition passes
```

## Meaning of `rho`

`rho` compares what the quadratic model predicted against what really happened:

- `predicted_reduction` says how much the local quadratic model expected to gain
- `actual_reduction` says how much the true objective actually dropped
- `rho` near `1` means the model was accurate
- small or negative `rho` means the model was poor

This is the standard trust-region acceptance signal.

## Parameter semantics

The current implementation uses the exposed trust parameters as follows:

- `trust_radius_init`:
  initial trust radius
- `trust_radius_min`, `trust_radius_max`:
  hard lower and upper radius bounds
- `trust_shrink`:
  multiplicative shrink factor applied after a rejected trust step
- `trust_expand`:
  multiplicative expansion factor applied after a successful trust step when
  `rho >= trust_eta_expand` and the accepted step is close to the current
  radius
- `trust_eta_shrink`:
  minimum acceptable `rho`
- `trust_eta_expand`:
  threshold above which a boundary-hitting accepted step may expand the radius
- `trust_max_reject`:
  maximum number of rejected trust steps per Newton iteration before the solver
  stops

## Notes

- The trust-region direction is still followed by a line search. This is a
  deliberate hybrid design, not a pure textbook trust-region method.
- The line search is clipped so the final accepted step remains inside the
  current trust ball.
- The trust-region step is currently built in a small subspace spanned by the
  Newton and gradient directions, not by solving the full trust-region
  subproblem in the full state space.
- The gradient fallback is still important for indefinite or poorly scaled
  Hessian models.

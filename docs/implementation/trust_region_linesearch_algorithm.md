# Nonlinear Globalization Contract

Last revised: 2026-07-10.

This note freezes the algorithms implemented by
`src/core/petsc/minimizers.py:newton`. It distinguishes three methods that must
not be conflated in experiment labels:

1. Newton with a one-dimensional line search;
2. Newton with an externally solved trust-region subproblem, normally PETSc
   Steihaug CG;
3. a reduced-subspace trust-region method used when no external trust
   subproblem callback is supplied.

The optional post-subproblem line search makes method 2 a hybrid. Method 3
follows its reduced trust step with the configured bounded Armijo or
golden-section search. The trust-region norm is the coefficient Euclidean norm
used by the subproblem callback. It is distinct from the optional Riesz metric
used to decide nonlinear convergence.

## Common discrete contract

At accepted state $x_k$, define

\[
f_k=f(x_k),\qquad g_k=\nabla f(x_k),\qquad H_k v
\]

through the supplied energy, gradient, and Hessian-action callbacks. Projection
and ghost-update callbacks are applied to every trial state. The convergence
metric supplies

\[
\lVert g_k\rVert_{V_h^*},\qquad
\lVert x_{k+1}-x_k\rVert_{V_h},\qquad
\lVert x_{k+1}\rVert_{V_h}.
\]

The absolute/relative gradient target is

\[
g_{\rm target}
=\max\{\mathtt{tolg},\mathtt{tolg\_rel}\lVert g_0\rVert_{V_h^*}\}
\]

when `tolg_rel>0`, and `tolg` otherwise. The relative accepted correction is

\[
s_k^{\rm rel}
=\frac{\lVert x_{k+1}-x_k\rVert_{V_h}}
       {\max\{u_{\rm scale},\lVert x_{k+1}\rVert_{V_h}\}}.
\]

With `require_all_convergence=True`, success requires an accepted step and all
three gates:

- $|f_k-f_{k+1}|<\mathtt{tolf}$;
- absolute or relative correction below `tolx_abs` or `tolx_rel`;
- a recomputed post-update dual residual below $g_{\rm target}$.

Without that option, a pre-iteration gradient gate or the configured legacy
energy gate may terminate the solve. Publication campaigns must use the full
contract and record the coefficient norms only as diagnostics.

Nonfinite energy, gradient, direction, or norm values are terminal failures.
Solver callbacks may raise a linear-solve exception; the problem wrapper owns
its classification and must preserve the last accepted state and a failure run
record.

## Method 1: Newton plus line search

The linear callback approximately solves

\[
H_k p_k=-g_k.
\]

An unusable or nonfinite direction is terminal. A direction whose coefficient
norm is at roundoff is successful only when the dual gradient gate is already
satisfied; otherwise it terminates as `Newton direction vanished before
gradient convergence`.

### Armijo mode

Let $d_k=g_k^\top p_k$. Starting from `armijo_alpha0`, clipped to the positive
part of the configured interval, repeatedly multiply by `armijo_shrink`. Accept
the first trial satisfying

\[
f(x_k+\alpha p_k)
\le f_k+c_1\alpha d_k.
\]

At most `armijo_max_ls` trials are evaluated. If no trial passes, the method
retains $x_k$, records the rejected iteration and evaluation count, and
terminates with an Armijo failure. It never repeats the same rejected state up
to the nonlinear iteration cap. If explicitly enabled, the optional fallback
repeats Armijo once along $-g_k$ and records that substitution.

### Residual-bisection modes

These legacy modes require $d_k<0$. They search $\alpha\in[0,1]$ for the last
trial with $\nabla f(x_k+\alpha p_k)^\top p_k<0$. The tolerance variant also
stops when the bracket width reaches `linesearch_tol`. A positive finite trial
is admitted according to this directional-residual rule. These modes are not
Armijo and must have separate experiment labels.

### Golden-section modes

The interval is repaired around $\alpha=0$ when needed, then a bounded
golden-section minimization is performed. The final trial must strictly reduce
the objective. A failed interval or nondecreasing trial is terminal.

A roundoff-limited nondecreasing trial may be accepted only under the full
convergence contract, only when its correction is small, its energy is within a
scale-aware roundoff/tolerance bound, and its recomputed dual residual passes.
The history marks `used_roundoff_acceptance=true`.

## Method 2: external trust-region subproblem

This is the production Steihaug/hybrid path. For trial radius $\Delta$, the
external callback receives $-g_k$ and returns a step $p$. The callback defines
its Krylov forcing, preconditioner, negative-curvature handling, and boundary
truncation; all must be recorded by the problem wrapper. The nonlinear method
then computes

\[
\ell=g_k^\top p,\qquad c=p^\top H_kp,
\qquad q(p)=\ell+\tfrac12c.
\]

Without post-subproblem line search, $\alpha=1$. With it, the configured
Armijo or golden search is clipped so $|\alpha|\lVert p\rVert_2\le\Delta$.
The predicted and actual reductions are

\[
\operatorname{pred}
=-\left(\alpha\ell+\tfrac12\alpha^2c\right),
\qquad
\operatorname{ared}=f_k-f(x_k+\alpha p),
\]

and $\rho=\operatorname{ared}/\operatorname{pred}$ only when the denominator is
positive and both quantities are finite.

The ordinary acceptance test is exactly

\[
f_{\rm trial}<\infty,qquad
\operatorname{ared}>0,qquad
\rho\ge\eta_{\rm accept}.
\]

Equality at `trust_eta_shrink` is accepted. A passing step expands the radius
only when

\[
\rho\ge\mathtt{trust\_eta\_expand}
\quad\text{and}\quad
\lVert\alpha p\rVert_2\ge0.9\Delta;
\]

otherwise the radius is unchanged. Rejection sets
$\Delta\leftarrow\max\{\Delta_{\min},
\mathtt{trust\_shrink}\,\Delta\}$ and resolves the subproblem. The method makes
at most `trust_max_reject+1` attempts in one nonlinear iteration.

After the final rejection, a small candidate step or negligible predicted
reduction is classified as radius exhaustion. It is not successful under the
full contract when the gradient gate still fails. A non-small rejected path is
terminal with `Trust-region rejected all candidate steps`. The last accepted
state is restored in every case.

Negative curvature is not itself a failure: the external subproblem may return
a boundary step, and the ordinary $\rho$ test decides acceptance. History
retains the model value, predicted/actual reductions, ratio, radius, and reject
count.

## Method 3: reduced-subspace trust region

When `trust_region=True` but no external subproblem callback exists, the method
builds a space from the approximate Newton direction and an orthogonalized
gradient direction. It solves the one- or two-dimensional quadratic trust
problem exactly up to the implemented boundary scan. A separate one-dimensional
gradient trust step is also constructed, and the lower quadratic-model value
is selected. The chosen direction is followed by the configured line search,
clipped so that $|\alpha|\lVert p\rVert_2\le\Delta$. In Armijo mode it uses the
same $c_1$, initial step, contraction, and evaluation cap as Method 1. In
golden-section mode it uses the same interval-repair and bounded minimization
contract as above. Acceptance, radius update, rejection retry, roundoff
handling, and termination then use the same formulas as Method 2.

This is not full-space Steihaug CG. Papers and tables must call it a
`reduced-subspace trust-region` method.

## Counter meanings

- `nit`: nonlinear iterations entered after the pre-iteration gradient gate;
- `ksp_its`: callback-reported linear/subproblem iterations for that nonlinear
  iteration;
- `ls_evals`: objective or directional-gradient trial evaluations accumulated
  within that iteration;
- `trust_rejects`: rejected trust candidates before acceptance or termination;
- `accepted_step`: whether the stored state changed;
- `used_gradient_fallback`: whether the explicit gradient substitute was used;
- `used_roundoff_acceptance`: whether the strict ordinary acceptance test failed
  but all roundoff/full-convergence guards passed.

Problem wrappers must additionally record recursive and independently
recomputed KSP residuals, convergence reasons, setup counts, and negative-
curvature events when the underlying PETSc method exposes them.

## Required experiment separation

`EXP-GLOB-001` has two tiers:

1. a controlled tier holding the discrete functional, Hessian action,
   preconditioner, forcing policy, start, and final accuracy fixed;
2. a production-bundle tier that may compare GMRES line search with STCG trust
   subproblems but cannot attribute differences to globalization alone.

Nonconvex endpoints are clustered by weighted state, energy, and problem
observables before time is compared. A failed line search, trust rejection,
linear failure, cap, or timeout remains a censored result, never a slow
converged observation.

## Regression coverage

Focused tests cover:

- inclusive trust acceptance at `rho == trust_eta_shrink`;
- rejection immediately below that threshold;
- a negative-curvature model step;
- accepted roundoff-limited hybrid termination;
- exhausted Armijo termination and last-state preservation;
- vanishing nonconverged Newton direction termination;
- reduced-subspace trust honoring its configured Armijo step rather than
  silently substituting a golden-section search;
- Riesz-scaled gradient and correction gates.

The publication protocol still requires problem-level tests of PETSc
negative-curvature reasons, failed linear solves, timeout checkpointing, and
all configured forcing policies before a controlled timing campaign is
admitted.

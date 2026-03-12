Given this `H`, I would not try to make AMG work on the raw matrix itself. I would solve
[
H,\delta z = r
]
with outer `MINRES`, and give PETSc a **different** preconditioning matrix `P` that is symmetric positive definite and much more AMG-friendly. PETSc explicitly supports distinct operator and preconditioner matrices through `KSPSetOperators(Amat,Pmat)`, and PETSc documents `KSPMINRES` as the right Krylov method for symmetric indefinite systems provided the preconditioner is itself symmetric positive definite. ([PETSc][1])

That recommendation is driven by the algebra you wrote down. Your raw `z`-Hessian is symmetric, but it is not SPD and it is not Laplacian-like: the double-well term, the sigmoid chain term, and the positive rank-one average couplings are exactly the kind of non-elliptic structure that makes generic AMG fragile. PETSc’s AMG guidance explicitly warns that non-elliptic operators are a common source of poor AMG performance, and it also notes that telling GAMG the matrix is symmetric avoids extra work during coarsening. ([PETSc][2])

## What I would precondition with

I would assemble a **surrogate SPD matrix** analytically from the element formulas, not by autodiff.

Write
[
u_e := D_e m = \frac13
\begin{bmatrix}
\theta'_i\ \theta'*j\ \theta'*k
\end{bmatrix},
\qquad
c*{\mathrm{inv},e} := e_e^{\mathrm{frozen}},p(p+1),\bar\theta_e^{-p-2},
\qquad
c*{W,e} := \alpha \ell^{-1} W''(\bar\theta_e).
]

Then use an element preconditioner of the form
[
P_e = K^{\mathrm{diff}}_e + R_e,
]
where (R_e) is diagonal and positive, and (K^{\mathrm{diff}}_e) is the only off-diagonal part.

A good first version is

[
K^{\mathrm{diff}}_e
===================

A_e \alpha \ell,\bar{\theta'_e}^{,2},K_e,
\qquad
\bar{\theta'_e}=\frac{\theta'_i+\theta'_j+\theta'_k}{3},
]

and

[
R_e
===

\operatorname{diag}(r_e),
]
with
[
r_e
===

A_e\bigl(c_{\mathrm{inv},e}+\max(c_{W,e},0)\bigr),u_e(\mathbf 1^T u_e)
;+;
A_e\frac{\mu}{3}\mathbf 1
;+;
\max(q_e\odot\theta''_e,0)
;+;
A_e\frac{\tau}{3}\mathbf 1 .
]

Two details matter here.

First, I would **lump every positive (m m^T) term to the diagonal** instead of keeping its dense (3\times3) block. That preserves the right size and positivity, but removes the positive off-diagonal average coupling that tends to confuse AMG coarsening.

Second, I would add a small floor (\tau>0) unless (\mu) already gives you a reliable reaction term everywhere. In saturated zones (\theta') can get tiny, so the diffusion part in `z`-space can become nearly singular.

If you want a closer surrogate, replace the scalarized diffusion by the exact SPD block
[
K^{\mathrm{diff}}_e = A_e \alpha \ell, D_e K_e D_e,
]
but in practice I would start with the scalarized coefficient (A_e \alpha\ell \bar{\theta'_e}^2 K_e), because it is a more standard variable-coefficient diffusion operator and usually coarsens better.

If computing (q_e) is inconvenient, drop the (\max(q_e\odot\theta''_e,0)) term from `P`. It is only a diagonal refinement.

## Why this works

This turns the preconditioner into a **scalar reaction-diffusion operator on the same mesh graph**, which is exactly the regime where AMG is strongest. The outer `MINRES` still sees the real indefinite Hessian `H`; AMG only has to invert the surrogate `P`.

In other words: treat the problem as **indefinite outer / elliptic inner**.

## PETSc setup I would use

I would try `PCGAMG` first, because your graph is fixed across iterations and PETSc exposes explicit interpolation reuse for GAMG. PETSc’s current GAMG docs provide `-pc_gamg_reuse_interpolation`, `-pc_gamg_repartition`, graph-threshold filtering, and aggressive-coarsening controls; PETSc also notes that thresholds in the (0.01)–(0.05) range are typical when tuning coarsening, while `-pc_gamg_threshold -1` is the simplest robust starting point. PETSc further warns that Chebyshev smoothers can fail when eigenvalue estimates are poor, so for a `MINRES` outer solve I would start with symmetric Richardson/Jacobi smoothing rather than clever smoothers. ([PETSc][3])

A solid first option set is:

```text
-topopt_ksp_type minres
-topopt_ksp_rtol 1e-8
-topopt_ksp_monitor_true_residual
-topopt_ksp_converged_reason

-topopt_pc_type gamg
-topopt_pc_gamg_type agg
-topopt_pc_gamg_repartition true
-topopt_pc_gamg_reuse_interpolation true
-topopt_pc_gamg_threshold -1.0
-topopt_pc_gamg_aggressive_coarsening 1

-topopt_mg_levels_ksp_type richardson
-topopt_mg_levels_ksp_max_it 2
-topopt_mg_levels_pc_type jacobi
-topopt_mg_coarse_ksp_type preonly
-topopt_mg_coarse_pc_type lu
```

After that, if the hierarchy gets too expensive or convergence is too slow, tune `-topopt_pc_gamg_threshold` upward into about `0.01`–`0.05`.

The other AMG candidate is BoomerAMG on the **same surrogate `P`**. PETSc says most BoomerAMG options are set only through the options database, that `-pc_hypre_boomeramg_max_iter` and `-pc_hypre_boomeramg_tol` control the number of V-cycles per preconditioner application, and that symmetric relaxation is the default, which is exactly what `MINRES` wants. PETSc also warns that BoomerAMG ignores an attached near-nullspace unless `-pc_hypre_boomeramg_nodal_coarsen` and `-pc_hypre_boomeramg_vec_interp_variant` are also set. hypre’s own documentation says the CPU defaults are HMIS coarsening and extended+i interpolation, and points to aggressive coarsening and non-Galerkin dropping as the main complexity-reduction knobs. ([PETSc][4])

So the BoomerAMG fallback is:

```text
-topopt_ksp_type minres
-topopt_ksp_rtol 1e-8
-topopt_pc_type hypre
-topopt_pc_hypre_type boomeramg
-topopt_pc_hypre_boomeramg_max_iter 1
-topopt_pc_hypre_boomeramg_tol 0.0
-topopt_pc_hypre_boomeramg_relax_type_all symmetric-SOR/Jacobi
```

I would still prefer `GAMG` first in your case because of the fixed graph and interpolation reuse.

## petsc4py sketch

```python
from petsc4py import PETSc
import numpy as np

def configure_topopt_ksp(H: PETSc.Mat, P: PETSc.Mat, coords_local: np.ndarray) -> PETSc.KSP:
    # coords_local shape: (n_local_owned, 2)

    for A in (H, P):
        A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        A.setOption(PETSc.Mat.Option.STRUCTURALLY_SYMMETRIC, True)
        # only set these if every future rebuild is truly symmetric too
        # A.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)
        # A.setOption(PETSc.Mat.Option.STRUCTURAL_SYMMETRY_ETERNAL, True)

    ksp = PETSc.KSP().create(comm=H.getComm())
    ksp.setOptionsPrefix("topopt_")
    ksp.setOperators(H, P)
    ksp.setType(PETSc.KSP.Type.MINRES)
    ksp.setTolerances(rtol=1e-8, atol=0.0, max_it=300)

    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.GAMG)
    pc.setCoordinates(coords_local)

    opts = PETSc.Options()
    opts["topopt_pc_gamg_type"] = "agg"
    opts["topopt_pc_gamg_repartition"] = "true"
    opts["topopt_pc_gamg_reuse_interpolation"] = "true"
    opts["topopt_pc_gamg_threshold"] = -1.0
    opts["topopt_pc_gamg_aggressive_coarsening"] = 1
    opts["topopt_mg_levels_ksp_type"] = "richardson"
    opts["topopt_mg_levels_ksp_max_it"] = 2
    opts["topopt_mg_levels_pc_type"] = "jacobi"
    opts["topopt_mg_coarse_ksp_type"] = "preonly"
    opts["topopt_mg_coarse_pc_type"] = "lu"

    ksp.setFromOptions()
    return ksp
```

```python
def surrogate_element_block(
    Ae, Ke, theta_p, theta_pp, theta_bar, efrozen, q_e,
    alpha, ell, p, mu, tau, scalarize_diffusion=True
):
    theta_p = np.asarray(theta_p, dtype=float)
    theta_pp = np.asarray(theta_pp, dtype=float)
    q_e = np.asarray(q_e, dtype=float)

    c_inv = efrozen * p * (p + 1) * theta_bar ** (-(p + 2))
    c_w   = alpha / ell * max(0.0, 2.0 - 12.0 * theta_bar + 12.0 * theta_bar**2)

    if scalarize_diffusion:
        a_e = alpha * ell * float(theta_p.mean() ** 2)
        Kloc = Ae * a_e * Ke
    else:
        D = np.diag(theta_p)
        Kloc = Ae * alpha * ell * (D @ Ke @ D)

    u = theta_p / 3.0
    r = Ae * (c_inv + c_w) * u * u.sum()   # lumped positive rank-1 terms
    r += Ae * (mu / 3.0)                   # lumped prox term
    r += np.maximum(q_e * theta_pp, 0.0)   # optional; may omit
    r += Ae * (tau / 3.0)                  # positivity floor

    return Kloc + np.diag(r)
```

petsc4py exposes exactly the calls you need here: `KSP.setOperators([A,P])`, `PC.setCoordinates(coordinates)`, and matrix update paths such as `Mat.setValuesCSR(...)` and `Mat.setPreallocationCOO(...)`/`Mat.setValuesCOO(...)`. PETSc’s FAQ also recommends COO preallocation/value insertion as a faster assembly path on CPUs and GPUs. ([PETSc][5])

## Assembly/update strategy

Because your graph is fixed, I would keep the same `Mat`, `KSP`, and `PC` objects alive across outer iterations.

For `H`:

* if JAX already gives you CSR values, update the PETSc matrix with `setValuesCSR` on the existing sparsity pattern.

For `P`:

* assemble it analytically in an element loop;
* use the same global sparsity pattern each time;
* if COO is more natural from your triangle loop, preallocate once with `setPreallocationCOO` and only update values afterward.

This is exactly the case where GAMG’s interpolation reuse is valuable. ([PETSc][3])

## Near-nullspace and coordinates

For this scalar P1 problem, you do **not** need elasticity-style nullspace vectors. PETSc notes that for smoothed aggregation the constant vector is the default near-nullspace for each variable when the block size is one, so a scalar problem already has the right default shape. Supplying your 2D nodal coordinates is still worthwhile, and PETSc’s `PCSetCoordinates()` is implemented for both GAMG and HYPRE. ([PETSc][1])

## One more practical point

At the size you quoted, `16205 x 16205` with about `1.1e5` nonzeros, I would absolutely keep a sparse direct solve around as a baseline on one rank. PETSc’s matrix manual notes that when the sparsity pattern stays fixed you can reuse the ordering/symbolic factorization and only redo the numeric factorization. Also, PETSc’s FAQ recommends roughly **10,000 unknowns per MPI rank minimum**, preferably **20,000+**, for good parallel efficiency, so this benchmark is still too small to tell you much about strong scaling beyond about one or two ranks. Use `-log_view` and `-pc_mg_log` while tuning. ([PETSc][6])

If you are allowed to solve a **modified** Newton system instead of the raw one, there is an even simpler route: add a small shift and solve
[
(H + \tau M_{\text{lumped}}),\delta z = r
]
with outer `CG` and AMG directly on the shifted operator. Given the spectra you reported, that is likely the easiest path to robustness. But if you need the raw linear system, the setup above is the one I would implement first: **outer MINRES on `H`, inner AMG on an SPD surrogate `P` built from diffusion plus lumped positive reaction.**

[1]: https://petsc.org/main/manual/ksp/ "https://petsc.org/main/manual/ksp/"
[2]: https://petsc.org/release/manual/ksp/ "https://petsc.org/release/manual/ksp/"
[3]: https://petsc.org/main/manualpages/PC/PCGAMG/ "https://petsc.org/main/manualpages/PC/PCGAMG/"
[4]: https://petsc.org/main/manualpages/PC/PCHYPRE/ "https://petsc.org/main/manualpages/PC/PCHYPRE/"
[5]: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.KSP.html "https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.KSP.html"
[6]: https://petsc.org/release/manual/mat/ "https://petsc.org/release/manual/mat/"

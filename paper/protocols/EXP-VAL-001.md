# EXP-VAL-001: Independent Verification and Optional Matched Comparator

## Status and scope

**Smooth scalar manufactured, hyperelastic affine-patch, and nonaffine
hyperelastic convergence blocks passed as dirty-worktree pilots; the matched
backend is outside the required manuscript scope.** The scalar blocks
independently assemble and solve smooth $p=3$
Laplace and branch-controlled Ginzburg--Landau problems and verify the expected
P1 spatial rates. The mechanics blocks
compare the production hyperelastic energy, residual, and Hessian with an
independent analytic first-Piola/tangent and boundary-traction assembly on an
exact affine deformation, then independently solve a nonaffine manufactured
weak problem and measure displacement, deformation-gradient, and stress rates.
These results do not validate the production rotating-beam solve or the
production L-shaped scalar load case.

The paper makes no DOLFINx/FEniCS comparison claim. If a matched-backend
companion is added later, it remains admission-blocked until the local
ADIOS2/DOLFINx ABI is repaired with explicit user approval. No dependency
rebuild is authorized by this card.

## Scientific questions

1. Does a P1 discretization of the stated smooth $p$-Laplace weak form achieve
   the expected $O(h^2)$ $L^2$ and $O(h)$ $H^1$-seminorm errors?
2. **Optional comparator question:** do production JAX, JAX/PETSc, and FEniCS
   backends agree when mesh, quadrature, load, constraints, initial state, and
   stopping contract are identical?
3. Do hyperelastic residuals, reactions, and tangents reproduce an independent
   affine patch and a nontrivial manufactured deformation while preserving
   positive determinant?
4. **Optional physical-model comparator question:** if a mechanics-validation
   claim is introduced for Plasticity3D, do an independently derived material
   model and global assembly reproduce the selected endpoint surrogate away
   from branch interfaces?

Questions 1 and 3, together with the branch-controlled Ginzburg--Landau block,
form the required independent-verification scope. Questions 2 and 4 are not
required by the current paper and remain unexecuted. Plasticity3D is presented
as a synthetic branch-structured discrete optimization functional, not as a
validated path-dependent constitutive model. Its formula, branch, derivative,
assembly, and distribution checks belong to EXP-MC-001, EXP-DERIV-001, and
EXP-DIST-001 and establish internal consistency only.

## Completed manufactured scalar block

The unit-square exact solution is

\[
u(x,y)=x+y+0.1\sin(\pi x)\sin(\pi y),
\]

with exact nonhomogeneous Dirichlet data and analytic source
$f=-\nabla\cdot(|\nabla u|\nabla u)$. Its gradient is bounded away from zero by
$1-0.1\pi>0.685$, avoiding the degenerate zero-gradient Hessian of the
$p=3$ operator. The verifier uses:

- uniform P1 triangular meshes with 8, 16, 32, and 64 subdivisions per axis;
- a seven-point degree-five Dunavant rule for the load and error integrals;
- independently assembled NumPy/SciPy weak residual and consistent tangent;
- damped Newton initialized by the affine boundary extension $x+y$;
- relative algebraic residual at most $10^{-8}$;
- no production AD, PETSc, MPI, or FEniCS assembly path.

The prespecified pilot gates are last-pair $L^2$ rate at least 1.75,
$H^1$-seminorm rate at least 0.85, tangent symmetry defect at most
$10^{-12}$, and minimum discrete element-gradient norm above 0.5.

## Pilot result

All four levels reached the relative residual target in three Newton steps.
The successive observed $L^2$ rates were 1.986, 1.996, and 1.999; the
$H^1$-seminorm rates were 0.990, 0.998, and 0.999. The finest errors were
$2.761\times10^{-5}$ and $5.451\times10^{-3}$, respectively. Every assembled
tangent was symmetric to the stored sparse arithmetic, and the minimum final
element-gradient norm exceeded 1.19.

Raw diagnostic:
`artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-VAL-001/plaplace_manufactured.json`.
The run came from a dirty worktree and is not publication evidence.

## Completed hyperelastic affine-patch block

The unit cube is split into six P1 tetrahedra and mapped by the constant
orientation-preserving deformation gradient

\[
F=\begin{bmatrix}
1.15&0.08&0\\
0.02&0.92&0.05\\
0&0.03&1.08
\end{bmatrix},\qquad \det F=1.139187,
\]

plus an arbitrary translation. For the manuscript density
$W=C_1(I_1-3-2\log J)+D_1(J-1)^2$, an independent NumPy implementation
assembles the analytic first Piola stress, material tangent, nodal internal
force, and constant boundary traction. The production JAX energy is
differentiated on the identical coordinates.

Relative discrepancies are $4.58\times10^{-15}$ in energy,
$2.67\times10^{-15}$ in the residual, $4.38\times10^{-16}$ in the Hessian, and
$1.75\times10^{-16}$ in boundary-traction balance. The Hessian symmetry defect
is $1.49\times10^{-16}$; net force is $1.92\times10^{-16}$; rigid-translation
Hessian actions are at most $1.19\times10^{-15}$. A rotated deformation gives
zero stored energy discrepancy and $7.94\times10^{-17}$ Piola covariance
error. All pass the $2\times10^{-11}$ pilot gate.

Raw diagnostic:
`artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-VAL-001/hyperelastic_affine_patch.json`.
The affine patch is exactly representable and does not establish nonaffine
spatial convergence.

## Completed hyperelastic nonaffine manufactured block

On the unit cube, the exact deformation is

\[
y(X,Y,Z)=\bigl(X+0.05\sin(\pi X)\sin(\pi Y)\sin(\pi Z),Y,Z\bigr).
\]

It has a strictly positive determinant. Exact values are imposed on the whole
boundary, and a closed-form body force $-\operatorname{Div}P(F_{\mathrm{exact}})$
makes the deformation a solution of the continuous compressible neo-Hookean
problem. A NumPy/SciPy implementation, independent of the production JAX,
PETSc, and mesh paths, assembles the first Piola stress and consistent tangent.
The source formula was additionally checked against a centered finite-
difference divergence of the analytic Piola field.

The Newton line search uses the stated Armijo test. A nondecreasing trial at
floating-point roundoff may be accepted only when its energy change is within
a scale-aware FP64 bound, its relative correction is at most
$\sqrt{\epsilon_{\mathrm{mach}}}$, and the independently reassembled trial
residual already passes the frozen algebraic stopping test. Such an event is
recorded explicitly as roundoff acceptance.

The 4, 8, 16, and 24 subdivision meshes have 81, 1,029, 10,125, and 36,501
free vector DOFs. All converge in four damped Newton steps. The last-pair rates
are 1.887 for displacement $L^2$, 1.006 for the deformation-gradient error,
and 0.983 for first-Piola stress, passing the prespecified 1.75/0.75/0.75
margins. The minimum discrete determinant is 0.844, and the largest tangent
symmetry defect is $8.60\times10^{-17}$.

The primary order-4 Duffy load was independently reassembled at orders 6 and
8 on every mesh, and the order-6 problem was re-solved. The maximum response
change was $8.29\times10^{-6}$ of the finite-element error, while the maximum
order-4-to-order-6 load change was $1.78\times10^{-5}$ of the exact-interpolant
consistency residual. Both occurred on the coarsest mesh and decreased under
refinement; the order-6-to-order-8 load difference was smaller again.

Raw diagnostic:
`artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-VAL-001/hyperelastic_nonaffine_quadrature_refinement_v2/result.json`.
This is a manufactured-formulation test with full Dirichlet data, not an
identical-functional production-backend comparison or a rotating-beam
validation.

## Completed Ginzburg--Landau manufactured block

The verifier uses

\[
u(x,y)=0.8+0.1\sin(\pi x)\sin(\pi y)
\]

with exact Dirichlet data and source
$f=-\epsilon\Delta u+u(u^2-1)$ for $\epsilon=0.04$. The independently
assembled solve uses the same symmetric three-point triangle rule as the
production JAX functional, while errors are post-evaluated with the seven-point
rule. The exact solution and every recorded computed endpoint have nodal values
at least 0.8, above $1/\sqrt{3}$. The recorded endpoints therefore lie in the
positive-curvature branch; the exact solution alone would not establish this
property for the computed states.

All four levels converge in four Newton steps. Successive $L^2$ rates are
2.010, 2.003, and 2.001; successive $H^1$-seminorm rates are 0.997, 0.999, and
1.000. Finest errors are $1.805\times10^{-5}$ and
$5.452\times10^{-3}$, and final relative residuals are at most
$2.38\times10^{-13}$. This verifies the smooth selected branch; it is not a
robustness study over nonconvex basins.

Raw diagnostic:
`artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-VAL-001/ginzburg_landau_manufactured.json`.

## Remaining execution order

1. Rerun every retained independent block from a clean frozen experiment
   commit and archive
   the strict run record, environment, commands, states, and hashes.
2. Only if a cross-backend claim is later retained, obtain authorization for
   the DOLFINx repair and export one canonical mesh and state for an
   identical-functional JAX, JAX/PETSc, and FEniCS comparison.
3. For that optional comparison, record weighted state, gradient, HVP,
   reaction, energy, and independently evaluated residual errors with
   conditioning-aware tolerances.

## Terminal decisions

- **PASS:** every retained central family passes its independent
  manufactured/patch gate; a matched backend is additionally required only if
  the paper makes the corresponding comparison claim.
- **SCOPED PASS:** the smooth scalar and hyperelastic checks pass, while
  synthetic plasticity remains a derivative benchmark rather than a validated
  mechanics model.
- **OPTIONAL COMPARATOR OMITTED:** a legally distributable or executable
  matched backend is unavailable; retain the passed independent mathematical
  checks and make no backend-validation claim.
- **FAIL:** expected rates, admissibility, residual agreement, or reaction
  balance fail; repair the model/discretization before any performance result
  using that family is admitted.

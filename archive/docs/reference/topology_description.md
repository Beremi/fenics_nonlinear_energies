## Implement a **staggered SIMP + phase-field regularization** benchmark

This is a very good showcase for your repository because:

* the **elasticity step** is an ordinary energy minimization over displacement DOFs;
* the **design step** is another ordinary scalar energy minimization over a fixed scalar field on the same mesh;
* both steps decompose into fixed-size element energies and keep a fixed sparsity graph;
* you can reuse ideas from your existing **hyperelasticity** and **Ginzburg–Landau** examples almost directly.

The part that is “easy in your approach, harder in plain FEniCS” is not the elasticity solve itself. It is the fact that the design update becomes a genuine FE energy subproblem with exact derivatives, instead of a residual PDE plus an external optimization wrapper. The COMET-FEniCS SIMP demo explicitly uses alternating minimization and (p)-continuation, because jointly minimizing the relaxed SIMP functional is hard in practice once (p>1). ([comet-fenics.readthedocs.io][2])

## The trap to avoid

Do **not** define
[
J(u,\theta)=\int_\Omega \frac12 \theta^p,\varepsilon(u):C_0:\varepsilon(u),dx-\ell(u)+R(\theta)
]
and minimize in both (u) and (\theta).

For force-controlled loading, if (K(\theta)u=f), then
[
\Pi(u^*(\theta),\theta)= -\frac12 f^T K(\theta)^{-1}f,
]
so minimizing the reduced potential makes the structure **softer**, not stiffer. That is the wrong physics for minimum-compliance topology optimization.

The correct classical problem is “minimize compliance subject to equilibrium,” not “jointly minimize potential in ((u,\theta)).” Phase-field and filtered approaches still treat it as a reduced design optimization, often using gradient flows or staggered updates with Ginzburg–Landau/perimeter regularization. ([TU Dresden][3])

---

## What I would implement

Use a fixed triangular mesh on a standard 2D cantilever:

* rectangle ( [0,L]\times[0,H] ),
* left edge clamped,
* downward traction on a short segment of the right edge,
* target volume fraction (\eta\approx 0.4).

That is a standard benchmark and matches the FEniCS demo geometry well enough for comparison. ([comet-fenics.readthedocs.io][2])

### 1. Unknowns

Use two separate problems, not one monolithic one.

For mechanics:

* (u \in [P1]^2) or ([P2]^2) on the mesh.

For design:

* a scalar field (z \in P1) on the same mesh,
* mapped to a material fraction
  [
  \theta(z)=\theta_{\min} + (1-\theta_{\min}),\sigma(z), \qquad \sigma(z)=\frac{1}{1+e^{-z}}.
  ]

Here:

* (\theta_{\min}>0) is the ersatz-void stiffness floor;
* (z) is unconstrained, which is good for your Newton/line-search solver;
* (\theta(z)\in(\theta_{\min},1)) for all trial states, so the energy stays finite even when the line search extrapolates.

This bound-handling choice is extremely important for your code because you already know the solver probes off-path trial points.

### 2. Material law

Use SIMP:
[
C(\theta)=\theta^p C_0,
]
with (p) continued from (1) toward (3) or (4).

For a first version, use small-strain linear elasticity:
[
\varepsilon(u)=\frac12(\nabla u+\nabla u^T).
]

Plane stress is fine for a clean benchmark.

### 3. Mechanics subproblem

At outer iteration (k), with current design (z_k), solve
[
u_{k+1} = \arg\min_u \Pi(u;z_k),
]
where
[
\Pi(u;z_k)=\int_\Omega \frac12,\theta(z_k)^p,\varepsilon(u):C_0:\varepsilon(u),dx-\ell(u).
]

This is just linear elasticity with spatially varying stiffness.

### 4. Design subproblem

After solving for (u_{k+1}), freeze the mechanics information and solve a scalar nonlinear design energy.

Define the frozen complementary-energy density
[
e_k(x)=\sigma_k(x):C_0^{-1}:\sigma_k(x),
\qquad
\sigma_k=\theta(z_k)^p C_0 \varepsilon(u_{k+1}).
]

Then solve
[
z_{k+1}=\arg\min_z G_k(z),
]
with
[
G_k(z)=
\int_\Omega
\left[
e_k,\theta(z)^{-p}
+\lambda_k,\theta(z)
+\alpha\left(\frac{\ell_{\rm pf}}{2}|\nabla \theta(z)|^2+\frac{1}{\ell_{\rm pf}}W(\theta(z))\right)
+\frac{\mu}{2}(z-z_k)^2
\right]dx.
]

Use
[
W(\theta)=\theta^2(1-\theta)^2.
]

Interpretation:

* (e_k,\theta^{-p}): the SIMP design term with mechanics frozen;
* (\lambda_k \theta): volume penalty through an outer Lagrange-multiplier update;
* phase-field/Ginzburg–Landau regularization: suppresses checkerboards and controls feature size;
* (\frac{\mu}{2}(z-z_k)^2): optional proximal/move-limit term if the outer alternation oscillates.

This design subproblem is the one that is very natural for your framework: it is a scalar FE energy on a fixed graph, with fixed-size local stencils and exact JAX derivatives.

---

## Why this version fits your repository well

### Mechanics step

It matches your existing “hyperelasticity-like” structure almost exactly:

* scalar energy,
* fixed full state vector with Dirichlet elimination,
* vector block size (2),
* fixed Hessian graph,
* element-local kernels,
* optional boundary facet load terms.

### Design step

It looks like a Ginzburg–Landau problem with a data-driven forcing term:

* scalar energy,
* fixed free-DOF vector,
* fixed graph,
* element-local kernels,
* exact Hessian from `jax.hessian(element_energy_design)`.

So this benchmark is basically:

* **elasticity** from your current mechanics examples,
* plus **Ginzburg–Landau** from your scalar-field examples,
* glued together by a simple outer loop.

That is why I think it is a strong “easy here, awkward in plain FEniCS” example.

---

## How to map it to your data structures

## A. Mechanics problem object

Use a standard vector FE problem.

State meaning:

* full vector length (2n_{\text{nodes}}),
* ordering
  [
  [ u_{x,0}, u_{y,0}, u_{x,1}, u_{y,1}, \dots ].
  ]

For P1 triangles with vertex connectivity `tri[e] = [i,j,k]`,

```python
elems_u[e] = [2*i, 2*i+1, 2*j, 2*j+1, 2*k, 2*k+1]
```

Provide:

* `u0_u`: full displacement vector with clamped values inserted;
* `freedofs_u`: all non-Dirichlet displacement DOFs;
* `elems_u`;
* `adjacency_u`: standard vector-elasticity adjacency;
* `ownership_block_size = 2`;
* coordinates and rigid-body near-nullspace for AMG.

Per-element data:

* geometry/Jacobian data,
* quadrature weights,
* current material factors (\theta(z_k)^p) at quadrature points.

Element kernel:
[
J^u_e(u_e)=\sum_q w_q |J_e|
\left(
\frac12,m_{e,q},\varepsilon_q(u_e):C_0:\varepsilon_q(u_e)

* b_q\cdot u_q
  \right)
  ]
  plus facet terms
  [
  J^{\Gamma}_f(u_f)= -\int_f t\cdot u,ds.
  ]

## B. Design problem object

Use a separate scalar FE problem.

State meaning:

* full vector length (n_{\text{nodes}}),
* one scalar latent design DOF per node.

For the same triangle connectivity,

```python
elems_z[e] = [i, j, k]
```

Provide:

* `u0_z`: full design vector with fixed non-design zones inserted;
* `freedofs_z`: all free design DOFs;
* `elems_z`;
* `adjacency_z`: scalar P1 graph.

Per-element data:

* geometry/Jacobian data,
* quadrature weights,
* frozen (e_k) values at quadrature points,
* previous design values if using the proximal term.

Element kernel:

```python
def design_element_energy(z_e, e_q, z_old_e, params):
    val = 0.0
    grad_z = dNdx.T @ z_e
    for q in range(nq):
        zq = N[q] @ z_e
        s = sigmoid(zq)
        theta = theta_min + (1.0 - theta_min) * s
        dtheta_dz = (1.0 - theta_min) * s * (1.0 - s)
        grad_theta = dtheta_dz * grad_z
        W = theta**2 * (1.0 - theta)**2
        zold_q = N[q] @ z_old_e
        val += w[q] * detJ * (
            e_q[q] * theta**(-p)
            + lam * theta
            + alpha * (0.5 * ell_pf * dot(grad_theta, grad_theta) + W / ell_pf)
            + 0.5 * mu_move * (zq - zold_q)**2
        )
    return val
```

That kernel is fully JAX-friendly.

---

## Outer algorithm

Use a simple staggered loop:

1. Initialize (z_0) as a uniform field giving the target volume fraction.
2. Set (p=1), choose (\lambda_0), maybe (\mu>0) small.
3. Repeat:

   1. build current (\theta(z_k)),
   2. solve mechanics step for (u_{k+1}),
   3. compute frozen (e_k) per element/quadrature point,
   4. solve design step for (z_{k+1}),
   5. compute current volume
      [
      V_{k+1}=\int_\Omega \theta(z_{k+1}),dx,
      ]
   6. update the scalar multiplier, e.g.
      [
      \lambda_{k+1}=\lambda_k+\beta_{\rm AL}\left(\frac{V_{k+1}}{|\Omega|}-\eta\right),
      ]
   7. every few outer iterations increase (p) toward (p_{\max}),
   8. stop when design change, compliance, and volume residual stabilize.

The important point is that **each inner solve is a problem your code already knows how to do**.

---

## What I would implement first

I would do it in this order:

1. **Linear elasticity only**, with current material field treated as fixed data.
2. Add a scalar **design-step energy** with just
   [
   e_k \theta^{-p} + \lambda \theta + \alpha |\nabla \theta|^2.
   ]
3. Add the **sigmoid parameterization** for robust bound handling.
4. Add the **double-well** (W(\theta)) if you want crisper black/white designs.
5. Add (p)-continuation.
6. Only after that think about Helmholtz filters, projections, or hyperelastic state equations.

That gets you to a working benchmark quickly.

---

## The main caveats

### 1. Classical topopt is not a native one-shot problem for your solver

This is the big one. The reduced compliance problem is global, constrained, and its reduced Hessian in the design variable is effectively dense/nonlocal. If you try to force the whole thing into one monolithic sparse Newton problem, you will either get the wrong objective or lose the locality that your JAX+PETSc element path depends on. ([wias-berlin.de][1])

### 2. Do not use a squared global volume penalty inside the inner design solve

A term like
[
\frac{\beta}{2}\left(\int_\Omega \theta - \eta |\Omega|\right)^2
]
adds a rank-one dense Hessian block over all design DOFs. That is mathematically fine, but it is a bad match to your sparse adjacency assumption.

Use an **outer scalar multiplier update** instead.

### 3. Avoid direct nonlocal filters at first

Convolution filters and explicit neighborhood filters are standard, but they introduce nonlocal couplings over the filter radius. Filters are used precisely because they regularize the problem and suppress checkerboards/mesh dependence, but from your implementation point of view they are less local than a phase-field or (H^1)-type regularization. ([ms.mcmaster.ca][4])

If later you want a Helmholtz-style filter while keeping locality, introduce an auxiliary filtered field as another FE unknown rather than eliminating it.

### 4. Keep (\theta_{\min}>0)

Do not allow exact zero stiffness. Even standard SIMP implementations keep a positive floor to avoid degeneracy. ([comet-fenics.readthedocs.io][2])

### 5. Use non-design solid regions

If the support or load application zone is allowed to dissolve into void, the problem becomes numerically awkward and often physically silly. Fix a small solid pad near the clamped edge and near the traction patch.

In your framework that is easy: mark those design DOFs as nonfree and insert their values into `u0_z`.

### 6. Do not use point loads for the first benchmark

Use a distributed edge traction on a short boundary segment. Point-load singularities contaminate the energy density and make the topology highly mesh-sensitive.

### 7. Resolve the interface length scale

In phase-field formulations, interface thickness is tied to the regularization length scale. If your (\ell_{\rm pf}) is smaller than roughly a few mesh sizes, the interface is under-resolved and the Newton step gets ugly. Phase-field papers explicitly tie interface width to the small parameter (\varepsilon). ([TU Dresden][3])

### 8. Expect local minima and use continuation

Topology optimization is nonconvex. Different initial conditions, mesh choices, and continuation schedules can yield different local optima. The standard SIMP practice is to start with (p=1) and increase it gradually. ([comet-fenics.readthedocs.io][2])

### 9. Watch out for sigmoid saturation

The unconstrained (z\mapsto \theta(z)) map is robust for line search, but if (z) gets very large in magnitude the derivatives flatten out. So:

* do not start with an overly sharp projection;
* do not over-weight the double-well term too early;
* use continuation instead of immediate black/white forcing.

### 10. Use enough quadrature

Because (\theta(z)) enters through sigmoid and powers, the design integrand is not a low-degree polynomial anymore. Do not rely on the absolute minimum quadrature rule.

### 11. Separate the two inner solves

Do not pack (u) and (z) into one mixed vector in the first implementation. The elasticity block wants block size (2), coordinates, and rigid-body modes; the design block wants block size (1) and a different preconditioner. Two separate solves are cleaner and more performant.

---

## My bottom-line recommendation

For your repository, the best topology-optimization example is:

* **2D cantilever**
* **linear elasticity**
* **P1 nodal design field on the same triangular mesh**
* **SIMP material law**
* **phase-field/H1 regularization**
* **outer alternating minimization**
* **outer scalar update for the volume multiplier**
* **(p)-continuation**

That is a real step up in complexity from p-Laplace and hyperelasticity, it stays very close to your native “energy over free DOFs” architecture, and it is more awkward in plain FEniCS than in your setup because plain FEniCS still needs the outer optimization machinery bolted on. If you specifically want a *single* monolithic variational-energy benchmark, phase-field fracture is actually a cleaner fit than topology optimization; but for a fixed-mesh material-layout problem, the staggered topology version above is the one I would build first.

[1]: https://www.wias-berlin.de/preprint/3219/wias_preprints_3219.pdf "Numerical analysis of the SIMP model for the topology optimization problem of minimizing compliance in linear elasticity"
[2]: https://comet-fenics.readthedocs.io/en/latest/demo/topology_optimization/simp_topology_optimization.html "Topology optimization using the SIMP method — Numerical tours of continuum mechanics using FEniCS master documentation"
[3]: https://tu-dresden.de/mn/math/wir/ressourcen/dateien/forschung/publikationen/pdf2012/phase_field_approaches_to_structural_topology.pdf "https://tu-dresden.de/mn/math/wir/ressourcen/dateien/forschung/publikationen/pdf2012/phase_field_approaches_to_structural_topology.pdf"
[4]: https://ms.mcmaster.ca/~bourdinb/downloads/Bourdin-2001a.pdf "https://ms.mcmaster.ca/~bourdinb/downloads/Bourdin-2001a.pdf"

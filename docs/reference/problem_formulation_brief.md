# Problem Formulation Brief

This note explains what kinds of problems fit the maintained solver stack and
how to formulate a new one so it works with the current JAX and JAX+PETSc
infrastructure.

## Core Solver Model

The repository solves finite-dimensional energy minimisation problems of the
form

$$
\min_{u\in\mathbb{R}^{n_{\mathrm{free}}}} J(u),
$$

where:

- `u` contains only free DOFs
- constrained DOFs are stored separately
- `J(u)` is a scalar energy
- Newton-type methods use the gradient and Hessian of `J`

Current examples live under:

- `src/problems/plaplace/`
- `src/problems/ginzburg_landau/`
- `src/problems/hyperelasticity/`
- `src/problems/topology/`

The common serial derivative machinery is in `src/core/serial/jax_diff.py`,
and the common distributed assembly/scaffolding lives under `src/core/petsc/`.

## Hard Requirements

### 1. The problem must be an energy minimisation problem

The maintained solvers are not generic residual-equation solvers. They expect a
scalar potential `J(u)`.

Good fit:

- elliptic energies
- hyperelastic stored-energy minimisation
- smooth reaction-diffusion energies
- smooth non-convex energies

Poor fit unless reformulated first:

- saddle-point systems with Lagrange multipliers
- contact and inequality constraints
- complementarity systems
- nonsmooth active-set logic

### 2. The unknown must be a fixed vector of free DOFs

Every maintained path assumes:

- one flat free-DOF vector
- a fixed constrained/free split during a solve
- a consistent map between the mathematical field and that flat vector

Changing Dirichlet values between load steps is supported. Changing which DOFs
are free during a solve is not part of the normal workflow.

### 3. The energy must stay finite on trial states

The nonlinear methods evaluate:

- the current iterate
- gradients
- Hessians or Hessian-vector products
- line-search trial points, often with step lengths in `[-0.5, 2.0]`

The energy therefore has to remain numerically meaningful under small
extrapolations near the current state. Non-convexity is acceptable; immediate
`NaN` or `inf` under ordinary trial steps is not.

### 4. The energy must be differentiable enough for JAX

For the recommended `JAX+PETSc element` path, assume the discrete energy is at
least `C^2` on the physically relevant region. In practice this means:

- JAX-compatible operations only
- fixed tensor shapes
- no Python side effects inside the energy kernel
- no hidden nonsmooth branching at the operating point

### 5. The Hessian sparsity pattern must be fixed

The maintained serial and distributed paths assume a known sparse structural
superset of the Hessian nonzeros. That pattern is used for:

- graph coloring
- sparse preallocation
- local coloring / COO extraction
- element-to-matrix scatter

If the true Hessian develops structural nonzeros missing from that pattern, the
assembly is wrong.

## Extra Requirements For The Serial JAX Path

The pure-JAX serial path is the least restrictive maintained solver family. It
needs:

- a scalar global energy `J(u, **params)`
- the parameters needed to rebuild the full state
- a fixed free-DOF adjacency pattern
- an initial guess

This is the model used by the maintained serial pLaplace and HyperElasticity
paths.

## Extra Requirements For The Distributed JAX+PETSc Path

This is the recommended production path for new scalable problems.

### 1. The total energy should decompose into local contributions

The distributed implementation expects something close to

$$
J(u)=\sum_e J_e(u_e;\text{element data}) + J_{\mathrm{simple\ global}}(u),
$$

where each element uses a fixed-size local stencil. Simple global terms such as
`-f \cdot u` are fine; genuinely dense nonlinear couplings are not.

### 2. Element connectivity must index directly into the full state vector

The distributed assemblers expect a rectangular connectivity array:

- scalar problems: scalar-DOF connectivity
- vector problems: usually expanded DOF connectivity
- mixed fields: only if flattened into a fixed local stencil

### 3. The recommended path wants an exact per-element kernel

The maintained production mode is `assembly_mode=element`. That path uses exact
element Hessians through JAX differentiation of a local element energy kernel.
If the discrete energy cannot be expressed that way, the problem may still fit
the serial path but will not fit the current scalable production path as well.

## Practical Advice For New Problems

When adding a new family, target this sequence:

1. write a clean scalar energy in JAX
2. define the constrained/free split explicitly
3. expose a fixed adjacency pattern
4. if MPI scaling matters, derive a fixed-stencil local element energy
5. only then add the solver CLI and the benchmark/report layer

If a new problem satisfies the requirements of the `JAX+PETSc element` path, it
will usually also fit the serial JAX path cleanly.

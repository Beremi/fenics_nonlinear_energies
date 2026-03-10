# Problem Formulation Brief for the JAX and JAX+PETSc Solvers

This note is meant to be handed to someone who will formulate a new problem for
this repository.

The short version is:

- the codebase does not solve an arbitrary nonlinear residual equation,
- it solves a finite-dimensional energy minimization problem,
- and the safest target is to formulate the problem so it fits the
  `JAX+PETSc element` path.

If a problem satisfies the `JAX+PETSc element` requirements, it will usually
also fit the pure-JAX serial path.

## 1. What the repository actually solves

The common solver model in this repository is:

$$
\min_{u \in \mathbb{R}^{n_{\mathrm{free}}}} J(u)
$$

where:

- `u` is the vector of free DOFs only,
- Dirichlet DOFs are removed from the unknown vector and stored separately,
- `J(u)` is a scalar energy,
- the solver uses Newton-type steps based on the gradient and Hessian of `J`.

This is visible in:

- pure JAX energy functions such as
  [pLaplace2D_jax/jax_energy.py](/home/michal/repos/fenics_nonlinear_energies/pLaplace2D_jax/jax_energy.py),
  [GinzburgLandau2D_jax/jax_energy.py](/home/michal/repos/fenics_nonlinear_energies/GinzburgLandau2D_jax/jax_energy.py),
  and [HyperElasticity3D_jax/jax_energy.py](/home/michal/repos/fenics_nonlinear_energies/HyperElasticity3D_jax/jax_energy.py),
- the serial JAX derivative builder in
  [tools/jax_diff.py](/home/michal/repos/fenics_nonlinear_energies/tools/jax_diff.py),
- and the generic MPI JAX+PETSc problem interface in
  [tools_petsc4py/jax_tools/parallel_assembler.py](/home/michal/repos/fenics_nonlinear_energies/tools_petsc4py/jax_tools/parallel_assembler.py).

## 2. Non-negotiable requirements

These are the hard requirements if the method is supposed to work at all.

### 2.1 The problem must be expressible as an energy minimization problem

The repository expects a scalar potential `J(u)`.

That means:

- a residual form by itself is not enough,
- a generic nonlinear PDE `R(u) = 0` is not enough unless it comes from an
  energy,
- saddle-point systems are not a natural fit unless they are reformulated into a
  true minimization problem or reduced first.

Good fit:

- elliptic energies,
- hyperelastic stored-energy minimization,
- reaction-diffusion type energies,
- non-convex energies, as long as they are still smooth enough.

Poor fit:

- incompressible mixed systems with Lagrange multipliers,
- contact/inequality constraints,
- complementarity problems,
- active-set logic embedded directly into the energy,
- nonsmooth energies without regularization.

### 2.2 The unknown must be a fixed-size vector of free DOFs

The core unknown is always a dense vector `u` containing only unconstrained DOFs.

The formulation must provide:

- a full DOF vector `u_0` containing Dirichlet values,
- a list `freedofs` telling which entries of `u_0` are free,
- a consistent mapping between the mathematical unknown and this flat vector.

The free-DOF set is assumed to be fixed during a solve. Changing Dirichlet
values between load steps is supported; changing which DOFs are free is not part
of the normal workflow.

### 2.3 The energy must stay finite on the trial states the solver will visit

The Newton methods in this repository do not only evaluate the current iterate.
They also evaluate:

- gradients,
- Hessian-vector products or exact Hessians,
- and many line-search trial points, typically with `alpha` in `[-0.5, 2.0]`.

So the formulation must remain numerically meaningful for off-path trial states
near the current iterate. If the energy instantly becomes `NaN` or `inf` under
small extrapolation, the method will be fragile.

Non-convexity is acceptable. Immediate non-finiteness is not.

### 2.4 The energy must be differentiable in the way JAX needs

At minimum, the method needs a reliable gradient. For the recommended
`JAX+PETSc element` path it also needs second derivatives.

Practical requirement:

- formulate the discrete energy using JAX-compatible operations only,
- avoid data-dependent shape changes,
- avoid Python side effects inside the energy kernel,
- avoid branching that creates nonsmooth behavior at the operating point,
- use a smooth enough energy on the region the solver will explore.

Strong recommendation:

- assume the problem should be at least `C^2` with respect to the free DOFs on
  the physically relevant region.

If the energy is not twice differentiable, the exact element-Hessian path is not
appropriate and even the SFD path becomes less trustworthy.

### 2.5 The Hessian sparsity pattern must be fixed and known

Both pure JAX and JAX+PETSc assume a fixed sparse adjacency pattern over the
free DOFs.

The formulation must provide a sparse matrix `adjacency` of shape
`(n_free, n_free)` that is a valid structural superset of the Hessian nonzeros.

This pattern is used for:

- graph coloring in the serial JAX path,
- PETSc sparse matrix preallocation,
- local coloring / COO extraction in the MPI path,
- and element-to-matrix scatter in the exact element-Hessian path.

If the true Hessian can develop structural nonzeros that are missing from
`adjacency`, the assembly is wrong.

## 3. Additional requirements for the pure JAX path

The pure-JAX serial path is the less restrictive one.

To support it, the problem needs:

- a global scalar energy function of the form
  `J(u, **params) -> scalar`,
- the `params` needed to rebuild the full state from `u`,
- the fixed free-DOF adjacency pattern,
- an initial guess `u_init`.

This is the model used by:

- [tools/jax_diff.py](/home/michal/repos/fenics_nonlinear_energies/tools/jax_diff.py),
- [pLaplace2D_jax/solve_pLaplace_jax_newton.py](/home/michal/repos/fenics_nonlinear_energies/pLaplace2D_jax/solve_pLaplace_jax_newton.py),
- [HyperElasticity3D_jax/solve_HE_jax_newton.py](/home/michal/repos/fenics_nonlinear_energies/HyperElasticity3D_jax/solve_HE_jax_newton.py).

For this path alone, the formulation does not strictly need explicit
per-element Hessians. A global energy plus adjacency is enough.

## 4. Additional requirements for the JAX+PETSc path

This is the important part if the problem is expected to scale with MPI or use
the repository's production path.

### 4.1 The total energy must decompose into local element contributions

The MPI implementation is built around local subdomains and local element
assembly. The expected structure is:

$$
J(u) = \sum_e J_e(u_e; \text{element data}) + J_{\text{global simple}}(u)
$$

where `u_e` is the fixed local stencil for one element.

In practice:

- each element contributes through a fixed-size local vector,
- each element has fixed-shape per-element data,
- the total energy is a sum over elements,
- optional simple global terms such as `-f dot u` are also allowed.

A boundary contribution is also fine if it can be written in the same style:
a sum of fixed-stencil local facet terms with a fixed connectivity pattern.
What does not fit well is a genuinely dense or globally coupled nonlinear term
that cannot be localized.

This is exactly how the current problem specs are written in:

- [pLaplace2D_jax_petsc/parallel_hessian_dof.py](/home/michal/repos/fenics_nonlinear_energies/pLaplace2D_jax_petsc/parallel_hessian_dof.py),
- [GinzburgLandau2D_jax_petsc/parallel_hessian_dof.py](/home/michal/repos/fenics_nonlinear_energies/GinzburgLandau2D_jax_petsc/parallel_hessian_dof.py),
- [HyperElasticity3D_jax_petsc/parallel_hessian_dof.py](/home/michal/repos/fenics_nonlinear_energies/HyperElasticity3D_jax_petsc/parallel_hessian_dof.py).

### 4.2 The element connectivity must index directly into the full state vector

The `elems` array is expected to be a rectangular connectivity array whose
entries index into the full vector used in the energy.

That means:

- for scalar problems, `elems` can be scalar-DOF connectivity,
- for vector problems, `elems` usually needs to be expanded to DOF connectivity,
- mixed fields are acceptable only if they are flattened into one fixed local
  DOF stencil per cell,
- all elements must use the same stencil size in a given problem.

Current examples:

- scalar P1 problems use `elems.shape = (n_elem, npe)`,
- 3D hyperelasticity expands each tetrahedron from `4` nodes to `12` vector DOFs
  before entering the PETSc layer in
  [HyperElasticity3D_petsc_support/mesh.py](/home/michal/repos/fenics_nonlinear_energies/HyperElasticity3D_petsc_support/mesh.py).

### 4.3 The recommended path needs an exact per-element energy kernel

The production MPI path uses `assembly_mode=element`, which computes exact
element Hessians through `jax.hessian(element_energy)`.

So the formulation should provide an element kernel of the form:

- `element_energy(v_e, element_data...) -> scalar`

with:

- fixed local DOF count per element,
- fixed-shape element metadata,
- JAX-differentiable operations only.

If this can be provided, the repository can use the fastest and cleanest JAX+PETSc path.

### 4.4 Dirichlet conditions must be encoded by elimination, not by penalty or by solving for all DOFs

The current infrastructure assumes:

- the full vector stores the imposed Dirichlet values,
- free DOFs are solved for separately,
- local vectors are rebuilt by inserting the free values into the full-state
  template.

Time-dependent Dirichlet data is supported only in the sense that `u_0` can be
updated between load steps while `freedofs`, `elems`, and `adjacency` stay
fixed.

This is exactly what the hyperelastic load stepping does in
[HyperElasticity3D_jax_petsc/solver.py](/home/michal/repos/fenics_nonlinear_energies/HyperElasticity3D_jax_petsc/solver.py)
and [tools_petsc4py/dof_partition.py](/home/michal/repos/fenics_nonlinear_energies/tools_petsc4py/dof_partition.py).

### 4.5 The problem must tolerate PETSc's block-distributed ownership model

The MPI path distributes contiguous ranges of free DOFs across ranks.

Implications:

- the free-DOF ordering must be globally consistent,
- vector-valued problems should declare a block size if components must stay
  grouped,
- the number of free DOFs must be divisible by that block size.

For elasticity, the code uses `ownership_block_size=3` and relies on `xyz`
triplets staying intact for AMG options such as coordinates and block size.

### 4.6 Some problems need extra linear-solver metadata

For scalar problems, the sparse matrix pattern is often enough.

For vector mechanics problems, better AMG behavior may require:

- near-nullspace vectors,
- nodal coordinates,
- knowledge of the block size,
- and a physically sensible initial guess / continuation strategy.

Hyperelasticity is the current example:

- rigid-body modes are provided as a near-nullspace,
- coordinates are attached for GAMG,
- boundary values are updated by continuation.

If the new problem has a known nullspace or near-nullspace, it should be
provided up front.

## 5. What kinds of formulations fit best

Best fit:

- finite-element energies assembled as a sum of element contributions,
- fixed mesh topology during each solve,
- fixed polynomial order and fixed element stencil,
- Dirichlet boundary conditions handled by elimination,
- optional continuation / load stepping with the same DOF pattern,
- smooth nonlinearities.

Acceptable but more delicate:

- non-convex energies,
- indefinite Hessians,
- strong continuation dependence,
- energies with logarithms or determinants, provided the iterates stay in a
  safe region.

Bad fit without reformulation or regularization:

- variational inequalities,
- contact,
- `L1`, total variation, max/min, or other nonsmooth terms,
- adaptivity that changes the DOF graph inside a solve,
- mixed formulations where the operator is not a true Hessian of one scalar
  energy,
- problems whose stable solution method fundamentally relies on constraints or
  active-set updates not present here.

## 6. Minimal deliverable package the formulator should hand back

If someone else is preparing a new problem for this repository, the output
should include all of the following.

### 6.1 Mathematical description

Provide:

- the continuous energy functional,
- the domain and boundary conditions,
- the discrete FE space,
- all material or model parameters,
- whether the problem is convex or non-convex,
- whether there is continuation / load stepping.

### 6.2 Discrete state definition

Provide:

- what one entry of the full state vector means,
- the DOF ordering convention,
- the full vector `u_0` with Dirichlet values inserted,
- the free-DOF index list `freedofs`,
- the initial guess `u_init` for the free DOFs.

### 6.3 Connectivity and geometry data

Provide:

- `elems`: element-to-full-state connectivity,
- all per-element arrays needed by the local energy kernel,
- any global arrays needed by the energy kernel,
- the sparse free-DOF adjacency pattern.

The per-element arrays must have fixed shapes and be indexable by the same
element numbering as `elems`.

### 6.4 Energy kernels

Provide both of these, at least conceptually:

1. A global discrete energy:
   `J(u_free, params) -> scalar`
2. A local element energy:
   `J_e(u_element, element_data) -> scalar`

The second one is what unlocks the recommended MPI element-Hessian path.

### 6.5 Solver-side notes

State explicitly:

- expected block size of the unknown,
- any known nullspace or near-nullspace,
- whether coordinates are available and meaningful for AMG,
- whether line-search extrapolation can leave the admissible region,
- whether the problem will need continuation to converge reliably.

## 7. Recommended target to give the formulator

The right instruction is not "make a nonlinear PDE."

The right instruction is:

1. Formulate the problem as a discrete energy minimization over free DOFs.
2. Make the total energy a sum of fixed-stencil element energies.
3. Encode Dirichlet constraints by elimination into `u_0` and `freedofs`.
4. Provide a fixed Hessian sparsity pattern over free DOFs.
5. Provide a JAX-differentiable element energy so `jax.hessian` can be used.
6. Keep mesh topology, free-DOF structure, and sparsity fixed during each solve.
7. If continuation is needed, vary only parameters or Dirichlet values between steps.

If they can deliver that, the problem is structurally compatible with the
repository.

## 8. Short screening checklist

Before spending implementation time, these answers should all be "yes":

- Is there a scalar energy to minimize?
- Can the unknown be represented as a fixed flat vector of free DOFs?
- Can Dirichlet constraints be handled by elimination?
- Is the discrete energy JAX-differentiable?
- Is the Hessian sparsity pattern fixed?
- Can the total energy be written as a sum of fixed-size element energies?
- Can a local element energy be evaluated from a fixed local stencil?
- Will the energy stay finite under Newton and line-search trial points near the solution path?

If any answer is "no", the problem likely needs reformulation before it will fit
this codebase.

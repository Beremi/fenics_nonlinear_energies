# PETSc PMG Setup Notes

## Purpose

This note describes the PETSc p-multigrid setup used for the Mohr-Coulomb plasticity experiments on the slope benchmark. It is written as a self-contained implementation note that can be copied into another project.

The focus is the working assembled PMG path:

- fine space: `P4`
- intermediate spaces: `P2`, `P1`
- optional extra coarse tail: one lower `h` level in `P1`
- outer Krylov solver: `FGMRES`
- preconditioner: `PCMG`
- MG cycle: multiplicative `V`

This note also lists what must be added if another codebase already has separate `P1`, `P2`, and `P4` builders.

## High-Level Design

The solver is a Newton method. At each Newton step it solves a linearized system

`A_k x = b_k`

with PETSc. The good PMG path uses:

- `A_k`: assembled fine tangent matrix on the current `P4` space
- `PCMG`: a PETSc multigrid hierarchy using mixed polynomial degrees
- coarse hierarchy: same mesh with degree reduction `P4 -> P2 -> P1`
- optional extra tail: one coarser mesh level in `P1`

The important idea is:

1. Build one PETSc level for each FE space.
2. Build prolongation and restriction operators between adjacent levels.
3. Let PETSc manage the V-cycle and smoothers.
4. Refresh only the fine matrix each Newton step in the legacy assembled path.

## Hierarchy Used Here

For the featured `L5` run, the working hierarchy is:

- `L4 P1`
- `L5 P1`
- `L5 P2`
- `L5 P4`

So this is a mixed `h/p` hierarchy:

- one coarser mesh level in `P1`
- then same-mesh degree reduction up to the fine `P4` space

For a pure same-mesh version, the hierarchy is simply:

- `P1`
- `P2`
- `P4`

on the same geometric mesh.

## What Data Each Level Must Provide

Each level must provide at least:

- node coordinates, shape `(n_nodes, dim)`
- scalar element connectivity, shape `(n_elems, n_local_scalar_dofs)`
- constrained/free DOF description
- full vector `u_0` so total DOF count is known
- adjacency or graph information for DOF reordering

For vector-valued displacement in 2D, the effective vector DOF count is:

- `2 * n_nodes` before constraints

Each level must also provide:

- `freedofs`: total DOF indices that remain after Dirichlet elimination
- a map from total DOF index to free-space index

The implementation here stores a vector

- `total_to_free_orig[total_dof] = free_dof_index or -1`

This map is essential when building prolongation operators in free-DOF coordinates.

## Reordering And Ownership

Each level is reordered independently before PETSc partitioning.

For vector elasticity/plasticity in 2D:

- use block size `2`
- reorder free DOFs in blocks of two so `ux, uy` stay together

Per level, define:

- `perm`: reordered position -> original free-DOF index
- `iperm`: original free-DOF index -> reordered position
- `lo, hi`: PETSc ownership range in reordered free-space indexing

If your other project already has its own partitioning, you still need a consistent local indexing per level so prolongation can be written in the same numbering that PETSc uses.

## How Prolongation Is Built

Do not build prolongation by assuming local DOF numbers match between `P1`, `P2`, and `P4`.

Instead, build prolongation geometrically:

1. Take one fine node.
2. Find the containing coarse triangle.
3. Express the fine point in the coarse triangle reference coordinates.
4. Evaluate the coarse Lagrange basis functions at that point.
5. Use those basis values as interpolation weights.

This works for:

- same mesh, different polynomial degree
- different meshes, as long as the fine node lies inside some coarse triangle

For the same-mesh `P4 -> P2 -> P1` case:

- `P1` to `P2`: evaluate `P1` basis at each `P2` node
- `P2` to `P4`: evaluate `P2` basis at each `P4` node

For vector fields, the scalar prolongation is duplicated componentwise:

- scalar weight `w` from coarse node `i` to fine node `j`
- becomes two entries:
  - `(2*j+0, 2*i+0) = w`
  - `(2*j+1, 2*i+1) = w`

Constrained DOFs are skipped:

- if either the fine or coarse total DOF is constrained, that prolongation entry is omitted

## Restriction

Restriction is assembled explicitly too.

In this implementation it is formed by transposing the scalar/vector transfer entries when building the sparse matrix.

For practical PETSc use, the restriction matrix is built explicitly rather than relying on PETSc to infer it.

## PETSc Matrix Construction For Transfers

The prolongation and restriction matrices are created as PETSc `MPIAIJ` matrices using owned rows only.

The pattern is:

1. Generate global COO triplets `(row, col, value)` in reordered free-space coordinates.
2. Keep only triplets whose row is locally owned.
3. Create the matrix with the full global size and owned local row size.
4. Preallocate with `setPreallocationCOO`.
5. Insert values with `setValuesCOO`.
6. Assemble.

This is a robust and simple way to create transfer operators in parallel.

## PETSc MG Configuration

The working legacy PMG setup is:

```python
ksp.setType("fgmres")

pc = ksp.getPC()
pc.setType(PETSc.PC.Type.MG)
pc.setMGLevels(nlevels)
pc.setMGType(PETSc.PC.MGType.MULTIPLICATIVE)
pc.setMGCycleType(PETSc.PC.MGCycleType.V)

for ell in range(1, nlevels):
    pc.setMGInterpolation(ell, prolongations[ell - 1])
    pc.setMGRestriction(ell, restrictions[ell - 1])
```

This means:

- multiplicative multigrid
- `V`-cycle
- level `0` is the coarsest
- level `nlevels - 1` is the finest

## Smoothers

The strongest working assembled smoother setup here is:

- for every noncoarse level:
  - KSP type: `richardson`
  - PC type: `sor`
  - iterations per smoothing phase: `3`

That is conceptually:

```python
for level_idx in range(1, nlevels):
    s = pc.getMGSmoother(level_idx)
    s.setType("richardson")
    s.setTolerances(max_it=3)
    s.getPC().setType("sor")
```

This uses PETSc's default level solve object on each noncoarse level.

Important practical note:

- this is matrix-based smoothing
- it works well for assembled fine matrices
- it is the main reason the assembled PMG path is strong

## Coarse Solve

The coarse solve policy used here is:

- on one MPI rank:
  - KSP: `preonly`
  - PC: `lu`
- on more than one MPI rank:
  - KSP: `cg`
  - PC: `jacobi`

Conceptually:

```python
coarse = pc.getMGCoarseSolve()
coarse.setType("preonly" if comm_size == 1 else "cg")
coarse.setTolerances(rtol=1e-10, max_it=200)
coarse.getPC().setType("lu" if comm_size == 1 else "jacobi")
```

This is conservative and portable.

## Fine Operator And Galerkin

In the legacy assembled PMG path:

- the fine Newton tangent is assembled on the fine space
- PETSc is given that matrix as the operator/preconditioner matrix
- coarse operators are then generated through Galerkin in PETSc

So the expected PETSc usage is:

```python
ksp.setOperators(Afine)
```

or, if operator and preconditioner matrices differ,

```python
ksp.setOperators(Amat, Pmat)
```

and for legacy PMG:

- Galerkin should be enabled so PETSc builds coarse operators from the fine `Pmat`

In practice this means the equivalent of:

- `pc_mg_galerkin = both`

This is the easiest path to reproduce in another project.

## Newton-Step Workflow

For the assembled legacy PMG path, each Newton step looks like:

1. Assemble the current fine `P4` tangent matrix.
2. Call `ksp.setOperators(A_k)`.
3. Call `ksp.setUp()`.
4. Solve the linear system.

The hierarchy itself is not rebuilt each Newton step.

What is reused across Newton steps:

- the PETSc `KSP` object
- the PETSc `PCMG` object
- all prolongation matrices
- all restriction matrices
- coarse and smoother KSP objects
- work vectors inside PETSc
- DOF orderings and ownership ranges

What changes each Newton step:

- the fine tangent matrix entries
- any operator-dependent PC setup that PETSc recomputes after `setOperators` / `setUp`

## What To Reuse Safely

If the mesh, spaces, ordering, and BC pattern do not change, you can safely reuse:

- hierarchy topology
- level metadata
- prolongation
- restriction
- KSP object
- PCMG object
- smoother objects
- coarse solver objects
- matrix shells or sparse containers with fixed sparsity

You should not assume you can safely reuse stale operator-dependent preconditioner setup across Newton steps unless you have tested the effect on iteration counts.

In this project, the safe reuse is:

- keep the PETSc objects alive
- still call `setOperators` and `setUp` each Newton step

The unsafe reuse experiment was:

- ask PETSc to reuse a stale preconditioner setup for changing Newton matrices

That degraded iteration counts significantly.

## What Another Project With Separate `P1`, `P2`, `P4` Builders Still Needs

If your other project already has separate builders for `P1`, `P2`, and `P4`, you still need to add:

### 1. A Unified Level Wrapper

Each builder should be wrapped into a common structure containing:

- `nodes`
- `elems_scalar`
- `freedofs`
- `u_0`
- adjacency
- `perm`
- `iperm`
- `lo`, `hi`
- `total_to_free_orig`

### 2. Consistent Free-DOF Conventions

Dirichlet handling must be consistent across all degrees:

- same geometric boundary interpretation
- same componentwise clamping rules

If `P1`, `P2`, and `P4` eliminate different DOFs on the same boundary, the transfer builder becomes messy or wrong.

### 3. Coordinate-Based Transfer Construction

You need a geometric search and basis-evaluation transfer builder.

For same mesh:

- map fine nodes to containing coarse elements
- evaluate coarse basis at fine-node coordinates

For different meshes:

- same, but triangle search must work across meshes

### 4. Vector Duplication Logic

If the problem is vector-valued, scalar transfer weights must be expanded componentwise.

### 5. PETSc-Owned Row Assembly For Transfers

Your builder must output row/col/value triplets in PETSc numbering and then assemble only owned rows locally.

### 6. Fine Operator Assembly Hook

Your Newton solve must expose a routine:

- assemble fine tangent at the current Newton state

This is the one thing that PMG absolutely needs every Newton step in the legacy assembled path.

### 7. Optional Lower-`h` Tail

If you want `L4 P1` below `L5 P1`, you need:

- an independent builder for the coarser mesh level
- the same transfer construction machinery between `L4 P1` and `L5 P1`

## Minimal Portable Implementation Plan

If you want the fastest route to reproduce this setup in another project, do this in order:

### Phase 1: Same-Mesh p-MG Only

Implement:

- `P1`, `P2`, `P4` level wrappers on the same mesh
- prolongation `P1 -> P2`
- prolongation `P2 -> P4`
- assembled fine `P4` tangent
- legacy PETSc `PCMG` with `richardson + sor`

Stop here first and verify it works.

### Phase 2: Add Optional h-Tail

Add:

- lower `h`-level `P1`
- prolongation from coarse `P1` mesh to fine `P1` mesh

Then use:

- `L4 P1 -> L5 P1 -> L5 P2 -> L5 P4`

### Phase 3: Add More Advanced Variants Only If Needed

Only after the assembled legacy path works, consider:

- explicit per-level operators
- matrix-free fine operator
- frozen or lagged operators
- custom smoothers

These are significantly more complicated than the legacy assembled path.

## Recommended PETSc Defaults To Copy

For the strong assembled PMG baseline, use:

- outer KSP: `fgmres`
- PC: `mg`
- MG type: multiplicative
- cycle type: `V`
- smoothers on noncoarse levels:
  - KSP: `richardson`
  - PC: `sor`
  - steps: `3`
- coarse solve:
  - 1 rank: `preonly + lu`
  - many ranks: `cg + jacobi`
- fine operator: assembled current tangent
- Galerkin coarse operators: enabled from fine `Pmat`

## What Not To Do First

Do not begin by trying to port:

- matrix-free fine `P4` with shell-only MG
- frozen fine `P4` preconditioner matrices
- staggered fine-level updates
- custom PETSc Python smoothers

Those experiments were useful here, but none of them were as strong or as simple as the assembled legacy PMG baseline.

## Bottom Line

The working recipe to share is:

- build level spaces for `P1`, `P2`, `P4`
- reorder each level in free-DOF space
- build geometric prolongation/restriction from coarse basis evaluation at fine nodes
- configure PETSc `PCMG` as multiplicative `V`-cycle
- use `richardson + sor` on noncoarse levels
- use a simple coarse solve
- assemble the fine tangent each Newton step
- reuse the hierarchy objects across Newton steps, but do not freeze stale operator-dependent PC setup unless you have measured that it stays effective

If another project already has separate `P1`, `P2`, and `P4` builders, the main missing work is not the FE spaces themselves. The missing work is:

- unified level metadata
- transfer construction
- free-DOF/reordering consistency
- PETSc hierarchy assembly


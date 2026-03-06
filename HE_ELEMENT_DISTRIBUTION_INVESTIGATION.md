# HyperElasticity Element Distribution Investigation

Date: 2026-03-06

## Scope

This is a stage-1, stage-2, and stage-3 investigation of the
**element-Hessian** path for `HyperElasticity3D_jax_petsc`.

The goal was to test, before changing the production solver, how much of the
performance depends on:

- the **ordering** of DOFs before PETSc ownership is assigned,
- the resulting **PETSc row distribution**,
- the local subdomain construction used to compute `J`, `grad`, and the
  element Hessian,
- the communication strategy used to turn local element contributions into a
  distributed matrix/vector.

This note records results from:

- [bench_he_element_distribution.py](/home/michal/repos/fenics_nonlinear_energies/experiment_scripts/bench_he_element_distribution.py)
- [run_he_element_distribution_step.py](/home/michal/repos/fenics_nonlinear_energies/experiment_scripts/run_he_element_distribution_step.py)

Stage 3 additionally records the production move of the winning idea into:

- [reordered_element_assembler.py](/home/michal/repos/fenics_nonlinear_energies/HyperElasticity3D_jax_petsc/reordered_element_assembler.py)
- [solver.py](/home/michal/repos/fenics_nonlinear_energies/HyperElasticity3D_jax_petsc/solver.py)
- [solve_HE_dof.py](/home/michal/repos/fenics_nonlinear_energies/HyperElasticity3D_jax_petsc/solve_HE_dof.py)

## Prototype strategies

The prototype keeps the same free-DOF PETSc matrix shape as the current JAX
solver, but changes how the subdomains are built and how element contributions
are combined.

### 1. `overlap_allgather`

- PETSc row ownership is defined first from the reordered free-DOF numbering.
- Each rank takes the **overlapping** local domain:
  all elements touching any owned row.
- The current free vector is rebuilt by `Allgatherv` of owned slices.
- Each rank computes:
  - weighted local `J`,
  - exact local `grad`,
  - exact local element Hessians,
  only for the rows it owns.
- No output reduction is needed for the matrix or gradient because the overlap
  is large enough to compute all owned-row contributions locally.

### 2. `nonoverlap_allreduce`

- PETSc row ownership is again defined first from the reordered free-DOF numbering.
- Elements are assigned **uniquely** to one rank by the owner of their minimum
  reordered free DOF.
- The current free vector is rebuilt by `Allgatherv` of owned slices.
- Each rank computes exact per-element `J`, `grad`, and Hessian on its disjoint
  element set.
- The partial gradient vector and global COO value vector are then summed by
  `Allreduce`.

This nonoverlap variant is more expensive in communication, but it provides a
clean reference because the element partition itself has no duplication.

## Reorder modes tested

All reorderings preserve the `xyz` triplet block structure (`bs=3`):

- `none`:
  original free-DOF order
- `block_rcm`:
  reverse Cuthill-McKee on the block graph
- `block_xyz`:
  geometric lexicographic ordering by coordinates
- `block_metis`:
  block graph partitioned by `pymetis`, then grouped by partition

## Stage 1 Benchmark Setup

These are **fixed-state** tests, not full Newton trajectories.

For each case:

- mesh level and MPI size are fixed,
- the state is the initial guess for load step `1 / 96`,
- exact per-element `J`, `grad`, and Hessian are computed,
- a single linear solve `H p = -g` is run with PETSc `GMRES + GAMG`.

Linear solver settings:

- `ksp_rtol=1e-1`
- `ksp_max_it=30`
- `pc_gamg_threshold=0.05`
- `pc_gamg_agg_nsmooths=1`

## Level 2, `np=4`

| Reorder | Strategy | Elem duplication | Node duplication | Hessian [s] | PC setup [s] | Solve [s] | KSP iters |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | overlap/allgather | 1.746 | 2.196 | 0.342 | 0.0482 | 0.0019 | 2 |
| `none` | nonoverlap/allreduce | 1.000 | 1.493 | 0.363 | 0.0476 | 0.0019 | 2 |
| `block_rcm` | overlap/allgather | 1.022 | 1.044 | 0.325 | 0.0291 | 0.0013 | 2 |
| `block_rcm` | nonoverlap/allreduce | 1.000 | 1.022 | 0.330 | 0.0292 | 0.0013 | 2 |
| `block_xyz` | overlap/allgather | 1.020 | 1.039 | 0.320 | 0.0287 | 0.0013 | 2 |
| `block_xyz` | nonoverlap/allreduce | 1.000 | 1.019 | 0.364 | 0.0288 | 0.0013 | 2 |
| `block_metis` | overlap/allgather | 1.019 | 1.038 | 0.316 | 0.0292 | 0.0009 | 1 |
| `block_metis` | nonoverlap/allreduce | 1.000 | 1.019 | 0.347 | 0.0292 | 0.0009 | 1 |

## Level 3, `np=8`

| Reorder | Strategy | Elem duplication | Node duplication | Hessian [s] | PC setup [s] | Solve [s] | KSP iters |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | overlap/allgather | 1.873 | 2.552 | 0.541 | 0.2395 | 0.0171 | 3 |
| `none` | nonoverlap/allreduce | 1.000 | 1.694 | 0.719 | 0.2313 | 0.0171 | 3 |
| `block_rcm` | overlap/allgather | 1.024 | 1.047 | 0.617 | 0.0941 | 0.0107 | 2 |
| `block_rcm` | nonoverlap/allreduce | 1.000 | 1.024 | 0.685 | 0.0939 | 0.0107 | 2 |
| `block_xyz` | overlap/allgather | 1.022 | 1.044 | 0.668 | 0.0968 | 0.0073 | 1 |
| `block_xyz` | nonoverlap/allreduce | 1.000 | 1.022 | 0.685 | 0.0957 | 0.0073 | 1 |
| `block_metis` | overlap/allgather | 1.022 | 1.044 | 0.624 | 0.0892 | 0.0111 | 2 |
| `block_metis` | nonoverlap/allreduce | 1.000 | 1.022 | 0.683 | 0.0905 | 0.0106 | 2 |

Cross-strategy checks on both levels:

- energy matches to roundoff
- gradient norm matches to roundoff

## Stage 1 Findings

### 1. Natural ordering is clearly bad

The `none` ordering is not competitive.

On `level 3`, `np=8`:

- overlap duplication jumps to `1.873x` in elements and `2.552x` in nodes
- `pc_setup` is about `0.24 s`
- solve time is about `0.017 s`

All block-aware reorderings are much better than this.

### 2. Reordering before PETSc ownership split matters a lot

Once the DOFs are reordered in a block-aware way, the PETSc distribution becomes
far more local.

On `level 3`, `np=8`:

- `pc_setup` drops from about `0.24 s` to about `0.09–0.10 s`
- solve time drops from about `0.017 s` to about `0.007–0.011 s`
- overlap duplication drops from `1.873x` to about `1.02x`

So the first answer to “what is the optimal PETSc matrix distribution?” is:

- **not** the natural free-DOF order
- a **block-aware reordered** free-DOF layout before PETSc ownership is assigned

### 3. The two communication strategies are both correct

For every tested reorder:

- `overlap_allgather` and `nonoverlap_allreduce` produced the same energy
- they produced the same gradient norm
- they produced the same linear iteration count and essentially the same
  `pc_setup` / solve time

So the prototype implementation is numerically consistent.

### 4. Overlap + allgather is cheaper than nonoverlap + allreduce

The nonoverlap strategy avoids duplicated elements, but its Hessian path is
consistently slower because it pays for the global reduction of the full
gradient/COO value vectors.

On `level 3`, `np=8`:

- `block_rcm`: `0.617 s` vs `0.685 s`
- `block_metis`: `0.624 s` vs `0.683 s`

So for the tested cases, the preferred assembly strategy is:

- **overlapping local domains**
- **input allgather**
- **no output reduction for owned rows**

### 5. No single block-aware reorder is a universal winner yet

The tested block-aware reorderings are all much better than `none`, but they do
not have the same best metric:

- `block_metis` gave the best `pc_setup` on the tested `level 3`, `np=8` case
- `block_xyz` gave the best solve time and even reduced the solve to one Krylov
  iteration in that case
- `block_rcm` stayed competitive and is robust

So the current recommendation is:

- treat `block_xyz`, `block_rcm`, and `block_metis` as the serious candidates
- do **not** keep the natural order

## Stage 2 Full Step `1 / 96`

The next check was the full nonlinear solve for load step `1 / 96`, first on
`level 3` and then on `level 4` if the strategy stayed stable.

For these runs, the experimental harness used the same nonlinear and linear
solver settings as the FEniCS custom comparison:

- Newton with `require_all_convergence=True`
- `tolf=1e-4`
- `tolg=1e-3`
- `tolg_rel=1e-3`
- `tolx_rel=1e-3`
- `tolx_abs=1e-10`
- line search on `[-0.5, 2.0]`
- `GMRES + GAMG`
- `ksp_rtol=1e-1`
- `ksp_max_it=30`
- `pc_setup_on_ksp_cap=True`
- `pc_gamg_threshold=0.05`
- `pc_gamg_agg_nsmooths=1`

The comparison rows are:

- `FEniCS custom`:
  [solve_HE_custom_jaxversion.py](/home/michal/repos/fenics_nonlinear_energies/HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py)
- `JAX element current`:
  [solve_HE_dof.py](/home/michal/repos/fenics_nonlinear_energies/HyperElasticity3D_jax_petsc/solve_HE_dof.py)
- experimental element-distribution runs from:
  [run_he_element_distribution_step.py](/home/michal/repos/fenics_nonlinear_energies/experiment_scripts/run_he_element_distribution_step.py)

### Level 3, `np=16`

| Variant | Elem dup | Newton | Sum KSP | Setup [s] | Assembly [s] | PC setup [s] | Solve [s] | Total [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FEniCS custom` | - | 12 | 75 | 0.064 | 0.164 | 0.074 | 0.271 | 0.660 |
| `JAX element current` | - | 11 | 84 | 2.501 | 0.413 | 0.244 | 1.146 | 2.363 |
| `none / overlap` | 2.202 | 11 | 84 | 0.616 | 0.453 | 0.207 | 1.064 | 2.252 |
| `none / nonoverlap` | 1.000 | 11 | 84 | 0.605 | 0.772 | 0.203 | 1.059 | 2.302 |
| `block_rcm / overlap` | 1.051 | 12 | 70 | 0.601 | 0.398 | 0.066 | 0.253 | 0.990 |
| `block_rcm / nonoverlap` | 1.000 | 12 | 70 | 0.606 | 0.815 | 0.066 | 0.247 | 1.413 |
| `block_xyz / overlap` | 1.047 | 16 | 84 | 0.599 | 0.517 | 0.061 | 0.311 | 1.258 |
| `block_xyz / nonoverlap` | 1.000 | 16 | 84 | 0.606 | 1.061 | 0.069 | 0.315 | 1.983 |
| `block_metis / overlap` | 1.067 | 13 | 76 | 0.594 | 0.434 | 0.067 | 0.305 | 1.122 |
| `block_metis / nonoverlap` | 1.000 | 13 | 76 | 0.599 | 0.875 | 0.067 | 0.320 | 1.585 |

### Level 3, `np=32`

| Variant | Elem dup | Newton | Sum KSP | Setup [s] | Assembly [s] | PC setup [s] | Solve [s] | Total [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FEniCS custom` | - | 14 | 81 | 0.088 | 0.134 | 0.070 | 0.250 | 0.580 |
| `JAX element current` | - | 12 | 84 | 1.942 | 0.255 | 0.212 | 1.197 | 2.065 |
| `none / overlap` | 2.392 | 12 | 84 | 0.606 | 0.319 | 0.229 | 1.149 | 2.252 |
| `none / nonoverlap` | 1.000 | 12 | 84 | 0.831 | 0.667 | 0.223 | 1.219 | 2.418 |
| `block_rcm / overlap` | 1.106 | 15 | 76 | 0.684 | 0.261 | 0.094 | 0.316 | 1.190 |
| `block_rcm / nonoverlap` | 1.000 | 15 | 76 | 0.594 | 0.860 | 0.094 | 0.310 | 1.710 |
| `block_xyz / overlap` | 1.098 | 12 | 71 | 0.597 | 0.210 | 0.089 | 0.245 | 0.956 |
| `block_xyz / nonoverlap` | 1.000 | 12 | 71 | 0.701 | 0.669 | 0.085 | 0.249 | 1.291 |
| `block_metis / overlap` | 1.138 | 13 | 64 | 0.629 | 0.233 | 0.078 | 0.279 | 0.963 |
| `block_metis / nonoverlap` | 1.000 | 13 | 64 | 0.778 | 0.748 | 0.095 | 0.279 | 1.479 |

### Level 3 Takeaways

- The full nonlinear run confirms the fixed-state result:
  `overlap_allgather` is consistently better end-to-end than
  `nonoverlap_allreduce`.
- Natural ordering is still bad on the real solve:
  at `np=32`, `none / overlap` is `2.252 s`, versus `0.956 s` for
  `block_xyz / overlap`.
- The reordered experimental path closes most of the original JAX gap:
  at `np=32`, `JAX element current` is `2.065 s`, while the best reordered
  variant is `0.956 s`.
- FEniCS custom is still the reference on `level 3`, but the gap drops from
  about `3.6x` to about `1.5x–1.7x`.

### Level 4, `np=16`

On `level 4`, only the `overlap_allgather` strategy was carried forward,
because on `level 3` the nonoverlap variant was never competitive.

| Variant | Elem dup | Newton | Sum KSP | Setup [s] | Assembly [s] | PC setup [s] | Solve [s] | Total [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FEniCS custom` | - | 15 | 84 | 0.259 | 1.536 | 0.464 | 4.114 | 7.622 |
| `JAX element current` | - | 13 | 72 | 10.709 | 3.950 | 1.610 | 12.503 | 25.509 |
| `none / overlap` | 1.936 | 13 | 72 | 1.095 | 4.692 | 1.613 | 12.542 | 26.628 |
| `block_rcm / overlap` | 1.024 | 14 | 72 | 1.064 | 3.928 | 0.422 | 3.590 | 11.893 |
| `block_xyz / overlap` | 1.023 | 14 | 69 | 1.043 | 3.914 | 0.486 | 3.413 | 11.766 |
| `block_metis / overlap` | 1.045 | 13 | 68 | 1.064 | 3.641 | 0.445 | 3.381 | 11.133 |

### Level 4, `np=32`

| Variant | Elem dup | Newton | Sum KSP | Setup [s] | Assembly [s] | PC setup [s] | Solve [s] | Total [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FEniCS custom` | - | 18 | 102 | 0.255 | 1.531 | 0.430 | 4.492 | 7.938 |
| `JAX element current` | - | 12 | 75 | 7.085 | 2.163 | 1.343 | 10.720 | 19.702 |
| `none / overlap` | 2.112 | 12 | 75 | 0.904 | 2.607 | 1.324 | 11.165 | 19.545 |
| `block_rcm / overlap` | 1.051 | 14 | 63 | 0.845 | 2.157 | 0.361 | 2.688 | 7.348 |
| `block_xyz / overlap` | 1.049 | 13 | 53 | 0.861 | 1.975 | 0.321 | 2.776 | 7.108 |
| `block_metis / overlap` | 1.086 | 16 | 82 | 0.853 | 2.485 | 0.400 | 3.209 | 8.705 |

### Level 4 Takeaways

- On the finest mesh, the reordered overlap strategy removes the large
  distribution regression.
- At `np=16`, the best experimental variant is `block_metis / overlap` at
  `11.133 s`, versus `25.509 s` for the current JAX element path and
  `7.622 s` for FEniCS custom.
- At `np=32`, the best experimental variant is `block_xyz / overlap` at
  `7.108 s`, versus `19.702 s` for the current JAX element path and
  `7.938 s` for FEniCS custom.
- So on the fine `np=32` case the experimental reordered element path is now
  slightly faster than FEniCS custom, while the natural-order and current JAX
  paths are still about `2.5x-2.8x` slower.
- The biggest remaining effect of reordering is still in `pc_setup` and
  `solve`, not in raw element assembly.

## Stage 3: Production Integration

The stage-2 winner was moved into the production HE JAX+PETSc solver.

Production behavior now is:

- `--assembly_mode element` uses a dedicated
  `HEReorderedElementAssembler`
- the element path uses the `overlap_allgather` strategy
- the element path accepts `--element_reorder_mode`
- if not set explicitly, production defaults to `block_xyz`
- the old reduced free-DOF SFD path is left unchanged for
  `--assembly_mode sfd`
- `solve_HE_dof.py` now configures the thread/XLA environment before importing
  the JAX solver stack

This was done in production rather than by further mutating the generic
assembler base because the winning design is currently specific to the HE
element path:

- exact vmapped element grad/Hessian
- reordered PETSc ownership
- overlap domain large enough to compute all owned rows locally
- `Allgatherv`-based state reconstruction

### Production verification

One production verification run was repeated on the main fine case:

- `level 4`
- `step 1 / 96`
- `np=32`
- `GMRES + GAMG`
- `--assembly_mode element`
- `--element_reorder_mode block_xyz`

| Variant | Newton | Sum KSP | Setup [s] | Assembly [s] | PC setup [s] | Solve [s] | Step [s] | Total [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FEniCS custom` | 18 | 102 | 0.192 | 1.141 | 0.299 | 2.774 | 5.142 | 5.334 |
| `JAX element current` | 12 | 75 | 6.722 | 2.163 | 1.267 | 9.550 | 16.397 | 23.120 |
| `JAX element production` | 13 | 53 | 7.499 | 1.978 | 0.346 | 1.818 | 5.856 | 14.112 |

Notes:

- The production `Step [s]` nearly closes the old gap to FEniCS custom.
- The linear part is now much closer to FEniCS than to the old JAX element
  implementation.
- The production `Setup [s]` is not directly comparable to the stage-2
  prototype `Setup [s]` column. The prototype setup tables started later and
  did not include the full reorder/layout/subdomain build cost that the real
  solver now reports.

### Updated Recommendation

For HyperElasticity element assembly, the current production recommendation is:

1. Use `--assembly_mode element`.
2. Keep the reordered ownership path.
3. Prefer `--element_reorder_mode block_xyz` by default.
4. Treat `block_rcm` as the first alternative to recheck on new machines.
5. Avoid the natural order on the fine meshes.

## Remaining Limit

The production change removed the large distribution/layout regression in the
element path, but it still uses the reduced free-DOF PETSc matrix layout.

So the remaining structural question is narrower now:

- not “does reordered ownership help?” — it does
- but whether a full-space PETSc matrix with constrained rows kept would
  remove the remaining setup and solver differences even further

## Artifacts

- The investigation harnesses are committed:
  [bench_he_element_distribution.py](/home/michal/repos/fenics_nonlinear_energies/experiment_scripts/bench_he_element_distribution.py)
  and
  [run_he_element_distribution_step.py](/home/michal/repos/fenics_nonlinear_energies/experiment_scripts/run_he_element_distribution_step.py).
- The raw JSON files used during the sweep were kept only as temporary local
  cache artifacts and are not part of the cleanup commit.

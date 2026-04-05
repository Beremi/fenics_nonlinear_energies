# Plasticity3D

This model card describes the heterogeneous 3D Mohr-Coulomb
strength-reduction benchmark implemented under
`src/problems/slope_stability_3d/`. It is the 3D companion to the maintained
2D plasticity page, but it follows the source benchmark more directly:
heterogeneous materials, source Gmsh physical-group labels, gravity along the
source `y` axis, same-mesh tetrahedral `P1/P2/P4` spaces, and autodiff of one
scalar 3D potential.

The primary documented result on this page is a corrected-frame
`P2(L1), lambda = 1.6` solve run from scratch with an elastic initial guess.
The detailed Octave-vs-JAX branch comparison is kept in the annex.

## Mathematical Formulation

The implemented 3D total potential keeps one scalar element energy and derives
the local residual and tangent from JAX autodiff:

$$
J(u_{\mathrm{free}})
= \sum_{e=1}^{n_{\mathrm{el}}} \Pi_e(u_e) - f_{\mathrm{ext}}^T u,
\qquad
\Pi_e(u_e)
= \sum_{q=1}^{n_q} w_{eq}\,
\psi_{\mathrm{MC},3D}\!\left(\varepsilon_{eq}(u_e), c_{\lambda,eq}, \sin\phi_{\lambda,eq},
\mu_{eq}, K_{eq}, \lambda_{eq}\right).
$$

The constitutive path is intentionally source-faithful:

- Davis-B reduction remains runtime-controlled by `lambda`
- the local density is the 3D scalar Mohr-Coulomb potential, not a hand-coded
  analytic tangent
- `jax.grad` and `jax.hessian` are taken directly on the scalar element energy
- the engineering-strain row order is `[xx, yy, zz, xy, yz, xz]`

## Geometry, Materials, And Boundary Conditions

- raw source assets: `data/meshes/SlopeStability3D/hetero_ssr/`
- maintained source meshes: `SSR_hetero_ada_L1.msh` through `L5`
- material physical groups map to logical IDs `0..3`
- boundary physical groups keep the source component-wise convention:
  - `x` constrained on labels `[1, 2]`
  - `y` constrained on label `[5]`
  - `z` constrained on labels `[3, 4]`
- gravity acts only in negative `y`
- the imported `.msh` meshes are treated as macro tetra meshes, then elevated
  locally to same-mesh `P1`, `P2`, and `P4`

## Discretisation Contract

- `P1`: `4` scalar nodes per tetra, `1` tet quadrature point
- `P2`: `10` scalar nodes per tetra, `11` tet quadrature points
- `P4`: `35` scalar nodes per tetra, `24` tet quadrature points
- hot-path geometry is stored as `dphix`, `dphiy`, `dphiz`, `quad_weight`
- heterogeneous material arrays are stored per quadrature point
- reordered PETSc assembly uses the target repo's overlap-domain scaffold

## Computed `L1` Hierarchy Card

The global hierarchy sizes below come from the generated
`hetero_ssr_L1_p{1,2,4}_same_mesh.h5` snapshots.

| space | nodes | macro tetrahedra | free DOFs | free `x` | free `y` | free `z` | tet quadrature |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `P1(L1)` | `3845` | `18419` | `10859` | `3732` | `3651` | `3476` | `1` |
| `P2(L1)` | `27605` | `18419` | `80362` | `27210` | `26883` | `26269` | `11` |
| `P4(L1)` | `208549` | `18419` | `616322` | `207077` | `205766` | `203479` | `24` |

The maintained same-mesh p-cascades are:

$$
P2(L1) \rightarrow P1(L1),
\qquad
P4(L1) \rightarrow P2(L1) \rightarrow P1(L1).
$$

Additional source-benchmark facts for the card:

- mesh name: `hetero_ssr_L1`
- material IDs present: `[0, 1, 2, 3]`
- gravity axis: `1` (`y`)
- `lambda_target_default = 1.0`

## Primary Solve Card: `P2(L1), lambda = 1.6`

This is the main validated 3D solve currently documented on the page. It is a
fresh run from scratch, not a continuation replay, and it uses an elastic
bootstrap before switching to the pure Mohr-Coulomb tangent.

### Solver stack

- mesh / space: `hetero_ssr_L1`, same-mesh `P2`
- nonlinear solver: Newton with Armijo line search
- initial guess: elastic solve on the same `P2 -> P1` PMG stack
- Newton tangent: pure plastic autodiff tangent
- Krylov solver: `fgmres`
- multigrid strategy: same-mesh `P2 -> P1`
- problem build mode: `root_bcast`
- MG level build mode: `root_bcast`
- transfer build mode: `owned_rows`
- distribution strategy: `overlap_p2p`
- reorder mode: `block_xyz`
- `P2` smoother: `richardson + sor(3)`
- coarse solve: `cg + hypre`
- Hypre coarse options:
  - `nodal_coarsen = 6`
  - `vec_interp_variant = 3`
  - `strong_threshold = 0.5`
  - `coarsen_type = HMIS`
  - `max_iter = 2`
  - `tol = 0.0`
  - `relax_type_all = symmetric-SOR/Jacobi`
- near-nullspace:
  - `P1` coarse level attached: `True`
  - `P2` level attached: `True`

### Outcome summary

Measured values from
`artifacts/raw_results/docs_showcase/plasticity3d_p2_l1_lambda1p6_from_scratch/output.json`:

| quantity | value |
| --- | ---: |
| MPI ranks | `1` |
| nonlinear iterations | `19` |
| final gradient norm | `5.584717e-03` |
| absolute gradient target | `1.0e-02` |
| energy | `-3.2221362908e6` |
| `omega` | `6.5042284288e6` |
| `u_max` | `1.0903602131e+00` |
| total linear iterations | `128` |
| nonlinear solve time | `277.26 s` |
| end-to-end wall time | `295.55 s` |

### Elastic bootstrap

The from-scratch run starts from the elastic solution instead of `u = 0`:

| quantity | value |
| --- | ---: |
| enabled | `True` |
| success | `True` |
| KSP / PC | `fgmres + mg` |
| linear iterations | `2` |
| RHS norm | `3.477654e5` |
| residual norm | `7.255930e2` |
| solution norm | `7.273820e1` |
| solve time | `2.04 s` |

### Representative Newton progress

The early iterations do the large shape change, the middle iterations settle
the slip zone, and the last iterations clean up the gradient to the requested
`1e-2` absolute target.

| it | energy | `||g||` at start | `||g||` after step | `alpha` | KSP its | true lin. rel. res. | step norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `-3.1542580262e6` | `3.040551e4` | `-` | `1.5625e-2` | `11` | `9.263e-3` | `6.964e0` |
| `2` | `-3.2156764582e6` | `4.573431e4` | `-` | `1.0` | `4` | `6.415e-3` | `4.108e1` |
| `9` | `-3.2218778287e6` | `6.268898e3` | `-` | `0.5` | `4` | `9.217e-3` | `8.943e-1` |
| `12` | `-3.2221321693e6` | `5.970165e2` | `-` | `0.5` | `9` | `7.381e-3` | `9.908e-1` |
| `17` | `-3.2221362830e6` | `1.516464e1` | `-` | `0.25` | `11` | `8.633e-3` | `1.862e-2` |
| `19` | `-3.2221362908e6` | `4.835178e-1` | `5.584717e-03` | `1.0` | `17` | `9.774e-3` | `1.278e-2` |

## Convergence And Final-State Visuals

The convergence panel below is from the saved from-scratch Newton history for
`P2(L1), lambda = 1.6`.

![P2(L1), lambda=1.6 convergence summary](../assets/plasticity3d/plasticity3d_p2_l1_lambda1p6_from_scratch_convergence.png)

The displacement plot shows the deformed outer boundary colored by
displacement magnitude `||u||`.

![P2(L1), lambda=1.6 displacement magnitude](../assets/plasticity3d/plasticity3d_p2_l1_lambda1p6_from_scratch_displacement.png)

The next three plots show separate source-style top-view slices of the
deviatoric-strain field. Regions where the plane does not intersect the
deformed tetra mesh are left blank on purpose.

### Deviatoric-strain slice: `x`

![P2(L1), lambda=1.6 deviatoric-strain x-slice](../assets/plasticity3d/plasticity3d_p2_l1_lambda1p6_from_scratch_deviatoric_strain_slice_x.png)

### Deviatoric-strain slice: `y`

![P2(L1), lambda=1.6 deviatoric-strain y-slice](../assets/plasticity3d/plasticity3d_p2_l1_lambda1p6_from_scratch_deviatoric_strain_slice_y.png)

### Deviatoric-strain slice: `z`

![P2(L1), lambda=1.6 deviatoric-strain z-slice](../assets/plasticity3d/plasticity3d_p2_l1_lambda1p6_from_scratch_deviatoric_strain_slice_z.png)

## Maintained Implementation Summary

| component | role |
| --- | --- |
| source mesh importer | ports source `.msh` assets and writes target-style HDF5 snapshots |
| same-mesh tetra elevation | builds `P1`, `P2`, and `P4` spaces on the imported macro tetra mesh |
| JAX constitutive kernel | evaluates the 3D scalar Mohr-Coulomb potential and autodiff derivatives |
| reordered PETSc assembler | assembles vector-valued residuals and tangents on reordered free DOFs |
| same-mesh PMG | supports `P2 -> P1` and `P4 -> P2 -> P1` owned-row transfer construction |

## Annex: Octave vs JAX `P2` Direct-Branch Comparison

The detailed source comparison is kept in the annex so the main card can stay
focused on the from-scratch `lambda = 1.6` solve. This annex compares the
source Octave direct-continuation branch against the JAX/PETSc replay at the
same accepted `lambda` values.

### Structural checks at final `lambda = 1.6`

| check | value |
| --- | --- |
| node map max abs diff | `1.421085e-14` |
| free-DOF mask exact | `True` |
| force relative diff | `5.422233e-16` |
| force max abs diff | `3.092282e-11` |
| macro P2 tet vertex connectivity exact | `True` |
| material IDs exact after vertex alignment | `True` |

Raw 10-node `P2` tetra arrays are not byte-identical because the higher-order
edge-node numbering differs between the two implementations.

### Accepted branch schedule

Both implementations accept the same direct-continuation lambda schedule:

`[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]`

### Branch table

| step | `lambda` | Octave work | JAX work | Octave direct `omega` | JAX direct `omega` | Octave `u_max` | JAX `u_max` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `1.0` | `6214942.30706` | `6214942.33106` | `32337.1518025` | `32337.0006857` | `0.814994708362` | `0.796656983182` |
| `2` | `1.1` | `6229234.33492` | `6229234.48852` | `72514.1502673` | `72514.0154024` | `0.814994708362` | `0.8149939666` |
| `3` | `1.2` | `6255872.56562` | `6255873.81475` | `130336.488982` | `130337.039189` | `0.841943200998` | `0.84193153619` |
| `4` | `1.3` | `6295404.82772` | `6295407.01458` | `190746.516473` | `190747.043537` | `0.870518103134` | `0.870513897092` |
| `5` | `1.4` | `6346885.00781` | `6346887.47692` | `247582.454206` | `247587.450324` | `0.900173290567` | `0.900043757977` |
| `6` | `1.5` | `6410935.53493` | `6410937.11614` | `305357.137731` | `305357.684177` | `0.936054607267` | `0.936262046721` |
| `7` | `1.6` | `6503975.85558` | `6504228.01611` | `386405.243445` | `386416.448986` | `1.08538061419` | `1.09035254789` |

Source note: in `init_phase_SSR_direct_continuation`, the Octave code stores
`Umax_hist(1)` from `U` instead of `U_old`, so the first-row source `u_max` is
a source-history bookkeeping quirk rather than a true state mismatch.

### Final `lambda = 1.6` comparison summary

- work relative diff: `3.877021e-05`
- displacement relative L2 diff: `3.517247e-03`
- deviatoric-strain relative L2 diff: `8.857115e-03`

### Branch visual summary

![Octave vs JAX branch summary](../assets/plasticity3d/plasticity3d_p2_compare_branch_summary.png)

### Final deformed boundary comparison

![Octave vs JAX deformed boundary comparison](../assets/plasticity3d/plasticity3d_p2_compare_deformed_boundary.png)

### Final deviatoric-strain slice comparison: `x`

![Octave vs JAX deviatoric-strain x-slice comparison](../assets/plasticity3d/plasticity3d_p2_compare_deviatoric_strain_slice_x.png)

### Final deviatoric-strain slice comparison: `y`

![Octave vs JAX deviatoric-strain y-slice comparison](../assets/plasticity3d/plasticity3d_p2_compare_deviatoric_strain_slice_y.png)

### Final deviatoric-strain slice comparison: `z`

![Octave vs JAX deviatoric-strain z-slice comparison](../assets/plasticity3d/plasticity3d_p2_compare_deviatoric_strain_slice_z.png)

## Caveats

- The primary published result on this page is a `P2(L1)` solve card. The
  `P4(L1) -> P2(L1) -> P1(L1)` cascade is implemented and smoke-tested, but the
  page is not currently presenting a full corrected-frame `P4` convergence
  study.
- The surface and slice plots focus on deviatoric strain because that is also
  the source benchmark's primary visual field for the 3D slope example.
- The current 3D implementation still uses zero plastic-history placeholders
  (`eps_p_old = 0`) rather than a full path-consistent history update loop.
- The maintained 2D plasticity path under `src/problems/slope_stability/`
  remains unchanged.

## Where To Go Next

- 2D plane-strain model card: [Plasticity](Plasticity.md)
- 2D maintained results: [Plasticity results](../results/Plasticity.md)
- setup and environment: [quickstart](../setup/quickstart.md)

## Reproduction Commands

Import the source `L1` mesh into same-mesh `P1/P2/P4` HDF5 snapshots:

```bash
./.venv/bin/python -m src.problems.slope_stability_3d.support.import_source_mesh \
  --mesh_name hetero_ssr_L1 --degree 1 --overwrite

./.venv/bin/python -m src.problems.slope_stability_3d.support.import_source_mesh \
  --mesh_name hetero_ssr_L1 --degree 2 --overwrite

./.venv/bin/python -m src.problems.slope_stability_3d.support.import_source_mesh \
  --mesh_name hetero_ssr_L1 --degree 4 --overwrite
```

Run the documented from-scratch `P2(L1), lambda = 1.6` solve with an elastic
initial guess:

```bash
./.venv/bin/python -u -m src.problems.slope_stability_3d.jax_petsc.solve_slope_stability_3d_dof \
  --mesh_name hetero_ssr_L1 --elem_degree 2 --lambda-target 1.6 \
  --profile performance --pc_type mg --mg_strategy same_mesh_p2_p1 \
  --ksp_type fgmres --ksp_rtol 1e-2 --ksp_max_it 80 \
  --problem_build_mode root_bcast --mg_level_build_mode root_bcast \
  --mg_transfer_build_mode owned_rows --distribution_strategy overlap_p2p \
  --element_reorder_mode block_xyz --line_search armijo \
  --elastic_initial_guess --no-regularized_newton_tangent \
  --mg_p1_smoother_ksp_type richardson --mg_p1_smoother_pc_type sor --mg_p1_smoother_steps 3 \
  --mg_p2_smoother_ksp_type richardson --mg_p2_smoother_pc_type sor --mg_p2_smoother_steps 3 \
  --mg_coarse_backend hypre --mg_coarse_ksp_type cg --mg_coarse_pc_type hypre \
  --mg_coarse_hypre_nodal_coarsen 6 --mg_coarse_hypre_vec_interp_variant 3 \
  --mg_coarse_hypre_strong_threshold 0.5 --mg_coarse_hypre_coarsen_type HMIS \
  --mg_coarse_hypre_max_iter 2 --mg_coarse_hypre_tol 0.0 \
  --mg_coarse_hypre_relax_type_all symmetric-SOR/Jacobi \
  --tolg 1e-2 --tolg_rel 0.0 --maxit 50 --save_history \
  --out artifacts/raw_results/docs_showcase/plasticity3d_p2_l1_lambda1p6_from_scratch/output.json \
  --state-out artifacts/raw_results/docs_showcase/plasticity3d_p2_l1_lambda1p6_from_scratch/state.npz \
  --progress-out artifacts/raw_results/docs_showcase/plasticity3d_p2_l1_lambda1p6_from_scratch/progress.json
```

Generate the documentation figures from the saved state and solver JSON:

```bash
./.venv/bin/python experiments/analysis/generate_plasticity3d_p2_lambda1p6_docs_assets.py \
  --state artifacts/raw_results/docs_showcase/plasticity3d_p2_l1_lambda1p6_from_scratch/state.npz \
  --result artifacts/raw_results/docs_showcase/plasticity3d_p2_l1_lambda1p6_from_scratch/output.json \
  --out-dir docs/assets/plasticity3d
```

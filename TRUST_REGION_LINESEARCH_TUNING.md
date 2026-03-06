# Trust-Region + Line-Search Tuning Notes

Date: 2026-03-05

## TL;DR

Settings that currently work best:

- p-Laplace:
  disable trust region and keep plain line search
  with `linesearch_interval=[-0.5, 2.0]`

- HyperElasticity, small case (`level 2`, `step 1 / 96`, `16` MPI):
  `use_trust_region=True`, `trust_radius_init=0.2`,
  `linesearch_interval=[0, 1]`

- HyperElasticity, large case (`level 4`, `step 1 / 24`, `32` MPI):
  simplest robust setting is
  `use_trust_region=False`, `linesearch_interval=[0, 1]`

- HyperElasticity, large case with trust region kept on:
  `use_trust_region=True`, `trust_radius_init` around `0.5` to `1.0`,
  `linesearch_interval=[0, 1]`

Settings to avoid on the large HE case:

- `trust_radius_init <= 0.2`
- default line-search interval `[-0.5, 2.0]`

## Scope

I reran smaller MPI cases to understand the behavior of the MATLAB-style trust-region
step that was ported from `trust-region-matlab-minimizer`.

Test problems:

- p-Laplace: `level 7`, `16` MPI ranks
- HyperElasticity: `level 2`, `16` MPI ranks, `step 1 / 96`

Implementations used:

- FEniCS custom Newton for the main sweep
- JAX + PETSc `element` mode for validation

Linear solver settings:

- p-Laplace: `CG + HYPRE`, `ksp_rtol=1e-3`
- HyperElasticity: `GMRES + GAMG`, `ksp_rtol=1e-3`, `ksp_max_it=10000`,
  `gamg_threshold=-1.0`, `gamg_agg_nsmooths=1`

## Important code-level finding

The current implementation is not a full trust-region method in the usual PETSc / Nocedal-Wright sense.

In `tools_petsc4py/minimizers.py`:

- `trust_shrink`, `trust_expand`, `trust_eta_shrink`, `trust_eta_expand`, and `trust_max_reject`
  are accepted as arguments but immediately discarded.
- the trust-radius update uses a hard-coded `dump = 10.0`
- `rho` is never computed
- step acceptance is based only on whether the line-searched step decreases the energy

So the *effective* tuning knobs right now are much smaller than the interface suggests:

- `trust_radius_init`
- `trust_radius_min` / `trust_radius_max`
- `linesearch_a`, `linesearch_b`
- `linesearch_tol`

Practically, this means the current algorithm behaves more like:

1. build a trust-region-constrained trial direction in a 2D subspace
2. run golden-section line search on that direction
3. accept only if the new energy is lower
4. shrink or relax the radius using the hard-coded `dump` rule

## Main sweep: FEniCS custom

### p-Laplace, level 7, 16 ranks

| Case | Trust region | LS interval | Radius init | Newton iters | Sum KSP iters | Total [s] | Result |
|---|---:|---|---:|---:|---:|---:|---|
| `ls_only_default` | no | `[-0.5, 2.0]` | 1.0 | 6 | 18 | 0.231 | converged |
| `tr_default` | yes | `[-0.5, 2.0]` | 1.0 | 9 | 27 | 0.359 | converged |
| `tr_r0_2` | yes | `[-0.5, 2.0]` | 0.2 | 11 | 33 | 0.440 | converged |
| `tr_r5` | yes | `[-0.5, 2.0]` | 5.0 | 7 | 21 | 0.289 | converged |
| `tr_pos_default` | yes | `[0.0, 2.0]` | 1.0 | 9 | 27 | 0.362 | converged |
| `tr_pos_r0_2` | yes | `[0.0, 2.0]` | 0.2 | 11 | 33 | 0.448 | converged |
| `tr_pos1_r0_2` | yes | `[0.0, 1.0]` | 0.2 | 100 | 296 | 3.577 | maxit |

Readout:

- Plain line search is clearly best.
- If trust region must stay enabled, `trust_radius_init=5.0` is the least bad of the tested settings.
- Restricting the line search to `[0, 1]` with a small initial radius is catastrophic here.

### HyperElasticity, level 2, step 1/96, 16 ranks

| Case | Trust region | LS interval | Radius init | Newton iters | Sum KSP iters | Total [s] | Result |
|---|---:|---|---:|---:|---:|---:|---|
| `ls_only_default` | no | `[-0.5, 2.0]` | 1.0 | 10 | 240 | 0.205 | converged |
| `tr_default` | yes | `[-0.5, 2.0]` | 1.0 | 10 | 240 | 0.213 | converged |
| `tr_r0_2` | yes | `[-0.5, 2.0]` | 0.2 | 10 | 240 | 0.214 | converged |
| `tr_r5` | yes | `[-0.5, 2.0]` | 5.0 | 10 | 240 | 0.246 | converged |
| `tr_pos_default` | yes | `[0.0, 2.0]` | 1.0 | 10 | 240 | 0.212 | converged |
| `tr_pos_r0_2` | yes | `[0.0, 2.0]` | 0.2 | 10 | 240 | 0.214 | converged |
| `tr_pos1_r0_2` | yes | `[0.0, 1.0]` | 0.2 | 9 | 183 | 0.185 | converged |

Readout:

- On this smaller HE case, most settings are nearly identical.
- The only tested setting that gave a real improvement was:
  `trust_radius_init=0.2`, `linesearch_interval=[0.0, 1.0]`
- Large initial radius (`5.0`) is slightly worse.

## Validation on JAX + PETSc element assembly

### p-Laplace, level 7, 16 ranks

| Case | Newton iters | Sum KSP iters | Total [s] | Result |
|---|---:|---:|---:|---|
| `ls_only_default` | 6 | 19 | 1.231 | converged |
| `tr_default` | 14 | 49 | 1.610 | converged |
| `tr_r5` | 10 | 35 | 1.423 | converged |
| `tr_pos1_r0_2` | 100 | 400 | 5.690 | maxit |

This confirms the p-Laplace trend from FEniCS:

- trust region hurts
- `tr_r5` is only a partial recovery
- `[0, 1]` with `radius=0.2` is again a failure mode

### HyperElasticity, level 2, step 1/96, 16 ranks

| Case | Newton iters | Sum KSP iters | Total [s] | Result |
|---|---:|---:|---:|---|
| `ls_only_default` | 10 | 192 | 1.795 | converged |
| `tr_default` | 10 | 193 | 1.797 | converged |
| `tr_r5` | 10 | 192 | 1.801 | converged |
| `tr_pos1_r0_2` | 9 | 163 | 1.752 | converged |

This confirms the HE trend from FEniCS:

- default trust-region settings do not help much
- `[0, 1]` with `radius=0.2` is the best of the tested trust-region settings
- the gain is modest on the small case, but consistent across both backends

## Line-search tolerance check

I also checked whether `linesearch_tol` is worth tuning aggressively.

### p-Laplace, line-search only, FEniCS custom

| `linesearch_tol` | Newton iters | Sum KSP iters | Total [s] |
|---:|---:|---:|---:|
| `1e-2` | 6 | 18 | 0.216 |
| `1e-3` | 6 | 18 | 0.231 |
| `1e-4` | 6 | 18 | 0.255 |

### HyperElasticity, best tested trust-region case, FEniCS custom

Case: `trust_radius_init=0.2`, `linesearch_interval=[0, 1]`

| `linesearch_tol` | Newton iters | Sum KSP iters | Total [s] |
|---:|---:|---:|---:|
| `1e-2` | 9 | 183 | 0.184 |
| `1e-3` | 9 | 183 | 0.185 |
| `1e-4` | 9 | 183 | 0.186 |

Readout:

- `linesearch_tol` did not change convergence behavior in these small tests
- tighter tolerance only increased runtime slightly
- this is a second-order knob compared with interval choice and initial trust radius

## Practical recommendations

### p-Laplace

Recommended:

- keep plain line search only
- if trust region must remain enabled, the least bad tested setting was:
  `trust_radius_init=5.0`, `linesearch_interval=[-0.5, 2.0]`

Not recommended:

- `trust_radius_init=0.2` with `linesearch_interval=[0, 1]`

### HyperElasticity

Best tested small-case setting:

- `trust_radius_init=0.2`
- `linesearch_interval=[0.0, 1.0]`
- `linesearch_tol=1e-2` or `1e-3`

But:

- this was only a small-case result
- it is **not** robust on the large `level 4`, `step 1 / 24` case
- for the large case, a separate sweep is needed

## 32-rank follow-up on larger cases

I also checked the recommended settings on larger runs.

### p-Laplace, finest mesh, 32 ranks

Setting used:

- trust region off
- `linesearch_interval=[-0.5, 2.0]`
- `linesearch_tol=1e-3`

| Backend | Mesh | Newton iters | Sum KSP iters | Total [s] | Result |
|---|---|---:|---:|---:|---|
| FEniCS custom | `level 9` | 6 | 23 | 1.130 | converged |
| JAX element | `level 9` | 7 | 26 | 3.902 | converged |

Readout:

- this confirms the earlier recommendation
- the large p-Laplace case is fine with plain line search
- there is still no evidence that trust region helps p-Laplace

### HyperElasticity, `step 1 / 24`, 32 ranks

Setting used:

- trust region on
- `trust_radius_init=0.2`
- `linesearch_interval=[0.0, 1.0]`
- `linesearch_tol=1e-3`

#### Coarse first: `level 2`

| Backend | Newton iters | Sum KSP iters | Total [s] | Result |
|---|---:|---:|---:|---|
| FEniCS custom | 23 | 1040 | 0.787 | converged |
| JAX element | 23 | 956 | 2.842 | converged |

#### Then finest mesh: `level 4`

| Backend | Newton iters | Sum KSP iters | Total [s] | Result |
|---|---:|---:|---:|---|
| FEniCS custom | 100 | 4512 | 64.836 | maxit |
| JAX element | 100 | 4385 | 152.426 | maxit |

Readout:

- the tuned HE setting does work on the coarse `1/24` case
- it does **not** transfer to the fine `level 4` `1/24` case
- both backends fail in the same way: they reach `maxit=100` with very large linear-iteration counts

At that point the conclusion was:

- p-Laplace recommendation is stable
- the specific HE setting `trust_radius_init=0.2`, `linesearch_interval=[0, 1]`
  is not robust to the fine `1/24` case
- this motivated a direct large-case sweep

## Focused large-case HE sweep

I then ran a focused sweep directly on the problematic case:

- HyperElasticity
- `level 4`
- `step 1 / 24`
- `32` MPI ranks

The sweep targeted the only knobs that are actually live in the current implementation:

- `use_trust_region`
- `trust_radius_init`
- `linesearch_interval`

### FEniCS custom sweep

| Case | Trust region | LS interval | Radius init | Newton iters | Sum KSP iters | Total [s] | Result |
|---|---:|---|---:|---:|---:|---:|---|
| `ls_default` | no | `[-0.5, 2.0]` | 1.0 | 25 | 12132 | 164.145 | converged |
| `ls_pos1` | no | `[0.0, 1.0]` | 1.0 | 26 | 2126 | 32.476 | converged |
| `tr_r0_05_pos1` | yes | `[0.0, 1.0]` | 0.05 | 100 | 4160 | 72.465 | maxit |
| `tr_r0_1_pos1` | yes | `[0.0, 1.0]` | 0.1 | 58 | 3240 | 54.511 | converged |
| `tr_r0_2_pos1` | yes | `[0.0, 1.0]` | 0.2 | 100 | 5211 | 94.156 | maxit |
| `tr_r0_5_pos1` | yes | `[0.0, 1.0]` | 0.5 | 25 | 2077 | 33.443 | converged |
| `tr_r1_pos1` | yes | `[0.0, 1.0]` | 1.0 | 25 | 2008 | 33.364 | converged |
| `tr_r5_pos1` | yes | `[0.0, 1.0]` | 5.0 | 25 | 2048 | 34.091 | converged |
| `tr_r0_2_pos2` | yes | `[0.0, 2.0]` | 0.2 | 50 | 2564 | 45.799 | converged |

Readout:

- The dominant improvement is changing the line-search interval to `[0, 1]`.
- The old `[-0.5, 2]` interval is disastrous here because it drives huge KSP work.
- With trust region enabled, `trust_radius_init=0.2` is actually too small on the large case.
- The large case wants either:
  - no trust region, `linesearch_interval=[0, 1]`, or
  - trust region with `trust_radius_init` around `0.5` to `1.0`

### JAX element validation

I validated the strongest large-case settings plus the old failing `r=0.2` case.

| Case | Trust region | LS interval | Radius init | Newton iters | Sum KSP iters | Total [s] | Result |
|---|---:|---|---:|---:|---:|---:|---|
| `ls_pos1` | no | `[0.0, 1.0]` | 1.0 | 23 | 1422 | 47.097 | converged |
| `tr_r0_2_pos1` | yes | `[0.0, 1.0]` | 0.2 | 100 | 4199 | 149.680 | maxit |
| `tr_r0_5_pos1` | yes | `[0.0, 1.0]` | 0.5 | 23 | 1387 | 47.032 | converged |
| `tr_r1_pos1` | yes | `[0.0, 1.0]` | 1.0 | 23 | 1389 | 46.889 | converged |

Readout:

- The JAX backend confirms the same pattern as FEniCS.
- The bad large-case setting is specifically `trust_radius_init=0.2`.
- A forward-only line search already fixes the large case.
- Trust region with `trust_radius_init` around `0.5` to `1.0` is also fine, but only marginally different from line-search-only.

### Updated HE recommendation

For the large `level 4`, `step 1 / 24`, `32`-rank case:

- simplest robust choice:
  `use_trust_region=False`, `linesearch_interval=[0, 1]`
- if trust region must stay enabled:
  `trust_radius_init` in the range `0.5` to `1.0`, with `linesearch_interval=[0, 1]`
- avoid:
  `trust_radius_init <= 0.2` on this large case

## What I think is actually going on

The current picture is more nuanced than I first thought.

There are still structural issues:

- the algorithm exposes trust-region parameters that are not wired into the actual update
- it does not use a standard model-vs-actual reduction ratio
- it always line-searches the trust-region direction instead of using trust-region acceptance directly

But the focused large-case sweep shows the HE failure was **not** purely structural.

The large-case HE behavior is strongly controlled by:

- the line-search interval
- the initial trust radius

What seems to be happening is:

- p-Laplace prefers pure Newton + line search
- HyperElasticity strongly prefers a forward-only line search on `[0, 1]`
- on the large HE case, too-small trust radii (`0.2` and below) over-constrain the step and can cause stagnation
- moderate trust radii (`0.5` to `1.0`) are acceptable, but they do not beat the simple `[0, 1]` line-search-only option by much
- the crude radius update is still a weakness, but it is not the only story

## Files with raw results

- `experiment_results_cache/trust_region_tuning_small_fenics_16r.json`
- `experiment_results_cache/trust_region_tuning_small_jax_element_16r.json`
- `experiment_results_cache/trust_region_linesearch_tol_small_fenics_16r.json`
- `experiment_results_cache/trust_region_followup_32r.json`
- `experiment_results_cache/he_large_case_param_sweep_fenics_32r.json`
- `experiment_results_cache/he_large_case_param_sweep_jax_32r.json`

## If I were to change the code next

I would do one of these two things, not stay in the current middle ground:

1. simplify: keep the line-search Newton path as the main method and drop the current trust-region branch
2. make it real: implement a standard trust-region acceptance/update using `rho`, and wire the exposed trust parameters into that logic

For the current code as written, the tuning advice I would carry forward is:

- p-Laplace: disable trust region
- HyperElasticity small case: `trust_radius_init=0.2`, `linesearch_interval=[0, 1]` is fine
- HyperElasticity large case: prefer `linesearch_interval=[0, 1]`, and if trust region stays on, use `trust_radius_init` around `0.5` to `1.0`

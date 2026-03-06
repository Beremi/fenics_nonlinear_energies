# Trust-Region + Line-Search Tuning Notes

Date: 2026-03-06

## TL;DR

Current recommendations after the `rho`-based trust-region implementation:

- p-Laplace:
  keep trust region off.
  Plain line search is still best, and now the trust-region variants are
  clearly worse on both FEniCS and JAX backends.

- HyperElasticity, small case (`level 2`, `step 1 / 96`, `16` MPI):
  the best tested setting is still
  `use_trust_region=True`, `trust_radius_init=0.2`,
  `linesearch_interval=[0, 1]`.

- HyperElasticity, coarse `1/24` follow-up (`level 2`, `step 1 / 24`, `32` MPI):
  that same setting still works on both backends.

- HyperElasticity, fine `1/96` case (`level 4`, `step 1 / 96`, `32` MPI):
  that same setting also works on both backends. The fine mesh itself is not
  the failure trigger.

- HyperElasticity, fine `1/24` case (`level 4`, `step 1 / 24`, `32` MPI):
  `trust_radius_init=0.2`, `linesearch_interval=[0, 1]` is still a bad setting.
  On the new code the FEniCS run was even worse: I aborted it after about
  10 minutes wall time.

## What changed in the code

The previous version exposed trust-region parameters but did not actually use a
standard trust-region acceptance/update rule.

That is now fixed in `tools_petsc4py/minimizers.py`:

- `rho = actual_reduction / predicted_reduction` is computed
- trust-step acceptance uses `rho`
- `trust_shrink`, `trust_expand`, `trust_eta_shrink`,
  `trust_eta_expand`, and `trust_max_reject` are all live
- rejected trust steps now shrink the radius and retry
- accepted trust steps can expand the radius when the model is accurate

The implemented algorithm is documented in:

- `TRUST_REGION_LINESEARCH_ALGORITHM.md`

## Important caveat about these reruns

The 32-rank wall-clock timings in this local rerun are much slower than the
older measurements in the previous note. The nonlinear iteration patterns are
still useful, but the absolute 32-rank times here should be treated as
environment-specific.

In particular, the finest p-Laplace FEniCS rerun on 32 ranks was much slower
than the older cached result, while the JAX/PETSc run was not. So for the large
cases, the iteration counts and convergence behavior are more trustworthy than
the raw seconds.

## Small sweep: 16 ranks

### p-Laplace, `level 7`

#### FEniCS custom

| Case | Trust region | LS interval | Radius init | Newton iters | Sum KSP iters | Total [s] | Result |
|---|---:|---|---:|---:|---:|---:|---|
| `ls_only_default` | no | `[-0.5, 2.0]` | 1.0 | 6 | 18 | 0.239 | converged |
| `tr_default` | yes | `[-0.5, 2.0]` | 1.0 | 11 | 33 | 0.460 | converged |
| `tr_r0_2` | yes | `[-0.5, 2.0]` | 0.2 | 15 | 45 | 0.599 | converged |
| `tr_r5` | yes | `[-0.5, 2.0]` | 5.0 | 9 | 27 | 0.370 | converged |
| `tr_pos_default` | yes | `[0.0, 2.0]` | 1.0 | 11 | 33 | 0.424 | converged |
| `tr_pos_r0_2` | yes | `[0.0, 2.0]` | 0.2 | 15 | 45 | 0.592 | converged |
| `tr_pos1_r0_2` | yes | `[0.0, 1.0]` | 0.2 | 16 | 48 | 0.603 | converged |

#### JAX + PETSc element validation

| Case | Newton iters | Sum KSP iters | Total [s] | Result |
|---|---:|---:|---:|---|
| `ls_only_default` | 6 | 19 | 0.409 | converged |
| `tr_default` | 18 | 65 | 1.076 | converged |
| `tr_r5` | 14 | 49 | 0.907 | converged |
| `tr_pos1_r0_2` | 24 | 87 | 1.485 | converged |

Readout:

- After switching to a real `rho`-based trust-region update, p-Laplace now
  clearly prefers plain line search.
- This is true on both backends.
- The old “trust region is only slightly worse” picture is gone.

### Line-search tolerance check: p-Laplace, line-search only, FEniCS custom

| `linesearch_tol` | Newton iters | Sum KSP iters | Total [s] |
|---:|---:|---:|---:|
| `1e-2` | 6 | 18 | 0.234 |
| `1e-3` | 6 | 18 | 0.267 |
| `1e-4` | 6 | 18 | 0.256 |

Readout:

- `linesearch_tol` is still a second-order knob here.
- It does not change convergence behavior.

### HyperElasticity, `level 2`, `step 1 / 96`

#### FEniCS custom

| Case | Trust region | LS interval | Radius init | Newton iters | Sum KSP iters | Total [s] | Result |
|---|---:|---|---:|---:|---:|---:|---|
| `ls_only_default` | no | `[-0.5, 2.0]` | 1.0 | 10 | 240 | 0.179 | converged |
| `tr_default` | yes | `[-0.5, 2.0]` | 1.0 | 10 | 240 | 0.208 | converged |
| `tr_r0_2` | yes | `[-0.5, 2.0]` | 0.2 | 10 | 240 | 0.191 | converged |
| `tr_r5` | yes | `[-0.5, 2.0]` | 5.0 | 10 | 240 | 0.193 | converged |
| `tr_pos_default` | yes | `[0.0, 2.0]` | 1.0 | 10 | 240 | 0.190 | converged |
| `tr_pos_r0_2` | yes | `[0.0, 2.0]` | 0.2 | 10 | 240 | 0.202 | converged |
| `tr_pos1_r0_2` | yes | `[0.0, 1.0]` | 0.2 | 8 | 153 | 0.161 | converged |

#### JAX + PETSc element validation

| Case | Newton iters | Sum KSP iters | Total [s] | Result |
|---|---:|---:|---:|---|
| `ls_only_default` | 10 | 192 | 0.563 | converged |
| `tr_default` | 10 | 192 | 0.597 | converged |
| `tr_r5` | 10 | 192 | 0.587 | converged |
| `tr_pos1_r0_2` | 8 | 137 | 0.496 | converged |

Readout:

- The small-case HE conclusion did not change.
- `trust_radius_init=0.2` with `linesearch_interval=[0, 1]` is still the best
  tested setting on both backends.
- The gain is modest but real.

### Line-search tolerance check: HyperElasticity, best small-case setting

Case:

- `use_trust_region=True`
- `trust_radius_init=0.2`
- `linesearch_interval=[0, 1]`

| `linesearch_tol` | Newton iters | Sum KSP iters | Total [s] |
|---:|---:|---:|---:|
| `1e-2` | 8 | 153 | 0.146 |
| `1e-3` | 8 | 153 | 0.147 |
| `1e-4` | 8 | 153 | 0.144 |

Readout:

- `linesearch_tol` again does not matter much.
- The interval and trust radius dominate the behavior.

## 32-rank follow-up

### p-Laplace, finest mesh, line-search only

Setting used:

- trust region off
- `linesearch_interval=[-0.5, 2.0]`

| Backend | Mesh | Newton iters | Sum KSP iters | Total [s] | Result |
|---|---|---:|---:|---:|---|
| FEniCS custom | `level 9` | 6 | 23 | 43.339 | converged |
| JAX element | `level 9` | 7 | 26 | 1.793 | converged |

Readout:

- The iteration pattern still confirms the p-Laplace recommendation:
  trust region is unnecessary.
- The absolute wall times here differ strongly from the older note, so these
  timings should not be compared directly to the earlier cached numbers.

### HyperElasticity, coarse `step 1 / 24`

Setting used:

- trust region on
- `trust_radius_init=0.2`
- `linesearch_interval=[0.0, 1.0]`

| Backend | Mesh | Newton iters | Sum KSP iters | Total [s] | Result |
|---|---|---:|---:|---:|---|
| FEniCS custom | `level 2` | 23 | 1050 | 0.812 | converged |
| JAX element | `level 2` | 23 | 958 | 2.259 | converged |

Readout:

- The small-case HE winner still transfers to the coarse `1/24` case.
- It is not a generally bad setting; it is specifically a bad **fine `1/24`**
  setting.

### HyperElasticity, finest mesh, `step 1 / 96`

Setting used:

- trust region on
- `trust_radius_init=0.2`
- `linesearch_interval=[0.0, 1.0]`

| Backend | Mesh | Newton iters | Sum KSP iters | Total [s] | Result |
|---|---|---:|---:|---:|---|
| FEniCS custom | `level 4` | 9 | 272 | 238.798 | converged |
| JAX element | `level 4` | 7 | 155 | 16.988 | converged |

Readout:

- The tuned small-case HE setting does transfer to the fine `1/96` case.
- Fine mesh alone is not the issue.
- The difficult case is the larger `1/24` load step on the fine mesh.

### HyperElasticity, fine `step 1 / 24`

Attempted setting:

- trust region on
- `trust_radius_init=0.2`
- `linesearch_interval=[0.0, 1.0]`

Result in this rerun:

- FEniCS custom, `level 4`, `32` ranks:
  aborted after about 10 minutes wall time
- I did not complete the symmetric JAX fine-case rerun in this pass after the
  FEniCS timeout made it clear that this remains a bad large-case setting

Readout:

- The same qualitative conclusion as before still holds:
  `trust_radius_init=0.2`, `linesearch_interval=[0, 1]` is not robust on the
  fine `level 4`, `step 1 / 24` case.
- On the new `rho`-based updater, the FEniCS version appears even less viable
  than before.

## Practical recommendations

### p-Laplace

Recommended:

- `use_trust_region=False`
- `linesearch_interval=[-0.5, 2.0]`

Not recommended:

- all tested trust-region variants in this post-`rho` rerun

### HyperElasticity

Recommended for the small case:

- `use_trust_region=True`
- `trust_radius_init=0.2`
- `linesearch_interval=[0, 1]`

Recommended for the coarse `1/24` follow-up:

- same as above

Recommended for the fine `1/96` case:

- same as above

Not recommended for the fine `level 4`, `step 1 / 24` case:

- `trust_radius_init=0.2`
- `linesearch_interval=[0, 1]`

The next thing to test, once the local 32-rank environment issue is out of the
way, is still the moderate-radius large-case family:

- no trust region, `linesearch_interval=[0, 1]`
- trust region with `trust_radius_init` around `0.5` to `1.0`

## Raw result files

- `experiment_results_cache/trust_region_rho_plaplace_small_16r.json`
- `experiment_results_cache/trust_region_rho_he_small_16r.json`
- `experiment_results_cache/trust_region_rho_he_small_fenics_detail_16r.json`
- `experiment_results_cache/plaplace_fenics_ls_only_default_l9_32r.json`
- `experiment_results_cache/plaplace_jax_ls_only_default_l9_32r.json`
- `experiment_results_cache/he_fenics_tr_pos1_r0_2_l2s1of24_32r.json`
- `experiment_results_cache/he_fenics_tr_pos1_r0_2_l2s1of24_32r_detail.json`
- `experiment_results_cache/he_jax_tr_pos1_r0_2_l2s1of24_32r.json`
- `experiment_results_cache/he_fenics_tr_pos1_r0_2_l4s1of96_32r_detail.json`
- `experiment_results_cache/he_jax_tr_pos1_r0_2_l4s1of96_32r.json`
- `experiment_results_cache/trust_region_rho_followup_32r_partial.json`

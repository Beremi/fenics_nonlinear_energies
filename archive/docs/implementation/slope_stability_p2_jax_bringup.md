# Slope Stability P2 JAX Bring-Up

## Goal

Add an experimental pure-JAX slope-stability family that:

- uses the external `run_2D_homo_SSR_capture` procedural `P2` mesh
- solves one fixed reduced-strength endpoint near the reported SSR benchmark
- integrates cleanly into this repository without promoting it to a maintained
  fifth benchmark family yet

This document records the actual implementation trail, including failed
attempts and the corrections made during validation.

## External Benchmark Facts Captured

Source repository examined:

- `https://github.com/Beremi/slope_stability_petsc4py`
- benchmark folder: `benchmarks/run_2D_homo_SSR_capture`
- procedural mesh source: `src/slope_stability/mesh/slope_2d.py`

Key benchmark facts copied into the prototype:

- analysis: 2D homogeneous SSR
- element type: `P2`
- Davis reduction type: `B`
- geometry:
  - `h = 1.0`
  - `x1 = 15.0`
  - `x2 = 10.0`
  - `x3 = 15.0`
  - `y1 = 10.0`
  - `y2 = 10.0`
  - `beta = 45 deg`
- raw material:
  - `c0 = 6.0`
  - `phi = 45.0 deg`
  - `psi = 0.0 deg`
  - `E = 40000.0`
  - `nu = 0.30`
  - `gamma = 20.0`
- reported PETSc final accepted load factor from the benchmark report:
  - `lambda = 1.21132390718`

Important finding:

- the external benchmark mesh is not a checked-in `.msh` file for this 2D
  case; it is generated procedurally from the geometry parameters above

## Implementation Choices

Implemented repository family:

- `src/problems/slope_stability/support/`
- `src/problems/slope_stability/jax/`
- frozen snapshot:
  `data/meshes/SlopeStability/ssr_homo_capture_p2_h1.h5`

What was matched exactly:

- procedural `P2` mesh topology and coordinates
- free-DOF mask convention from the external code
- 7-point `P2` triangle quadrature
- gravity load assembly sign and magnitude
- Davis-`B` strength reduction formula

What was intentionally simplified:

- constitutive model stayed aligned with the pasted `triangle_energy_mc` idea
  rather than the external PETSc total-strain constitutive operator
- plastic history is frozen to
  `eps_p_old = 0` at every element/quadrature point
- the solver targets one fixed endpoint solve at `lambda = 1.21`
  instead of running SSR continuation to the reported terminal value

Consequence:

- this is a zero-history endpoint prototype, not a path-consistent plasticity
  benchmark reproduction

## Trials, Failures, Corrections

### Trial 1: Freeze the external procedural mesh into repo-native HDF5

Ported the mesh generator and assembled:

- `nodes`
- `elems_scalar`
- vector `elems`
- `elem_B`
- `quad_weight`
- `force`
- `freedofs`
- `u_0`
- `eps_p_old`
- free-DOF adjacency

Result:

- success
- frozen snapshot written to
  `data/meshes/SlopeStability/ssr_homo_capture_p2_h1.h5`

Validated counts:

- nodes: `2721`
- triangles: `1300`
- free `x` DOFs: `2580`
- free `y` DOFs: `2640`
- total free DOFs: `5220`

### Trial 2: First end-to-end pure-JAX solve

Command used:

```bash
./.venv/bin/python -u src/problems/slope_stability/jax/solve_slope_stability_jax.py \
  --quiet \
  --json artifacts/tmp_slope_stability_smoke/output.json \
  --state-out artifacts/tmp_slope_stability_smoke/state.npz
```

Observed result:

- status: `completed`
- final energy: `-212.44500996086214`
- `u_max`: `0.07670122680558222`
- Newton iterations: `8`
- linear iterations: `39`
- total wall time: about `2.72 s`

This confirmed:

- the mesh/load assembly was internally consistent
- the elastic initial guess was good enough for the prototype endpoint solve
- the trust-region serial path behaved well on this case

### Failure 1: NaN gradient at the zero-displacement state in tests

Failing test:

- `tests/test_slope_stability_jax.py`

Symptom:

- energy values were finite
- gradient at the zero state came back as all `NaN`

Root cause:

- `two_theta = arctan2(2*txy, sxx-syy)` was being computed outside the plastic
  branch
- at the zero-stress elastic state this meant differentiating through
  `arctan2(0, 0)` even though the plastic branch was not physically active

Correction:

- moved the `two_theta` computation inside the plastic branch only

Result after correction:

- zero-state and elastic-initial-state energy, gradient, and sparse Hessian all
  became finite

### Failure 2: Runner summary path failed for temp directories outside the repo

Failing test:

- `tests/test_slope_stability_runner.py`

Symptom:

- `json_out.relative_to(REPO_ROOT)` raised when the runner wrote into a pytest
  temp directory outside the repository tree

Correction:

- added a display-path helper that falls back to an absolute path when the
  output is outside the repo

Result after correction:

- runner summary generation worked both in-repo and in temporary test output
  directories

## Final Verification

Targeted verification run:

```bash
./.venv/bin/pytest -q \
  tests/test_problem_data_hdf5.py \
  tests/test_cli_output_directories.py \
  tests/test_docs_asset_cli_exports.py \
  tests/test_slope_stability_support.py \
  tests/test_slope_stability_jax.py \
  tests/test_slope_stability_runner.py
```

Outcome:

- `18 passed`

Additional docs-surface verification:

```bash
./.venv/bin/pytest -q \
  tests/test_current_doc_paths.py \
  tests/test_docs_publication.py::test_docs_use_only_current_repo_relative_paths \
  tests/test_docs_publication.py::test_current_docs_have_one_problem_and_one_results_page_per_family
```

Outcome:

- `3 passed`

## Current Limitations

- no continuation path to the reported PETSc terminal `lambda`
- no quadrature-point history evolution
- no maintained docs/results family page
- no claim of numerical parity with the external PETSc constitutive benchmark

## Next Logical Upgrade

If this prototype is extended, the highest-value next step is:

1. replace the zero-history endpoint constitutive path with a history-aware
   continuation formulation
2. keep the frozen mesh snapshot, CLI surface, loader, and runner unchanged
3. compare the resulting continuation history against the external PETSc report

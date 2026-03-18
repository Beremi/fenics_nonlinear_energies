# jax_petsc_element_l7_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `7` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `49665` |
| Free DOFs | `48641` |
| Setup time [s] | `0.753` |
| Total solve time [s] | `0.707` |
| Total Newton iterations | `7` |
| Total linear iterations | `14` |
| Total assembly time [s] | `0.067` |
| Total PC init time [s] | `0.362` |
| Total KSP solve time [s] | `0.115` |
| Total line-search time [s] | `0.147` |
| Final energy | `-7.958292` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l7_np1.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l7_np1.log` |

## Frozen Settings

| Setting | Value |
|---|---|
| `ksp_type` | `cg` |
| `pc_type` | `hypre` |
| `ksp_rtol` | `0.1` |
| `ksp_max_it` | `30` |
| `use_trust_region` | `False` |
| `trust_subproblem_line_search` | `False` |
| `linesearch_interval` | `[-0.5, 2.0]` |
| `linesearch_tol` | `0.1` |
| `trust_radius_init` | `1.0` |
| `local_hessian_mode` | `element` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.707 | 7 | 14 | -7.958292 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.707` |
| Newton iterations | `7` |
| Linear iterations | `14` |
| Energy | `-7.958292` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.384065 | 521138.922978 | 28062.385991 | nan | 1.956948 | 2 | yes | 9 | 0.013 | 0.062 | 0.017 | 0.021 | 1.000000 | nan |
| 2 | -3.923058 | 9.307123 | 13.011986 | nan | 0.862712 | 2 | yes | 9 | 0.012 | 0.055 | 0.016 | 0.021 | 1.000000 | nan |
| 3 | -7.347342 | 3.424284 | 9.356425 | nan | 0.809497 | 2 | yes | 9 | 0.008 | 0.059 | 0.018 | 0.021 | 1.000000 | nan |
| 4 | -7.927872 | 0.580530 | 3.171588 | nan | 1.002033 | 2 | yes | 9 | 0.008 | 0.053 | 0.017 | 0.021 | 1.000000 | nan |
| 5 | -7.957650 | 0.029779 | 0.652471 | nan | 1.055248 | 2 | yes | 9 | 0.008 | 0.047 | 0.016 | 0.021 | 1.000000 | nan |
| 6 | -7.958285 | 0.000635 | 0.082565 | nan | 1.088137 | 2 | yes | 9 | 0.008 | 0.044 | 0.016 | 0.021 | 1.000000 | nan |
| 7 | -7.958292 | 0.000008 | 0.007239 | 0.000366 | 1.055248 | 2 | yes | 9 | 0.010 | 0.044 | 0.016 | 0.021 | 1.000000 | nan |

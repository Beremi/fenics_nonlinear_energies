# jax_petsc_element_l7_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `7` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `49665` |
| Free DOFs | `48641` |
| Setup time [s] | `0.425` |
| Total solve time [s] | `0.328` |
| Total Newton iterations | `7` |
| Total linear iterations | `14` |
| Total assembly time [s] | `0.022` |
| Total PC init time [s] | `0.179` |
| Total KSP solve time [s] | `0.049` |
| Total line-search time [s] | `0.070` |
| Final energy | `-7.958292` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l7_np4.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l7_np4.log` |

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
| 1 | 0.328 | 7 | 14 | -7.958292 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.328` |
| Newton iterations | `7` |
| Linear iterations | `14` |
| Energy | `-7.958292` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.402543 | 521138.904500 | 28062.385991 | nan | 1.956948 | 2 | yes | 9 | 0.004 | 0.029 | 0.007 | 0.011 | 1.000000 | nan |
| 2 | -3.944636 | 9.347179 | 13.011639 | nan | 0.862712 | 2 | yes | 9 | 0.004 | 0.026 | 0.007 | 0.010 | 1.000000 | nan |
| 3 | -7.346908 | 3.402272 | 9.260810 | nan | 0.809497 | 2 | yes | 9 | 0.003 | 0.027 | 0.007 | 0.010 | 1.000000 | nan |
| 4 | -7.928141 | 0.581233 | 3.145850 | nan | 1.002033 | 2 | yes | 9 | 0.003 | 0.026 | 0.007 | 0.010 | 1.000000 | nan |
| 5 | -7.957658 | 0.029517 | 0.644505 | nan | 1.055248 | 2 | yes | 9 | 0.003 | 0.024 | 0.007 | 0.010 | 1.000000 | nan |
| 6 | -7.958285 | 0.000627 | 0.081840 | nan | 1.088137 | 2 | yes | 9 | 0.003 | 0.024 | 0.007 | 0.010 | 1.000000 | nan |
| 7 | -7.958292 | 0.000007 | 0.007170 | 0.000348 | 1.055248 | 2 | yes | 9 | 0.003 | 0.024 | 0.007 | 0.010 | 1.000000 | nan |

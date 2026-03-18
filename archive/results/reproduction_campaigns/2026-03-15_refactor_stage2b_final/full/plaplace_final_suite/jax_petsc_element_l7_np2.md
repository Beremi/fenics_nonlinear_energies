# jax_petsc_element_l7_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `7` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `49665` |
| Free DOFs | `48641` |
| Setup time [s] | `0.532` |
| Total solve time [s] | `0.445` |
| Total Newton iterations | `7` |
| Total linear iterations | `14` |
| Total assembly time [s] | `0.037` |
| Total PC init time [s] | `0.236` |
| Total KSP solve time [s] | `0.070` |
| Total line-search time [s] | `0.092` |
| Final energy | `-7.958292` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l7_np2.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l7_np2.log` |

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
| 1 | 0.445 | 7 | 14 | -7.958292 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.445` |
| Newton iterations | `7` |
| Linear iterations | `14` |
| Energy | `-7.958292` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.438106 | 521138.868937 | 28062.385991 | nan | 1.956948 | 2 | yes | 9 | 0.005 | 0.039 | 0.010 | 0.013 | 1.000000 | nan |
| 2 | -3.911265 | 9.349371 | 13.013050 | nan | 0.862712 | 2 | yes | 9 | 0.008 | 0.035 | 0.010 | 0.013 | 1.000000 | nan |
| 3 | -7.343915 | 3.432649 | 9.349464 | nan | 0.809497 | 2 | yes | 9 | 0.004 | 0.037 | 0.011 | 0.013 | 1.000000 | nan |
| 4 | -7.927456 | 0.583542 | 3.176365 | nan | 1.002033 | 2 | yes | 9 | 0.006 | 0.034 | 0.010 | 0.013 | 1.000000 | nan |
| 5 | -7.957639 | 0.030183 | 0.658291 | nan | 1.055248 | 2 | yes | 9 | 0.004 | 0.031 | 0.010 | 0.013 | 1.000000 | nan |
| 6 | -7.958285 | 0.000646 | 0.083574 | nan | 1.088137 | 2 | yes | 9 | 0.005 | 0.030 | 0.010 | 0.013 | 1.000000 | nan |
| 7 | -7.958292 | 0.000008 | 0.007360 | 0.000415 | 1.088137 | 2 | yes | 9 | 0.005 | 0.030 | 0.010 | 0.013 | 1.000000 | nan |

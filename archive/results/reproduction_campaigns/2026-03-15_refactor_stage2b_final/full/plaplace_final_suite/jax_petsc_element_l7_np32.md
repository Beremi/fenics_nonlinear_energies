# jax_petsc_element_l7_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `7` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `49665` |
| Free DOFs | `48641` |
| Setup time [s] | `0.420` |
| Total solve time [s] | `0.414` |
| Total Newton iterations | `7` |
| Total linear iterations | `14` |
| Total assembly time [s] | `0.009` |
| Total PC init time [s] | `0.253` |
| Total KSP solve time [s] | `0.064` |
| Total line-search time [s] | `0.080` |
| Final energy | `-7.958292` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l7_np32.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l7_np32.log` |

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
| 1 | 0.414 | 7 | 14 | -7.958292 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.414` |
| Newton iterations | `7` |
| Linear iterations | `14` |
| Energy | `-7.958292` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.393331 | 521138.913712 | 28062.385991 | nan | 1.956948 | 2 | yes | 9 | 0.002 | 0.050 | 0.009 | 0.011 | 1.000000 | nan |
| 2 | -3.952888 | 9.346218 | 13.016851 | nan | 0.862712 | 2 | yes | 9 | 0.001 | 0.035 | 0.010 | 0.012 | 1.000000 | nan |
| 3 | -7.349719 | 3.396831 | 9.225148 | nan | 0.809497 | 2 | yes | 9 | 0.001 | 0.033 | 0.008 | 0.012 | 1.000000 | nan |
| 4 | -7.928308 | 0.578590 | 3.139429 | nan | 1.002033 | 2 | yes | 9 | 0.001 | 0.033 | 0.009 | 0.011 | 1.000000 | nan |
| 5 | -7.957663 | 0.029354 | 0.634869 | nan | 1.055248 | 2 | yes | 9 | 0.001 | 0.034 | 0.010 | 0.011 | 1.000000 | nan |
| 6 | -7.958285 | 0.000622 | 0.080389 | nan | 1.088137 | 2 | yes | 9 | 0.001 | 0.033 | 0.008 | 0.012 | 1.000000 | nan |
| 7 | -7.958292 | 0.000008 | 0.007280 | 0.000365 | 1.055248 | 2 | yes | 9 | 0.001 | 0.035 | 0.009 | 0.011 | 1.000000 | nan |

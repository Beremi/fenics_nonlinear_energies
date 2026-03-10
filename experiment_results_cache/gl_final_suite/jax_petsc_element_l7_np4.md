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
| Total DOFs | `66049` |
| Free DOFs | `65025` |
| Setup time [s] | `0.694` |
| Total solve time [s] | `0.253` |
| Total Newton iterations | `6` |
| Total linear iterations | `21` |
| Total assembly time [s] | `0.026` |
| Total PC init time [s] | `0.090` |
| Total KSP solve time [s] | `0.041` |
| Total line-search time [s] | `0.090` |
| Final energy | `0.345662` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_element_l7_np4.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_element_l7_np4.log` |

## Frozen Settings

| Setting | Value |
|---|---|
| `ksp_type` | `gmres` |
| `pc_type` | `hypre` |
| `ksp_rtol` | `0.001` |
| `ksp_max_it` | `200` |
| `use_trust_region` | `False` |
| `trust_subproblem_line_search` | `False` |
| `linesearch_interval` | `[-0.5, 2.0]` |
| `linesearch_tol` | `0.001` |
| `trust_radius_init` | `1.0` |
| `local_hessian_mode` | `element` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.253 | 6 | 21 | 0.345662 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.253` |
| Newton iterations | `6` |
| Linear iterations | `21` |
| Energy | `0.345662` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.494535 | 0.170775 | 0.003705 | nan | -0.331106 | 7 | yes | 19 | 0.005 | 0.017 | 0.013 | 0.015 | 1.000000 | nan |
| 2 | 0.482327 | 0.012208 | 0.003516 | nan | -0.496252 | 5 | yes | 19 | 0.005 | 0.015 | 0.009 | 0.015 | 1.000000 | nan |
| 3 | 0.376069 | 0.106257 | 0.005124 | nan | 0.198784 | 3 | yes | 19 | 0.004 | 0.014 | 0.006 | 0.015 | 1.000000 | nan |
| 4 | 0.346286 | 0.029783 | 0.003644 | nan | 0.913446 | 2 | yes | 19 | 0.004 | 0.014 | 0.004 | 0.015 | 1.000000 | nan |
| 5 | 0.345662 | 0.000624 | 0.000469 | nan | 1.028774 | 2 | yes | 19 | 0.004 | 0.014 | 0.004 | 0.015 | 1.000000 | nan |
| 6 | 0.345662 | 0.000000 | 0.000006 | 0.000000 | 1.000416 | 2 | yes | 19 | 0.004 | 0.014 | 0.005 | 0.015 | 1.000000 | nan |

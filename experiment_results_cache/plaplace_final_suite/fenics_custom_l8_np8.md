# fenics_custom_l8_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `8` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `197633` |
| Free DOFs | `197413` |
| Setup time [s] | `0.556` |
| Total solve time [s] | `0.651` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.060` |
| Total PC init time [s] | `0.274` |
| Total KSP solve time [s] | `0.111` |
| Total line-search time [s] | `0.177` |
| Final energy | `-7.959556` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/fenics_custom_l8_np8.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/fenics_custom_l8_np8.log` |

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
| `local_hessian_mode` | `-` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.651 | 6 | 12 | -7.959556 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.651` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.959556` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -3.939269 | 8.232049 | 11.226822 | nan | 0.723392 | 2 | yes | 9 | 0.011 | 0.050 | 0.018 | 0.029 | 1.000000 | nan |
| 2 | -7.314101 | 3.374832 | 8.827655 | nan | 0.776608 | 2 | yes | 9 | 0.010 | 0.047 | 0.019 | 0.033 | 1.000000 | nan |
| 3 | -7.920638 | 0.606537 | 3.337280 | nan | 0.948817 | 2 | yes | 9 | 0.010 | 0.043 | 0.018 | 0.029 | 1.000000 | nan |
| 4 | -7.958477 | 0.037839 | 0.716709 | nan | 1.002033 | 2 | yes | 9 | 0.010 | 0.042 | 0.018 | 0.029 | 1.000000 | nan |
| 5 | -7.959526 | 0.001049 | 0.107149 | nan | 1.055248 | 2 | yes | 9 | 0.010 | 0.045 | 0.019 | 0.029 | 1.000000 | nan |
| 6 | -7.959556 | 0.000030 | 0.014991 | 0.001132 | 1.141353 | 2 | yes | 9 | 0.010 | 0.047 | 0.019 | 0.029 | 1.000000 | nan |

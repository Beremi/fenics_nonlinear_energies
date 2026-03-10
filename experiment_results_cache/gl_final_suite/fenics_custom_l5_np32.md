# fenics_custom_l5_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `5` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `4225` |
| Free DOFs | `4225` |
| Setup time [s] | `0.035` |
| Total solve time [s] | `0.317` |
| Total Newton iterations | `10` |
| Total linear iterations | `38` |
| Total assembly time [s] | `0.003` |
| Total PC init time [s] | `0.171` |
| Total KSP solve time [s] | `0.096` |
| Total line-search time [s] | `0.025` |
| Final energy | `0.346232` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l5_np32.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l5_np32.log` |

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
| `local_hessian_mode` | `-` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.317 | 10 | 38 | 0.346232 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.317` |
| Newton iterations | `10` |
| Linear iterations | `38` |
| Energy | `0.346232` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.566343 | 0.099126 | 0.014806 | nan | -0.259130 | 10 | yes | 19 | 0.001 | 0.016 | 0.021 | 0.002 | 1.000000 | nan |
| 2 | 0.511454 | 0.054890 | 0.016340 | nan | 0.344736 | 5 | yes | 19 | 0.000 | 0.017 | 0.013 | 0.003 | 1.000000 | nan |
| 3 | 0.432418 | 0.079036 | 0.007957 | nan | -0.492586 | 2 | yes | 19 | 0.000 | 0.018 | 0.005 | 0.002 | 1.000000 | nan |
| 4 | 0.401713 | 0.030705 | 0.013572 | nan | 0.041403 | 6 | yes | 19 | 0.000 | 0.018 | 0.014 | 0.003 | 1.000000 | nan |
| 5 | 0.399428 | 0.002285 | 0.013127 | nan | 0.174792 | 3 | yes | 19 | 0.000 | 0.017 | 0.008 | 0.002 | 1.000000 | nan |
| 6 | 0.379137 | 0.020291 | 0.010902 | nan | 0.093053 | 3 | yes | 19 | 0.000 | 0.016 | 0.009 | 0.003 | 1.000000 | nan |
| 7 | 0.353656 | 0.025481 | 0.011681 | nan | 0.388787 | 3 | yes | 19 | 0.000 | 0.017 | 0.007 | 0.002 | 1.000000 | nan |
| 8 | 0.346316 | 0.007341 | 0.006687 | nan | 1.034705 | 2 | yes | 19 | 0.000 | 0.016 | 0.006 | 0.002 | 1.000000 | nan |
| 9 | 0.346232 | 0.000083 | 0.000561 | nan | 1.023276 | 2 | yes | 19 | 0.000 | 0.017 | 0.007 | 0.003 | 1.000000 | nan |
| 10 | 0.346232 | 0.000000 | 0.000006 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.000 | 0.020 | 0.007 | 0.003 | 1.000000 | nan |

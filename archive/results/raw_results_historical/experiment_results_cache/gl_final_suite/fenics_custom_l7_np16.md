# fenics_custom_l7_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `7` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `66049` |
| Free DOFs | `65898` |
| Setup time [s] | `0.174` |
| Total solve time [s] | `0.304` |
| Total Newton iterations | `7` |
| Total linear iterations | `30` |
| Total assembly time [s] | `0.013` |
| Total PC init time [s] | `0.136` |
| Total KSP solve time [s] | `0.079` |
| Total line-search time [s] | `0.052` |
| Final energy | `0.345662` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l7_np16.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l7_np16.log` |

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
| 1 | 0.304 | 7 | 30 | 0.345662 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.304` |
| Newton iterations | `7` |
| Linear iterations | `30` |
| Energy | `0.345662` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.534048 | 0.131262 | 0.003705 | nan | -0.299350 | 9 | yes | 19 | 0.002 | 0.022 | 0.022 | 0.008 | 1.000000 | nan |
| 2 | 0.510373 | 0.023675 | 0.003901 | nan | 0.352932 | 6 | yes | 19 | 0.002 | 0.020 | 0.015 | 0.007 | 1.000000 | nan |
| 3 | 0.440206 | 0.070167 | 0.002325 | nan | -0.455331 | 5 | yes | 19 | 0.002 | 0.020 | 0.013 | 0.007 | 1.000000 | nan |
| 4 | 0.369406 | 0.070800 | 0.003791 | nan | 0.238304 | 4 | yes | 19 | 0.002 | 0.019 | 0.011 | 0.007 | 1.000000 | nan |
| 5 | 0.346165 | 0.023241 | 0.002773 | nan | 0.911613 | 2 | yes | 19 | 0.002 | 0.019 | 0.006 | 0.007 | 1.000000 | nan |
| 6 | 0.345663 | 0.000502 | 0.000380 | nan | 1.043169 | 2 | yes | 19 | 0.002 | 0.019 | 0.006 | 0.008 | 1.000000 | nan |
| 7 | 0.345662 | 0.000000 | 0.000010 | 0.000000 | 1.001549 | 2 | yes | 19 | 0.002 | 0.018 | 0.006 | 0.007 | 1.000000 | nan |

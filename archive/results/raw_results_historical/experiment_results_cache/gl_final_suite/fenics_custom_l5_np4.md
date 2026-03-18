# fenics_custom_l5_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `5` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `4225` |
| Free DOFs | `4181` |
| Setup time [s] | `0.021` |
| Total solve time [s] | `0.087` |
| Total Newton iterations | `7` |
| Total linear iterations | `27` |
| Total assembly time [s] | `0.004` |
| Total PC init time [s] | `0.035` |
| Total KSP solve time [s] | `0.019` |
| Total line-search time [s] | `0.015` |
| Final energy | `0.346232` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l5_np4.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l5_np4.log` |

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
| 1 | 0.087 | 7 | 27 | 0.346232 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.087` |
| Newton iterations | `7` |
| Linear iterations | `27` |
| Energy | `0.346232` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.504288 | 0.161181 | 0.014806 | nan | -0.305981 | 8 | yes | 19 | 0.001 | 0.005 | 0.005 | 0.002 | 1.000000 | nan |
| 2 | 0.503178 | 0.001111 | 0.014312 | nan | -0.252333 | 5 | yes | 19 | 0.000 | 0.005 | 0.003 | 0.002 | 1.000000 | nan |
| 3 | 0.468743 | 0.034434 | 0.017846 | nan | -0.175125 | 6 | yes | 19 | 0.000 | 0.005 | 0.004 | 0.002 | 1.000000 | nan |
| 4 | 0.385297 | 0.083446 | 0.020360 | nan | 0.267260 | 2 | yes | 19 | 0.000 | 0.005 | 0.002 | 0.002 | 1.000000 | nan |
| 5 | 0.347450 | 0.037848 | 0.014671 | nan | 0.905415 | 2 | yes | 19 | 0.000 | 0.005 | 0.002 | 0.002 | 1.000000 | nan |
| 6 | 0.346233 | 0.001217 | 0.002418 | nan | 1.050233 | 2 | yes | 19 | 0.000 | 0.005 | 0.002 | 0.002 | 1.000000 | nan |
| 7 | 0.346232 | 0.000001 | 0.000057 | 0.000000 | 1.001116 | 2 | yes | 19 | 0.000 | 0.005 | 0.002 | 0.002 | 1.000000 | nan |

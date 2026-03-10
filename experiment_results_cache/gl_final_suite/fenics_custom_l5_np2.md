# fenics_custom_l5_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `5` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `4225` |
| Free DOFs | `4102` |
| Setup time [s] | `0.021` |
| Total solve time [s] | `0.072` |
| Total Newton iterations | `7` |
| Total linear iterations | `28` |
| Total assembly time [s] | `0.006` |
| Total PC init time [s] | `0.020` |
| Total KSP solve time [s] | `0.012` |
| Total line-search time [s] | `0.022` |
| Final energy | `0.346232` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l5_np2.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l5_np2.log` |

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
| 1 | 0.072 | 7 | 28 | 0.346232 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.072` |
| Newton iterations | `7` |
| Linear iterations | `28` |
| Energy | `0.346232` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.503394 | 0.162076 | 0.014806 | nan | -0.319244 | 8 | yes | 19 | 0.001 | 0.004 | 0.003 | 0.003 | 1.000000 | nan |
| 2 | 0.500927 | 0.002467 | 0.014536 | nan | -0.281989 | 6 | yes | 19 | 0.001 | 0.003 | 0.002 | 0.003 | 1.000000 | nan |
| 3 | 0.456337 | 0.044590 | 0.018337 | nan | -0.069127 | 5 | yes | 19 | 0.001 | 0.003 | 0.002 | 0.003 | 1.000000 | nan |
| 4 | 0.392625 | 0.063712 | 0.019271 | nan | 0.141470 | 3 | yes | 19 | 0.001 | 0.003 | 0.001 | 0.003 | 1.000000 | nan |
| 5 | 0.348493 | 0.044132 | 0.015740 | nan | 0.813812 | 2 | yes | 19 | 0.001 | 0.003 | 0.001 | 0.003 | 1.000000 | nan |
| 6 | 0.346233 | 0.002260 | 0.003507 | nan | 1.066462 | 2 | yes | 19 | 0.001 | 0.003 | 0.001 | 0.003 | 1.000000 | nan |
| 7 | 0.346232 | 0.000001 | 0.000110 | 0.000000 | 1.000684 | 2 | yes | 19 | 0.001 | 0.003 | 0.001 | 0.003 | 1.000000 | nan |

# jax_petsc_element_l5_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `5` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `4225` |
| Free DOFs | `3969` |
| Setup time [s] | `0.545` |
| Total solve time [s] | `0.080` |
| Total Newton iterations | `8` |
| Total linear iterations | `33` |
| Total assembly time [s] | `0.004` |
| Total PC init time [s] | `0.021` |
| Total KSP solve time [s] | `0.012` |
| Total line-search time [s] | `0.041` |
| Final energy | `0.346231` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_element_l5_np2.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_element_l5_np2.log` |

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
| 1 | 0.080 | 8 | 33 | 0.346231 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.080` |
| Newton iterations | `8` |
| Linear iterations | `33` |
| Energy | `0.346231` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.566939 | 0.098531 | 0.014806 | nan | -0.256864 | 9 | yes | 19 | 0.001 | 0.003 | 0.003 | 0.005 | 1.000000 | nan |
| 2 | 0.511131 | 0.055807 | 0.016193 | nan | 0.318478 | 6 | yes | 19 | 0.001 | 0.003 | 0.002 | 0.005 | 1.000000 | nan |
| 3 | 0.429210 | 0.081921 | 0.008793 | nan | -0.320109 | 4 | yes | 19 | 0.000 | 0.003 | 0.001 | 0.005 | 1.000000 | nan |
| 4 | 0.386588 | 0.042623 | 0.012844 | nan | 0.088687 | 5 | yes | 19 | 0.000 | 0.002 | 0.002 | 0.005 | 1.000000 | nan |
| 5 | 0.353522 | 0.033066 | 0.011966 | nan | 0.395418 | 3 | yes | 19 | 0.000 | 0.002 | 0.001 | 0.005 | 1.000000 | nan |
| 6 | 0.346286 | 0.007235 | 0.006395 | nan | 1.049101 | 2 | yes | 19 | 0.000 | 0.002 | 0.001 | 0.005 | 1.000000 | nan |
| 7 | 0.346231 | 0.000055 | 0.000474 | nan | 1.013246 | 2 | yes | 19 | 0.000 | 0.002 | 0.001 | 0.005 | 1.000000 | nan |
| 8 | 0.346231 | 0.000000 | 0.000004 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.000 | 0.002 | 0.001 | 0.005 | 1.000000 | nan |

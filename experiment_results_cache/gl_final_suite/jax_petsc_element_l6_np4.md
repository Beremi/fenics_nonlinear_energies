# jax_petsc_element_l6_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `6` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `16641` |
| Free DOFs | `16129` |
| Setup time [s] | `0.563` |
| Total solve time [s] | `0.091` |
| Total Newton iterations | `5` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.005` |
| Total PC init time [s] | `0.033` |
| Total KSP solve time [s] | `0.011` |
| Total line-search time [s] | `0.038` |
| Final energy | `0.345777` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_element_l6_np4.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_element_l6_np4.log` |

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
| 1 | 0.091 | 5 | 12 | 0.345777 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.091` |
| Newton iterations | `5` |
| Linear iterations | `12` |
| Energy | `0.345777` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.499824 | 0.165517 | 0.007408 | nan | -0.499650 | 3 | yes | 19 | 0.001 | 0.008 | 0.003 | 0.007 | 1.000000 | nan |
| 2 | 0.372583 | 0.127241 | 0.008373 | nan | 0.205415 | 3 | yes | 19 | 0.001 | 0.007 | 0.003 | 0.008 | 1.000000 | nan |
| 3 | 0.346214 | 0.026369 | 0.005339 | nan | 0.842870 | 2 | yes | 19 | 0.001 | 0.006 | 0.002 | 0.008 | 1.000000 | nan |
| 4 | 0.345777 | 0.000437 | 0.000761 | nan | 1.036971 | 2 | yes | 19 | 0.001 | 0.006 | 0.002 | 0.008 | 1.000000 | nan |
| 5 | 0.345777 | 0.000000 | 0.000017 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.001 | 0.007 | 0.002 | 0.008 | 1.000000 | nan |

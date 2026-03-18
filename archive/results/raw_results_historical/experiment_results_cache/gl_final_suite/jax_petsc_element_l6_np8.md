# jax_petsc_element_l6_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `6` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `16641` |
| Free DOFs | `16129` |
| Setup time [s] | `0.586` |
| Total solve time [s] | `0.190` |
| Total Newton iterations | `9` |
| Total linear iterations | `42` |
| Total assembly time [s] | `0.007` |
| Total PC init time [s] | `0.072` |
| Total KSP solve time [s] | `0.039` |
| Total line-search time [s] | `0.068` |
| Final energy | `0.345777` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_element_l6_np8.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_element_l6_np8.log` |

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
| 1 | 0.190 | 9 | 42 | 0.345777 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.190` |
| Newton iterations | `9` |
| Linear iterations | `42` |
| Energy | `0.345777` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.566666 | 0.098676 | 0.007408 | nan | -0.230606 | 11 | yes | 19 | 0.001 | 0.009 | 0.010 | 0.008 | 1.000000 | nan |
| 2 | 0.528159 | 0.038506 | 0.008049 | nan | 0.245801 | 6 | yes | 19 | 0.001 | 0.009 | 0.006 | 0.008 | 1.000000 | nan |
| 3 | 0.470486 | 0.057674 | 0.005434 | nan | -0.499650 | 6 | yes | 19 | 0.001 | 0.008 | 0.005 | 0.008 | 1.000000 | nan |
| 4 | 0.453827 | 0.016658 | 0.008306 | nan | 0.063562 | 6 | yes | 19 | 0.001 | 0.008 | 0.005 | 0.007 | 1.000000 | nan |
| 5 | 0.397879 | 0.055949 | 0.007966 | nan | 0.283056 | 4 | yes | 19 | 0.001 | 0.008 | 0.004 | 0.008 | 1.000000 | nan |
| 6 | 0.353017 | 0.044862 | 0.006224 | nan | 0.800384 | 3 | yes | 19 | 0.001 | 0.007 | 0.003 | 0.008 | 1.000000 | nan |
| 7 | 0.346022 | 0.006995 | 0.003380 | nan | 0.888754 | 2 | yes | 19 | 0.001 | 0.008 | 0.002 | 0.007 | 1.000000 | nan |
| 8 | 0.345777 | 0.000245 | 0.000502 | nan | 1.057998 | 2 | yes | 19 | 0.001 | 0.008 | 0.002 | 0.007 | 1.000000 | nan |
| 9 | 0.345777 | 0.000000 | 0.000013 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.001 | 0.008 | 0.002 | 0.007 | 1.000000 | nan |

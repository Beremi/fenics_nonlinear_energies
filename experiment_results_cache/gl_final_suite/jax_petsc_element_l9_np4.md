# jax_petsc_element_l9_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `9` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `1050625` |
| Free DOFs | `1046529` |
| Setup time [s] | `3.465` |
| Total solve time [s] | `5.857` |
| Total Newton iterations | `7` |
| Total linear iterations | `27` |
| Total assembly time [s] | `0.734` |
| Total PC init time [s] | `1.484` |
| Total KSP solve time [s] | `1.301` |
| Total line-search time [s] | `2.138` |
| Final energy | `0.345626` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_element_l9_np4.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_element_l9_np4.log` |

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
| 1 | 5.857 | 7 | 27 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `5.857` |
| Newton iterations | `7` |
| Linear iterations | `27` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.503061 | 0.162238 | 0.000926 | nan | -0.325608 | 7 | yes | 19 | 0.105 | 0.239 | 0.312 | 0.310 | 1.000000 | nan |
| 2 | 0.502212 | 0.000849 | 0.000929 | nan | -0.243169 | 6 | yes | 19 | 0.104 | 0.207 | 0.271 | 0.303 | 1.000000 | nan |
| 3 | 0.472838 | 0.029375 | 0.001142 | nan | -0.188120 | 5 | yes | 19 | 0.106 | 0.208 | 0.229 | 0.312 | 1.000000 | nan |
| 4 | 0.365951 | 0.106887 | 0.001325 | nan | 0.350667 | 3 | yes | 19 | 0.104 | 0.207 | 0.151 | 0.308 | 1.000000 | nan |
| 5 | 0.345942 | 0.020009 | 0.000756 | nan | 0.994485 | 2 | yes | 19 | 0.106 | 0.209 | 0.112 | 0.298 | 1.000000 | nan |
| 6 | 0.345627 | 0.000315 | 0.000074 | nan | 1.023708 | 2 | yes | 19 | 0.104 | 0.208 | 0.112 | 0.301 | 1.000000 | nan |
| 7 | 0.345626 | 0.000000 | 0.000001 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.105 | 0.206 | 0.113 | 0.307 | 1.000000 | nan |

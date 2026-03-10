# jax_petsc_element_l9_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `9` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `1050625` |
| Free DOFs | `1046529` |
| Setup time [s] | `5.483` |
| Total solve time [s] | `7.407` |
| Total Newton iterations | `7` |
| Total linear iterations | `29` |
| Total assembly time [s] | `0.971` |
| Total PC init time [s] | `2.169` |
| Total KSP solve time [s] | `1.575` |
| Total line-search time [s] | `2.463` |
| Final energy | `0.345626` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_element_l9_np2.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_element_l9_np2.log` |

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
| 1 | 7.407 | 7 | 29 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `7.407` |
| Newton iterations | `7` |
| Linear iterations | `29` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.536130 | 0.129170 | 0.000926 | nan | -0.296817 | 8 | yes | 19 | 0.139 | 0.340 | 0.394 | 0.351 | 1.000000 | nan |
| 2 | 0.509876 | 0.026255 | 0.000970 | nan | 0.355465 | 6 | yes | 19 | 0.138 | 0.303 | 0.305 | 0.348 | 1.000000 | nan |
| 3 | 0.439764 | 0.070112 | 0.000570 | nan | -0.426808 | 5 | yes | 19 | 0.140 | 0.310 | 0.266 | 0.353 | 1.000000 | nan |
| 4 | 0.365202 | 0.074562 | 0.000921 | nan | 0.268660 | 4 | yes | 19 | 0.139 | 0.304 | 0.219 | 0.353 | 1.000000 | nan |
| 5 | 0.345849 | 0.019352 | 0.000638 | nan | 0.956797 | 2 | yes | 19 | 0.139 | 0.301 | 0.130 | 0.348 | 1.000000 | nan |
| 6 | 0.345626 | 0.000223 | 0.000064 | nan | 1.016644 | 2 | yes | 19 | 0.138 | 0.305 | 0.129 | 0.354 | 1.000000 | nan |
| 7 | 0.345626 | 0.000000 | 0.000000 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.138 | 0.305 | 0.131 | 0.354 | 1.000000 | nan |

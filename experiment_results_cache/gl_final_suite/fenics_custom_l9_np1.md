# fenics_custom_l9_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `9` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `1050625` |
| Free DOFs | `1046529` |
| Setup time [s] | `2.881` |
| Total solve time [s] | `17.819` |
| Total Newton iterations | `7` |
| Total linear iterations | `22` |
| Total assembly time [s] | `2.386` |
| Total PC init time [s] | `2.658` |
| Total KSP solve time [s] | `2.438` |
| Total line-search time [s] | `9.398` |
| Final energy | `0.345626` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l9_np1.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l9_np1.log` |

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
| 1 | 17.819 | 7 | 22 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `17.819` |
| Newton iterations | `7` |
| Linear iterations | `22` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.495168 | 0.170132 | 0.000926 | nan | -0.369494 | 6 | yes | 19 | 0.360 | 0.426 | 0.593 | 1.345 | 1.000000 | nan |
| 2 | 0.487515 | 0.007652 | 0.000946 | nan | -0.330674 | 6 | yes | 19 | 0.338 | 0.373 | 0.593 | 1.344 | 1.000000 | nan |
| 3 | 0.468169 | 0.019346 | 0.001226 | nan | -0.141969 | 1 | yes | 19 | 0.338 | 0.373 | 0.165 | 1.343 | 1.000000 | nan |
| 4 | 0.389793 | 0.078376 | 0.001215 | nan | 0.139370 | 3 | yes | 19 | 0.338 | 0.373 | 0.335 | 1.340 | 1.000000 | nan |
| 5 | 0.347144 | 0.042649 | 0.000991 | nan | 0.864762 | 2 | yes | 19 | 0.337 | 0.370 | 0.253 | 1.329 | 1.000000 | nan |
| 6 | 0.345627 | 0.001517 | 0.000180 | nan | 1.055465 | 2 | yes | 19 | 0.336 | 0.369 | 0.249 | 1.342 | 1.000000 | nan |
| 7 | 0.345626 | 0.000001 | 0.000005 | 0.000000 | 1.000684 | 2 | yes | 19 | 0.340 | 0.374 | 0.250 | 1.355 | 1.000000 | nan |

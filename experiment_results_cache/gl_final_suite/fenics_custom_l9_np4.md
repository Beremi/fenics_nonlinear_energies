# fenics_custom_l9_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `9` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `1050625` |
| Free DOFs | `1049338` |
| Setup time [s] | `3.263` |
| Total solve time [s] | `5.303` |
| Total Newton iterations | `6` |
| Total linear iterations | `22` |
| Total assembly time [s] | `0.588` |
| Total PC init time [s] | `1.213` |
| Total KSP solve time [s] | `1.039` |
| Total line-search time [s] | `2.223` |
| Final energy | `0.345626` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l9_np4.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l9_np4.log` |

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
| 1 | 5.303 | 6 | 22 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `5.303` |
| Newton iterations | `6` |
| Linear iterations | `22` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.496363 | 0.168937 | 0.000926 | nan | -0.317844 | 8 | yes | 19 | 0.104 | 0.228 | 0.343 | 0.370 | 1.000000 | nan |
| 2 | 0.485217 | 0.011147 | 0.000907 | nan | -0.485255 | 5 | yes | 19 | 0.097 | 0.197 | 0.222 | 0.368 | 1.000000 | nan |
| 3 | 0.390899 | 0.094318 | 0.001324 | nan | 0.122009 | 3 | yes | 19 | 0.097 | 0.196 | 0.147 | 0.370 | 1.000000 | nan |
| 4 | 0.347084 | 0.043815 | 0.001088 | nan | 0.866162 | 2 | yes | 19 | 0.096 | 0.196 | 0.109 | 0.371 | 1.000000 | nan |
| 5 | 0.345627 | 0.001457 | 0.000187 | nan | 1.052066 | 2 | yes | 19 | 0.097 | 0.198 | 0.109 | 0.373 | 1.000000 | nan |
| 6 | 0.345626 | 0.000001 | 0.000005 | 0.000000 | 1.000416 | 2 | yes | 19 | 0.097 | 0.197 | 0.109 | 0.371 | 1.000000 | nan |

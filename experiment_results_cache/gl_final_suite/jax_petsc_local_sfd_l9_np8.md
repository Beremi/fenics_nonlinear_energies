# jax_petsc_local_sfd_l9_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `9` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `1050625` |
| Free DOFs | `1046529` |
| Setup time [s] | `3.882` |
| Total solve time [s] | `7.939` |
| Total Newton iterations | `8` |
| Total linear iterations | `35` |
| Total assembly time [s] | `2.578` |
| Total PC init time [s] | `1.053` |
| Total KSP solve time [s] | `1.551` |
| Total line-search time [s] | `2.543` |
| Final energy | `0.345626` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l9_np8.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l9_np8.log` |

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
| `local_hessian_mode` | `sfd_local` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 7.939 | 8 | 35 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `7.939` |
| Newton iterations | `8` |
| Linear iterations | `35` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.561989 | 0.103311 | 0.000926 | nan | -0.269694 | 9 | yes | 19 | 0.323 | 0.152 | 0.370 | 0.320 | 1.000000 | nan |
| 2 | 0.510037 | 0.051952 | 0.001031 | nan | 0.336704 | 6 | yes | 19 | 0.326 | 0.129 | 0.254 | 0.314 | 1.000000 | nan |
| 3 | 0.447872 | 0.062165 | 0.000564 | nan | -0.240636 | 6 | yes | 19 | 0.326 | 0.129 | 0.254 | 0.317 | 1.000000 | nan |
| 4 | 0.388714 | 0.059158 | 0.000857 | nan | 0.173392 | 5 | yes | 19 | 0.328 | 0.128 | 0.215 | 0.315 | 1.000000 | nan |
| 5 | 0.353640 | 0.035074 | 0.000819 | nan | 0.460064 | 3 | yes | 19 | 0.327 | 0.130 | 0.142 | 0.318 | 1.000000 | nan |
| 6 | 0.345690 | 0.007950 | 0.000415 | nan | 1.038371 | 2 | yes | 19 | 0.317 | 0.127 | 0.106 | 0.321 | 1.000000 | nan |
| 7 | 0.345626 | 0.000063 | 0.000032 | nan | 1.013246 | 2 | yes | 19 | 0.315 | 0.128 | 0.106 | 0.319 | 1.000000 | nan |
| 8 | 0.345626 | 0.000000 | 0.000000 | 0.000000 | 1.000416 | 2 | yes | 19 | 0.317 | 0.130 | 0.106 | 0.319 | 1.000000 | nan |

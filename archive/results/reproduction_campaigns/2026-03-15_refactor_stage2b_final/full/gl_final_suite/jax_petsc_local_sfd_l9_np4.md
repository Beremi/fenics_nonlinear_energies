# jax_petsc_local_sfd_l9_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `9` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `1050625` |
| Free DOFs | `1046529` |
| Setup time [s] | `6.700` |
| Total solve time [s] | `8.359` |
| Total Newton iterations | `7` |
| Total linear iterations | `27` |
| Total assembly time [s] | `2.810` |
| Total PC init time [s] | `1.770` |
| Total KSP solve time [s] | `1.346` |
| Total line-search time [s] | `2.228` |
| Final energy | `0.345626` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l9_np4.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l9_np4.log` |

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
| 1 | 8.359 | 7 | 27 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `8.359` |
| Newton iterations | `7` |
| Linear iterations | `27` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.503061 | 0.162238 | 0.000926 | nan | -0.325608 | 7 | yes | 19 | 0.705 | 0.279 | 0.324 | 0.317 | 1.000000 | nan |
| 2 | 0.502212 | 0.000849 | 0.000929 | nan | -0.243169 | 6 | yes | 19 | 0.357 | 0.249 | 0.279 | 0.317 | 1.000000 | nan |
| 3 | 0.472838 | 0.029375 | 0.001142 | nan | -0.188120 | 5 | yes | 19 | 0.339 | 0.248 | 0.238 | 0.318 | 1.000000 | nan |
| 4 | 0.365951 | 0.106887 | 0.001325 | nan | 0.350667 | 3 | yes | 19 | 0.340 | 0.248 | 0.156 | 0.321 | 1.000000 | nan |
| 5 | 0.345942 | 0.020009 | 0.000756 | nan | 0.994485 | 2 | yes | 19 | 0.357 | 0.248 | 0.116 | 0.322 | 1.000000 | nan |
| 6 | 0.345627 | 0.000315 | 0.000074 | nan | 1.023708 | 2 | yes | 19 | 0.355 | 0.250 | 0.116 | 0.317 | 1.000000 | nan |
| 7 | 0.345626 | 0.000000 | 0.000001 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.357 | 0.248 | 0.116 | 0.317 | 1.000000 | nan |

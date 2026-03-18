# jax_petsc_local_sfd_l5_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `5` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `4225` |
| Free DOFs | `3969` |
| Setup time [s] | `0.839` |
| Total solve time [s] | `0.246` |
| Total Newton iterations | `8` |
| Total linear iterations | `33` |
| Total assembly time [s] | `0.162` |
| Total PC init time [s] | `0.017` |
| Total KSP solve time [s] | `0.011` |
| Total line-search time [s] | `0.052` |
| Final energy | `0.346231` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l5_np1.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l5_np1.log` |

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
| 1 | 0.246 | 8 | 33 | 0.346231 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.246` |
| Newton iterations | `8` |
| Linear iterations | `33` |
| Energy | `0.346231` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.563701 | 0.101769 | 0.014806 | nan | -0.260530 | 9 | yes | 19 | 0.153 | 0.003 | 0.003 | 0.007 | 1.000000 | nan |
| 2 | 0.512226 | 0.051475 | 0.016164 | nan | 0.325274 | 6 | yes | 19 | 0.002 | 0.002 | 0.002 | 0.006 | 1.000000 | nan |
| 3 | 0.446643 | 0.065583 | 0.008885 | nan | -0.311480 | 5 | yes | 19 | 0.001 | 0.002 | 0.002 | 0.006 | 1.000000 | nan |
| 4 | 0.402183 | 0.044460 | 0.013808 | nan | 0.108314 | 4 | yes | 19 | 0.001 | 0.002 | 0.001 | 0.006 | 1.000000 | nan |
| 5 | 0.357061 | 0.045122 | 0.013271 | nan | 0.359996 | 3 | yes | 19 | 0.001 | 0.002 | 0.001 | 0.007 | 1.000000 | nan |
| 6 | 0.346330 | 0.010731 | 0.007555 | nan | 1.012979 | 2 | yes | 19 | 0.001 | 0.002 | 0.001 | 0.006 | 1.000000 | nan |
| 7 | 0.346231 | 0.000099 | 0.000655 | nan | 1.013246 | 2 | yes | 19 | 0.001 | 0.002 | 0.001 | 0.007 | 1.000000 | nan |
| 8 | 0.346231 | 0.000000 | 0.000004 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.001 | 0.002 | 0.001 | 0.006 | 1.000000 | nan |

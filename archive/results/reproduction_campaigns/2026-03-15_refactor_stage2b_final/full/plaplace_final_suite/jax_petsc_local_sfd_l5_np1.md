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
| Total DOFs | `3201` |
| Free DOFs | `2945` |
| Setup time [s] | `0.552` |
| Total solve time [s] | `0.192` |
| Total Newton iterations | `6` |
| Total linear iterations | `6` |
| Total assembly time [s] | `0.140` |
| Total PC init time [s] | `0.023` |
| Total KSP solve time [s] | `0.004` |
| Total line-search time [s] | `0.022` |
| Final energy | `-7.942969` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_local_sfd_l5_np1.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_local_sfd_l5_np1.log` |

## Frozen Settings

| Setting | Value |
|---|---|
| `ksp_type` | `cg` |
| `pc_type` | `hypre` |
| `ksp_rtol` | `0.1` |
| `ksp_max_it` | `30` |
| `use_trust_region` | `False` |
| `trust_subproblem_line_search` | `False` |
| `linesearch_interval` | `[-0.5, 2.0]` |
| `linesearch_tol` | `0.1` |
| `trust_radius_init` | `1.0` |
| `local_hessian_mode` | `sfd_local` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.192 | 6 | 6 | -7.942969 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.192` |
| Newton iterations | `6` |
| Linear iterations | `6` |
| Energy | `-7.942969` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.077151 | 8662.155668 | 1796.149353 | nan | 1.956948 | 1 | yes | 9 | 0.136 | 0.004 | 0.001 | 0.004 | 1.000000 | nan |
| 2 | -6.000210 | 5.923059 | 1.076054 | nan | 0.219327 | 1 | yes | 9 | 0.001 | 0.004 | 0.001 | 0.004 | 1.000000 | nan |
| 3 | -7.748865 | 1.748655 | 4.670113 | nan | 0.584072 | 1 | yes | 9 | 0.001 | 0.004 | 0.001 | 0.004 | 1.000000 | nan |
| 4 | -7.940342 | 0.191478 | 1.971581 | nan | 1.055248 | 1 | yes | 9 | 0.001 | 0.004 | 0.001 | 0.004 | 1.000000 | nan |
| 5 | -7.942960 | 0.002618 | 0.176147 | nan | 1.055248 | 1 | yes | 9 | 0.001 | 0.004 | 0.001 | 0.004 | 1.000000 | nan |
| 6 | -7.942969 | 0.000008 | 0.009402 | 0.000169 | 1.002033 | 1 | yes | 9 | 0.001 | 0.004 | 0.001 | 0.004 | 1.000000 | nan |

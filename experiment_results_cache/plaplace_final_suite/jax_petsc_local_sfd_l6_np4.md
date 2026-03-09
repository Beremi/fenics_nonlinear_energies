# jax_petsc_local_sfd_l6_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `6` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `12545` |
| Free DOFs | `12033` |
| Setup time [s] | `0.705` |
| Total solve time [s] | `0.117` |
| Total Newton iterations | `7` |
| Total linear iterations | `10` |
| Total assembly time [s] | `0.006` |
| Total PC init time [s] | `0.062` |
| Total KSP solve time [s] | `0.014` |
| Total line-search time [s] | `0.031` |
| Final energy | `-7.954564` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/jax_petsc_local_sfd_l6_np4.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/jax_petsc_local_sfd_l6_np4.log` |

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
| 1 | 0.117 | 7 | 10 | -7.954564 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.117` |
| Newton iterations | `7` |
| Linear iterations | `10` |
| Energy | `-7.954564` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.174548 | 65342.831658 | 6978.482874 | nan | 1.956948 | 2 | yes | 9 | 0.001 | 0.010 | 0.002 | 0.005 | 1.000000 | nan |
| 2 | -5.842377 | 6.016925 | 3.268084 | nan | 0.411863 | 1 | yes | 9 | 0.001 | 0.009 | 0.002 | 0.004 | 1.000000 | nan |
| 3 | -7.712817 | 1.870440 | 5.418072 | nan | 0.637288 | 2 | yes | 9 | 0.001 | 0.009 | 0.003 | 0.004 | 1.000000 | nan |
| 4 | -7.941930 | 0.229113 | 2.139070 | nan | 0.948817 | 1 | yes | 9 | 0.001 | 0.009 | 0.002 | 0.004 | 1.000000 | nan |
| 5 | -7.954305 | 0.012375 | 0.463164 | nan | 1.055248 | 2 | yes | 9 | 0.001 | 0.009 | 0.003 | 0.004 | 1.000000 | nan |
| 6 | -7.954562 | 0.000257 | 0.057779 | nan | 1.088137 | 1 | yes | 9 | 0.001 | 0.009 | 0.002 | 0.004 | 1.000000 | nan |
| 7 | -7.954564 | 0.000001 | 0.003064 | 0.000136 | 1.055248 | 1 | yes | 9 | 0.001 | 0.009 | 0.002 | 0.004 | 1.000000 | nan |

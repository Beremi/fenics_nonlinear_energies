# jax_petsc_local_sfd_l6_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `6` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `12545` |
| Free DOFs | `12033` |
| Setup time [s] | `0.725` |
| Total solve time [s] | `0.126` |
| Total Newton iterations | `7` |
| Total linear iterations | `11` |
| Total assembly time [s] | `0.005` |
| Total PC init time [s] | `0.070` |
| Total KSP solve time [s] | `0.017` |
| Total line-search time [s] | `0.030` |
| Final energy | `-7.954564` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/jax_petsc_local_sfd_l6_np8.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/jax_petsc_local_sfd_l6_np8.log` |

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
| 1 | 0.126 | 7 | 11 | -7.954564 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.126` |
| Newton iterations | `7` |
| Linear iterations | `11` |
| Energy | `-7.954564` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.188846 | 65342.817360 | 6978.482874 | nan | 1.956948 | 2 | yes | 9 | 0.001 | 0.011 | 0.003 | 0.005 | 1.000000 | nan |
| 2 | -5.778321 | 5.967167 | 3.268162 | nan | 0.411863 | 1 | yes | 9 | 0.001 | 0.010 | 0.002 | 0.004 | 1.000000 | nan |
| 3 | -7.703459 | 1.925138 | 5.514274 | nan | 0.637288 | 2 | yes | 9 | 0.001 | 0.011 | 0.003 | 0.004 | 1.000000 | nan |
| 4 | -7.941036 | 0.237577 | 2.195275 | nan | 0.948817 | 2 | yes | 9 | 0.001 | 0.010 | 0.003 | 0.004 | 1.000000 | nan |
| 5 | -7.954318 | 0.013282 | 0.491565 | nan | 1.088137 | 2 | yes | 9 | 0.001 | 0.010 | 0.003 | 0.004 | 1.000000 | nan |
| 6 | -7.954562 | 0.000244 | 0.056879 | nan | 1.088137 | 1 | yes | 9 | 0.001 | 0.010 | 0.002 | 0.004 | 1.000000 | nan |
| 7 | -7.954564 | 0.000001 | 0.003111 | 0.000090 | 1.002033 | 1 | yes | 9 | 0.001 | 0.010 | 0.002 | 0.004 | 1.000000 | nan |

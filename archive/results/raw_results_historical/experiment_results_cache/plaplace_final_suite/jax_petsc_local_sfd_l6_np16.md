# jax_petsc_local_sfd_l6_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `6` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `12545` |
| Free DOFs | `12033` |
| Setup time [s] | `0.711` |
| Total solve time [s] | `0.162` |
| Total Newton iterations | `7` |
| Total linear iterations | `11` |
| Total assembly time [s] | `0.004` |
| Total PC init time [s] | `0.096` |
| Total KSP solve time [s] | `0.024` |
| Total line-search time [s] | `0.034` |
| Final energy | `-7.954564` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/jax_petsc_local_sfd_l6_np16.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/jax_petsc_local_sfd_l6_np16.log` |

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
| 1 | 0.162 | 7 | 11 | -7.954564 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.162` |
| Newton iterations | `7` |
| Linear iterations | `11` |
| Energy | `-7.954564` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.213246 | 65342.792960 | 6978.482874 | nan | 1.956948 | 2 | yes | 9 | 0.001 | 0.016 | 0.004 | 0.005 | 1.000000 | nan |
| 2 | -5.793649 | 6.006895 | 3.267612 | nan | 0.411863 | 1 | yes | 9 | 0.001 | 0.014 | 0.003 | 0.005 | 1.000000 | nan |
| 3 | -7.699670 | 1.906022 | 5.482087 | nan | 0.637288 | 2 | yes | 9 | 0.001 | 0.013 | 0.004 | 0.005 | 1.000000 | nan |
| 4 | -7.940681 | 0.241010 | 2.198814 | nan | 0.948817 | 2 | yes | 9 | 0.000 | 0.014 | 0.004 | 0.005 | 1.000000 | nan |
| 5 | -7.954309 | 0.013628 | 0.493413 | nan | 1.088137 | 2 | yes | 9 | 0.000 | 0.013 | 0.004 | 0.005 | 1.000000 | nan |
| 6 | -7.954562 | 0.000254 | 0.057719 | nan | 1.088137 | 1 | yes | 9 | 0.001 | 0.013 | 0.003 | 0.005 | 1.000000 | nan |
| 7 | -7.954564 | 0.000001 | 0.003125 | 0.000093 | 1.002033 | 1 | yes | 9 | 0.000 | 0.013 | 0.003 | 0.005 | 1.000000 | nan |

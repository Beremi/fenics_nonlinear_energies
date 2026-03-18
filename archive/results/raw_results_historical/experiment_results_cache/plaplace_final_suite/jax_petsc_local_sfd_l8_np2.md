# jax_petsc_local_sfd_l8_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `8` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `197633` |
| Free DOFs | `195585` |
| Setup time [s] | `1.847` |
| Total solve time [s] | `1.155` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.125` |
| Total PC init time [s] | `0.588` |
| Total KSP solve time [s] | `0.191` |
| Total line-search time [s] | `0.224` |
| Final energy | `-7.959556` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/jax_petsc_local_sfd_l8_np2.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/jax_petsc_local_sfd_l8_np2.log` |

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
| 1 | 1.155 | 6 | 12 | -7.959556 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `1.155` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.959556` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42.020822 | 4134620.949391 | 111906.518518 | nan | 1.956948 | 2 | yes | 9 | 0.021 | 0.115 | 0.033 | 0.039 | 1.000000 | nan |
| 2 | -5.543509 | 47.564330 | 51.890385 | nan | 1.817627 | 2 | yes | 9 | 0.021 | 0.105 | 0.032 | 0.037 | 1.000000 | nan |
| 3 | -7.584607 | 2.041098 | 7.919839 | nan | 0.690503 | 2 | yes | 9 | 0.021 | 0.108 | 0.034 | 0.037 | 1.000000 | nan |
| 4 | -7.956610 | 0.372003 | 3.067761 | nan | 1.174242 | 2 | yes | 9 | 0.020 | 0.092 | 0.031 | 0.037 | 1.000000 | nan |
| 5 | -7.959530 | 0.002920 | 0.238830 | nan | 1.055248 | 2 | yes | 9 | 0.021 | 0.084 | 0.030 | 0.037 | 1.000000 | nan |
| 6 | -7.959556 | 0.000026 | 0.016576 | 0.001272 | 1.088137 | 2 | yes | 9 | 0.021 | 0.084 | 0.031 | 0.037 | 1.000000 | nan |

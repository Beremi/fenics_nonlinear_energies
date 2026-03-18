# jax_petsc_element_l5_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `5` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `3201` |
| Free DOFs | `2945` |
| Setup time [s] | `0.258` |
| Total solve time [s] | `0.093` |
| Total Newton iterations | `6` |
| Total linear iterations | `6` |
| Total assembly time [s] | `0.002` |
| Total PC init time [s] | `0.051` |
| Total KSP solve time [s] | `0.011` |
| Total line-search time [s] | `0.026` |
| Final energy | `-7.942969` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l5_np8.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l5_np8.log` |

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
| `local_hessian_mode` | `element` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.093 | 6 | 6 | -7.942969 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.093` |
| Newton iterations | `6` |
| Linear iterations | `6` |
| Energy | `-7.942969` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.158554 | 8661.919962 | 1796.149353 | nan | 1.956948 | 1 | yes | 9 | 0.000 | 0.009 | 0.002 | 0.004 | 1.000000 | nan |
| 2 | -5.785533 | 5.944087 | 1.213762 | nan | 0.219327 | 1 | yes | 9 | 0.000 | 0.008 | 0.002 | 0.004 | 1.000000 | nan |
| 3 | -7.694394 | 1.908861 | 4.641741 | nan | 0.551183 | 1 | yes | 9 | 0.000 | 0.008 | 0.002 | 0.004 | 1.000000 | nan |
| 4 | -7.936419 | 0.242025 | 2.129261 | nan | 1.002033 | 1 | yes | 9 | 0.000 | 0.009 | 0.002 | 0.004 | 1.000000 | nan |
| 5 | -7.942901 | 0.006482 | 0.283956 | nan | 1.088137 | 1 | yes | 9 | 0.000 | 0.009 | 0.002 | 0.004 | 1.000000 | nan |
| 6 | -7.942969 | 0.000068 | 0.022729 | 0.001095 | 1.055248 | 1 | yes | 9 | 0.000 | 0.008 | 0.002 | 0.004 | 1.000000 | nan |

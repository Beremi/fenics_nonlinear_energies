# jax_petsc_element_l6_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `6` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `12545` |
| Free DOFs | `12033` |
| Setup time [s] | `0.377` |
| Total solve time [s] | `0.196` |
| Total Newton iterations | `7` |
| Total linear iterations | `10` |
| Total assembly time [s] | `0.017` |
| Total PC init time [s] | `0.095` |
| Total KSP solve time [s] | `0.024` |
| Total line-search time [s] | `0.054` |
| Final energy | `-7.954564` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l6_np1.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l6_np1.log` |

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
| 1 | 0.196 | 7 | 10 | -7.954564 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.196` |
| Newton iterations | `7` |
| Linear iterations | `10` |
| Energy | `-7.954564` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.179866 | 65342.826341 | 6978.482874 | nan | 1.956948 | 2 | yes | 9 | 0.004 | 0.016 | 0.004 | 0.008 | 1.000000 | nan |
| 2 | -5.813350 | 5.993215 | 3.268111 | nan | 0.411863 | 1 | yes | 9 | 0.003 | 0.014 | 0.003 | 0.008 | 1.000000 | nan |
| 3 | -7.706286 | 1.892937 | 5.449764 | nan | 0.637288 | 2 | yes | 9 | 0.002 | 0.015 | 0.004 | 0.008 | 1.000000 | nan |
| 4 | -7.942070 | 0.235784 | 2.186848 | nan | 0.948817 | 1 | yes | 9 | 0.002 | 0.014 | 0.003 | 0.008 | 1.000000 | nan |
| 5 | -7.954328 | 0.012259 | 0.453944 | nan | 1.055248 | 2 | yes | 9 | 0.002 | 0.013 | 0.004 | 0.008 | 1.000000 | nan |
| 6 | -7.954562 | 0.000234 | 0.054070 | nan | 1.088137 | 1 | yes | 9 | 0.002 | 0.013 | 0.003 | 0.008 | 1.000000 | nan |
| 7 | -7.954564 | 0.000001 | 0.002918 | 0.000132 | 1.055248 | 1 | yes | 9 | 0.002 | 0.012 | 0.003 | 0.008 | 1.000000 | nan |

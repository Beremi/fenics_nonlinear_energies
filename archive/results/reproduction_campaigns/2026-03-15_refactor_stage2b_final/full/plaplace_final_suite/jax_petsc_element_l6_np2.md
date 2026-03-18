# jax_petsc_element_l6_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `6` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `12545` |
| Free DOFs | `12033` |
| Setup time [s] | `0.317` |
| Total solve time [s] | `0.152` |
| Total Newton iterations | `7` |
| Total linear iterations | `10` |
| Total assembly time [s] | `0.010` |
| Total PC init time [s] | `0.077` |
| Total KSP solve time [s] | `0.018` |
| Total line-search time [s] | `0.043` |
| Final energy | `-7.954564` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l6_np2.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l6_np2.log` |

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
| 1 | 0.152 | 7 | 10 | -7.954564 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.152` |
| Newton iterations | `7` |
| Linear iterations | `10` |
| Energy | `-7.954564` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.182567 | 65342.823639 | 6978.482874 | nan | 1.956948 | 2 | yes | 9 | 0.002 | 0.012 | 0.003 | 0.006 | 1.000000 | nan |
| 2 | -5.801397 | 5.983964 | 3.267965 | nan | 0.411863 | 1 | yes | 9 | 0.002 | 0.011 | 0.002 | 0.006 | 1.000000 | nan |
| 3 | -7.707674 | 1.906277 | 5.499268 | nan | 0.637288 | 2 | yes | 9 | 0.001 | 0.012 | 0.003 | 0.006 | 1.000000 | nan |
| 4 | -7.942271 | 0.234597 | 2.185379 | nan | 0.948817 | 1 | yes | 9 | 0.001 | 0.011 | 0.002 | 0.006 | 1.000000 | nan |
| 5 | -7.954334 | 0.012062 | 0.448604 | nan | 1.055248 | 2 | yes | 9 | 0.001 | 0.011 | 0.003 | 0.006 | 1.000000 | nan |
| 6 | -7.954562 | 0.000229 | 0.052955 | nan | 1.088137 | 1 | yes | 9 | 0.001 | 0.010 | 0.002 | 0.006 | 1.000000 | nan |
| 7 | -7.954564 | 0.000001 | 0.003121 | 0.000131 | 1.055248 | 1 | yes | 9 | 0.001 | 0.010 | 0.002 | 0.006 | 1.000000 | nan |

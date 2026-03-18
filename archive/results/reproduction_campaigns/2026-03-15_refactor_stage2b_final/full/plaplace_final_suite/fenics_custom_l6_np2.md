# fenics_custom_l6_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `6` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `12545` |
| Free DOFs | `12280` |
| Setup time [s] | `0.054` |
| Total solve time [s] | `0.137` |
| Total Newton iterations | `5` |
| Total linear iterations | `8` |
| Total assembly time [s] | `0.013` |
| Total PC init time [s] | `0.056` |
| Total KSP solve time [s] | `0.016` |
| Total line-search time [s] | `0.044` |
| Final energy | `-7.954564` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l6_np2.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l6_np2.log` |

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
| `local_hessian_mode` | `-` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.137 | 5 | 8 | -7.954564 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.137` |
| Newton iterations | `5` |
| Linear iterations | `8` |
| Energy | `-7.954564` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.851553 | 6.063039 | 0.753185 | nan | 0.186438 | 1 | yes | 9 | 0.003 | 0.012 | 0.002 | 0.009 | 1.000000 | nan |
| 2 | -7.688304 | 1.836751 | 5.705305 | nan | 0.584072 | 2 | yes | 9 | 0.003 | 0.011 | 0.004 | 0.009 | 1.000000 | nan |
| 3 | -7.950594 | 0.262290 | 2.604415 | nan | 1.088137 | 2 | yes | 9 | 0.003 | 0.011 | 0.004 | 0.009 | 1.000000 | nan |
| 4 | -7.954535 | 0.003941 | 0.238825 | nan | 1.088137 | 1 | yes | 9 | 0.003 | 0.011 | 0.003 | 0.009 | 1.000000 | nan |
| 5 | -7.954564 | 0.000028 | 0.018928 | 0.000485 | 1.002033 | 2 | yes | 9 | 0.003 | 0.011 | 0.004 | 0.009 | 1.000000 | nan |

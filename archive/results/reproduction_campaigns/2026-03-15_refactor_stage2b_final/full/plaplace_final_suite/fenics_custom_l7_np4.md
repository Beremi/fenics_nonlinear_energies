# fenics_custom_l7_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `7` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `49665` |
| Free DOFs | `49353` |
| Setup time [s] | `0.170` |
| Total solve time [s] | `0.354` |
| Total Newton iterations | `6` |
| Total linear iterations | `11` |
| Total assembly time [s] | `0.039` |
| Total PC init time [s] | `0.147` |
| Total KSP solve time [s] | `0.047` |
| Total line-search time [s] | `0.104` |
| Final energy | `-7.958292` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l7_np4.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l7_np4.log` |

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
| 1 | 0.354 | 6 | 11 | -7.958292 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.354` |
| Newton iterations | `6` |
| Linear iterations | `11` |
| Energy | `-7.958292` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.363979 | 6.029234 | 2.800481 | nan | 0.358647 | 1 | yes | 9 | 0.007 | 0.028 | 0.006 | 0.018 | 1.000000 | nan |
| 2 | -7.618955 | 2.254976 | 6.209906 | nan | 0.584072 | 2 | yes | 9 | 0.006 | 0.025 | 0.008 | 0.017 | 1.000000 | nan |
| 3 | -7.940304 | 0.321349 | 2.739069 | nan | 0.948817 | 2 | yes | 9 | 0.006 | 0.023 | 0.008 | 0.017 | 1.000000 | nan |
| 4 | -7.958005 | 0.017701 | 0.548632 | nan | 1.055248 | 2 | yes | 9 | 0.006 | 0.024 | 0.008 | 0.017 | 1.000000 | nan |
| 5 | -7.958289 | 0.000283 | 0.056028 | nan | 1.088137 | 2 | yes | 9 | 0.006 | 0.023 | 0.008 | 0.017 | 1.000000 | nan |
| 6 | -7.958292 | 0.000004 | 0.004810 | 0.000341 | 1.088137 | 2 | yes | 9 | 0.006 | 0.023 | 0.008 | 0.017 | 1.000000 | nan |

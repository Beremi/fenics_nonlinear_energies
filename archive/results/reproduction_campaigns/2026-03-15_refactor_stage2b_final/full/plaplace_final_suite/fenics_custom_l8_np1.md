# fenics_custom_l8_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `8` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `197633` |
| Free DOFs | `195585` |
| Setup time [s] | `0.552` |
| Total solve time [s] | `4.031` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.444` |
| Total PC init time [s] | `1.286` |
| Total KSP solve time [s] | `0.493` |
| Total line-search time [s] | `1.572` |
| Final energy | `-7.959556` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l8_np1.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l8_np1.log` |

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
| 1 | 4.031 | 6 | 12 | -7.959556 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `4.031` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.959556` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -3.968347 | 8.272959 | 11.247358 | nan | 0.723392 | 2 | yes | 9 | 0.079 | 0.246 | 0.084 | 0.263 | 1.000000 | nan |
| 2 | -7.323956 | 3.355609 | 8.789222 | nan | 0.776608 | 2 | yes | 9 | 0.073 | 0.244 | 0.088 | 0.262 | 1.000000 | nan |
| 3 | -7.920161 | 0.596205 | 3.296795 | nan | 0.948817 | 2 | yes | 9 | 0.073 | 0.213 | 0.081 | 0.262 | 1.000000 | nan |
| 4 | -7.958393 | 0.038232 | 0.724299 | nan | 1.002033 | 2 | yes | 9 | 0.073 | 0.198 | 0.080 | 0.262 | 1.000000 | nan |
| 5 | -7.959531 | 0.001138 | 0.114542 | nan | 1.055248 | 2 | yes | 9 | 0.074 | 0.193 | 0.080 | 0.262 | 1.000000 | nan |
| 6 | -7.959556 | 0.000025 | 0.014107 | 0.000931 | 1.088137 | 2 | yes | 9 | 0.073 | 0.192 | 0.081 | 0.261 | 1.000000 | nan |

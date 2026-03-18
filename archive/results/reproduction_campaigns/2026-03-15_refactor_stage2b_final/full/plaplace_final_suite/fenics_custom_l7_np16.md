# fenics_custom_l7_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `7` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `49665` |
| Free DOFs | `49530` |
| Setup time [s] | `0.151` |
| Total solve time [s] | `0.219` |
| Total Newton iterations | `6` |
| Total linear iterations | `10` |
| Total assembly time [s] | `0.010` |
| Total PC init time [s] | `0.133` |
| Total KSP solve time [s] | `0.038` |
| Total line-search time [s] | `0.032` |
| Final energy | `-7.958292` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l7_np16.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l7_np16.log` |

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
| 1 | 0.219 | 6 | 10 | -7.958292 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.219` |
| Newton iterations | `6` |
| Linear iterations | `10` |
| Energy | `-7.958292` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.356312 | 6.038952 | 2.859689 | nan | 0.358647 | 1 | yes | 9 | 0.002 | 0.023 | 0.005 | 0.005 | 1.000000 | nan |
| 2 | -7.638572 | 2.282260 | 6.165305 | nan | 0.584072 | 2 | yes | 9 | 0.002 | 0.023 | 0.008 | 0.006 | 1.000000 | nan |
| 3 | -7.941821 | 0.303249 | 2.674194 | nan | 0.948817 | 2 | yes | 9 | 0.002 | 0.022 | 0.007 | 0.005 | 1.000000 | nan |
| 4 | -7.958001 | 0.016180 | 0.524397 | nan | 1.055248 | 2 | yes | 9 | 0.002 | 0.021 | 0.007 | 0.005 | 1.000000 | nan |
| 5 | -7.958290 | 0.000289 | 0.056718 | nan | 1.141353 | 1 | yes | 9 | 0.002 | 0.021 | 0.005 | 0.005 | 1.000000 | nan |
| 6 | -7.958292 | 0.000003 | 0.004925 | 0.000177 | 1.002033 | 2 | yes | 9 | 0.001 | 0.022 | 0.007 | 0.005 | 1.000000 | nan |

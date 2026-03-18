# fenics_custom_l6_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `6` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `12545` |
| Free DOFs | `12545` |
| Setup time [s] | `0.062` |
| Total solve time [s] | `0.184` |
| Total Newton iterations | `5` |
| Total linear iterations | `8` |
| Total assembly time [s] | `0.003` |
| Total PC init time [s] | `0.127` |
| Total KSP solve time [s] | `0.040` |
| Total line-search time [s] | `0.010` |
| Final energy | `-7.954564` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l6_np32.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l6_np32.log` |

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
| 1 | 0.184 | 5 | 8 | -7.954564 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.184` |
| Newton iterations | `5` |
| Linear iterations | `8` |
| Energy | `-7.954564` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.770730 | 5.983674 | 0.772147 | nan | 0.186438 | 1 | yes | 9 | 0.001 | 0.027 | 0.007 | 0.002 | 1.000000 | nan |
| 2 | -7.678258 | 1.907529 | 5.763129 | nan | 0.551183 | 2 | yes | 9 | 0.000 | 0.023 | 0.010 | 0.002 | 1.000000 | nan |
| 3 | -7.949453 | 0.271194 | 2.655249 | nan | 1.088137 | 1 | yes | 9 | 0.000 | 0.025 | 0.008 | 0.002 | 1.000000 | nan |
| 4 | -7.954521 | 0.005068 | 0.266744 | nan | 1.088137 | 2 | yes | 9 | 0.000 | 0.025 | 0.007 | 0.002 | 1.000000 | nan |
| 5 | -7.954564 | 0.000042 | 0.021001 | 0.000764 | 1.055248 | 2 | yes | 9 | 0.001 | 0.027 | 0.007 | 0.002 | 1.000000 | nan |

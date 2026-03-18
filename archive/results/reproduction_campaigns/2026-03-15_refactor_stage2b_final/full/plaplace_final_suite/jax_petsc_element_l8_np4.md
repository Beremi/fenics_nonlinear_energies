# jax_petsc_element_l8_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `8` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `197633` |
| Free DOFs | `195585` |
| Setup time [s] | `1.060` |
| Total solve time [s] | `0.928` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.096` |
| Total PC init time [s] | `0.484` |
| Total KSP solve time [s] | `0.145` |
| Total line-search time [s] | `0.180` |
| Final energy | `-7.959556` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l8_np4.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l8_np4.log` |

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
| 1 | 0.928 | 6 | 12 | -7.959556 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.928` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.959556` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41.941714 | 4134621.028498 | 111906.518518 | nan | 1.956948 | 2 | yes | 9 | 0.017 | 0.092 | 0.024 | 0.032 | 1.000000 | nan |
| 2 | -5.541372 | 47.483086 | 51.889760 | nan | 1.817627 | 2 | yes | 9 | 0.018 | 0.083 | 0.024 | 0.029 | 1.000000 | nan |
| 3 | -7.586335 | 2.044964 | 7.962674 | nan | 0.723392 | 2 | yes | 9 | 0.016 | 0.087 | 0.026 | 0.030 | 1.000000 | nan |
| 4 | -7.956896 | 0.370561 | 2.980958 | nan | 1.174242 | 2 | yes | 9 | 0.015 | 0.077 | 0.024 | 0.030 | 1.000000 | nan |
| 5 | -7.959535 | 0.002639 | 0.226301 | nan | 1.055248 | 2 | yes | 9 | 0.016 | 0.073 | 0.023 | 0.029 | 1.000000 | nan |
| 6 | -7.959556 | 0.000021 | 0.015032 | 0.001061 | 1.088137 | 2 | yes | 9 | 0.015 | 0.072 | 0.023 | 0.029 | 1.000000 | nan |

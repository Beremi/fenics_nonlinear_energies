# fenics_custom_l5_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `5` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `3201` |
| Free DOFs | `3172` |
| Setup time [s] | `0.026` |
| Total solve time [s] | `0.067` |
| Total Newton iterations | `5` |
| Total linear iterations | `5` |
| Total assembly time [s] | `0.002` |
| Total PC init time [s] | `0.045` |
| Total KSP solve time [s] | `0.011` |
| Total line-search time [s] | `0.007` |
| Final energy | `-7.942969` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l5_np8.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l5_np8.log` |

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
| 1 | 0.067 | 5 | 5 | -7.942969 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.067` |
| Newton iterations | `5` |
| Linear iterations | `5` |
| Energy | `-7.942969` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.961653 | 6.111858 | 0.564733 | nan | 0.100333 | 1 | yes | 9 | 0.001 | 0.010 | 0.002 | 0.001 | 1.000000 | nan |
| 2 | -7.714151 | 1.752498 | 6.146158 | nan | 0.637288 | 1 | yes | 9 | 0.000 | 0.009 | 0.002 | 0.001 | 1.000000 | nan |
| 3 | -7.940640 | 0.226489 | 2.357308 | nan | 1.141353 | 1 | yes | 9 | 0.000 | 0.009 | 0.002 | 0.001 | 1.000000 | nan |
| 4 | -7.942964 | 0.002324 | 0.186315 | nan | 1.002033 | 1 | yes | 9 | 0.000 | 0.008 | 0.002 | 0.001 | 1.000000 | nan |
| 5 | -7.942969 | 0.000005 | 0.004134 | 0.000264 | 1.002033 | 1 | yes | 9 | 0.000 | 0.009 | 0.002 | 0.001 | 1.000000 | nan |

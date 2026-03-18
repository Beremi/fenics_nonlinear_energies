# fenics_custom_l5_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `5` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `3201` |
| Free DOFs | `2945` |
| Setup time [s] | `0.014` |
| Total solve time [s] | `0.056` |
| Total Newton iterations | `5` |
| Total linear iterations | `5` |
| Total assembly time [s] | `0.006` |
| Total PC init time [s] | `0.020` |
| Total KSP solve time [s] | `0.004` |
| Total line-search time [s] | `0.021` |
| Final energy | `-7.942969` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l5_np1.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l5_np1.log` |

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
| 1 | 0.056 | 5 | 5 | -7.942969 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.056` |
| Newton iterations | `5` |
| Linear iterations | `5` |
| Energy | `-7.942969` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.932077 | 6.083632 | 0.563493 | nan | 0.100333 | 1 | yes | 9 | 0.001 | 0.004 | 0.001 | 0.004 | 1.000000 | nan |
| 2 | -7.705583 | 1.773506 | 5.969522 | nan | 0.637288 | 1 | yes | 9 | 0.001 | 0.004 | 0.001 | 0.004 | 1.000000 | nan |
| 3 | -7.940525 | 0.234941 | 2.304564 | nan | 1.141353 | 1 | yes | 9 | 0.001 | 0.004 | 0.001 | 0.004 | 1.000000 | nan |
| 4 | -7.942960 | 0.002435 | 0.177592 | nan | 1.002033 | 1 | yes | 9 | 0.001 | 0.004 | 0.001 | 0.004 | 1.000000 | nan |
| 5 | -7.942969 | 0.000009 | 0.005880 | 0.000166 | 1.002033 | 1 | yes | 9 | 0.001 | 0.004 | 0.001 | 0.004 | 1.000000 | nan |

# fenics_custom_l9_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `9` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `788481` |
| Free DOFs | `788421` |
| Setup time [s] | `2.471` |
| Total solve time [s] | `1.128` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.095` |
| Total PC init time [s] | `0.486` |
| Total KSP solve time [s] | `0.239` |
| Total line-search time [s] | `0.240` |
| Final energy | `-7.960006` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l9_np32.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l9_np32.log` |

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
| 1 | 1.128 | 6 | 12 | -7.960006 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `1.128` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.960006` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.141061 | 38.288367 | 44.828231 | nan | 1.678307 | 2 | yes | 9 | 0.016 | 0.087 | 0.032 | 0.039 | 1.000000 | nan |
| 2 | -7.543733 | 2.402672 | 8.635813 | nan | 0.723392 | 2 | yes | 9 | 0.016 | 0.086 | 0.036 | 0.039 | 1.000000 | nan |
| 3 | -7.952884 | 0.409151 | 3.156605 | nan | 1.141353 | 2 | yes | 9 | 0.013 | 0.075 | 0.045 | 0.039 | 1.000000 | nan |
| 4 | -7.959902 | 0.007017 | 0.358637 | nan | 1.141353 | 2 | yes | 9 | 0.016 | 0.083 | 0.062 | 0.042 | 1.000000 | nan |
| 5 | -7.960005 | 0.000103 | 0.036861 | nan | 1.088137 | 2 | yes | 9 | 0.016 | 0.077 | 0.032 | 0.041 | 1.000000 | nan |
| 6 | -7.960006 | 0.000001 | 0.003119 | 0.000162 | 1.055248 | 2 | yes | 9 | 0.017 | 0.078 | 0.032 | 0.041 | 1.000000 | nan |

# jax_petsc_element_l5_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `5` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `3201` |
| Free DOFs | `2945` |
| Setup time [s] | `0.251` |
| Total solve time [s] | `0.154` |
| Total Newton iterations | `7` |
| Total linear iterations | `9` |
| Total assembly time [s] | `0.003` |
| Total PC init time [s] | `0.091` |
| Total KSP solve time [s] | `0.022` |
| Total line-search time [s] | `0.034` |
| Final energy | `-7.942969` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l5_np16.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l5_np16.log` |

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
| 1 | 0.154 | 7 | 9 | -7.942969 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.154` |
| Newton iterations | `7` |
| Linear iterations | `9` |
| Energy | `-7.942969` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.048141 | 8662.126658 | 1796.149353 | nan | 1.956948 | 1 | yes | 9 | 0.000 | 0.014 | 0.003 | 0.005 | 1.000000 | nan |
| 2 | -5.856179 | 5.808038 | 1.322786 | nan | 0.219327 | 1 | yes | 9 | 0.000 | 0.012 | 0.003 | 0.005 | 1.000000 | nan |
| 3 | -7.685835 | 1.829655 | 4.507877 | nan | 0.551183 | 1 | yes | 9 | 0.000 | 0.014 | 0.003 | 0.005 | 1.000000 | nan |
| 4 | -7.934888 | 0.249053 | 2.234790 | nan | 1.002033 | 2 | yes | 9 | 0.000 | 0.012 | 0.004 | 0.005 | 1.000000 | nan |
| 5 | -7.942698 | 0.007811 | 0.305904 | nan | 1.002033 | 1 | yes | 9 | 0.000 | 0.013 | 0.003 | 0.005 | 1.000000 | nan |
| 6 | -7.942967 | 0.000269 | 0.048644 | nan | 1.088137 | 2 | yes | 9 | 0.000 | 0.013 | 0.004 | 0.005 | 1.000000 | nan |
| 7 | -7.942969 | 0.000002 | 0.003594 | 0.000074 | 1.002033 | 1 | yes | 9 | 0.000 | 0.013 | 0.003 | 0.005 | 1.000000 | nan |

# fenics_custom_l9_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `9` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `788481` |
| Free DOFs | `784385` |
| Setup time [s] | `2.443` |
| Total solve time [s] | `13.577` |
| Total Newton iterations | `5` |
| Total linear iterations | `10` |
| Total assembly time [s] | `1.463` |
| Total PC init time [s] | `4.273` |
| Total KSP solve time [s] | `1.681` |
| Total line-search time [s] | `5.339` |
| Final energy | `-7.960005` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l9_np1.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l9_np1.log` |

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
| 1 | 13.577 | 5 | 10 | -7.960005 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `13.577` |
| Newton iterations | `5` |
| Linear iterations | `10` |
| Energy | `-7.960005` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.132170 | 38.283916 | 44.793319 | nan | 1.678307 | 2 | yes | 9 | 0.311 | 1.002 | 0.357 | 1.068 | 1.000000 | nan |
| 2 | -7.544088 | 2.411918 | 8.641301 | nan | 0.723392 | 2 | yes | 9 | 0.290 | 0.991 | 0.359 | 1.067 | 1.000000 | nan |
| 3 | -7.952877 | 0.408789 | 3.153469 | nan | 1.141353 | 2 | yes | 9 | 0.286 | 0.762 | 0.313 | 1.065 | 1.000000 | nan |
| 4 | -7.959905 | 0.007028 | 0.360585 | nan | 1.141353 | 2 | yes | 9 | 0.286 | 0.748 | 0.320 | 1.066 | 1.000000 | nan |
| 5 | -7.960005 | 0.000099 | 0.036497 | 0.002931 | 1.088137 | 2 | yes | 9 | 0.290 | 0.771 | 0.331 | 1.072 | 1.000000 | nan |

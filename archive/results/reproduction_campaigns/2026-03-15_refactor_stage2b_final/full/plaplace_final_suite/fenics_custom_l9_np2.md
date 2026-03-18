# fenics_custom_l9_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `9` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `788481` |
| Free DOFs | `786335` |
| Setup time [s] | `3.314` |
| Total solve time [s] | `7.534` |
| Total Newton iterations | `5` |
| Total linear iterations | `10` |
| Total assembly time [s] | `0.828` |
| Total PC init time [s] | `2.491` |
| Total KSP solve time [s] | `0.903` |
| Total line-search time [s] | `2.887` |
| Final energy | `-7.960005` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l9_np2.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l9_np2.log` |

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
| 1 | 7.534 | 5 | 10 | -7.960005 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `7.534` |
| Newton iterations | `5` |
| Linear iterations | `10` |
| Energy | `-7.960005` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.149756 | 38.346245 | 44.838861 | nan | 1.678307 | 2 | yes | 9 | 0.190 | 0.581 | 0.188 | 0.608 | 1.000000 | nan |
| 2 | -7.549405 | 2.399649 | 8.548720 | nan | 0.723392 | 2 | yes | 9 | 0.161 | 0.572 | 0.194 | 0.608 | 1.000000 | nan |
| 3 | -7.953125 | 0.403720 | 3.109526 | nan | 1.141353 | 2 | yes | 9 | 0.159 | 0.448 | 0.171 | 0.609 | 1.000000 | nan |
| 4 | -7.959908 | 0.006783 | 0.348427 | nan | 1.141353 | 2 | yes | 9 | 0.159 | 0.444 | 0.173 | 0.530 | 1.000000 | nan |
| 5 | -7.960005 | 0.000097 | 0.035411 | 0.002794 | 1.088137 | 2 | yes | 9 | 0.159 | 0.446 | 0.177 | 0.532 | 1.000000 | nan |

# fenics_custom_l6_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `6` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `12545` |
| Free DOFs | `12033` |
| Setup time [s] | `0.048` |
| Total solve time [s] | `0.177` |
| Total Newton iterations | `5` |
| Total linear iterations | `9` |
| Total assembly time [s] | `0.023` |
| Total PC init time [s] | `0.058` |
| Total KSP solve time [s] | `0.020` |
| Total line-search time [s] | `0.066` |
| Final energy | `-7.954564` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/fenics_custom_l6_np1.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/fenics_custom_l6_np1.log` |

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
| 1 | 0.177 | 5 | 9 | -7.954564 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.177` |
| Newton iterations | `5` |
| Linear iterations | `9` |
| Energy | `-7.954564` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.844615 | 6.055609 | 0.751353 | nan | 0.186438 | 1 | yes | 9 | 0.005 | 0.013 | 0.003 | 0.014 | 1.000000 | nan |
| 2 | -7.671548 | 1.826933 | 5.957966 | nan | 0.584072 | 2 | yes | 9 | 0.004 | 0.012 | 0.004 | 0.013 | 1.000000 | nan |
| 3 | -7.950015 | 0.278468 | 2.712124 | nan | 1.088137 | 2 | yes | 9 | 0.004 | 0.011 | 0.004 | 0.013 | 1.000000 | nan |
| 4 | -7.954536 | 0.004521 | 0.262336 | nan | 1.088137 | 2 | yes | 9 | 0.004 | 0.011 | 0.004 | 0.013 | 1.000000 | nan |
| 5 | -7.954564 | 0.000027 | 0.016929 | 0.000471 | 1.002033 | 2 | yes | 9 | 0.004 | 0.011 | 0.004 | 0.013 | 1.000000 | nan |

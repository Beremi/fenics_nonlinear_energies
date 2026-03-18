# fenics_custom_l6_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `6` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `16641` |
| Free DOFs | `16563` |
| Setup time [s] | `0.054` |
| Total solve time [s] | `0.246` |
| Total Newton iterations | `8` |
| Total linear iterations | `31` |
| Total assembly time [s] | `0.006` |
| Total PC init time [s] | `0.121` |
| Total KSP solve time [s] | `0.074` |
| Total line-search time [s] | `0.026` |
| Final energy | `0.345777` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l6_np16.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l6_np16.log` |

## Frozen Settings

| Setting | Value |
|---|---|
| `ksp_type` | `gmres` |
| `pc_type` | `hypre` |
| `ksp_rtol` | `0.001` |
| `ksp_max_it` | `200` |
| `use_trust_region` | `False` |
| `trust_subproblem_line_search` | `False` |
| `linesearch_interval` | `[-0.5, 2.0]` |
| `linesearch_tol` | `0.001` |
| `trust_radius_init` | `1.0` |
| `local_hessian_mode` | `-` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.246 | 8 | 31 | 0.345777 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.246` |
| Newton iterations | `8` |
| Linear iterations | `31` |
| Energy | `0.345777` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.504296 | 0.161046 | 0.007408 | nan | -0.325875 | 7 | yes | 19 | 0.001 | 0.025 | 0.024 | 0.005 | 1.000000 | nan |
| 2 | 0.504222 | 0.000074 | 0.007500 | nan | -0.133772 | 6 | yes | 19 | 0.001 | 0.014 | 0.011 | 0.003 | 1.000000 | nan |
| 3 | 0.499023 | 0.005199 | 0.008476 | nan | -0.210713 | 6 | yes | 19 | 0.001 | 0.014 | 0.012 | 0.003 | 1.000000 | nan |
| 4 | 0.419011 | 0.080012 | 0.010072 | nan | 0.056498 | 4 | yes | 19 | 0.001 | 0.014 | 0.008 | 0.003 | 1.000000 | nan |
| 5 | 0.351309 | 0.067702 | 0.009162 | nan | 0.643703 | 2 | yes | 19 | 0.001 | 0.014 | 0.005 | 0.003 | 1.000000 | nan |
| 6 | 0.345793 | 0.005516 | 0.003047 | nan | 1.076924 | 2 | yes | 19 | 0.001 | 0.014 | 0.005 | 0.003 | 1.000000 | nan |
| 7 | 0.345777 | 0.000016 | 0.000172 | nan | 1.005482 | 2 | yes | 19 | 0.001 | 0.013 | 0.004 | 0.003 | 1.000000 | nan |
| 8 | 0.345777 | 0.000000 | 0.000001 | 0.000000 | 1.000416 | 2 | yes | 19 | 0.001 | 0.013 | 0.004 | 0.003 | 1.000000 | nan |

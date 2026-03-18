# fenics_custom_l5_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `5` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `3201` |
| Free DOFs | `3066` |
| Setup time [s] | `0.028` |
| Total solve time [s] | `0.037` |
| Total Newton iterations | `5` |
| Total linear iterations | `5` |
| Total assembly time [s] | `0.003` |
| Total PC init time [s] | `0.018` |
| Total KSP solve time [s] | `0.004` |
| Total line-search time [s] | `0.010` |
| Final energy | `-7.942969` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/fenics_custom_l5_np2.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/fenics_custom_l5_np2.log` |

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
| 1 | 0.037 | 5 | 5 | -7.942969 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.037` |
| Newton iterations | `5` |
| Linear iterations | `5` |
| Energy | `-7.942969` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.947235 | 6.099057 | 0.563154 | nan | 0.100333 | 1 | yes | 9 | 0.001 | 0.004 | 0.001 | 0.002 | 1.000000 | nan |
| 2 | -7.701531 | 1.754296 | 6.060160 | nan | 0.637288 | 1 | yes | 9 | 0.001 | 0.004 | 0.001 | 0.002 | 1.000000 | nan |
| 3 | -7.940332 | 0.238801 | 2.395063 | nan | 1.141353 | 1 | yes | 9 | 0.001 | 0.003 | 0.001 | 0.002 | 1.000000 | nan |
| 4 | -7.942959 | 0.002627 | 0.188414 | nan | 1.055248 | 1 | yes | 9 | 0.001 | 0.003 | 0.001 | 0.002 | 1.000000 | nan |
| 5 | -7.942969 | 0.000010 | 0.012442 | 0.000186 | 1.002033 | 1 | yes | 9 | 0.001 | 0.003 | 0.001 | 0.002 | 1.000000 | nan |

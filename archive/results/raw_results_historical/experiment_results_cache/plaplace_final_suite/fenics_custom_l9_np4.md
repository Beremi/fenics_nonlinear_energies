# fenics_custom_l9_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `9` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `788481` |
| Free DOFs | `787184` |
| Setup time [s] | `2.535` |
| Total solve time [s] | `4.077` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.400` |
| Total PC init time [s] | `1.421` |
| Total KSP solve time [s] | `0.668` |
| Total line-search time [s] | `1.389` |
| Final energy | `-7.960006` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/fenics_custom_l9_np4.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/fenics_custom_l9_np4.log` |

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
| 1 | 4.077 | 6 | 12 | -7.960006 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `4.077` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.960006` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.151387 | 38.318922 | 44.826888 | nan | 1.678307 | 2 | yes | 9 | 0.071 | 0.282 | 0.112 | 0.223 | 1.000000 | nan |
| 2 | -7.549155 | 2.397768 | 8.595540 | nan | 0.723392 | 2 | yes | 9 | 0.066 | 0.263 | 0.112 | 0.223 | 1.000000 | nan |
| 3 | -7.953221 | 0.404066 | 3.131350 | nan | 1.141353 | 2 | yes | 9 | 0.066 | 0.217 | 0.105 | 0.250 | 1.000000 | nan |
| 4 | -7.959901 | 0.006681 | 0.349018 | nan | 1.088137 | 2 | yes | 9 | 0.066 | 0.210 | 0.108 | 0.223 | 1.000000 | nan |
| 5 | -7.960005 | 0.000103 | 0.034381 | nan | 1.141353 | 2 | yes | 9 | 0.065 | 0.224 | 0.114 | 0.249 | 1.000000 | nan |
| 6 | -7.960006 | 0.000001 | 0.003211 | 0.000191 | 1.055248 | 2 | yes | 9 | 0.066 | 0.225 | 0.116 | 0.221 | 1.000000 | nan |

# fenics_custom_l9_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `9` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `788481` |
| Free DOFs | `787928` |
| Setup time [s] | `2.220` |
| Total solve time [s] | `2.600` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.230` |
| Total PC init time [s] | `0.942` |
| Total KSP solve time [s] | `0.586` |
| Total line-search time [s] | `0.728` |
| Final energy | `-7.960006` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/fenics_custom_l9_np8.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/fenics_custom_l9_np8.log` |

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
| 1 | 2.600 | 6 | 12 | -7.960006 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `2.600` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.960006` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.103275 | 38.327102 | 44.892839 | nan | 1.678307 | 2 | yes | 9 | 0.046 | 0.172 | 0.095 | 0.122 | 1.000000 | nan |
| 2 | -7.537074 | 2.433799 | 8.687503 | nan | 0.723392 | 2 | yes | 9 | 0.036 | 0.158 | 0.096 | 0.121 | 1.000000 | nan |
| 3 | -7.952727 | 0.415653 | 3.179134 | nan | 1.141353 | 2 | yes | 9 | 0.036 | 0.133 | 0.089 | 0.122 | 1.000000 | nan |
| 4 | -7.959893 | 0.007166 | 0.360692 | nan | 1.141353 | 2 | yes | 9 | 0.039 | 0.159 | 0.100 | 0.121 | 1.000000 | nan |
| 5 | -7.960004 | 0.000112 | 0.037979 | nan | 1.088137 | 2 | yes | 9 | 0.036 | 0.158 | 0.101 | 0.121 | 1.000000 | nan |
| 6 | -7.960006 | 0.000001 | 0.003333 | 0.000174 | 1.055248 | 2 | yes | 9 | 0.036 | 0.162 | 0.103 | 0.121 | 1.000000 | nan |

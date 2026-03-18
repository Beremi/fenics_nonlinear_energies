# jax_petsc_element_l8_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `8` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `197633` |
| Free DOFs | `195585` |
| Setup time [s] | `0.529` |
| Total solve time [s] | `0.462` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.040` |
| Total PC init time [s] | `0.212` |
| Total KSP solve time [s] | `0.057` |
| Total line-search time [s] | `0.135` |
| Final energy | `-7.959556` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/jax_petsc_element_l8_np16.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/jax_petsc_element_l8_np16.log` |

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
| 1 | 0.462 | 6 | 12 | -7.959556 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.462` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.959556` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41.945895 | 4134621.024317 | 111906.518518 | nan | 1.956948 | 2 | yes | 9 | 0.006 | 0.038 | 0.010 | 0.023 | 1.000000 | nan |
| 2 | -5.550586 | 47.496481 | 51.892605 | nan | 1.817627 | 2 | yes | 9 | 0.007 | 0.036 | 0.010 | 0.023 | 1.000000 | nan |
| 3 | -7.587132 | 2.036546 | 7.883818 | nan | 0.723392 | 2 | yes | 9 | 0.007 | 0.037 | 0.010 | 0.023 | 1.000000 | nan |
| 4 | -7.956978 | 0.369846 | 2.951112 | nan | 1.174242 | 2 | yes | 9 | 0.007 | 0.035 | 0.010 | 0.023 | 1.000000 | nan |
| 5 | -7.959535 | 0.002557 | 0.217816 | nan | 1.055248 | 2 | yes | 9 | 0.007 | 0.034 | 0.009 | 0.022 | 1.000000 | nan |
| 6 | -7.959556 | 0.000021 | 0.014669 | 0.001082 | 1.088137 | 2 | yes | 9 | 0.007 | 0.032 | 0.009 | 0.022 | 1.000000 | nan |

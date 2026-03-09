# jax_petsc_element_l9_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `9` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `788481` |
| Free DOFs | `784385` |
| Setup time [s] | `1.840` |
| Total solve time [s] | `2.557` |
| Total Newton iterations | `6` |
| Total linear iterations | `11` |
| Total assembly time [s] | `0.376` |
| Total PC init time [s] | `0.967` |
| Total KSP solve time [s] | `0.522` |
| Total line-search time [s] | `0.588` |
| Final energy | `-7.960005` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/jax_petsc_element_l9_np8.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/jax_petsc_element_l9_np8.log` |

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
| 1 | 2.557 | 6 | 11 | -7.960005 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `2.557` |
| Newton iterations | `6` |
| Linear iterations | `11` |
| Energy | `-7.960005` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 331.447128 | 33077101.435235 | 448769.786628 | nan | 1.956948 | 2 | yes | 9 | 0.064 | 0.185 | 0.093 | 0.100 | 1.000000 | nan |
| 2 | -4.960074 | 336.407203 | 208.098040 | nan | 1.956948 | 2 | yes | 9 | 0.064 | 0.164 | 0.094 | 0.097 | 1.000000 | nan |
| 3 | -7.532000 | 2.571926 | 2.319552 | nan | 0.411863 | 1 | yes | 9 | 0.062 | 0.173 | 0.062 | 0.098 | 1.000000 | nan |
| 4 | -7.951637 | 0.419637 | 2.329233 | nan | 0.948817 | 2 | yes | 9 | 0.062 | 0.153 | 0.091 | 0.097 | 1.000000 | nan |
| 5 | -7.959971 | 0.008334 | 0.314764 | nan | 1.088137 | 2 | yes | 9 | 0.062 | 0.147 | 0.091 | 0.097 | 1.000000 | nan |
| 6 | -7.960005 | 0.000034 | 0.017126 | 0.001144 | 1.055248 | 2 | yes | 9 | 0.062 | 0.144 | 0.091 | 0.099 | 1.000000 | nan |

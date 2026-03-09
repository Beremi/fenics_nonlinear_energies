# jax_petsc_element_l9_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `9` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `788481` |
| Free DOFs | `784385` |
| Setup time [s] | `1.251` |
| Total solve time [s] | `1.388` |
| Total Newton iterations | `6` |
| Total linear iterations | `11` |
| Total assembly time [s] | `0.189` |
| Total PC init time [s] | `0.581` |
| Total KSP solve time [s] | `0.198` |
| Total line-search time [s] | `0.364` |
| Final energy | `-7.960005` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/jax_petsc_element_l9_np16.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/jax_petsc_element_l9_np16.log` |

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
| 1 | 1.388 | 6 | 11 | -7.960005 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `1.388` |
| Newton iterations | `6` |
| Linear iterations | `11` |
| Energy | `-7.960005` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 331.382362 | 33077101.500001 | 448769.786628 | nan | 1.956948 | 2 | yes | 9 | 0.032 | 0.109 | 0.036 | 0.062 | 1.000000 | nan |
| 2 | -4.965502 | 336.347864 | 208.099408 | nan | 1.956948 | 2 | yes | 9 | 0.032 | 0.099 | 0.036 | 0.060 | 1.000000 | nan |
| 3 | -7.538722 | 2.573220 | 2.309048 | nan | 0.411863 | 1 | yes | 9 | 0.031 | 0.104 | 0.024 | 0.061 | 1.000000 | nan |
| 4 | -7.952045 | 0.413322 | 2.290385 | nan | 0.948817 | 2 | yes | 9 | 0.031 | 0.092 | 0.035 | 0.061 | 1.000000 | nan |
| 5 | -7.959973 | 0.007929 | 0.302382 | nan | 1.055248 | 2 | yes | 9 | 0.031 | 0.089 | 0.034 | 0.060 | 1.000000 | nan |
| 6 | -7.960005 | 0.000032 | 0.017124 | 0.001053 | 1.055248 | 2 | yes | 9 | 0.031 | 0.087 | 0.034 | 0.061 | 1.000000 | nan |

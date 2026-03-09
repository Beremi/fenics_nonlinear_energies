# jax_petsc_element_l9_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `9` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `788481` |
| Free DOFs | `784385` |
| Setup time [s] | `4.230` |
| Total solve time [s] | `4.933` |
| Total Newton iterations | `6` |
| Total linear iterations | `11` |
| Total assembly time [s] | `0.595` |
| Total PC init time [s] | `2.528` |
| Total KSP solve time [s] | `0.799` |
| Total line-search time [s] | `0.881` |
| Final energy | `-7.960005` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/jax_petsc_element_l9_np2.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/jax_petsc_element_l9_np2.log` |

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
| 1 | 4.933 | 6 | 11 | -7.960005 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `4.933` |
| Newton iterations | `6` |
| Linear iterations | `11` |
| Energy | `-7.960005` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 331.391595 | 33077101.490768 | 448769.786628 | nan | 1.956948 | 2 | yes | 9 | 0.097 | 0.491 | 0.145 | 0.150 | 1.000000 | nan |
| 2 | -4.961125 | 336.352721 | 208.093747 | nan | 1.956948 | 2 | yes | 9 | 0.100 | 0.460 | 0.145 | 0.147 | 1.000000 | nan |
| 3 | -7.524305 | 2.563180 | 2.298523 | nan | 0.411863 | 1 | yes | 9 | 0.100 | 0.465 | 0.099 | 0.146 | 1.000000 | nan |
| 4 | -7.951217 | 0.426912 | 2.326601 | nan | 0.948817 | 2 | yes | 9 | 0.101 | 0.398 | 0.138 | 0.145 | 1.000000 | nan |
| 5 | -7.959962 | 0.008745 | 0.317225 | nan | 1.088137 | 2 | yes | 9 | 0.100 | 0.356 | 0.133 | 0.147 | 1.000000 | nan |
| 6 | -7.960005 | 0.000042 | 0.018726 | 0.001933 | 1.055248 | 2 | yes | 9 | 0.098 | 0.358 | 0.139 | 0.146 | 1.000000 | nan |

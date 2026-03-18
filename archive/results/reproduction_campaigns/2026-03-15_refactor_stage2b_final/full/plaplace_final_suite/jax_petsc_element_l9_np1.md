# jax_petsc_element_l9_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `9` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `788481` |
| Free DOFs | `784385` |
| Setup time [s] | `10.062` |
| Total solve time [s] | `10.102` |
| Total Newton iterations | `6` |
| Total linear iterations | `11` |
| Total assembly time [s] | `1.120` |
| Total PC init time [s] | `5.094` |
| Total KSP solve time [s] | `1.664` |
| Total line-search time [s] | `2.002` |
| Final energy | `-7.960004` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l9_np1.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l9_np1.log` |

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
| 1 | 10.102 | 6 | 11 | -7.960004 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `10.102` |
| Newton iterations | `6` |
| Linear iterations | `11` |
| Energy | `-7.960004` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 331.418508 | 33077101.463856 | 448769.786628 | nan | 1.956948 | 2 | yes | 9 | 0.187 | 1.006 | 0.300 | 0.339 | 1.000000 | nan |
| 2 | -4.961755 | 336.380263 | 208.093625 | nan | 1.956948 | 2 | yes | 9 | 0.188 | 0.959 | 0.306 | 0.336 | 1.000000 | nan |
| 3 | -7.534270 | 2.572515 | 2.317338 | nan | 0.411863 | 1 | yes | 9 | 0.186 | 0.964 | 0.209 | 0.335 | 1.000000 | nan |
| 4 | -7.951696 | 0.417426 | 2.314233 | nan | 0.948817 | 2 | yes | 9 | 0.187 | 0.806 | 0.291 | 0.330 | 1.000000 | nan |
| 5 | -7.959957 | 0.008261 | 0.312475 | nan | 1.055248 | 2 | yes | 9 | 0.186 | 0.704 | 0.277 | 0.330 | 1.000000 | nan |
| 6 | -7.960004 | 0.000047 | 0.019654 | 0.002159 | 1.088137 | 2 | yes | 9 | 0.186 | 0.656 | 0.281 | 0.332 | 1.000000 | nan |

# jax_petsc_local_sfd_l9_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `9` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `788481` |
| Free DOFs | `784385` |
| Setup time [s] | `1.611` |
| Total solve time [s] | `1.048` |
| Total Newton iterations | `6` |
| Total linear iterations | `11` |
| Total assembly time [s] | `0.109` |
| Total PC init time [s] | `0.459` |
| Total KSP solve time [s] | `0.123` |
| Total line-search time [s] | `0.318` |
| Final energy | `-7.960003` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/jax_petsc_local_sfd_l9_np32.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/jax_petsc_local_sfd_l9_np32.log` |

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
| `local_hessian_mode` | `sfd_local` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 1.048 | 6 | 11 | -7.960003 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `1.048` |
| Newton iterations | `6` |
| Linear iterations | `11` |
| Energy | `-7.960003` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 331.496885 | 33077101.385478 | 448769.786628 | nan | 1.956948 | 2 | yes | 9 | 0.017 | 0.080 | 0.021 | 0.054 | 1.000000 | nan |
| 2 | -4.948124 | 336.445010 | 208.109121 | nan | 1.956948 | 2 | yes | 9 | 0.018 | 0.076 | 0.022 | 0.053 | 1.000000 | nan |
| 3 | -7.527004 | 2.578880 | 2.312219 | nan | 0.411863 | 1 | yes | 9 | 0.017 | 0.086 | 0.017 | 0.054 | 1.000000 | nan |
| 4 | -7.951224 | 0.424220 | 2.345686 | nan | 0.948817 | 2 | yes | 9 | 0.018 | 0.075 | 0.021 | 0.053 | 1.000000 | nan |
| 5 | -7.959941 | 0.008717 | 0.317703 | nan | 1.088137 | 2 | yes | 9 | 0.020 | 0.071 | 0.020 | 0.053 | 1.000000 | nan |
| 6 | -7.960003 | 0.000062 | 0.019544 | 0.002606 | 1.088137 | 2 | yes | 9 | 0.018 | 0.071 | 0.020 | 0.053 | 1.000000 | nan |

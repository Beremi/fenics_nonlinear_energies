# jax_petsc_local_sfd_l9_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `9` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `788481` |
| Free DOFs | `784385` |
| Setup time [s] | `3.671` |
| Total solve time [s] | `3.410` |
| Total Newton iterations | `6` |
| Total linear iterations | `11` |
| Total assembly time [s] | `0.505` |
| Total PC init time [s] | `1.595` |
| Total KSP solve time [s] | `0.610` |
| Total line-search time [s] | `0.601` |
| Final energy | `-7.960005` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/jax_petsc_local_sfd_l9_np4.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/jax_petsc_local_sfd_l9_np4.log` |

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
| 1 | 3.410 | 6 | 11 | -7.960005 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `3.410` |
| Newton iterations | `6` |
| Linear iterations | `11` |
| Energy | `-7.960005` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 331.369849 | 33077101.512515 | 448769.786628 | nan | 1.956948 | 2 | yes | 9 | 0.084 | 0.304 | 0.109 | 0.101 | 1.000000 | nan |
| 2 | -4.968032 | 336.337880 | 208.093499 | nan | 1.956948 | 2 | yes | 9 | 0.084 | 0.281 | 0.109 | 0.099 | 1.000000 | nan |
| 3 | -7.544087 | 2.576056 | 2.316394 | nan | 0.411863 | 1 | yes | 9 | 0.085 | 0.289 | 0.073 | 0.101 | 1.000000 | nan |
| 4 | -7.952412 | 0.408325 | 2.272104 | nan | 0.948817 | 2 | yes | 9 | 0.084 | 0.248 | 0.104 | 0.099 | 1.000000 | nan |
| 5 | -7.959969 | 0.007556 | 0.295317 | nan | 1.055248 | 2 | yes | 9 | 0.084 | 0.239 | 0.106 | 0.100 | 1.000000 | nan |
| 6 | -7.960005 | 0.000036 | 0.017843 | 0.001852 | 1.088137 | 2 | yes | 9 | 0.084 | 0.235 | 0.108 | 0.100 | 1.000000 | nan |

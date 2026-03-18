# jax_petsc_local_sfd_l8_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `8` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `197633` |
| Free DOFs | `195585` |
| Setup time [s] | `3.224` |
| Total solve time [s] | `2.724` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.583` |
| Total PC init time [s] | `1.220` |
| Total KSP solve time [s] | `0.406` |
| Total line-search time [s] | `0.468` |
| Final energy | `-7.959556` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_local_sfd_l8_np1.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_local_sfd_l8_np1.log` |

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
| 1 | 2.724 | 6 | 12 | -7.959556 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `2.724` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.959556` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41.885321 | 4134621.084891 | 111906.518518 | nan | 1.956948 | 2 | yes | 9 | 0.380 | 0.238 | 0.070 | 0.080 | 1.000000 | nan |
| 2 | -5.513351 | 47.398672 | 51.887977 | nan | 1.817627 | 2 | yes | 9 | 0.042 | 0.225 | 0.070 | 0.078 | 1.000000 | nan |
| 3 | -7.579611 | 2.066260 | 8.097798 | nan | 0.723392 | 2 | yes | 9 | 0.041 | 0.230 | 0.074 | 0.076 | 1.000000 | nan |
| 4 | -7.956696 | 0.377085 | 3.040997 | nan | 1.174242 | 2 | yes | 9 | 0.040 | 0.187 | 0.065 | 0.078 | 1.000000 | nan |
| 5 | -7.959532 | 0.002836 | 0.238666 | nan | 1.055248 | 2 | yes | 9 | 0.039 | 0.173 | 0.064 | 0.077 | 1.000000 | nan |
| 6 | -7.959556 | 0.000024 | 0.015492 | 0.001067 | 1.088137 | 2 | yes | 9 | 0.040 | 0.166 | 0.063 | 0.080 | 1.000000 | nan |

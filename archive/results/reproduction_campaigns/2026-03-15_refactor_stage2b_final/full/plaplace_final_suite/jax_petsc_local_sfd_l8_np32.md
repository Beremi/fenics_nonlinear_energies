# jax_petsc_local_sfd_l8_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `8` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `197633` |
| Free DOFs | `195585` |
| Setup time [s] | `1.262` |
| Total solve time [s] | `0.742` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.211` |
| Total PC init time [s] | `0.310` |
| Total KSP solve time [s] | `0.080` |
| Total line-search time [s] | `0.127` |
| Final energy | `-7.959556` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_local_sfd_l8_np32.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_local_sfd_l8_np32.log` |

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
| 1 | 0.742 | 6 | 12 | -7.959556 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.742` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.959556` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41.955652 | 4134621.014560 | 111906.518518 | nan | 1.956948 | 2 | yes | 9 | 0.188 | 0.054 | 0.012 | 0.022 | 1.000000 | nan |
| 2 | -5.521617 | 47.477270 | 51.899792 | nan | 1.817627 | 2 | yes | 9 | 0.005 | 0.054 | 0.016 | 0.021 | 1.000000 | nan |
| 3 | -7.575237 | 2.053620 | 8.033874 | nan | 0.690503 | 2 | yes | 9 | 0.004 | 0.055 | 0.013 | 0.021 | 1.000000 | nan |
| 4 | -7.956467 | 0.381230 | 3.132989 | nan | 1.174242 | 2 | yes | 9 | 0.004 | 0.051 | 0.012 | 0.021 | 1.000000 | nan |
| 5 | -7.959530 | 0.003064 | 0.251726 | nan | 1.055248 | 2 | yes | 9 | 0.004 | 0.049 | 0.013 | 0.021 | 1.000000 | nan |
| 6 | -7.959556 | 0.000026 | 0.017055 | 0.001347 | 1.088137 | 2 | yes | 9 | 0.005 | 0.047 | 0.013 | 0.021 | 1.000000 | nan |

# jax_petsc_local_sfd_l6_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `6` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `12545` |
| Free DOFs | `12033` |
| Setup time [s] | `0.727` |
| Total solve time [s] | `0.516` |
| Total Newton iterations | `7` |
| Total linear iterations | `13` |
| Total assembly time [s] | `0.178` |
| Total PC init time [s] | `0.214` |
| Total KSP solve time [s] | `0.065` |
| Total line-search time [s] | `0.053` |
| Final energy | `-7.954564` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_local_sfd_l6_np32.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_local_sfd_l6_np32.log` |

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
| 1 | 0.516 | 7 | 13 | -7.954564 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.516` |
| Newton iterations | `7` |
| Linear iterations | `13` |
| Energy | `-7.954564` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.196888 | 65342.809318 | 6978.482874 | nan | 1.956948 | 2 | yes | 9 | 0.174 | 0.032 | 0.011 | 0.007 | 1.000000 | nan |
| 2 | -5.826087 | 6.022975 | 3.268912 | nan | 0.411863 | 1 | yes | 9 | 0.001 | 0.031 | 0.007 | 0.008 | 1.000000 | nan |
| 3 | -7.714359 | 1.888272 | 5.410014 | nan | 0.637288 | 2 | yes | 9 | 0.001 | 0.031 | 0.010 | 0.008 | 1.000000 | nan |
| 4 | -7.942175 | 0.227816 | 2.138874 | nan | 0.948817 | 2 | yes | 9 | 0.001 | 0.027 | 0.009 | 0.008 | 1.000000 | nan |
| 5 | -7.954354 | 0.012179 | 0.462226 | nan | 1.088137 | 2 | yes | 9 | 0.001 | 0.034 | 0.011 | 0.008 | 1.000000 | nan |
| 6 | -7.954563 | 0.000209 | 0.051364 | nan | 1.088137 | 2 | yes | 9 | 0.001 | 0.033 | 0.009 | 0.006 | 1.000000 | nan |
| 7 | -7.954564 | 0.000001 | 0.002752 | 0.000071 | 1.002033 | 2 | yes | 9 | 0.001 | 0.026 | 0.008 | 0.008 | 1.000000 | nan |

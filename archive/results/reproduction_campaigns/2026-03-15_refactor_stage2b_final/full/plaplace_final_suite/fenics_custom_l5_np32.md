# fenics_custom_l5_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `5` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `3201` |
| Free DOFs | `3194` |
| Setup time [s] | `0.050` |
| Total solve time [s] | `0.135` |
| Total Newton iterations | `5` |
| Total linear iterations | `5` |
| Total assembly time [s] | `0.002` |
| Total PC init time [s] | `0.100` |
| Total KSP solve time [s] | `0.022` |
| Total line-search time [s] | `0.007` |
| Final energy | `-7.942969` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l5_np32.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l5_np32.log` |

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
| `local_hessian_mode` | `-` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.135 | 5 | 5 | -7.942969 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.135` |
| Newton iterations | `5` |
| Linear iterations | `5` |
| Energy | `-7.942969` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -6.049561 | 6.194216 | 0.565238 | nan | 0.100333 | 1 | yes | 9 | 0.001 | 0.023 | 0.005 | 0.001 | 1.000000 | nan |
| 2 | -7.740327 | 1.690766 | 5.437894 | nan | 0.637288 | 1 | yes | 9 | 0.000 | 0.020 | 0.005 | 0.002 | 1.000000 | nan |
| 3 | -7.940765 | 0.200438 | 2.173535 | nan | 1.088137 | 1 | yes | 9 | 0.000 | 0.018 | 0.004 | 0.001 | 1.000000 | nan |
| 4 | -7.942961 | 0.002195 | 0.161261 | nan | 1.055248 | 1 | yes | 9 | 0.000 | 0.019 | 0.004 | 0.002 | 1.000000 | nan |
| 5 | -7.942969 | 0.000008 | 0.010087 | 0.000304 | 1.002033 | 1 | yes | 9 | 0.000 | 0.019 | 0.004 | 0.001 | 1.000000 | nan |

# fenics_custom_l8_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `8` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `197633` |
| Free DOFs | `196986` |
| Setup time [s] | `0.672` |
| Total solve time [s] | `1.218` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.132` |
| Total PC init time [s] | `0.450` |
| Total KSP solve time [s] | `0.163` |
| Total line-search time [s] | `0.410` |
| Final energy | `-7.959556` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l8_np4.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l8_np4.log` |

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
| 1 | 1.218 | 6 | 12 | -7.959556 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `1.218` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.959556` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -4.029409 | 8.321377 | 11.233000 | nan | 0.776608 | 2 | yes | 9 | 0.024 | 0.090 | 0.027 | 0.077 | 1.000000 | nan |
| 2 | -7.369241 | 3.339832 | 9.178317 | nan | 0.809497 | 2 | yes | 9 | 0.022 | 0.082 | 0.028 | 0.067 | 1.000000 | nan |
| 3 | -7.929189 | 0.559948 | 3.154281 | nan | 1.002033 | 2 | yes | 9 | 0.022 | 0.071 | 0.027 | 0.067 | 1.000000 | nan |
| 4 | -7.958777 | 0.029589 | 0.653791 | nan | 1.055248 | 2 | yes | 9 | 0.022 | 0.069 | 0.027 | 0.067 | 1.000000 | nan |
| 5 | -7.959542 | 0.000765 | 0.091497 | nan | 1.088137 | 2 | yes | 9 | 0.021 | 0.068 | 0.027 | 0.066 | 1.000000 | nan |
| 6 | -7.959556 | 0.000014 | 0.010435 | 0.000853 | 1.088137 | 2 | yes | 9 | 0.022 | 0.069 | 0.027 | 0.066 | 1.000000 | nan |

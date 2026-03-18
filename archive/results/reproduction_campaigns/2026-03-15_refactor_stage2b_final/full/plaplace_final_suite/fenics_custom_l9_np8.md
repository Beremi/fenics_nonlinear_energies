# fenics_custom_l9_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `9` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `788481` |
| Free DOFs | `787928` |
| Setup time [s] | `2.542` |
| Total solve time [s] | `2.959` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.258` |
| Total PC init time [s] | `1.104` |
| Total KSP solve time [s] | `0.604` |
| Total line-search time [s] | `0.863` |
| Final energy | `-7.960006` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l9_np8.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l9_np8.log` |

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
| 1 | 2.959 | 6 | 12 | -7.960006 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `2.959` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.960006` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.103275 | 38.327102 | 44.892839 | nan | 1.678307 | 2 | yes | 9 | 0.046 | 0.202 | 0.098 | 0.143 | 1.000000 | nan |
| 2 | -7.537074 | 2.433799 | 8.687503 | nan | 0.723392 | 2 | yes | 9 | 0.042 | 0.187 | 0.098 | 0.143 | 1.000000 | nan |
| 3 | -7.952727 | 0.415653 | 3.179134 | nan | 1.141353 | 2 | yes | 9 | 0.042 | 0.156 | 0.093 | 0.144 | 1.000000 | nan |
| 4 | -7.959893 | 0.007166 | 0.360692 | nan | 1.141353 | 2 | yes | 9 | 0.042 | 0.187 | 0.104 | 0.145 | 1.000000 | nan |
| 5 | -7.960004 | 0.000112 | 0.037979 | nan | 1.088137 | 2 | yes | 9 | 0.042 | 0.185 | 0.105 | 0.144 | 1.000000 | nan |
| 6 | -7.960006 | 0.000001 | 0.003333 | 0.000174 | 1.055248 | 2 | yes | 9 | 0.042 | 0.187 | 0.106 | 0.144 | 1.000000 | nan |

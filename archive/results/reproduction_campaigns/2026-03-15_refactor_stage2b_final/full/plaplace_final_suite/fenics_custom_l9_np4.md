# fenics_custom_l9_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `9` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `788481` |
| Free DOFs | `787184` |
| Setup time [s] | `2.998` |
| Total solve time [s] | `4.843` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.511` |
| Total PC init time [s] | `1.738` |
| Total KSP solve time [s] | `0.714` |
| Total line-search time [s] | `1.636` |
| Final energy | `-7.960006` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l9_np4.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l9_np4.log` |

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
| 1 | 4.843 | 6 | 12 | -7.960006 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `4.843` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.960006` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.151387 | 38.318922 | 44.826888 | nan | 1.678307 | 2 | yes | 9 | 0.091 | 0.345 | 0.121 | 0.273 | 1.000000 | nan |
| 2 | -7.549155 | 2.397768 | 8.595540 | nan | 0.723392 | 2 | yes | 9 | 0.084 | 0.325 | 0.120 | 0.273 | 1.000000 | nan |
| 3 | -7.953221 | 0.404066 | 3.131350 | nan | 1.141353 | 2 | yes | 9 | 0.084 | 0.264 | 0.113 | 0.273 | 1.000000 | nan |
| 4 | -7.959901 | 0.006681 | 0.349018 | nan | 1.088137 | 2 | yes | 9 | 0.084 | 0.256 | 0.115 | 0.273 | 1.000000 | nan |
| 5 | -7.960005 | 0.000103 | 0.034381 | nan | 1.141353 | 2 | yes | 9 | 0.085 | 0.273 | 0.122 | 0.273 | 1.000000 | nan |
| 6 | -7.960006 | 0.000001 | 0.003211 | 0.000191 | 1.055248 | 2 | yes | 9 | 0.084 | 0.274 | 0.123 | 0.272 | 1.000000 | nan |

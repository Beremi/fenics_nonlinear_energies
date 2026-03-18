# fenics_custom_l7_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `7` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `49665` |
| Free DOFs | `48641` |
| Setup time [s] | `0.135` |
| Total solve time [s] | `1.021` |
| Total Newton iterations | `6` |
| Total linear iterations | `11` |
| Total assembly time [s] | `0.115` |
| Total PC init time [s] | `0.333` |
| Total KSP solve time [s] | `0.116` |
| Total line-search time [s] | `0.396` |
| Final energy | `-7.958292` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l7_np1.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l7_np1.log` |

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
| 1 | 1.021 | 6 | 11 | -7.958292 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `1.021` |
| Newton iterations | `6` |
| Linear iterations | `11` |
| Energy | `-7.958292` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.362796 | 6.031619 | 2.814561 | nan | 0.358647 | 1 | yes | 9 | 0.021 | 0.065 | 0.013 | 0.066 | 1.000000 | nan |
| 2 | -7.624973 | 2.262177 | 6.233213 | nan | 0.584072 | 2 | yes | 9 | 0.019 | 0.060 | 0.021 | 0.066 | 1.000000 | nan |
| 3 | -7.940510 | 0.315536 | 2.740784 | nan | 0.948817 | 2 | yes | 9 | 0.019 | 0.053 | 0.020 | 0.066 | 1.000000 | nan |
| 4 | -7.957957 | 0.017447 | 0.561445 | nan | 1.055248 | 2 | yes | 9 | 0.019 | 0.053 | 0.020 | 0.066 | 1.000000 | nan |
| 5 | -7.958289 | 0.000332 | 0.064457 | nan | 1.141353 | 2 | yes | 9 | 0.019 | 0.051 | 0.021 | 0.066 | 1.000000 | nan |
| 6 | -7.958292 | 0.000004 | 0.005655 | 0.000257 | 1.055248 | 2 | yes | 9 | 0.019 | 0.052 | 0.021 | 0.066 | 1.000000 | nan |

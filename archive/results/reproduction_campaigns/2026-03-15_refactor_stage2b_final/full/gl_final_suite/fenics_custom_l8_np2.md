# fenics_custom_l8_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `8` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `263169` |
| Free DOFs | `262132` |
| Setup time [s] | `1.064` |
| Total solve time [s] | `3.538` |
| Total Newton iterations | `8` |
| Total linear iterations | `33` |
| Total assembly time [s] | `0.480` |
| Total PC init time [s] | `0.658` |
| Total KSP solve time [s] | `0.580` |
| Total line-search time [s] | `1.656` |
| Final energy | `0.345634` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l8_np2.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l8_np2.log` |

## Frozen Settings

| Setting | Value |
|---|---|
| `ksp_type` | `gmres` |
| `pc_type` | `hypre` |
| `ksp_rtol` | `0.001` |
| `ksp_max_it` | `200` |
| `use_trust_region` | `False` |
| `trust_subproblem_line_search` | `False` |
| `linesearch_interval` | `[-0.5, 2.0]` |
| `linesearch_tol` | `0.001` |
| `trust_radius_init` | `1.0` |
| `local_hessian_mode` | `-` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 3.538 | 8 | 33 | 0.345634 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `3.538` |
| Newton iterations | `8` |
| Linear iterations | `33` |
| Energy | `0.345634` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.560775 | 0.104527 | 0.001852 | nan | -0.270127 | 9 | yes | 19 | 0.064 | 0.098 | 0.145 | 0.207 | 1.000000 | nan |
| 2 | 0.508986 | 0.051789 | 0.002031 | nan | 0.329640 | 6 | yes | 19 | 0.060 | 0.081 | 0.099 | 0.207 | 1.000000 | nan |
| 3 | 0.443295 | 0.065691 | 0.001111 | nan | -0.305281 | 5 | yes | 19 | 0.059 | 0.080 | 0.085 | 0.206 | 1.000000 | nan |
| 4 | 0.381891 | 0.061404 | 0.001730 | nan | 0.198084 | 4 | yes | 19 | 0.059 | 0.080 | 0.070 | 0.207 | 1.000000 | nan |
| 5 | 0.349980 | 0.031912 | 0.001562 | nan | 0.619710 | 3 | yes | 19 | 0.059 | 0.079 | 0.056 | 0.207 | 1.000000 | nan |
| 6 | 0.345655 | 0.004325 | 0.000594 | nan | 1.071260 | 2 | yes | 19 | 0.059 | 0.080 | 0.042 | 0.207 | 1.000000 | nan |
| 7 | 0.345634 | 0.000021 | 0.000037 | nan | 1.007047 | 2 | yes | 19 | 0.060 | 0.080 | 0.042 | 0.208 | 1.000000 | nan |
| 8 | 0.345634 | 0.000000 | 0.000000 | 0.000000 | 1.004082 | 2 | yes | 19 | 0.059 | 0.079 | 0.042 | 0.208 | 1.000000 | nan |

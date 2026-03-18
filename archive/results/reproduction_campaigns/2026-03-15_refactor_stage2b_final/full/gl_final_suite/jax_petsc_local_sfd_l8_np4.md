# jax_petsc_local_sfd_l8_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `8` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `263169` |
| Free DOFs | `261121` |
| Setup time [s] | `2.158` |
| Total solve time [s] | `1.844` |
| Total Newton iterations | `7` |
| Total linear iterations | `24` |
| Total assembly time [s] | `0.678` |
| Total PC init time [s] | `0.404` |
| Total KSP solve time [s] | `0.221` |
| Total line-search time [s] | `0.500` |
| Final energy | `0.345634` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l8_np4.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l8_np4.log` |

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
| `local_hessian_mode` | `sfd_local` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 1.844 | 7 | 24 | 0.345634 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `1.844` |
| Newton iterations | `7` |
| Linear iterations | `24` |
| Energy | `0.345634` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.501516 | 0.163785 | 0.001852 | nan | -0.327708 | 6 | yes | 19 | 0.271 | 0.067 | 0.051 | 0.071 | 1.000000 | nan |
| 2 | 0.500357 | 0.001159 | 0.001847 | nan | -0.264896 | 5 | yes | 19 | 0.065 | 0.057 | 0.043 | 0.070 | 1.000000 | nan |
| 3 | 0.466496 | 0.033861 | 0.002324 | nan | -0.160030 | 5 | yes | 19 | 0.067 | 0.056 | 0.043 | 0.071 | 1.000000 | nan |
| 4 | 0.358450 | 0.108046 | 0.002635 | nan | 0.437471 | 2 | yes | 19 | 0.065 | 0.056 | 0.021 | 0.071 | 1.000000 | nan |
| 5 | 0.345745 | 0.012705 | 0.001241 | nan | 1.032440 | 2 | yes | 19 | 0.067 | 0.056 | 0.021 | 0.072 | 1.000000 | nan |
| 6 | 0.345634 | 0.000111 | 0.000090 | nan | 1.012546 | 2 | yes | 19 | 0.070 | 0.057 | 0.022 | 0.072 | 1.000000 | nan |
| 7 | 0.345634 | 0.000000 | 0.000000 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.071 | 0.056 | 0.021 | 0.073 | 1.000000 | nan |

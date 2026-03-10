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
| Total DOFs | `263169` |
| Free DOFs | `261121` |
| Setup time [s] | `3.976` |
| Total solve time [s] | `2.559` |
| Total Newton iterations | `7` |
| Total linear iterations | `25` |
| Total assembly time [s] | `0.603` |
| Total PC init time [s] | `0.650` |
| Total KSP solve time [s] | `0.513` |
| Total line-search time [s] | `0.713` |
| Final energy | `0.345634` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l8_np1.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l8_np1.log` |

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
| 1 | 2.559 | 7 | 25 | 0.345634 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `2.559` |
| Newton iterations | `7` |
| Linear iterations | `25` |
| Energy | `0.345634` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.500078 | 0.165223 | 0.001852 | nan | -0.334072 | 7 | yes | 19 | 0.087 | 0.106 | 0.131 | 0.104 | 1.000000 | nan |
| 2 | 0.498645 | 0.001434 | 0.001824 | nan | -0.252066 | 5 | yes | 19 | 0.087 | 0.089 | 0.096 | 0.102 | 1.000000 | nan |
| 3 | 0.457824 | 0.040821 | 0.002272 | nan | -0.135338 | 5 | yes | 19 | 0.086 | 0.091 | 0.098 | 0.101 | 1.000000 | nan |
| 4 | 0.376056 | 0.081769 | 0.002511 | nan | 0.326674 | 2 | yes | 19 | 0.085 | 0.092 | 0.047 | 0.101 | 1.000000 | nan |
| 5 | 0.346402 | 0.029653 | 0.001654 | nan | 0.957065 | 2 | yes | 19 | 0.088 | 0.092 | 0.047 | 0.101 | 1.000000 | nan |
| 6 | 0.345634 | 0.000768 | 0.000230 | nan | 1.041337 | 2 | yes | 19 | 0.085 | 0.092 | 0.048 | 0.101 | 1.000000 | nan |
| 7 | 0.345634 | 0.000000 | 0.000005 | 0.000000 | 1.001116 | 2 | yes | 19 | 0.086 | 0.089 | 0.046 | 0.104 | 1.000000 | nan |

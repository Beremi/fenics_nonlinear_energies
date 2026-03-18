# Representative Matrix Checkpoint

Campaign: `2026-03-15_refactor_stage2`

This checkpoint freezes the first beyond-smoke direct-entrypoint matrix used for the second half of Phase 2.

| Case | Result | Key metrics | Notes |
| --- | --- | --- | --- |
| pLaplace pure JAX level 5 np=1 | `completed` | dofs=2945; solve_time_s=0.102500; newton_iters=6; energy=-7.943000 |  |
| pLaplace FEniCS custom level 5 np=2 | `completed` | dofs=3201; solve_time_s=0.060300; newton_iters=5; linear_iters=15; energy=-7.942969 |  |
| pLaplace JAX+PETSc level 5 np=2 | `completed` | dofs=3201; solve_time_s=0.174600; newton_iters=6; linear_iters=17; energy=-7.942969 |  |
| GL FEniCS custom level 5 np=2 | `completed` | dofs=4225; solve_time_s=0.079700; newton_iters=7; linear_iters=28; energy=0.346232 |  |
| GL JAX+PETSc level 5 np=2 | `completed` | dofs=4225; solve_time_s=0.240322; newton_iters=12; energy=0.345987 |  |
| HE pure JAX level 1 steps 24/24 np=1 | `completed` | dofs=2187; solve_time_s=41.595491; newton_iters=559; linear_iters=2284 |  |
| HE FEniCS custom level 1 steps 24/24 np=2 | `failed` | failure_mode=PETSc signal 11 / MPI abort (no JSON output) | log: `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/representative_matrix/logs/he_fenics_custom_l1_steps24_np2.log` |
| HE JAX+PETSc level 1 steps 24/24 np=2 | `completed` | dofs=2187; solve_time_s=20.663140; newton_iters=528 |  |
| Topology pure JAX 192x96 np=1 | `completed` | wall_time_s=17.787321; outer_iterations=121; final_p=4.000000; final_compliance=4.155706; final_volume_fraction=0.400000 | Current direct CLI result differs from the existing benchmark markdown values; the markdown report is stale relative to the maintained solver path. |
| Topology parallel 768x384 np=2 | `completed` | wall_time_s=138.601570; outer_iterations=72; final_p=5.600000; final_compliance=8.947271; final_volume_fraction=0.393204 | Matches the current parallel benchmark qualitative state and final metrics; only wall time drifted. |

## Findings

- Nine of the ten representative direct-entrypoint cases completed successfully.
- `HyperElasticity` FEniCS custom `level 1`, `24/24` steps, `np=2` crashed with a PETSc segmentation violation before writing JSON.
- The direct `192 x 96` pure-JAX topology benchmark now lands at a different final state than the current markdown report, so the benchmark doc is stale relative to the maintained solver path.
- The `768 x 384`, `np=2` parallel topology representative reproduces the current qualitative state and final metrics from the parallel report, with only wall-time drift.

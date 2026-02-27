# Step-1 Timing Breakdown (Level 3, np=16, GAMG)

Input runs:
- `experiment_scripts/he_fenics_l3_step1_np16_gamg_timing.json`
- `experiment_scripts/he_jax_petsc_l3_step1_np16_gamg_timing.json`

## Newton-level Breakdown

| Solver | Step Time [s] | Newton iters | KSP iters | Grad [s] | Hess callback [s] | Line search [s] | Update [s] | Overhead [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FEniCS custom GAMG np16 | 2.123100 | 40 | 414 | 0.035226 | 1.701550 | 0.365444 | 0.018250 | 0.001712 |
| JAX+PETSc GAMG np16 | 42.189392 | 39 | 402 | 0.499072 | 38.561745 | 2.960301 | 0.149893 | 0.017056 |

## Linear-Callback Breakdown (sum over Newton iterations)

| Solver | Hess assembly [s] | setOperators [s] | setTolerances [s] | PC setup [s] | KSP solve [s] | Linear total [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FEniCS custom GAMG np16 | 0.615968 | 0.000217 | 0.000000 | 0.064120 | 1.019717 | 1.700012 |
| JAX+PETSc GAMG np16 | 20.787788 | 0.017068 | 0.000635 | 1.280878 | 16.472861 | 38.559229 |

## JAX+PETSc Assembly Sub-breakdown

| Component | Time [s] | Share of assembly [%] |
| --- | ---: | ---: |
| P2P exchange | 0.018797 | 0.09 |
| HVP compute | 20.097207 | 96.68 |
| extraction | 0.405546 | 1.95 |
| COO assembly | 0.258683 | 1.24 |

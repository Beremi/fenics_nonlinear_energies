# Step-1 Timing Investigation (L3, np=16, GAMG)

Runs:
- FEniCS custom baseline:                         `experiment_scripts/he_fenics_l3_step1_np16_gamg_timing.json`
- JAX+PETSc previous (batched, implicit matrix finalization): `experiment_scripts/he_jax_petsc_l3_step1_np16_gamg_timing.json`
- JAX+PETSc patched (sequential HVP + explicit A.assemble): `experiment_scripts/he_jax_petsc_l3_step1_np16_gamg_timing_fix1.json`
- JAX+PETSc patched + batched HVP (for control): `experiment_scripts/he_jax_petsc_l3_step1_np16_gamg_timing_fix1_batched.json`

## Newton-level totals

| Solver | Step [s] | Newton | KSP sum | Grad [s] | Hess [s] | LS [s] | Update [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FEniCS custom | 2.1231 | 40 | 414 | 0.03522595524555072 | 1.701550348836463 | 0.3654435549979098 | 0.018250399036332965 |
| JAX+PETSc previous | 42.189392 | 39 | 402 | 0.4990715988096781 | 38.56174478499452 | 2.9603009037673473 | 0.14989337313454598 |
| JAX+PETSc patched (sequential) | 28.075207 | 39 | 402 | 0.5007499278872274 | 24.440081344917417 | 2.9686771529959515 | 0.14803856582148 |
| JAX+PETSc patched (batched) | 41.884078 | 39 | 402 | 0.491469944070559 | 38.33427000103984 | 2.896927593043074 | 0.1426134390057996 |

## Hess-callback internals (sum over Newton)

| Solver | Assembly [s] | HVP [s] | COO/finalize [s] | KSP solve [s] | Linear total [s] |
| --- | ---: | ---: | ---: | ---: | ---: |
| FEniCS custom | 0.615968 | 0 | 0 | 1.019717 | 1.7000120000000005 |
| JAX+PETSc previous | 20.787787781911902 | 20.097206754086073 | 0.25868268113117665 | 16.472861024085432 | 38.55922942660982 |
| JAX+PETSc patched (sequential) | 18.971605406317394 | 5.922214759862982 | 12.815858518064488 | 5.136156533786561 | 24.438180260302033 |
| JAX+PETSc patched (batched) | 33.02489551511826 | 20.894498446024954 | 11.761194041930139 | 4.981207336997613 | 38.33235699223587 |

## Key findings

- Explicit matrix finalization moved major cost from `KSP.solve` into matrix finalize (COO assembly block), reducing solve-phase time significantly.
- Sequential per-color HVP is much faster than batched `vmap` for this HE case.
- Combined patch reduced JAX+PETSc step-1 time from **42.189392 s** to **28.075207 s** (about **1.50x** faster), with identical Newton/KSP iteration counts.
